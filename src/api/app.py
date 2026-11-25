"""
app.py  —  the RentWise UK public REST API (Milestone 5).

    uvicorn src.api.app:app --port 8000        # docs at /docs (OpenAPI)

Data endpoints (header  X-API-Key: rw_...)
    GET  /v1/areas                     areas the model serves (code, name, region)
    GET  /v1/areas/{la_code}           one area: income, spatial profile, property types
    POST /v1/predict                   official ONS rent (or a model estimate + interval where ONS
                                       publishes none) + affordability + SHAP factors
Public
    GET  /v1/health                    liveness + data/model freshness
    GET  /v1/model                     what is deployed and how well it performs (honest metrics)
    GET  /v1/sources                   source, period and release of every input ("as of")
Accounts (Authorization: Bearer <token>)
    POST /v1/users                     sign up
    POST /v1/auth/login | /v1/auth/logout
    GET  /v1/me
    POST/GET/DELETE /v1/me/api-keys    self-service API keys (plaintext shown once)
    GET/POST/DELETE /v1/me/saved-areas
    GET/POST/DELETE /v1/me/alerts      rent / affordability alerts
    GET  /v1/me/notifications ; POST /v1/me/notifications/{id}/read

Design notes
* Only the areas the model was trained on are served: conformal intervals are
  calibrated for those (80% -> 80% coverage) but would under-cover unseen areas.
* Secrets are never stored in plaintext (scrypt passwords, SHA-256 of random keys/tokens).
* Per-key rate limiting on data endpoints; per-client limiting on login/sign-up.
* Configuration via env: RENTWISE_API_DB, RENTWISE_RATE_LIMIT, RENTWISE_ALLOW_SIGNUP,
  RENTWISE_SESSION_DAYS.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402  (also loads .env so RENTWISE_* settings apply)
from src.api import security as sec  # noqa: E402
from src.api.db import Database, now_iso  # noqa: E402
from src.api.ratelimit import SlidingWindowLimiter  # noqa: E402

API_VERSION = "1.0.0"
DISCLAIMER = "Indicative estimate from open-data models; not financial advice."
FACTOR_LABELS = {
    "bedrooms": "Number of bedrooms", "median_annual_income": "Local income",
    "crime_per_1000": "Crime per 1,000 residents",
    "property_type_ord": "Property type / size", "income_band": "Area income band",
    "school_quality": "Local school quality (historic Ofsted)",
    "share_good_or_better": "Share of Good/Outstanding schools",
}


AMENITY_LABELS = {
    "rail_station": "rail stations", "bus_stop": "bus stops", "supermarket": "supermarkets",
    "convenience": "convenience shops", "gp_clinic": "GP surgeries & clinics", "hospital": "hospitals",
    "pharmacy": "pharmacies", "restaurant_cafe": "restaurants & cafés", "pub_bar": "pubs & bars",
    "park": "parks", "gym": "gyms",
}


def db_path_from_env() -> str:
    p = Path(os.environ.get("RENTWISE_API_DB", "data/api/rentwise_api.sqlite3"))
    return str(p if p.is_absolute() or str(p) == ":memory:" else config.ROOT / p)


def factor_label(f: str) -> str:
    if f.endswith("__missing"):
        base = f[: -len("__missing")]
        if base.startswith(("school", "share_good")):
            return "School data not available (Wales — schools are inspected by Estyn)"
        return f"{FACTOR_LABELS.get(base, base)}: no data for this area"
    if f.startswith("region_"):
        return "Being in " + f[len("region_"):]
    if f.startswith("dens_"):
        return "Density of " + AMENITY_LABELS.get(f[5:], f[5:].replace("_", " "))
    return FACTOR_LABELS.get(f, f)


# ------------------------------------------------------------------ schemas
class PredictIn(BaseModel):
    la_code: str = Field(..., pattern=r"^[EW]\d{8}$", examples=["E07000223"])
    property_type: str = Field(..., examples=["Flat"])


class Factor(BaseModel):
    feature: str
    label: str
    impact_gbp: float


class Interval(BaseModel):
    low: int
    high: int
    level: float


class PredictOut(BaseModel):
    la_code: str
    la_name: str
    region: str
    property_type: str
    bedrooms: int
    rent_gbp_month: int
    kind: Literal["official", "modelled"]
    official_rent_gbp_month: int | None
    model_rent_gbp_month: int
    as_of: str | None
    source: str | None
    interval: Interval | None
    rent_to_income: float
    affordable: bool
    benchmark: float = 0.30
    factors: list[Factor]
    model: str
    disclaimer: str = DISCLAIMER


class SignupIn(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., min_length=sec.MIN_PASSWORD_LEN, max_length=256)


class LoginIn(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=256)


class KeyIn(BaseModel):
    name: str = Field("default", min_length=1, max_length=60)


class SavedAreaIn(BaseModel):
    la_code: str = Field(..., pattern=r"^[EW]\d{8}$")
    label: str | None = Field(None, max_length=80)


class AlertIn(BaseModel):
    la_code: str = Field(..., pattern=r"^[EW]\d{8}$")
    property_type: str
    metric: Literal["rent", "ratio"]
    op: Literal["above", "below"]
    threshold: float = Field(..., gt=0)


# ------------------------------------------------------------------ factory
def create_app(db: Database | None = None, predictor=None, rate_limit: int | None = None,
               allow_signup: bool | None = None, auth_limit: int = 10) -> FastAPI:
    """Build the API. Tests inject an in-memory DB and/or a predictor."""
    app = FastAPI(title="RentWise UK API", version=API_VERSION,
                  description="UK rent estimates with calibrated uncertainty, affordability and "
                              "plain-English explanations, from open government data.")
    app.state.db = db or Database(db_path_from_env())
    app.state.predictor = predictor
    app.state.limiter = SlidingWindowLimiter(limit=rate_limit or int(os.environ.get("RENTWISE_RATE_LIMIT", "60")))
    app.state.auth_limiter = SlidingWindowLimiter(limit=auth_limit, window_s=60)
    app.state.allow_signup = (os.environ.get("RENTWISE_ALLOW_SIGNUP", "1") == "1"
                              if allow_signup is None else allow_signup)
    session_days = int(os.environ.get("RENTWISE_SESSION_DAYS", "7"))

    def get_predictor():
        if app.state.predictor is None:
            from src.models.predict import RentWisePredictor
            app.state.predictor = RentWisePredictor()
        return app.state.predictor

    # ---- dependencies --------------------------------------------------
    def require_api_key(response: Response, x_api_key: str | None = Header(None, alias="X-API-Key")) -> dict:
        if not x_api_key:
            raise HTTPException(401, "missing X-API-Key header")
        row = app.state.db.one("SELECT id, user_id, revoked FROM api_keys WHERE key_hash = ?",
                               (sec.sha256(x_api_key),))
        if not row or row["revoked"]:
            raise HTTPException(401, "invalid or revoked API key")
        ok, remaining, retry = app.state.limiter.check(f"key:{row['id']}")
        response.headers["X-RateLimit-Limit"] = str(app.state.limiter.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        if not ok:
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(int(retry) + 1)})
        return row

    def require_user(authorization: str | None = Header(None)) -> dict:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
        tok = authorization.split(" ", 1)[1].strip()
        row = app.state.db.one(
            "SELECT u.id, u.email, u.created_at, s.expires_at FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ?", (sec.sha256(tok),))
        if not row or row["expires_at"] < now_iso():
            raise HTTPException(401, "invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
        return row

    def auth_throttle(request: Request):
        client = request.client.host if request.client else "unknown"
        ok, _, retry = app.state.auth_limiter.check(f"auth:{client}")
        if not ok:
            raise HTTPException(429, "too many attempts, slow down", headers={"Retry-After": str(int(retry) + 1)})

    def check_area_and_type(la_code: str, property_type: str | None = None):
        p = get_predictor()
        area = p.area_by_code(la_code)
        if area is None:
            raise HTTPException(404, f"area {la_code} is not served by this model (see GET /v1/areas)")
        if property_type is not None and property_type not in p.property_types:
            raise HTTPException(422, f"property_type must be one of {p.property_types}")
        return area

    # ---- public ----------------------------------------------------------
    @app.get("/v1/health", tags=["public"])
    def health():
        from src.health import check
        h = check()
        return {"status": h["status"], "serviceable": h["serviceable"], "model": h["model"],
                "data": {k: h["data"].get(k) for k in ("backend", "generated_at", "age_days")},
                "api_version": API_VERSION}

    @app.get("/v1/sources", tags=["public"])
    def sources():
        """Every input's source, period and release — the 'as of' for each figure."""
        p = get_predictor()
        return {"vintages": getattr(p, "vintages", {}) or {},
                "licences": {"government": "Open Government Licence v3.0",
                             "openstreetmap": "ODbL — © OpenStreetMap contributors"}}

    @app.get("/v1/model", tags=["public"])
    def model_info():
        info = get_predictor().model_info()
        info["served_areas"] = len(get_predictor().areas)
        info["note"] = ("Intervals are calibrated for the served areas only; random_cv = areas seen in "
                        "training, grouped_cv = whole areas held out.")
        return info

    # ---- data (API key) --------------------------------------------------
    @app.get("/v1/areas", tags=["data"])
    def areas(_key: dict = Depends(require_api_key)):
        a = get_predictor().areas
        return {"count": len(a), "areas": [{"la_code": r.la_code, "la_name": r.la_name, "region": r.region}
                                           for r in a.sort_values("la_name").itertuples()]}

    @app.get("/v1/areas/{la_code}", tags=["data"])
    def area(la_code: str, _key: dict = Depends(require_api_key)):
        p = get_predictor()
        a = check_area_and_type(la_code)
        return {"la_code": la_code, "la_name": a["la_name"], "region": a["region"],
                "median_annual_income": float(a["income"]), "crime_per_1000": float(a["crime"]),
                "property_types": {t: {"typical_bedrooms": p.typical_bedrooms(t)} for t in p.property_types},
                "spatial_profile": p.spatial_profile(a["la_name"])}

    @app.post("/v1/predict", response_model=PredictOut, tags=["data"])
    def predict(body: PredictIn, _key: dict = Depends(require_api_key)):
        p = get_predictor()
        a = check_area_and_type(body.la_code, body.property_type)
        r = p.predict(a["la_name"], body.property_type)
        return PredictOut(
            la_code=body.la_code, la_name=a["la_name"], region=r["region"],
            property_type=body.property_type, bedrooms=p.typical_bedrooms(body.property_type),
            rent_gbp_month=r["rent"], kind=r["kind"], official_rent_gbp_month=r["official_rent"],
            model_rent_gbp_month=r["model_rent"], as_of=r.get("as_of"), source=r.get("source"),
            interval=r["interval"], rent_to_income=round(r["ratio"], 4),
            affordable=r["affordable"],
            factors=[Factor(feature=f, label=factor_label(f), impact_gbp=round(float(v), 2))
                     for f, v in r["contributions"][:8]],
            model=p.model_info()["model"])

    # ---- accounts --------------------------------------------------------
    @app.post("/v1/users", status_code=201, tags=["accounts"], dependencies=[Depends(auth_throttle)])
    def signup(body: SignupIn):
        if not app.state.allow_signup:
            raise HTTPException(403, "sign-up is disabled")
        try:
            email = sec.validate_email(body.email)
            sec.validate_password(body.password)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        if app.state.db.one("SELECT id FROM users WHERE email = ?", (email,)):
            raise HTTPException(409, "an account with this email already exists")
        uid = app.state.db.run("INSERT INTO users (email, pw_hash, created_at) VALUES (?, ?, ?)",
                               (email, sec.hash_password(body.password), now_iso()))
        return {"id": uid, "email": email}

    @app.post("/v1/auth/login", tags=["accounts"], dependencies=[Depends(auth_throttle)])
    def login(body: LoginIn):
        email = (body.email or "").strip().lower()
        u = app.state.db.one("SELECT id, pw_hash FROM users WHERE email = ?", (email,))
        # verify against a dummy hash when the user doesn't exist (no user-enumeration timing gap)
        stored = u["pw_hash"] if u else _DUMMY_HASH
        if not sec.verify_password(body.password, stored) or not u:
            raise HTTPException(401, "invalid email or password")
        tok, h = sec.new_session_token()
        exp = (datetime.now(timezone.utc) + timedelta(days=session_days)).isoformat(timespec="seconds")
        app.state.db.run("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)", (h, u["id"], exp))
        app.state.db.run("DELETE FROM sessions WHERE expires_at < ?", (now_iso(),))      # housekeeping
        return {"access_token": tok, "token_type": "bearer", "expires_at": exp}

    @app.post("/v1/auth/logout", status_code=204, tags=["accounts"])
    def logout(authorization: str | None = Header(None), user: dict = Depends(require_user)):
        app.state.db.run("DELETE FROM sessions WHERE token_hash = ?", (sec.sha256(authorization.split(" ", 1)[1]),))
        return Response(status_code=204)

    @app.get("/v1/me", tags=["accounts"])
    def me(user: dict = Depends(require_user)):
        return {"id": user["id"], "email": user["email"], "created_at": user["created_at"]}

    @app.post("/v1/me/api-keys", status_code=201, tags=["accounts"])
    def create_key(body: KeyIn, user: dict = Depends(require_user)):
        n = app.state.db.one("SELECT COUNT(*) c FROM api_keys WHERE user_id = ? AND revoked = 0", (user["id"],))["c"]
        if n >= 5:
            raise HTTPException(409, "limit of 5 active API keys reached; revoke one first")
        key, prefix, h = sec.new_api_key()
        kid = app.state.db.run("INSERT INTO api_keys (user_id, name, prefix, key_hash, created_at) "
                               "VALUES (?, ?, ?, ?, ?)", (user["id"], body.name, prefix, h, now_iso()))
        return {"id": kid, "name": body.name, "prefix": prefix, "api_key": key,
                "note": "Store this key now — it will not be shown again."}

    @app.get("/v1/me/api-keys", tags=["accounts"])
    def list_keys(user: dict = Depends(require_user)):
        return app.state.db.all("SELECT id, name, prefix, created_at, revoked FROM api_keys WHERE user_id = ? "
                                "ORDER BY id", (user["id"],))

    @app.delete("/v1/me/api-keys/{key_id}", status_code=204, tags=["accounts"])
    def revoke_key(key_id: int, user: dict = Depends(require_user)):
        if not app.state.db.run("UPDATE api_keys SET revoked = 1 WHERE id = ? AND user_id = ?", (key_id, user["id"])):
            raise HTTPException(404, "key not found")
        return Response(status_code=204)

    # ---- saved areas -------------------------------------------------------
    @app.get("/v1/me/saved-areas", tags=["saved areas"])
    def saved(user: dict = Depends(require_user)):
        rows = app.state.db.all("SELECT la_code, label, created_at FROM saved_areas WHERE user_id = ? "
                                "ORDER BY created_at", (user["id"],))
        p = get_predictor()
        for r in rows:
            a = p.area_by_code(r["la_code"])
            r["la_name"] = a["la_name"] if a is not None else None
        return rows

    @app.post("/v1/me/saved-areas", status_code=201, tags=["saved areas"])
    def save_area(body: SavedAreaIn, user: dict = Depends(require_user)):
        a = check_area_and_type(body.la_code)
        n = app.state.db.one("SELECT COUNT(*) c FROM saved_areas WHERE user_id = ?", (user["id"],))["c"]
        if n >= 50:
            raise HTTPException(409, "limit of 50 saved areas reached")
        if app.state.db.one("SELECT 1 x FROM saved_areas WHERE user_id = ? AND la_code = ?",
                            (user["id"], body.la_code)):
            raise HTTPException(409, "area already saved")
        app.state.db.run("INSERT INTO saved_areas (user_id, la_code, label, created_at) VALUES (?, ?, ?, ?)",
                         (user["id"], body.la_code, body.label, now_iso()))
        return {"la_code": body.la_code, "la_name": a["la_name"], "label": body.label}

    @app.delete("/v1/me/saved-areas/{la_code}", status_code=204, tags=["saved areas"])
    def unsave(la_code: str, user: dict = Depends(require_user)):
        if not app.state.db.run("DELETE FROM saved_areas WHERE user_id = ? AND la_code = ?", (user["id"], la_code)):
            raise HTTPException(404, "area not saved")
        return Response(status_code=204)

    # ---- alerts & notifications -------------------------------------------
    @app.get("/v1/me/alerts", tags=["alerts"])
    def list_alerts(user: dict = Depends(require_user)):
        return app.state.db.all("SELECT id, la_code, property_type, metric, op, threshold, active, last_value, "
                                "last_state, last_checked_at, created_at FROM alerts WHERE user_id = ? ORDER BY id",
                                (user["id"],))

    @app.post("/v1/me/alerts", status_code=201, tags=["alerts"])
    def create_alert(body: AlertIn, user: dict = Depends(require_user)):
        check_area_and_type(body.la_code, body.property_type)
        if body.metric == "ratio" and body.threshold > 3:
            raise HTTPException(422, "ratio thresholds are fractions, e.g. 0.30 for 30%")
        n = app.state.db.one("SELECT COUNT(*) c FROM alerts WHERE user_id = ? AND active = 1", (user["id"],))["c"]
        if n >= 20:
            raise HTTPException(409, "limit of 20 active alerts reached")
        aid = app.state.db.run(
            "INSERT INTO alerts (user_id, la_code, property_type, metric, op, threshold, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user["id"], body.la_code, body.property_type, body.metric, body.op, body.threshold, now_iso()))
        return {"id": aid, **body.model_dump(), "active": True}

    @app.delete("/v1/me/alerts/{alert_id}", status_code=204, tags=["alerts"])
    def delete_alert(alert_id: int, user: dict = Depends(require_user)):
        if not app.state.db.run("DELETE FROM alerts WHERE id = ? AND user_id = ?", (alert_id, user["id"])):
            raise HTTPException(404, "alert not found")
        return Response(status_code=204)

    @app.get("/v1/me/notifications", tags=["alerts"])
    def notifications(unread_only: bool = False, user: dict = Depends(require_user)):
        sql = ("SELECT id, alert_id, message, value, created_at, read_at, delivered_via FROM notifications "
               "WHERE user_id = ?" + (" AND read_at IS NULL" if unread_only else "") + " ORDER BY id DESC LIMIT 100")
        return app.state.db.all(sql, (user["id"],))

    @app.post("/v1/me/notifications/{nid}/read", status_code=204, tags=["alerts"])
    def mark_read(nid: int, user: dict = Depends(require_user)):
        if not app.state.db.run(
                "UPDATE notifications SET read_at = ? WHERE id = ? AND user_id = ? AND read_at IS NULL",
                (now_iso(), nid, user["id"])):
            raise HTTPException(404, "notification not found or already read")
        return Response(status_code=204)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):        # never leak stack traces
        import logging
        logging.getLogger("rentwise.api").exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    return app


_DUMMY_HASH = sec.hash_password("dummy-password-for-timing")

# Module-level app for `uvicorn src.api.app:app` — built lazily so importing this module
# (e.g. in tests) doesn't create the default database file.
def __getattr__(name):
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
