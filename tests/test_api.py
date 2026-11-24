"""RentWise public API (Milestone 5): contract, auth, security, accounts, alerts.

Uses a small deterministic fake predictor so every assertion is exact; a separate
class runs the same endpoints against the real deployed model."""
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.api import security as sec  # noqa: E402
from src.api.alerts import evaluate_all  # noqa: E402
from src.api.app import create_app, factor_label  # noqa: E402
from src.api.db import Database  # noqa: E402


class FakePredictor:
    property_types = ["Flat", "Terraced"]

    def __init__(self):
        self.areas = pd.DataFrame({"la_code": ["E07000223", "W06000015"], "la_name": ["Adur", "Cardiff"],
                                   "region": ["South East", "Wales"], "income": [30000.0, 28000.0],
                                   "crime": [3.1, 5.0]})
        self.rent = {"Adur": 1100, "Cardiff": 900}

    def area_by_code(self, code):
        m = self.areas[self.areas.la_code == code]
        return None if m.empty else m.iloc[0]

    def typical_bedrooms(self, t):
        return 2 if t == "Flat" else 3

    def spatial_profile(self, name):
        return {"dens_pub_bar": 0.5, "school_quality": None if name == "Cardiff" else 3.1}

    def model_info(self):
        return {"model": "fake", "spatial": True}

    def predict(self, name, ptype, bedrooms=None):
        rent = self.rent[name] + (100 if ptype == "Terraced" else 0)
        inc = float(self.areas.set_index("la_name").loc[name, "income"])
        return {"rent": rent, "region": "x", "ratio": rent * 12 / inc, "affordable": rent * 12 / inc <= 0.3,
                "interval": {"low": rent - 100, "high": rent + 100, "level": 0.8},
                "contributions": [("median_annual_income", 120.0), ("school_quality__missing", -30.0)]}


@pytest.fixture
def ctx():
    db = Database(":memory:")
    pred = FakePredictor()
    app = create_app(db=db, predictor=pred, rate_limit=5, auth_limit=50)
    client = TestClient(app)
    key, prefix, h = sec.new_api_key()
    db.run("INSERT INTO api_keys (user_id, name, prefix, key_hash, created_at) VALUES (NULL,'admin',?,?,'t')",
           (prefix, h))
    return client, db, pred, {"X-API-Key": key}


def _login(client, email="kesh@example.com", pw="correct-horse-battery"):
    assert client.post("/v1/users", json={"email": email, "password": pw}).status_code == 201
    tok = client.post("/v1/auth/login", json={"email": email, "password": pw}).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


# ------------------------------------------------------------------ public + data
def test_health_and_model_are_public(ctx):
    c, *_ = ctx
    assert c.get("/v1/health").status_code == 200
    m = c.get("/v1/model").json()
    assert m["served_areas"] == 2 and "grouped_cv" in m["note"]


def test_data_endpoints_require_valid_key(ctx):
    c, db, _, H = ctx
    assert c.get("/v1/areas").status_code == 401
    assert c.get("/v1/areas", headers={"X-API-Key": "rw_wrong"}).status_code == 401
    r = c.get("/v1/areas", headers=H)
    assert r.status_code == 200 and r.json()["count"] == 2
    assert r.headers["X-RateLimit-Limit"] == "5"


def test_predict_contract_and_labels(ctx):
    c, _, _, H = ctx
    r = c.post("/v1/predict", headers=H, json={"la_code": "W06000015", "property_type": "Terraced"})
    assert r.status_code == 200
    j = r.json()
    assert j["rent_gbp_month"] == 1000 and j["interval"] == {"low": 900, "high": 1100, "level": 0.8}
    assert j["bedrooms"] == 3 and j["benchmark"] == 0.30 and "not financial advice" in j["disclaimer"]
    assert j["factors"][1]["label"].startswith("School data not available")


@pytest.mark.parametrize("body,code", [
    ({"la_code": "E09000033", "property_type": "Flat"}, 404),          # not served
    ({"la_code": "E07000223", "property_type": "Castle"}, 422),        # bad type
    ({"la_code": "'; DROP TABLE users;--", "property_type": "Flat"}, 422),  # rejected by schema
    ({"property_type": "Flat"}, 422),
])
def test_predict_rejects_bad_input(ctx, body, code):
    c, db, _, H = ctx
    assert c.post("/v1/predict", headers=H, json=body).status_code == code
    assert db.one("SELECT COUNT(*) c FROM api_keys")["c"] == 1          # DB intact


def test_rate_limit_returns_429_with_retry_after(ctx):
    c, _, _, H = ctx
    codes = [c.get("/v1/areas", headers=H).status_code for _ in range(6)]
    assert codes[:5] == [200] * 5 and codes[5] == 429
    r = c.get("/v1/areas", headers=H)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


# ------------------------------------------------------------------ accounts
def test_signup_validation_and_duplicates(ctx):
    c, *_ = ctx
    assert c.post("/v1/users", json={"email": "not-an-email", "password": "long-enough-pw"}).status_code == 422
    assert c.post("/v1/users", json={"email": "a@b.co", "password": "short"}).status_code == 422
    assert c.post("/v1/users", json={"email": "A@B.co", "password": "long-enough-pw"}).status_code == 201
    assert c.post("/v1/users", json={"email": "a@b.co", "password": "long-enough-pw"}).status_code == 409


def test_secrets_never_stored_in_plaintext(ctx):
    c, db, *_ = ctx
    H = _login(c)
    key = c.post("/v1/me/api-keys", headers=H, json={"name": "mine"}).json()["api_key"]
    u = db.one("SELECT pw_hash FROM users")
    assert u["pw_hash"].startswith("scrypt$") and "correct-horse" not in u["pw_hash"]
    tok = H["Authorization"].split()[1]
    dump = str(db.all("SELECT * FROM api_keys")) + str(db.all("SELECT * FROM sessions"))
    assert key not in dump and tok not in dump
    assert key[:11] in dump                                              # only the public prefix


def test_login_failures_are_generic(ctx):
    c, *_ = ctx
    _login(c)
    bad_pw = c.post("/v1/auth/login", json={"email": "kesh@example.com", "password": "wrong-password!"})
    no_user = c.post("/v1/auth/login", json={"email": "ghost@example.com", "password": "wrong-password!"})
    assert bad_pw.status_code == no_user.status_code == 401
    assert bad_pw.json() == no_user.json()                               # no user enumeration


def test_logout_and_expired_tokens(ctx):
    c, db, *_ = ctx
    H = _login(c)
    assert c.get("/v1/me", headers=H).status_code == 200
    assert c.post("/v1/auth/logout", headers=H).status_code == 204
    assert c.get("/v1/me", headers=H).status_code == 401
    H2 = {"Authorization": "Bearer " + c.post("/v1/auth/login", json={
        "email": "kesh@example.com", "password": "correct-horse-battery"}).json()["access_token"]}
    db.run("UPDATE sessions SET expires_at = '2000-01-01T00:00:00+00:00'")
    assert c.get("/v1/me", headers=H2).status_code == 401


def test_self_service_keys_work_and_revoke(ctx):
    c, *_ = ctx
    H = _login(c)
    k = c.post("/v1/me/api-keys", headers=H, json={"name": "app"}).json()
    assert c.get("/v1/areas", headers={"X-API-Key": k["api_key"]}).status_code == 200
    listed = c.get("/v1/me/api-keys", headers=H).json()
    assert listed[0]["prefix"] == k["prefix"] and "api_key" not in listed[0]
    assert c.delete(f"/v1/me/api-keys/{k['id']}", headers=H).status_code == 204
    assert c.get("/v1/areas", headers={"X-API-Key": k["api_key"]}).status_code == 401


def test_users_are_isolated(ctx):
    c, *_ = ctx
    A = _login(c, "a@example.com"); B = _login(c, "b@example.com")
    ka = c.post("/v1/me/api-keys", headers=A, json={"name": "a"}).json()
    al = c.post("/v1/me/alerts", headers=A, json={"la_code": "E07000223", "property_type": "Flat",
                                                 "metric": "rent", "op": "above", "threshold": 1000}).json()
    assert c.delete(f"/v1/me/api-keys/{ka['id']}", headers=B).status_code == 404
    assert c.delete(f"/v1/me/alerts/{al['id']}", headers=B).status_code == 404
    assert c.get("/v1/me/alerts", headers=B).json() == []


# ------------------------------------------------------------------ saved areas + alerts
def test_saved_areas_crud(ctx):
    c, *_ = ctx
    H = _login(c)
    assert c.post("/v1/me/saved-areas", headers=H, json={"la_code": "E07000223", "label": "home"}).status_code == 201
    assert c.post("/v1/me/saved-areas", headers=H, json={"la_code": "E07000223"}).status_code == 409
    assert c.post("/v1/me/saved-areas", headers=H, json={"la_code": "E09000033"}).status_code == 404
    rows = c.get("/v1/me/saved-areas", headers=H).json()
    assert rows[0]["la_name"] == "Adur" and rows[0]["label"] == "home"
    assert c.delete("/v1/me/saved-areas/E07000223", headers=H).status_code == 204
    assert c.get("/v1/me/saved-areas", headers=H).json() == []


def test_alert_validation(ctx):
    c, *_ = ctx
    H = _login(c)
    base = {"la_code": "E07000223", "property_type": "Flat", "metric": "ratio", "op": "above"}
    assert c.post("/v1/me/alerts", headers=H, json={**base, "threshold": 30}).status_code == 422    # % not fraction
    assert c.post("/v1/me/alerts", headers=H, json={**base, "threshold": 0.3, "metric": "price"}).status_code == 422
    assert c.post("/v1/me/alerts", headers=H, json={**base, "threshold": 0.3}).status_code == 201


def test_alerts_are_edge_triggered_and_notify_once(ctx):
    c, db, pred, _ = ctx
    H = _login(c)
    c.post("/v1/me/alerts", headers=H, json={"la_code": "E07000223", "property_type": "Flat",
                                             "metric": "rent", "op": "above", "threshold": 1050})
    sent = []

    class Capture:
        name = "test"

        def send(self, email, subject, body):
            sent.append((email, body)); return True

    s1 = evaluate_all(db, pred, Capture())                    # rent 1100 > 1050 -> fires
    s2 = evaluate_all(db, pred, Capture())                    # still true -> no repeat
    assert (s1["fired"], s2["fired"]) == (1, 0) and len(sent) == 1
    assert "£1,100" in sent[0][1] and sent[0][0] == "kesh@example.com"
    pred.rent["Adur"] = 1000; evaluate_all(db, pred, Capture())          # condition clears
    pred.rent["Adur"] = 1200; s4 = evaluate_all(db, pred, Capture())     # crosses again -> fires
    assert s4["fired"] == 1
    notes = c.get("/v1/me/notifications", headers=H).json()
    assert len(notes) == 2 and notes[0]["delivered_via"] == "test"
    assert c.post(f"/v1/me/notifications/{notes[0]['id']}/read", headers=H).status_code == 204
    assert len(c.get("/v1/me/notifications?unread_only=true", headers=H).json()) == 1


def test_ratio_alert_message():
    from src.api.alerts import describe
    msg = describe({"metric": "ratio", "property_type": "Flat", "op": "above", "threshold": 0.3}, "Adur", 0.44)
    assert "44%" in msg and "30%" in msg


def test_factor_labels():
    assert factor_label("dens_gym") == "Density of gyms"
    assert factor_label("region_London") == "Being in London"
    assert factor_label("school_quality__missing").startswith("School data not available")


def test_signup_can_be_disabled():
    app = create_app(db=Database(":memory:"), predictor=FakePredictor(), allow_signup=False)
    assert TestClient(app).post("/v1/users", json={"email": "a@b.co", "password": "long-enough-pw"}).status_code == 403


def test_auth_endpoints_are_throttled():
    app = create_app(db=Database(":memory:"), predictor=FakePredictor(), auth_limit=3)
    c = TestClient(app)
    codes = [c.post("/v1/auth/login", json={"email": "x@y.co", "password": "nope-nope-nope"}).status_code
             for _ in range(4)]
    assert codes == [401, 401, 401, 429]


def test_admin_cli_create_list_revoke(capsys):
    from src.api.admin import main
    db = Database(":memory:")
    assert main(["create-key", "--name", "partner"], db=db) == 0
    out = capsys.readouterr().out
    key = out.split(": ")[1].split()[0]
    assert key.startswith("rw_") and db.one("SELECT key_hash FROM api_keys")["key_hash"] == sec.sha256(key)
    assert main(["revoke-key", "1"], db=db) == 0
    assert db.one("SELECT revoked FROM api_keys")["revoked"] == 1
    assert main(["revoke-key", "99"], db=db) == 1


# ------------------------------------------------------------------ real model
import config  # noqa: E402

real = pytest.mark.skipif(not (config.MODELS_DIR / "rent_model_meta.json").exists(),
                          reason="spatial model not built")


@pytest.fixture(scope="module")
def client():
    db = Database(":memory:")
    key, prefix, h = sec.new_api_key()
    db.run("INSERT INTO api_keys (user_id, name, prefix, key_hash, created_at) VALUES (NULL,'t',?,?,'t')",
           (prefix, h))
    return TestClient(create_app(db=db, rate_limit=10_000)), {"X-API-Key": key}


@real
class TestAgainstRealModel:

    def test_api_matches_predictor_for_every_area(self, client):
        from src.models.predict import RentWisePredictor
        c, H = client
        p = RentWisePredictor()
        areas = c.get("/v1/areas", headers=H).json()["areas"]
        assert len(areas) == len(p.areas) == 314
        for a in areas[::13]:                                   # every 13th area, both types
            for t in ("Flat", "Terraced"):
                j = c.post("/v1/predict", headers=H, json={"la_code": a["la_code"], "property_type": t}).json()
                assert j["rent_gbp_month"] == p.predict(a["la_name"], t)["rent"]
                assert j["interval"]["low"] <= j["rent_gbp_month"] <= j["interval"]["high"]

    def test_area_detail_and_model(self, client):
        c, H = client
        d = c.get("/v1/areas/W06000015", headers=H).json()
        assert d["la_name"] == "Cardiff" and d["spatial_profile"]["school_quality"] is None
        m = c.get("/v1/model").json()
        assert m["model"] == "xgboost-spatial" and m["served_areas"] == 314

    def test_openapi_schema_lists_all_routes(self, client):
        c, _ = client
        paths = c.get("/openapi.json").json()["paths"]
        for p in ("/v1/predict", "/v1/areas", "/v1/me/alerts", "/v1/me/api-keys", "/v1/auth/login"):
            assert p in paths
