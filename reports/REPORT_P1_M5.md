# RentWise UK — Report: Priority 1 (spatial model live) + Milestone 5 (public API, accounts, alerts)

**Author:** Keshav Kandel · **Date:** 24 November 2026 · **Status:** complete, tested, validated

---

## 1. Summary

Two pieces of work were delivered, tested and validated:

1. **Priority 1 — the improved rent model is now live.** The rent estimator now uses Module 3's real
   spatial features (OpenStreetMap amenities + historic Ofsted school quality) alongside the Module 2
   features. Accuracy improved on both the served areas and unseen areas, and the confidence range got about
   20% tighter while staying correctly calibrated.
2. **Milestone 5 — RentWise is now a product with an API.** A public REST API serves estimates with
   API keys and rate limiting; users can sign up, create their own keys, save areas and set rent or
   affordability alerts, which a scheduled job evaluates and delivers.

Final state: **105 automated tests passing**, the dashboard verified in all three views, the API verified as
a running server over HTTP, and the model verified in an environment identical to the live deployment.

---

## 2. Priority 1 — deploying the spatial rent model

### 2.1 Results (real data, 1,198 rows, 314 local authorities)

| | Previous live model | **New live model (Modules 2+3)** | Change |
|---|---|---|---|
| R², areas seen in training (what the app serves) | 0.913 | **0.947** | +0.034 |
| MAE, areas seen in training | £89 | **£68** | −24% |
| R², whole areas held out (unseen) | 0.775 | **0.817** | +0.042 |
| MAE, unseen areas | £152 | **£136** | −11% |
| 80% confidence range | ±£140 | **±£113** | ≈20% tighter |
| Coverage of the 80% / 90% range (served areas) | 80% / 90% | **80.1% / 90.2%** | on target |

All figures are out-of-fold (5-fold cross-validation). The "unseen" rows hold out whole local authorities,
because each area appears up to four times (once per property type) and a random split would let the model
recognise an area it has already seen.

**What drives estimates (mean absolute SHAP impact):** local income (£191), region (£127), bedrooms (£122),
property type (£55), then gym density (£39), school quality (£36), rail-station density (£34), hospital
density (£27), park density (£17) and crime (£15). These are associations across areas, not causes —
gym density, for example, is most likely a marker of affluent urban areas.

### 2.2 What changed in the product

- **Model artefacts:** `models/rent_xgb_spatial.json` (portable XGBoost JSON, no pickle),
  `models/rent_model_meta.json` (feature order, training time, full evaluation) and a recalibrated
  `models/conformal.json`. The legacy model remains as an automatic fallback.
- **Dashboard:** a new **Area map** view (affordability, school quality and 11 amenity layers by area);
  a **What's nearby** profile under each estimate; readable labels for every factor; updated About panel.
  *System results* now shows the deployed model's real summary, SHAP chart and Module 3 ablation chart.
- **Honest explanations:** where a feature has no data for an area (Ofsted doesn't inspect Welsh schools —
  Estyn does), the explanation says *"School data not available (Wales — schools are inspected by Estyn)"*
  instead of presenting the model's missing-value handling as a fact about the area.
- **Health banner:** model age now comes from the recorded training time (file dates reset on every deploy,
  which previously showed a wrong age).
- **Pipeline:** fixed a bug where the full pipeline overwrote the real Module 3 summary with an old
  crime-only text; `make spatial` and the full pipeline now rebuild Module 3 and the deployed model.

### 2.3 Validation

| Check | Result |
|---|---|
| Every served area × every property type (314 × 4 = 1,256 predictions) | all plausible (£250–£10,000) and inside their own interval |
| Same sweep in an environment identical to the live deployment (XGBoost 2.1.1, numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1) | all 1,256 pass; identical values |
| Streamlit `AppTest`: Rent estimator (Cardiff, Camden), Area map (two layers), System results | renders with **no exceptions** |
| Unknown area / invalid property type | clear `UnknownAreaError` / `ValueError` |

**A deployment fault was caught and fixed during validation.** The model had first been saved by XGBoost
3.2.0, while the live app pins 2.1.1 — an older XGBoost is not guaranteed to load a newer model file, which
could have broken the live app on deploy. The model was retrained in a deployment-matched environment
(identical metrics), now loads in both versions, and a new test fails CI if the versions drift again.

---

## 3. Milestone 5 — public API, accounts and alerts

### 3.1 Endpoints (`uvicorn src.api.app:app`, interactive docs at `/docs`)

| Area | Endpoints | Auth |
|---|---|---|
| Public | `GET /v1/health`, `GET /v1/model` (deployed model + honest metrics) | none |
| Data | `GET /v1/areas`, `GET /v1/areas/{la_code}`, `POST /v1/predict` | `X-API-Key` |
| Accounts | `POST /v1/users`, `POST /v1/auth/login`, `POST /v1/auth/logout`, `GET /v1/me` | bearer token |
| API keys | `POST/GET /v1/me/api-keys`, `DELETE /v1/me/api-keys/{id}` | bearer token |
| Saved areas | `GET/POST /v1/me/saved-areas`, `DELETE /v1/me/saved-areas/{la_code}` | bearer token |
| Alerts | `GET/POST /v1/me/alerts`, `DELETE /v1/me/alerts/{id}`, `GET /v1/me/notifications`, `POST /v1/me/notifications/{id}/read` | bearer token |

`POST /v1/predict` returns the rent, the calibrated 80% interval, rent-to-income vs the 30% benchmark,
the top SHAP factors with plain-English labels, the model name and a not-financial-advice disclaimer.

**Alerts** watch one area + property type and fire when the predicted rent or the rent-to-income ratio goes
above or below a threshold. They are **edge-triggered**: one notification when the condition becomes true,
not one every run. A job (`make alerts`) evaluates them after each data refresh; notifications are always
stored for the API, and are emailed when SMTP is configured.

### 3.2 Security design

| Concern | How it's handled |
|---|---|
| Passwords | scrypt (memory-hard) with a per-user random salt; constant-time comparison |
| API keys & session tokens | 256-bit random secrets; **only SHA-256 hashes stored**; plaintext shown once; public prefix for listings |
| Brute force | login/sign-up throttled per client; data endpoints rate-limited per key (`429` + `Retry-After`) |
| User enumeration | identical error and similar timing for "wrong password" and "no such user" |
| Injection | every query parameterised; inputs validated by schema (e.g. area codes must match `^[EW]\d{8}$`) |
| Isolation | every account query is scoped to the caller; cross-user access returns `404` |
| Error leakage | unhandled errors return a generic `500`; details go to the server log only |
| Abuse limits | 5 active keys, 50 saved areas, 20 active alerts per user |
| Containers | API image runs as an unprivileged user; only its database directory is writable |
| Statistical honesty | only the 314 trained areas are served, because intervals are calibrated for them (see §4) |

### 3.3 Validation

| Check | Result |
|---|---|
| API test suite (contract, bad input incl. SQL-injection attempt, rate limit, secret storage, generic login errors, logout & expiry, key revocation, user isolation, saved areas, alert validation, edge-triggered alerts, admin CLI, auth throttling) | **25 / 25 pass** |
| API vs dashboard predictor on the real model (every 13th area × 2 types) | identical rents; intervals contain the estimate |
| **Live server** (uvicorn over HTTP) | health 200; Camden flat → £2,381 (£2,268–£2,493), 65% of income; no key → 401; `/docs` → 200; sign-up, login, alert creation OK |
| **Alert job** on the live database | first run fired: *"Rent-to-income for a flat in Camden is now 65% — above your 50% threshold."*; second run: no duplicate |

---

## 4. Test evidence

**105 tests pass** (`python -m pytest -q`). New in this round: 25 API, 6 deployed-predictor, 4 spatial-model
training, plus a health test. Docker images could not be built in the development sandbox (no Docker there);
the Dockerfiles and `docker-compose.yml` follow the existing, already-working image and were syntax-checked —
**build them once locally** (`docker compose up --build`) before relying on them.

| Test file | Tests | Covers |
|---|---|---|
| `test_api.py` | 25 | the whole public API, security behaviour, alerts, admin CLI, real-model parity |
| `test_store.py` | 9 | DuckDB store, CSV fallback, drift guard |
| `test_epc_bulk.py` | 7 | EPC streaming from zip, postcode filter, per-file reduction |
| `test_predict_spatial.py` | 6 | deployed model over all 314 × 4, Welsh labelling, errors, **XGBoost version guard** |
| `test_health.py` | 6 | health status, data backend, model age from metadata |
| `test_rent_spatial.py` | 4 | training frame, calibration coverage, portable artefacts |
| others (13 files) | 48 | data sources, conformal, forecasting, geo, schools, OSM, Module 3, maps, registry |

---

## 5. Limitations (stated, not hidden)

1. **Unseen areas:** the 80% range covers only **54%** of truths for areas the model never saw (68% for 90%).
   The app and API therefore serve only the 314 trained areas. Serving new areas needs wider, separately
   calibrated intervals.
2. **Single instance:** the rate limiter is in-process and storage is SQLite — correct for one server;
   move to Redis + Postgres before running several replicas.
3. **Accounts are minimal:** no email verification or password reset yet.
4. **Spatial features:** amenity density is per km² (favours compact urban areas; per-capita needs ONS
   population); OSM completeness varies by area; Ofsted grades are historic (median inspection 2023); Wales
   has no school data (Estyn not yet integrated).
5. **Area-level model:** estimates describe a typical property of a type in an area, not a specific home.
6. **Docker images** not built in the sandbox (see §4).

---

## 6. How to run and deploy

```bash
make app                                         # dashboard
pip install -r requirements-api.txt && make api  # API at http://localhost:8000/docs
python -m src.api.admin create-key --name me     # first API key
make alerts                                      # evaluate alerts (schedule monthly after the refresh)
docker compose up --build                        # dashboard :8501 + API :8000
```

Streamlit Cloud picks up the new model automatically from `main`. The API needs a host that runs a
container or Python process (behind HTTPS); configuration is via environment variables (see `DEPLOYMENT.md`).

---

## 7. Recommended next steps

1. **Deploy the API** to a small cloud host behind HTTPS and give it a persistent volume.
2. **Add email verification and password reset**, then connect SMTP so alerts reach inboxes.
3. **Interval policy for unseen areas** (grouped-conformal widths) so more areas can be served safely.
4. **Per-capita amenity features** (ONS mid-year population) and **Estyn** grades for Wales.
5. **Postgres + Redis** when there is real traffic; add API usage metrics.
