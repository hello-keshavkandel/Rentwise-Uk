# Deploying RentWise UK to Streamlit Community Cloud (Phase 10)

The app is a standard Streamlit project, so deployment is a few clicks once the
repo is on GitHub. No server admin needed.

## Prerequisites
1. The `Rentwise-UK` folder is pushed to a GitHub repo (see GITHUB_SETUP.md).
2. `requirements.txt` lists every dependency (it does).
3. The models exist. Streamlit Cloud runs `python run_pipeline.py`? No — instead,
   commit the trained models, OR add a first-run step. Simplest for a student demo:
   commit the small model files by removing `models/*.joblib` from `.gitignore`
   before pushing (a few MB), plus the small processed CSVs the dashboard reads
   (`data/processed/*.csv`) and the `reports/` figures, so the app loads the real
   trained model and results immediately. Rebuild them first with
   `python run_real_pipeline.py` (real data) or `python run_pipeline.py` (synthetic).

## Steps
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**.
3. Repository: `infokeshavkandel-bit/rentwise-uk`  ·  Branch: `main`.
4. Main file path: `app/streamlit_app.py`.
5. Click **Deploy**. First build installs the requirements (a few minutes).
6. You get a public URL like `https://rentwise-uk.streamlit.app` — put this link
   in your final report as the live artefact.

## Notes
- If the build times out installing heavy packages, pin lighter versions or
  remove `jupyter` from requirements (not needed at runtime).
- The app degrades gracefully: if a model file is missing it tells the user to run
  the pipeline, rather than crashing.
- Keep the "indicative, not financial advice" disclaimer visible (it is, in-app).

---

## Production hardening (Milestone 4)

### Health & freshness
- `python -m src.health` (or `make health`) checks the required artifacts are present and reports data/model freshness; exits non-zero if the app can't serve. It's also the container `HEALTHCHECK`.
- The dashboard shows a live **status banner** (🟢/🟡/🔴 + data/model age) and an **About & methodology** panel.

### Option A — Streamlit Community Cloud (current)
Auto-deploys from `main` on push. Keep the Python version at **3.11** and the committed
artifacts (`models/best_model.joblib`, `models/conformal.json`, `data/processed/*.csv`)
in the repo so the app serves without the raw data.

### Option B — Any container host (Render / Fly.io / Railway / a VM)
```bash
docker build -t rentwise-uk .
docker run -p 8501:8501 rentwise-uk      # http://localhost:8501
```
The image pins Python 3.11, installs `requirements.txt`, bundles the serving
artifacts, exposes 8501, and uses the health check above. Point a custom domain at
the host and enable its HTTPS.

### CI/CD
- `.github/workflows/ci.yml` runs the test suite on every push/PR.
- `.github/workflows/refresh-data.yml` runs monthly: fetch (what's automatable) → validate → publish the data manifest.

## Data store (DuckDB)

The app reads its processed data from the committed DuckDB store
`data/processed/rentwise.duckdb` (via `src.data.store.load_table`), falling back to
the processed CSVs if the store is missing, unreadable, or lacks a table. The
sidebar banner shows which backend is live (`data via duckdb` / `data via csv`).

After changing processed data, **rebuild and commit the store together with the CSVs**:

```bash
make build        # pipeline also rebuilds the store at the end
make store        # or rebuild the store on its own
git add data/processed/rentwise.duckdb data/processed/*.csv
```

`tests/test_store.py::test_committed_store_matches_processed_csvs` fails CI if the
committed store and CSVs drift apart. Keep `duckdb` pinned in `requirements.txt`
(currently 1.5.6) — newer DuckDB reads older store files, but not the reverse.

## Deployed rent model (Modules 2 + 3)

The live estimator uses `models/rent_xgb_spatial.json` (portable XGBoost JSON) + `models/rent_model_meta.json`
+ `models/conformal.json`, built by `make spatial`. If those files are absent the app falls back to the legacy
`best_model.joblib`.

**Version rule:** retrain with the **same XGBoost version pinned in `requirements.txt`** (currently 2.1.1).
A model written by a newer major version may not load on Streamlit Cloud. Simplest way:

```bash
python3 -m venv /tmp/venv_deploy && /tmp/venv_deploy/bin/pip install -r requirements.txt
/tmp/venv_deploy/bin/python -m src.models.rent_spatial
```

`tests/test_predict_spatial.py::test_model_saved_with_the_deployed_xgboost_version` fails CI if this drifts.

## Public REST API (Milestone 5)

```bash
pip install -r requirements.txt -r requirements-api.txt
python -m src.api.admin create-key --name me          # bootstrap/service key (shown once)
uvicorn src.api.app:app --host 0.0.0.0 --port 8000    # or: docker compose up --build
```

| Setting (env) | Default | Purpose |
|---|---|---|
| `RENTWISE_API_DB` | `data/api/rentwise_api.sqlite3` | users, keys, saved areas, alerts, notifications (git-ignored; mount a volume) |
| `RENTWISE_RATE_LIMIT` | `60` | requests per minute per API key |
| `RENTWISE_ALLOW_SIGNUP` | `1` | set `0` to disable public sign-up |
| `RENTWISE_SESSION_DAYS` | `7` | login token lifetime |
| `RENTWISE_SMTP_HOST`/`_PORT`/`_USER`/`_PASSWORD`/`_FROM` | unset | email delivery for alerts (else stored + logged) |

**Alerts** are evaluated by a job, not on request: run `python -m src.api.alerts` (or
`docker compose run --rm api python -m src.api.alerts`) on the API host after each monthly data refresh
(e.g. cron `0 9 3 * *`). Alerts are edge-triggered: one notification when a condition becomes true.

**Run behind HTTPS** (a reverse proxy such as Caddy/Nginx or the platform's TLS); the API relies on TLS for
transport security of keys and tokens.

**Known single-instance limits** (fine for launch, change before scaling out): the rate limiter is in-process
(use Redis with >1 replica) and storage is SQLite (move to Postgres; the SQL is plain and parameterised).
The API serves only the 314 trained areas because its intervals are calibrated for those.
