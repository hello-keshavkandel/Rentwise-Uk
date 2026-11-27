# RentWise UK 🏠

**A data-driven UK housing & long-term rental intelligence system — predict, explain, and measure affordability from open government data.**

[![CI](https://github.com/hello-keshavkandel/Rentwise-Uk/actions/workflows/ci.yml/badge.svg)](https://github.com/hello-keshavkandel/Rentwise-Uk/actions/workflows/ci.yml)
&nbsp;**Live app:** https://rentwiseuk.streamlit.app

RentWise UK predicts long-term rental prices and residential sale prices across England & Wales, expresses rent as an **affordability measure against local incomes**, **explains every estimate with SHAP**, attaches a **calibrated confidence range**, and forecasts regional trends — all from **open data only** (ONS, HM Land Registry, data.police.uk, OpenStreetMap, Ofsted/DfE). Four interconnected ML modules feed an interactive Streamlit dashboard and a **public REST API** with accounts and rent alerts.

> Originally an MSc project; now developed as a solo product by **Keshav Kandel**. The dissertation state is preserved at the `dissertation-submission` git tag.

---

## What it does

| Module | What | Headline |
|---|---|---|
| **2 · Rent & affordability** (deployed, with Module 3 features) | **Official ONS average rent first** (Oct 2026, 316 areas × 4 types); model explains it with SHAP and estimates only where ONS publishes none; affordability vs the 30% benchmark | Model R² **0.945** / MAE £79 for served areas, **0.858** / £152 for unseen areas; 80% interval ±£106, coverage 80% |
| **1 · Sale price** | HM Land Registry price model enriched with **EPC floor area + rooms** (86% of sales matched) | XGBoost R² **0.568 → 0.732** with EPC; 90% conformal intervals |
| **3 · Geospatial** | 11 OSM amenity densities (506k amenities) + historic Ofsted school quality + crime per 1,000 residents + LA boundary maps | Unseen-area MAE **−8%** (£169 → £156) every repeat; R² gain small (0.847 → 0.853); choropleths |
| **4 · Forecasting** | ARIMA / Holt-Winters regional rent trends; rolling-origin backtest + conformal intervals | mean MAPE ~0.6%, recalibrated 80/95% bands |

The dashboard has five views:

- **Rent check** — the official ONS average rent for any area by **bedroom count (1–4+) or property type**, with its annual change; affordability against **your own income** (or local full-time median pay, the ONS basis) and the rent you could afford at 30%; median **sale price and gross yield**; the **trend since 2015** against region and nation; the region's **12-month forecast** with honest bands; and what drives rent there (SHAP).
- **Compare areas** — 2–4 areas side by side (rent, change, affordability, prices, yield, crime, schools, amenities) with trends and **CSV download**.
- **Area map** — official affordability, rent, rent change, yield, school quality and 11 amenity layers.
- **Methodology & data** — what every number is, its source, date and limits.
- **System results** — the research modules' latest results, with a live **health/freshness banner**.

## Public API (Milestone 5)

```bash
pip install -r requirements.txt -r requirements-api.txt
python -m src.api.admin create-key --name me      # prints an API key once
make api                                          # http://localhost:8000/docs
curl -X POST localhost:8000/v1/predict -H "X-API-Key: rw_..." -H 'Content-Type: application/json' \
     -d '{"la_code":"E09000007","property_type":"Flat"}'
```

Official rents for every measure with affordability against your income, sale prices and yield (`/v1/areas/{code}/rents`), trends since 2015 (`/trend`), regional forecasts (`/forecast`), side-by-side comparison (`/v1/compare`), and rent estimates with calibrated intervals and SHAP factors (`/v1/predict`); self-service **accounts**, API keys, **saved areas** and **rent / affordability alerts** (`/v1/me/...`, evaluated by `make alerts` after each refresh). Hashed secrets, per-key rate limiting, served-areas-only by design. Full report: `reports/REPORT_P1_M5.md`.

## Quickstart

```bash
git clone https://github.com/hello-keshavkandel/Rentwise-Uk.git && cd Rentwise-Uk
make setup        # builds .venv with Python 3.11 + all dependencies (uses uv if installed)
make verify       # proves it works: env, health, all tests, live API + live dashboard
make demo         # dashboard http://localhost:8501 + API http://localhost:8000/docs, demo account included
```

`make verify` prints a PASS / WARN / FAIL table (also saved to `reports/verify_report.md`) and is what CI runs on
every push. `make help` lists every target, including the data pipeline (`make build`, `make spatial`, …). Raw
data is git-ignored — `DATA_SOURCES.md` lists exactly what to download and from where.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `make setup` says Python 3.11 was not found | `brew install uv` (recommended — it fetches Python 3.11 for you) or `brew install python@3.11`, then `make setup` |
| `pip` tries to compile numpy/pandas | You are on Python 3.12+; the pins need 3.11 — use `make setup`, which creates a 3.11 `.venv` |
| `make verify` warns about package versions | Run `make setup` again inside the project; the warning lists what differs |
| Port 8000 or 8501 already in use | `python scripts/demo.py --api-port 8010 --app-port 8510` |
| Dashboard shows "Could not load the trained model" | Run `make verify` — the Health line names the missing file |
| Docker: `docker compose up --build` fails | Make sure Docker Desktop is running; images use Python 3.11 and need ~2 GB free |

## Architecture

```
open data ─▶ fetch + validate ─▶ shared feature store ─▶ four ML modules ─┬▶ Streamlit dashboard
 (sources.py)   (_manifest.json)     (data contract)      (1 sale · 2 rent ·  │  (estimator, map, results)
                                      + DuckDB store)       3 geo · 4 forecast)└▶ FastAPI public API
                                                                                 (keys, accounts, alerts)
```

- **Object-oriented pipeline** — each stage is a class with a `run()` method; the whole system rebuilds from `python run_real_pipeline.py` (`make build`).
- **Provenance & validation** — `src/data/sources.py` records bytes/sha256/row-counts/validation per source in `data/raw/_manifest.json`.
- **Processed store** — `src/data/store.py` (DuckDB) is the app's **primary data source**: `data/processed/rentwise.duckdb` is committed and deployed, read via `load_table()` with automatic CSV fallback. The pipeline rebuilds it after every run; a test guards it against drifting from the CSVs.
- **Calibrated uncertainty** — `src/models/conformal.py` (cross-conformal) gives honest intervals; empirically 90%→90%, 80%→80% on real data.
- **Metrics history** — `src/models/registry.py` logs every run to `reports/metrics_history.jsonl`.
- **Maps** — `src/viz/maps.py` renders folium choropleths from LA-boundary GeoJSON.

## Data sources

Open data only, no proprietary access, no ethical approval required — HM Land Registry Price Paid, ONS Price Index of Private Rents, ONS ASHE earnings, data.police.uk crime, ONS geography; optional EPC (floor area), OpenStreetMap & Ofsted (geospatial). See **`DATA_SOURCES.md`**.

## Testing

```bash
make verify      # env + health + 140 tests (unit, integration, data-quality, dashboard, API) + live servers
```
CI runs the suite on every push. Per-milestone test reports are in `reports/TESTING_REPORT_*.md`.

## Deployment

- **Streamlit Community Cloud** (live) — auto-deploys from `main`; Python 3.11.
- **Any container host** — `docker compose up --build` runs the dashboard (:8501) and the API (:8000) together; or build each from `Dockerfile` / `Dockerfile.api`.

See **`DEPLOYMENT.md`** and **`ROADMAP.md`** (the product roadmap).

## Licence & author

MIT (see `LICENSE`). Built by **Keshav Kandel** — *indicative estimates, not financial advice.*
