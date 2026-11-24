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
| **2 · Rent & affordability** (deployed, with Module 3 features) | Predicts LA-level rents, scores them vs the 30% income benchmark, explains with SHAP | XGBoost **R² 0.947** / MAE £68 for served areas (**0.817** / £136 for unseen areas); **80% interval ±£113**, coverage 80% |
| **1 · Sale price** | HM Land Registry price model enriched with **EPC floor area + rooms** (86% of sales matched) | XGBoost R² **0.568 → 0.732** with EPC; 90% conformal intervals |
| **3 · Geospatial** | 11 OSM amenity densities (506k amenities) + historic Ofsted school quality + LA boundary maps | Unseen-LA rent R² **0.775 → 0.833**, MAE −14%; choropleths |
| **4 · Forecasting** | ARIMA / Holt-Winters regional rent trends; rolling-origin backtest + conformal intervals | mean MAPE ~0.6%, recalibrated 80/95% bands |

The dashboard has three views: a **Rent estimator** (local authority + property type → predicted rent, affordability verdict, **confidence range**, plain-English SHAP explanation and a *what's nearby* profile), an **Area map** (affordability, school quality and 11 amenity layers by local authority) and a **System results** overview — with a live **health/freshness banner**.

## Public API (Milestone 5)

```bash
pip install -r requirements.txt -r requirements-api.txt
python -m src.api.admin create-key --name me      # prints an API key once
make api                                          # http://localhost:8000/docs
curl -X POST localhost:8000/v1/predict -H "X-API-Key: rw_..." -H 'Content-Type: application/json' \
     -d '{"la_code":"E09000007","property_type":"Flat"}'
```

Rent estimates with calibrated intervals, affordability and SHAP factors (`/v1/predict`, `/v1/areas`); self-service **accounts**, API keys, **saved areas** and **rent / affordability alerts** (`/v1/me/...`, evaluated by `make alerts` after each refresh). Hashed secrets, per-key rate limiting, served-areas-only by design. Full report: `reports/REPORT_P1_M5.md`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

make validate     # check/validate the open data in data/raw (see DATA_SOURCES.md)
make build        # run the real-data pipeline (all four modules)
make store        # build the DuckDB query store
make app          # launch the dashboard
make api          # launch the public REST API (needs requirements-api.txt)
make test         # run the test suite
make health       # artifacts + data/model freshness
```

Run `make help` for every target. The real raw data is git-ignored — `DATA_SOURCES.md` lists exactly what to download and from where.

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
make test        # pytest — 105 tests (unit, integration, API end-to-end); data-dependent ones skip without the raw data
```
CI runs the suite on every push. Per-milestone test reports are in `reports/TESTING_REPORT_*.md`.

## Deployment

- **Streamlit Community Cloud** (live) — auto-deploys from `main`; Python 3.11.
- **Any container host** — `docker compose up --build` runs the dashboard (:8501) and the API (:8000) together; or build each from `Dockerfile` / `Dockerfile.api`.

See **`DEPLOYMENT.md`** and **`ROADMAP.md`** (the product roadmap).

## Licence & author

MIT (see `LICENSE`). Built by **Keshav Kandel** — *indicative estimates, not financial advice.*
