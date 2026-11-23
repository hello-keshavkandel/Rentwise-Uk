# RentWise UK 🏠

**A data-driven UK housing & long-term rental intelligence system — predict, explain, and measure affordability from open government data.**

[![CI](https://github.com/hello-keshavkandel/Rentwise-Uk/actions/workflows/ci.yml/badge.svg)](https://github.com/hello-keshavkandel/Rentwise-Uk/actions/workflows/ci.yml)
&nbsp;**Live app:** https://rentwiseuk.streamlit.app

RentWise UK predicts long-term rental prices and residential sale prices across England & Wales, expresses rent as an **affordability measure against local incomes**, **explains every estimate with SHAP**, attaches a **calibrated confidence range**, and forecasts regional trends — all from **open government data only** (ONS, HM Land Registry, data.police.uk). Four interconnected ML modules feed one interactive Streamlit dashboard.

> Originally an MSc project; now developed as a solo product by **Keshav Kandel**. The dissertation state is preserved at the `dissertation-submission` git tag.

---

## What it does

| Module | What | Headline |
|---|---|---|
| **2 · Rent & affordability** | Predicts LA-level rents, scores them vs the 30% income benchmark, explains with SHAP | XGBoost **R² 0.90**, +29% over baseline; **calibrated 80% intervals** |
| **1 · Sale price** | HM Land Registry price model enriched with **EPC floor area + rooms** (86% of sales matched) | XGBoost R² **0.568 → 0.732** with EPC; 90% conformal intervals |
| **3 · Geospatial** | Spatial drivers of rent (crime live; OSM/Ofsted + maps in progress) | SHAP + ablation; interactive choropleths |
| **4 · Forecasting** | ARIMA / Holt-Winters regional rent trends; rolling-origin backtest + conformal intervals | mean MAPE ~0.6%, recalibrated 80/95% bands |

The dashboard has two views: a **Rent estimator** (pick a local authority + property type → predicted rent, affordability verdict, a **confidence range**, and a plain-English SHAP explanation) and a **System results** overview of all four modules, with a live **health/freshness banner**.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

make validate     # check/validate the open data in data/raw (see DATA_SOURCES.md)
make build        # run the real-data pipeline (all four modules)
make store        # build the DuckDB query store
make app          # launch the dashboard
make test         # run the test suite
make health       # artifacts + data/model freshness
```

Run `make help` for every target. The real raw data is git-ignored — `DATA_SOURCES.md` lists exactly what to download and from where.

## Architecture

```
open data ─▶ fetch + validate ─▶ shared feature store ─▶ four ML modules ─▶ Streamlit dashboard
 (sources.py)   (_manifest.json)     (data contract)      (1 sale · 2 rent ·      (+ health banner,
                                      + DuckDB store)       3 geo · 4 forecast)     SHAP, intervals, maps)
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
make test        # pytest — 60 tests (unit + integration); data-dependent ones skip without the raw data
```
CI runs the suite on every push. Per-milestone test reports are in `reports/TESTING_REPORT_*.md`.

## Deployment

- **Streamlit Community Cloud** (live) — auto-deploys from `main`; Python 3.11.
- **Any container host** — `docker build -t rentwise-uk . && docker run -p 8501:8501 rentwise-uk`.

See **`DEPLOYMENT.md`** and **`ROADMAP.md`** (the product roadmap).

## Licence & author

MIT (see `LICENSE`). Built by **Keshav Kandel** — *indicative estimates, not financial advice.*
