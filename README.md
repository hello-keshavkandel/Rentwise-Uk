# RentWise UK

A data-driven housing & long-term rental intelligence system for the UK market.
MSc Masters Project (Group 27), University of the West of Scotland. Open data only — no ethical approval required.

## What it does
Predicts UK long-term rental prices and residential sale prices, expresses rent as an affordability measure against local incomes, explains the drivers with SHAP, analyses geospatial drivers, forecasts regional trends, and presents the results in an interactive Streamlit dashboard — four interconnected machine-learning modules over one shared open-data pipeline.

## Project structure
```
Rentwise-UK/
├── config.py              # central settings (paths, thresholds, success criteria)
├── requirements.txt       # Python dependencies
├── run_pipeline.py        # SYNTHETIC development build (all four modules)
├── run_real_pipeline.py   # REAL-DATA build (all four modules) — see below
├── DATA_SOURCES.md        # data dictionary: what/where/why for every dataset
├── data_contract.md       # shared schema between modules
├── data/
│   ├── raw/               # source files (git-ignored): real ONS/HMLR/police + lookup
│   └── processed/         # cleaned & merged data
├── notebooks/             # exploration (EDA)
├── src/
│   ├── data/              # data generation, cleaning, real-data loading
│   ├── features/          # feature engineering
│   ├── models/            # training, evaluation, SHAP, affordability, prediction
│   └── modules/           # Modules 1 (sale price), 3 (geospatial), 4 (forecasting)
├── tests/                 # unit tests (data contract, affordability engine)
├── models/                # saved trained model (best_model.joblib)
├── reports/               # metrics, summaries, and figures
└── app/                   # Streamlit dashboard (app/streamlit_app.py)
```

## Setup (once)
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python config.py                     # sanity-check settings
```

## Two ways to build the system

**A — Synthetic development build** (instant; builds/tests the whole pipeline without any downloads):
```bash
python run_pipeline.py               # generates synthetic data and runs all four modules
```

**B — Real-data build** (the results reported in the dissertation). Place these real open files in `data/raw/`:

| File in `data/raw/` | Source | Used by |
|---|---|---|
| `ons_pipr_monthly.xlsx` | ONS Price Index of Private Rents — monthly price statistics | Module 2 (rent by LA & property type) |
| `ons_pipr_historical.xlsx` | ONS PIPR historical series (monthly, from 2005) | Module 4 (forecasting) |
| `ashe_table8_7a.xlsx` | ONS ASHE Table 8.7a — annual gross pay by LA of residence | Module 2 (affordability) |
| `land_registry_pp_2026.csv` | HM Land Registry Price Paid | Module 1 (sale price) |
| `police/*street.csv` | data.police.uk street-level crime | Module 3 (crime driver) |
| `lsoa_to_lad.csv` | ONS LSOA(2021)→LAD(2024) lookup | maps crime LSOA → local authority |

Then:
```bash
python run_real_pipeline.py          # builds + evaluates all four modules on real data
streamlit run app/streamlit_app.py   # launch the dashboard
```

## Results (real open data)
- **Module 2 — Rent & affordability:** XGBoost **R² 0.900** (RMSE £137), beating a Ridge baseline (R² 0.804) by **29%** — both success criteria met. Top driver: local income. ~95% of local authorities exceed the 30% affordability benchmark.
- **Module 1 — Sale price:** XGBoost **R² 0.567** on a 90,000-transaction sample of ~216,000 cleaned HM Land Registry sales; includes SHAP importance (local area price level and detached type dominate) and a leave-one-region-out cross-regional test (mean RMSE ~£184k — a deliberately strict generalisation check). Lower R² reflects open data lacking floor area/bedrooms — a documented limitation.
- **Module 4 — Forecasting:** ARIMA / Holt-Winters, mean **MAPE ≈ 0.6%** across nine regions, with **80%/95% prediction intervals** (empirical coverage 68% / 89% — intervals slightly over-confident).
- **Module 3 — Geospatial:** real crime driver (SHAP ≈ £57; largely collinear with income/region). Transport, schools, green space, amenities and interactive maps are future work (need OSM/Ofsted).

*Development note: the system was built iteratively — first on synthetic data (`run_pipeline.py`) to develop and test the pipeline, then validated on real open data (`run_real_pipeline.py`).*

## Team (per project specification)
- **Keshav Kandel** — technical lead: system design, development, testing, integration (Module 2: rental price & affordability).
- **Abdullah** — Module 1 (sale price): data sourcing/documentation, results interpretation, write-up.
- **Milan** — Module 3 (geospatial): geospatial features/analysis, interpretation, write-up.
- **Harish** — Module 4 (forecasting + dashboard): requirements, usability testing, deployment docs, project management.

## Dashboard
`streamlit run app/streamlit_app.py` opens two views: a **Rent estimator** (pick a local authority and property type — the bedroom count is derived from the type, matching the ONS data — returning predicted rent, affordability verdict and a SHAP explanation) and a **System results** overview of all four modules.

## Status
0. Setup ✅  1. Data ✅  2. Cleaning ✅  3. EDA ✅  4. Features ✅  5. Modelling ✅  6. Evaluation ✅  7. SHAP ✅  8. Affordability ✅  9. Dashboard ✅  · Real-data validation ✅ · 10. Deploy ▶ (Streamlit Community Cloud)

## Tests
```bash
python -m pytest tests/              # data-contract and affordability-engine unit tests
```
