# Update — Real-data validation complete

The four modules have now been validated on **real open data** via `run_real_pipeline.py` (ONS Price Index of Private Rents, ONS ASHE earnings, HM Land Registry Price Paid, data.police.uk crime + ONS LSOA→LAD lookup). Headline real results:

- **Module 2 (rent & affordability):** XGBoost **R² 0.900** (RMSE £137), beating the Ridge baseline (R² 0.804) by **29%** — both success criteria met on real data. Top driver: local income. ~95% of local authorities exceed the 30% affordability benchmark.
- **Module 1 (sale price):** XGBoost R² 0.567 on ~216,000 HM Land Registry transactions (open data lacks floor area/bedrooms — documented limitation).
- **Module 4 (forecasting):** ARIMA/Holt-Winters, mean MAPE ≈ 0.6% across nine regions.
- **Module 3 (geospatial):** real crime driver (SHAP ≈ £57, largely collinear with income/region); transport/schools/green space/amenities and maps remain future work (OSM/Ofsted).

The 'swap synthetic data for real ONS/HM Land Registry data and re-run' item below is therefore **DONE**. The sections that follow describe the earlier synthetic-development state and remain accurate as project history.

---

# RentWise UK (Module 2) — Code Review & Validation (through Phase 9)

I inspected every file, refactored the whole codebase, re-ran it end to end, and
validated it against the build scope and dissertation plan. This is the record.

## 1. What I changed in this review pass
- **Object-Oriented refactor.** Before, only `predict.py` used a class; the other
  modules were procedural. I converted **every module into a class** with a clear
  responsibility and a `run()` method, so the code now follows OOP consistently.
- **First-person, per-block comments.** Every module docstring and code block now
  explains what I do and *why*, in the first person, concisely.
- **Shared, reusable engines.** `RegressionMetrics`, `DataSplitter` and
  `AffordabilityEngine` are used by several stages, so logic lives in one place.
- **Pipeline orchestrator.** New `run_pipeline.py` runs all phases in order by
  composing the classes — one command reproduces every result.
- **Re-ran the full pipeline** — identical results to before (Test RMSE £66,
  R² 0.932), proving the refactor changed structure, not behaviour.

## 2. The OOP structure now
| File | Class | Responsibility |
|---|---|---|
| data/make_sample_data.py | `SampleDataGenerator` | Phase 1 synthetic data |
| data/clean_data.py | `DataCleaner` | Phase 2 cleaning |
| data/download_data.py | `DataDownloader` | Phase 1 real-data download |
| features/build_features.py | `FeatureBuilder` | Phase 4 features |
| analysis/eda.py | `DataExplorer` | Phase 3 EDA |
| models/prepare.py | `DataSplitter` | shared train/val/test split |
| models/metrics.py | `RegressionMetrics` | shared scoring |
| models/train_baseline.py | `BaselineTrainer` | Phase 5 baseline |
| models/train_advanced.py | `AdvancedTrainer` | Phase 5 RF + XGBoost |
| models/evaluate.py | `ModelEvaluator` | Phase 6 evaluation |
| models/explain_shap.py | `ShapExplainer` | Phase 7 SHAP |
| models/affordability.py | `AffordabilityEngine` | Phase 8 affordability |
| models/predict.py | `RentWisePredictor` | inference for the app |
| run_pipeline.py | `Pipeline` | orchestrates all phases |

Note: `config.py` is intentionally a settings module (constants), which is the
standard, accepted pattern for configuration and is imported everywhere.

## 3. Strengths (what is already good)
- Clean separation of concerns; each phase isolated and independently runnable.
- Reproducible (fixed seed, documented steps, single command).
- Honest methodology: baseline comparison, validation-based model selection,
  held-out test, cross-regional validation, and explicit success criteria.
- Transparency built in (SHAP) and an ethics disclaimer in the app.
- Secure dashboard: constrained inputs only, model loaded read-only and cached.

## 4. Things I could still do better (suggestions)
1. **Automated tests** — add a small `pytest` suite (e.g. affordability maths,
   feature-column count, predictor output keys). Currently I validate by re-running.
2. **Persist the feature schema** — save the income-band edges and column order to
   a file so `predict.py` doesn't recompute them from the CSV.
3. **Type hints + docstring args/returns** on every public method (mostly done).
4. **Logging** instead of `print` for a more professional feel (optional).
5. **Config as a frozen dataclass** if I want strict OOP for settings too.

## 5. Tasks still remaining (within/after this implementation phase)
- **Real data ingestion (Phase 1, Step 3)** — download the real ONS/HM Land
  Registry files and re-run; currently synthetic only (clearly labelled).
- **Explicit multi-table merge (Phase 2, Step 5)** — the synthetic data is born as
  one table, so a real `merge` on `la_code` is written when real sources arrive.
- **Regenerate models locally** — so the saved `.joblib` matches the local
  scikit-learn version (removes the version-mismatch warning).
- **Phase 10 — Deploy & finalise** — Streamlit Community Cloud + repo tidy-up.
- **GitHub push** — still outstanding.

## 6. Validation against the build scope & dissertation plan
| Plan phase / step | Required | Status | Evidence |
|---|---|---|---|
| 0 Setup | env + repo | DONE | config, requirements, structure |
| 1 Data acquisition | sources + downloader | MECHANISM built; synthetic in use | download_data.py, DATA_SOURCES.md |
| 2 Clean & merge | clean; merge on LA | Clean DONE; merge pending real data | clean_data.py |
| 3 EDA | stats + charts | DONE | eda.py, 4 figures, eda_summary.md |
| 4 Features | encodings, bands | DONE | build_features.py, features.csv |
| 5 Modelling | baseline + RF + XGBoost + tuning | DONE | train_baseline/advanced, comparison |
| 6 Evaluation | metrics + cross-regional | DONE | evaluate.py, cross_regional.md |
| 7 Explainability | SHAP global + local | DONE | explain_shap.py, shap figures |
| 8 Affordability | ratio + 30% flag | DONE | affordability.py, summary |
| 9 Dashboard | interactive app | DONE | streamlit_app.py, predict.py |
| 10 Deploy | cloud + finalise | PENDING | — |

**Marking-scheme fit:** Implementation (25%) — a complete, working, OOP artefact;
Research Design (15%) — reproducible methodology; Testing (10%) and Results (10–15%)
— evaluation, cross-regional validation and figures are all produced. The one
honest caveat carried throughout is that results are on synthetic development data
and will be regenerated on real ONS data.

## 7. Conclusion
Through Phase 9 the implementation is complete, consistent, object-oriented,
well-commented, reproducible, and matches the build scope and dissertation plan.
The remaining work is real-data ingestion and Phase 10 (deployment), both already
scoped.

---

## 8. Update — four-module system completed
All four modules are now implemented and run on synthetic data (pending real ONS/HM Land Registry):
- **Module 1 (sale price):** `src/modules/module1_sale_price.py` — OLS/RF/XGBoost + bootstrapped prediction intervals. Best OLS, R² ~0.90.
- **Module 2 (rental + affordability):** the deepest module (cross-regional validation, SHAP, dashboard). R² 0.932.
- **Module 3 (geospatial):** `src/modules/module3_geospatial.py` — ablation (RMSE 67.7→65.2) + spatial SHAP (transport top).
- **Module 4 (forecasting):** `src/modules/module4_forecasting.py` — ARIMA + Holt-Winters, mean MAPE 2.6% / 0.6% across 10 regions.

`python run_pipeline.py` builds all four; the Streamlit dashboard now has a **System results** view integrating them. Deployment files added (`.streamlit/config.toml`, `DEPLOYMENT.md`).

### Remaining (human-only) tasks
1. Regenerate models locally (`python run_pipeline.py` in your own venv) so the saved `.joblib` matches your scikit-learn version.
2. Push the repo to GitHub (see `GITHUB_SETUP.md`).
3. Deploy to Streamlit Community Cloud (see `DEPLOYMENT.md`) and put the live URL in the report.
4. Swap synthetic data for real ONS/HM Land Registry data and re-run.
