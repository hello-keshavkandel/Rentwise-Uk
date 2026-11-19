# Module 1 (EPC Enrichment) — Testing Report

**Branch:** `feature/module1-epc-enrichment` · **Date:** 2026-11-19
**Scope:** the EPC floor-area/rooms enrichment, its join + feature engineering, the
shared trainer, and the wiring into `run_real_pipeline.py` and the dashboard.

## 1. Summary

| | |
|---|---|
| Automated tests | **15 / 15 passing** (15 collected, 0 failed, 0 skipped) |
| Runtime | ~17 s (full suite) |
| Build checks | `py_compile` clean on all new/changed modules |
| Pipeline wiring | `run_real_pipeline.run_module1()` confirmed to call the shared trainer |
| Dashboard wiring | System-results panel reads the regenerated `module1_summary.md` + `m1_predicted_vs_actual.png` |
| Lint | `ruff` not installed in this environment (lint not run) |

## 2. Test inventory & results

### Unit — affordability engine (`test_affordability.py`) — 3/3 PASS
- `test_ratio_is_rent_over_income` — ratio = annual rent ÷ annual income.
- `test_threshold_boundary_is_affordable` — 30.00% affordable, 30.01% not.
- `test_clearly_affordable_and_unaffordable` — 22.5% affordable, 45% not.

### Unit — data contract (`test_data_contract.py`) — 6/6 PASS
- Required columns present · no nulls in key/target · primary key unique · regions valid · value ranges · dataset reasonably sized.

### Unit — EPC join & features (`test_epc_join.py`) — 4/4 PASS
- `test_match_rate_and_flag` — join on (postcode, house number) matches the right rows; `has_epc` flag + match rate correct (0.75 on the fixture).
- `test_no_missing_after_impute` — unmatched sales get imputed floor area/rooms; no nulls; `floor_area_per_room > 0`.
- `test_feature_columns_toggle` — EPC feature set toggles on/off correctly.
- `test_encode_is_train_only` — district median-price encoding uses training rows only (no leakage) + one-hot property type.

### Integration — shared trainer on REAL data (`test_module1_integration.py`) — 2/2 PASS
*(skipped automatically if the Price Paid file is absent, so CI without data still passes)*
- `test_baseline_end_to_end` — full path on real Price Paid (8k sample): best R² in a sane band, **conformal 90% interval coverage = 90% (±8%)**, summary written.
- `test_epc_enrichment_lifts_accuracy` — with a synthetic floor-area proxy: EPC actually joins (match rate > 0.3) and **enriched R² > baseline R²** (mechanism verified).

## 3. System verification (manual, on real data)

- **Baseline reproduced** on the real 216,172-sale dataset (90k model sample):

  | Model | Test RMSE (£) | Test MAE (£) | Test R² |
  |---|---|---|---|
  | OLS (baseline) | 150,749 | 97,005 | 0.505 |
  | Random Forest | 141,050 | 88,776 | 0.567 |
  | **XGBoost (best)** | **140,886** | **88,714** | **0.568** |

  Matches the dissertation's Module 1 (R² 0.567) — the refactor changed nothing it shouldn't.
- **Calibrated uncertainty:** split-conformal 90% interval ≈ **±£193,696**, empirical coverage **90%**.
- **Cross-regional (leave-one-region-out):** mean RMSE **£184,313** (strict generalisation test), unchanged from baseline.
- **Figures regenerated:** `m1_predicted_vs_actual.png`, `m1_shap.png`, `m1_cross_regional.png`.
- **Dashboard:** unchanged code, but it renders the regenerated summary + figure, so it will automatically show the enriched model (lift + intervals) once EPC data is added.

## 4. EPC enrichment — status

The enrichment **code is complete and fully tested**, but the real accuracy gain
needs the real EPC dataset (free, user-supplied — see `DATA_SOURCES.md`).

- **Mechanism validated** with a synthetic floor-area proxy (floor area correlated
  with price): baseline R² 0.563 → **0.801** at a 90% match rate, floor area ranked
  the #1 SHAP driver (£115k). *These numbers are illustrative of the plumbing only
  — not a real result.*
- **Real result:** will be produced by `python -m src.data.epc` then
  `python run_module1_enriched.py` once EPC bulk/API data is in place.

## 5. Known gaps / not yet covered

- Real EPC accuracy is **not yet measured** (blocked on the free EPC download).
- `ruff`/`black` linting not run here (no `ruff` in this environment) — will be part of Milestone 1 CI.
- No property-level address fuzzy-matching yet (we match on postcode + house number; flats/named houses that carry no number don't match and are reported honestly).

## 6. Verdict

The Module 1 enrichment is **implementation-complete and 100% green on all 15 tests**,
wired into both the pipeline and the dashboard, with the baseline faithfully
reproduced and calibrated uncertainty added. The only thing between here and the
real accuracy jump is the one-off EPC data download.
