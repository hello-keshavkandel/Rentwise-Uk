# Module 1 - Sale Price Prediction (summary) [REAL HM Land Registry + EPC floor area]

## Model comparison (held-out test set)
| Model | Test RMSE (£) | Test MAE (£) | Test R² |
|---|---|---|---|
| OLS (baseline) | 124,309 | 78,931 | 0.664 |
| Random Forest | 116,735 | 72,353 | 0.703 |
| XGBoost | 110,933 | 68,235 | 0.732 |

**Best model:** XGBoost (R² 0.732).

**EPC enrichment:** floor area + rooms lift XGBoost from R² 0.568 to **0.732** (+0.164); EPC match rate 86% of sales.

## SHAP feature importance (best model, mean £ impact on sale price)
- **local area price level**: £98,079
- **floor area (m²)**: £68,290
- **detached**: £30,820
- **has EPC match**: £14,184
- **freehold tenure**: £10,424
- **semi-detached**: £6,842
- **habitable rooms**: £6,822
- **flat**: £6,562

## Calibrated uncertainty (split-conformal, 90%)
- 90% prediction-interval half-width ≈ £154,252; empirical coverage 90% on the test set.

## Cross-regional validation (leave-one-region-out)
| Region | RMSE (£) | R² | n |
|---|---|---|---|
| East Midlands | 104,259 | 0.458 | 3,570 |
| East of England | 168,425 | 0.334 | 4,563 |
| London | 380,422 | -0.875 | 3,774 |
| North East | 138,116 | -0.092 | 1,890 |
| North West | 111,369 | 0.470 | 5,192 |
| South East | 199,280 | 0.273 | 6,878 |
| South West | 133,309 | 0.460 | 4,732 |
| Wales | 120,065 | 0.091 | 1,891 |
| West Midlands | 100,447 | 0.598 | 3,486 |
| Yorkshire and The Humber | 121,477 | 0.387 | 4,024 |

Mean leave-one-region-out RMSE £157,717 — a deliberately strict generalisation test.

*Modelled on 90,000 of 216,172 cleaned standard residential sales; district median-price target encoding from training data only.*