# Module 1 - Sale Price Prediction (summary) [REAL HM Land Registry data]

## Model comparison (held-out test set)
| Model | Test RMSE (£) | Test MAE (£) | Test R² |
|---|---|---|---|
| OLS (baseline) | 150,749 | 97,005 | 0.505 |
| Random Forest | 141,050 | 88,776 | 0.567 |
| XGBoost | 140,886 | 88,714 | 0.568 |

**Best model:** XGBoost (R² 0.568).

## SHAP feature importance (best model, mean £ impact on sale price)
- **local area price level**: £105,763
- **detached**: £69,700
- **freehold tenure**: £19,113
- **flat**: £17,324
- **semi-detached**: £10,892
- **terraced**: £7,032
- **new build**: £65

## Calibrated uncertainty (split-conformal, 90%)
- 90% prediction-interval half-width ≈ £193,696; empirical coverage 90% on the test set.

## Cross-regional validation (leave-one-region-out)
| Region | RMSE (£) | R² | n |
|---|---|---|---|
| East Midlands | 137,330 | 0.060 | 3,570 |
| East of England | 196,043 | 0.098 | 4,563 |
| London | 413,737 | -1.218 | 3,774 |
| North East | 158,469 | -0.437 | 1,890 |
| North West | 136,337 | 0.206 | 5,192 |
| South East | 227,633 | 0.052 | 6,878 |
| South West | 161,369 | 0.208 | 4,732 |
| Wales | 133,856 | -0.129 | 1,891 |
| West Midlands | 129,403 | 0.333 | 3,486 |
| Yorkshire and The Humber | 148,949 | 0.078 | 4,024 |

Mean leave-one-region-out RMSE £184,313 — a deliberately strict generalisation test.

*Modelled on 90,000 of 216,172 cleaned standard residential sales; district median-price target encoding from training data only.*