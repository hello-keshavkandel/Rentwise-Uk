# Module 1 - Sale Price Prediction (summary) [REAL HM Land Registry data]

## Model comparison (held-out test set)
| Model | Test RMSE (£) | Test MAE (£) | Test R² |
|---|---|---|---|
| OLS (baseline) | 150,750 | 97,004 | 0.505 |
| Random Forest | 141,050 | 88,776 | 0.567 |
| XGBoost | 140,954 | 88,721 | 0.567 |

**Best model:** XGBoost (R² 0.567).

## SHAP feature importance (best model, mean £ impact on sale price)
- **local area price level**: £105,836
- **detached**: £76,728
- **freehold tenure**: £20,323
- **semi-detached**: £18,314
- **flat**: £11,150
- **terraced**: £2,473

## Cross-regional validation (leave-one-region-out)
| Region | RMSE (£) | R² | n |
|---|---|---|---|
| East Midlands | 138,204 | 0.048 | 3,570 |
| East of England | 195,928 | 0.099 | 4,563 |
| London | 414,440 | -1.225 | 3,774 |
| North East | 157,358 | -0.417 | 1,890 |
| North West | 138,244 | 0.184 | 5,192 |
| South East | 227,514 | 0.053 | 6,878 |
| South West | 160,444 | 0.217 | 4,732 |
| Wales | 133,136 | -0.117 | 1,891 |
| West Midlands | 129,496 | 0.332 | 3,486 |
| Yorkshire and The Humber | 148,157 | 0.088 | 4,024 |

Mean leave-one-region-out RMSE £184,292 — higher than the in-sample test error, because a wholly held-out region carries no local price signal (district encoding falls back to the national median). This is a deliberately strict generalisation test.

*Real HM Land Registry Price Paid, England & Wales (modelled on a 90,000-transaction random sample of 216,172 cleaned standard residential sales; district median-price target encoding from training data only). Real sale prices need floor area/bedrooms for higher accuracy - a documented limitation.*