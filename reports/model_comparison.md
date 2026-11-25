# Model Comparison - RentWise UK (Module 2) [REAL ONS/police data]

| Model | Test RMSE (£) | Test MAE (£) | Test R² |
|---|---|---|---|
| Ridge (baseline) | 217 | 164 | 0.870 |
| XGBoost (deployed) | 130 | 85 | 0.953 |

**Deployed model:** XGBoost — beats Ridge by 40% RMSE on real data.

*Real data: ONS Price Index of Private Rents, ONS ASHE earnings, data.police.uk crime. Transport pending (OSM, future work).*