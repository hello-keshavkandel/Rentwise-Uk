# Model Comparison - RentWise UK (Module 2) [REAL ONS/police data]

| Model | Test RMSE (£) | Test MAE (£) | Test R² |
|---|---|---|---|
| Ridge (baseline) | 192 | 149 | 0.804 |
| XGBoost (deployed) | 137 | 97 | 0.900 |

**Deployed model:** XGBoost — beats Ridge by 29% RMSE on real data.

*Real data: ONS Price Index of Private Rents, ONS ASHE earnings, data.police.uk crime. Transport pending (OSM, future work).*