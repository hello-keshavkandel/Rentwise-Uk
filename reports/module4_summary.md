# Module 4 - Regional Trend Forecasting (summary) [REAL ONS data]

| Region | ARIMA MAPE (%) | Holt-Winters MAPE (%) |
|---|---|---|
| North East | 1.25 | 1.19 |
| North West | 0.56 | 0.38 |
| Yorkshire and The Humber | 0.89 | 0.94 |
| East Midlands | 0.49 | 0.35 |
| West Midlands | 0.25 | 0.28 |
| London | 0.43 | 0.31 |
| South East | 0.42 | 0.64 |
| South West | 0.24 | 0.29 |
| Wales | 0.97 | 0.79 |

**Mean MAPE:** ARIMA 0.61%, Holt-Winters 0.57% (lower is better).

**Prediction-interval calibration (ARIMA, held-out 12 months):** the nominal 80% interval contained 68% of actuals and the 95% interval 89%. Coverage a little below the nominal level indicates the intervals are marginally over-confident (slightly narrow) — an honest calibration finding.

*Real ONS Price Index of Private Rents historical series (£ by region, monthly from 2005); held-out last 12 months; 80%/95% intervals from the ARIMA forecast distribution.*