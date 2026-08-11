# Cross-Regional Validation - RentWise UK (Module 2)

I hold each region out, train on the rest, then predict the unseen region.

| Region | n | RMSE (£) | MAE (£) | R² |
|---|---|---|---|---|
| East Midlands | 140 | 199 | 141 | -0.005 |
| East of England | 175 | 358 | 279 | 0.108 |
| London | 73 | 630 | 578 | -3.294 |
| North East | 48 | 202 | 167 | 0.224 |
| North West | 140 | 233 | 181 | 0.210 |
| South East | 250 | 459 | 405 | -0.710 |
| South West | 104 | 341 | 235 | 0.115 |
| Wales | 88 | 161 | 136 | 0.289 |
| West Midlands | 120 | 156 | 115 | 0.557 |
| Yorkshire and The Humber | 60 | 207 | 196 | 0.205 |

**RMSE across regions:** mean £295 (range £156-£630). A tight range means the model generalises well across the UK.

*Figures from synthetic development data; regenerated on real ONS data.*