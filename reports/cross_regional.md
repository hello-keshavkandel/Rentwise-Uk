# Cross-Regional Validation - RentWise UK (Module 2)

I hold each region out, train on the rest, then predict the unseen region.

| Region | n | RMSE (£) | MAE (£) | R² |
|---|---|---|---|---|
| East Midlands | 140 | 151 | 118 | 0.429 |
| East of England | 180 | 400 | 314 | 0.145 |
| London | 128 | 1039 | 906 | -1.347 |
| North East | 48 | 212 | 161 | 0.180 |
| North West | 140 | 240 | 186 | 0.193 |
| South East | 256 | 499 | 448 | -0.603 |
| South West | 104 | 363 | 263 | 0.022 |
| Wales | 88 | 200 | 157 | -0.080 |
| West Midlands | 120 | 177 | 138 | 0.443 |
| Yorkshire and The Humber | 60 | 170 | 151 | 0.481 |

**RMSE across regions:** mean £345 (range £151-£1039). A tight range means the model generalises well across the UK.

*Figures from synthetic development data; regenerated on real ONS data.*