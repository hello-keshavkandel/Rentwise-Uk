# Module 1 — EPC-Enriched Sale Price Prediction

Price Paid: land_registry_pp_2026.csv · modelled on 90,000 sales (60/20/20 train/calibration/test).

## Model comparison (held-out test set)
| Feature set | Test RMSE (£) | Test MAE (£) | Test R² |
|---|---|---|---|
| Baseline (no floor area) | 140,411 | 88,915 | 0.563 |
| **EPC-enriched (floor area + rooms)** | **94,780** | **64,435** | **0.801** |

**EPC lift:** R² +0.238 (0.563 → 0.801); RMSE −32%. EPC match rate: 90% of sales.

## Calibrated uncertainty (split-conformal, 90%)
- 90% prediction interval half-width ≈ £138,236; empirical coverage 90% on the test set.

## SHAP feature importance (enriched model, mean £ impact)
- **floor area (m²)**: £115,005
- **local area price level**: £57,262
- **detached**: £22,310
- **has EPC match**: £14,009
- **flat**: £10,664
- **habitable rooms**: £9,581
- **freehold**: £6,430
- **room size**: £3,578