# Module 2 - Rent model (deployed) [REAL ONS + police + OSM + Ofsted]

**Deployed:** XGBoost on Module 2 features + Module 3 spatial features, trained 2026-11-24
on 1198 rows (314 local authorities).

| Evaluation (5-fold, out-of-fold) | R² | MAE | RMSE |
|---|---|---|---|
| LAs seen in training (what the app/API serve) | 0.947 | £68 | £101 |
| Whole LAs held out (unseen areas) | 0.817 | £136 | £188 |

**Calibrated intervals (conformal):** 80% → ±£113, 90% → ±£161.
Coverage for served areas 80% / 90% (on target). For unseen areas they would
cover only 54% / 68% — which is why the app and API serve only the 314 trained areas.

Previous deployed model (no spatial features): R² 0.913 / MAE £89 (seen), 0.775 / £152 (unseen), 80% interval ±£140.

*Spatial features: log density of 11 OpenStreetMap amenity types + historic Ofsted school quality.
Associations, not causes.*
