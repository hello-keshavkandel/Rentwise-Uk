# Module 2 - Rent model (deployed) [REAL ONS + police + OSM + Ofsted]

**Deployed:** XGBoost on Module 2 features + Module 3 spatial features, trained 2026-11-25
on 1264 rows (316 local authorities).

| Evaluation (5-fold, out-of-fold) | R² | MAE | RMSE |
|---|---|---|---|
| LAs seen in training (what the app/API serve) | 0.945 | £79 | £144 |
| Whole LAs held out (unseen areas) | 0.858 | £152 | £231 |

**Calibrated intervals (conformal):** 80% → ±£106, 90% → ±£175.
Coverage for served areas 80% / 90% (on target). For unseen areas they would
cover only 51% / 70% — which is why the app and API serve only the
316 trained areas.

Before Phase A (Nov 2026 audit) the model trained on 1,198 rows after 66 genuine high rents had been
deleted as "outliers", with a one-month raw crime count and a constant transport placeholder:
R² 0.947 / MAE £68 (seen), 0.817 / £136 (unseen). Phase A restored every official rent and replaced
crime with a true rate, so the figures are not directly comparable — the restored rows are the most
expensive in the country.

*Spatial features: log density of 11 OpenStreetMap amenity types + historic Ofsted school quality.
Associations, not causes.*
