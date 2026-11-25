# Module 3 - Geospatial features (summary) [REAL OSM + Ofsted + ONS boundaries]

**Spatial features consistently reduce the typical error for unseen local authorities; the R² gain is small and not consistent across repeats:**
R² 0.847 → **0.853** (+0.006), MAE £169 → **£156**
(-8%).
R² change per repeat: +0.009, -0.003, +0.006, +0.012 · MAE reduction per repeat:
£12, £10, £13, £18.

## Ablation — whole LAs held out (grouped 5-fold CV × 4 repeats, XGBoost)
| Features | R² (mean ± sd) | MAE |
|---|---|---|
| base | 0.847 ± 0.001 | £169 ± 1 |
| +amenities | 0.854 ± 0.002 | £157 ± 1 |
| +schools | 0.848 ± 0.008 | £166 ± 5 |
| +both | 0.853 ± 0.004 | £156 ± 2 |

## Why grouped CV
Each LA appears up to 4 times (one row per property type) with identical LA-level features.
A random split scores LAs the model has already seen: base R² 0.936 / +spatial
0.950 (MAE £97 / £78). Holding out whole LAs is the honest
test of generalisation; the dashboard only offers LAs present in training, so its in-sample
accuracy sits between the two.

## Features (316 LAs, 1264 rows)
- **Amenity density** — log(1 + count per km²) for 11 OSM categories (rail stations, bus stops,
  supermarkets, convenience, GP/clinics, hospitals, pharmacies, restaurants/cafés, pubs/bars,
  parks, gyms); 506,395 amenities from Geofabrik England + Wales (Nov 2026).
- **School quality** — mean historic Ofsted grade (last 2019-framework grade, labelled historic)
  and share Good-or-better; 18,263 schools. Missing for Welsh LAs (Estyn) — left as NaN.

## Caveats
- OSM completeness varies by area; densities are relative indicators, not a census.
- Density per km² favours compact urban LAs; per-capita measures need ONS population (future).
- School grades are historic (median inspection year 2023).
- Associations, not causes: these features describe places, they don't set rents.
