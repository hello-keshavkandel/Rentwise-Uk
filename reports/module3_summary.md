# Module 3 - Geospatial features (summary) [REAL OSM + Ofsted + ONS boundaries]

**Spatial features improve rent prediction for unseen local authorities:**
R² 0.775 → **0.833** (+0.059), MAE £152 → **£130**
(-14%). Gain in every repeat: +0.053, +0.053, +0.054, +0.074.

## Ablation — whole LAs held out (grouped 5-fold CV × 4 repeats, XGBoost)
| Features | R² (mean ± sd) | MAE |
|---|---|---|
| base | 0.775 ± 0.008 | £152 ± 3 |
| +amenities | 0.824 ± 0.006 | £134 ± 2 |
| +schools | 0.803 ± 0.011 | £143 ± 4 |
| +both | 0.833 ± 0.005 | £130 ± 3 |

## Why grouped CV
Each LA appears up to 4 times (one row per property type) with identical LA-level features.
A random split scores LAs the model has already seen: base R² 0.913 / +spatial
0.946 (MAE £89 / £69). Holding out whole LAs is the honest
test of generalisation; the dashboard only offers LAs present in training, so its in-sample
accuracy sits between the two.

## Features (314 LAs, 1198 rows)
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
