# Data Contract — RentWise UK

Every inter-module data exchange follows this contract, so the four modules can be
built and tested independently and still integrate. All exchanges are CSV, UTF-8,
one row per observation. The Module 2 development schema below is produced by
`SampleDataGenerator` (synthetic) or, later, by `RealDataLoader` (real ONS data) —
both write the **same** columns, which is what lets the data source be swapped with
no downstream change.

## Primary key
`(la_code, property_type, year)` uniquely identifies a row. `la_code` alone is the
key for area-level joins (income, crime, transport).

## Module 2 schema — `data/processed/rentwise_sample.csv` / `rentwise_clean.csv`

| Column | Type | Meaning | Valid range / values |
|---|---|---|---|
| `la_code` | string | ONS-style local-authority code | non-null, e.g. `E09000007` |
| `la_name` | string | Local authority name | non-null |
| `region` | string | One of the ten modelled regions | see list below |
| `year` | int | Observation year | 2022–2024 |
| `property_type` | string | Dwelling type | Room, Flat, Terraced, Semi-detached, Detached |
| `bedrooms` | int | Bedroom count | 1–6 |
| `median_annual_income` | float | Local median gross annual income (£) | 20,000–80,000 |
| `crime_per_1000` | float | Crimes per 1,000 population | 0–200 |
| `transport_score` | float | Transport/amenity access score | 0.0–1.0 |
| `median_monthly_rent` | float | **Target** — median monthly rent (£) | ≥ 250 |

Regions: North East, North West, Yorkshire and The Humber, East Midlands,
West Midlands, East of England, London, South East, South West, Wales.

## Feature schema — `data/processed/rentwise_features.csv`
All Module 2 columns above (minus the identifiers used only for joins) plus the
engineered features: `property_type_ord` (ordinal 1–5), `income_band` (1–4),
and one-hot `region_*` dummy columns. The target `median_monthly_rent` is retained.

## Cross-module contract
Modules 1, 3 and 4 key their area-level outputs on `la_code` / `region` so they can
be merged into the shared feature store and surfaced together on the dashboard:
Module 1 adds `predicted_sale_price` (+ interval), Module 3 adds spatial features
(`transport_score`, `crime_per_1000`, `school_quality`, `green_space`,
`amenity_density`), and Module 4 adds `forecast_rent` by region and horizon.

## Quality rules (enforced in `tests/`)
No nulls in key or target columns; the primary key is unique; every value sits in
the range above; `region` is drawn only from the ten-region list.

*Version 1.0. The synthetic development build (`run_pipeline.py`) and the real build
(`run_real_pipeline.py`, which assembles this schema from ONS PIPR rents, ASHE
earnings and data.police.uk crime) both honour this same contract — so cleaning,
features, models, SHAP, affordability and the dashboard run unchanged on either.*
