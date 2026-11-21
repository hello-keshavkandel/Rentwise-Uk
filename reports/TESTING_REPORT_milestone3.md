# Milestone 3 (Model/Data Gaps — in-sandbox slice) — Testing Report

**Branch:** `feature/module1-epc-enrichment` · **Date:** 2026-11-20

## Summary
| | |
|---|---|
| Automated tests | **34 / 34 passing** (+5 since Milestone 2: conformal, registry, geo, maps) |
| Delivered now (real data) | calibrated rent intervals (shown in app) + model metrics registry |
| Scaffolded (data fetched later) | Module 3 geospatial features + interactive choropleth maps |

## 1. Calibrated prediction intervals — Module 2 (DONE, real data)
- `src/models/conformal.py`: K-fold **cross-conformal** interval widths, finite-sample corrected.
- Calibrated on the real processed data: **90% → ±£199 (coverage 90.2%)**, **80% → ±£140 (coverage 80.1%)** — essentially perfect calibration.
- `predict.py` now returns an `interval`; the dashboard shows an **80% confidence range** under each rent estimate (e.g. Adur Flat £1,104 → £963–£1,244).
- Tests: `test_conformal.py` (4) — width ordering, coverage near nominal, calibrate/load round-trip.

## 2. Model registry / metrics history (DONE)
- `src/models/registry.py`: appends each run's metrics + git commit to `reports/metrics_history.jsonl`; `history()` returns a DataFrame.
- Wired into the Module 1 trainer (every run is recorded).
- Tests: `test_registry.py` (2).

## 3. Module 3 geospatial scaffold (CODE DONE; data later, like EPC)
- `src/modules/module3_geo.py`: per-LA amenity aggregation, Ofsted school-quality scoring, feature-store merge, and a **PCA composite spatial index** (pure pandas/sklearn). Optional geopandas `assign_lad` for point-in-polygon.
- `src/viz/maps.py`: **folium choropleth** renderer (reads GeoJSON directly — no geopandas), saves interactive HTML.
- Registered 3 **optional** sources (OSM, Ofsted, LA boundaries) with validators; the manifest stays `all_ok=True` without them.
- Tests: `test_module3_geo.py` (3) — aggregation, school scoring, PCA orientation; `test_maps.py` (2) — choropleth renders a real Leaflet map to HTML.

## 4. Honest status
- The **real** Module 3 spatial drivers + maps need the OSM/Ofsted/boundary downloads (same pattern as EPC); the engineering + tests are ready for them.
- `geopandas` is kept **optional** (only for lat/lon → LA assignment); everything else runs on the current dependency set (folium already in requirements).

## 5. Verdict
Milestone 3's **data-free** items (calibrated uncertainty + registry) are complete and
live in the app; the geospatial pieces are fully scaffolded and tested, waiting only on
the external data. **34/34 tests green.**
