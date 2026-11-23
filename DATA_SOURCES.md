# RentWise UK — Data Sources (the "what, where and why")

Every dataset is open, public, and licensed for reuse (Open Government Licence, or OpenStreetMap's ODbL), which is why the project needs **no ethical approval**. This file is the data dictionary for the real files used in the final build.

## Real datasets used (placed in `data/raw/`, read by `run_real_pipeline.py`)

| # | File in `data/raw/` | Dataset (provider) | What it gives us | Used by | Link |
|---|---|---|---|---|---|
| 1 | `ons_pipr_monthly.xlsx` | Price Index of Private Rents — monthly price statistics (ONS) | **Median monthly rent (£)** by local authority **and by property type/bedroom**, monthly to Jun 2026 | Module 2 (rent target) | https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/priceindexofprivaterentsukmonthlypricestatistics |
| 2 | `ons_pipr_historical.xlsx` | Price Index of Private Rents — historical series (ONS) | Regional rent **£ levels and index**, monthly from 2005 | Module 4 (forecasting) | https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/priceindexofprivaterentsukhistoricalseries |
| 3 | `ashe_table8_7a.xlsx` | ASHE Table 8.7a — annual pay, gross, place of residence (ONS) | **Median gross annual earnings** by local authority | Module 2 (affordability ratio) | https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/placeofresidencebylocalauthorityashetable8 |
| 4 | `land_registry_pp_2026.csv` | Price Paid Data (HM Land Registry) | Every residential **sale price** (England & Wales) | Module 1 (sale price) | https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads |
| 5 | `police/*street.csv` | Street-level crime (data.police.uk) | Recorded crimes by LSOA/location | Module 3 (crime driver) | https://data.police.uk/data/ |
| 6 | `lsoa_to_lad.csv` | LSOA (2021) → LAD (2024) lookup (ONS Open Geography Portal) | Maps crime LSOAs up to local authority | Module 3 (crime aggregation) | https://geoportal.statistics.gov.uk/ |
| — | *(not sourced — future work)* | OpenStreetMap / Ofsted | Transport, schools, green space, amenity proximity | Module 3 spatial layers + maps | https://download.geofabrik.de/europe/great-britain.html |

## Key idea (important for the viva)
ONS rent data is published at **local-authority** level (and by property type), not per postcode. So the system **models rent at local-authority × property-type granularity** — it never claims an exact postcode-level "true" rent. This is stated as a limitation in the report and is the point the supervisor specifically asked us to address.

## How the real data is built (iterative, design-science)
1. **Development (synthetic):** `python run_pipeline.py` generates a synthetic sample with the same schema as the real data, so the whole pipeline could be built and tested before the official files were obtained. Clearly labelled synthetic; not used for final results.
2. **Real validation:** with the six real files above in `data/raw/`, `python run_real_pipeline.py` assembles them into the shared schema and runs all four modules on real data. It:
   - reads rent (£) by LA and property type from the PIPR monthly file (latest month),
   - joins median gross annual earnings from ASHE Table 8.7a (by LA code),
   - aggregates data.police.uk crime from LSOA to local authority via the ONS lookup and normalises it to a crime index,
   - trains Ridge / Random Forest / XGBoost, evaluates (test + leave-one-region-out), runs SHAP and the affordability engine,
   - trains Module 1 on the real Price Paid transactions and Module 4 on the real PIPR historical series.

## Honest data notes (documented in the report)
- ONS marks unavailable cells `[x]` and not-applicable `[z]`; these are treated as missing.
- The crime feature is a **normalised index** (no population denominator was sourced), and the transport feature is a placeholder (OSM not sourced — future work).
- ASHE earnings are **individual** employee earnings, not household income, so the affordability ratio is an area-level indicator, not an individual verdict.
- HM Land Registry Price Paid lacks floor area / bedroom counts, which caps Module 1's accuracy.
- Coverage is England & Wales (the ten modelled regions); Scotland/NI are out of scope for the joined dataset.

## EPC — Energy Performance Certificates (Module 1 enrichment)

Adds **floor area** and **habitable rooms** to the sale-price model — the two
strongest price drivers that HM Land Registry omits.

- **Source:** https://get-energy-performance-data.communities.gov.uk — GOV.UK "Get energy
  performance of buildings data" (beta), which replaced epc.opendatacommunities.org.
  England & Wales, Open Government Licence; **sign-in with GOV.UK One Login required**.
- **Fields used:** `total_floor_area`, `number_habitable_rooms` (joined to Price Paid by postcode + house number)
- **How to add it:**
  1. Sign in → **Download files** → **Domestic Energy Performance Certificates** →
     **Download all** (`domestic-csv.zip`, ~7.7 GB, Sep 2008 – latest month).
     **Move the zip into `data/raw/epc/` — do not unzip it**; the loader streams the CSVs
     straight out of the archive (extracted, it would be tens of GB). Extracted CSVs also
     work. Recommendation tables are ignored.
  2. Build the lookup:  `python -m src.data.epc`
     (reads only the needed columns, in chunks, keeping only postcodes found in Price Paid)
  3. Train the enriched model:  `python run_module1_enriched.py`
- The legacy API path (`EPC_EMAIL`/`EPC_API_KEY`) targets the old service and is not
  verified against the new GOV.UK developer API — use the bulk download.

## Module 3 geospatial enrichment (optional — fetched later)

### LA boundaries (done)
- **Source:** ONS "Local Authority Districts (May 2024) Boundaries UK BGC" (GeoJSON), via
  https://ckan.publishing.service.gov.uk/dataset/local-authority-districts-may-2024-boundaries-uk-bgc
  — May 2024 vintage to match the `LAD24CD` codes used throughout. Contains OS data © Crown copyright and database right.
- Save as `data/raw/geo/lad_boundaries.geojson`, then `make geo` (`python -m src.data.geo`): reprojects
  British National Grid → WGS84, keeps England & Wales, maps Barnsley/Sheffield to their April-2025 codes
  (E08000038/39, as used by ONS rent data) and writes the committed `data/processed/lad_boundaries_wgs84.geojson`.

These power the richer spatial drivers and interactive maps. They are optional:
the pipeline validates them when present and runs without them otherwise.

| File in `data/raw/` | Source | Columns / format |
|---|---|---|
| `osm_amenities.csv` | OpenStreetMap (Overpass / osmnx export) | `lad_code, category[, lat, lon]` |
| `ofsted_schools.csv` | Ofsted / Get Information About Schools | `lad_code, ofsted_rating` |
| `geo/lad_boundaries.geojson` | ONS Open Geography Portal (LAD 2024) | GeoJSON FeatureCollection with `LAD24CD` |

Then: `python -m src.data.sources manifest` validates them, and
`src/modules/module3_geo.py` + `src/viz/maps.py` build the features and choropleth maps.
*(`assign_lad` point-in-polygon needs the optional `geopandas`; if your amenity rows
already carry `lad_code`, it isn't required.)*
