# Module 3 - Geospatial Driver Analysis (summary) [REAL crime data]

## Real spatial feature: crime (data.police.uk, LSOA -> local authority)
- Crime ablation (XGBoost on real rent): test RMSE £137 with crime vs £143 without (marginal ~£5).
- Module 2 SHAP on real data attributes about £57 of rent to local crime level.

## Future work (open data not yet sourced)
- Transport proximity, school quality, green space and amenity density need OSM/Ofsted layers; interactive choropleth maps need LA boundary geometry. These remain future work.
