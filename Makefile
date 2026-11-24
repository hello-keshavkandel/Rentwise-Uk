# RentWise UK — pipeline orchestration
# Usage:  make <target>   (run from the repo root)

PY ?= python3

.PHONY: help setup fetch validate build geo schools osm spatial api alerts store module1 refresh test lint app clean

help:               ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:              ## install dependencies
	$(PY) -m pip install -r requirements.txt

fetch:              ## download source data where URLs are automatable
	$(PY) -m src.data.sources fetch

validate:           ## validate raw data + (re)write data/raw/_manifest.json
	$(PY) -m src.data.sources manifest

build:              ## run the full real-data pipeline (all four modules)
	$(PY) run_real_pipeline.py

geo:                ## reproject LA boundaries for maps (needs data/raw/geo/lad_boundaries.geojson)
	$(PY) -m src.data.geo

schools:            ## historic Ofsted grade per LA (needs data/raw/schools/: Ofsted MI + GIAS)
	$(PY) -m src.data.schools

osm:                ## OSM amenity counts per LA (needs data/raw/osm/*.osm.pbf + requirements-data.txt)
	$(PY) -m src.data.osm

spatial:            ## Module 3 features + retrain/recalibrate the deployed spatial rent model
	$(PY) -m src.modules.module3_spatial && $(PY) -m src.models.rent_spatial

api:                ## run the public REST API locally (docs at http://localhost:8000/docs)
	$(PY) -m uvicorn src.api.app:app --reload --port 8000

alerts:             ## evaluate users' rent/affordability alerts (run after each refresh)
	$(PY) -m src.api.alerts

store:              ## (re)build the DuckDB processed-data store
	$(PY) -m src.data.store build

module1:            ## train the EPC-enriched Module 1 (needs EPC lookup)
	$(PY) run_module1_enriched.py

refresh: validate build store   ## end-to-end refresh: validate -> rebuild -> store
	@echo "RentWise UK refresh complete."

health:             ## check artifacts + data/model freshness
	$(PY) -m src.health

test:               ## run the test suite
	$(PY) -m pytest -q

app:                ## launch the Streamlit dashboard
	streamlit run app/streamlit_app.py

clean:              ## rebuild the DuckDB store from scratch (it is committed; never leave it deleted)
	rm -f data/processed/rentwise.duckdb && $(PY) -m src.data.store build
