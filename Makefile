# RentWise UK — pipeline orchestration
# Usage:  make <target>   (run from the repo root)

PY ?= python3

.PHONY: help setup fetch validate build store module1 refresh test lint app clean

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

store:              ## (re)build the DuckDB processed-data store
	$(PY) -m src.data.store build

module1:            ## train the EPC-enriched Module 1 (needs EPC lookup)
	$(PY) run_module1_enriched.py

refresh: validate build store   ## end-to-end refresh: validate -> rebuild -> store
	@echo "RentWise UK refresh complete."

test:               ## run the test suite
	$(PY) -m pytest -q

app:                ## launch the Streamlit dashboard
	streamlit run app/streamlit_app.py

clean:              ## remove the DuckDB store
	rm -f data/processed/rentwise.duckdb
