# Milestone 2 (Automated Data Pipeline) — Testing Report

**Branch:** `feature/module1-epc-enrichment` · **Date:** 2026-11-19
**Scope:** data-source registry (fetch + validate + provenance), DuckDB processed
store, Makefile orchestration, and CI / scheduled-refresh workflows.

## 1. Summary

| | |
|---|---|
| Automated tests | **23 / 23 passing** (was 15; +8 for sources & store) |
| Runtime | ~18 s |
| Live checks (real local data) | `make validate` → **all_ok=True** (6/6 sources); `make store` builds 5 tables + manifest |
| Network | Sandbox is network-restricted, so live downloads were **mocked**; the real fetch runs on your machine / GitHub Actions |

## 2. What was built

- **`src/data/sources.py`** — a registry of all six sources (HM Land Registry, ONS PIPR monthly + historical, ONS ASHE, ONS geography lookup, data.police.uk) with:
  - retrying downloaders (skip-if-cached; clear "download by hand" message for ONS URLs that change each release),
  - a per-source **validator** that runs on the local file,
  - a **provenance manifest** (`data/raw/_manifest.json`) recording bytes, sha256, row counts, validation status and fetch time.
- **`src/data/store.py`** — a **DuckDB** store (`data/processed/rentwise.duckdb`) built from the processed CSVs + the source manifest, with `table()`, `query()`, `info()` accessors.
- **`Makefile`** — `fetch · validate · build · store · refresh · test · app`.
- **`.github/workflows/`** — `ci.yml` (tests on every push/PR) and `refresh-data.yml` (monthly cron: fetch what's automatable, validate, publish the manifest).

## 3. Test inventory & results

### Source registry (`test_sources.py`) — 5/5 PASS
- `test_registry_shape` — 6 sources; HM Land Registry has a real URL, ONS sources are manual.
- `test_manifest_on_real_files` — validates the **real** raw files; `all_ok=True`; the BOM-safe lookup validator passes; manifest written.
- `test_fetch_is_mocked` — download path works with a mocked `requests.get`; second call is served from cache (no re-download).
- `test_manual_source_not_fetched` — URL-less sources report `manual`, never crash.
- `test_validator_flags_bad_file` — a malformed Price Paid file (<16 cols) is correctly rejected.

### DuckDB store (`test_store.py`) — 3/3 PASS
- `test_build_and_info` — builds tables from CSVs; row counts correct.
- `test_query_roundtrip` — SQL aggregation returns expected ordering; `table()` returns a DataFrame.
- `test_missing_csv_is_skipped` — absent CSVs are skipped gracefully, not fatal.

### Regression — everything from Milestone 1 still green
- Affordability (3), data contract (6), EPC join (4), Module 1 integration (2) — all PASS.

## 4. Live verification on real data (sandbox)

```
make validate   ->  all_ok=True
  [OK] land_registry   16 cols, 252,062 rows
  [OK] pipr_monthly    Table 1 present
  [OK] pipr_historical Table 3 present
  [OK] ashe            'All' sheet present
  [OK] lsoa_to_lad     LSOA21CD + LAD24CD present (BOM handled)
  [OK] police          42 street.csv files
make store      ->  rentwise_clean 1198 · features 1198 · sample 1264 · affordability 314 · source_manifest 6
```
A sample query via the store (`London £2,004` top average rent) confirms the SQL layer.

## 5. Known limitations (honest)

- **ONS & police auto-fetch:** ONS changes file URLs per release and data.police.uk ships bulk archives, so those stay **manual** downloads; the registry validates them once present and the monthly workflow reports them as pending rather than failing.
- **Sandbox network:** live downloads could not be exercised here (restricted network); the fetch logic is covered by a mocked test and will run for real on your machine / in Actions.
- The DuckDB store currently **mirrors** the processed CSVs; migrating the app's reads from CSV to DuckDB is a follow-up polish (Milestone 4).

## 6. Verdict

Milestone 2 is **implementation-complete and 100% green on all 23 tests**, with the
validation + provenance + store layers proven against the real local data and the
fetch path covered by mocks. The pipeline is now reproducible (`make refresh`),
checkable (manifest + validators), and queryable (DuckDB), with CI and a monthly
refresh wired up.
