# RentWise UK — Product Roadmap

**Owner:** Keshav Kandel · **Status:** post-dissertation, solo build
**Goal:** a polished, open-source portfolio project now → a real public product later.
**Strategy (chosen):** keep the Python/Streamlit stack, but make it production-grade by (2) closing the data & model gaps, (3) automating the live data pipeline, and (4) hardening & deploying properly. No full re-architecture yet — that comes in Phase 5 when there are users.

> The submitted MSc dissertation stays frozen and credited to Group 27. This roadmap is the **product** continuation, led solely by Keshav. We tag the dissertation commit (`dissertation-submission`) and evolve the branding to a personal project from Milestone 1.

---

## Where we are today (baseline)

| Area | Current state | Gap to close |
|---|---|---|
| **Module 1 — Sale price** | XGBoost R² **0.567** on HM Land Registry (2026 H1, 90k sample) → **0.732 with real EPC** (done) | Wide absolute intervals (±£154k at 90%); weak London generalisation |
| **Module 2 — Rent & affordability** | XGBoost R² **0.900** (random split; **0.775** with whole LAs held out), SHAP, 30% benchmark, conformal intervals (done) | Area-level only; headline R² is in-sample-LA |
| **Module 3 — Geospatial** | **Done (real data):** 11 OSM amenity densities + historic Ofsted quality + LA boundaries; unseen-LA R² 0.775 → **0.833** | Spatial features not yet in the live estimator; density per km² (no population yet); Wales schools (Estyn) missing |
| **Module 4 — Forecasting** | ARIMA + Holt-Winters, MAPE ~0.6% | Intervals slightly mis-calibrated |
| **Data** | Manual download → `data/raw/` → `run_real_pipeline.py` | No automation, no refresh, CSV-only, no validation |
| **App / deploy** | Streamlit two-view, free Community Cloud | Sleeps; no CI/CD, monitoring, analytics, domain |
| **Engineering** | 2 unit tests, pinned reqs | No CI, limited tests, no logging/config discipline |

---

## Milestone 1 — Foundation & "make it mine" (quick, unblocks everything)

*Turn the dissertation artefact into a clean, personal, well-engineered base.*

- **Rebrand to a solo project:** new `README.md` (your authorship, product framing, screenshots, live link), add `LICENSE` (MIT), `CONTRIBUTING.md`; drop group-specific wording from product docs. Tag the dissertation state: `git tag dissertation-submission`.
- **Project hygiene:** `pyproject.toml` (or keep `requirements.txt` but split dev/prod), `.env` + `config` for paths/keys, structured **logging**, consistent error handling, `ruff`/`black` + `pre-commit`.
- **Test suite:** expand `tests/` — data contract, affordability engine, feature builder, prediction engine (rent floor, derived bedrooms, inert-factor hiding), model-smoke tests.
- **CI:** GitHub Actions — run tests + lint on every push.
- **Definition of done:** green CI badge, one-command local setup, README a stranger can follow.

## Milestone 2 — Automated live data pipeline (focus #3)

*No more manual downloads — the system refreshes itself and fails loudly on bad data.*

- **Source fetchers** (one module each, resumable, cached): ONS Price Index of Private Rents, ONS ASHE earnings, HM Land Registry Price Paid (monthly), data.police.uk crime, ONS geography lookups & LA boundaries.
- **Local data store:** move processed data from loose CSVs to **DuckDB** (or SQLite) — fast, file-based, zero-infra; keep raw files versioned/snapshotted.
- **Data validation:** schema + range + freshness checks (lightweight, custom or `pandera`) so a malformed ONS release can't silently corrupt a model.
- **Orchestration:** turn `run_real_pipeline.py` into a small DAG (start simple: a `Makefile` or `prefect`-lite) with incremental, idempotent steps.
- **Schedule:** GitHub Actions cron (monthly, aligned to ONS release dates) that refreshes data, re-runs the pipeline, and commits updated `reports/` + model artifacts.
- **Definition of done:** `make refresh` (or a scheduled run) pulls the latest real data, validates it, retrains, and updates the app's artifacts — unattended.

## Milestone 3 — Close the data & model gaps (focus #2)

*The biggest quality + portfolio leap.*

- **Module 1 — add floor area & bedrooms (highest impact):** join **EPC (Energy Performance Certificate) open data** (total floor area, habitable rooms, property age/type) to Price Paid by postcode/address; extend history beyond 2026 H1. Expect a real R² jump. Add calibrated **prediction intervals** via conformal prediction (`MAPIE`).
- **Module 3 — real geospatial features + maps:** source **OpenStreetMap** (transport stops, green space, amenities via `osmnx`/Overpass) and **Ofsted** school ratings; compute per-LA features; build the **PCA composite spatial index**; add **interactive choropleth maps** (Folium/Plotly over ONS LA boundaries) to the dashboard.
- **Module 2 & 4 — uncertainty:** calibrated intervals for rent (conformal) and recalibrated forecast intervals (rolling-window backtesting).
- **Model ops:** hyperparameter tuning, a simple **model registry** (versioned artifacts + a metrics log so you can see quality trend over time).
- **Definition of done:** Module 1 R² meaningfully improved with honest intervals; Module 3 shows ≥4 real spatial drivers on an interactive map; every prediction carries calibrated uncertainty.

## Milestone 4 — Production hardening & deploy (focus #4, deepened)

*Make it fast, reliable, observable, and properly live.*

- **Hosting:** keep Streamlit but move to an owned/paid host that doesn't sleep (Render / Railway / Fly.io / Hugging Face Spaces), with model + data artifacts loaded from storage, health checks, friendly error states, and caching.
- **CI/CD:** auto-deploy on push to `main` after tests pass.
- **Observability:** usage analytics, logging, and a lightweight data/model-freshness banner in the app.
- **Polish:** custom domain, refreshed UI, About/methodology page, demo GIF in the README.
- **Definition of done:** always-on app on your domain, auto-deployed, monitored, that you'd happily put at the top of your CV.

## Phase 5 — Toward a real public product

**Milestone 5 — done (Nov 2026):** public **REST API** (FastAPI, OpenAPI docs) with hashed API keys and
per-key rate limiting; **user accounts** (scrypt passwords, expiring bearer tokens), self-service keys,
**saved areas** and edge-triggered **rent / affordability alerts** with pluggable email delivery; Docker +
compose. See `reports/REPORT_P1_M5.md`.

**Next (when there are users):**
- Postgres + Redis (shared rate limiting) + cloud host behind HTTPS; API monitoring.
- Password reset / email verification; optional React/Next frontend; usage dashboards.
- Wider intervals (or refusal) for unseen areas; per-capita amenity features (ONS population); Estyn for Wales.
- Monetisation options (freemium API, embeds for agents/portals).

---

## Suggested order & rough effort

1. **Milestone 1 — Foundation** (small) ← start here
2. **Milestone 2 — Data pipeline** (medium)
3. **Milestone 3 — Data & model gaps** (large; EPC + maps are the headline wins)
4. **Milestone 4 — Hardening & deploy** (medium)
5. **Phase 5 — Product** (future)

We build one milestone at a time, each on its own branch, each ending green and deployable.
