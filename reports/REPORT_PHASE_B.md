# RentWise UK — Phase B report: a reliable local environment

**Date:** 25 November 2026 · **Author:** Keshav Kandel · **Status:** complete in the sandbox; two checks for the Mac

Phase B makes RentWise run the same way everywhere — a laptop, CI, Docker — and gives three commands that
anyone (including a recruiter) can use:

```bash
make setup     # Python 3.11 environment with exact package pins
make verify    # proves everything works, including live servers
make demo      # dashboard + API with a ready demo account
```

## 1. What was built

| Piece | What it does |
|---|---|
| `make setup` (`scripts/setup.sh`) | Builds `.venv` with **Python 3.11** — the version the live app, Docker and CI use. Uses `uv` (fetches 3.11 automatically), else `python3.11`, else prints exact install steps. Creates `.env` from `.env.example`. |
| `.python-version` | Pins 3.11 for uv/pyenv. Needed because macOS ships 3.13, and several pins (e.g. numpy 1.26.4) have no 3.13 builds. |
| `make verify` (`scripts/verify.py`) | Six checks with a PASS / WARN / FAIL table: Python version, package pins, health, **full test suite**, **live API** (starts uvicorn, creates a temporary key, calls health, sources and predict), **live dashboard** (starts Streamlit, checks its health endpoint). Saves `reports/verify_report.md`; exit code 1 on any failure. |
| `make demo` (`scripts/demo.py`) | Seeds a separate demo database (never the real one) with a demo user, API key, saved area and an affordability alert, evaluates the alert once so a notification is waiting, then starts the API and dashboard. |
| `tests/test_data_quality.py` | Locks in the Phase A standard on the shipped data: one official row per area × type, Westminster present, rents untrimmed, crime never a fake zero and every gap declared, no placeholder columns, every input dated, every area has spatial features and a map shape, store matches CSV. |
| `tests/test_app.py` | Dashboard tests (Streamlit AppTest): official rent with date, an area with missing crime still answers, sources panel lists no-data areas, map and results views render. |
| `tests/test_demo.py` | The demo seed creates a working account, key, saved area and alert. |
| CI | Runs `make verify`'s script on every push (Python 3.11): all tests plus live API and dashboard; uploads the verify report. |
| Docker | Both images run as a non-root user; the image no longer bundles the 50 MB EPC lookup or the 10 MB rent history (pipeline-only). |
| `.env.example` | Every setting documented (API database, rate limit, sign-up, session length, SMTP, EPC). Relative paths resolve from the project root. |
| README | Three-command quickstart and a troubleshooting table. |

## 2. Validation

| Check | Result |
|---|---|
| `make verify` in the working repo (sandbox) | API, dashboard, health and **140 tests** pass; correctly warns that the sandbox uses Python 3.10 and lacks some pinned packages |
| **Fresh clone**, clean environment from the requirement files | environment built in **26 s**; all 16 pins exact; `make verify` **PASS** in **40 s** — 131 passed, 5 skipped (need git-ignored raw data or optional mapping libraries) |
| `make demo` live | Westminster flat £2,974 (official, Oct 2026) via the API; demo login shows the waiting alert *"Rent-to-income for a flat in Camden is now 66% — above your 50% threshold"*; dashboard healthy |

**Not verifiable in the sandbox (two checks for the Mac):**

1. **Python 3.11 via `make setup`.** The sandbox network blocks the Python 3.11 download; the fresh-clone test
   therefore used Python 3.10 with every package pin identical. On the Mac, `uv` downloads 3.11 normally. CI on
   GitHub also runs 3.11 and will confirm on the next push.
2. **`docker compose up --build`.** Docker is not available in the sandbox.

## 3. Time from clone to running app

Clean environment in 26 s plus `make verify` in 40 s in the sandbox — well inside the 10-minute target. On a Mac
the first `make setup` also downloads Python 3.11 (one-off, typically under a minute with uv).
