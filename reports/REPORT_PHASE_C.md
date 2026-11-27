# RentWise UK — Phase C report: making it useful

**Date:** 27 November 2026 · **Author:** Keshav Kandel · **Status:** complete; `make verify` PASS on the Mac

Phase C turns RentWise from "a model with a dashboard" into a tool that answers the questions renters,
landlords, analysts and councils actually ask — with official figures first, every number dated and
sourced, and missing data shown as missing.

## 1. What was built

| Question a user asks | Answer in the app / API | Data |
|---|---|---|
| "What does a **2-bed** cost here?" | Official ONS median for **all properties, 1, 2, 3, 4+ bed** and the four property types, with the published annual change | ONS PIPR, Oct 2026 — 316 LAs × 9 measures, plus 9 regions and 3 nations |
| "Can **I** afford it?" | Rent as a share of **your** gross income (optional, never stored), the band (within / above / well above 30%) and the **rent you could pay at 30%**; also against local median pay | ASHE 2025 |
| "Is it cheaper to buy?" / "What yield would I get?" | Median sale price (standard-price sales, ≥ 20 sales) and indicative **gross yield** for the same property type | HM Land Registry Price Paid, Jan–Jun 2026 (216,686 sales) |
| "How has rent moved?" | Monthly trend **since Jan 2015** for the area, its region and the nation, any measure | ONS PIPR (419k rows, shipped as a 0.6 MB parquet) |
| "Where is it going?" | The region's **12-month forecast** with 80/95% bands and an honest coverage note | `models/rent_forecasts.json` |
| "How do these places compare?" | 2–4 areas side by side + trends + **CSV download**; every area × measure as one analyst CSV | all of the above + crime, schools, amenities |
| "What exactly is this number?" | **Methodology & data** page: definition, source, date and limits of every figure | — |

API additions (all behind the API key, same rate limit): `GET /v1/areas/{code}/rents?income_gbp_year=`,
`/trend?measure=`, `/forecast`, `GET /v1/compare?codes=`, and `income_gbp_year` on `POST /v1/predict`
(returns `your_affordability`).

## 2. Forecast method and honest evaluation

Three models on log rent, compared on the same rolling-origin backtest (18 origins × 12 series, 1–12 months ahead):

| Model | Pooled MAPE |
|---|---|
| ARIMA(1,1,1) | 1.45% |
| ARIMA(1,1,0) + drift | 1.49% |
| **Damped-trend ETS (chosen)** | **0.87%** |

One model is chosen for all series, because picking the best per series on the same backtest would flatter the
result. Error rises from 0.19% at one month to 1.63% at twelve. Bands are **conformal** (per-horizon quantiles of
the backtest's own errors). **Out-of-time check** (calibrated on older origins, tested on newer): nominal 80% →
**71%**, nominal 95% → **93%**. The bands are slightly narrow because rent growth has been slowing — the app
says so beside the chart rather than presenting 80% as exact. Forecasts are regional (the level Module 4
validated); area-level forecasts are a candidate for later.

## 3. Findings worth knowing

- *Superseded by `reports/VALIDATION_REAL_WORLD.md` (27 Nov): affordability now uses full-time pay, the ONS
  basis — 198 of 316 areas (63%) above 30%, median area 33%; and the ONS figures are averages, not medians.*
- **Affordability is worse than "30%" headlines suggest when measured against one earner.** On official
  October 2026 rents, the all-properties median is above 30% of local median pay in **276 of 316 areas (87%)**;
  the median area is at **40%** (range 24%–97%; e.g. Leeds 44%, Manchester 55%, Bristol 67%). The app states the method (individual gross pay, stricter than household
  measures) next to every ratio.
- **Gross yields are plausible** and match market intuition: median 4.5% overall, flats highest (6.5%),
  detached lowest (4.0%).
- **The map now uses official figures.** Its affordability layer previously came from model predictions
  averaged over types (`affordability.csv`); it now uses the official all-properties median ÷ local pay.

## 4. Bugs found and fixed during Phase C

| Bug | Effect | Fix |
|---|---|---|
| `GET /v1/areas/{code}` called `float()` on a missing crime rate | 500 error for the 10 Greater Manchester boroughs (NaN is not valid JSON) | returns `null`; regression test |
| Area-map affordability came from model estimates | map disagreed with the official-first rent page | map uses official rent ÷ ASHE pay |
| A 2-bed rent was compared with the all-homes sale price | misleading yield for bedroom measures | yield shown only where rent and price describe the same property type |

## 5. Accessibility and mobile

Every input has a visible label and help text; a focus outline is enforced; charts use a colour-blind-safe
palette **and** distinct dash patterns (never colour alone); every chart has a titled axis and a "show as a
table" alternative; verdicts carry text and an icon, not just colour; metric labels were shortened so nothing
truncates. Checked in a 375 px mobile viewport: metrics stack, tables scroll, and a hint points to the sidebar.
A full WCAG audit with assistive technology remains for Phase E.

## 6. Validation

| Check | Result |
|---|---|
| New tests (`tests/test_phase_c.py`, updated `tests/test_app.py`) | PIPR parsing incl. `[x]` → missing; 316 × 9 coverage; type rents = model dataset (1,264/1,264); trends 2015 → latest; district matching; Price Paid rules (category B excluded, < 20 sales → missing, unmatched reported); affordability bands; conformal coverage; forecast sanity; comparison keeps missing as missing; API rents/trend/forecast/compare/predict-with-income; Greater Manchester crime → `null` |
| Full suite on the Mac (Python 3.11, pinned) | **165 passed**, 2 skipped |
| `make verify` | **PASS** — packages, health, tests, live API, live dashboard |
| Visual check (built-in browser, desktop + 375 px) | Rent check, Compare, Methodology render; forecast and trend charts legible |

## 7. Limits carried forward

Regional (not area) forecasts; Price Paid covers Jan–Jun 2026 and has no bedroom counts; Scotland and Northern
Ireland not covered; a printable one-page summary moved to Phase D (case study) as it suits the portfolio.
