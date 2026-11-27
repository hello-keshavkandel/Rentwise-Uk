# RentWise UK — real-world validation against official and public sources

**Date:** 27 November 2026 · **Author:** Keshav Kandel · **Scope:** the live app (https://rentwiseuk.streamlit.app) after
the Phase C release, checked view by view, and every figure type checked against the source that publishes it.

**Bottom line.** Every official rent RentWise shows is **exactly** the figure ONS published. The check found
**two real correctness problems**, both now fixed: (1) rents were labelled "median" when ONS publishes an
**average** (geometric-mean) rent, and (2) "local pay" used all employee jobs, including part-time, where ONS's own
affordability statistics use **full-time** pay. Sale prices, crime and forecasts are consistent with the official
and market sources once the methodological differences are understood.

## 1. Functional check of the live app (27 Nov 2026)

| View | Checked | Result |
|---|---|---|
| Rent check | Leeds (default); Westminster × 2 bedrooms × income £65,000 | ✅ £3,330; 61% of income (3,330 × 12 ÷ 65,000 = 61.5%); rent at 30% £1,625 (correct) |
| Compare areas | default trio, table, both CSV downloads | ✅ no errors; downloads present |
| Area map | choropleth renders, all layers selectable | ✅ |
| Methodology & data | full text, CSV download | ✅ |
| System results | all module summaries | ✅ |
| Health banner | "System ok · data 1d · model 1d" | ✅ |

No exceptions in any view. (Driven in a 1280 px browser; mobile layout was checked at 375 px in Phase C.)

## 2. Rents — compared with ONS "Private rent and house prices, UK: November 2026"

| Figure (October 2026) | ONS bulletin | RentWise | Match |
|---|---|---|---|
| UK average rent, annual change | £1,411, +3.8% | £1,411, +3.8% | ✅ exact |
| England | £1,472, +4.0% | £1,472, +4.0% | ✅ |
| Wales | £849, +3.9% | £849, +3.9% | ✅ |
| London / North East | £2,350 / £793 | £2,350 / £793 | ✅ |
| Highest / lowest regional growth | North West 5.6% / South West 3.3% | 5.6% / 3.3% | ✅ |
| Highest local rent | Kensington and Chelsea £3,765 | £3,765 (highest of 316) | ✅ |
| Highest outside London | Oxford £1,956 | £1,956 | ✅ |
| UK detached / flats | £1,603 / £1,377 | £1,603 / £1,377 | ✅ |
| UK 4+ bed / 1 bed | £2,097 / £1,145 | £2,097 / £1,145 | ✅ |

These are now locked in by `tests/test_validation_official.py` (15 golden checks).

**Problem found — wrong statistic name.** The bulletin states that PIPR rent prices measure *"the geometric mean
rent price for the 'average' privately-rented property"*, adjusted for the mix of homes. RentWise called them
"median". **Fixed:** every label, help text, the methodology page, API notes and docs now say **"official average
rent"**, with a plain explanation (rents actually paid, new and existing tenancies, not asking rents).

**New source to add next.** On 18 November 2026 ONS launched a separate *Private rent price median and
quartiles, England and Wales* dataset (snapshot medians and quartiles by area and bedrooms). It answers "what
range should I expect to pay?" — ideal for renters. ONS advises: PIPR for change over time, medians/quartiles for
comparing places at one point in time. Recommended for the next release (needs a manual download).

## 3. Earnings and affordability

**Problem found — earnings basis.** RentWise's local pay was the ASHE median for **all employee jobs**
(full- and part-time): UK £32,890. ONS housing affordability statistics use **full-time** employees: UK £39,039.
Including part-time jobs made every area look less affordable than on the official basis.

**Fixed:** affordability now uses the **full-time median gross annual pay of residents** (ASHE 2025 Table 8.7a,
"Full-Time" sheet, residence-based) — new `data/processed/earnings.csv`, used by the app, the map, the API and
alerts. The rent model keeps its all-jobs feature (it was trained on it), clearly documented.

| Area | Avg rent (all homes) | Full-time median pay | Rent ÷ pay, RentWise (one earner) | ONS private-rental affordability (household income, FYE 2024) |
|---|---|---|---|---|
| England | £1,472 | £39,243 | 45.0% | 36.3% |
| London | £2,350 | £46,414 | 60.8% | 41.6% |
| North East | £793 | £34,833 | 27.3% | 19.8% |
| Wales | £849 | £36,354 | 28.0% | 25.9% |

The ranking agrees (London least affordable, North East most). RentWise's figure is higher because it compares
rent with **one** full-time earner, while ONS uses the **household** income of private renters (many have two
earners) — the app now says this explicitly and invites users to enter their household income.

After the fix: **198 of 316 areas (63%)** have average rent above 30% of one local full-time median salary
(median area 33%; range 20%–86%). Examples: Leeds 38%, Manchester 46%, Bristol 58%, Hartlepool 21%.
*(Phase C's "276 of 316 / median 40%" used all-jobs pay and is superseded.)*

### 3a. Rent ranges added (ONS median & quartiles, September 2026 snapshot)

Integrated the new ONS dataset so users see **what most people pay**: lower quartile, median and upper quartile
by bedroom count for all 316 areas (Studio for 234 where published). Cross-check against the PIPR averages —
two different ONS measures built from the same rents:

| Measure | Median ÷ PIPR average (median across LAs) | Correlation across LAs | PIPR average inside the middle half |
|---|---|---|---|
| 1 bed | 1.00 | 0.998 | 98% |
| 2 bed | 0.99 | 0.999 | 99% |
| 3 bed | 0.99 | 0.999 | 99% |
| 4+ bed | 0.98 | 0.994 | 98% |
| All homes | 0.94 | 0.990 | 90% |

By bedroom count the two agree closely. For "all homes" the median is about 6% lower, and nationally much
lower (England median £1,052 vs average £1,472), because PIPR weights and mix-adjusts to a fixed basket while the
median reflects the homes in that month's data — ONS warns the two are not directly comparable. The app shows both,
with that explanation, and uses the range only to compare places at one point in time (ONS's guidance).
Example — Leeds, 2 bedrooms: average £978; median £962; middle half £913–£1,015 (3,098 rents).

## 4. Sale prices — compared with ONS house price statistics

| | ONS (HPSSA median, year to Sept 2025) | RentWise (Land Registry Price Paid median, Jan–Jun 2026, standard sales) |
|---|---|---|
| England | £300,000 | £290,000 |
| Wales | £213,000 | £215,000 |

Wales agrees within 1%. England is 3% lower, consistent with the period difference: ONS reports London prices
falling 3.5% (Inner London −7.2%) in the year to September 2026, and H1 excludes the stronger summer months.
UK HPI averages (England £293,000, Sept 2026) are a different, mix-adjusted measure and are not directly comparable.
**Verdict:** consistent. Gross yields (median 4.5%; flats 6.5%, detached 4.0%) are in the range commonly quoted for
UK buy-to-let; they are shown as indicative.

## 5. Crime — compared with national police-recorded crime

RentWise's 12-month data.police.uk street-level crime totals about **5.65 million** offences for the 306 areas
with data (≈ 96 per 1,000 residents). The Criminal Justice System statistics report **7.08 million** police-recorded
crimes in the year to June 2026 **including fraud and computer misuse**, which police.uk street-level data does not
contain; Greater Manchester (10 boroughs) is also missing from police.uk. Area rates look right (Westminster 399 per
1,000 — the highest; Rutland 46). **Verdict:** consistent in scale and ranking; it is a street-crime rate, not total
recorded crime, which the methodology page states.

## 6. Forecasts — compared with published market forecasts

| Forecaster | Measure | Forecast |
|---|---|---|
| **RentWise** (damped-trend ETS on ONS PIPR) | all tenancies, UK, Oct 2026 → Oct 2027 | **+4.3%** (£1,472; 80% band £1,436–£1,508) |
| Hamptons | new-let rents, 2027 | +4.5% (peak), then 3.5% and 2.5% |
| Hamptons | whole private rented sector, Q4 2026–Q4 2029 | +10.9% (≈ 3.5% a year) |
| Knight Frank | UK rents, five years | +18.1% (≈ 3.4% a year) |

RentWise sits at the **upper end** of published forecasts. Its 12-month figure follows the recent pick-up in the
official series (UK rents grew at an annualised 5.3% over the last three months versus 3.8% over the year). The
backtest shows the model over-forecast by about 1% at 12 months in the most recent periods, and its 80% band covered
71% out of time — both stated in the app. **Verdict:** plausible and honestly caveated; treat as a guide.

## 7. What changed as a result of this validation

| Fix | Files |
|---|---|
| "Median" → "average" (geometric mean) everywhere users or API clients see it | `app/streamlit_app.py`, `src/api/app.py`, `src/models/*.py`, docs |
| Full-time median pay for all affordability figures (app, map, API, alerts) | `src/data/earnings.py` → `data/processed/earnings.csv`; `src/models/area_data.py`; `src/models/predict.py` |
| Methodology page: comparison with ONS household-based affordability; "checked against ONS" statement | `app/streamlit_app.py` |
| Golden tests against the ONS bulletin and ASHE | `tests/test_validation_official.py` |
| Pipeline builds earnings automatically | `run_real_pipeline.py`, `make rents` |

Tests: **181 passed**, 2 skipped (Python 3.11, pinned). 

## 8. Guidance for real users (now reflected in the app)

- The rent shown is what tenants **actually pay on average** (new and existing lets). New lets you see advertised
  usually cost more.
- "Share of local pay" assumes **one full-time earner**; enter your household income for your own answer.
- Sale prices and yields are indicative; forecasts are regional extrapolations, not promises.

## Sources

- ONS, [Private rent and house prices, UK: November 2026](https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/privaterentandhousepricesuk/november2026)
- ONS, [Private rental affordability, England, Wales and Northern Ireland: 2024](https://www.ons.gov.uk/peoplepopulationandcommunity/housing/bulletins/privaterentalaffordabilityengland/2024)
- ONS, [Housing affordability in England and Wales: 2025](https://www.ons.gov.uk/peoplepopulationandcommunity/housing/bulletins/housingaffordabilityinenglandandwales/2025)
- ONS ASHE 2025 Table 8.7a (file `ashe_table8_7a.xlsx`, sheets "All" and "Full-Time")
- Ministry of Justice, [Criminal justice statistics quarterly: June 2026](https://www.gov.uk/government/statistics/criminal-justice-statistics-quarterly-june-2026/criminal-justice-statistics-quarterly-june-2026-html)
- Hamptons / Knight Frank forecasts as reported in [Mortgage Professional](https://www.mpamag.com/uk/mortgage-types/buy-to-let/rents-to-rise-twice-as-fast-as-house-prices-forecasts-show/598421) and [BigGo Finance](https://finance.biggo.com/news/efc0526c-4ad3-41c1-9775-a96dbb8bfcfd) — secondary reports; verify against the firms' own publications before quoting.
