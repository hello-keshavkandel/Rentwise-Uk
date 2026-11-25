# RentWise UK — Phase A report: correctness

**Date:** 25 November 2026 · **Author:** Keshav Kandel · **Status:** complete, tested, validated

Phase A makes every number RentWise shows either an **official figure** or a **clearly labelled model
estimate**, with its source and date, and removes three inputs that were wrong. It was triggered by the
product-readiness audit (`PRODUCT_READINESS_PLAN.md`).

---

## 1. What was wrong, and what changed

| # | Problem found | Effect on users | Fix |
|---|---|---|---|
| 1 | The cleaner deleted **66 genuine official rents** (every rent above £2,472; 55 in London) as "outliers" | Westminster and Kensington & Chelsea vanished; the model never saw the top of the market | Official statistics are never trimmed; only impossible values (outside £100–£20,000) are dropped, and reported |
| 2 | "Crime" was a **one-month raw count** scaled 0–100, with no population denominator | Big cities scored high for being big (Birmingham = 100) | Crime = recorded street crimes, **12 months, annualised, per 1,000 residents** (ONS mid-2025 population) |
| 3 | **Greater Manchester** police data absent; a handful of its crimes logged by neighbouring forces made 10 boroughs look like the **safest areas in England** (≈0.1 per 1,000) | Wrong explanations; biased model | Coverage judged by each area's **home force**; a rate below 5 per 1,000 (real range 39–399) means the force did not publish → **"no police data"**, never zero |
| 4 | **Gloucestershire** published 4 of 12 months; counting "any record" undercounted its areas (18–46 vs national median 85) | Understated crime | Rate annualised from the home force's 4 real months; flagged "based on 4 months" |
| 5 | **"Transport access"** was a constant 0.5 placeholder | A dead input presented as a feature | Removed; real transport signal = OSM rail-station and bus-stop density |
| 6 | The cleaner **filled gaps with the median** | Missing data would have been shown as invented values | Gaps stay gaps; models handle them internally (XGBoost natively; the Ridge baseline via a train-only imputer + "was missing" flag) |
| 7 | The app showed a **model estimate** where ONS publishes the figure (mean gap £20, max £110) | A less accurate number than the official one | **Official-first:** the ONS median is the headline; the model explains it and estimates only where ONS publishes nothing |
| 8 | Rents were from the **July 2026** release; no dates anywhere | Users could not judge freshness | Rents refreshed to **October 2026** (released 18 Nov 2026); "Data sources & dates" panel in the app; `GET /v1/sources` in the API |

## 2. Data after Phase A

| Input | Vintage | Coverage |
|---|---|---|
| ONS Price Index of Private Rents | October 2026 (released 18 Nov 2026); history from Jan 2015 | 316 areas × 4 property types = 1,264 official rents; 179,488 monthly history rows |
| ONS ASHE Table 8.7a earnings | 2025 | 316 areas |
| ONS mid-year population | mid-2025 (released 29 Jul 2026) | 318 areas, total 62,009,993 (matches ONS England & Wales total) |
| data.police.uk street crime | Oct 2025 – Sep 2026 | 306 areas with a rate (39.4–399.4 per 1,000); 10 Greater Manchester boroughs without police data; 8 areas on partial months |

## 3. Model after Phase A (deployed, XGBoost 2.1.1)

| | Before Phase A | After Phase A |
|---|---|---|
| Training data | 1,198 rows, 314 areas (66 high rents deleted) | **1,264 rows, 316 areas (all official rents)** |
| R² / MAE — areas served | 0.947 / £68 | **0.945 / £79** |
| R² / MAE — whole areas held out | 0.817 / £136 | **0.858 / £152** |
| 80% / 90% interval | ±£113 / ±£161 | **±£106 / ±£175** |
| Coverage — areas served | 80.1% / 90.2% | **80.1% / 90.1%** |

The two columns are not directly comparable: Phase A restored the 66 most expensive rents in the country,
which raises errors in pounds while making the model honest about the whole market.

**Module 3 on corrected data:** spatial features reduce the typical error for unseen areas by **8%**
(£169 → £156) in **every** repeat (£10–£18); the R² gain is small (0.847 → 0.853) and not consistent. With a real
crime rate and the full rent range, the base model is much stronger, so amenities add less than before. The
summary states exactly that rather than the earlier "gain in every repeat".

## 4. Validation

| Check | Result |
|---|---|
| Automated tests | **116 passing** (new: crime rate, home-force coverage, missing-not-zero, implausible-rate rule, PIPR vintage, population reader, official-mode cleaner, official-first predictor and API, sources endpoint) |
| Every area × property type in the deployment environment (XGBoost 2.1.1) | 1,264 / 1,264 answers **equal the official ONS figure**; 0 implausible |
| Dashboard (Streamlit AppTest) — estimator (Westminster, Manchester), area map, system results | no exceptions; official headline with source and date; "no police data" list shown |
| Population cross-check | 62,009,993 = ONS England & Wales mid-2025 total |
| Fallback model | loads in the deployed XGBoost 2.1.1 |

## 5. Still open (by design or for later phases)

- **Greater Manchester crime:** no police data; shown as missing. A future option is the ONS police-force-area
  lookup plus another source (e.g. ONS recorded-crime tables) — verification required.
- **Bedroom-level official rents** (1, 2, 3, 4+ bed) exist in the ONS file and will replace the property-type →
  bedroom assumption in Phase C.
- **Affordability methodology note** moves to the Phase C methodology page.
- Interval offered only for modelled answers; official answers carry no model range.
