# RentWise UK — Product Readiness Audit & Plan

**Owner:** Keshav Kandel · **Audit date:** 25 Nov 2026 · **Direction chosen:** portfolio first → real users later ·
**Pace:** as soon as possible · **Hosting budget:** free only

This document answers three questions: *what state is RentWise in*, *what must change before real people rely
on it*, and *in what order to do it*. Every finding below was checked against the code and data, not assumed.

---

## 1. Where RentWise stands today

| Area | State | Verdict |
|---|---|---|
| Data pipeline (validation, provenance, DuckDB store, monthly workflow) | Built, tested | ✅ solid |
| Module 1 — sale price (Land Registry + real EPC, R² 0.73) | Built, report only | ⚠️ not usable in the app |
| Module 2+3 — rent model with spatial features (R² 0.947 served / 0.817 unseen, 80% interval ±£113) | Live | ⚠️ crime input wrong (see §2) |
| Module 4 — regional forecasts (MAPE ~0.6%, recalibrated bands) | Built, report only | ⚠️ not usable in the app |
| Dashboard — estimator, area map, system results | Live | ⚠️ shows a model estimate where an official figure exists |
| Public API, accounts, saved areas, alerts | Built, tested locally | ⏸ not hosted (planned later) |
| Tests | 105 passing, CI on every push | ✅ |
| Docker / compose | Written, not yet built | ⚠️ build once on the Mac |

**Bottom line:** the engineering is strong and well tested. The things standing between RentWise and "useful and
trustworthy for real people" are **data correctness** and **presenting official numbers honestly**, not more
infrastructure.

---

## 2. Audit findings

### Critical — users would get wrong answers

1. **The "crime" input is not a crime rate, and 15 areas show zero crime.**
   It is a raw count of crimes for **one month (June 2026)**, min–max scaled to 0–100 with **no population
   denominator**, so large cities score high for being large (Birmingham = 100). The police files for
   **Greater Manchester** and **Gloucestershire** are absent from the downloaded data, so 15 areas — including
   Manchester, Salford, Bolton, Bury, Oldham, Rochdale, Stockport, Tameside, Gloucester, Stroud, Cotswold and
   Forest of Dean — get **0**, i.e. "the safest places in England". The model learned from this and the app
   explains rents with it.
   *Fix:* 12 months of crime, divided by ONS mid-year population → true crimes per 1,000 residents; areas without
   police data become **missing** (labelled "no police data"), never zero; retrain and revalidate.
   *(Whether those forces publish to data.police.uk now: verification required when we re-download.)*

2. **For all 314 areas the app shows a model estimate even though ONS publishes the official figure.**
   The model's in-sample estimate differs from the official ONS median by **£20 on average, up to £110**
   (e.g. Vale of Glamorgan detached: official £1,343, app £1,233). For a served area and property type, the
   **official ONS median should be the headline**; the model's role is to *explain* it (SHAP), to fill gaps
   (34 areas lack some property types), and to give the uncertainty range.
   *Fix:* show "Official ONS median (Jun 2026): £X" first; show the model only where no official figure exists,
   clearly labelled "model estimate"; keep the factor explanation for both.

### High — misleading or incomplete

3. **"Transport access" is a placeholder** — a constant 0.5 for every area. It has no effect but should not
   exist; real transport signals (rail-station and bus-stop density from OpenStreetMap) are already available.
   *Fix:* remove the placeholder from features, labels and docs.
4. **No "data as of" dates in the app.** Rents: ONS PIPR published 22 Jul 2026; crime: Jun 2026; Ofsted:
   Oct 2026; OSM: Nov 2026; boundaries: May 2024. Users can't judge freshness.
   *Fix:* a data-vintage panel and "as of" labels beside every figure. Refresh PIPR to the latest release.
5. **Affordability method needs stating precisely.** The ratio is *annual rent ÷ median gross annual earnings
   of employees (ASHE)* — individual earnings, not household income — so it is stricter than ONS's own
   household-based affordability measures. This is legitimate, but must be explained in the app.
6. **Modules 1 and 4 are invisible to users.** Sale-price estimates and rent forecasts exist only as reports.

### Medium — needed before real users (not for the portfolio demo)

7. **UK GDPR basics for accounts:** no account-deletion or data-export endpoint, no privacy notice.
8. **Single-instance limits:** in-process rate limiter and SQLite (documented); fine for a demo.
9. **Docker images not yet built** — compose is written but unverified.
10. **Licensing attribution:** OGL (ONS, Land Registry, Ofsted/DfE, police.uk) needs attribution in the app;
    the OSM-derived counts are under **ODbL** (attribution + share-alike for the derived database if published).

### What "exact results" can and can't mean

RentWise models **typical rents by area and property type**, from official medians. It cannot value one
specific flat — no open dataset holds individual rents. The right promise is: **correct, sourced, dated, with
an honest uncertainty range and a plain-English reason** — and to say clearly what it does not do.

---

## 3. The plan (portfolio first, ASAP, free)

### Phase A — Correctness *(done 25 Nov 2026 — see reports/REPORT_PHASE_A.md)*
- [x] Crime: 12 months, per 1,000 residents (ONS mid-year population), missing forces → missing, not 0.
- [x] Remove the transport placeholder; keep OSM transport densities.
- [x] Official-first display: ONS median as headline where it exists; model estimate only where it doesn't.
- [x] Data-vintage panel + "as of" labels; refresh ONS PIPR to the latest release.
- [ ] Affordability methodology note in the app and API docs. *(moved to Phase C methodology page)*
- [x] Retrain, recalibrate, re-run the full validation (all areas × types, AppTest, API parity), update reports.

### Phase B — A 100% reliable local environment *(done — `make verify` PASS on the Mac, 26 Nov 2026: Python 3.11.18, 17/17 pins, 136 tests, live API + dashboard; Docker optional, pending)*
- [x] `make setup` → `make verify`: one command runs tests, the dashboard AppTest and an API smoke test.
- [x] `make demo`: starts dashboard + API with a seeded demo account and API key.
- [ ] Build `docker compose up` on the Mac and fix anything it reveals. *(Keshav — Docker unavailable in the sandbox)*
- [x] CI runs the dashboard AppTest and the API tests on every push.
- [x] `.env.example` complete; troubleshooting section in the README.

### Phase C — Make it genuinely useful *(done 27 Nov 2026 — reports/REPORT_PHASE_C.md)*
- [x] **Official rents by bedroom count (1/2/3/4+)** and all properties, with annual change — 316 areas × 9 measures.
- [x] **Affordability calculator** with the user's own income (never stored) + the rent affordable at 30%.
- [x] **Compare areas** side by side (2–4 areas: rent, change, ratio, prices, yield, crime, schools, amenities).
- [x] **Download** CSV: one area, a comparison, and every area × measure (analysts). *(printable summary → Phase D)*
- [x] **Trends since 2015** and **regional 12-month forecasts** (backtested model choice, conformal bands,
      out-of-time coverage reported); **sale-price context** + gross yield from HM Land Registry.
- [x] **Methodology & data page** (incl. the affordability method note carried over from Phase A).
- [x] Accessibility basics (labels/help on every input, focus outline, charts never colour-only, text
      alternatives for charts) and a mobile layout (checked at 375 px).
- [x] API: `/v1/areas/{code}/rents|trend|forecast`, `/v1/compare`, personal affordability in `/v1/predict`.

### Phase D — Portfolio case study *(~1–2 days, in parallel with C)*
- [ ] Case study page: problem → data → approach → **honest evaluation** → results → impact → next steps.
- [ ] Highlight the engineering judgement stories: grouped-CV leakage finding (0.90 vs 0.78), the crime-data
      bug found and fixed, the XGBoost version mismatch caught before deploy, Welsh-school honesty labels.
- [ ] 2-minute demo video/GIF, architecture diagram, polished README, CV bullets, LinkedIn post.
- [ ] **Credit accurately:** the MSc dissertation was a Group 27 project; the product phase (Milestones 1–5,
      this plan) is solo work by Keshav. Recruiters value this clarity.

### Phase E — Free live deployment *(when ready; ~1 day)*
- [ ] Dashboard stays on Streamlit Community Cloud (free, already live).
- [ ] API on a free container host behind HTTPS; free-tier email for alerts and password reset.
      *(Current free-tier offers change often — verification required at the time; expect cold starts.)*
- [ ] Email verification, password reset, account deletion/export, privacy notice (needed once real users sign up).

### Phase F — Real-world extension *(later, when there are users)*
- Postgres + Redis, monitoring and backups; Scotland & Northern Ireland data; Estyn grades for Wales;
  per-capita amenities; council/analyst partnerships; possible freemium API.

---

## 4. Definition of done — portfolio launch

- [ ] No area shows a fabricated or misleading input (crime, transport) — verified by test.
- [ ] Every figure in the app shows its source and date; official figures shown as official.
- [ ] `make verify` passes on a fresh clone on the Mac; `docker compose up` works.
- [ ] Live app updated; case study, demo video and README published.
- [ ] Limitations stated in the app, the README and the case study.

## 5. What's needed from Keshav

- Approve starting **Phase A** now.
- During Phase A: download 12 months of police data and the ONS mid-year population estimates (I'll give exact
  links and steps, as before), and the latest ONS PIPR release.
- Run `docker compose up --build` once on the Mac in Phase B.
