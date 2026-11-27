"""
rent_dataset.py  —  build the official Module 2 dataset from real open data (Phase A).

Replaces the dissertation-era builder, fixing four correctness problems found in the
Nov 2026 audit:

1. CRIME is now a true rate: recorded street crimes over the months available,
   annualised, per 1,000 residents (ONS mid-year population). Areas with NO police
   data (e.g. a force absent from the download) are MISSING, never zero.
2. The constant TRANSPORT placeholder (0.5 everywhere) is gone; real transport
   signals come from OpenStreetMap rail/bus density in Module 3.
3. Official ONS rents are kept exactly as published — none are trimmed as
   "outliers" (that previously deleted 66 genuine rents, incl. all of Westminster).
4. Every input's VINTAGE is recorded (data period + release) in data_vintages.json,
   so the app can show "as of" beside each figure.

Outputs (data/processed/):
  rentwise_sample.csv   one row per local authority x property type (latest month)
  rent_history.csv      monthly official rents per LA x property type since Jan 2015
  data_vintages.json    source, period and release for every input
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.geo import CODE_ALIASES

log = logging.getLogger("rentwise.dataset")

LA_PREFIX = ("E06", "E07", "E08", "E09", "W06")
REGIONS = {"North East", "North West", "Yorkshire and The Humber", "East Midlands", "West Midlands",
           "East of England", "London", "South East", "South West", "Wales"}
TYPE_COL = {"Detached": "Rental price detached", "Semi-detached": "Rental price semidetached",
            "Terraced": "Rental price terraced", "Flat": "Rental price flat maisonette"}
TYPE_BEDS = {"Flat": 2, "Terraced": 3, "Semi-detached": 3, "Detached": 4}
MISSING_TOKENS = {None, "[x]", "[z]", "[c]", "[u]", ""}


def _alias(code):
    return CODE_ALIASES.get(code, code)


def _num(v):
    if v in MISSING_TOKENS:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ rents (ONS PIPR)
def read_pipr(path: str | Path):
    """Official rents: (latest LA x type table, full monthly history, meta)."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    cover = [str(c) for r in wb["Cover sheet"].iter_rows(max_row=15, values_only=True) for c in r if c]
    published = next((m.group(1) for t in cover
                      for m in [re.search(r"published at [0-9:apm]+ on (\d{1,2} \w+ \d{4})", t)] if m), None)
    it = wb["Table 1"].iter_rows(min_row=3, values_only=True)
    hdr = list(next(it))
    ci = {h: i for i, h in enumerate(hdr) if h}
    hist = []
    for r in it:
        code, period = r[ci["Area code"]], r[ci["Time period"]]
        if code is None or period is None or str(code)[:3] not in LA_PREFIX:
            continue
        region = r[ci["Region or country name"]]
        if region not in REGIONS:
            continue
        for pt, col in TYPE_COL.items():
            v = _num(r[ci[col]])
            if v is not None:
                hist.append((code, r[ci["Area name"]], region, period, pt, v))
    wb.close()
    h = pd.DataFrame(hist, columns=["la_code", "la_name", "region", "period", "property_type", "rent"])
    h["period"] = pd.to_datetime(h["period"]).dt.strftime("%Y-%m")
    latest = h["period"].max()
    cur = h[h["period"] == latest].copy()
    meta = {"source": "ONS Price Index of Private Rents (PIPR), monthly price statistics",
            "period": latest, "history_from": h["period"].min(), "released": published,
            "file": Path(path).name}
    return cur, h, meta


def latest_pipr_file(raw_dir: Path) -> Path:
    """The ONS PIPR workbook with the most recent data month (not file date)."""
    files = sorted(glob.glob(str(Path(raw_dir) / "ons_pipr_monthly*.xlsx")))
    if not files:
        raise FileNotFoundError("no ons_pipr_monthly*.xlsx in data/raw")
    if len(files) == 1:
        return Path(files[0])
    best, best_p = None, ""
    for f in files:
        _, h, m = read_pipr(f)
        if m["period"] > best_p:
            best, best_p = Path(f), m["period"]
    return best


# ------------------------------------------------------------------ earnings (ASHE)
def read_ashe(path: str | Path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["All"]
    title = next(ws.iter_rows(max_row=1, values_only=True))[0] or ""
    inc = {}
    for r in ws.iter_rows(min_row=6, values_only=True):
        if r[1] and str(r[1])[:3] in LA_PREFIX:
            v = _num(r[3])
            if v is not None:
                inc[_alias(r[1])] = v
    wb.close()
    year = (re.search(r"(20\d\d)", str(title)) or [None, None])[1]
    return pd.Series(inc, name="median_annual_income"), {
        "source": "ONS ASHE Table 8.7a, median gross annual pay, all employee jobs, by place of residence",
        "period": year, "file": Path(path).name}


# ------------------------------------------------------------------ population (ONS MYE)
def read_population(path: str | Path):
    """Mid-year population per LA (all ages). Finds the LA table by its header."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    pop, label = {}, None
    for ws in wb.worksheets:
        rows = ws.iter_rows(max_row=12, values_only=True)
        for i, r in enumerate(rows):
            cells = [str(c).strip() if c is not None else "" for c in r]
            if "Code" in cells and any(c.lower().startswith("all ages") or c.lower().startswith("mid-20")
                                       for c in cells):
                code_i = cells.index("Code")
                val_i = next(j for j, c in enumerate(cells)
                             if c.lower().startswith("all ages") or c.lower().startswith("mid-20"))
                label = f"{ws.title}: {cells[val_i]}"
                for rr in ws.iter_rows(min_row=i + 2, values_only=True):
                    code = rr[code_i]
                    if code and str(code)[:3] in LA_PREFIX:
                        v = _num(rr[val_i])
                        if v:
                            pop[_alias(code)] = v
                break
        if pop:
            break
    wb.close()
    if not pop:
        raise ValueError(f"could not find a local-authority population table in {path}")
    yr = (re.search(r"(20\d\d)", Path(path).name.replace("mye", "mye20")) or [None, None])[1]
    return pd.Series(pop, name="population"), {
        "source": "ONS mid-year population estimates, England and Wales", "period": f"mid-{yr}" if yr else None,
        "table": label, "file": Path(path).name}


# ------------------------------------------------------------------ crime (police.uk)
def crime_counts(police_dir: str | Path, lsoa_to_lad: dict) -> pd.DataFrame:
    """Street-crime counts per LA, judged by each area's HOME police force.

    A few crimes in an area are logged by a neighbouring force, so counting "any
    record" makes an area whose own force published nothing (e.g. Greater Manchester)
    look covered with a near-zero rate. Instead:
      * home force  = the force area the LA's crimes "Fall within" (most common value),
      * crimes      = records REPORTED BY that home force,
      * months      = months in which the home force actually published data.
    An area whose home force published no months gets months_covered = 0 (-> missing).
    """
    files = sorted(glob.glob(str(Path(police_dir) / "**" / "*-street.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no *-street.csv files under {police_dir}")
    falls = collections.defaultdict(collections.Counter)      # lad -> Counter(force area)
    by_reporter = collections.defaultdict(collections.Counter) # lad -> Counter(reporting force)
    published = collections.defaultdict(set)                   # force -> months it reported
    all_months = set()
    for fp in files:
        month = Path(fp).name[:7]                              # YYYY-MM-<force>-street.csv
        all_months.add(month)
        with open(fp, newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                rep = (row.get("Reported by") or "").strip()
                if rep:
                    published[rep].add(month)
                lad = lsoa_to_lad.get(row.get("LSOA code"))
                if not lad:
                    continue
                lad = _alias(lad)
                fw = (row.get("Falls within") or rep).strip()
                falls[lad][fw] += 1
                by_reporter[lad][rep] += 1
    recs = {}
    for lad, fc in falls.items():
        home = fc.most_common(1)[0][0]
        recs[lad] = {"home_force": home, "crimes": float(by_reporter[lad][home]),
                     "months_covered": float(len(published.get(home, ())))}
    df = pd.DataFrame.from_dict(recs, orient="index")
    df.attrs["months"] = sorted(all_months)
    df.attrs["force_months"] = {f: len(m) for f, m in published.items()}
    return df


# Annual recorded street crime per 1,000 residents ranges from ~45 (rural Rutland) to
# ~400 (central Westminster) where a force publishes. A rate below this floor means the
# area's own force did not publish (its few records were logged by a neighbouring force,
# e.g. Greater Manchester): the rate is unknown, not low.
MIN_PLAUSIBLE_RATE = 5.0


def crime_rate(counts: pd.DataFrame, population: pd.Series, min_months: int = 3,
               min_plausible: float = MIN_PLAUSIBLE_RATE) -> pd.Series:
    """Annualised crimes per 1,000 residents. Missing (NaN) when an LA has fewer than
    `min_months` months from its home force, no population, or an implausibly low
    rate (see MIN_PLAUSIBLE_RATE) — never zero."""
    df = counts.reindex(population.index)
    months = df["months_covered"].fillna(0)
    annual = df["crimes"] / months.where(months > 0) * 12
    rate = annual / population * 1000
    ok = (months >= min_months) & (rate >= min_plausible)
    return rate.where(ok).rename("crime_per_1000")


# ------------------------------------------------------------------ assemble
def build(raw_dir: str | Path, processed_dir: str | Path, police_dir: str | Path | None = None,
          population_file: str | Path | None = None) -> dict:
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    pipr = latest_pipr_file(raw_dir)
    cur, hist, rent_meta = read_pipr(pipr)
    income, ashe_meta = read_ashe(raw_dir / "ashe_table8_7a.xlsx")

    pop_file = Path(population_file) if population_file else next(
        iter(sorted(glob.glob(str(raw_dir / "population" / "*.xlsx")))), None)
    if pop_file is None:
        raise FileNotFoundError("ONS mid-year population workbook missing (data/raw/population/)")
    population, pop_meta = read_population(pop_file)

    pdir = Path(police_dir) if police_dir else raw_dir / "police_12m"
    lookup = pd.read_csv(raw_dir / "lsoa_to_lad.csv", usecols=["LSOA21CD", "LAD24CD"], encoding="utf-8-sig")
    counts = crime_counts(pdir, dict(zip(lookup.LSOA21CD, lookup.LAD24CD)))
    rate = crime_rate(counts, population)
    months = counts.attrs["months"]

    cur = cur.assign(la_code=cur.la_code.map(_alias))
    cur["year"] = int(rent_meta["period"][:4])
    cur["bedrooms"] = cur.property_type.map(TYPE_BEDS)
    cur["median_annual_income"] = cur.la_code.map(income)
    cur["crime_per_1000"] = cur.la_code.map(rate)
    cur["median_monthly_rent"] = cur.rent.round().astype(int)
    no_income = cur.median_annual_income.isna()
    if no_income.any():
        log.warning("dropping %d rows with no ASHE earnings: %s", int(no_income.sum()),
                    sorted(cur.loc[no_income, "la_name"].unique()))
    cur = cur[~no_income]
    cols = ["la_code", "la_name", "region", "year", "property_type", "bedrooms",
            "median_annual_income", "crime_per_1000", "median_monthly_rent"]
    out = cur[cols].sort_values(["la_name", "property_type"]).reset_index(drop=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(processed_dir / "rentwise_sample.csv", index=False)
    hist.assign(la_code=hist.la_code.map(_alias)).to_csv(processed_dir / "rent_history.csv", index=False)
    missing_crime = sorted(out.loc[out.crime_per_1000.isna(), "la_name"].unique())
    # areas whose crime rate rests on fewer months than the full window (e.g. a force
    # that stopped publishing part-way), so the app can say "based on N months"
    has_rate = set(out.loc[out.crime_per_1000.notna(), "la_code"])
    names = dict(zip(out.la_code, out.la_name))
    partial = {names[c]: int(m) for c, m in counts["months_covered"].items()
               if c in has_rate and m < len(months)}
    # Phase C: every official measure (all / 1-4+ bed / 4 types) + history since 2015,
    # and median sale prices for buy-vs-rent context and gross yield.
    from src.data import official_rents, sale_prices
    rent_meta["measures"] = official_rents.build(raw_dir, processed_dir, pipr)["measures"]
    sales_meta = None
    if (raw_dir / "land_registry_pp_2026.csv").exists():
        sales_meta = sale_prices.build(raw_dir, processed_dir)
    vintages = {
        "rents": rent_meta, "earnings": ashe_meta, "population": pop_meta,
        **({"sale_prices": sales_meta} if sales_meta else {}),
        "crime": {"source": "data.police.uk street-level crime, all forces",
                  "period": f"{months[0]} to {months[-1]}" if months else None, "months": len(months),
                  "measure": "recorded street crimes per 1,000 residents, annualised",
                  "areas_without_police_data": missing_crime,
                  "areas_with_partial_months": partial},
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (processed_dir / "data_vintages.json").write_text(json.dumps(vintages, indent=2))
    summary = {"rows": len(out), "las": int(out.la_code.nunique()), "rent_period": rent_meta["period"],
               "crime_months": len(months), "las_without_crime": len(missing_crime),
               "history_rows": len(hist)}
    log.info("dataset built: %s", summary)
    return summary


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    print(build(config.DATA_RAW, config.DATA_PROCESSED))
