"""
rent_quartiles.py  —  what most people actually pay: ONS median and quartile rents (Nov 2026).

ONS launched "Private rent price median and quartiles, England and Wales" on 18 November 2026. For each
local authority (and England, Wales) and bedroom category it gives the lower quartile, median and
upper quartile monthly rent, and the number of rents used, for one month at a time.

How RentWise uses it — following ONS's own guidance:
  * It answers "what range should I expect?" for ONE point in time: "half of 2-bed homes here rent
    for £X–£Y, median £Z". It is a snapshot, NOT a trend: ONS states these figures are not comparable
    between months (no mix adjustment), so only the latest month is kept.
  * It complements, and does not replace, the PIPR average (geometric mean, mix-adjusted, the right
    measure for change over time). The two measure different things and can differ noticeably.
  * Suppressed cells stay missing: "[c]" counts (≤10 rents), "[u]" unreliable quartiles (≤10 rents),
    "[x]" no data. Nothing is imputed.

Output: data/processed/rent_quartiles.csv
  area_code, area_name, level, period, measure, count, lower_quartile, median, upper_quartile

    python -m src.data.rent_quartiles
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger("rentwise.quartiles")

# measure -> (count column, lower quartile, median, upper quartile) header text in Table 1
MEASURE_COLS = {
    "All": ("Count of All categories", "All Lower Quartile (£)", "All Median (£)", "All Upper quartile (£)"),
    "Studio": ("Count of rents Studio", "Lower quartile Studio (£)", "Median Studio (£)",
               "Upper quartile Studio (£)"),
    "1 bed": ("Count of rents One Bedrooms", "Lower quartile One Bedrooms (£)", "Median One Bedrooms (£)",
              "Upper quartile One Bedrooms (£)"),
    "2 bed": ("Count of rents Two Bedrooms", "Lower quartile Two Bedrooms (£)", "Median Two Bedrooms (£)",
              "Upper quartile Two Bedrooms (£)"),
    "3 bed": ("Count of rents Three Bedrooms", "Lower quartile Three Bedrooms (£)", "Median Three Bedrooms (£)",
              "Upper quartile Three Bedrooms (£)"),
    "4+ bed": ("Count of Four or more Bedrooms", "Lower quartile Four or more bedrooms (£)",
               "Median Four or more bedrooms (£)", "Upper quartile Four or more bedrooms (£)"),
}
LA_PREFIX = ("E06", "E07", "E08", "E09", "W06")
NATIONS = {"E92000001", "W92000004"}
SUPPRESSED = {"[c]", "[u]", "[x]", "[z]", "", None}


def _num(v):
    if v in SUPPRESSED:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_quartiles(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Latest month of ONS median/quartile rents -> long table + meta."""
    import openpyxl
    from src.data.geo import CODE_ALIASES
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    cover = [str(c) for r in wb["Cover sheet"].iter_rows(max_row=15, values_only=True) for c in r if c]
    released = next((m.group(1) for t in cover
                      for m in [re.search(r"published at [0-9:apm]+ on (\d{1,2} \w+ \d{4})", t)] if m), None)
    it = wb["Table 1"].iter_rows(min_row=3, values_only=True)
    norm = {re.sub(r"\s+", " ", str(h)).strip().lower(): i for i, h in enumerate(next(it)) if h}

    def col(name):
        return norm.get(re.sub(r"\s+", " ", name).strip().lower())
    cols = {m: tuple(col(c) for c in cs) for m, cs in MEASURE_COLS.items() if all(col(c) is not None for c in cs)}
    rows = []
    for r in it:
        period, code = r[col("Time period")], r[col("Area code")]
        if period is None or code is None:
            continue
        level = "la" if str(code)[:3] in LA_PREFIX else ("nation" if code in NATIONS else None)
        if level is None:
            continue
        for m, (ci, lq, md, uq) in cols.items():
            rows.append((CODE_ALIASES.get(code, code), r[col("Area name")], level, period, m,
                         _num(r[ci]), _num(r[lq]), _num(r[md]), _num(r[uq])))
    wb.close()
    df = pd.DataFrame(rows, columns=["area_code", "area_name", "level", "period", "measure", "count",
                                     "lower_quartile", "median", "upper_quartile"])
    df["period"] = pd.to_datetime(df["period"]).dt.strftime("%Y-%m")
    latest = df["period"].max()
    df = df[df.period == latest].reset_index(drop=True)
    for c in ("lower_quartile", "median", "upper_quartile"):
        df[c] = df[c].round()
    meta = {"source": "ONS Private rent price median and quartiles, England and Wales (from PIPR data)",
            "period": latest, "released": released, "file": Path(path).name, "measures": list(cols),
            "note": "Snapshot for one month; not comparable over time; not the same measure as the PIPR average."}
    return df, meta


def build(raw_dir: str | Path, processed_dir: str | Path) -> dict:
    src = Path(raw_dir) / "privaterentpricemedianandquartilesenglandandwales.xlsx"
    df, meta = read_quartiles(src)
    df.to_csv(Path(processed_dir) / "rent_quartiles.csv", index=False)
    la = df[df.level == "la"]
    meta["areas"] = int(la.area_code.nunique())
    meta["published_medians"] = int(la["median"].notna().sum())
    log.info("rent quartiles built: %s", meta)
    return meta


if __name__ == "__main__":  # pragma: no cover
    import json
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    m = build(config.DATA_RAW, config.DATA_PROCESSED)
    vp = config.DATA_PROCESSED / "data_vintages.json"
    v = json.loads(vp.read_text())
    v["rent_quartiles"] = m
    vp.write_text(json.dumps(v, indent=2))
    print(m)
