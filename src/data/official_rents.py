"""
official_rents.py  —  every official ONS rent RentWise can show, latest and since 2015 (Phase C).

The ONS Price Index of Private Rents (PIPR) publishes, for each local authority, region and
nation, a monthly average rent (the PIPR price level, a geometric-mean measure) for:
  * all properties,
  * each bedroom count (1, 2, 3, 4 or more bed),
  * each property type (detached, semi-detached, terraced, flat/maisonette),
plus the annual % change of each. Phase A used only the four property types; this module
keeps all nine measures so the app and API can answer "what does a 2-bed cost here?" with
the official figure rather than a model.

Outputs (data/processed/, both small and committed so the live app can read them):
  official_rents.csv     latest month: one row per area x measure, with annual change
  rent_trends.parquet    monthly history since Jan 2015 for every area x measure

    python -m src.data.official_rents          # rebuild from the newest PIPR workbook
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger("rentwise.official")

# measure -> PIPR column suffix ("Rental price <suffix>", "Annual change <suffix>")
MEASURES = {
    "All": "",
    "1 bed": "one bed", "2 bed": "two bed", "3 bed": "three bed", "4+ bed": "four or more bed",
    "Detached": "detached", "Semi-detached": "semidetached", "Terraced": "terraced",
    "Flat": "flat maisonette",
}
BEDROOM_MEASURES = ["1 bed", "2 bed", "3 bed", "4+ bed"]
TYPE_MEASURES = ["Detached", "Semi-detached", "Terraced", "Flat"]
LA_PREFIX = ("E06", "E07", "E08", "E09", "W06")
REGION_PREFIX = ("E12",)
NATION_CODES = {"E92000001": "England", "W92000004": "Wales", "K02000001": "United Kingdom"}
MISSING = {None, "", "[x]", "[z]", "[c]", "[u]"}


def _num(v):
    if v in MISSING:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _level(code: str) -> str | None:
    code = str(code)
    if code[:3] in LA_PREFIX:
        return "la"
    if code[:3] in REGION_PREFIX:
        return "region"
    if code in NATION_CODES:
        return "nation"
    return None


def read_pipr_all(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One pass over PIPR Table 1 -> (latest official rents, full history, meta)."""
    import openpyxl
    from src.data.geo import CODE_ALIASES

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    cover = [str(c) for r in wb["Cover sheet"].iter_rows(max_row=15, values_only=True) for c in r if c]
    released = next((m.group(1) for t in cover
                      for m in [re.search(r"published at [0-9:apm]+ on (\d{1,2} \w+ \d{4})", t)] if m), None)
    it = wb["Table 1"].iter_rows(min_row=3, values_only=True)
    hdr = [str(h).strip() if h else "" for h in next(it)]
    ci = {h: i for i, h in enumerate(hdr) if h}
    cols = {}
    for m, suf in MEASURES.items():
        price = f"Rental price {suf}".strip()
        if price in ci:                                   # older/test files may lack some measures
            cols[m] = (ci[price], ci.get(f"Annual change {suf}".strip()))
    rows = []
    for r in it:
        code, period = r[ci["Area code"]], r[ci["Time period"]]
        if code is None or period is None:
            continue
        level = _level(code)
        if level is None:
            continue
        name = r[ci["Area name"]]
        region = r[ci["Region or country name"]] if level == "la" else None
        for m, (pi, ai) in cols.items():
            v = _num(r[pi])
            if v is not None:
                rows.append((CODE_ALIASES.get(code, code), name, level, region, period, m, v,
                             _num(r[ai]) if ai is not None else None))
    wb.close()
    h = pd.DataFrame(rows, columns=["area_code", "area_name", "level", "region", "period", "measure",
                                    "rent", "annual_change_pct"])
    h["period"] = pd.to_datetime(h["period"]).dt.strftime("%Y-%m")
    h["rent"] = h["rent"].round().astype(int)
    latest = h["period"].max()
    cur = h[h["period"] == latest].reset_index(drop=True)
    meta = {"source": "ONS Price Index of Private Rents (PIPR), monthly price statistics",
            "period": latest, "history_from": h["period"].min(), "released": released,
            "file": Path(path).name, "measures": list(cols)}
    return cur, h, meta


def build(raw_dir: str | Path, processed_dir: str | Path, pipr_file: str | Path | None = None) -> dict:
    from src.data.rent_dataset import latest_pipr_file
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    src = Path(pipr_file) if pipr_file else latest_pipr_file(raw_dir)
    cur, hist, meta = read_pipr_all(src)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cur.to_csv(processed_dir / "official_rents.csv", index=False)
    trends = hist[["area_code", "level", "period", "measure", "rent"]].copy()
    for c in ("area_code", "level", "measure"):
        trends[c] = trends[c].astype("category")
    trends.to_parquet(processed_dir / "rent_trends.parquet", index=False, compression="zstd")
    names = hist.drop_duplicates("area_code")[["area_code", "area_name", "level", "region"]]
    names.to_csv(processed_dir / "rent_areas.csv", index=False)
    summary = {"period": meta["period"], "history_from": meta["history_from"], "measures": meta["measures"],
               "latest_rows": len(cur), "history_rows": len(hist),
               "las": int(cur.loc[cur.level == "la", "area_code"].nunique())}
    log.info("official rents built: %s", summary)
    return summary


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    print(build(config.DATA_RAW, config.DATA_PROCESSED))
