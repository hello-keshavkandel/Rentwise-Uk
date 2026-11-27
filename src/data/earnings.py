"""
earnings.py  —  local pay for affordability, on the same basis ONS uses (validation fix, Nov 2026).

The real-world validation against ONS found that RentWise's "local pay" was the ASHE median for
ALL employee jobs (full- and part-time: £32,890 for the UK in 2025), while ONS housing
affordability ratios use the median for FULL-TIME employees (£39,039 UK). Mixing in part-time
jobs made every rent look less affordable than on the official basis.

The user-facing affordability now uses the full-time median (residence-based: people who live in
the area). The all-jobs median is kept, because the rent model was trained on it.

Output: data/processed/earnings.csv  (la_code, la_name, median_pay_fulltime, median_pay_all)

    python -m src.data.earnings
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger("rentwise.earnings")
LA_PREFIX = ("E06", "E07", "E08", "E09", "W06")


def read_sheet(path: str | Path, sheet: str) -> pd.Series:
    import openpyxl
    from src.data.geo import CODE_ALIASES
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {}
    for r in wb[sheet].iter_rows(min_row=6, values_only=True):
        if r[1] and str(r[1])[:3] in LA_PREFIX:
            try:
                out[CODE_ALIASES.get(r[1], r[1])] = float(r[3])
            except (TypeError, ValueError):
                pass                                   # ASHE marks unreliable cells 'x' -> missing
    wb.close()
    return pd.Series(out)


def build(raw_dir: str | Path, processed_dir: str | Path) -> dict:
    src = Path(raw_dir) / "ashe_table8_7a.xlsx"
    ft, al = read_sheet(src, "Full-Time"), read_sheet(src, "All")
    import openpyxl
    wb = openpyxl.load_workbook(src, read_only=True)
    title = str(next(wb["Full-Time"].iter_rows(max_row=1, values_only=True))[0])
    wb.close()
    year = (re.search(r"(20\d\d)", title) or [None, None])[1]
    clean = pd.read_csv(Path(processed_dir) / "rentwise_clean.csv").drop_duplicates("la_code")
    df = clean[["la_code", "la_name"]].assign(median_pay_fulltime=lambda d: d.la_code.map(ft),
                                              median_pay_all=lambda d: d.la_code.map(al))
    df.to_csv(Path(processed_dir) / "earnings.csv", index=False)
    meta = {"source": "ONS ASHE Table 8.7a, median gross annual pay, FULL-TIME employees, by place of residence",
            "period": year, "file": src.name, "areas": int(df.median_pay_fulltime.notna().sum()),
            "missing": sorted(df.loc[df.median_pay_fulltime.isna(), "la_name"])}
    log.info("earnings built: %s", meta)
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
    v["earnings_fulltime"] = m
    vp.write_text(json.dumps(v, indent=2))
    print(m)
