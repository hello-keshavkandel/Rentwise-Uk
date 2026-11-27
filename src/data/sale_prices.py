"""
sale_prices.py  —  what homes sell for in each area, next to what they rent for (Phase C).

Module 1 predicts individual sale prices; for the app, the useful and honest figure is the
**median price actually paid** in each local authority, from HM Land Registry Price Paid
data. Next to the official ONS rent this gives:

  * buy-vs-rent context for renters ("a typical flat here sells for £X"),
  * an indicative **gross rental yield** for landlords and analysts
    (annual official median rent / median sale price).

Rules
  * Standard-price sales only (PPD category A). Category B (repossessions, buy-to-let
    portfolios, transfers) would distort a "typical price".
  * A median is published only with at least MIN_SALES sales; otherwise it is missing.
  * District names in Price Paid are matched to ONS local-authority names (normalised, plus
    two known spelling differences); unmatched districts are reported, never guessed.

The yield is indicative: rents and sales are medians over different homes (renters and
buyers occupy different stock), and the yield is gross (before costs, voids and tax).

Output: data/processed/sale_prices.csv  (one row per LA x property type, plus "All")

    python -m src.data.sale_prices
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger("rentwise.sales")

PPD_COLS = ["id", "price", "date", "postcode", "ptype", "new_build", "tenure", "paon", "saon", "street",
            "locality", "town", "district", "county", "category", "status"]
PTYPE = {"D": "Detached", "S": "Semi-detached", "T": "Terraced", "F": "Flat"}
MIN_SALES = 20
# Price Paid district name -> ONS local-authority name, where normalising is not enough
DISTRICT_ALIASES = {"RHONDDA CYNON TAFF": "RHONDDA CYNON TAF", "WREKIN": "TELFORD AND WREKIN"}


def norm_name(s: str) -> str:
    """Comparable form of an LA / district name ('Bristol, City of' == 'CITY OF BRISTOL')."""
    s = str(s).upper().replace("&", " AND ")
    s = re.sub(r",\s*(CITY|COUNTY) OF$", "", s)
    s = re.sub(r"^(CITY|COUNTY) OF ", "", s)
    s = re.sub(r"^THE ", "", s)
    s = re.sub(r"[^A-Z ]", " ", s)
    s = re.sub(r"\bST\b", "SAINT", s)
    s = re.sub(r"\s+", " ", s).strip()
    return DISTRICT_ALIASES.get(s, s)


def summarise(ppd: pd.DataFrame, la_names: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Median price per LA x type from Price Paid rows. la_names: la_code -> ONS name."""
    lookup = {norm_name(n): c for c, n in la_names.items()}
    d = ppd[(ppd["category"] == "A") & ppd["ptype"].isin(PTYPE)].copy()
    d["la_code"] = d["district"].map(lambda x: lookup.get(norm_name(x)))
    unmatched = sorted(d.loc[d.la_code.isna(), "district"].dropna().unique())
    d = d.dropna(subset=["la_code"])
    d["property_type"] = d["ptype"].map(PTYPE)
    by_type = d.groupby(["la_code", "property_type"])["price"].agg(["median", "count"]).reset_index()
    overall = d.groupby("la_code")["price"].agg(["median", "count"]).reset_index().assign(property_type="All")
    out = pd.concat([overall, by_type], ignore_index=True).rename(
        columns={"median": "median_price", "count": "sales"})
    out.loc[out.sales < MIN_SALES, "median_price"] = None
    out["la_name"] = out.la_code.map(la_names)
    dates = pd.to_datetime(d["date"])
    meta = {"source": "HM Land Registry Price Paid Data (standard-price sales, category A)",
            "period": f"{dates.min():%Y-%m} to {dates.max():%Y-%m}" if len(d) else None,
            "sales": int(len(d)), "min_sales_for_median": MIN_SALES, "unmatched_districts": unmatched}
    return out[["la_code", "la_name", "property_type", "median_price", "sales"]], meta


def build(raw_dir: str | Path, processed_dir: str | Path, ppd_file: str | Path | None = None) -> dict:
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    src = Path(ppd_file) if ppd_file else raw_dir / "land_registry_pp_2026.csv"
    ppd = pd.read_csv(src, header=None, names=PPD_COLS,
                      usecols=["price", "date", "ptype", "district", "category"])
    official = pd.read_csv(processed_dir / "official_rents.csv")
    la = official[official.level == "la"].drop_duplicates("area_code")
    out, meta = summarise(ppd, dict(zip(la.area_code, la.area_name)))
    meta["file"] = src.name
    out.to_csv(processed_dir / "sale_prices.csv", index=False)
    log.info("sale prices built: %s rows, %s", len(out), {k: meta[k] for k in ("period", "sales")})
    if meta["unmatched_districts"]:
        log.info("districts not in the ONS rent areas (not served): %s", meta["unmatched_districts"])
    return meta


if __name__ == "__main__":  # pragma: no cover
    import json
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    m = build(config.DATA_RAW, config.DATA_PROCESSED)
    # record the vintage beside the others
    vp = config.DATA_PROCESSED / "data_vintages.json"
    v = json.loads(vp.read_text()) if vp.exists() else {}
    v["sale_prices"] = m
    vp.write_text(json.dumps(v, indent=2))
    print(m)
