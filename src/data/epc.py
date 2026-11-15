"""
epc.py  —  Energy Performance Certificate (EPC) open data for Module 1.

WHY this exists
---------------
Module 1 predicts residential sale prices from HM Land Registry Price Paid data,
which records price, postcode, property type, tenure and a new-build flag — but
NOT floor area or the number of rooms, two of the strongest drivers of price.
EPC open data (England & Wales) fills exactly that gap: every certificate carries
`total_floor_area` and `number_habitable_rooms`, keyed by address and postcode.

This module turns the raw EPC data into a clean, de-duplicated lookup table that
`src/features/property_attributes.py` can join onto Price Paid by (postcode, house
number).

DATA ACQUISITION (free, one-off)
--------------------------------
Register at https://epc.opendatacommunities.org (instant, just an email), then
EITHER:
  * Bulk  — download the domestic certificates (whole country, or per local
            authority) and unzip into `data/raw/epc/` so the tree looks like:
                data/raw/epc/domestic-E06000001-Hartlepool/certificates.csv
                data/raw/epc/domestic-E06000002-.../certificates.csv
            `load_bulk()` reads every certificates.csv it finds.
  * API   — put your credentials in `.env` (EPC_EMAIL, EPC_API_KEY) and call
            `fetch_api(postcodes)` to pull only the postcodes you need.

Either way, `build_epc_lookup()` writes a consolidated
`data/processed/epc_lookup.parquet` that the feature join consumes.
"""

from __future__ import annotations

import os
import glob
import base64
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("rentwise.epc")

# The only EPC columns Module 1 needs. Bulk files use UPPER_SNAKE, the API uses
# lower-hyphen; `_canon()` maps both onto these canonical names.
WANTED = {
    "postcode": "postcode",
    "address1": "address1",
    "address2": "address2",
    "address3": "address3",
    "total_floor_area": "total_floor_area",
    "number_habitable_rooms": "number_habitable_rooms",
    "number_heated_rooms": "number_heated_rooms",
    "property_type": "epc_property_type",
    "built_form": "built_form",
    "construction_age_band": "construction_age_band",
    "tenure": "epc_tenure",
    "inspection_date": "inspection_date",
    "lodgement_date": "lodgement_date",
}


def _canon(col: str) -> str:
    """Normalise an EPC column name (bulk UPPER_SNAKE or API lower-hyphen)."""
    return col.strip().lower().replace("-", "_").replace(" ", "_")


def _norm_postcode(s: pd.Series) -> pd.Series:
    """Uppercase, strip all spaces — a stable postcode join key (NG8 2DT -> NG82DT)."""
    return s.astype("string").str.upper().str.replace(r"\s+", "", regex=True)


def _house_number(addr: pd.Series) -> pd.Series:
    """
    Extract the primary house number from an EPC address line.

    EPC ADDRESS1 looks like '336, Trowell Road' or 'Flat 2, 14 High Street'.
    HM Land Registry stores the house number in PAON ('336'). Matching on the
    first run of digits is a simple, high-recall key; flats/named houses that
    carry no number simply won't match and are reported in the match rate.
    """
    return (addr.astype("string")
                .str.extract(r"(\d+)", expand=False)
                .str.lstrip("0"))


def _keep_wanted(df: pd.DataFrame) -> pd.DataFrame:
    """Select + rename the canonical EPC columns that are present."""
    df = df.rename(columns={c: _canon(c) for c in df.columns})
    present = {src: dst for src, dst in WANTED.items() if src in df.columns}
    out = df[list(present)].rename(columns=present)
    for c in WANTED.values():               # guarantee a stable schema
        if c not in out.columns:
            out[c] = pd.NA
    return out


# --------------------------------------------------------------------------- bulk
def load_bulk(epc_dir: str | Path) -> pd.DataFrame:
    """Read and concatenate every `certificates.csv` under `epc_dir`."""
    epc_dir = Path(epc_dir)
    files = sorted(glob.glob(str(epc_dir / "**" / "certificates.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No certificates.csv found under {epc_dir}. Download EPC bulk data "
            "from https://epc.opendatacommunities.org and unzip it there "
            "(see src/data/epc.py docstring)."
        )
    usecols = None  # bulk files are wide; we read all then trim (robust to layout changes)
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, dtype=str, low_memory=False, usecols=usecols)
            frames.append(_keep_wanted(df))
        except Exception as e:                           # noqa: BLE001
            log.warning("skipped %s: %s", fp, e)
    log.info("EPC bulk: read %d certificate file(s)", len(frames))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- api
def _auth_header(email: str, api_key: str) -> dict:
    token = base64.b64encode(f"{email}:{api_key}".encode()).decode()
    return {"Authorization": f"Basic {token}", "Accept": "text/csv"}


def fetch_api(postcodes, email=None, api_key=None, cache_csv=None) -> pd.DataFrame:
    """
    Pull certificates for a set of postcodes from the EPC API (text/csv).

    Credentials come from args or the EPC_EMAIL / EPC_API_KEY environment
    variables. Results are cached to `cache_csv` so repeat runs are free.
    """
    import io
    import time
    import requests  # local import: only needed on the API path

    email = email or os.environ.get("EPC_EMAIL")
    api_key = api_key or os.environ.get("EPC_API_KEY")
    if not (email and api_key):
        raise RuntimeError(
            "EPC API needs EPC_EMAIL and EPC_API_KEY (register free at "
            "https://epc.opendatacommunities.org). Or use the bulk download path."
        )
    base = "https://epc.opendatacommunities.org/api/v1/domestic/search"
    headers = _auth_header(email, api_key)
    out = []
    pcs = sorted({str(p).upper().strip() for p in postcodes if str(p).strip()})
    for i, pc in enumerate(pcs, 1):
        try:
            r = requests.get(base, headers=headers, params={"postcode": pc, "size": 5000}, timeout=30)
            if r.status_code == 200 and r.text.strip():
                out.append(pd.read_csv(io.StringIO(r.text), dtype=str))
            elif r.status_code == 429:                   # rate limited — back off
                time.sleep(2)
        except Exception as e:                           # noqa: BLE001
            log.warning("EPC API %s: %s", pc, e)
        if i % 200 == 0:
            log.info("EPC API: %d/%d postcodes", i, len(pcs))
    df = pd.concat([_keep_wanted(x) for x in out], ignore_index=True) if out else pd.DataFrame()
    if cache_csv and not df.empty:
        df.to_csv(cache_csv, index=False)
    return df


# ------------------------------------------------------------------ consolidate
def consolidate(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw EPC rows into one best record per (postcode, house number).

    * numeric floor area / room counts, with implausible values dropped,
    * most RECENT certificate kept per property (EPCs are re-lodged over time),
    * a compact join key (`pc_key`, `house_no`).
    """
    df = raw.copy()
    df["total_floor_area"] = pd.to_numeric(df["total_floor_area"], errors="coerce")
    for c in ("number_habitable_rooms", "number_heated_rooms"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # plausibility guards (EPC has data-entry noise)
    df.loc[(df["total_floor_area"] < 10) | (df["total_floor_area"] > 2000), "total_floor_area"] = np.nan
    df.loc[(df["number_habitable_rooms"] < 1) | (df["number_habitable_rooms"] > 20), "number_habitable_rooms"] = np.nan

    df["pc_key"] = _norm_postcode(df["postcode"])
    df["house_no"] = _house_number(df["address1"])
    date = pd.to_datetime(df.get("lodgement_date"), errors="coerce").fillna(
        pd.to_datetime(df.get("inspection_date"), errors="coerce"))
    df["_date"] = date

    df = df.dropna(subset=["pc_key", "house_no", "total_floor_area"])
    df = (df.sort_values("_date")
            .drop_duplicates(subset=["pc_key", "house_no"], keep="last"))

    keep = ["pc_key", "house_no", "total_floor_area", "number_habitable_rooms",
            "number_heated_rooms", "built_form", "construction_age_band"]
    out = df[keep].reset_index(drop=True)
    log.info("EPC consolidated: %d unique properties", len(out))
    return out


def build_epc_lookup(epc_raw_dir: str | Path, out_parquet: str | Path) -> pd.DataFrame:
    """End-to-end: read bulk EPC -> consolidate -> write parquet lookup."""
    lookup = consolidate(load_bulk(epc_raw_dir))
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    try:
        lookup.to_parquet(out_parquet, index=False)
    except Exception:                                    # parquet engine missing -> csv fallback
        out_parquet = out_parquet.with_suffix(".csv")
        lookup.to_csv(out_parquet, index=False)
    log.info("EPC lookup written -> %s (%d rows)", out_parquet, len(lookup))
    return lookup


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    raw_dir = config.DATA_RAW / "epc"
    out = config.DATA_PROCESSED / "epc_lookup.parquet"
    build_epc_lookup(raw_dir, out)
