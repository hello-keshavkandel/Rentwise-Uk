"""
property_attributes.py  —  Module 1 features: Price Paid + EPC floor area/rooms.

I load HM Land Registry Price Paid, clean it the same way the dissertation build
did, then LEFT-JOIN the EPC lookup (src/data/epc.py) by (postcode, house number)
to add `total_floor_area` and `number_habitable_rooms`. Unmatched sales keep a
sensible imputed value plus a `has_epc` flag, so the model can still use them and
learn how much the real attribute adds.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("rentwise.module1.features")

# HM Land Registry Price Paid is a 16-column CSV with NO header.
PP_COLS = ["tuid", "price", "date", "postcode", "ptype", "is_new_raw", "duration",
           "paon", "saon", "street", "locality", "town", "district", "county",
           "ppd_cat", "status"]
PT_MAP = {"D": "Detached", "S": "Semi-detached", "T": "Terraced", "F": "Flat"}


def _norm_postcode(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.replace(r"\s+", "", regex=True)


def _house_number(paon: pd.Series) -> pd.Series:
    """First run of digits in PAON ('336', 'FLAT 2, 14' -> '2'? no -> '14')."""
    return paon.astype("string").str.extract(r"(\d+)", expand=False).str.lstrip("0")


def load_price_paid(path: str | Path,
                    price_min: int = 20_000, price_max: int = 2_000_000) -> pd.DataFrame:
    """Parse + clean Price Paid into a tidy DataFrame (standard residential, cat A)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if len(r) < 16 or r[14] != "A" or r[15] != "A" or r[4] not in PT_MAP:
                continue
            try:
                price = int(r[1])
            except ValueError:
                continue
            if not (price_min <= price <= price_max):
                continue
            rows.append((price, PT_MAP[r[4]], 1 if r[5] == "Y" else 0,
                         1 if r[6] == "F" else 0, (r[12] or "").upper(),
                         r[3], r[7]))
    df = pd.DataFrame(rows, columns=["price", "ptype", "is_new", "is_freehold",
                                     "district", "postcode", "paon"])
    df["pc_key"] = _norm_postcode(df["postcode"])
    df["house_no"] = _house_number(df["paon"])
    log.info("Price Paid: %d cleaned standard residential sales", len(df))
    return df


def add_epc_features(pp: pd.DataFrame, epc_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join EPC attributes onto Price Paid and engineer Module 1 features.

    Adds: total_floor_area, number_habitable_rooms, floor_area_per_room,
    has_epc (1/0). Missing floor area/rooms are imputed by the
    (property type x district) median, then the national median — so every row
    is usable and the `has_epc` flag lets the model separate measured from imputed.
    """
    epc = epc_lookup.copy()
    epc["house_no"] = epc["house_no"].astype("string")
    pp = pp.copy()
    pp["house_no"] = pp["house_no"].astype("string")

    merged = pp.merge(
        epc[["pc_key", "house_no", "total_floor_area", "number_habitable_rooms"]],
        on=["pc_key", "house_no"], how="left", validate="m:1",
    )
    merged["has_epc"] = merged["total_floor_area"].notna().astype(int)
    match_rate = merged["has_epc"].mean()
    log.info("EPC match rate: %.1f%% of sales (%d of %d)",
             100 * match_rate, int(merged["has_epc"].sum()), len(merged))

    # impute floor area / rooms: (ptype x district) median -> ptype median -> global
    for col in ("total_floor_area", "number_habitable_rooms"):
        g1 = merged.groupby(["ptype", "district"])[col].transform("median")
        g2 = merged.groupby("ptype")[col].transform("median")
        merged[col] = merged[col].fillna(g1).fillna(g2).fillna(merged[col].median())

    merged["floor_area_per_room"] = (
        merged["total_floor_area"] / merged["number_habitable_rooms"].clip(lower=1))

    merged.attrs["epc_match_rate"] = float(match_rate)
    return merged


def feature_columns(use_epc: bool) -> list[str]:
    """The model feature set, with or without EPC attributes (for A/B comparison)."""
    base = ["is_new", "is_freehold", "district_med",
            "pt_Detached", "pt_Semi-detached", "pt_Terraced", "pt_Flat"]
    if use_epc:
        base += ["total_floor_area", "number_habitable_rooms",
                 "floor_area_per_room", "has_epc"]
    return base


def encode(df: pd.DataFrame, train_idx, seed: int = 42) -> pd.DataFrame:
    """District median-price target encoding (train-only) + one-hot property type."""
    enc = df.loc[train_idx].groupby("district")["price"].median()
    glob = df.loc[train_idx, "price"].median()
    df = df.copy()
    df["district_med"] = df["district"].map(enc).fillna(glob)
    for p in PT_MAP.values():
        df[f"pt_{p}"] = (df["ptype"] == p).astype(int)
    return df
