"""
Unit tests for the Module 1 EPC enrichment (src/features/property_attributes.py).

These use tiny in-memory frames — no downloads — so the join logic, imputation
and feature set are verified deterministically in CI.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import property_attributes as pa


@pytest.fixture
def price_paid():
    # four sales; two will match EPC on (postcode, house number)
    return pd.DataFrame({
        "price": [200_000, 350_000, 275_000, 500_000],
        "ptype": ["Terraced", "Detached", "Flat", "Detached"],
        "is_new": [0, 0, 0, 1], "is_freehold": [1, 1, 0, 1],
        "district": ["NOTTINGHAM", "RUSHCLIFFE", "NOTTINGHAM", "RUSHCLIFFE"],
        "pc_key": ["NG82DT", "NG123UJ", "NG82DT", "NG123UJ"],
        "house_no": ["336", "153", "10", "153"],
    })


@pytest.fixture
def epc_lookup():
    return pd.DataFrame({
        "pc_key": ["NG82DT", "NG123UJ"],
        "house_no": ["336", "153"],
        "total_floor_area": [78.0, 142.0],
        "number_habitable_rooms": [4, 6],
    })


def test_match_rate_and_flag(price_paid, epc_lookup):
    out = pa.add_epc_features(price_paid, epc_lookup)
    # rows 0 and 1/3 share keys with EPC; row at (NG123UJ,153) appears twice -> 3 matches
    assert out["has_epc"].sum() == 3
    assert out.attrs["epc_match_rate"] == pytest.approx(0.75)


def test_no_missing_after_impute(price_paid, epc_lookup):
    out = pa.add_epc_features(price_paid, epc_lookup)
    assert out["total_floor_area"].notna().all()
    assert out["number_habitable_rooms"].notna().all()
    assert (out["floor_area_per_room"] > 0).all()


def test_feature_columns_toggle():
    assert "total_floor_area" not in pa.feature_columns(use_epc=False)
    assert "total_floor_area" in pa.feature_columns(use_epc=True)
    assert "has_epc" in pa.feature_columns(use_epc=True)


def test_encode_is_train_only(price_paid):
    df = pa.encode(price_paid, train_idx=[0, 1], seed=1)
    assert "district_med" in df.columns
    assert {"pt_Detached", "pt_Flat", "pt_Terraced", "pt_Semi-detached"} <= set(df.columns)
