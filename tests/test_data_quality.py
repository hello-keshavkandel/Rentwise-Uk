"""Data-quality rules on the SHIPPED processed data (what the live app serves).

These encode the Phase A correctness standard so it can never silently regress:
official rents are never trimmed or invented, crime is never a fake zero, every
input is dated, and no placeholder input exists."""
import json

import pandas as pd
import pytest

import config

P = config.DATA_PROCESSED
pytestmark = pytest.mark.skipif(not (P / "rentwise_clean.csv").exists(), reason="processed data not built")


@pytest.fixture(scope="module")
def clean():
    return pd.read_csv(P / "rentwise_clean.csv")


@pytest.fixture(scope="module")
def vintages():
    return json.loads((P / "data_vintages.json").read_text())


def test_one_official_row_per_area_and_type(clean):
    assert not clean.duplicated(["la_code", "property_type"]).any()
    assert clean.la_code.nunique() >= 316          # every LA with ONS rents + earnings
    assert {"Westminster", "Kensington and Chelsea"} <= set(clean.la_name)   # never trimmed again


def test_rents_are_plausible_and_untrimmed(clean):
    r = clean.median_monthly_rent
    assert r.between(100, 20_000).all()
    assert r.max() > 2_472                          # the old IQR cut-off no longer applies


def test_crime_is_a_rate_and_missing_is_never_zero(clean, vintages):
    c = clean.drop_duplicates("la_code").set_index("la_name").crime_per_1000
    assert (c.dropna() >= 5).all()                  # plausibility floor (real range ~39-400)
    assert (c.dropna() <= 1000).all()
    no_data = set(vintages["crime"]["areas_without_police_data"])
    assert set(c[c.isna()].index) == no_data        # every gap is declared, nothing else missing


def test_no_placeholder_inputs(clean):
    assert "transport_score" not in clean.columns
    for col in ("median_annual_income", "crime_per_1000"):
        s = clean.drop_duplicates("la_code")[col].dropna()
        assert s.nunique() > 50, f"{col} looks like a constant placeholder"


def test_every_input_is_dated(vintages):
    for k in ("rents", "earnings", "population", "crime"):
        assert vintages[k].get("period"), f"{k} has no period"
    assert vintages["rents"].get("released")


def test_served_areas_have_spatial_features_and_shapes(clean):
    sp = pd.read_csv(P / "la_spatial_features.csv").set_index("lad_code")
    geo = json.loads((P / "lad_boundaries_wgs84.geojson").read_text())
    shapes = {f["properties"]["LAD24CD"] for f in geo["features"]}
    codes = set(clean.la_code)
    assert codes <= set(sp.index), sorted(codes - set(sp.index))
    assert codes <= shapes, sorted(codes - shapes)


def test_store_matches_csv(clean):
    from src.data.store import load_table
    st = load_table("rentwise_clean")
    assert len(st) == len(clean) and set(st.la_code) == set(clean.la_code)
