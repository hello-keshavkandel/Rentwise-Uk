"""
test_data_contract.py  —  I check that the generated dataset obeys data_contract.md.

These are the unit tests referenced in the Testing chapter: they fail loudly if a
data exchange breaks the shared contract, which is what lets the four modules be
developed independently and still integrate.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from src.data.make_sample_data import SampleDataGenerator

REQUIRED = ["la_code", "la_name", "region", "year", "property_type", "bedrooms",
            "median_annual_income", "crime_per_1000", "transport_score",
            "median_monthly_rent"]
REGIONS = {"North East", "North West", "Yorkshire and The Humber", "East Midlands",
           "West Midlands", "East of England", "London", "South East", "South West",
           "Wales"}


@pytest.fixture(scope="module")
def df():
    # I generate a fresh dataset so the tests never depend on stale files.
    return SampleDataGenerator().generate()


def test_required_columns_present(df):
    assert list(df.columns) == REQUIRED


def test_no_nulls_in_key_or_target(df):
    for col in ["la_code", "property_type", "year", "median_monthly_rent"]:
        assert df[col].notna().all(), f"nulls found in {col}"


def test_primary_key_unique(df):
    key = df[["la_code", "property_type", "year"]]
    assert not key.duplicated().any(), "primary key (la_code, property_type, year) not unique"


def test_regions_are_valid(df):
    assert set(df["region"].unique()).issubset(REGIONS)


def test_value_ranges(df):
    assert df["median_monthly_rent"].min() >= 250
    assert df["bedrooms"].between(1, 6).all()
    assert df["transport_score"].between(0, 1).all()
    assert df["median_annual_income"].between(20000, 80000).all()
    assert df["year"].between(2022, 2024).all()


def test_dataset_is_reasonably_sized(df):
    # I expect the broad development set (~117 LAs x 5 types x 3 years).
    assert len(df) >= 1500
    assert df["la_name"].nunique() >= 100
