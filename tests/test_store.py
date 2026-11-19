"""
Tests for the DuckDB processed-data store (src/data/store.py).

Build a store in a tmp directory from tiny CSVs, then query it — no reliance on the
real data, so this runs anywhere.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip("duckdb")
from src.data import store as st


def _tiny_processed(tmp_path):
    (tmp_path / "rentwise_clean.csv").write_text(
        "la_name,region,median_monthly_rent\nAdur,South East,1100\nLeeds,Yorkshire and The Humber,850\n")
    (tmp_path / "affordability.csv").write_text(
        "la_name,ratio\nAdur,0.44\nLeeds,0.31\n")
    return tmp_path


def test_build_and_info(tmp_path):
    proc = _tiny_processed(tmp_path)
    db = tmp_path / "t.duckdb"
    counts = st.build_store(processed_dir=proc, db_path=db)
    assert counts["rentwise_clean"] == 2
    assert counts["affordability"] == 2
    assert st.info(db)["rentwise_clean"] == 2


def test_query_roundtrip(tmp_path):
    proc = _tiny_processed(tmp_path)
    db = tmp_path / "t.duckdb"
    st.build_store(processed_dir=proc, db_path=db)
    df = st.query("SELECT region, max(median_monthly_rent) m FROM rentwise_clean GROUP BY 1 ORDER BY m DESC", db_path=db)
    assert list(df["m"]) == [1100, 850]
    assert isinstance(st.table("affordability", db_path=db), pd.DataFrame)


def test_missing_csv_is_skipped(tmp_path):
    (tmp_path / "rentwise_clean.csv").write_text("la_name,region,median_monthly_rent\nAdur,South East,1100\n")
    db = tmp_path / "t.duckdb"
    counts = st.build_store(processed_dir=tmp_path, db_path=db)   # only one CSV present
    assert "rentwise_clean" in counts and "affordability" not in counts
