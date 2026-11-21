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


# ---- store-first reads with CSV fallback (the app's read path) ----------------

def test_load_table_prefers_store(tmp_path):
    proc = _tiny_processed(tmp_path)
    db = proc / "rentwise.duckdb"
    st.build_store(processed_dir=proc, db_path=db)
    # mutate the CSV after building: a store read must NOT see this change
    (proc / "rentwise_clean.csv").write_text("region,median_monthly_rent\nX,1\n")
    df = st.load_table("rentwise_clean", db_path=db)
    assert len(df) > 1 and "X" not in set(df["region"])
    assert st.source_of("rentwise_clean", db_path=db) == "duckdb"


def test_load_table_falls_back_to_csv_when_store_missing(tmp_path):
    proc = _tiny_processed(tmp_path)
    db = proc / "rentwise.duckdb"            # never built
    df = st.load_table("rentwise_clean", db_path=db)
    assert len(df) == len(pd.read_csv(proc / "rentwise_clean.csv"))
    assert st.source_of("rentwise_clean", db_path=db) == "csv"


def test_load_table_falls_back_when_table_missing(tmp_path):
    proc = _tiny_processed(tmp_path)
    db = proc / "rentwise.duckdb"
    con = st.connect(db); con.execute("CREATE TABLE other AS SELECT 1 x"); con.close()
    df = st.load_table("rentwise_clean", db_path=db)
    assert "region" in df.columns


def test_load_table_survives_corrupt_store(tmp_path):
    proc = _tiny_processed(tmp_path)
    db = proc / "rentwise.duckdb"
    db.write_bytes(b"not a duckdb file")
    df = st.load_table("rentwise_clean", db_path=db)
    assert "region" in df.columns
    assert st.source_of("rentwise_clean", db_path=db) == "csv"


def test_load_table_unknown_name_raises(tmp_path):
    with pytest.raises(KeyError):
        st.load_table("nope", db_path=tmp_path / "missing.duckdb")


def test_committed_store_matches_processed_csvs():
    """The committed store must be in sync with the committed CSVs (guards drift)."""
    import numpy as np
    import config
    db = config.DATA_PROCESSED / "rentwise.duckdb"
    if not db.exists():
        pytest.skip("store not present")
    for t in ("rentwise_features", "rentwise_clean"):
        csv = config.DATA_PROCESSED / f"{t}.csv"
        if not csv.exists():
            continue
        a, b = st.load_table(t, db_path=db), pd.read_csv(csv)
        assert list(a.columns) == list(b.columns) and a.shape == b.shape, t
        for c in b.columns:
            if pd.api.types.is_numeric_dtype(b[c]):
                assert np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True), (t, c)
            else:
                assert (a[c].astype(str) == b[c].astype(str)).all(), (t, c)
