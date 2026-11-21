"""
Tests for the data-acquisition registry (src/data/sources.py).

Validators run against the REAL raw files when present (skipped otherwise). The
network fetch is tested with a mocked `requests.get`, so no download happens.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from src.data import sources as S


def test_registry_shape():
    assert len(S.SOURCES) == 9                              # 6 required + 3 optional (Module 3)
    assert len([s for s in S.SOURCES if not s.optional]) == 6
    assert S.BY_KEY["land_registry"].url is not None       # has a real URL
    assert S.BY_KEY["pipr_monthly"].url is None            # manual (ONS URL changes)
    assert S.BY_KEY["osm_amenities"].optional is True      # geospatial enrichment is optional


def test_manifest_on_real_files():
    if not (config.DATA_RAW / "land_registry_pp_2026.csv").exists():
        pytest.skip("raw data not present")
    m = S.build_manifest(config.DATA_RAW)
    assert m["all_ok"] is True
    assert m["sources"]["lsoa_to_lad"]["ok"] is True       # BOM-safe validator
    assert m["sources"]["land_registry"]["rows"] > 1000
    assert (config.DATA_RAW / "_manifest.json").exists()


def test_fetch_is_mocked(tmp_path, monkeypatch):
    import requests

    class _Resp:
        content = b"col1,col2\n1,2\n"
        def raise_for_status(self): pass

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    src = S.Source("x", "X", "x.csv", S.v_lookup, url="http://example.test/x.csv")
    rec = src.fetch(tmp_path, force=True)
    assert rec["status"] == "downloaded"
    assert (tmp_path / "x.csv").read_bytes() == _Resp.content
    # second call without force -> cached, no download
    assert src.fetch(tmp_path)["status"] == "cached"


def test_manual_source_not_fetched(tmp_path):
    src = S.Source("m", "Manual", "m.xlsx", S.v_ashe, url=None, note="download by hand")
    assert src.fetch(tmp_path)["status"] == "manual"


def test_validator_flags_bad_file(tmp_path):
    bad = tmp_path / "land_registry_pp_2026.csv"
    bad.write_text("only,three,cols\n1,2,3\n")               # <16 cols -> not ok
    src = S.BY_KEY["land_registry"]
    assert src.check(tmp_path)["ok"] is False
