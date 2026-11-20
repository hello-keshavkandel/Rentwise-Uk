"""Tests for the choropleth renderer (src/viz/maps.py)."""
import sys
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip("folium")
from src.viz.maps import choropleth


def _geojson(tmp_path):
    g = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"LAD24CD": "E1"},
         "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}},
        {"type": "Feature", "properties": {"LAD24CD": "E2"},
         "geometry": {"type": "Polygon", "coordinates": [[[1, 0], [1, 1], [2, 1], [2, 0], [1, 0]]]}},
    ]}
    p = tmp_path / "boundaries.geojson"
    p.write_text(json.dumps(g))
    return p


def test_choropleth_saves_html(tmp_path):
    gj = _geojson(tmp_path)
    out = tmp_path / "map.html"
    choropleth({"E1": 0.44, "E2": 0.31}, gj, value_name="Rent-to-income", out_html=out)
    assert out.exists() and out.stat().st_size > 1000
    assert "leaflet" in out.read_text().lower()      # a real folium/Leaflet map


def test_choropleth_accepts_series(tmp_path):
    import pandas as pd
    gj = _geojson(tmp_path)
    s = pd.Series({"E1": 1.0, "E2": 2.0})
    m = choropleth(s, gj, value_name="idx")
    assert m is not None
