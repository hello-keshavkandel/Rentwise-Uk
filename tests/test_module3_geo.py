"""Tests for Module 3 geospatial feature engineering (src/modules/module3_geo.py)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modules import module3_geo as g3


def test_aggregate_points():
    pts = pd.DataFrame({"lad_code": ["A", "A", "A", "B"],
                        "category": ["transport", "transport", "school", "transport"]})
    out = g3.aggregate_points(pts).set_index("lad_code")
    assert out.loc["A", "n_transport"] == 2
    assert out.loc["A", "n_school"] == 1
    assert out.loc["B", "n_transport"] == 1


def test_school_quality():
    sc = pd.DataFrame({"lad_code": ["A", "A", "B"],
                       "ofsted_rating": ["Outstanding", "Good", "Inadequate"]})
    out = g3.school_quality(sc).set_index("lad_code")
    assert out.loc["A", "school_quality"] == 3.5     # (4+3)/2
    assert out.loc["B", "school_quality"] == 1.0


def test_build_fills_missing_and_pca_orients():
    base = pd.DataFrame({"lad_code": ["A", "B", "C"]})
    amen = pd.DataFrame({"lad_code": ["A", "B"], "n_transport": [8, 1], "n_school": [4, 1]})
    feat = g3.build_spatial_features(base, amen)
    assert set(feat["lad_code"]) == {"A", "B", "C"}
    assert feat.loc[feat.lad_code == "C", "n_transport"].iloc[0] == 0   # filled with 0
    out = g3.pca_composite(feat, ["n_transport", "n_school"])
    assert "spatial_index" in out.columns and len(out) == 3
    idx = out.set_index("lad_code")["spatial_index"]
    assert idx["A"] > idx["B"]                        # more amenities -> higher index
