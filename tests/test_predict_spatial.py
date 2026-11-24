"""Deployed predictor with the spatial model (skips if artefacts absent)."""
import math

import pytest

import config

pytestmark = pytest.mark.skipif(
    not (config.MODELS_DIR / "rent_model_meta.json").exists()
    or not (config.DATA_PROCESSED / "la_spatial_features.csv").exists(),
    reason="spatial model artefacts not built")


@pytest.fixture(scope="module")
def predictor():
    from src.models.predict import RentWisePredictor
    return RentWisePredictor()


def test_uses_spatial_model(predictor):
    info = predictor.model_info()
    assert info["model"] == "xgboost-spatial" and info["spatial"] and info["trained_at"]
    assert predictor.spatial_cols and set(predictor.spatial_cols) <= set(predictor.feature_cols)


def test_every_served_area_predicts_sanely(predictor):
    for la in predictor.area_names():
        for ptype in predictor.property_types:
            r = predictor.predict(la, ptype)
            assert 250 <= r["rent"] <= 10_000, (la, ptype, r["rent"])
            assert r["interval"]["low"] <= r["rent"] <= r["interval"]["high"]
            assert math.isfinite(r["ratio"]) and r["ratio"] > 0


def test_welsh_school_gap_is_labelled_missing(predictor):
    r = predictor.predict("Cardiff", "Terraced")
    feats = [f for f, _ in r["contributions"]]
    assert not any(f == "school_quality" for f in feats)
    assert predictor.spatial_profile("Cardiff")["school_quality"] is None


def test_unknown_area_raises(predictor):
    from src.models.predict import UnknownAreaError
    with pytest.raises(UnknownAreaError):
        predictor.predict("Atlantis", "Flat")
    with pytest.raises(ValueError):
        predictor.predict("Cardiff", "Castle")


def test_area_by_code_roundtrip(predictor):
    row = predictor.area_by_code("E07000223")
    assert row is not None and row["la_name"] == "Adur"
    assert predictor.area_by_code("E99999999") is None
