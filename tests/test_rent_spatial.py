"""Spatial rent model: frame alignment, evaluation/calibration, portable artefacts."""
import json

import numpy as np
import pandas as pd
import pytest

from src.models.rent_spatial import TARGET, evaluate_and_calibrate, train_and_save, training_frame


def _synthetic(n_la=40, seed=0):
    rng = np.random.default_rng(seed)
    las = [f"E{i:08d}" for i in range(n_la)]
    clean = pd.DataFrame({"la_code": np.repeat(las, 4)})
    amen = rng.normal(size=n_la)
    feats = pd.DataFrame({"bedrooms": np.tile([1, 2, 3, 4], n_la)})
    feats[TARGET] = 600 + 150 * feats["bedrooms"] + 200 * np.repeat(amen, 4) + rng.normal(0, 30, len(feats))
    spatial = pd.DataFrame({"lad_code": las, "area_km2": 1.0, "dens_pub_bar": amen,
                            "school_quality": rng.normal(3, 0.2, n_la), "share_good_or_better": 0.9})
    spatial.loc[0, ["school_quality", "share_good_or_better"]] = np.nan     # a 'Welsh' LA
    return clean, feats, spatial


def test_training_frame_aligns_and_keeps_nan_schools():
    clean, feats, spatial = _synthetic()
    X, y, g = training_frame(clean, feats, spatial)
    assert list(X.columns) == ["bedrooms", "dens_pub_bar", "school_quality", "share_good_or_better"]
    assert len(X) == len(y) == len(g) == 160
    assert X["school_quality"].isna().sum() == 4                 # NaN kept for XGBoost, not imputed


def test_training_frame_rejects_unknown_la():
    clean, feats, spatial = _synthetic()
    with pytest.raises(ValueError):
        training_frame(clean, feats, spatial[spatial.lad_code != "E00000005"])


def test_calibration_hits_nominal_coverage_on_random_cv():
    clean, feats, spatial = _synthetic(n_la=60)
    ev = evaluate_and_calibrate(*training_frame(clean, feats, spatial))
    assert 0.76 <= ev["coverage_random_cv"]["0.2"] <= 0.84
    assert ev["conformal_q"]["0.1"] > ev["conformal_q"]["0.2"] > 0
    assert ev["random_cv"]["r2"] > 0.8


def test_train_and_save_writes_portable_artefacts(tmp_path):
    from xgboost import XGBRegressor
    clean, feats, spatial = _synthetic()
    meta = train_and_save(clean, feats, spatial, tmp_path)
    m = XGBRegressor(); m.load_model(str(tmp_path / meta["model_file"]))
    saved = json.loads((tmp_path / "rent_model_meta.json").read_text())
    assert saved["feature_cols"] == meta["feature_cols"] and "trained_at" in saved
    conf = json.loads((tmp_path / "conformal.json").read_text())
    assert set(conf["alpha_to_q"]) == {"0.1", "0.2"}
    X, _, _ = training_frame(clean, feats, spatial)
    assert m.predict(X.to_numpy(dtype=float)[:3]).shape == (3,)
