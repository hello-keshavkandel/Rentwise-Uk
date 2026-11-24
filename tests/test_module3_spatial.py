"""Module 3 spatial features + grouped evaluation (synthetic, fast)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.modules.module3_spatial import build_la_spatial_features, grouped_cv


def test_build_features_density_and_school_join():
    osm = pd.DataFrame({"lad_code": ["E1", "W1"], "pub_bar": [10, 0], "park": [0, 4]})
    schools = pd.DataFrame({"lad_code": ["E1"], "school_quality": [3.2], "share_good_or_better": [0.9]})
    out = build_la_spatial_features(osm, schools, {"E1": 10.0, "W1": 2.0}).set_index("lad_code")
    assert out.loc["E1", "dens_pub_bar"] == np.log1p(1.0)          # 10 per 10 km²
    assert out.loc["W1", "dens_park"] == np.log1p(2.0)
    assert out.loc["E1", "school_quality"] == 3.2
    assert np.isnan(out.loc["W1", "school_quality"])               # Wales: no Ofsted -> NaN


def test_grouped_cv_holds_out_whole_groups():
    """A feature that only identifies the group must NOT help when groups are held out."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 4)
    group_effect = rng.normal(0, 10, 40)[groups]
    x = rng.normal(size=len(groups))
    y = 2 * x + group_effect
    X_id = pd.DataFrame({"x": x, "gid": groups.astype(float)})     # gid is pure leakage
    lr = lambda seed: LinearRegression()                            # noqa: E731
    r = grouped_cv(X_id, y, groups, repeats=2, folds=4, model_factory=lr)
    assert r["r2"] < 0.3                                            # group effect is unpredictable out-of-group
    assert len(r["r2_runs"]) == 2
