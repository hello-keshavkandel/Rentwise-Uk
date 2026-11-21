"""
conformal.py  —  calibrated prediction intervals for the rent model (Module 2).

A point estimate ("£1,100/month") hides how sure the model is. Cross-conformal
prediction turns the model's own errors into an honest interval with a guaranteed
long-run coverage (an 80% interval contains the true rent ~80% of the time), with
NO distributional assumptions.

K-fold cross-conformal: every row gets an out-of-fold residual (no train-on-test
leakage); the finite-sample-corrected quantile of |residual| is the interval
half-width `q`, saved to `models/conformal.json` and applied symmetrically at
prediction time.

The calibration uses a fresh model of the SAME family/params as the deployed one
(conformal guarantees are about the procedure, not a single fitted artifact) — this
also sidesteps cloning a model pickled by an older library version.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

log = logging.getLogger("rentwise.conformal")

DEFAULT_ALPHAS = (0.1, 0.2)   # -> 90% and 80% intervals


def default_factory():
    """A fresh XGBRegressor matching the deployed Module 2 model."""
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                        random_state=42, verbosity=0, n_jobs=4)


def _oof_residuals(make_model, X, y, k, seed):
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    resid = np.empty(len(y)); pred = np.empty(len(y))
    for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(X):
        m = make_model(); m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te]); resid[te] = np.abs(y[te] - pred[te])
    return resid, pred


def cross_conformal_q(make_model, X, y, alphas=DEFAULT_ALPHAS, k=5, seed=42) -> dict:
    """Out-of-fold |residual| quantiles (finite-sample corrected) per alpha."""
    resid, _ = _oof_residuals(make_model, X, y, k, seed)
    n = len(resid)
    out = {}
    for a in alphas:
        level = min(np.ceil((n + 1) * (1 - a)) / n, 1.0)   # conformal correction
        out[str(a)] = float(np.quantile(resid, level))
    return out


def empirical_coverage(make_model, X, y, alphas=DEFAULT_ALPHAS, k=5, seed=42) -> dict:
    """Out-of-fold coverage — the fraction of truths actually inside each interval."""
    resid, pred = _oof_residuals(make_model, X, y, k, seed)
    y = np.asarray(y, dtype=float); n = len(resid)
    cov = {}
    for a in alphas:
        level = min(np.ceil((n + 1) * (1 - a)) / n, 1.0)
        q = float(np.quantile(resid, level))
        cov[str(a)] = float(((y >= pred - q) & (y <= pred + q)).mean())
    return cov


def calibrate(make_model, X, y, out_json: Path, target: str = "median_monthly_rent",
              alphas=DEFAULT_ALPHAS, seed=42) -> dict:
    qmap = cross_conformal_q(make_model, X, y, alphas=alphas, seed=seed)
    payload = {"target": target, "n": int(len(y)), "alpha_to_q": qmap,
               "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out_json = Path(out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    log.info("conformal widths -> %s  %s", out_json, qmap)
    return payload


def load_q(conformal_json: Path, alpha: float = 0.2) -> float | None:
    """Interval half-width for `alpha` (default 80%), or None if unavailable."""
    p = Path(conformal_json)
    if not p.exists():
        return None
    qmap = json.loads(p.read_text()).get("alpha_to_q", {})
    if not qmap:
        return None
    if str(alpha) in qmap:
        return float(qmap[str(alpha)])
    return float(qmap[min(qmap, key=lambda s: abs(float(s) - alpha))])


def calibrate_deployed(features_csv: Path, out_json: Path, make_model=None,
                       target: str = "median_monthly_rent") -> dict:
    """Calibrate against the processed feature table using a fresh model."""
    feats = pd.read_csv(features_csv)
    X = feats[[c for c in feats.columns if c != target]]
    y = feats[target]
    return calibrate(make_model or default_factory, X, y, out_json, target=target)


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    payload = calibrate_deployed(config.DATA_PROCESSED / "rentwise_features.csv",
                                 config.MODELS_DIR / "conformal.json")
    print("calibrated:", payload["alpha_to_q"])
