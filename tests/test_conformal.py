"""
Tests for calibrated prediction intervals (src/models/conformal.py).

Synthetic, fast, deterministic — verifies the conformal widths order correctly and
that empirical coverage lands near the nominal level.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import conformal as cf


def _data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=1.0, size=n)
    return X, y


def _lin():
    from sklearn.linear_model import LinearRegression
    return LinearRegression()


def test_widths_order_by_alpha():
    X, y = _data()
    q = cf.cross_conformal_q(_lin, X, y, alphas=(0.1, 0.2))
    assert q["0.1"] > q["0.2"] > 0          # 90% interval wider than 80%


def test_coverage_near_nominal():
    X, y = _data()
    cov = cf.empirical_coverage(_lin, X, y, alphas=(0.1, 0.2))
    assert 0.82 <= cov["0.1"] <= 0.97       # ~90%
    assert 0.70 <= cov["0.2"] <= 0.90       # ~80%


def test_calibrate_writes_and_loads(tmp_path):
    X, y = _data()
    out = tmp_path / "conformal.json"
    payload = cf.calibrate(_lin, X, y, out, target="rent")
    assert out.exists() and payload["target"] == "rent"
    q80 = cf.load_q(out, alpha=0.2)
    assert q80 == pytest.approx(payload["alpha_to_q"]["0.2"])
    # nearest-alpha fallback
    assert cf.load_q(out, alpha=0.15) in (payload["alpha_to_q"]["0.1"], payload["alpha_to_q"]["0.2"])


def test_load_q_missing_returns_none(tmp_path):
    assert cf.load_q(tmp_path / "nope.json") is None
