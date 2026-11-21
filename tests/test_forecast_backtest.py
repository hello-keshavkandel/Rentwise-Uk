"""
Tests for src/models/forecast_backtest.py

These use a synthetic trend + seasonal + noise series so they run fast and need
no ONS data. The key property we assert is that conformal interval widths built
from backtest errors achieve close to their NOMINAL coverage on fresh data —
the whole point of the recalibration.
"""

import numpy as np
import pytest

from src.models.forecast_backtest import (
    BacktestResult,
    conformal_interval_widths,
    empirical_coverage,
    rolling_origin_backtest,
)


def _make_series(n=180, seed=0):
    """Monthly-ish series: linear trend + 12-period seasonal + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 1000 + 2.0 * t
    seasonal = 25.0 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 8.0, size=n)
    return trend + seasonal + noise


def _naive_seasonal_forecaster(history, horizon):
    """Last value carried forward with the seasonal shape of the last 12 months."""
    history = np.asarray(history, dtype=float)
    last = history[-1]
    if history.size >= 24:
        season = history[-12:] - history[-24:-12]
        reps = int(np.ceil(horizon / 12))
        shape = np.tile(season, reps)[:horizon]
    else:
        shape = np.zeros(horizon)
    return last + np.cumsum(np.where(np.arange(horizon) % 12 == 0, 0, 0)) + shape


def test_backtest_runs_and_reports_finite_metrics():
    y = _make_series()
    res = rolling_origin_backtest(
        y, _naive_seasonal_forecaster, horizon=12, n_origins=6, step=6
    )
    assert isinstance(res, BacktestResult)
    assert res.n_origins >= 3
    assert np.isfinite(res.mape) and res.mape >= 0
    assert np.isfinite(res.rmse) and res.rmse >= 0
    assert res.errors.size == res.n_origins * 12
    d = res.as_dict()
    assert d["horizon"] == 12 and d["n_points"] == res.errors.size


def test_backtest_raises_when_series_too_short():
    short = _make_series(n=20)
    with pytest.raises(ValueError):
        rolling_origin_backtest(short, _naive_seasonal_forecaster,
                                horizon=12, n_origins=6, step=3, min_train=24)


def test_conformal_widths_increase_with_level():
    rng = np.random.default_rng(1)
    errors = rng.normal(0, 10, size=500)
    widths = conformal_interval_widths(errors, levels=(0.80, 0.95))
    assert widths[0.80] > 0
    assert widths[0.95] > widths[0.80]   # higher confidence => wider band


def test_conformal_widths_achieve_nominal_coverage_out_of_sample():
    """Calibrate widths on one half of the errors, check coverage on the other."""
    rng = np.random.default_rng(7)
    # Heavy-ish tails so parametric (gaussian) bands would misfire but conformal won't.
    cal = rng.standard_t(df=5, size=4000) * 12
    test = rng.standard_t(df=5, size=4000) * 12
    widths = conformal_interval_widths(cal, levels=(0.80, 0.95))
    cov80 = empirical_coverage(test, np.zeros_like(test), widths[0.80])
    cov95 = empirical_coverage(test, np.zeros_like(test), widths[0.95])
    assert 0.76 <= cov80 <= 0.84   # within ~4pts of nominal 80%
    assert 0.92 <= cov95 <= 0.975  # within ~3pts of nominal 95%


def test_empirical_coverage_bounds():
    actual = np.array([10.0, 11.0, 9.0, 10.5])
    forecast = np.array([10.0, 10.0, 10.0, 10.0])
    assert empirical_coverage(actual, forecast, width=0.4) == 0.25   # only 10.0 within
    assert empirical_coverage(actual, forecast, width=5.0) == 1.0
