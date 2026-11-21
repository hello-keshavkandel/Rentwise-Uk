"""
forecast_backtest.py — honest evaluation + calibrated intervals for Module 4.

WHY this exists:
The original Module 4 judged its forecasts on a SINGLE held-out window (the last
12 months) and drew its 80%/95% bands from ARIMA's own parametric formula. Both
are fragile: one window is a sample of size one, and the parametric bands assume
the model is correctly specified, so they tend to be over-confident on real rent
series.

This module fixes both with two standard, model-agnostic tools:

1. rolling_origin_backtest() — refit at several expanding origins and forecast
   forward each time. Averaging the error over many origins is a far more stable
   estimate of real forecast skill than a single split.

2. conformal_interval_widths() — take the backtest's own forecast errors and use
   their empirical quantiles as the half-width of the prediction interval. This
   is distribution-free: an 80% interval built this way covers ~80% by
   construction, whatever the model's parametric assumptions.

Everything works on a plain 1-D sequence and takes the forecaster as a callable,
so it is unit-tested on synthetic series with no heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

# A forecaster maps (history, horizon) -> array of `horizon` point forecasts.
ForecastFn = Callable[[np.ndarray, int], np.ndarray]


def default_arima_forecaster(order=(1, 1, 1)) -> ForecastFn:
    """ARIMA(order) point forecaster, matching Module 4's live model.

    Imported lazily so that importing this module (and its tests) does not drag
    in statsmodels unless an ARIMA backtest is actually requested.
    """

    def _fn(history: np.ndarray, horizon: int) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA

        fit = ARIMA(np.asarray(history, dtype=float), order=order).fit()
        return np.asarray(fit.forecast(horizon), dtype=float)

    return _fn


@dataclass
class BacktestResult:
    """Outcome of a rolling-origin backtest."""

    horizon: int
    n_origins: int
    mape: float                       # mean abs % error, pooled over all origins
    rmse: float                       # pooled root mean squared error
    errors: np.ndarray = field(repr=False)        # signed errors (actual-forecast)
    abs_pct_errors: np.ndarray = field(repr=False)  # |error|/|actual|, per point

    def as_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "n_origins": self.n_origins,
            "mape": round(self.mape, 4),
            "rmse": round(self.rmse, 4),
            "n_points": int(self.errors.size),
        }


def rolling_origin_backtest(
    series: Sequence[float],
    forecast_fn: ForecastFn,
    horizon: int = 12,
    n_origins: int = 6,
    step: int = 3,
    min_train: int | None = None,
) -> BacktestResult:
    """Expanding-window rolling-origin backtest.

    We place `n_origins` cutoffs near the end of the series, `step` apart. At each
    cutoff we fit on everything up to it and forecast the next `horizon` points,
    then compare to the real values. Errors from every origin are pooled.

    Parameters
    ----------
    series       : the full 1-D series (chronological).
    forecast_fn  : (history, horizon) -> point forecasts.
    horizon      : steps to forecast at each origin.
    n_origins    : how many cutoffs to evaluate.
    step         : spacing between cutoffs (in time steps).
    min_train    : minimum training length for the earliest origin; defaults to
                   half the series, floored at 24.

    Returns
    -------
    BacktestResult with pooled MAPE / RMSE and the raw error arrays (the latter
    feed conformal_interval_widths()).
    """
    y = np.asarray(series, dtype=float)
    n = y.size
    if min_train is None:
        min_train = max(24, n // 2)

    # The latest origin leaves exactly `horizon` points to score; earlier origins
    # step back from there. Keep only origins that still have `min_train` history.
    latest = n - horizon
    origins = [latest - k * step for k in range(n_origins)]
    origins = sorted(o for o in origins if o >= min_train)
    if not origins:
        raise ValueError(
            f"series too short: n={n}, horizon={horizon}, min_train={min_train} "
            "leaves no valid rolling origin. Lower horizon/min_train/n_origins."
        )

    signed, abs_pct = [], []
    used = 0
    for cut in origins:
        train = y[:cut]
        actual = y[cut:cut + horizon]
        if actual.size < horizon:
            continue
        pred = np.asarray(forecast_fn(train, horizon), dtype=float)[:horizon]
        err = actual - pred
        signed.append(err)
        denom = np.where(np.abs(actual) < 1e-9, np.nan, np.abs(actual))
        abs_pct.append(np.abs(err) / denom)
        used += 1

    signed = np.concatenate(signed)
    abs_pct = np.concatenate(abs_pct)
    mape = float(np.nanmean(abs_pct) * 100.0)
    rmse = float(np.sqrt(np.mean(signed ** 2)))
    return BacktestResult(
        horizon=horizon, n_origins=used, mape=mape, rmse=rmse,
        errors=signed, abs_pct_errors=abs_pct,
    )


def conformal_interval_widths(
    errors: Sequence[float],
    levels: Sequence[float] = (0.80, 0.95),
) -> dict[float, float]:
    """Distribution-free interval half-widths from backtest errors.

    For a nominal level L we take the finite-sample-corrected empirical quantile
    of |error| at probability L. The symmetric band (forecast ± width) then
    covers about L of future points, regardless of the model's parametric
    assumptions.

    The correction uses the (n+1) quantile position standard in conformal
    prediction, clipped to 1.0 so small samples simply take the max |error|.
    """
    e = np.abs(np.asarray(errors, dtype=float))
    e = e[~np.isnan(e)]
    n = e.size
    if n == 0:
        raise ValueError("no errors provided for conformal width calibration")
    out: dict[float, float] = {}
    for L in levels:
        q = min(1.0, np.ceil((n + 1) * L) / n)   # finite-sample conformal quantile
        out[float(L)] = float(np.quantile(e, q))
    return out


def empirical_coverage(
    actual: Sequence[float],
    forecast: Sequence[float],
    width: float,
) -> float:
    """Fraction of `actual` falling inside forecast ± width."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    inside = np.abs(a - f) <= width
    return float(np.mean(inside))
