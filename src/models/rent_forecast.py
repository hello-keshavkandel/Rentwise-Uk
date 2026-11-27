"""
rent_forecast.py  —  12-month rent forecasts the app can show, with honest bands (Phase C).

Module 4 showed that regional rent series are forecastable a year ahead (MAPE ~0.6%) but
that model-based intervals were over-confident. This module turns that research into a
product feature, on the latest official data:

* Series: the ONS PIPR average rent ("All" properties) for the 9 English regions, England,
  Wales and the UK, Jan 2015 -> latest month. Forecasts are REGIONAL — the level Module 4
  validated; an area's own trend is shown as history, next to its region's forecast.
* Candidates (on log rent): ARIMA(1,1,1); ARIMA(1,1,0) with drift; damped-trend ETS.
* Selection: ONE model for all series, chosen by pooled rolling-origin backtest MAPE
  (choosing per series on the same backtest would flatter the result).
* Bands: conformal — the empirical quantiles of the backtest's relative errors at each
  horizon, pooled over series. Coverage is checked out-of-time: widths calibrated on the
  older half of the origins, coverage measured on the newer half.

Output: models/rent_forecasts.json

    python -m src.models.rent_forecast
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("rentwise.forecast")
HORIZON = 12
LEVELS = (0.80, 0.95)


# ------------------------------------------------------------------ forecasters (log space)
def _arima111(y, h):
    from statsmodels.tsa.arima.model import ARIMA
    return np.asarray(ARIMA(y, order=(1, 1, 1)).fit().forecast(h))


def _arima_drift(y, h):
    from statsmodels.tsa.arima.model import ARIMA
    return np.asarray(ARIMA(y, order=(1, 1, 0), trend="t").fit().forecast(h))


def _ets_damped(y, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    return np.asarray(ExponentialSmoothing(y, trend="add", damped_trend=True).fit().forecast(h))


MODELS = {"ARIMA(1,1,1)": _arima111, "ARIMA(1,1,0)+drift": _arima_drift, "Damped-trend ETS": _ets_damped}


def forecast(series: np.ndarray, model: str, h: int = HORIZON) -> np.ndarray:
    """Point forecast in £ (model fitted on log rent)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.exp(MODELS[model](np.log(np.asarray(series, dtype=float)), h))


# ------------------------------------------------------------------ backtest + calibration
def backtest(series: np.ndarray, model: str, h: int = HORIZON, n_origins: int = 18, step: int = 2):
    """Relative errors (actual/forecast - 1), shape (origins, h), oldest origin first."""
    y = np.asarray(series, dtype=float)
    latest = len(y) - h
    origins = sorted(latest - k * step for k in range(n_origins))
    rel = [y[o:o + h] / forecast(y[:o], model, h) - 1 for o in origins if o >= 36]
    return np.array(rel)


def conformal_widths(rel: np.ndarray, levels=LEVELS) -> dict[float, np.ndarray]:
    """Per-horizon half-widths (relative) = finite-sample-corrected quantile of |error|."""
    a = np.abs(rel)
    n = a.shape[0]
    return {L: np.quantile(a, min(1.0, np.ceil((n + 1) * L) / n), axis=0) for L in levels}


def coverage(rel: np.ndarray, widths: dict[float, np.ndarray]) -> dict[float, float]:
    return {L: float(np.mean(np.abs(rel) <= w)) for L, w in widths.items()}


# ------------------------------------------------------------------ build
def load_series(trends_path: Path, areas_path: Path) -> dict[str, tuple[str, pd.Series]]:
    t = pd.read_parquet(trends_path)
    t = t[(t["measure"] == "All") & (t["level"].isin(["region", "nation"]))]
    names = pd.read_csv(areas_path).set_index("area_code")["area_name"].to_dict()
    out = {}
    for code, g in t.groupby("area_code", observed=True):
        s = g.sort_values("period").set_index("period")["rent"].astype(float)
        out[str(code)] = (names.get(code, code), s)
    return out


def build(processed_dir: Path, models_dir: Path, h: int = HORIZON) -> dict:
    series = load_series(processed_dir / "rent_trends.parquet", processed_dir / "rent_areas.csv")
    # 1) backtest every candidate on every series
    rels = {m: {c: backtest(s.values, m, h) for c, (_, s) in series.items()} for m in MODELS}
    mape = {m: float(np.mean([np.mean(np.abs(r)) for r in rels[m].values()]) * 100) for m in MODELS}
    chosen = min(mape, key=mape.get)
    pooled = np.concatenate(list(rels[chosen].values()), axis=0)          # (series*origins, h)
    # 2) out-of-time coverage check: calibrate on older origins, test on newer ones
    per = [r for r in rels[chosen].values()]
    half = per[0].shape[0] // 2
    old = np.concatenate([r[:half] for r in per]); new = np.concatenate([r[half:] for r in per])
    oot = coverage(new, conformal_widths(old))
    # 3) final widths from all origins; forecast from the latest month
    widths = conformal_widths(pooled)
    out = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
           "measure": "All properties (ONS PIPR average monthly rent)", "horizon_months": h,
           "method": {"chosen": chosen, "backtest_mape_pct": {k: round(v, 3) for k, v in mape.items()},
                      "origins_per_series": int(per[0].shape[0]),
                      "selection": "one model for all series, lowest pooled rolling-origin MAPE",
                      "bands": "conformal: per-horizon quantiles of backtest relative errors, pooled over series",
                      "out_of_time_coverage": {f"{int(k * 100)}%": round(v, 3) for k, v in oot.items()},
                      "mape_by_horizon_pct": [round(float(x) * 100, 3) for x in np.mean(np.abs(pooled), axis=0)]},
           "series": {}}
    for code, (name, s) in series.items():
        fc = forecast(s.values, chosen, h)
        start = pd.Period(s.index[-1], "M")
        pts = []
        for i, f in enumerate(fc):
            p = {"period": str(start + i + 1), "rent": int(round(f))}
            for L, w in widths.items():
                k = int(L * 100)
                p[f"low{k}"], p[f"high{k}"] = int(round(f * (1 - w[i]))), int(round(f * (1 + w[i])))
            pts.append(p)
        out["series"][code] = {"name": name, "last_period": s.index[-1], "last_rent": int(s.iloc[-1]),
                               "annual_change_pct": round((s.iloc[-1] / s.iloc[-13] - 1) * 100, 2),
                               "forecast": pts}
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "rent_forecasts.json").write_text(json.dumps(out, indent=2))
    log.info("forecasts built: chosen %s, MAPE %s, out-of-time coverage %s", chosen, mape, oot)
    return out


def write_summary(fc: dict, path: Path) -> None:
    m = fc["method"]
    lines = ["# Rent forecasts — regional, 12 months ahead (Phase C)", "",
             f"*Generated {fc['generated_at'][:10]} from ONS PIPR, all properties, monthly since Jan 2015.*", "",
             "| Model (log rent) | Rolling-origin MAPE, 1–12 months |", "|---|---|"]
    lines += [f"| {k}{' **(chosen)**' if k == m['chosen'] else ''} | {v:.2f}% |"
              for k, v in m["backtest_mape_pct"].items()]
    lines += ["", f"Error grows with the horizon: {m['mape_by_horizon_pct'][0]:.2f}% one month ahead, "
                  f"{m['mape_by_horizon_pct'][-1]:.2f}% twelve months ahead.", "",
              "**Bands are conformal**, built from the backtest's own errors at each horizon. "
              "Out-of-time check (calibrated on older origins, tested on newer): "
              + ", ".join(f"nominal {k} → **{v * 100:.0f}%** actual coverage"
                          for k, v in m["out_of_time_coverage"].items()) + ".", "",
              "| Area | Latest | Annual change | Forecast in 12 months (80% band) |", "|---|---|---|---|"]
    for s in sorted(fc["series"].values(), key=lambda s: -s["last_rent"]):
        f = s["forecast"][-1]
        lines.append(f"| {s['name']} | £{s['last_rent']:,} ({s['last_period']}) | {s['annual_change_pct']:+.1f}% | "
                     f"£{f['rent']:,} (£{f['low80']:,}–£{f['high80']:,}) |")
    lines += ["", "*Forecasts are statistical extrapolations of the official series, not predictions of policy "
                  "effects (e.g. the Renters' Rights Act). Indicative only.*"]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    fc = build(config.DATA_PROCESSED, config.MODELS_DIR)
    write_summary(fc, config.ROOT / "reports" / "rent_forecast_summary.md")
    print(json.dumps(fc["method"], indent=2))
