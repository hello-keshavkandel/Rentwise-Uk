"""
run_module1_enriched.py  —  Module 1, enriched with EPC floor area & rooms.

What it does
------------
1. Loads HM Land Registry Price Paid (real).
2. Loads the EPC lookup (data/processed/epc_lookup.parquet) built by
   `python -m src.data.epc`. If it's missing, it runs the BASELINE only and tells
   you how to add EPC.
3. Trains XGBoost twice — on the baseline feature set and on the EPC-enriched set
   — and reports the improvement (the whole point of this milestone).
4. Adds split-conformal prediction intervals (honest, model-agnostic uncertainty)
   and refreshes SHAP importance.
5. Writes reports/module1_enriched_summary.md.

Run:  python run_module1_enriched.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.features import property_attributes as pa

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("rentwise.module1")

SAMPLE_N = 90_000
SEED = getattr(config, "RANDOM_SEED", 42)


def _find_price_paid() -> Path | None:
    for name in ("land_registry_pp_2026.csv", "hm_land_registry_price_paid_latest.csv"):
        p = config.DATA_RAW / name
        if p.exists():
            return p
    return None


def _load_epc_lookup() -> pd.DataFrame | None:
    for ext in (".parquet", ".csv"):
        p = (config.DATA_PROCESSED / "epc_lookup").with_suffix(ext)
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p, dtype={"house_no": str})
    return None


def _fit_xgb(Xtr, ytr):
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                     subsample=0.9, colsample_bytree=0.9,
                     random_state=SEED, verbosity=0, n_jobs=4)
    m.fit(Xtr, ytr)
    return m


def _metrics(y, p):
    return dict(rmse=float(np.sqrt(mean_squared_error(y, p))),
                mae=float(mean_absolute_error(y, p)),
                r2=float(r2_score(y, p)))


def _conformal_interval(model, Xcal, ycal, Xte, alpha=0.1):
    """Split-conformal: width = (1-alpha) quantile of |residual| on a calibration set."""
    resid = np.abs(ycal - model.predict(Xcal))
    q = float(np.quantile(resid, 1 - alpha))
    pred = model.predict(Xte)
    lo, hi = pred - q, pred + q
    return pred, lo, hi, q


def main() -> None:
    pp_path = _find_price_paid()
    if pp_path is None:
        log.error("No Price Paid file in %s. Add land_registry_pp_2026.csv.", config.DATA_RAW)
        sys.exit(1)

    pp = pa.load_price_paid(pp_path)
    epc = _load_epc_lookup()
    have_epc = epc is not None
    if have_epc:
        pp = pa.add_epc_features(pp, epc)
        match_rate = pp.attrs.get("epc_match_rate", float("nan"))
        log.info("EPC lookup loaded (%d properties). Match rate %.1f%%.",
                 len(epc), 100 * match_rate)
    else:
        match_rate = 0.0
        log.warning("No EPC lookup found — running BASELINE only. "
                    "Build it with:  python -m src.data.epc  (after placing EPC "
                    "bulk data in data/raw/epc/). See src/data/epc.py.")

    if len(pp) > SAMPLE_N:
        pp = pp.sample(SAMPLE_N, random_state=SEED).reset_index(drop=True)

    # train / calibration / test split (60/20/20) — calibration feeds conformal
    idx = np.arange(len(pp))
    tr, tmp = train_test_split(idx, test_size=0.40, random_state=SEED)
    cal, te = train_test_split(tmp, test_size=0.50, random_state=SEED)
    pp = pa.encode(pp, tr, seed=SEED)
    y = pp["price"].values

    def run(use_epc):
        cols = pa.feature_columns(use_epc and have_epc)
        X = pp[cols]
        m = _fit_xgb(X.iloc[tr], y[tr])
        return cols, m, _metrics(y[te], m.predict(X.iloc[te]))

    base_cols, base_m, base_s = run(False)
    log.info(f"BASELINE   R2={base_s['r2']:.3f} RMSE=£{base_s['rmse']:,.0f}")

    lines = ["# Module 1 — EPC-Enriched Sale Price Prediction\n",
             f"Price Paid: {pp_path.name} · modelled on {len(pp):,} sales "
             f"(60/20/20 train/calibration/test).\n",
             "## Model comparison (held-out test set)",
             "| Feature set | Test RMSE (£) | Test MAE (£) | Test R² |",
             "|---|---|---|---|",
             f"| Baseline (no floor area) | {base_s['rmse']:,.0f} | {base_s['mae']:,.0f} | {base_s['r2']:.3f} |"]

    if have_epc:
        epc_cols, epc_m, epc_s = run(True)
        log.info(f"EPC-ENRICHED R2={epc_s['r2']:.3f} RMSE=£{epc_s['rmse']:,.0f}")
        d_r2 = epc_s["r2"] - base_s["r2"]
        d_rmse = (base_s["rmse"] - epc_s["rmse"]) / base_s["rmse"] * 100
        lines.append(f"| **EPC-enriched (floor area + rooms)** | **{epc_s['rmse']:,.0f}** | "
                     f"**{epc_s['mae']:,.0f}** | **{epc_s['r2']:.3f}** |")
        lines.append(f"\n**EPC lift:** R² +{d_r2:.3f} ({base_s['r2']:.3f} → {epc_s['r2']:.3f}); "
                     f"RMSE −{d_rmse:.0f}%. EPC match rate: {100*match_rate:.0f}% of sales.\n")

        # conformal intervals on the enriched model
        X = pp[epc_cols]
        pred, lo, hi, q = _conformal_interval(epc_m, X.iloc[cal], y[cal], X.iloc[te], alpha=0.1)
        cover = float(((y[te] >= lo) & (y[te] <= hi)).mean())
        lines += ["## Calibrated uncertainty (split-conformal, 90%)",
                  f"- 90% prediction interval half-width ≈ £{q:,.0f}; empirical coverage "
                  f"{100*cover:.0f}% on the test set.\n"]

        # SHAP on enriched model
        try:
            import xgboost as xgb
            Xs = X.iloc[te].sample(min(3000, len(te)), random_state=SEED)
            contrib = epc_m.get_booster().predict(xgb.DMatrix(Xs), pred_contribs=True)[:, :-1]
            imp = pd.Series(np.abs(contrib).mean(axis=0), index=epc_cols).sort_values(ascending=False)
            lines.append("## SHAP feature importance (enriched model, mean £ impact)")
            label = {"district_med": "local area price level", "total_floor_area": "floor area (m²)",
                     "number_habitable_rooms": "habitable rooms", "floor_area_per_room": "room size",
                     "pt_Detached": "detached", "pt_Semi-detached": "semi-detached",
                     "pt_Terraced": "terraced", "pt_Flat": "flat", "is_freehold": "freehold",
                     "is_new": "new build", "has_epc": "has EPC match"}
            lines += [f"- **{label.get(f, f)}**: £{v:,.0f}" for f, v in imp.head(8).items()]
        except Exception as e:  # noqa: BLE001
            log.warning("SHAP step skipped: %s", e)
    else:
        lines.append("\n_EPC enrichment pending — add EPC bulk data to run the comparison._\n")

    out = config.ROOT / "reports" / "module1_enriched_summary.md"
    out.write_text("\n".join(lines))
    log.info("Wrote %s", out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
