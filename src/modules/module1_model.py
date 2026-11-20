"""
module1_model.py  —  the single, shared Module 1 trainer (sale price).

Both `run_real_pipeline.py` and `run_module1_enriched.py` call `train_and_report`
so there is one source of truth. It is EPC-aware: if an EPC lookup is supplied it
adds floor area + rooms and reports the lift; otherwise it runs the baseline.

Produces (in `reports/`):
  * module1_summary.md         — metrics, SHAP, conformal intervals, cross-regional
  * figures/m1_predicted_vs_actual.png
  * figures/m1_shap.png
  * figures/m1_cross_regional.png
and returns a metrics dict for tests / callers.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.features import property_attributes as pa

log = logging.getLogger("rentwise.module1")

LABEL = {"is_new": "new build", "is_freehold": "freehold tenure",
         "district_med": "local area price level", "total_floor_area": "floor area (m²)",
         "number_habitable_rooms": "habitable rooms", "floor_area_per_room": "room size",
         "has_epc": "has EPC match", "pt_Detached": "detached",
         "pt_Semi-detached": "semi-detached", "pt_Terraced": "terraced", "pt_Flat": "flat"}


def _xgb(seed, n=400, depth=6):
    return XGBRegressor(n_estimators=n, max_depth=depth, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9,
                        random_state=seed, verbosity=0, n_jobs=4)


def _m(y, p):
    return (float(np.sqrt(mean_squared_error(y, p))),
            float(mean_absolute_error(y, p)), float(r2_score(y, p)))


def train_and_report(pp_path, epc_lookup=None, reports_dir=None, seed=42,
                     sample_n=90_000, region_map=None, make_figures=True) -> dict:
    """Train Module 1 (EPC-enriched when `epc_lookup` given) and write the summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports_dir = Path(reports_dir) if reports_dir else Path("reports")
    figs = reports_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    df = pa.load_price_paid(pp_path)
    n_clean = len(df)
    use_epc = epc_lookup is not None and len(epc_lookup) > 0
    match_rate = 0.0
    if use_epc:
        df = pa.add_epc_features(df, epc_lookup)
        match_rate = df.attrs.get("epc_match_rate", 0.0)

    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=seed).reset_index(drop=True)

    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.15, random_state=seed)
    df = pa.encode(df, tr, seed=seed)
    feat = pa.feature_columns(use_epc)
    y = df["price"].values
    X = df[feat]

    # ---- three-model comparison (always) + a baseline-vs-enriched A/B when EPC ----
    results = []
    for name, mdl, scale in [
        ("OLS (baseline)", LinearRegression(), True),
        ("Random Forest", RandomForestRegressor(n_estimators=200, max_depth=20, random_state=seed, n_jobs=4), False),
        ("XGBoost", _xgb(seed), False),
    ]:
        est = make_pipeline(StandardScaler(), mdl) if scale else mdl
        est.fit(X.iloc[tr], y[tr])
        results.append((name, *_m(y[te], est.predict(X.iloc[te]))))
    best = min(results, key=lambda r: r[1])
    best_name = best[0]

    ab = None
    if use_epc:
        base_feat = pa.feature_columns(False)
        bm = _xgb(seed).fit(df[base_feat].iloc[tr], y[tr])
        base_r2 = _m(y[te], bm.predict(df[base_feat].iloc[te]))[2]
        xm = _xgb(seed).fit(X.iloc[tr], y[tr])
        enr_r2 = _m(y[te], xm.predict(X.iloc[te]))[2]
        ab = {"baseline_r2": base_r2, "enriched_r2": enr_r2, "match_rate": match_rate}

    # ---- SHAP on best XGB ----
    imp = None
    try:
        import xgboost as xgb
        bmod = _xgb(seed).fit(X.iloc[tr], y[tr])
        Xs = X.iloc[te].sample(min(3000, len(te)), random_state=seed)
        contrib = bmod.get_booster().predict(xgb.DMatrix(Xs), pred_contribs=True)[:, :-1]
        imp = pd.Series(np.abs(contrib).mean(axis=0), index=feat).sort_values(ascending=False)
        if make_figures:
            top = imp.head(8)[::-1]
            plt.figure(figsize=(6.6, 4))
            plt.barh([LABEL.get(f, f) for f in top.index], top.values, color="#2E74B5")
            plt.xlabel("Mean |SHAP| (£ impact on sale price)")
            plt.title("Module 1: SHAP feature importance"); plt.tight_layout()
            plt.savefig(figs / "m1_shap.png", dpi=130); plt.close()
    except Exception as e:  # noqa: BLE001
        log.warning("SHAP skipped: %s", e)

    # ---- predicted vs actual figure (best XGB) ----
    pred_te = _xgb(seed).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    if make_figures:
        s = np.random.default_rng(seed).choice(len(te), size=min(2000, len(te)), replace=False)
        plt.figure(figsize=(5.2, 5))
        plt.scatter(y[te][s], pred_te[s], s=6, alpha=0.3, color="#2E74B5")
        lim = [0, max(y[te].max(), pred_te.max())]
        plt.plot(lim, lim, "r--", lw=1, label="perfect")
        plt.xlabel("Actual sale price (£)"); plt.ylabel("Predicted (£)")
        plt.title(f"Module 1: predicted vs actual (R² {best[3]:.3f})")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(figs / "m1_predicted_vs_actual.png", dpi=130); plt.close()

    # ---- split-conformal 90% intervals ----
    interval = None
    try:
        trf, cal = train_test_split(tr, test_size=0.25, random_state=seed)
        cm = _xgb(seed).fit(X.iloc[trf], y[trf])
        q = float(np.quantile(np.abs(y[cal] - cm.predict(X.iloc[cal])), 0.9))
        pr = cm.predict(X.iloc[te])
        cover = float(((y[te] >= pr - q) & (y[te] <= pr + q)).mean())
        interval = {"half_width": q, "coverage": cover}
    except Exception as e:  # noqa: BLE001
        log.warning("conformal skipped: %s", e)

    # ---- cross-regional leave-one-region-out ----
    xreg = []
    if region_map is not None:
        def _norm(s):
            s = str(s).lower().replace(" ua", "").replace("city of", "").replace("county of", "")
            return re.sub(r"[^a-z0-9]+", " ", s).strip()
        dfx = df.copy()
        dfx["region"] = dfx["district"].apply(lambda d: region_map.get(_norm(d)))
        dfx = dfx.dropna(subset=["region"])
        if len(dfx) > 40000:
            dfx = dfx.sample(40000, random_state=seed)
        for reg in sorted(dfx["region"].unique()):
            trn, tst = dfx[dfx.region != reg], dfx[dfx.region == reg]
            if len(tst) < 200:
                continue
            enc2 = trn.groupby("district")["price"].median(); g2 = trn["price"].median()
            def prep(d):
                d = d.copy(); d["district_med"] = d["district"].map(enc2).fillna(g2)
                for p in ("Detached", "Semi-detached", "Terraced", "Flat"):
                    d[f"pt_{p}"] = (d["ptype"] == p).astype(int)
                return d
            trn2, tst2 = prep(trn), prep(tst)
            mdl = _xgb(seed, n=150).fit(trn2[feat], trn2["price"])
            pr = mdl.predict(tst2[feat])
            xreg.append((reg, float(np.sqrt(mean_squared_error(tst2["price"], pr))),
                         float(r2_score(tst2["price"], pr)), len(tst2)))
        if xreg and make_figures:
            mx = np.mean([x[1] for x in xreg]); xs = sorted(xreg, key=lambda x: x[1])
            plt.figure(figsize=(6.6, 3.8))
            plt.barh([r for r, *_ in xs], [v for _, v, _, _ in xs], color="#2E74B5")
            plt.axvline(mx, color="black", ls="--", lw=1, label=f"mean £{mx:,.0f}")
            plt.xlabel("Leave-one-region-out RMSE (£)")
            plt.title("Module 1: cross-regional validation"); plt.legend(fontsize=8)
            plt.tight_layout(); plt.savefig(figs / "m1_cross_regional.png", dpi=130); plt.close()

    # ---- write summary ----
    tag = "REAL HM Land Registry + EPC floor area" if use_epc else "REAL HM Land Registry data"
    md = [f"# Module 1 - Sale Price Prediction (summary) [{tag}]\n",
          "## Model comparison (held-out test set)",
          "| Model | Test RMSE (£) | Test MAE (£) | Test R² |", "|---|---|---|---|"]
    md += [f"| {n} | {rm:,.0f} | {ma:,.0f} | {r2:.3f} |" for n, rm, ma, r2 in results]
    md.append(f"\n**Best model:** {best_name} (R² {best[3]:.3f}).")
    if ab:
        md.append(f"\n**EPC enrichment:** floor area + rooms lift XGBoost from R² {ab['baseline_r2']:.3f} "
                  f"to **{ab['enriched_r2']:.3f}** (+{ab['enriched_r2']-ab['baseline_r2']:.3f}); "
                  f"EPC match rate {100*ab['match_rate']:.0f}% of sales.")
    if imp is not None:
        md.append("\n## SHAP feature importance (best model, mean £ impact on sale price)")
        md += [f"- **{LABEL.get(f, f)}**: £{v:,.0f}" for f, v in imp.head(8).items()]
    if interval:
        md += ["\n## Calibrated uncertainty (split-conformal, 90%)",
               f"- 90% prediction-interval half-width ≈ £{interval['half_width']:,.0f}; "
               f"empirical coverage {100*interval['coverage']:.0f}% on the test set."]
    if xreg:
        mx = np.mean([x[1] for x in xreg])
        md += ["\n## Cross-regional validation (leave-one-region-out)",
               "| Region | RMSE (£) | R² | n |", "|---|---|---|---|"]
        md += [f"| {r} | {rm:,.0f} | {r2:.3f} | {n:,} |" for r, rm, r2, n in xreg]
        md.append(f"\nMean leave-one-region-out RMSE £{mx:,.0f} — a deliberately strict generalisation test.")
    md.append(f"\n*Modelled on {len(df):,} of {n_clean:,} cleaned standard residential sales; "
              "district median-price target encoding from training data only.*")
    (reports_dir / "module1_summary.md").write_text("\n".join(md))

    # record this run in the metrics history (quality over time)
    try:
        from src.models.registry import log_run
        log_run("module1", {"r2": best[3], "rmse": best[1], "mae": best[2],
                            "use_epc": use_epc, "match_rate": match_rate, "n": len(df)})
    except Exception as e:  # noqa: BLE001
        log.warning("metrics logging skipped: %s", e)

    return {"models": results, "best": best_name, "best_r2": best[3],
            "ab": ab, "interval": interval, "cross_regional": xreg,
            "use_epc": use_epc, "match_rate": match_rate, "n_modelled": len(df)}
