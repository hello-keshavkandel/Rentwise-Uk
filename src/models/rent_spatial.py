"""
rent_spatial.py  —  the deployed rent model, now with Module 3 spatial features.

Trains XGBoost on the Module 2 features PLUS the per-LA spatial features
(OSM amenity density + historic Ofsted quality), and writes three artefacts the
app and API load:

  models/rent_xgb_spatial.json      XGBoost booster in XGBoost's portable JSON format
                                    (no pickle -> no "older XGBoost version" warnings)
  models/rent_model_meta.json       feature order, trained_at, params, evaluation
  models/conformal.json             calibrated 80/90% interval half-widths for THIS model

Evaluation reported in the metadata:
  * random 5-fold CV    — LAs seen in training (what the dashboard serves)
  * grouped 5-fold CV   — whole LAs held out (generalisation to unseen areas)
  * interval coverage   — under both regimes, so the intervals' scope is explicit

    python -m src.models.rent_spatial
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("rentwise.rent_spatial")

TARGET = "median_monthly_rent"
PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=4)
MODEL_FILE = "rent_xgb_spatial.json"
META_FILE = "rent_model_meta.json"


def make_model():
    from xgboost import XGBRegressor
    return XGBRegressor(**PARAMS)


def spatial_columns(spatial: pd.DataFrame) -> list[str]:
    return [c for c in spatial.columns if c.startswith("dens_")] + ["school_quality", "share_good_or_better"]


def training_frame(clean: pd.DataFrame, features: pd.DataFrame, spatial: pd.DataFrame):
    """X (Module 2 features + spatial), y, LA groups — rows aligned with `clean`."""
    if len(clean) != len(features):
        raise ValueError("rentwise_clean and rentwise_features are not row-aligned")
    cols = spatial_columns(spatial)
    S = spatial.set_index("lad_code")[cols].reindex(clean["la_code"]).reset_index(drop=True)
    missing = clean.loc[S.isna().all(axis=1).to_numpy(), "la_code"].unique()
    if len(missing):
        raise ValueError(f"no spatial features for LAs: {list(missing)[:5]}")
    X = pd.concat([features.drop(columns=[TARGET]).reset_index(drop=True), S], axis=1)
    return X, features[TARGET].to_numpy(dtype=float), clean["la_code"].to_numpy()


def _oof(X, y, folds_of):
    pred = np.zeros(len(y))
    for k in np.unique(folds_of):
        tr, te = folds_of != k, folds_of == k
        pred[te] = make_model().fit(X[tr], y[tr]).predict(X[te])
    return pred


def _metrics(y, pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"r2": float(r2_score(y, pred)), "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred)))}


def _coverage(y, pred, q):
    return float(((y >= pred - q) & (y <= pred + q)).mean())


def evaluate_and_calibrate(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                           alphas=(0.1, 0.2), seed: int = 42) -> dict:
    """Random + grouped OOF metrics; conformal widths from random OOF residuals."""
    rng = np.random.default_rng(seed)
    rand_folds = rng.permutation(len(y)) % 5
    uniq = pd.unique(groups)
    g_fold = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % 5))
    grp_folds = np.array([g_fold[g] for g in groups])

    Xv = X.to_numpy(dtype=float)
    p_rand, p_grp = _oof(Xv, y, rand_folds), _oof(Xv, y, grp_folds)
    resid = np.abs(y - p_rand)
    n = len(resid)
    q = {str(a): float(np.quantile(resid, min(np.ceil((n + 1) * (1 - a)) / n, 1.0))) for a in alphas}
    return {
        "random_cv": _metrics(y, p_rand),
        "grouped_cv": _metrics(y, p_grp),
        "conformal_q": q,
        "coverage_random_cv": {a: _coverage(y, p_rand, w) for a, w in q.items()},
        "coverage_grouped_cv": {a: _coverage(y, p_grp, w) for a, w in q.items()},
    }


FRIENDLY = {"median_annual_income": "Local income", "crime_per_1000": "Crime per 1,000 residents",
            "bedrooms": "Bedrooms",
            "property_type_ord": "Property type", "income_band": "Income band",
            "school_quality": "School quality (Ofsted)", "share_good_or_better": "Share Good+ schools"}


def shap_figure(model, X: pd.DataFrame, path: Path, top: int = 12) -> list[tuple[str, float]]:
    """Mean |SHAP| (exact XGBoost contributions) for the deployed model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xgboost as xgb
    contrib = model.get_booster().predict(xgb.DMatrix(X.to_numpy(dtype=float)), pred_contribs=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=X.columns)
    # collapse the region one-hots into one bar
    reg = imp[[c for c in imp.index if c.startswith("region_")]].sum()
    imp = imp[[c for c in imp.index if not c.startswith("region_")]]
    imp["Region"] = reg
    imp = imp.sort_values(ascending=False).head(top)
    names = [FRIENDLY.get(c, (c[5:].replace("_", " ").capitalize() + " density") if c.startswith("dens_") else c)
             for c in imp.index]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(names[::-1], imp.values[::-1], color="#2e86c1")
    ax.set_xlabel("Mean |impact| on rent estimate (£)"); ax.set_title("Deployed rent model — what drives estimates")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)
    return list(zip(names, [float(v) for v in imp.values]))


def write_summary(meta: dict, path: Path) -> None:
    ev = meta["evaluation"]; r, g = ev["random_cv"], ev["grouped_cv"]
    q, cr, cg = ev["conformal_q"], ev["coverage_random_cv"], ev["coverage_grouped_cv"]
    Path(path).write_text(f"""# Module 2 - Rent model (deployed) [REAL ONS + police + OSM + Ofsted]

**Deployed:** XGBoost on Module 2 features + Module 3 spatial features, trained {meta['trained_at'][:10]}
on {meta['n_rows']} rows ({meta['n_las']} local authorities).

| Evaluation (5-fold, out-of-fold) | R² | MAE | RMSE |
|---|---|---|---|
| LAs seen in training (what the app/API serve) | {r['r2']:.3f} | £{r['mae']:.0f} | £{r['rmse']:.0f} |
| Whole LAs held out (unseen areas) | {g['r2']:.3f} | £{g['mae']:.0f} | £{g['rmse']:.0f} |

**Calibrated intervals (conformal):** 80% → ±£{q['0.2']:.0f}, 90% → ±£{q['0.1']:.0f}.
Coverage for served areas {cr['0.2']*100:.0f}% / {cr['0.1']*100:.0f}% (on target). For unseen areas they would
cover only {cg['0.2']*100:.0f}% / {cg['0.1']*100:.0f}% — which is why the app and API serve only the
{meta['n_las']} trained areas.

Before Phase A (Nov 2026 audit) the model trained on 1,198 rows after 66 genuine high rents had been
deleted as "outliers", with a one-month raw crime count and a constant transport placeholder:
R² 0.947 / MAE £68 (seen), 0.817 / £136 (unseen). Phase A restored every official rent and replaced
crime with a true rate, so the figures are not directly comparable — the restored rows are the most
expensive in the country.

*Spatial features: log density of 11 OpenStreetMap amenity types + historic Ofsted school quality.
Associations, not causes.*
""")


def train_and_save(clean, features, spatial, models_dir: Path) -> dict:
    import xgboost
    X, y, groups = training_frame(clean, features, spatial)
    ev = evaluate_and_calibrate(X, y, groups)
    model = make_model().fit(X.to_numpy(dtype=float), y)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(models_dir / MODEL_FILE))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {"model_file": MODEL_FILE, "target": TARGET, "feature_cols": list(X.columns),
            "spatial_cols": spatial_columns(spatial), "trained_at": now, "n_rows": int(len(y)),
            "n_las": int(pd.unique(groups).size), "params": PARAMS,
            "xgboost_version": xgboost.__version__, "evaluation": ev}
    (models_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    (models_dir / "conformal.json").write_text(json.dumps(
        {"target": TARGET, "n": int(len(y)), "alpha_to_q": ev["conformal_q"],
         "model": MODEL_FILE, "generated_at": now}, indent=2))
    meta["shap_top"] = shap_figure(model, X, models_dir.parent / "reports" / "figures" / "rent_shap_importance.png")
    (models_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    log.info("rent model saved: %s", {k: ev[k] for k in ("random_cv", "grouped_cv")})
    return meta


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    from src.data.store import load_table
    P = config.DATA_PROCESSED
    meta = train_and_save(load_table("rentwise_clean"), load_table("rentwise_features"),
                          pd.read_csv(P / "la_spatial_features.csv"), config.MODELS_DIR)
    ev = meta["evaluation"]
    write_summary(meta, config.ROOT / "reports" / "rent_model_summary.md")
    try:
        from src.models.registry import log_run
        log_run("module2_spatial", {"r2_random": ev["random_cv"]["r2"], "r2_grouped": ev["grouped_cv"]["r2"],
                                    "mae_random": ev["random_cv"]["mae"], "mae_grouped": ev["grouped_cv"]["mae"],
                                    "q80": ev["conformal_q"]["0.2"],
                                    "cov80_random": ev["coverage_random_cv"]["0.2"],
                                    "cov80_grouped": ev["coverage_grouped_cv"]["0.2"]})
    except Exception as e:  # noqa: BLE001
        log.warning("registry log skipped: %s", e)
    print(json.dumps({k: ev[k] for k in ev}, indent=1))
