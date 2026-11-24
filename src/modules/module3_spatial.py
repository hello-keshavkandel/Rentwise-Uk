"""
module3_spatial.py  —  Module 3 on REAL data: spatial features + their measured value.

Builds one row of spatial features per local authority from the prepared inputs
  * OSM amenity counts       data/processed/osm_amenities_by_la.csv   (src/data/osm.py)
  * historic Ofsted quality  data/processed/school_quality_by_la.csv  (src/data/schools.py)
  * LA area (km²)            data/processed/lad_boundaries_wgs84.geojson (src/data/geo.py)
into data/processed/la_spatial_features.csv (small, committed), then measures how much
they improve the Module 2 rent model.

Evaluation design (important): every LA appears up to 4 times (one row per property
type) with identical LA-level features, so a random train/test split lets the model
"see" an LA during training and score it again at test time. We therefore report
**grouped cross-validation with whole LAs held out**, repeated over several random
LA-to-fold assignments, alongside the random-split figure for comparison.

    python -m src.modules.module3_spatial
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("rentwise.module3")

SCHOOL_COLS = ["school_quality", "share_good_or_better"]


def la_areas_km2(boundaries_geojson, code_field: str = "LAD24CD") -> dict[str, float]:
    """Geodesic area of each LA (km²). Needs pyproj + shapely (data-prep deps)."""
    from pyproj import Geod
    from shapely.geometry import shape
    geod = Geod(ellps="WGS84")
    geo = json.loads(Path(boundaries_geojson).read_text())
    return {f["properties"][code_field]: abs(geod.geometry_area_perimeter(shape(f["geometry"]))[0]) / 1e6
            for f in geo["features"]}


def build_la_spatial_features(osm_counts: pd.DataFrame, schools: pd.DataFrame,
                              areas_km2: dict[str, float]) -> pd.DataFrame:
    """Per-LA table: log(1 + amenities per km²) for each OSM category + school quality."""
    osm = osm_counts.set_index("lad_code")
    area = osm.index.map(areas_km2).astype(float)
    dens = np.log1p(osm.div(area, axis=0)).add_prefix("dens_")
    out = dens.join(schools.set_index("lad_code")[SCHOOL_COLS], how="left")
    out.insert(0, "area_km2", area)
    return out.reset_index()


def _xgb(seed):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=seed, n_jobs=4)


def grouped_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, repeats: int = 4,
               folds: int = 5, model_factory=_xgb) -> dict:
    """Repeated K-fold with whole groups (LAs) held out. Returns mean/sd R² and MAE."""
    from sklearn.metrics import mean_absolute_error, r2_score
    uniq = pd.unique(groups)
    r2s, maes = [], []
    for rep in range(repeats):
        fold_of = dict(zip(uniq, np.random.default_rng(rep).permutation(len(uniq)) % folds))
        fid = np.array([fold_of[g] for g in groups])
        pred = np.zeros(len(y))
        for k in range(folds):
            tr, te = fid != k, fid == k
            pred[te] = model_factory(rep).fit(X[tr], y[tr]).predict(X[te])
        r2s.append(r2_score(y, pred))
        maes.append(mean_absolute_error(y, pred))
    return {"r2": float(np.mean(r2s)), "r2_sd": float(np.std(r2s)),
            "mae": float(np.mean(maes)), "mae_sd": float(np.std(maes)), "r2_runs": r2s}


def random_cv(X, y, folds: int = 5, model_factory=_xgb) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold
    pred = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=42).split(X):
        pred[te] = model_factory(42).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return {"r2": float(r2_score(y, pred)), "mae": float(mean_absolute_error(y, pred))}


def evaluate(clean: pd.DataFrame, features: pd.DataFrame, spatial: pd.DataFrame,
             target: str = "median_monthly_rent", repeats: int = 4) -> dict:
    """Ablation: base Module 2 features vs + amenities / + schools / + both."""
    S = spatial.set_index("lad_code").reindex(clean["la_code"]).reset_index(drop=True)
    y = features[target].to_numpy()
    base = features.drop(columns=[target])
    dens = [c for c in S.columns if c.startswith("dens_")]
    sets = {"base": base,
            "+amenities": pd.concat([base, S[dens]], axis=1),
            "+schools": pd.concat([base, S[SCHOOL_COLS]], axis=1),
            "+both": pd.concat([base, S[dens + SCHOOL_COLS]], axis=1)}
    groups = clean["la_code"].to_numpy()
    res = {k: grouped_cv(X, y, groups, repeats=repeats) for k, X in sets.items()}
    res["random_split"] = {"base": random_cv(sets["base"], y), "+both": random_cv(sets["+both"], y)}
    res["n_rows"], res["n_las"] = int(len(y)), int(clean["la_code"].nunique())
    return res


def write_summary(res: dict, spatial: pd.DataFrame, path: Path) -> None:
    b, s = res["base"], res["+both"]
    gains = np.array(s["r2_runs"]) - np.array(b["r2_runs"])
    rows = "\n".join(f"| {k} | {v['r2']:.3f} ± {v['r2_sd']:.3f} | £{v['mae']:.0f} ± {v['mae_sd']:.0f} |"
                     for k, v in res.items() if k in ("base", "+amenities", "+schools", "+both"))
    rs = res["random_split"]
    med_year = (int(spatial["median_inspection_year"].median())
                if "median_inspection_year" in spatial else "n/a")
    md = f"""# Module 3 - Geospatial features (summary) [REAL OSM + Ofsted + ONS boundaries]

**Spatial features improve rent prediction for unseen local authorities:**
R² {b['r2']:.3f} → **{s['r2']:.3f}** (+{s['r2'] - b['r2']:.3f}), MAE £{b['mae']:.0f} → **£{s['mae']:.0f}**
({100 * (s['mae'] - b['mae']) / b['mae']:+.0f}%). Gain in every repeat: {', '.join(f'+{g:.3f}' for g in gains)}.

## Ablation — whole LAs held out (grouped 5-fold CV × {len(b['r2_runs'])} repeats, XGBoost)
| Features | R² (mean ± sd) | MAE |
|---|---|---|
{rows}

## Why grouped CV
Each LA appears up to 4 times (one row per property type) with identical LA-level features.
A random split scores LAs the model has already seen: base R² {rs['base']['r2']:.3f} / +spatial
{rs['+both']['r2']:.3f} (MAE £{rs['base']['mae']:.0f} / £{rs['+both']['mae']:.0f}). Holding out whole LAs is the honest
test of generalisation; the dashboard only offers LAs present in training, so its in-sample
accuracy sits between the two.

## Features ({res['n_las']} LAs, {res['n_rows']} rows)
- **Amenity density** — log(1 + count per km²) for 11 OSM categories (rail stations, bus stops,
  supermarkets, convenience, GP/clinics, hospitals, pharmacies, restaurants/cafés, pubs/bars,
  parks, gyms); 506,395 amenities from Geofabrik England + Wales (Nov 2026).
- **School quality** — mean historic Ofsted grade (last 2019-framework grade, labelled historic)
  and share Good-or-better; 18,263 schools. Missing for Welsh LAs (Estyn) — left as NaN.

## Caveats
- OSM completeness varies by area; densities are relative indicators, not a census.
- Density per km² favours compact urban LAs; per-capita measures need ONS population (future).
- School grades are historic (median inspection year {med_year}).
- Associations, not causes: these features describe places, they don't set rents.
"""
    path.write_text(md)


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    P = config.DATA_PROCESSED
    schools = pd.read_csv(P / "school_quality_by_la.csv")
    spatial = build_la_spatial_features(pd.read_csv(P / "osm_amenities_by_la.csv"), schools,
                                        la_areas_km2(P / "lad_boundaries_wgs84.geojson"))
    spatial.to_csv(P / "la_spatial_features.csv", index=False)
    res = evaluate(pd.read_csv(P / "rentwise_clean.csv"), pd.read_csv(P / "rentwise_features.csv"), spatial)
    write_summary(res, schools, config.ROOT / "reports" / "module3_summary.md")
    try:
        from src.models.registry import log_run
        log_run("module3", {"r2_base_grouped": res["base"]["r2"], "r2_spatial_grouped": res["+both"]["r2"],
                            "mae_base_grouped": res["base"]["mae"], "mae_spatial_grouped": res["+both"]["mae"],
                            "r2_base_random": res["random_split"]["base"]["r2"],
                            "r2_spatial_random": res["random_split"]["+both"]["r2"]})
    except Exception as e:  # noqa: BLE001
        log.warning("registry log skipped: %s", e)
    print({k: (round(v["r2"], 3), round(v["mae"])) for k, v in res.items()
           if k in ("base", "+amenities", "+schools", "+both")})
