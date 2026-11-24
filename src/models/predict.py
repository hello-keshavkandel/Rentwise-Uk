"""
predict.py  —  the inference engine used by my dashboard (Phase 9).

I wrap everything Module 2 produced behind one class, RentWisePredictor, so my
dashboard stays thin. Given a plain choice (local authority, property type,
bedrooms) I return the predicted rent, the affordability verdict, and the SHAP
contributions that explain it. Keeping this out of the UI guarantees I build the
model's inputs exactly as training did - the most common source of silent bugs.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.models.affordability import AffordabilityEngine
from src.data.store import load_table

TARGET = "median_monthly_rent"
PROPERTY_ORDER = {"Room": 1, "Flat": 2, "Terraced": 3, "Semi-detached": 4, "Detached": 5}
# I never return an implausible rent to a user. The linear baseline can extrapolate
# below zero for extreme low-value inputs, so I floor every estimate at the same
# minimum the data uses (£250). This is a documented safeguard, not a data fix.
RENT_FLOOR = 250.0


class UnknownAreaError(KeyError):
    """Requested area is not one the deployed model was trained on (not served)."""


class RentWisePredictor:
    """I load my trained model and use it to answer one property at a time."""

    def __init__(self):
        # I load the model, the exact feature order, the income bands, and a
        # per-area lookup of the figures a user can't be expected to know.
        # Preferred: the spatial model (Module 2 + Module 3 features) saved as portable
        # XGBoost JSON with a metadata file. Fallback: the legacy pickled model.
        self.meta = self._load_meta()
        feats = load_table("rentwise_features")   # DuckDB store, CSV fallback
        if self.meta:
            from xgboost import XGBRegressor
            self.model = XGBRegressor()
            self.model.load_model(str(config.MODELS_DIR / self.meta["model_file"]))
            self.feature_cols = list(self.meta["feature_cols"])
            self.spatial_cols = list(self.meta.get("spatial_cols", []))
            sp = pd.read_csv(config.DATA_PROCESSED / "la_spatial_features.csv")
            self.spatial = sp.set_index("lad_code")
        else:
            self.model = joblib.load(config.MODELS_DIR / "best_model.joblib")
            self.feature_cols = [c for c in feats.columns if c != TARGET]
            self.spatial_cols, self.spatial = [], None
        _, self.income_edges = pd.qcut(feats["median_annual_income"], 4,
                                       retbins=True, labels=False)

        clean = load_table("rentwise_clean")
        self.areas = (clean.groupby("la_name")
                      .agg(la_code=("la_code", "first"), region=("region", "first"),
                           income=("median_annual_income", "first"),
                           crime=("crime_per_1000", "first"),
                           transport=("transport_score", "first"))
                      .reset_index())

        # I only offer property types the model was actually trained on, and I derive
        # a typical bedroom count per type from the data. In the ONS rent data the
        # bedroom count is tied to the property type, so I do not expose a free
        # bedroom control that would push the model outside its training domain.
        self.type_beds = (clean.groupby("property_type")["bedrooms"].median()
                          .round().astype(int).to_dict())
        self.property_types = [t for t in PROPERTY_ORDER if t in self.type_beds]
        self.affordability = AffordabilityEngine()      # I reuse the shared engine
        # I keep a small background sample so a linear model can be explained too.
        # (Only needed for linear models; the spatial model is explained exactly via XGBoost.)
        base_cols = [c for c in self.feature_cols if c in feats.columns]
        self._background = feats[base_cols].sample(
            min(len(feats), 200), random_state=config.RANDOM_SEED)
        # Calibrated 80% prediction interval half-width (conformal), if available.
        from src.models.conformal import load_q
        self.interval_q = load_q(config.MODELS_DIR / "conformal.json", alpha=0.2)
        self.interval_level = 0.80

    @staticmethod
    def _load_meta():
        p = config.MODELS_DIR / "rent_model_meta.json"
        if not p.exists():
            return None
        import json
        meta = json.loads(p.read_text())
        if not (config.MODELS_DIR / meta.get("model_file", "")).exists():
            return None
        return meta

    def model_info(self) -> dict:
        """What is deployed: model type, training time, honest evaluation figures."""
        if not self.meta:
            return {"model": "xgboost-legacy", "spatial": False}
        ev = self.meta.get("evaluation", {})
        return {"model": "xgboost-spatial", "spatial": True, "trained_at": self.meta.get("trained_at"),
                "n_las": self.meta.get("n_las"), "random_cv": ev.get("random_cv"),
                "grouped_cv": ev.get("grouped_cv"), "interval_coverage_seen_las": ev.get("coverage_random_cv")}

    def area_by_code(self, la_code: str):
        """Area row for an LA code, or None if the LA is not served (not in training)."""
        m = self.areas.loc[self.areas["la_code"] == la_code]
        return None if m.empty else m.iloc[0]

    def spatial_profile(self, la_name: str) -> dict:
        """Raw spatial features for an area (empty if the legacy model is deployed)."""
        if self.spatial is None:
            return {}
        row = self.spatial.loc[self._area(la_name)["la_code"]]
        return {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}

    def area_names(self):
        # I give the dashboard the list of areas to show in its dropdown.
        return sorted(self.areas["la_name"].tolist())

    def typical_bedrooms(self, property_type) -> int:
        # I return the bedroom count the data associates with this property type.
        return int(self.type_beds.get(property_type, 2))

    def _area(self, la_name):
        m = self.areas.loc[self.areas["la_name"] == la_name]
        if m.empty:
            raise UnknownAreaError(f"'{la_name}' is not one of the {len(self.areas)} areas this model serves")
        return m.iloc[0]

    def _build_row(self, la_name, property_type, bedrooms, year=2024) -> pd.DataFrame:
        # I rebuild the exact one-row feature vector the model was trained on.
        a = self._area(la_name)
        if property_type not in PROPERTY_ORDER:
            raise ValueError(f"unknown property type '{property_type}'")
        row = {c: 0 for c in self.feature_cols}          # I start all-zero
        row["year"] = year
        row["bedrooms"] = bedrooms
        row["median_annual_income"] = a["income"]
        row["crime_per_1000"] = a["crime"]
        row["transport_score"] = a["transport"]
        row["property_type_ord"] = PROPERTY_ORDER[property_type]
        row["income_band"] = int(np.clip(
            np.searchsorted(self.income_edges, a["income"], side="right"), 1, 4))
        region_col = f"region_{a['region']}"
        if region_col in row:
            row[region_col] = 1
        if self.spatial is not None:                      # Module 3 features for this LA
            sp = self.spatial.loc[a["la_code"]]
            for c in self.spatial_cols:
                row[c] = float(sp[c]) if pd.notna(sp[c]) else np.nan
        return pd.DataFrame([row])[self.feature_cols]

    def _predict_raw(self, X1) -> float:
        # The spatial model was trained on a plain float matrix; pass the same.
        if self.meta:
            return float(self.model.predict(X1.to_numpy(dtype=float))[0])
        return float(self.model.predict(X1)[0])

    def _shap_row(self, X1):
        # I explain one prediction with the explainer that matches the model type,
        # so the dashboard works whether the baseline or a tree model is deployed.
        name = type(self.model).__name__
        if name.startswith("XGB"):
            import xgboost as xgb
            data = X1.to_numpy(dtype=float) if self.meta else X1
            c = self.model.get_booster().predict(xgb.DMatrix(data), pred_contribs=True)
            return c[0, :-1]
        if name in ("Ridge", "LinearRegression", "Lasso", "ElasticNet"):
            expl = shap.LinearExplainer(self.model, shap.maskers.Independent(self._background))
            return np.array(expl.shap_values(X1))[0]
        try:
            return np.array(shap.TreeExplainer(self.model).shap_values(X1))[0]
        except Exception:
            expl = shap.Explainer(self.model.predict, self._background)
            return np.array(expl(X1).values)[0]

    def predict(self, la_name, property_type, bedrooms=None, year=2024) -> dict:
        # I predict the rent, judge affordability, and explain the result.
        if bedrooms is None:
            bedrooms = self.typical_bedrooms(property_type)
        X1 = self._build_row(la_name, property_type, bedrooms, year)
        rent = max(self._predict_raw(X1), RENT_FLOOR)  # I floor to a valid rent

        area = self._area(la_name)
        income = float(area["income"])
        ratio = self.affordability.ratio(rent * 12, income)

        shap_vals = self._shap_row(X1)
        # I drop region dummies that are 0 for this property, so the explanation
        # only shows the region it is actually in (not "being in London" for a
        # non-London property).
        row0 = X1.iloc[0]
        # I drop region dummies that are 0 for this property, and I also hide any
        # feature whose impact rounds to £0. On the real data `year` is a constant
        # (one ONS monthly snapshot), so it carries no signal and would otherwise
        # show a confusing "£0" line in the explanation.
        # A feature with NO data for this area (e.g. Ofsted school quality in Wales,
        # where schools are inspected by Estyn) is re-labelled "<feature>__missing" so
        # the UI never presents the model's missing-value handling as a fact about the area.
        pairs = [(f + "__missing" if pd.isna(row0.get(f)) else f, v)
                 for f, v in zip(self.feature_cols, shap_vals)
                 if not (f.startswith("region_") and row0.get(f, 0) == 0)
                 and abs(v) >= 0.5]
        contribs = sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)
        # Calibrated prediction interval (80% by default), floored to a valid rent.
        interval = None
        if self.interval_q is not None:
            interval = {"low": max(round(rent - self.interval_q), int(RENT_FLOOR)),
                        "high": round(rent + self.interval_q), "level": self.interval_level}
        return {"rent": round(rent), "region": area["region"], "la_code": area["la_code"], "income": income,
                "ratio": ratio, "affordable": bool(self.affordability.is_affordable(ratio)),
                "contributions": contribs, "interval": interval}


if __name__ == "__main__":
    # I run a quick self-test.
    p = RentWisePredictor()
    demo = p.predict(p.area_names()[0], "Flat", 2)
    print("Area:", p.area_names()[0])
    print(f"Predicted rent: £{demo['rent']}/month")
    print(f"Rent-to-income: {round(demo['ratio']*100)}% ->",
          "AFFORDABLE" if demo["affordable"] else "ABOVE 30%")
    print("Top driver:", demo["contributions"][0])
