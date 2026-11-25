"""
explain_shap.py  —  Phase 7, Step 11: Explainability with SHAP.

I open up my model with SHAP so it can explain WHY it predicted a rent. I produce
global importance (which features matter across all predictions) and a local,
plain-English explanation of a single prediction. Transparency is the core promise
of RentWise UK, so this step is essential, not optional.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.models.prepare import DataSplitter


class ShapExplainer:
    """I use SHAP to turn my model's numbers into explanations, measured in pounds.

    Why SHAP (I explain this in the viva): it is based on cooperative game theory
    (Lundberg & Lee, 2017) and fairly splits each prediction into a contribution
    per feature, in the same units as the target. So I can say 'transport added
    £40, being in London added £120' - something a renter or examiner can trust.
    """

    FRIENDLY = {
        "bedrooms": "number of bedrooms", "median_annual_income": "local income",
        "crime_per_1000": "crime per 1,000 residents",
        "property_type_ord": "property type/size", "income_band": "area income band",
        "year": "year",
    }

    def __init__(self):
        self.fig_dir = config.ROOT / "reports" / "figures"

    def _friendly(self, feature: str) -> str:
        # I turn a raw column name into a human-readable label.
        if feature.startswith("region_"):
            return "being in " + feature.replace("region_", "")
        return self.FRIENDLY.get(feature, feature)

    def _chart_global(self, importance):
        top = importance.head(10)[::-1]
        plt.figure(figsize=(7, 5))
        plt.barh([self._friendly(f) for f in top.index], top.values, color="#2E74B5")
        plt.title("Global feature importance (mean |SHAP|, £)")
        plt.xlabel("Average impact on predicted rent (£)"); plt.tight_layout()
        out = self.fig_dir / "07_shap_importance.png"; plt.savefig(out, dpi=120); plt.close()
        return out

    def _shap_values(self, model, X):
        # I pick the right SHAP explainer for whichever model won, so this step
        # works for the linear baseline or either tree model, in any environment.
        name = type(model).__name__
        if name in ("Ridge", "LinearRegression", "Lasso", "ElasticNet"):
            masker = shap.maskers.Independent(X, max_samples=min(len(X), 500))
            expl = shap.LinearExplainer(model, masker)
            return np.asarray(expl.shap_values(X)), float(np.array(expl.expected_value).ravel()[0])
        if name.startswith("XGB"):
            # I read exact TreeSHAP straight from the booster - this avoids a
            # known SHAP/XGBoost version clash and is fast and exact.
            import xgboost as xgb
            contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
            return np.asarray(contribs[:, :-1]), float(contribs[0, -1])
        try:
            expl = shap.TreeExplainer(model)
            return np.asarray(expl.shap_values(X)), float(np.array(expl.expected_value).ravel()[0])
        except Exception:
            bg = shap.utils.sample(X, 100, random_state=config.RANDOM_SEED)
            expl = shap.Explainer(model.predict, bg)
            ex = expl(X)
            return np.asarray(ex.values), float(np.array(ex.base_values).ravel()[0])

    def run(self):
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        X, _ = DataSplitter().load_xy()
        model = joblib.load(config.MODELS_DIR / "best_model.joblib")

        # I compute exact SHAP values for every row, using the explainer that
        # matches the deployed model type.
        shap_values, base_value = self._shap_values(model, X)

        # Global importance = average absolute SHAP per feature.
        importance = pd.Series(np.abs(shap_values).mean(axis=0),
                               index=X.columns).sort_values(ascending=False)
        print("Global feature importance (average £ impact on rent):")
        for feat, val in importance.head(8).items():
            print(f"  {self._friendly(feat):<28} £{val:.0f}")
        self._chart_global(importance)

        # Local explanation for one example property (the first row).
        contribs = pd.Series(shap_values[0], index=X.columns)
        top = contribs.reindex(contribs.abs().sort_values(ascending=False).index).head(6)
        prediction = base_value + contribs.sum()
        lines = [f"The model starts from a baseline of about £{base_value:.0f} and "
                 f"adjusts to **£{prediction:.0f}** for this property:"]
        for feat, val in top.items():
            lines.append(f"- {self._friendly(feat).capitalize()} "
                         f"{'increases' if val > 0 else 'decreases'} the estimate by £{abs(val):.0f}.")
        print("\nExample explanation (property #0):")
        for l in lines:
            print("  " + l.replace("**", ""))

        md = ["# SHAP Explainability - RentWise UK (Module 2)\n",
              "## Global importance (average £ impact on predicted rent)"]
        md += [f"- **{self._friendly(f)}**: £{v:.0f}" for f, v in importance.head(10).items()]
        md.append("\n## Example: explaining one prediction")
        md += lines
        md.append("\n*SHAP values are in pounds and sum to the model's prediction. "
                  "Figures from synthetic development data; regenerated on real ONS data.*")
        (config.ROOT / "reports" / "shap_explanation.md").write_text("\n".join(md))
        print("\nI saved -> reports/figures/07_shap_importance.png, reports/shap_explanation.md")


if __name__ == "__main__":
    ShapExplainer().run()
