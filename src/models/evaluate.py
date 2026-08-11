"""
evaluate.py  —  Phase 6, Step 10: Evaluation & Testing.

I evaluate my best model honestly: I check the success criteria on the held-out
test set, draw diagnostic charts, and run cross-regional (leave-one-region-out)
validation to prove the model generalises across the UK rather than overfitting
one area.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.base import clone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.models.prepare import DataSplitter
from src.models.metrics import RegressionMetrics


class ModelEvaluator:
    """I test the best model and check it works everywhere, not just on average."""

    def __init__(self):
        self.splitter = DataSplitter()
        self.fig_dir = config.ROOT / "reports" / "figures"

    def _region_labels(self, X: pd.DataFrame) -> pd.Series:
        # I rebuild each row's region name from its one-hot 'region_*' columns.
        region_cols = [c for c in X.columns if c.startswith("region_")]
        return X[region_cols].idxmax(axis=1).str.replace("region_", "", regex=False)

    def _chart_pred_vs_actual(self, y_true, y_pred):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, color="#2E74B5")
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        plt.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
        plt.title("Predicted vs actual rent (test set)"); plt.xlabel("Actual rent (£)")
        plt.ylabel("Predicted rent (£)"); plt.legend(); plt.tight_layout()
        out = self.fig_dir / "05_predicted_vs_actual.png"; plt.savefig(out, dpi=120); plt.close()
        return out

    def _chart_residuals(self, y_true, y_pred):
        residuals = y_true - y_pred
        plt.figure(figsize=(7, 4))
        plt.scatter(y_pred, residuals, alpha=0.6, color="#8E44AD")
        plt.axhline(0, color="red", linestyle="--")
        plt.title("Residuals (error) vs predicted rent"); plt.xlabel("Predicted rent (£)")
        plt.ylabel("Actual - predicted (£)"); plt.tight_layout()
        out = self.fig_dir / "06_residuals.png"; plt.savefig(out, dpi=120); plt.close()
        return out

    def _cross_regional(self, best_model):
        # I hold out each region, train on the rest, and predict the unseen region.
        X, y = self.splitter.load_xy()
        regions = self._region_labels(X)
        rows = []
        for r in sorted(regions.unique()):
            is_r = regions == r
            model = clone(best_model)               # fresh copy, same settings
            model.fit(X[~is_r], y[~is_r])
            m = RegressionMetrics.compute(y[is_r], model.predict(X[is_r]))
            m["region"] = r; m["n"] = int(is_r.sum())
            rows.append(m)
        return pd.DataFrame(rows)[["region", "n", "rmse", "mae", "r2"]]

    def run(self):
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        _, _, X_test, _, _, y_test = self.splitter.split()
        best = joblib.load(config.MODELS_DIR / "best_model.joblib")
        baseline = json.loads((config.ROOT / "reports" / "metrics_baseline.json").read_text())["test"]

        test = RegressionMetrics.compute(y_test, best.predict(X_test))
        improvement = (baseline["rmse"] - test["rmse"]) / baseline["rmse"]
        print("1) Held-out TEST performance")
        print("-" * 40)
        RegressionMetrics.show("Best model", test)
        print(f"   R2 >= 0.80 target : {'PASS' if test['r2'] >= config.TARGET_R2 else 'FAIL'}")
        print(f"   Beat baseline 10% : {improvement*100:+.1f}% "
              f"({'PASS' if improvement >= config.TARGET_RMSE_IMPROVEMENT else 'not met'})")

        y_pred = best.predict(X_test)
        self._chart_pred_vs_actual(y_test, y_pred)
        self._chart_residuals(y_test, y_pred)

        cr = self._cross_regional(best)
        print("\n2) Cross-regional validation (leave-one-region-out)")
        print("-" * 40)
        print(cr.round(2).to_string(index=False))
        print(f"\n   RMSE across regions: mean £{cr['rmse'].mean():.0f}, "
              f"spread £{cr['rmse'].min():.0f}-£{cr['rmse'].max():.0f}")

        md = ["# Cross-Regional Validation - RentWise UK (Module 2)\n",
              "I hold each region out, train on the rest, then predict the unseen region.\n",
              "| Region | n | RMSE (£) | MAE (£) | R² |", "|---|---|---|---|---|"]
        for _, row in cr.iterrows():
            md.append(f"| {row['region']} | {int(row['n'])} | {row['rmse']:.0f} | {row['mae']:.0f} | {row['r2']:.3f} |")
        md.append(f"\n**RMSE across regions:** mean £{cr['rmse'].mean():.0f} "
                  f"(range £{cr['rmse'].min():.0f}-£{cr['rmse'].max():.0f}). A tight range "
                  "means the model generalises well across the UK.")
        md.append("\n*Figures from synthetic development data; regenerated on real ONS data.*")
        (config.ROOT / "reports" / "cross_regional.md").write_text("\n".join(md))
        (config.ROOT / "reports" / "evaluation.json").write_text(json.dumps({
            "test": test, "baseline_test": baseline,
            "rmse_improvement_vs_baseline": improvement,
            "meets_r2_target": bool(test["r2"] >= config.TARGET_R2),
            "cross_regional_rmse_mean": float(cr["rmse"].mean()),
            "cross_regional_rmse_min": float(cr["rmse"].min()),
            "cross_regional_rmse_max": float(cr["rmse"].max())}, indent=2))
        print("\nI saved -> reports/figures/05_,06_, reports/cross_regional.md, reports/evaluation.json")


if __name__ == "__main__":
    ModelEvaluator().run()
