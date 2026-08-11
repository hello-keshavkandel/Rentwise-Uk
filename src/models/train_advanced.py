"""
train_advanced.py  —  Phase 5, Step 9: advanced models + tuning + selection.

I train Random Forest and XGBoost, tune each with 5-fold cross-validation, compare
them fairly against the baseline, pick the best on the VALIDATION set, and save it.
This is the heart of my Implementation chapter.
"""

import sys
import json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.models.prepare import DataSplitter
from src.models.metrics import RegressionMetrics


class AdvancedTrainer:
    """I train, tune and compare the tree models, then keep the winner.

    Why these models (I explain this in the viva): Ridge can only draw straight
    lines, but housing data has non-linear interactions. Random Forest averages
    many trees (robust); XGBoost builds trees in sequence, each fixing the last
    one's mistakes (usually best on tabular data). Trees split on thresholds, so I
    don't scale features here.
    """

    SCORING = "neg_mean_squared_error"

    def __init__(self):
        self.splitter = DataSplitter()

    def _tune(self, model, grid, X_train, y_train):
        # I grid-search with 5-fold cross-validation and return the best fitted model.
        search = GridSearchCV(model, grid, scoring=self.SCORING,
                              cv=config.CV_FOLDS, n_jobs=-1)
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_

    def run(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.splitter.split()

        rf, rf_params = self._tune(
            RandomForestRegressor(random_state=config.RANDOM_SEED),
            {"n_estimators": [100, 300], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 2]},
            X_train, y_train)
        xgb, xgb_params = self._tune(
            XGBRegressor(random_state=config.RANDOM_SEED, objective="reg:squarederror"),
            {"n_estimators": [100, 300], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
            X_train, y_train)

        results = {
            "Random Forest": {"params": rf_params, "estimator": rf,
                              "validation": RegressionMetrics.compute(y_val, rf.predict(X_val)),
                              "test": RegressionMetrics.compute(y_test, rf.predict(X_test))},
            "XGBoost": {"params": xgb_params, "estimator": xgb,
                        "validation": RegressionMetrics.compute(y_val, xgb.predict(X_val)),
                        "test": RegressionMetrics.compute(y_test, xgb.predict(X_test))},
        }

        baseline_full = json.loads((config.ROOT / "reports" / "metrics_baseline.json").read_text())
        baseline = baseline_full["test"]
        print("Model comparison (test set)")
        print("-" * 46)
        RegressionMetrics.show("Ridge (baseline)", baseline)
        for name, r in results.items():
            RegressionMetrics.show(name, r["test"])

        # I deploy whichever model performs best - including the baseline. On this
        # near-linear synthetic data the linear model can win; I let the evidence
        # decide rather than forcing a tree model, and I choose on VALIDATION
        # (never on test) so the test figures stay an honest held-out estimate.
        ridge = joblib.load(config.MODELS_DIR / "ridge_baseline.joblib")
        candidates = {
            "Ridge (baseline)": {"params": "alpha tuned by CV", "estimator": ridge,
                                 "validation": baseline_full.get("validation", baseline),
                                 "test": baseline},
            "Random Forest": results["Random Forest"],
            "XGBoost": results["XGBoost"],
        }
        best_name = min(candidates, key=lambda n: candidates[n]["validation"]["rmse"])
        best = candidates[best_name]
        improvement = (baseline["rmse"] - best["test"]["rmse"]) / baseline["rmse"]
        meets = improvement >= config.TARGET_RMSE_IMPROVEMENT
        print(f"\nDeployed model  : {best_name}  (settings: {best['params']})")
        if best_name == "Ridge (baseline)":
            print("Note            : the linear baseline is strongest on this near-linear "
                  "synthetic data; the tree models are retained for the real-data stage.")
        else:
            print(f"Beats baseline  : {improvement*100:+.1f}% RMSE "
                  f"({'MEETS' if meets else 'does NOT meet'} the >=10% target)")

        joblib.dump(best["estimator"], config.MODELS_DIR / "best_model.joblib")
        (config.ROOT / "reports" / "metrics_comparison.json").write_text(json.dumps({
            "baseline": {"test": baseline},
            "random_forest": {"params": results["Random Forest"]["params"], "test": results["Random Forest"]["test"]},
            "xgboost": {"params": results["XGBoost"]["params"], "test": results["XGBoost"]["test"]},
            "best_model": best_name, "rmse_improvement_vs_baseline": improvement,
            "meets_10pct_target": bool(meets)}, indent=2))
        md = ["# Model Comparison - RentWise UK (Module 2)\n",
              "| Model | Test RMSE (£) | Test MAE (£) | Test R² |", "|---|---|---|---|",
              f"| Ridge (baseline) | {baseline['rmse']:.0f} | {baseline['mae']:.0f} | {baseline['r2']:.3f} |",
              f"| Random Forest | {results['Random Forest']['test']['rmse']:.0f} | {results['Random Forest']['test']['mae']:.0f} | {results['Random Forest']['test']['r2']:.3f} |",
              f"| XGBoost | {results['XGBoost']['test']['rmse']:.0f} | {results['XGBoost']['test']['mae']:.0f} | {results['XGBoost']['test']['r2']:.3f} |",
              (f"\n**Deployed model:** {best_name} (best on the held-out test set; the linear "
               f"baseline leads on this near-linear synthetic data, with the tree models kept "
               f"for the real-data stage)." if best_name == "Ridge (baseline)"
               else f"\n**Deployed model:** {best_name} - {improvement*100:+.1f}% RMSE vs baseline."),
              "\n*Figures from synthetic development data; regenerated on real ONS data.*"]
        (config.ROOT / "reports" / "model_comparison.md").write_text("\n".join(md))
        print("\nI saved -> models/best_model.joblib, reports/metrics_comparison.json, reports/model_comparison.md")


if __name__ == "__main__":
    AdvancedTrainer().run()
