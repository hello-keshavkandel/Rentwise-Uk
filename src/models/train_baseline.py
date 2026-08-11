"""
train_baseline.py  —  Phase 5, Step 8: the baseline model (Ridge Regression).

I train a Ridge Regression as my simplest sensible model, so my stronger models
have a fair benchmark to beat. My success rule is that the best model must beat
this baseline's RMSE by at least 10%.
"""

import sys
import json
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.models.prepare import DataSplitter
from src.models.metrics import RegressionMetrics


class BaselineTrainer:
    """I build, train and save the Ridge baseline.

    Why Ridge (I explain this in the viva): it is linear (easy to explain), fast,
    and 'regularised' - the alpha setting shrinks the weights so it doesn't
    overreact to any single feature. I scale features first because linear models
    are sensitive to scale (income is in tens of thousands, transport is 0-1).
    """

    def __init__(self):
        self.splitter = DataSplitter()
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=config.RANDOM_SEED)),
        ])

    def run(self):
        # I split the data, train on TRAIN, and score on validation and test.
        X_train, X_val, X_test, y_train, y_val, y_test = self.splitter.split()
        self.model.fit(X_train, y_train)
        val = RegressionMetrics.compute(y_val, self.model.predict(X_val))
        test = RegressionMetrics.compute(y_test, self.model.predict(X_test))

        print("Baseline model: Ridge Regression")
        print("-" * 40)
        RegressionMetrics.show("Validation", val)
        RegressionMetrics.show("Test", test)

        # I save the model and its scores so my advanced models can compare.
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        (config.ROOT / "reports").mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, config.MODELS_DIR / "ridge_baseline.joblib")
        (config.ROOT / "reports" / "metrics_baseline.json").write_text(
            json.dumps({"model": "Ridge (baseline)", "validation": val, "test": test}, indent=2))
        print("\nI saved -> models/ridge_baseline.joblib and reports/metrics_baseline.json")
        print("This test RMSE is the number my advanced models must beat by >=10%.")


if __name__ == "__main__":
    BaselineTrainer().run()
