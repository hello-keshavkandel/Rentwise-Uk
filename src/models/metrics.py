"""
metrics.py  —  shared scoring, used by every model step.

I put all my regression scoring in one small class so that the baseline, the
advanced models and the evaluator all measure success in exactly the same way.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


class RegressionMetrics:
    """I compute and present the three metrics from my success criteria.

    Why these three (I explain this in the viva):
      * RMSE - typical error in £, but I let it punish big misses harder.
      * MAE  - typical error in £, treating every miss equally.
      * R2   - 0..1, the share of rent variation I actually explain (target >= 0.80).
    I report all three because together they give an honest, rounded picture.
    """

    @staticmethod
    def compute(y_true, y_pred) -> dict:
        # I turn predictions vs truth into a small dictionary of scores.
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # sqrt(MSE) = RMSE
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"rmse": rmse, "mae": mae, "r2": r2}

    @staticmethod
    def show(name: str, metrics: dict) -> None:
        # I print one model's scores on a single tidy line.
        print(f"{name:<22} RMSE £{metrics['rmse']:.0f}   "
              f"MAE £{metrics['mae']:.0f}   R2 {metrics['r2']:.3f}")
