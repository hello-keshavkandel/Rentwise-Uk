"""
module1_sale_price.py  —  Module 1: Residential Sale Price Prediction (Abdullah).

I predict UK residential SALE prices (the sibling of Module 2's rents). I follow
the same open-data -> clean -> features -> models -> evaluate pipeline pattern, and
I add bootstrapped prediction intervals so each estimate carries an uncertainty
range. Developed on synthetic HM-Land-Registry-style data pending the real files.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.models.metrics import RegressionMetrics

PROPERTY_ORDER = {"Flat": 1, "Terraced": 2, "Semi-detached": 3, "Detached": 4}
REGIONS = ["London", "North West", "Yorkshire and The Humber", "South West",
           "West Midlands", "East Midlands", "East of England", "South East",
           "North East", "Wales"]


class SalePriceModule:
    """I build, train and evaluate the sale-price model, with prediction intervals."""

    TARGET = "sale_price"

    def __init__(self, n_per_group: int = 30):
        # I fix the seed so my synthetic data and results are reproducible.
        self.rng = np.random.default_rng(config.RANDOM_SEED)
        self.n = n_per_group
        self.fig_dir = config.ROOT / "reports" / "figures"

    def generate(self) -> pd.DataFrame:
        # I generate synthetic transactions with a transparent price formula, so a
        # good model should recover the drivers (a check while on synthetic data).
        rows = []
        region_premium = {r: (250000 if r == "London" else self.rng.normal(90000, 30000))
                          for r in REGIONS}
        for region in REGIONS:
            for _ in range(self.n):
                ptype = self.rng.choice(list(PROPERTY_ORDER))
                bedrooms = int(np.clip(PROPERTY_ORDER[ptype] + self.rng.integers(-1, 2), 1, 6))
                tenure = self.rng.choice(["Freehold", "Leasehold"])
                new_build = int(self.rng.random() < 0.1)
                year = int(self.rng.integers(2019, 2025))
                price = (120000
                         + bedrooms * 42000
                         + region_premium[region]
                         + (0 if tenure == "Freehold" else -15000)   # leasehold discount
                         + new_build * 30000                          # new-build premium
                         + (year - 2019) * 6000
                         + self.rng.normal(0, 25000))
                rows.append({"region": region, "property_type": ptype, "bedrooms": bedrooms,
                             "tenure": tenure, "new_build": new_build, "year": year,
                             "sale_price": round(max(60000, price), -3)})
        return pd.DataFrame(rows)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # I encode: property type ordinal, tenure binary, region one-hot, keep the rest.
        df = df.copy()
        df["property_type_ord"] = df["property_type"].map(PROPERTY_ORDER)
        df["is_leasehold"] = (df["tenure"] == "Leasehold").astype(int)
        df = pd.concat([df, pd.get_dummies(df["region"], prefix="region").astype(int)], axis=1)
        return df.drop(columns=["property_type", "tenure", "region"])

    def _prediction_interval(self, model, X_train, y_train, X_test, n_boot=40):
        # I bootstrap the training set to build an 80% prediction interval.
        preds = []
        for _ in range(n_boot):
            idx = self.rng.integers(0, len(X_train), len(X_train))
            m = RandomForestRegressor(n_estimators=100, random_state=int(self.rng.integers(1e6)))
            m.fit(X_train.iloc[idx], y_train.iloc[idx])
            preds.append(m.predict(X_test))
        preds = np.array(preds)
        return np.percentile(preds, 10, axis=0), np.percentile(preds, 90, axis=0)

    def run(self):
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        feats = self.build_features(self.generate())
        X = feats.drop(columns=[self.TARGET]); y = feats[self.TARGET]
        X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_SEED)

        # Baseline OLS, then tuned RF and XGBoost.
        ols = LinearRegression().fit(X_tr, y_tr)
        rf = GridSearchCV(RandomForestRegressor(random_state=config.RANDOM_SEED),
                          {"n_estimators": [150], "max_depth": [None]},
                          scoring="neg_mean_squared_error", cv=5, n_jobs=-1).fit(X_tr, y_tr).best_estimator_
        xgb = GridSearchCV(XGBRegressor(random_state=config.RANDOM_SEED, objective="reg:squarederror"),
                           {"n_estimators": [200], "max_depth": [4], "learning_rate": [0.1]},
                           scoring="neg_mean_squared_error", cv=5, n_jobs=-1).fit(X_tr, y_tr).best_estimator_

        scores = {"OLS (baseline)": RegressionMetrics.compute(y_test, ols.predict(X_test)),
                  "Random Forest": RegressionMetrics.compute(y_test, rf.predict(X_test)),
                  "XGBoost": RegressionMetrics.compute(y_test, xgb.predict(X_test))}
        print("Module 1 - Sale Price Prediction")
        print("-" * 42)
        for name, m in scores.items():
            RegressionMetrics.show(name, m)

        best_name = min(scores, key=lambda n: scores[n]["rmse"])
        best = {"OLS (baseline)": ols, "Random Forest": rf, "XGBoost": xgb}[best_name]
        print(f"\nBest model: {best_name} (R2 {scores[best_name]['r2']:.3f})")

        # Prediction interval demo on the first few test rows.
        lo, hi = self._prediction_interval(best if best_name != "OLS (baseline)" else rf, X_tr, y_tr, X_test.head(5))

        # Chart: predicted vs actual for the best model.
        yp = best.predict(X_test)
        plt.figure(figsize=(6, 6)); plt.scatter(y_test, yp, alpha=0.5, color="#2E74B5")
        lohi = [min(y_test.min(), yp.min()), max(y_test.max(), yp.max())]
        plt.plot(lohi, lohi, "r--", label="perfect"); plt.legend()
        plt.title("Module 1: predicted vs actual sale price"); plt.xlabel("Actual (£)"); plt.ylabel("Predicted (£)")
        plt.tight_layout(); chart = self.fig_dir / "m1_predicted_vs_actual.png"; plt.savefig(chart, dpi=120); plt.close()

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best, config.MODELS_DIR / "module1_best.joblib")
        md = ["# Module 1 - Sale Price Prediction (summary)\n",
              "| Model | Test RMSE (£) | Test MAE (£) | Test R² |", "|---|---|---|---|"]
        for name, m in scores.items():
            md.append(f"| {name} | {m['rmse']:,.0f} | {m['mae']:,.0f} | {m['r2']:.3f} |")
        md.append(f"\n**Best model:** {best_name}. Example 80% prediction intervals (first 5 test homes): "
                  + ", ".join(f"£{l:,.0f}-£{h:,.0f}" for l, h in zip(lo, hi)))
        md.append("\n*Figures from synthetic HM-Land-Registry-style data; regenerated on real data.*")
        (config.ROOT / "reports" / "module1_summary.md").write_text("\n".join(md))
        print(f"\nI saved -> models/module1_best.joblib, {chart.name}, reports/module1_summary.md")


if __name__ == "__main__":
    SalePriceModule().run()
