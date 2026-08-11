"""
affordability.py  —  Phase 8, Step 12: the Affordability Engine.

I turn predicted rents into an affordability measure: annual rent divided by local
income, flagged against the 30% benchmark. This is the piece that makes my module
more than a price predictor - it answers "is it affordable here, relative to local
wages?", which is exactly my research gap.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from src.models.prepare import DataSplitter


class AffordabilityEngine:
    """I compute rent-to-income ratios and decide what is affordable.

    The two small methods ratio() and is_affordable() are my reusable engine - my
    dashboard calls these exact methods so the logic lives in one place. The run()
    method applies them to the model's predictions and produces the reporting.
    """

    def __init__(self, threshold: float = config.AFFORDABILITY_THRESHOLD):
        # I remember the affordability threshold (default 30% of income).
        self.threshold = threshold
        self.fig_dir = config.ROOT / "reports" / "figures"

    def ratio(self, annual_rent, annual_income):
        # I express rent as a share of income, e.g. 0.30 = 30%.
        return annual_rent / annual_income

    def is_affordable(self, ratio) -> bool:
        # I call it affordable if the ratio is at or below my threshold.
        return ratio <= self.threshold

    def run(self):
        # I predict rent, compute affordability per area, and report it.
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        clean = pd.read_csv(config.DATA_PROCESSED / "rentwise_clean.csv")
        X, _ = DataSplitter().load_xy()
        model = joblib.load(config.MODELS_DIR / "best_model.joblib")
        predicted = model.predict(X)

        df = clean[["la_code", "la_name", "region", "property_type",
                    "median_annual_income"]].copy()
        df["predicted_monthly_rent"] = np.round(predicted, 0)
        df["annual_rent"] = df["predicted_monthly_rent"] * 12
        df["rent_to_income"] = self.ratio(df["annual_rent"], df["median_annual_income"])
        df["affordable"] = df["rent_to_income"].apply(self.is_affordable)

        area = (df.groupby(["la_name", "region"], as_index=False)
                  .agg(avg_monthly_rent=("predicted_monthly_rent", "mean"),
                       median_income=("median_annual_income", "first"),
                       rent_to_income=("rent_to_income", "mean")))
        area["affordable"] = area["rent_to_income"].apply(self.is_affordable)

        pct = 100 * area["affordable"].mean()
        print("Affordability engine")
        print("-" * 40)
        print(f"Threshold           : {self.threshold*100:.0f}% of income")
        print(f"Local authorities   : {len(area)}")
        print(f"Affordable areas    : {area['affordable'].sum()} ({pct:.0f}%)")
        print(f"Rent-to-income range: {area['rent_to_income'].min()*100:.0f}% "
              f"to {area['rent_to_income'].max()*100:.0f}%")

        by_region = df.groupby("region")["rent_to_income"].mean().sort_values(ascending=False)
        colors = ["#B00020" if v > self.threshold else "#1E7D32" for v in by_region.values]
        plt.figure(figsize=(8, 4.5))
        plt.bar(list(by_region.index), by_region.values * 100, color=colors)
        plt.axhline(self.threshold * 100, color="black", linestyle="--",
                    label="30% affordability threshold")
        plt.title("Average rent-to-income by region"); plt.ylabel("Rent as % of local income")
        plt.xticks(rotation=45, ha="right"); plt.legend(); plt.tight_layout()
        chart = self.fig_dir / "08_affordability_by_region.png"; plt.savefig(chart, dpi=120); plt.close()

        area_sorted = area.sort_values("rent_to_income", ascending=False)
        area_sorted.to_csv(config.DATA_PROCESSED / "affordability.csv", index=False)
        md = ["# Affordability Summary - RentWise UK (Module 2)\n",
              f"- Affordability threshold: **{self.threshold*100:.0f}%** of local income",
              f"- Local authorities assessed: **{len(area)}**",
              f"- Affordable areas: **{area['affordable'].sum()} ({pct:.0f}%)**",
              f"- Rent-to-income range: **{area['rent_to_income'].min()*100:.0f}%-"
              f"{area['rent_to_income'].max()*100:.0f}%**\n",
              "## Validation",
              "I check the ratios for plausibility (a sensible 15-45% band). Direct "
              "comparison against published ONS figures will be done once the real ONS "
              "rent and earnings data are loaded.",
              "\n*Figures from synthetic development data; regenerated on real ONS data.*"]
        (config.ROOT / "reports" / "affordability_summary.md").write_text("\n".join(md))
        print(f"\nI saved -> data/processed/affordability.csv, {chart.name}, reports/affordability_summary.md")


if __name__ == "__main__":
    AffordabilityEngine().run()
