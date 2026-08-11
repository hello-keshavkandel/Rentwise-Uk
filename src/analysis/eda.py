"""
eda.py  —  Phase 3, Step 6: Exploratory Data Analysis.

Before I model, I explore. I print summary statistics, measure how each feature
relates to rent, and save four charts plus a findings file. Modelling without EDA
is guessing, so this is where I show I understand my data.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # I save charts to files instead of opening a window.
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config


class DataExplorer:
    """I explore the cleaned data and produce evidence for my Results chapter."""

    TARGET = "median_monthly_rent"

    def __init__(self):
        self.data_path = config.DATA_PROCESSED / "rentwise_clean.csv"
        self.fig_dir = config.ROOT / "reports" / "figures"
        self.summary_path = config.ROOT / "reports" / "eda_summary.md"

    def load(self) -> pd.DataFrame:
        # I load the cleaned dataset from Phase 2.
        if not self.data_path.exists():
            raise FileNotFoundError(f"{self.data_path} not found. I need clean_data.py first.")
        return pd.read_csv(self.data_path)

    def summarise(self, df: pd.DataFrame):
        # I print the shape and the numeric summary statistics.
        print(f"Rows: {len(df)}   Columns: {len(df.columns)}")
        print("\nSummary statistics (numeric):")
        print(df.select_dtypes("number").describe().round(1).to_string())

    def correlations(self, df: pd.DataFrame) -> pd.Series:
        # I measure how strongly each numeric feature moves with rent.
        num = df.select_dtypes("number")
        corr = num.corr()[self.TARGET].drop(self.TARGET).sort_values(ascending=False)
        print("\nCorrelation of each feature with rent:")
        print(corr.round(3).to_string())
        return corr

    def _chart_distribution(self, df):
        plt.figure(figsize=(7, 4))
        plt.hist(df[self.TARGET], bins=25, color="#2E74B5", edgecolor="white")
        plt.title("Distribution of monthly rent"); plt.xlabel("Monthly rent (£)")
        plt.ylabel("Number of records"); plt.tight_layout()
        out = self.fig_dir / "01_rent_distribution.png"; plt.savefig(out, dpi=120); plt.close()
        return out

    def _chart_by_type(self, df):
        types = df["property_type"].unique()
        data = [df.loc[df["property_type"] == t, self.TARGET] for t in types]
        plt.figure(figsize=(7, 4)); plt.boxplot(data, tick_labels=types)
        plt.title("Rent by property type"); plt.xlabel("Property type")
        plt.ylabel("Monthly rent (£)"); plt.tight_layout()
        out = self.fig_dir / "02_rent_by_property_type.png"; plt.savefig(out, dpi=120); plt.close()
        return out

    def _chart_vs_income(self, df):
        plt.figure(figsize=(7, 4))
        plt.scatter(df["median_annual_income"], df[self.TARGET], alpha=0.5, color="#1E7D32")
        plt.title("Rent vs local income"); plt.xlabel("Local median annual income (£)")
        plt.ylabel("Monthly rent (£)"); plt.tight_layout()
        out = self.fig_dir / "03_rent_vs_income.png"; plt.savefig(out, dpi=120); plt.close()
        return out

    def _chart_heatmap(self, df):
        corr = df.select_dtypes("number").corr()
        plt.figure(figsize=(6.5, 5.5))
        im = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr)), corr.columns)
        for i in range(len(corr)):
            for j in range(len(corr)):
                plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        plt.title("Correlation heatmap (numeric features)"); plt.tight_layout()
        out = self.fig_dir / "04_correlation_heatmap.png"; plt.savefig(out, dpi=120); plt.close()
        return out

    def _write_summary(self, df, corr, charts):
        # I write a short, plain-English findings file for my Results chapter.
        lines = ["# EDA Summary - RentWise UK (Module 2)\n",
                 f"- Records analysed: **{len(df)}**",
                 f"- Average monthly rent: **£{df[self.TARGET].mean():.0f}** "
                 f"(min £{df[self.TARGET].min():.0f}, max £{df[self.TARGET].max():.0f})\n",
                 "## Strongest drivers of rent (by correlation)"]
        for feat, val in corr.items():
            lines.append(f"- **{feat}**: {val:+.2f} - as it rises, rent "
                         f"{'increases' if val > 0 else 'decreases'}.")
        lines.append("\n## Charts")
        lines += [f"- {Path(c).name}" for c in charts]
        lines.append("\n*Figures from synthetic development data; regenerated on real ONS data.*")
        self.summary_path.write_text("\n".join(lines))

    def run(self):
        # I run the whole exploration and save charts + summary.
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        df = self.load()
        self.summarise(df)
        corr = self.correlations(df)
        charts = [self._chart_distribution(df), self._chart_by_type(df),
                  self._chart_vs_income(df), self._chart_heatmap(df)]
        self._write_summary(df, corr, charts)
        print("\nI saved charts to:", self.fig_dir)


if __name__ == "__main__":
    DataExplorer().run()
