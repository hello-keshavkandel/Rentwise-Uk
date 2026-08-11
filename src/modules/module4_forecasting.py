"""
module4_forecasting.py  —  Module 4: Regional Trend Forecasting (Harish).

I forecast where regional UK rents are heading. I build a monthly regional rental
index, then fit two classical forecasters - ARIMA and Holt-Winters (ETS) - and
measure each with MAPE on a held-out 12-month test period. (Prophet is the spec's
alternative second model; the code is structured so it can be swapped in for ETS.)
Developed on synthetic ONS-IPHRP-style monthly data.
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

warnings.filterwarnings("ignore")
REGIONS = ["London", "North West", "Yorkshire and The Humber", "South West",
           "West Midlands", "East Midlands", "East of England", "South East",
           "North East", "Wales"]


class ForecastingModule:
    """I build a rental index and forecast it with ARIMA and Holt-Winters."""

    def __init__(self):
        self.rng = np.random.default_rng(config.RANDOM_SEED)
        self.fig_dir = config.ROOT / "reports" / "figures"

    def build_series(self) -> pd.DataFrame:
        # I make a monthly index (Jan 2015=100) per region: gentle upward trend +
        # a seasonal peak in late summer (academic-year demand) + noise.
        months = pd.date_range("2015-01-01", "2024-12-01", freq="MS")
        t = np.arange(len(months))
        data = {}
        for r in REGIONS:
            trend = 100 + t * (0.35 if r == "London" else 0.22)
            seasonal = 3 * np.sin(2 * np.pi * (months.month - 9) / 12)  # peak ~Sep
            noise = self.rng.normal(0, 0.8, len(months))
            data[r] = trend + seasonal + noise
        return pd.DataFrame(data, index=months)

    @staticmethod
    def _mape(actual, pred):
        return float(np.mean(np.abs((actual - pred) / actual)) * 100)

    def _forecast_region(self, series):
        # I hold out the last 12 months, fit both models on the rest, and score MAPE.
        train, test = series[:-12], series[-12:]
        arima = ARIMA(train, order=(1, 1, 1)).fit()
        arima_fc = arima.forecast(12)
        ets = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
        ets_fc = ets.forecast(12)
        return {"arima_mape": self._mape(test.values, arima_fc.values),
                "ets_mape": self._mape(test.values, ets_fc.values),
                "test": test, "arima_fc": arima_fc, "ets_fc": ets_fc}

    def run(self):
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        df = self.build_series()
        rows = []
        example = None
        for r in REGIONS:
            res = self._forecast_region(df[r])
            rows.append({"region": r, "ARIMA_MAPE": res["arima_mape"], "ETS_MAPE": res["ets_mape"]})
            if r == "London":
                example = (df[r], res)
        table = pd.DataFrame(rows)

        print("Module 4 - Regional Trend Forecasting")
        print("-" * 42)
        print(table.round(2).to_string(index=False))
        print(f"\nMean MAPE  ARIMA {table['ARIMA_MAPE'].mean():.2f}%   ETS {table['ETS_MAPE'].mean():.2f}%")

        # Chart: London actual vs 12-month forecasts.
        series, res = example
        plt.figure(figsize=(9, 4))
        plt.plot(series.index, series.values, color="#555", label="actual index")
        plt.plot(res["test"].index, res["arima_fc"].values, "--", color="#2E74B5", label="ARIMA forecast")
        plt.plot(res["test"].index, res["ets_fc"].values, "--", color="#1E7D32", label="Holt-Winters forecast")
        plt.title("Module 4: London rental index - 12-month forecast vs actual")
        plt.ylabel("Rental price index (Jan 2015=100)"); plt.legend(); plt.tight_layout()
        chart = self.fig_dir / "m4_forecast_london.png"; plt.savefig(chart, dpi=120); plt.close()

        md = ["# Module 4 - Regional Trend Forecasting (summary)\n",
              "| Region | ARIMA MAPE (%) | Holt-Winters MAPE (%) |", "|---|---|---|"]
        for _, r in table.iterrows():
            md.append(f"| {r['region']} | {r['ARIMA_MAPE']:.2f} | {r['ETS_MAPE']:.2f} |")
        md.append(f"\n**Mean MAPE:** ARIMA {table['ARIMA_MAPE'].mean():.2f}%, "
                  f"Holt-Winters {table['ETS_MAPE'].mean():.2f}% (lower is better).")
        md.append("\n*Prophet is the specification's alternative second model; ETS is used here "
                  "for a clean install, and the code is structured to swap Prophet in.*")
        md.append("\n*Figures from synthetic ONS-IPHRP-style monthly data; regenerated on real data.*")
        (config.ROOT / "reports" / "module4_summary.md").write_text("\n".join(md))
        print(f"\nI saved -> {chart.name}, reports/module4_summary.md")


if __name__ == "__main__":
    ForecastingModule().run()
