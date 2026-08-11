"""
real_data.py  —  Phase 1 (REAL data adapter).

I read the REAL open datasets (once they are placed in data/raw/) and reshape them
into the exact same columns my synthetic generator produces. That is the whole
point of my two-stage design: swap the data source here, and every downstream
step - cleaning, features, models, SHAP, affordability, dashboard - runs unchanged.

I can only run if the real files exist. If they do not, I stop with a clear message
telling exactly what to download and the simple column names I expect. Preparing
those small tidy CSVs from the ONS downloads is a genuine non-coding task a teammate
can own (Abdullah/Milan), which keeps the group contribution honest.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

# The simple, tidy files I expect in data/raw/ (prepared from the ONS/gov downloads).
EXPECTED = {
    "prms_rents.csv":    "PRMS median rents. Columns: la_code, la_name, region, bedroom_category, median_monthly_rent",
    "ashe_earnings.csv": "ASHE earnings. Columns: la_code, median_annual_income",
    "crime.csv":         "(optional) data.police.uk. Columns: la_code, crime_per_1000",
    "transport.csv":     "(optional) OSM/DfT. Columns: la_code, transport_score",
}
# I map the ONS bedroom category to my property_type / bedrooms schema.
BEDROOM_MAP = {
    "Room": ("Room", 1), "One Bedroom": ("Flat", 1), "Two Bedrooms": ("Terraced", 2),
    "Three Bedrooms": ("Semi-detached", 3), "Four or More Bedrooms": ("Detached", 4),
}


class RealDataLoader:
    """I turn the real ONS/gov files into the RentWise development schema."""

    def __init__(self):
        self.raw = config.DATA_RAW

    def _require(self, name: str) -> Path:
        # I check a required file exists, or explain exactly what to provide.
        path = self.raw / name
        if not path.exists():
            lines = ["\nI cannot find the real data files. Please place these tidy CSVs in",
                     f"  {self.raw}", "with these exact simple columns:"]
            for f, desc in EXPECTED.items():
                lines.append(f"  - {f}: {desc}")
            lines.append("\nDownload sources are listed in DATA_SOURCES.md. Prepare each file by")
            lines.append("exporting the ONS table and renaming the columns to those above.")
            raise FileNotFoundError("\n".join(lines))
        return path

    def load(self) -> pd.DataFrame:
        # I read rents + earnings (required) and crime + transport (optional).
        rents = pd.read_csv(self._require("prms_rents.csv"))
        earn = pd.read_csv(self._require("ashe_earnings.csv"))
        df = rents.merge(earn, on="la_code", how="left")

        # I attach optional location features, defaulting sensibly if absent.
        for name, col, default in [("crime.csv", "crime_per_1000", None),
                                   ("transport.csv", "transport_score", None)]:
            path = self.raw / name
            if path.exists():
                df = df.merge(pd.read_csv(path), on="la_code", how="left")
            else:
                df[col] = df.get(col, pd.Series([float("nan")] * len(df)))
        # I fill any missing optional feature with its national median (documented limitation).
        for col in ["crime_per_1000", "transport_score"]:
            if df[col].isna().all():
                df[col] = 0.5 if col == "transport_score" else 60.0
            df[col] = df[col].fillna(df[col].median())

        # I map bedroom category -> property_type + bedrooms.
        mapped = df["bedroom_category"].map(BEDROOM_MAP)
        df["property_type"] = mapped.map(lambda t: t[0] if isinstance(t, tuple) else "Flat")
        df["bedrooms"] = mapped.map(lambda t: t[1] if isinstance(t, tuple) else 1)
        df["year"] = 2024  # I stamp the release year (adjust to the data you download)

        cols = ["la_code", "la_name", "region", "year", "property_type", "bedrooms",
                "median_annual_income", "crime_per_1000", "transport_score",
                "median_monthly_rent"]
        return df[cols]

    def run(self) -> Path:
        # I write the real data into the SAME file the synthetic generator writes,
        # so the rest of the pipeline runs with no other change.
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out = config.DATA_PROCESSED / "rentwise_sample.csv"
        df = self.load()
        df.to_csv(out, index=False)
        print(f"[RealDataLoader] I wrote {len(df)} REAL rows to {out}")
        print("[RealDataLoader] Now run the rest of the pipeline (clean -> features -> models).")
        return out


if __name__ == "__main__":
    RealDataLoader().run()
