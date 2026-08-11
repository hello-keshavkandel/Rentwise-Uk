"""
clean_data.py  —  Phase 2, Step 4: cleaning & preparation.

I turn a raw/synthetic dataset into a clean, analysis-ready table. I standardise
column names, drop duplicates, fill missing values, remove rent outliers, and run
sanity checks. I do it in a documented, repeatable class so a marker can re-run it
and get exactly the same result - that reproducibility is part of the marks.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config


class DataCleaner:
    """I clean one dataset end to end and report what I changed.

    Why I clean first (I explain this in the viva): real government data is messy -
    missing cells, duplicates, the odd impossible value. A model trained on dirt
    learns the dirt, so I clean before anything else.
    """

    RENT_COL = "median_monthly_rent"

    def __init__(self, input_name: str = "rentwise_sample.csv"):
        # I remember which file I clean and where the result goes.
        self.input_path = config.DATA_PROCESSED / input_name
        self.output_path = config.DATA_PROCESSED / "rentwise_clean.csv"

    def load(self) -> pd.DataFrame:
        # I load the dataset I am going to clean. This is the ONLY place that knows
        # where data comes from - so swapping in real ONS data changes just this.
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"{self.input_path} not found. I need make_sample_data.py first.")
        return pd.read_csv(self.input_path)

    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # I make column names lower-case and underscored so they are consistent.
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        return df

    def _fill_missing(self, df: pd.DataFrame):
        # I fill gaps: median for numbers, mode (most common) for text.
        df = df.copy()
        missing_before = int(df.isna().sum().sum())
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])
        return df, missing_before

    def _remove_rent_outliers(self, df: pd.DataFrame):
        # I drop rent values outside the IQR fence (1.5x the interquartile range).
        q1, q3 = df[self.RENT_COL].quantile(0.25), df[self.RENT_COL].quantile(0.75)
        iqr = q3 - q1
        keep = df[self.RENT_COL].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return df[keep].copy(), int((~keep).sum())

    def _sanity_checks(self, df: pd.DataFrame):
        # I flag anything that should be impossible, as a final guard.
        warnings = []
        if (df[self.RENT_COL] <= 0).any():
            warnings.append("Found rent values <= 0.")
        if not df["bedrooms"].between(0, 10).all():
            warnings.append("Found bedrooms outside 0-10.")
        if (df["median_annual_income"] <= 0).any():
            warnings.append("Found income values <= 0.")
        return warnings

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # I run every cleaning step in order and print a short report.
        n_start = len(df)
        df = self._standardise_columns(df)
        dupes = int(df.duplicated().sum())
        df = df.drop_duplicates()
        df, missing_filled = self._fill_missing(df)
        df, outliers = self._remove_rent_outliers(df)
        warnings = self._sanity_checks(df)

        print("Cleaning report")
        print("-" * 32)
        print(f"Rows in                : {n_start}")
        print(f"Duplicate rows removed : {dupes}")
        print(f"Missing values filled  : {missing_filled}")
        print(f"Rent outliers removed  : {outliers}")
        print(f"Rows out               : {len(df)}")
        print(f"Sanity warnings        : {warnings if warnings else 'none'}")
        return df

    def run(self) -> Path:
        # I load, clean and save the cleaned dataset.
        clean_df = self.clean(self.load())
        clean_df.to_csv(self.output_path, index=False)
        print(f"\nI saved clean data -> {self.output_path}")
        return self.output_path


if __name__ == "__main__":
    DataCleaner().run()
