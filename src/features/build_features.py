"""
build_features.py  —  Phase 4, Step 7: feature engineering.

I turn the cleaned table into a fully numeric, model-ready table, because models
only understand numbers. How I convert each column matters, so I keep the choices
explicit in one class.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config


class FeatureBuilder:
    """I engineer the model inputs from the cleaned data.

    Why my choices matter 
      * Ordinal encoding for property type - it has a real size order (Room<..<Detached).
      * One-hot encoding for region - it has NO order, so I make yes/no columns and
        avoid implying London > Wales.
      * Income binned into quartiles - a simple low/mid/high area signal.
      * I DROP identifiers (la_name) so the model learns patterns, not memorised areas.
    """

    TARGET = "median_monthly_rent"
    PROPERTY_ORDER = {"Room": 1, "Flat": 2, "Terraced": 3, "Semi-detached": 4, "Detached": 5}

    def __init__(self):
        self.input_path = config.DATA_PROCESSED / "rentwise_clean.csv"
        self.output_path = config.DATA_PROCESSED / "rentwise_features.csv"

    def load(self) -> pd.DataFrame:
        # I load the cleaned data produced in Phase 2.
        if not self.input_path.exists():
            raise FileNotFoundError(f"{self.input_path} not found. I need clean_data.py first.")
        return pd.read_csv(self.input_path)

    def _encode_property_type(self, df: pd.DataFrame) -> pd.DataFrame:
        # I map property type to a single ordered number using the size order.
        df = df.copy()
        df["property_type_ord"] = df["property_type"].map(self.PROPERTY_ORDER)
        return df

    def _add_income_band(self, df: pd.DataFrame) -> pd.DataFrame:
        # I split income into 4 quartile bands (1=lowest area .. 4=highest).
        df = df.copy()
        df["income_band"] = pd.qcut(df["median_annual_income"], q=4, labels=[1, 2, 3, 4]).astype(int)
        return df

    def _one_hot_region(self, df: pd.DataFrame) -> pd.DataFrame:
        # I turn region into separate 0/1 columns (no false ordering).
        dummies = pd.get_dummies(df["region"], prefix="region").astype(int)
        return pd.concat([df, dummies], axis=1)

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        # I run every feature step and keep only model-ready columns + target.
        df = self._encode_property_type(df)
        df = self._add_income_band(df)
        df = self._one_hot_region(df)
        drop_cols = ["la_code", "la_name", "region", "property_type"]
        feats = df.drop(columns=[c for c in drop_cols if c in df.columns])
        cols = [c for c in feats.columns if c != self.TARGET] + [self.TARGET]
        return feats[cols]

    def run(self) -> Path:
        # I build the features and save the model-ready file.
        feats = self.build(self.load())
        feature_cols = [c for c in feats.columns if c != self.TARGET]
        print("Feature engineering complete")
        print("-" * 34)
        print(f"Rows            : {len(feats)}")
        print(f"Feature columns : {len(feature_cols)}")
        for c in feature_cols:
            print("  -", c)
        feats.to_csv(self.output_path, index=False)
        print(f"\nI saved model-ready data -> {self.output_path}")
        return self.output_path


if __name__ == "__main__":
    FeatureBuilder().run()
