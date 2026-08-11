"""
prepare.py  —  shared data preparation for modelling (used by Steps 8, 9 & 10).

I keep the loading and the train/validation/test split in one class so every
model step trains and tests on exactly the same, reproducible split.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config


class DataSplitter:
    """I load the model-ready features and split them into train/val/test.

    Why I split three ways (I explain this in the viva):
      * TRAIN (70%)      - the model learns from this.
      * VALIDATION (15%) - I compare and tune models on this, never on test.
      * TEST (15%)       - I lock this away and use it once, for the honest score.
    A fixed random seed makes my split reproducible.
    """

    TARGET = "median_monthly_rent"

    def __init__(self):
        # I remember where the features live and the split proportions.
        self.features_path = config.DATA_PROCESSED / "rentwise_features.csv"
        self.test_size = config.TEST_SIZE
        self.val_size = config.VALIDATION_SIZE
        self.seed = config.RANDOM_SEED

    def load_xy(self):
        # I load the file and separate the inputs (X) from the target (y).
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"{self.features_path} not found. I need build_features.py to run first.")
        df = pd.read_csv(self.features_path)
        X = df.drop(columns=[self.TARGET])
        y = df[self.TARGET]
        return X, y

    def split(self):
        # I return X_train, X_val, X_test, y_train, y_val, y_test (70/15/15).
        X, y = self.load_xy()

        # First I peel off the TEST set (15%).
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed)

        # Then I take VALIDATION out of what's left, so it's 15% of the whole.
        val_fraction = self.val_size / (1 - self.test_size)  # ~0.176
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=self.seed)
        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # I run a quick check of the split sizes.
    splitter = DataSplitter()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split()
    print("Data split (rows):")
    print(f"  train      : {len(X_train)}")
    print(f"  validation : {len(X_val)}")
    print(f"  test       : {len(X_test)}")
    print(f"  features   : {X_train.shape[1]}")
