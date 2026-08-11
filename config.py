"""
config.py — central settings for RentWise UK.

WHY this file exists:
Every script needs to know where the data lives and what the key numbers are
(e.g. the 30% affordability threshold). Putting them in ONE place means we
never hard-code the same value in five files and get inconsistencies.
Change a setting here and the whole project follows.
"""

from pathlib import Path

# ---- Folder paths (built relative to this file, so they work on any computer) ----
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"            # untouched downloaded files
DATA_PROCESSED = ROOT / "data" / "processed"  # cleaned/merged files we create
MODELS_DIR = ROOT / "models"                 # saved trained models

# ---- Project constants (the numbers from our specification) ----
AFFORDABILITY_THRESHOLD = 0.30   # rent above 30% of income = "unaffordable"
RANDOM_SEED = 42                 # fixes randomness so results are reproducible

# ---- Data mode (governs run_pipeline.py, the synthetic development build) ----
# "synthetic" = generate development data (default, instant).
# For the REAL build use run_real_pipeline.py, which reads the real ONS/HM Land
# Registry/police files in data/raw/ and produces the results in the dissertation
# (see DATA_SOURCES.md). run_real_pipeline.py does not depend on DATA_MODE.
DATA_MODE = "synthetic"
TEST_SIZE = 0.15                 # 15% of data held out for final testing
VALIDATION_SIZE = 0.15           # 15% for validation/tuning
CV_FOLDS = 5                     # 5-fold cross-validation

# ---- Success criteria (from supervisor feedback) ----
TARGET_R2 = 0.80                 # want test R-squared >= 0.80
TARGET_RMSE_IMPROVEMENT = 0.10   # best model >=10% better RMSE than baseline

# ---- Scope (England & Wales, by region) ----
REGIONS = [
    "North East", "North West", "Yorkshire and The Humber",
    "East Midlands", "West Midlands", "East of England",
    "London", "South East", "South West", "Wales",
]

if __name__ == "__main__":
    # Running `python config.py` prints the settings — a quick sanity check.
    print("RentWise UK config")
    print("Raw data folder     :", DATA_RAW)
    print("Processed data folder:", DATA_PROCESSED)
    print("Affordability limit :", AFFORDABILITY_THRESHOLD)
    print("Regions covered     :", len(REGIONS))
