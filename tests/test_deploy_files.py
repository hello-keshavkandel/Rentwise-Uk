"""Deployment guard: every file the live app/API reads must be shippable by git.

Streamlit Cloud and Docker build from the repository, so a file that git ignores is
simply absent in production (this caught the Area map's affordability.csv)."""
import shutil
import subprocess

import pytest

import config

DEPLOY_FILES = [
    "data/processed/rentwise_clean.csv", "data/processed/rentwise_features.csv",
    "data/processed/rentwise.duckdb", "data/processed/affordability.csv",
    "data/processed/data_vintages.json", "data/processed/la_spatial_features.csv",
    "data/processed/lad_boundaries_wgs84.geojson",
    "data/processed/official_rents.csv", "data/processed/rent_trends.parquet",
    "data/processed/rent_areas.csv", "data/processed/sale_prices.csv", "models/rent_forecasts.json",
    "models/rent_xgb_spatial.json", "models/rent_model_meta.json", "models/conformal.json",
    "models/best_model.joblib",
]

pytestmark = pytest.mark.skipif(shutil.which("git") is None or not (config.ROOT / ".git").exists(),
                                reason="not a git checkout")


def _git(*args):
    return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True)


@pytest.mark.parametrize("rel", DEPLOY_FILES)
def test_deploy_file_is_shippable(rel):
    assert (config.ROOT / rel).exists(), f"{rel} missing locally — run the pipeline"
    tracked = _git("ls-files", "--error-unmatch", rel).returncode == 0
    ignored = _git("check-ignore", "-q", rel).returncode == 0
    assert tracked or not ignored, f"{rel} is git-ignored, so production will not have it"
