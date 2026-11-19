"""
Integration tests for the shared Module 1 trainer (src/modules/module1_model.py).

These run on the REAL Price Paid file when it is present (skipped otherwise, so CI
without the data still passes). They exercise the whole path end to end:
load -> (EPC join) -> encode -> train/compare -> conformal intervals -> summary.

The EPC-enriched case uses a SYNTHETIC floor-area proxy (floor area correlated with
price) purely to prove the enrichment MECHANISM lifts accuracy; the real magnitude
comes from real EPC data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from src.features import property_attributes as pa
from src.modules import module1_model as m1

PP = None
for _name in ("land_registry_pp_2026.csv", "hm_land_registry_price_paid_latest.csv"):
    if (config.DATA_RAW / _name).exists():
        PP = config.DATA_RAW / _name
        break

pytestmark = pytest.mark.skipif(PP is None, reason="HM Land Registry Price Paid not present")

SAMPLE = 8000


def _synthetic_epc(pp_path):
    """A small synthetic EPC lookup keyed to real sales (floor area ~ price proxy)."""
    df = pa.load_price_paid(pp_path).dropna(subset=["house_no"]).drop_duplicates(["pc_key", "house_no"])
    rng = np.random.default_rng(0)
    base = {"Flat": 55, "Terraced": 85, "Semi-detached": 95, "Detached": 135}
    fa = (df["ptype"].map(base).astype(float) + 0.00012 * df["price"]
          + rng.normal(0, 15, len(df))).clip(25, 400)
    return pd.DataFrame({"pc_key": df["pc_key"].values, "house_no": df["house_no"].values,
                         "total_floor_area": fa.round(1).values,
                         "number_habitable_rooms": (fa / 22).round().clip(1, 12).astype(int).values})


def test_baseline_end_to_end(tmp_path):
    res = m1.train_and_report(PP, epc_lookup=None, reports_dir=tmp_path,
                              seed=42, sample_n=SAMPLE, make_figures=False)
    assert res["use_epc"] is False
    assert 0.35 < res["best_r2"] < 0.85            # sane sale-price R²
    assert res["interval"]["coverage"] == pytest.approx(0.90, abs=0.08)  # conformal calibrated
    assert (tmp_path / "module1_summary.md").exists()


def test_epc_enrichment_lifts_accuracy(tmp_path):
    epc = _synthetic_epc(PP)
    res = m1.train_and_report(PP, epc_lookup=epc, reports_dir=tmp_path,
                              seed=42, sample_n=SAMPLE, make_figures=False)
    assert res["use_epc"] is True
    assert res["match_rate"] > 0.3                 # join actually matched
    assert res["ab"] is not None
    # floor area must improve the model (mechanism check)
    assert res["ab"]["enriched_r2"] > res["ab"]["baseline_r2"] + 0.02
    assert "EPC" in (tmp_path / "module1_summary.md").read_text()
