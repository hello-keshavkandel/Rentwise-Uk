"""
run_module1_enriched.py  —  standalone runner for the EPC-enriched Module 1.

Thin wrapper around the shared trainer `src/modules/module1_model.train_and_report`
(the same code `run_real_pipeline.py` uses). It loads Price Paid, the EPC lookup
(if built) and a region map, trains, and prints the baseline-vs-enriched result.

Run:
    python -m src.data.epc          # build data/processed/epc_lookup.parquet first
    python run_module1_enriched.py
"""

from __future__ import annotations

import re
import sys
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.modules.module1_model import train_and_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("rentwise.module1")


def _price_paid() -> Path | None:
    for name in ("land_registry_pp_2026.csv", "hm_land_registry_price_paid_latest.csv"):
        p = config.DATA_RAW / name
        if p.exists():
            return p
    return None


def _epc_lookup():
    for ext in (".parquet", ".csv"):
        p = (config.DATA_PROCESSED / "epc_lookup").with_suffix(ext)
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p, dtype={"house_no": str})
    return None


def _region_map():
    try:
        clean = pd.read_csv(config.DATA_PROCESSED / "rentwise_clean.csv")
        def _norm(s):
            s = str(s).lower().replace(" ua", "").replace("city of", "").replace("county of", "")
            return re.sub(r"[^a-z0-9]+", " ", s).strip()
        return {_norm(n): r for n, r in zip(clean.la_name, clean.region)}
    except Exception:
        return None


def main() -> None:
    pp = _price_paid()
    if pp is None:
        log.error("No Price Paid file in %s.", config.DATA_RAW); sys.exit(1)
    epc = _epc_lookup()
    if epc is None:
        log.warning("No EPC lookup found — running BASELINE only. Build it with "
                    "`python -m src.data.epc` after adding EPC data to data/raw/epc/.")
    res = train_and_report(pp, epc_lookup=epc, reports_dir=config.ROOT / "reports",
                           seed=config.RANDOM_SEED, region_map=_region_map())
    print("\n--- Module 1 result ---")
    print(f"Best model: {res['best']}  (R² {res['best_r2']:.3f})  on {res['n_modelled']:,} sales")
    if res["ab"]:
        ab = res["ab"]
        print(f"EPC lift: R² {ab['baseline_r2']:.3f} -> {ab['enriched_r2']:.3f} "
              f"(+{ab['enriched_r2']-ab['baseline_r2']:.3f}); match rate {100*ab['match_rate']:.0f}%")
    if res["interval"]:
        iv = res["interval"]
        print(f"90% conformal interval: ±£{iv['half_width']:,.0f}  (coverage {100*iv['coverage']:.0f}%)")
    print(f"Summary written to reports/module1_summary.md")


if __name__ == "__main__":
    main()
