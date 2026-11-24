"""
sources.py  —  the RentWise UK data acquisition registry (Milestone 2).

One place that knows every open dataset the system uses: where it comes from, what
file it lands as, how to validate it, and a provenance record. Replaces the ad-hoc
download script with something reproducible and checkable.

Usage
-----
    python -m src.data.sources fetch     # download everything it can (needs network)
    python -m src.data.sources validate  # sanity-check whatever is on disk
    python -m src.data.sources manifest  # (re)write data/raw/_manifest.json

Each source carries a `validate` function that runs on the local file, so a broken
or stale government release fails loudly instead of silently poisoning a model.
Downloads are skipped if the file already exists (use force=True to refresh); ONS
file names change per release, so those URLs may need updating — the fetcher says
so clearly rather than crashing.
"""

from __future__ import annotations

import csv
import json
import time
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("rentwise.sources")

UA = {"User-Agent": "RentWiseUK/2.0 (+https://github.com/hello-keshavkandel/Rentwise-Uk)"}


# --------------------------------------------------------------------- helpers
def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _csv_rows(path: Path, cap: int = 5_000_000) -> int:
    n = 0
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
            if n >= cap:
                break
    return n


def _xlsx_sheets(path: Path) -> list[str]:
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    try:
        return list(wb.sheetnames)
    finally:
        wb.close()


# ------------------------------------------------------------------- validators
def v_price_paid(p: Path) -> dict:
    with open(p, newline="") as f:
        first = next(csv.reader(f), [])
    ok = len(first) >= 16
    rows = _csv_rows(p)
    return {"ok": ok and rows > 1000, "rows": rows,
            "detail": f"{len(first)} cols, {rows:,} rows"}


def v_pipr_monthly(p: Path) -> dict:
    sheets = _xlsx_sheets(p)
    ok = any(s.strip().lower() in ("table 1", "1a") for s in sheets)
    return {"ok": ok, "rows": 0, "detail": f"sheets={sheets[:6]}"}


def v_pipr_historical(p: Path) -> dict:
    sheets = _xlsx_sheets(p)
    ok = any(s.strip().lower().startswith("table 3") or s.strip() == "3" for s in sheets)
    return {"ok": ok, "rows": 0, "detail": f"sheets={sheets[:6]}"}


def v_ashe(p: Path) -> dict:
    sheets = _xlsx_sheets(p)
    ok = any(s.strip().lower() == "all" for s in sheets)
    return {"ok": ok, "rows": 0, "detail": f"sheets={sheets[:8]}"}


def v_lookup(p: Path) -> dict:
    with open(p, newline="", encoding="utf-8-sig") as f:   # utf-8-sig strips a BOM
        header = next(csv.reader(f), [])
    cols = {c.strip().upper().lstrip("﻿") for c in header}
    ok = "LSOA21CD" in cols and "LAD24CD" in cols
    rows = _csv_rows(p)
    return {"ok": ok and rows > 1000, "rows": rows, "detail": f"cols={sorted(cols)[:6]}"}


def v_police_dir(p: Path) -> dict:
    files = list(p.glob("**/*street.csv")) if p.exists() else []
    return {"ok": len(files) > 0, "rows": len(files),
            "detail": f"{len(files)} street.csv files"}


def _csv_header(p: Path) -> set:
    with open(p, newline="", encoding="utf-8-sig") as f:
        return {c.strip().lower() for c in next(csv.reader(f), [])}


def v_osm(p: Path) -> dict:                      # Module 3 (optional)
    cols = _csv_header(p)
    return {"ok": {"lad_code", "category"} <= cols, "rows": _csv_rows(p),
            "detail": f"cols={sorted(cols)[:6]}"}


def v_ofsted(p: Path) -> dict:                   # Module 3 (optional)
    cols = _csv_header(p)
    ok = "lad_code" in cols and bool({"ofsted_rating", "ofsted_grade"} & cols)
    return {"ok": ok, "rows": _csv_rows(p),
            "detail": f"cols={sorted(cols)[:6]}"}


def v_geojson(p: Path) -> dict:                  # Module 3 (optional)
    try:
        g = json.loads(p.read_text())
        feats = g.get("features", [])
        props = feats[0].get("properties", {}) if feats else {}
        ok = g.get("type") == "FeatureCollection" and any(
            k.upper().startswith("LAD") and k.upper().endswith("CD") for k in props)
        return {"ok": ok, "rows": len(feats), "detail": f"{len(feats)} features"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "rows": 0, "detail": f"invalid geojson: {e}"}


# ----------------------------------------------------------------------- model
@dataclass
class Source:
    key: str
    name: str
    filename: str                       # relative to data/raw/
    validate_fn: Callable[[Path], dict]
    url: Optional[str] = None           # None => manual download only
    note: str = ""
    is_dir: bool = False
    optional: bool = False              # optional sources don't affect all_ok

    def path(self, raw_dir: Path) -> Path:
        return Path(raw_dir) / self.filename

    def fetch(self, raw_dir: Path, force: bool = False, retries: int = 3) -> dict:
        target = self.path(raw_dir)
        if self.url is None:
            return {"key": self.key, "status": "manual", "detail": self.note}
        if target.exists() and not force:
            return {"key": self.key, "status": "cached", "bytes": target.stat().st_size}
        import requests  # local import: only needed on the network path
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(self.url, headers=UA, timeout=60)
                r.raise_for_status()
                target.write_bytes(r.content)
                return {"key": self.key, "status": "downloaded", "bytes": len(r.content)}
            except Exception as e:  # noqa: BLE001
                log.warning("fetch %s attempt %d/%d failed: %s", self.key, attempt, retries, e)
                time.sleep(2 * attempt)
        return {"key": self.key, "status": "failed",
                "detail": f"could not download {self.url} — fetch it by hand into {target}"}

    def check(self, raw_dir: Path) -> dict:
        target = self.path(raw_dir)
        if not target.exists():
            return {"key": self.key, "present": False, "ok": False, "detail": "missing"}
        try:
            v = self.validate_fn(target)
        except Exception as e:  # noqa: BLE001
            v = {"ok": False, "rows": 0, "detail": f"validator error: {e}"}
        rec = {"key": self.key, "present": True, **v}
        if not self.is_dir:
            rec["bytes"] = target.stat().st_size
            rec["sha256"] = sha256(target)[:16]
        return rec


# ------------------------------------------------------------------- registry
SOURCES: list[Source] = [
    Source("land_registry", "HM Land Registry Price Paid (monthly)",
           "land_registry_pp_2026.csv", v_price_paid,
           url="http://prod1.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-monthly-update-new-version.csv",
           note="Stable S3 URL; rename to the dated file you want to model."),
    Source("pipr_monthly", "ONS Price Index of Private Rents (monthly tables)",
           "ons_pipr_monthly.xlsx", v_pipr_monthly, url=None,
           note="ONS release URL changes each month — download the latest xlsx by hand."),
    Source("pipr_historical", "ONS PIPR historical series",
           "ons_pipr_historical.xlsx", v_pipr_historical, url=None,
           note="ONS time-series xlsx — download the latest by hand."),
    Source("ashe", "ONS ASHE Table 8.7a (earnings by residence)",
           "ashe_table8_7a.xlsx", v_ashe, url=None,
           note="ONS ASHE supplementary table — download the latest by hand."),
    Source("lsoa_to_lad", "ONS LSOA(2021)->LAD(2024) lookup",
           "lsoa_to_lad.csv", v_lookup, url=None,
           note="ONS Open Geography Portal export."),
    Source("police", "data.police.uk street-level crime",
           "police", v_police_dir, url=None, is_dir=True,
           note="Bulk archive from data.police.uk — unzip street.csv files into data/raw/police/."),
    # --- Module 3 geospatial enrichment (optional; fetched later like EPC) ---
    Source("osm_amenities", "OpenStreetMap amenities by local authority",
           "osm_amenities.csv", v_osm, url=None, optional=True,
           note="Export OSM points (Overpass/osmnx) as rows of lad_code,category,[lat,lon]."),
    Source("ofsted", "Ofsted school ratings by local authority",
           "ofsted_schools.csv", v_ofsted, url=None, optional=True,
           note="Built by `python -m src.data.schools` from Ofsted MI + GIAS (lad_code, ofsted_grade)."),
    Source("la_boundaries", "ONS LA (2024) boundary geometry (GeoJSON)",
           "geo/lad_boundaries.geojson", v_geojson, url=None, optional=True,
           note="ONS Open Geography Portal LAD(2024) boundaries as GeoJSON (needs LAD24CD)."),
]

BY_KEY = {s.key: s for s in SOURCES}


# ------------------------------------------------------------------- orchestration
def fetch_all(raw_dir: Path, force: bool = False) -> list[dict]:
    return [s.fetch(raw_dir, force=force) for s in SOURCES]


def validate_all(raw_dir: Path) -> list[dict]:
    return [s.check(raw_dir) for s in SOURCES]


def build_manifest(raw_dir: Path) -> dict:
    raw_dir = Path(raw_dir)
    checks = validate_all(raw_dir)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {c["key"]: {**c, "name": BY_KEY[c["key"]].name,
                               "url": BY_KEY[c["key"]].url,
                               "optional": BY_KEY[c["key"]].optional} for c in checks},
        # required (non-optional) sources must all pass; optional ones are pending enrichment
        "all_ok": all(c.get("ok") for c in checks if not BY_KEY[c["key"]].optional),
    }
    (raw_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _print_manifest(m: dict) -> None:
    print(f"\nData manifest ({m['generated_at']}) — all_ok={m['all_ok']}")
    print("-" * 72)
    for key, s in m["sources"].items():
        flag = "OK " if s.get("ok") else ("-- " if not s.get("present") else "BAD")
        print(f"  [{flag}] {key:16} {s.get('detail', '')}")


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    cmd = sys.argv[1] if len(sys.argv) > 1 else "manifest"
    if cmd == "fetch":
        for r in fetch_all(config.DATA_RAW, force="--force" in sys.argv):
            print(" ", r)
    _print_manifest(build_manifest(config.DATA_RAW))
