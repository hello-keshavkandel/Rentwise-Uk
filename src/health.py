"""
health.py  —  RentWise UK health check & freshness (Milestone 4 observability).

One place to answer "is the app able to serve, and how fresh is its data/model?".
Used by the dashboard (freshness banner), by `make health`, and as a container
HEALTHCHECK. Pure-stdlib + pandas-free so it is cheap to call on every page load.

    python -m src.health        # prints a report, exits non-zero if not serviceable
    from src.health import check
    h = check()                 # dict: status, artifacts, data/model freshness
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# artifacts the app needs to serve predictions
REQUIRED = {
    "model": ROOT / "models" / "best_model.joblib",
    "features": ROOT / "data" / "processed" / "rentwise_features.csv",
    "clean": ROOT / "data" / "processed" / "rentwise_clean.csv",
}
OPTIONAL = {
    "conformal": ROOT / "models" / "conformal.json",
    "duckdb": ROOT / "data" / "processed" / "rentwise.duckdb",
}
MANIFEST = ROOT / "data" / "raw" / "_manifest.json"


def _age_days(ts: datetime) -> float:
    return round((datetime.now(timezone.utc) - ts).total_seconds() / 86400, 1)


def _mtime(p: Path):
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc) if p.exists() else None


def check() -> dict:
    """Return a health dict: status in {ok, degraded, down} + artifact/freshness detail."""
    artifacts = {k: p.exists() for k, p in {**REQUIRED, **OPTIONAL}.items()}
    required_ok = all(artifacts[k] for k in REQUIRED)

    # data freshness from the provenance manifest
    data = {"present": MANIFEST.exists()}
    if MANIFEST.exists():
        try:
            m = json.loads(MANIFEST.read_text())
            gen = m.get("generated_at")
            data["generated_at"] = gen
            data["all_ok"] = bool(m.get("all_ok"))
            if gen:
                data["age_days"] = _age_days(datetime.fromisoformat(gen))
        except Exception as e:  # noqa: BLE001
            data["error"] = str(e)

    # model freshness
    mt = _mtime(REQUIRED["model"])
    model = {"present": mt is not None}
    if mt:
        model["updated_at"] = mt.isoformat(timespec="seconds")
        model["age_days"] = _age_days(mt)

    if not required_ok:
        status = "down"
    elif MANIFEST.exists() and data.get("all_ok") is False:
        status = "degraded"
    else:
        status = "ok"

    return {"status": status, "serviceable": required_ok,
            "artifacts": artifacts, "data": data, "model": model,
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def summary_line(h: dict | None = None) -> str:
    """A compact one-liner for the dashboard banner."""
    h = h or check()
    bits = [f"status: {h['status']}"]
    if h["model"].get("age_days") is not None:
        bits.append(f"model {h['model']['age_days']}d old")
    if h["data"].get("age_days") is not None:
        bits.append(f"data {h['data']['age_days']}d old")
    return " · ".join(bits)


if __name__ == "__main__":  # pragma: no cover
    h = check()
    print(json.dumps(h, indent=2))
    sys.exit(0 if h["serviceable"] else 1)
