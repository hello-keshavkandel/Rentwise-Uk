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

# artifacts the app needs to serve predictions. Data is served from the DuckDB
# store (primary) with the processed CSVs as fallback — either one is enough.
REQUIRED = {
    "model": ROOT / "models" / "best_model.joblib",   # legacy fallback (always shipped)
}
MODEL_META = ROOT / "models" / "rent_model_meta.json"
DATA_STORE = ROOT / "data" / "processed" / "rentwise.duckdb"
DATA_CSVS = {
    "features": ROOT / "data" / "processed" / "rentwise_features.csv",
    "clean": ROOT / "data" / "processed" / "rentwise_clean.csv",
}
OPTIONAL = {
    "conformal": ROOT / "models" / "conformal.json",
    "duckdb": DATA_STORE,
    **DATA_CSVS,
}
MANIFEST = ROOT / "data" / "raw" / "_manifest.json"


def _age_days(ts: datetime) -> float:
    return round((datetime.now(timezone.utc) - ts).total_seconds() / 86400, 1)


def _mtime(p: Path):
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc) if p.exists() else None


def _store_manifest() -> dict | None:
    """Read data provenance from the store's source_manifest table (deployed path,
    where the raw manifest file is not shipped). Returns None if unavailable."""
    if not DATA_STORE.exists():
        return None
    try:
        import duckdb
        con = duckdb.connect(str(DATA_STORE), read_only=True)
        try:
            rows = con.execute("SELECT ok, generated_at FROM source_manifest").fetchall()
        finally:
            con.close()
    except Exception:
        return None
    if not rows:
        return None
    gens = [g for _, g in rows if g]
    return {"generated_at": max(gens) if gens else None,
            "all_ok": None,  # optional sources make a raw all() misleading; report as unknown
            "source": "store"}


def data_backend() -> str | None:
    """Which backend serves data: 'duckdb', 'csv', or None if neither is present."""
    if DATA_STORE.exists():
        return "duckdb"
    if all(p.exists() for p in DATA_CSVS.values()):
        return "csv"
    return None


def check() -> dict:
    """Return a health dict: status in {ok, degraded, down} + artifact/freshness detail."""
    artifacts = {k: p.exists() for k, p in {**REQUIRED, **OPTIONAL}.items()}
    backend = data_backend()
    required_ok = all(artifacts[k] for k in REQUIRED) and backend is not None

    # data freshness: raw provenance manifest if present, else the store's copy
    data = {"present": MANIFEST.exists(), "backend": backend}
    if not MANIFEST.exists():
        sm = _store_manifest()
        if sm and sm.get("generated_at"):
            data.update(present=True, generated_at=sm["generated_at"], source="store")
            try:
                data["age_days"] = _age_days(datetime.fromisoformat(sm["generated_at"]))
            except Exception:  # noqa: BLE001
                pass
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

    # model freshness: the deployed model's recorded training time (file mtimes reset
    # on every deploy/clone, so they are not a trustworthy age), else legacy mtime.
    model = {"present": False}
    trained = None
    if MODEL_META.exists():
        try:
            meta = json.loads(MODEL_META.read_text())
            trained = datetime.fromisoformat(meta["trained_at"])
            model.update(present=True, name="xgboost-spatial")
        except Exception:  # noqa: BLE001
            trained = None
    if trained is None:
        trained = _mtime(REQUIRED["model"])
        if trained:
            model.update(present=True, name="xgboost-legacy")
    if trained:
        model["updated_at"] = trained.isoformat(timespec="seconds")
        model["age_days"] = _age_days(trained)

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
    if h["data"].get("backend"):
        bits.append(f"data: {h['data']['backend']}")
    if h["model"].get("age_days") is not None:
        bits.append(f"model {h['model']['age_days']}d old")
    if h["data"].get("age_days") is not None:
        bits.append(f"data {h['data']['age_days']}d old")
    return " · ".join(bits)


if __name__ == "__main__":  # pragma: no cover
    h = check()
    print(json.dumps(h, indent=2))
    sys.exit(0 if h["serviceable"] else 1)
