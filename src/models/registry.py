"""
registry.py  —  a lightweight model/metrics registry.

Every training run appends one JSON line to `reports/metrics_history.jsonl`, so the
quality of each module is tracked over time (and against the git commit that
produced it). No MLflow, no server — just an auditable, greppable history that the
pipeline writes and the dashboard can chart later.

    from src.models.registry import log_run, history
    log_run("module2", {"r2": 0.900, "rmse": 137})
    df = history("module2")           # pandas DataFrame, newest last
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _default_path() -> Path:
    # Overridable so tests (see tests/conftest.py) never write to the real history.
    import os
    env = os.environ.get("RENTWISE_METRICS_HISTORY")
    if env:
        return Path(env)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    return config.ROOT / "reports" / "metrics_history.jsonl"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        return None


def log_run(name: str, metrics: dict, extra: dict | None = None,
            history_path: Path | str | None = None) -> dict:
    """Append one metrics record for module `name`."""
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "module": name, "commit": _git_commit(),
           "metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                       for k, v in metrics.items()}}
    if extra:
        rec["extra"] = extra
    path = Path(history_path) if history_path else _default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def read_all(history_path: Path | str | None = None) -> list[dict]:
    path = Path(history_path) if history_path else _default_path()
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def history(name: str | None = None, history_path: Path | str | None = None):
    """Return the run history (optionally for one module) as a DataFrame."""
    import pandas as pd
    rows = read_all(history_path)
    if name:
        rows = [r for r in rows if r.get("module") == name]
    if not rows:
        return pd.DataFrame(columns=["ts", "module", "commit"])
    flat = []
    for r in rows:
        base = {"ts": r["ts"], "module": r["module"], "commit": r.get("commit")}
        base.update({f"metric_{k}": v for k, v in r.get("metrics", {}).items()})
        flat.append(base)
    return pd.DataFrame(flat)


def latest(name: str, history_path: Path | str | None = None) -> dict | None:
    rows = [r for r in read_all(history_path) if r.get("module") == name]
    return rows[-1] if rows else None
