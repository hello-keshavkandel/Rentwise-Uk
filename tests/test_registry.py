"""Tests for the metrics-history registry (src/models/registry.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import registry as reg


def test_log_and_history(tmp_path):
    hp = tmp_path / "metrics_history.jsonl"
    reg.log_run("module1", {"r2": 0.567, "rmse": 140886}, history_path=hp)
    reg.log_run("module1", {"r2": 0.72, "rmse": 110000}, history_path=hp)
    reg.log_run("module2", {"r2": 0.900}, history_path=hp)

    df = reg.history("module1", history_path=hp)
    assert len(df) == 2
    assert "metric_r2" in df.columns
    assert df["metric_r2"].iloc[-1] == 0.72       # newest last

    assert reg.latest("module1", history_path=hp)["metrics"]["r2"] == 0.72
    assert len(reg.history(history_path=hp)) == 3  # all modules


def test_history_empty(tmp_path):
    df = reg.history("module1", history_path=tmp_path / "none.jsonl")
    assert df.empty
