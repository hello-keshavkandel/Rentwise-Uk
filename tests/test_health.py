"""Tests for the health check (src/health.py) — paths patched to a tmp dir."""
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.health as H


def _required(tmp):
    return {"model": tmp / "m.joblib", "features": tmp / "f.csv", "clean": tmp / "c.csv"}


def test_down_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "REQUIRED", _required(tmp_path))
    monkeypatch.setattr(H, "OPTIONAL", {})
    monkeypatch.setattr(H, "MANIFEST", tmp_path / "none.json")
    h = H.check()
    assert h["status"] == "down" and h["serviceable"] is False


def test_ok_when_all_present(tmp_path, monkeypatch):
    for p in _required(tmp_path).values():
        p.write_text("x")
    monkeypatch.setattr(H, "REQUIRED", _required(tmp_path))
    monkeypatch.setattr(H, "OPTIONAL", {})
    monkeypatch.setattr(H, "MANIFEST", tmp_path / "none.json")
    h = H.check()
    assert h["status"] == "ok" and h["serviceable"] is True


def test_degraded_when_manifest_not_ok(tmp_path, monkeypatch):
    for p in _required(tmp_path).values():
        p.write_text("x")
    man = tmp_path / "man.json"
    man.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "all_ok": False}))
    monkeypatch.setattr(H, "REQUIRED", _required(tmp_path))
    monkeypatch.setattr(H, "OPTIONAL", {})
    monkeypatch.setattr(H, "MANIFEST", man)
    h = H.check()
    assert h["status"] == "degraded"
    assert h["data"]["all_ok"] is False


def test_summary_line_is_string(tmp_path, monkeypatch):
    for p in _required(tmp_path).values():
        p.write_text("x")
    monkeypatch.setattr(H, "REQUIRED", _required(tmp_path))
    monkeypatch.setattr(H, "OPTIONAL", {})
    monkeypatch.setattr(H, "MANIFEST", tmp_path / "none.json")
    assert "status:" in H.summary_line()
