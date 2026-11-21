"""Shared pytest fixtures.

Redirect the metrics registry to a temp file for every test, so test runs
(e.g. the Module 1 integration test on a synthetic EPC fixture) never write
synthetic results into the real reports/metrics_history.jsonl.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_metrics_history(tmp_path, monkeypatch):
    monkeypatch.setenv("RENTWISE_METRICS_HISTORY", str(tmp_path / "metrics_history.jsonl"))
