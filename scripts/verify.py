"""
verify.py  —  one command that proves RentWise UK works on this machine.

    make verify            (or: python scripts/verify.py)

Checks, in order, and prints a PASS / WARN / FAIL table (exit code 1 on any FAIL):
  1. Python version (3.11 = the version the live app, Docker and CI use)
  2. Installed packages match the pins in requirements.txt
  3. Health: model + data present, freshness
  4. Full automated test suite (unit, integration, data-quality, dashboard, API)
  5. Live API: starts uvicorn, creates a temporary key, calls /v1/health, /v1/sources, /v1/predict
  6. Live dashboard: starts Streamlit headless and checks its health endpoint

A copy of the table is written to reports/verify_report.md.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = sys.executable
results: list[tuple[str, str, str]] = []          # (check, status, detail)


def record(check, status, detail):
    results.append((check, status, detail))
    icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[status]
    print(f"{icon} {check:<22} {detail}", flush=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(url, data=None, headers=None, timeout=10):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
        return r.status, (json.loads(body) if body.strip().startswith(("{", "[")) else body)


def wait_for(url, seconds=60):
    end = time.time() + seconds
    while time.time() < end:
        try:
            return http(url, timeout=3)
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"{url} did not answer within {seconds}s")


# ---------------------------------------------------------------- checks
def check_python():
    v = sys.version_info
    if (v.major, v.minor) == (3, 11):
        record("Python", "PASS", f"{v.major}.{v.minor}.{v.micro} ({PY})")
    else:
        record("Python", "WARN", f"{v.major}.{v.minor} — the live app uses 3.11; run `make setup` for a 3.11 .venv")


def check_pins():
    pins = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==([\d.]+)", (ROOT / "requirements.txt").read_text(), re.M))
    wrong, missing = [], []
    for name, want in pins.items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if have != want:
            wrong.append(f"{name} {have}≠{want}")
    if missing:
        record("Packages", "FAIL", "missing: " + ", ".join(missing) + " — run `make setup`")
    elif wrong:
        record("Packages", "WARN", f"{len(wrong)} differ from pins: " + ", ".join(wrong[:4])
               + " — run `make setup` to match them")
    else:
        record("Packages", "PASS", f"all {len(pins)} pinned packages match")


def check_health():
    from src.health import check
    h = check()
    st = "PASS" if h["status"] == "ok" else ("WARN" if h["serviceable"] else "FAIL")
    record("Health", st, f"status {h['status']} · data via {h['data'].get('backend')} · "
                         f"model {h['model'].get('name')} ({h['model'].get('updated_at', '?')[:10]})")


def check_tests():
    t0 = time.time()
    p = subprocess.run([PY, "-m", "pytest", "-q", "-W", "ignore::DeprecationWarning", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    tail = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-1:] or ["no output"]
    summary = re.sub(r"=+", "", tail[0]).strip()
    record("Test suite", "PASS" if p.returncode == 0 else "FAIL", f"{summary} ({time.time() - t0:.0f}s)")
    if p.returncode != 0:
        print(p.stdout[-3000:])


def check_api():
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        record("Live API", "WARN", "fastapi/uvicorn not installed — `make setup` installs them")
        return
    tmp = tempfile.mkdtemp(prefix="rentwise-verify-")
    env = {**os.environ, "RENTWISE_API_DB": str(Path(tmp) / "verify.sqlite3"), "RENTWISE_RATE_LIMIT": "1000"}
    out = subprocess.run([PY, "-m", "src.api.admin", "create-key", "--name", "verify"], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    key = re.search(r"(rw_[A-Za-z0-9_\-]+)", out.stdout)
    port = free_port()
    proc = subprocess.Popen([PY, "-m", "uvicorn", "src.api.app:app", "--port", str(port), "--log-level", "warning"],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    try:
        _, h = wait_for(base + "/v1/health")
        _, src = http(base + "/v1/sources")
        _, pr = http(base + "/v1/predict", {"la_code": "E09000007", "property_type": "Flat"},
                     {"X-API-Key": key.group(1) if key else ""})
        ok = h.get("status") == "ok" and pr.get("rent_gbp_month", 0) > 0 and src["vintages"].get("rents")
        record("Live API", "PASS" if ok else "FAIL",
               f"{base} · Camden flat £{pr.get('rent_gbp_month'):,} ({pr.get('kind')}, {pr.get('as_of')})")
    except Exception as e:  # noqa: BLE001
        err = proc.stderr.read().decode()[-400:] if proc.poll() is not None and proc.stderr else ""
        record("Live API", "FAIL", f"{e} {err}")
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def check_dashboard():
    try:
        import streamlit  # noqa: F401
    except ImportError:
        record("Live dashboard", "FAIL", "streamlit not installed — run `make setup`")
        return
    port = free_port()
    proc = subprocess.Popen([PY, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", str(port),
                             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        status, body = wait_for(f"http://127.0.0.1:{port}/_stcore/health", 90)
        record("Live dashboard", "PASS" if str(body).strip() == "ok" else "FAIL",
               f"http://127.0.0.1:{port} answered {status} {str(body).strip()}")
    except Exception as e:  # noqa: BLE001
        record("Live dashboard", "FAIL", str(e))
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def main() -> int:
    print(f"RentWise UK — verify ({datetime.now():%Y-%m-%d %H:%M})\n")
    for fn in (check_python, check_pins, check_health, check_tests, check_api, check_dashboard):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — one broken check must not hide the others
            record(fn.__name__.replace("check_", "").title(), "FAIL", f"{type(e).__name__}: {e}")
    fails = sum(s == "FAIL" for _, s, _ in results)
    warns = sum(s == "WARN" for _, s, _ in results)
    verdict = "FAIL" if fails else ("PASS with warnings" if warns else "PASS")
    print(f"\nResult: {verdict}  ({len(results) - fails - warns} pass, {warns} warn, {fails} fail)")
    lines = [f"# Verify report — {datetime.now():%Y-%m-%d %H:%M}", "", f"**Result: {verdict}**", "",
             "| Check | Status | Detail |", "|---|---|---|"]
    lines += [f"| {c} | {s} | {d.replace('|', '/')} |" for c, s, d in results]
    (ROOT / "reports" / "verify_report.md").write_text("\n".join(lines) + "\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
