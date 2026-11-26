"""
demo.py  —  run RentWise UK locally with a ready-made demo account.

    make demo                 (or: python scripts/demo.py)
    python scripts/demo.py --seed-only     # just (re)create the demo data

Creates a separate demo database (data/api/demo.sqlite3, never your real one) with:
  * a demo user  demo@rentwise.local / rentwise-demo-2026
  * an API key   (printed once)
  * a saved area (Westminster) and an affordability alert (Camden flat above 50% of income),
    evaluated once so a notification is waiting
then starts the public API (http://localhost:8000/docs) and the dashboard
(http://localhost:8501). Ctrl+C stops both.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEMO_EMAIL = "demo@rentwise.local"
DEMO_PASSWORD = "rentwise-demo-2026"      # local demo only — never used outside data/api/demo.sqlite3


def seed(db_path: Path) -> dict:
    from src.api import security as sec
    from src.api.alerts import LogNotifier, evaluate_all
    from src.api.db import Database, now_iso
    from src.models.predict import RentWisePredictor

    if db_path.exists():
        db_path.unlink()
    db = Database(db_path)
    uid = db.run("INSERT INTO users (email, pw_hash, created_at) VALUES (?, ?, ?)",
                 (DEMO_EMAIL, sec.hash_password(DEMO_PASSWORD), now_iso()))
    key, prefix, h = sec.new_api_key()
    db.run("INSERT INTO api_keys (user_id, name, prefix, key_hash, created_at) VALUES (?, ?, ?, ?, ?)",
           (uid, "demo", prefix, h, now_iso()))
    db.run("INSERT INTO saved_areas (user_id, la_code, label, created_at) VALUES (?, ?, ?, ?)",
           (uid, "E09000033", "Westminster (demo)", now_iso()))
    db.run("INSERT INTO alerts (user_id, la_code, property_type, metric, op, threshold, created_at) "
           "VALUES (?, ?, ?, ?, ?, ?, ?)", (uid, "E09000007", "Flat", "ratio", "above", 0.5, now_iso()))
    summary = evaluate_all(db, RentWisePredictor(), LogNotifier())
    return {"api_key": key, "alerts": summary}


def port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def pick_port(preferred: int, name: str) -> int:
    """Use the preferred port, or the next free one (e.g. an old API still running on 8000)."""
    for port in range(preferred, preferred + 50):
        if port_free(port):
            if port != preferred:
                print(f"Note: port {preferred} is busy (something else is running there) "
                      f"— {name} will use {port} instead.")
            return port
    raise SystemExit(f"No free port near {preferred} for the {name}.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-only", action="store_true")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--app-port", type=int, default=8501)
    args = ap.parse_args()

    db_path = ROOT / "data" / "api" / "demo.sqlite3"
    info = seed(db_path)
    print(f"""
RentWise UK demo
----------------
Demo account : {DEMO_EMAIL} / {DEMO_PASSWORD}
API key      : {info['api_key']}
Seeded       : saved area Westminster; alert "Camden flat rent above 50% of income"
               -> {info['alerts']['fired']} notification(s) waiting
""")
    if args.seed_only:
        return 0

    args.api_port = pick_port(args.api_port, "API")
    args.app_port = pick_port(args.app_port, "dashboard")
    env = {**os.environ, "RENTWISE_API_DB": str(db_path)}
    py = sys.executable
    procs = [
        subprocess.Popen([py, "-m", "uvicorn", "src.api.app:app", "--port", str(args.api_port),
                          "--log-level", "warning"], cwd=ROOT, env=env),
        subprocess.Popen([py, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port",
                          str(args.app_port), "--server.headless", "true",
                          "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env),
    ]
    time.sleep(3)
    print(f"""Dashboard : http://localhost:{args.app_port}
API docs  : http://localhost:{args.api_port}/docs
Try it    : curl -X POST http://localhost:{args.api_port}/v1/predict \\
              -H "X-API-Key: {info['api_key']}" -H "Content-Type: application/json" \\
              -d '{{"la_code":"E09000007","property_type":"Flat"}}'
Press Ctrl+C to stop.
""", flush=True)
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
