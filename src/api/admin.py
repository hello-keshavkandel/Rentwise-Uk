"""
admin.py  —  operator commands for the RentWise API (no HTTP needed).

    python -m src.api.admin create-key --name partner-x     # service key, printed once
    python -m src.api.admin list-keys
    python -m src.api.admin revoke-key 3
    python -m src.api.admin stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api import security as sec  # noqa: E402
from src.api.db import Database, now_iso  # noqa: E402


def main(argv=None, db: Database | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rentwise-admin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-key"); c.add_argument("--name", required=True)
    sub.add_parser("list-keys")
    r = sub.add_parser("revoke-key"); r.add_argument("key_id", type=int)
    sub.add_parser("stats")
    args = ap.parse_args(argv)
    if db is None:
        from src.api.app import db_path_from_env
        db = Database(db_path_from_env())

    if args.cmd == "create-key":
        key, prefix, h = sec.new_api_key()
        kid = db.run("INSERT INTO api_keys (user_id, name, prefix, key_hash, created_at) VALUES (NULL, ?, ?, ?, ?)",
                     (args.name, prefix, h, now_iso()))
        print(f"API key #{kid} ({args.name}): {key}\nStore it now — it is not stored and will not be shown again.")
    elif args.cmd == "list-keys":
        for k in db.all("SELECT id, user_id, name, prefix, created_at, revoked FROM api_keys ORDER BY id"):
            print(k)
    elif args.cmd == "revoke-key":
        n = db.run("UPDATE api_keys SET revoked = 1 WHERE id = ?", (args.key_id,))
        print("revoked" if n else "no such key")
        return 0 if n else 1
    elif args.cmd == "stats":
        for t in ("users", "api_keys", "saved_areas", "alerts", "notifications"):
            print(t, db.one(f"SELECT COUNT(*) c FROM {t}")["c"])   # table names are fixed literals
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
