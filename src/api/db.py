"""
db.py  —  storage for the RentWise public API (Milestone 5).

SQLite via the standard library: zero extra services, one file, transactional.
The SQL is deliberately plain so the same schema moves to Postgres when there are
real users (Phase 5 roadmap). Every query is parameterised (no string-built SQL).

Tables
  users          id, email (unique, lower-cased), pw_hash, created_at
  sessions       token_hash (sha256 of bearer token), user_id, expires_at
  api_keys       id, user_id (nullable for admin keys), name, prefix, key_hash, created_at, revoked
  saved_areas    user_id, la_code, label, created_at          (unique per user+area)
  alerts         id, user_id, la_code, property_type, metric ('rent'|'ratio'), op ('above'|'below'),
                 threshold, active, last_value, last_state, last_checked_at, created_at
  notifications  id, user_id, alert_id, message, value, created_at, read_at, delivered_via
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    pw_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS saved_areas (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    la_code TEXT NOT NULL,
    label TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, la_code)
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    la_code TEXT NOT NULL,
    property_type TEXT NOT NULL,
    metric TEXT NOT NULL CHECK (metric IN ('rent', 'ratio')),
    op TEXT NOT NULL CHECK (op IN ('above', 'below')),
    threshold REAL NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    last_value REAL,
    last_state INTEGER,
    last_checked_at TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    alert_id INTEGER REFERENCES alerts(id) ON DELETE SET NULL,
    message TEXT NOT NULL,
    value REAL,
    created_at TEXT NOT NULL,
    read_at TEXT,
    delivered_via TEXT
);
CREATE INDEX IF NOT EXISTS ix_alerts_user ON alerts(user_id);
CREATE INDEX IF NOT EXISTS ix_notif_user ON notifications(user_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Thin, thread-safe wrapper: one connection per operation, WAL journal."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mem = sqlite3.connect(":memory:", check_same_thread=False) if self.path == ":memory:" else None
        with self.connect() as con:
            con.executescript(SCHEMA)
            if self._mem is None:
                con.execute("PRAGMA journal_mode=WAL")

    @contextmanager
    def connect(self):
        if self._mem is not None:                     # tests: shared in-memory DB
            with self._lock:
                self._mem.row_factory = sqlite3.Row
                self._mem.execute("PRAGMA foreign_keys = ON")
                try:
                    yield self._mem
                    self._mem.commit()
                except Exception:
                    self._mem.rollback()
                    raise
            return
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def one(self, sql: str, params=()):
        with self.connect() as con:
            r = con.execute(sql, params).fetchone()
            return dict(r) if r else None

    def all(self, sql: str, params=()):
        with self.connect() as con:
            return [dict(r) for r in con.execute(sql, params).fetchall()]

    def run(self, sql: str, params=()) -> int:
        """Execute a write; returns lastrowid (inserts) or rowcount (updates/deletes)."""
        with self.connect() as con:
            cur = con.execute(sql, params)
            return cur.lastrowid if sql.lstrip().upper().startswith("INSERT") else cur.rowcount
