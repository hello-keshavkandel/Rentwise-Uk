"""
store.py  —  the RentWise UK processed-data store (DuckDB).

Milestone 2 moves processed data off loose CSVs into a single, queryable DuckDB
file (`data/processed/rentwise.duckdb`). CSVs stay as the human-readable pipeline
output; this store is the fast, SQL-queryable layer the app and analyses read.

    python -m src.data.store build     # (re)build the store from processed CSVs
    python -m src.data.store info      # list tables + row counts

    from src.data.store import table, query
    df = table("rentwise_clean")
    hi = query("SELECT region, avg(median_monthly_rent) r FROM rentwise_clean GROUP BY 1 ORDER BY r DESC")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb

log = logging.getLogger("rentwise.store")

# processed CSV -> DuckDB table name
PROCESSED_TABLES = {
    "rentwise_clean": "rentwise_clean.csv",
    "rentwise_features": "rentwise_features.csv",
    "rentwise_sample": "rentwise_sample.csv",
    "affordability": "affordability.csv",
}


def _default_db() -> Path:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import config
    return config.DATA_PROCESSED / "rentwise.duckdb"


def _processed_dir() -> Path:
    return _default_db().parent


def connect(db_path: Path | str | None = None, read_only: bool = False):
    db_path = Path(db_path) if db_path else _default_db()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path), read_only=read_only)


def build_store(processed_dir: Path | str | None = None,
                db_path: Path | str | None = None) -> dict:
    """(Re)build the DuckDB store from the processed CSVs + the raw manifest."""
    processed_dir = Path(processed_dir) if processed_dir else _processed_dir()
    con = connect(db_path)
    counts: dict[str, int] = {}
    try:
        for table_name, csv_name in PROCESSED_TABLES.items():
            csv_path = processed_dir / csv_name
            if not csv_path.exists():
                log.warning("skip %s (missing %s)", table_name, csv_path.name)
                continue
            con.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS "
                f"SELECT * FROM read_csv_auto(?, header=true)", [str(csv_path)])
            counts[table_name] = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]

        # fold the raw-data provenance manifest in, if present
        manifest = processed_dir.parent / "raw" / "_manifest.json"
        if manifest.exists():
            m = json.loads(manifest.read_text())
            rows = [{"key": k, "name": v.get("name"), "ok": bool(v.get("ok")),
                     "rows": v.get("rows"), "bytes": v.get("bytes"),
                     "sha256": v.get("sha256"), "url": v.get("url"),
                     "generated_at": m.get("generated_at")} for k, v in m["sources"].items()]
            con.execute("CREATE OR REPLACE TABLE source_manifest "
                        "(key VARCHAR, name VARCHAR, ok BOOLEAN, rows BIGINT, bytes BIGINT, "
                        " sha256 VARCHAR, url VARCHAR, generated_at VARCHAR)")
            con.executemany(
                "INSERT INTO source_manifest VALUES (?,?,?,?,?,?,?,?)",
                [[r["key"], r["name"], r["ok"], r["rows"], r["bytes"],
                  r["sha256"], r["url"], r["generated_at"]] for r in rows])
            counts["source_manifest"] = len(rows)
        log.info("store built: %s", counts)
        return counts
    finally:
        con.close()


def table(name: str, db_path: Path | str | None = None):
    con = connect(db_path, read_only=True)
    try:
        return con.execute(f"SELECT * FROM {name}").df()
    finally:
        con.close()


def query(sql: str, db_path: Path | str | None = None, params=None):
    con = connect(db_path, read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


def info(db_path: Path | str | None = None) -> dict:
    con = connect(db_path, read_only=True)
    try:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
    finally:
        con.close()


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    if cmd == "build":
        print("built:", build_store())
    print("tables:", info())
