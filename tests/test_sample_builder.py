"""A sample contains only the status facts needed for its own statutes."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sqlite3

spec = importlib.util.spec_from_file_location(
    "build_sample_db", Path(__file__).resolve().parents[1] / "scripts/build_sample_db.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_status_records_exclude_other_laws_and_operational_notes():
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript("""
            ATTACH DATABASE ':memory:' AS src;
            CREATE TABLE st_statutes (id INTEGER PRIMARY KEY, law_id TEXT);
            INSERT INTO st_statutes VALUES (1, 'sampled');
            CREATE TABLE src.st_backfill_log (statute_id INTEGER, status TEXT, note TEXT);
            INSERT INTO src.st_backfill_log VALUES (1, 'ok', 'private load note'), (2, 'ok', 'other');
            CREATE TABLE src.st_lapsed (law_id TEXT, lapsed_on TEXT, reason TEXT, note TEXT, noted_at TEXT);
            INSERT INTO src.st_lapsed VALUES
                ('sampled', '20260701', 'sunset', 'private load note', 'timestamp'),
                ('other', '20260701', 'sunset', 'other', 'timestamp');
        """)
        builder.copy_statute_status_tables(conn)
        assert conn.execute("SELECT * FROM main.st_backfill_log").fetchall() == [(1, "ok")]
        assert conn.execute("SELECT * FROM main.st_lapsed").fetchall() == [("sampled", "20260701", "sunset")]
        assert [r[1] for r in conn.execute("PRAGMA main.table_info(st_lapsed)")] == [
            "law_id", "lapsed_on", "reason"]
    finally:
        conn.close()


def test_older_sources_without_status_tables_are_supported():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("ATTACH DATABASE ':memory:' AS src")
        builder.copy_statute_status_tables(conn)
        assert conn.execute("SELECT name FROM main.sqlite_master").fetchall() == []
    finally:
        conn.close()
