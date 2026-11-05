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


def _versions_db():
    """A source corpus attached as ``src``, with the shape `test_statute_history_rules` uses."""
    hist = importlib.import_module("test_statute_history_rules")
    conn = hist._corpus()
    conn.execute("ATTACH DATABASE ':memory:' AS src")
    for table in ("st_statutes", "st_articles"):
        conn.execute(f"CREATE TABLE src.{table} AS SELECT * FROM main.{table}")
    return hist, conn


def test_the_snapshot_pick_mirrors_the_tool():
    import importlib as _il
    statutes = _il.import_module("lawful_mcp.tools.statutes")
    hist, conn = _versions_db()
    try:
        group = conn.execute("SELECT id, effective_date, history_status, change_kind, mst, serial_id "
                             "FROM st_statutes WHERE law_id=?", (hist.LAW,)).fetchall()
        for as_of in ("19990101", "20120101", "20170101", "20260601", statutes._today_iso(), "29991231"):
            want = statutes._pick_current_version(conn, group, as_of)["id"]
            assert builder.pick_current_snapshot(conn, hist.LAW, as_of) == want, as_of
    finally:
        conn.close()


def test_later_full_text_versions_without_articles_are_dropped():
    """A future version loaded whole would take over once its date passed and
    serve a law with no articles; repeal and delta rows stay."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript("""
            CREATE TABLE st_statutes (id INTEGER PRIMARY KEY, law_id TEXT, effective_date TEXT,
                history_status TEXT, change_kind TEXT, mst TEXT, serial_id TEXT);
            CREATE TABLE charge_legal_map (statute_id INTEGER);
            INSERT INTO st_statutes VALUES
                (1, 'L', '20200101', '현행', '일부개정', '10', '10'),
                (2, 'L', '20270101', '시행예정', '일부개정', '20', '20'),
                (3, 'L', '20270601', '시행예정', '일부개정', '30', ''),
                (4, 'L', '20280101', '연혁', '폐지', '40', ''),
                (5, 'L', '20290101', '시행예정', '전부개정', '50', ''),
                (6, 'L', '20300101', '시행예정', '일부개정', '60', '60'),
                (7, 'L', '20100101', '연혁', '일부개정', '5', '5');
            INSERT INTO charge_legal_map VALUES (6);
        """)
        assert builder.drop_unshipped_full_texts(conn, [1]) == 2
        assert [r[0] for r in conn.execute("SELECT id FROM st_statutes ORDER BY id")] == [1, 3, 4, 6, 7]
    finally:
        conn.close()


def test_treaties_keep_the_chosen_partners_and_the_multilateral_ones_without_run_stamps():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("ATTACH DATABASE ':memory:' AS src")
        conn.executescript("""
            CREATE TABLE src.st_treaties (id INTEGER PRIMARY KEY, treaty_no TEXT, name TEXT NOT NULL,
                name_en TEXT, country TEXT, role TEXT NOT NULL, signed_date TEXT, effective_date TEXT,
                attachment_url TEXT, listed_at TEXT NOT NULL);
            CREATE TABLE src.st_treaty_articles (id INTEGER PRIMARY KEY, treaty_id INTEGER NOT NULL,
                article_seq INTEGER NOT NULL, article_no INTEGER, article_branch INTEGER NOT NULL DEFAULT 0,
                title TEXT, article_text TEXT NOT NULL);
            CREATE VIRTUAL TABLE src.st_treaty_articles_fts USING fts5(
                article_text, content='st_treaty_articles', content_rowid='id', tokenize='trigram');
            CREATE TABLE src.st_treaty_country_names (name TEXT PRIMARY KEY, country TEXT NOT NULL);
            INSERT INTO src.st_treaties VALUES
                (1, NULL, '베트남 협정', NULL, '베트남', '협약', NULL, '19940909', NULL, 'run stamp'),
                (2, NULL, '일본 협약', NULL, '일본', '협약', NULL, '19700101', NULL, 'run stamp'),
                (3, NULL, '다자협약', NULL, NULL, '개정', NULL, '20200901', NULL, 'run stamp');
            INSERT INTO src.st_treaty_articles (treaty_id, article_seq, article_no, article_text) VALUES
                (1, 0, 4, '제4조 거주자'), (2, 0, 4, '제4조 거주자'), (3, 0, 1, '제1조 적용 대상 조세협정');
            INSERT INTO src.st_treaty_country_names VALUES ('베트남', '베트남'), ('일본', '일본');
        """)
        assert builder.copy_treaty_tables(conn, ("베트남",)) == 2
        assert [r[0] for r in conn.execute("SELECT id FROM main.st_treaties ORDER BY id")] == [1, 3]
        assert "listed_at" not in [r[1] for r in conn.execute("PRAGMA main.table_info(st_treaties)")]
        assert conn.execute("SELECT COUNT(*) FROM main.st_treaty_articles").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM main.st_treaty_country_names").fetchone()[0] == 2
        hits = conn.execute("SELECT rowid FROM main.st_treaty_articles_fts WHERE st_treaty_articles_fts "
                            "MATCH '\"거주자\"'").fetchall()
        assert len(hits) == 1
    finally:
        conn.close()


def test_older_sources_without_treaties_are_supported():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("ATTACH DATABASE ':memory:' AS src")
        assert builder.copy_treaty_tables(conn, None) is None
        assert conn.execute("SELECT name FROM main.sqlite_master").fetchall() == []
    finally:
        conn.close()
