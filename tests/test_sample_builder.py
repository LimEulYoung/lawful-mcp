"""The sample builder: which statute versions ship, and what it copies from the status, annex and treaty tables."""
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_sample_db", Path(__file__).resolve().parents[1] / "scripts/build_sample_db.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def _with_source(script: str) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("ATTACH DATABASE ':memory:' AS src")
    conn.executescript(script)
    return conn


def test_the_current_whole_text_and_later_amendments_ship_and_pending_whole_texts_are_left_out():
    conn = _with_source("""
        CREATE TABLE src.st_statutes (id INTEGER PRIMARY KEY, law_id TEXT, effective_date TEXT,
            change_kind TEXT, serial_id TEXT);
        INSERT INTO src.st_statutes VALUES
            (1, 'L', '20100101', '제정', '1'),          -- an older whole text: row only
            (2, 'L', '20150101', '일부개정', ''),        -- an amendment before the current text: row only
            (3, 'L', '20200101', '일부개정', '3'),       -- the current whole text
            (4, 'L', '20210101', '일부개정', ''),        -- an amendment in force since
            (5, 'L', '20270101', '일부개정', ''),        -- pending, changed articles only: ships
            (6, 'L', '20280101', '일부개정', '6'),       -- pending, stored whole: left out
            (7, 'L', '20290101', '폐지', '');            -- pending repeal row: ships (no articles)
    """)
    try:
        ship, left_out = builder.plan_versions(conn, "L", "20261108")
        assert sorted(ship) == [3, 4, 5, 7]
        assert left_out == [6]
        # A law not yet in force ships from its first version.
        conn.executescript("""INSERT INTO src.st_statutes VALUES (8, 'N', '20270101', '제정', '8'),
                                                                  (9, 'N', '20280101', '일부개정', '');""")
        assert builder.plan_versions(conn, "N", "20261108") == ([8, 9], [])
    finally:
        conn.close()


def test_status_tables_cover_the_sampled_laws_without_loader_notes():
    conn = _with_source("""
        CREATE TABLE main.st_statutes (id INTEGER PRIMARY KEY, law_id TEXT);
        CREATE TABLE main.st_articles (id INTEGER PRIMARY KEY, statute_id INTEGER);
        INSERT INTO main.st_statutes VALUES (1, 'shipped'), (2, 'row-only');
        INSERT INTO main.st_articles VALUES (10, 1);
        CREATE TABLE src.st_lapsed (law_id TEXT PRIMARY KEY, lapsed_on TEXT NOT NULL, reason TEXT NOT NULL,
            note TEXT, noted_at TEXT);
        INSERT INTO src.st_lapsed VALUES ('row-only', '20260701', 'sunset', 'private load note', 'stamp'),
                                         ('other', '20260701', 'sunset', 'other', 'stamp');
        CREATE TABLE src.st_addenda (id INTEGER PRIMARY KEY, law_id TEXT NOT NULL, article_no INTEGER NOT NULL,
            article_text TEXT NOT NULL);
        INSERT INTO src.st_addenda VALUES (1, 'shipped', 1, '제1조(시행일)'), (2, 'row-only', 1, '제1조'),
                                          (3, 'other', 1, '제1조');
    """)
    try:
        builder.copy_statute_status_tables(conn)
        assert conn.execute("SELECT * FROM main.st_lapsed").fetchall() == [
            ("row-only", "20260701", "sunset", None, None)]
        # Addenda only for the laws whose articles ship.
        assert conn.execute("SELECT law_id FROM main.st_addenda").fetchall() == [("shipped",)]
    finally:
        conn.close()


def test_the_annex_index_covers_sampled_laws_and_rules_without_run_stamps():
    conn = _with_source("""
        CREATE TABLE main.st_statutes (id INTEGER PRIMARY KEY, law_id TEXT);
        CREATE TABLE main.st_notices (id INTEGER PRIMARY KEY, notice_id TEXT);
        INSERT INTO main.st_statutes VALUES (1, '000100');
        INSERT INTO main.st_notices VALUES (5, '2100000001');
        CREATE TABLE src.st_byeolpyo (id INTEGER PRIMARY KEY, scope TEXT NOT NULL, owner_id TEXT NOT NULL,
            kind TEXT, title TEXT, file_url TEXT, listed_at TEXT);
        INSERT INTO src.st_byeolpyo VALUES (1, 'law', '000100', '별표', 'a', 'u1', 'stamp'),
            (2, 'admrul', '2100000001', '별표', 'b', 'u2', 'stamp'), (3, 'law', '000200', '별표', 'c', 'u3', 'stamp'),
            (4, 'admrul', '000100', '별표', 'd', 'u4', 'stamp');
    """)
    try:
        builder.copy_annex_index(conn)
        rows = conn.execute("SELECT id, listed_at FROM main.st_byeolpyo ORDER BY id").fetchall()
        assert rows == [(1, None), (2, None)]
    finally:
        conn.close()


def test_treaties_keep_the_chosen_partners_and_the_multilateral_ones_without_run_stamps():
    conn = _with_source("""
        CREATE TABLE src.st_treaties (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT,
            role TEXT NOT NULL, effective_date TEXT, attachment_url TEXT, listed_at TEXT);
        CREATE TABLE src.st_treaty_articles (id INTEGER PRIMARY KEY, treaty_id INTEGER NOT NULL,
            article_seq INTEGER NOT NULL, article_no INTEGER, article_branch INTEGER NOT NULL DEFAULT 0,
            title TEXT, article_text TEXT NOT NULL);
        CREATE VIRTUAL TABLE src.st_treaty_articles_fts USING fts5(
            article_text, content='st_treaty_articles', content_rowid='id', tokenize='trigram');
        CREATE TABLE src.st_treaty_country_names (name TEXT PRIMARY KEY, country TEXT NOT NULL);
        INSERT INTO src.st_treaties VALUES
            (1, '베트남 협정', '베트남', '협약', '19940909', NULL, 'run stamp'),
            (2, '일본 협약', '일본', '협약', '19700101', NULL, 'run stamp'),
            (3, '다자협약', NULL, '개정', '20200901', NULL, 'run stamp');
        INSERT INTO src.st_treaty_articles (treaty_id, article_seq, article_no, article_text) VALUES
            (1, 0, 4, '제4조 거주자'), (2, 0, 4, '제4조 거주자'), (3, 0, 1, '제1조 적용 대상 조세협정');
        INSERT INTO src.st_treaty_country_names VALUES ('베트남', '베트남'), ('일본', '일본');
    """)
    try:
        assert builder.copy_treaty_tables(conn, ("베트남",)) == 2
        assert conn.execute("SELECT id, listed_at FROM main.st_treaties ORDER BY id").fetchall() == [
            (1, None), (3, None)]
        assert conn.execute("SELECT COUNT(*) FROM main.st_treaty_articles").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM main.st_treaty_country_names").fetchone()[0] == 2
        hits = conn.execute("SELECT rowid FROM main.st_treaty_articles_fts WHERE st_treaty_articles_fts "
                            "MATCH '\"거주자\"'").fetchall()
        assert len(hits) == 1
    finally:
        conn.close()


def test_a_source_missing_a_table_the_tools_read_stops_the_build():
    conn = _with_source("CREATE TABLE main.st_statutes (id INTEGER PRIMARY KEY, law_id TEXT);")
    try:
        with pytest.raises(SystemExit, match="st_byeolpyo"):
            builder.copy_annex_index(conn)
    finally:
        conn.close()
