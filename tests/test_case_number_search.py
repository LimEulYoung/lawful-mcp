"""Case-number identity, joined cases and honest keyword fallback."""
from __future__ import annotations

import importlib
import sqlite3

import pytest

ps = importlib.import_module("lawful_mcp.tools.precedent_search")


def _case_corpus():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE prec_cases (
            id INTEGER PRIMARY KEY, case_number TEXT, case_name TEXT, court_name TEXT,
            court_level TEXT, decision_date TEXT, decision_year INTEGER, case_year INTEGER,
            reference_statute TEXT, summary TEXT, generated_summary TEXT,
            content_md TEXT, holdings TEXT
        );
        CREATE INDEX idx_prec_cases_casenum_norm
            ON prec_cases(REPLACE(case_number, ' ', ''));
        CREATE VIRTUAL TABLE prec_cases_fts USING fts5(
            case_name, content_md, summary, reference_statute, generated_summary,
            content='prec_cases', content_rowid='id', tokenize='trigram');
        CREATE TABLE prec_meta (
            id INTEGER PRIMARY KEY, court_level TEXT, court_name TEXT,
            decision_year INTEGER, case_year INTEGER
        );
    """)
    cases = [
        (1, "2010다89012", "손해배상", "손해배상 청구에 관한 판결이다."),
        (2, "93다37342", "손해배상", "대법원 2010다89012 판결을 따른다."),
        (3, "2009다84608, 84615, 84622", "구상금", "구상금 사건이다."),
        (4, "85다카1181(본소),1182(반소)", "소유권이전등기", "소유권이전등기 사건이다."),
        (5, "서울행법2005구합34725", "부당해고", "부당해고 사건이다."),
        (6, "2002노5461-1", "사기", "분리된 사기 사건이다."),
        (7, "2010다89012·89013", "손해배상", "병합된 손해배상 사건이다."),
    ]
    for cid, number, name, body in cases:
        conn.execute(
            "INSERT INTO prec_cases (id, case_number, case_name, court_name, court_level,"
            " decision_date, decision_year, case_year, content_md)"
            " VALUES (?,?,?,'대법원','대법원','2011-01-01',2011,2010,?)",
            (cid, number, name, body))
        conn.execute("INSERT INTO prec_meta VALUES (?,'대법원','대법원',2011,2010)", (cid,))
    conn.execute("INSERT INTO prec_cases_fts(prec_cases_fts) VALUES('rebuild')")
    conn.commit()
    return conn


@pytest.mark.parametrize("number, expected", [
    ("2010다89012", [1]),  # An exact row takes precedence over joined rows.
    ("93다373", []),
    ("2009다84608", [3]),
    ("2009다84615", [3]),
    ("2009다8461", []),
    ("85다카1182", [4]),
    ("2005구합34725", [5]),
    ("05구합34725", []),
    ("2002노5461", [6]),
    ("2010다89013", [7]),
])
def test_only_the_requested_case_number_matches(number, expected):
    conn = _case_corpus()
    try:
        assert ps._case_number_ids(conn, number, 5) == expected
    finally:
        conn.close()


def test_case_number_candidate_lookup_uses_the_index():
    """An expression change must not turn this into a scan of full bodies."""
    conn = _case_corpus()
    try:
        plan = " ".join(r[3] for r in conn.execute(
            "EXPLAIN QUERY PLAN " + ps._CASE_NO_CANDIDATES_SQL, ("%93다%", "%373%")))
        assert "idx_prec_cases_casenum_norm" in plan
    finally:
        conn.close()


def test_misses_filters_and_keyword_results_are_distinguished(monkeypatch):
    monkeypatch.setattr(ps, "open_db", _case_corpus)
    monkeypatch.setattr(ps, "USE_DENSE", False)

    missing = ps.precedent_search(None, case_number="93다373")
    assert "93다37342" not in missing
    assert "코퍼스에 수록되지 않은 판결" in missing

    fallback = ps.precedent_search(None, case_number="93다373", query="손해배상")
    assert "case_no:" in fallback
    assert "아래는 query 키워드로 찾은 판례입니다" in fallback

    filtered = ps.precedent_search(None, case_number="2010다89012", court_name="서울고법")
    assert "판례는 있으나 지정한 심급·법원·연도 조건에 맞지 않습니다" in filtered
    assert "조건을 빼고 다시 조회하세요" in filtered

    cited = ps.precedent_search(None, query="2010다89012")
    assert "case_no: 93다37342" in cited
    assert "case_number 인자에 넣으세요" in cited
