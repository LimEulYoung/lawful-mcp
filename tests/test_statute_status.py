"""Repeal, lapse, historical lookup and response-budget contracts."""
from __future__ import annotations

import sqlite3

import pytest

from lawful_mcp.tools import statutes


@pytest.fixture
def corpus(monkeypatch):
    monkeypatch.setattr(statutes, "_today_iso", lambda: "20260929")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE st_statutes (
            id INTEGER PRIMARY KEY, law_id TEXT, effective_date TEXT,
            history_status TEXT, change_kind TEXT
        );
        INSERT INTO st_statutes VALUES
            (1, 'repealed', '20200101', NULL, '제정'),
            (2, 'repealed', '20260901', '연혁', '폐지'),
            (3, 'expired', '20200101', NULL, '제정'),
            (4, 'pending', '20200101', NULL, '제정'),
            (5, 'pending', '20990101', '시행예정', '타법폐지'),
            (6, 'renamed', '20200101', '연혁', '제정'),
            (7, 'renamed', '20260101', '현행', '전부개정');
    """)
    yield conn
    conn.close()


def test_repeal_does_not_depend_on_the_backfill_register(corpus):
    """A repeal edition is not a living successor; a future repeal is pending."""
    assert statutes._backfilled_sids(corpus, [1, 4, 6]) == set()
    assert statutes._repealed_sids(corpus, [1, 4, 6]) == {1}
    corpus.executescript("""
        CREATE TABLE st_backfill_log (statute_id INTEGER, status TEXT);
        INSERT INTO st_backfill_log VALUES (1, 'ok'), (6, 'ok');
    """)
    assert statutes._repealed_sids(corpus, [1, 4, 6]) == {1}
    assert statutes._repeal_label(corpus, "repealed") == "폐지"


@pytest.mark.parametrize("reason, label", [
    ("sunset", "유효기간 만료"),
    ("amending_act", "일괄개정 반영 완료"),
    ("reenacted", "같은 이름의 새 법으로 대체"),
    ("parent_lapsed", "모법 효력 상실"),
])
def test_lapse_respects_the_date_and_later_reinstatement(corpus, reason, label):
    assert statutes._lapse(corpus, "expired", "20260929") is None
    corpus.execute("CREATE TABLE st_lapsed (law_id TEXT PRIMARY KEY, lapsed_on TEXT, reason TEXT)")
    corpus.execute("INSERT INTO st_lapsed VALUES ('expired','20260701',?)", (reason,))
    assert not statutes._is_repealed_as_of(corpus, "expired", "20260630")
    assert statutes._is_repealed_as_of(corpus, "expired", "20260701")
    assert statutes._repealed_sids(corpus, [3]) == {3}
    assert statutes._repeal_label(corpus, "expired") == f"효력 상실({label})"
    assert statutes._is_repealed_at(corpus, "expired", "20260929") == {
        "effective_date": "20260701", "change_kind": f"효력 상실({label})"}

    corpus.execute("INSERT INTO st_statutes VALUES (8,'expired','20260801','현행','일부개정')")
    assert not statutes._is_repealed_as_of(corpus, "expired", "20260929")
    assert statutes._repealed_sids(corpus, [3, 8]) == set()
    # Reinstatement does not rewrite the interval during which it had lapsed.
    assert statutes._is_repealed_as_of(corpus, "expired", "20260715")


def test_lapse_is_named_in_list_and_detail_responses():
    label = "효력 상실(유효기간 만료)"
    listed = statutes._format_response_md({
        "status": "ok", "mode": "list", "matches": [{
            "statute_id": 1, "name": "예시법", "is_repealed": True, "repeal_label": label}]})
    detail = statutes._format_response_md({
        "status": "ok", "mode": "detail", "statute": {"id": 1, "name": "예시법"},
        "repealed": {"change_kind": label, "note": label}})
    assert f"{label} — 지금은 효력이 없는 법령입니다" in listed
    assert "폐지된 법령" not in listed
    assert f"- 효력 상실: {label}" in detail
    assert "- 폐지:" not in detail


def test_response_budget_is_enforced_at_the_tool_boundary(monkeypatch):
    """Exercise the public call so leaving its returns unbounded is caught."""
    monkeypatch.setattr(statutes, "open_db", lambda: sqlite3.connect(":memory:"))
    huge = {"status": "ok", "mode": "outline", "statute": {"name": "예시법", "id": 1},
            "articles": [{"no": str(i), "title": "제" * 120} for i in range(500)]}
    monkeypatch.setattr(statutes, "_statute_lookup_impl", lambda *a, **k: huge)
    raw = statutes._format_response_md(huge)
    out = statutes.statute_lookup(None, statute_id=1)
    assert len(raw) > statutes.RESPONSE_MAX_CHARS
    assert len(out) <= statutes.RESPONSE_MAX_CHARS
    assert "꼬리를 생략" in out
    assert f"전체 {len(raw):,}자" in out
    assert out.endswith("articles=[...] 로 나눠 재호출하세요.")
    kept = out.split("\n\n- message:")[0]
    assert raw.startswith(kept + "\n")

    small = {"status": "missing_input", "message": "짧은 응답"}
    assert statutes._bounded_response_md(small) == statutes._format_response_md(small)
    # Even a single overlong line must fit, including the notice itself.
    assert len(statutes._bounded_response_md({"message": "가" * 60_000})) <= statutes.RESPONSE_MAX_CHARS
