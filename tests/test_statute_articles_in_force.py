"""Article membership across full revisions, partial amendments and future editions.

Synthetic histories keep these checks independent of the bundled corpus and of
today's date. A full revision can remove an old number without a deletion row.
"""
from __future__ import annotations

import sqlite3

import pytest

from lawful_mcp.tools import statutes


LAW, SPARSE_LAW, LARGE_LAW, PENDING_LAW = "example", "sparse", "large", "pending"
OLD_FULL, OLD_DELTA, REVISION, SNAPSHOT, NEW_DELTA = 101, 102, 103, 104, 105
LARGE_SNAPSHOT, PENDING_SNAPSHOT = 304, 403


@pytest.fixture
def corpus(monkeypatch):
    monkeypatch.setattr(statutes, "_today_iso", lambda: "20261001")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE st_statutes (
            id INTEGER PRIMARY KEY, law_id TEXT, name TEXT, short_name TEXT,
            kind TEXT, issuing_agency TEXT, promulgation_date TEXT,
            effective_date TEXT, history_status TEXT, change_kind TEXT, mst TEXT
        );
        CREATE TABLE st_articles (
            id INTEGER PRIMARY KEY, statute_id INTEGER, article_no TEXT,
            article_no_num INTEGER, article_branch INTEGER, title TEXT,
            article_text TEXT, article_changed TEXT, article_eff_date TEXT
        );
        CREATE TABLE st_backfill_log (statute_id INTEGER, status TEXT);
    """)

    def version(sid, law_id, eff, status, kind, name="예시법"):
        conn.execute(
            "INSERT INTO st_statutes (id, law_id, name, kind, effective_date, history_status,"
            " change_kind, mst) VALUES (?, ?, ?, '법률', ?, ?, ?, ?)",
            (sid, law_id, name, eff, status, kind, str(sid) if status else None))

    def article(sid, num, branch, title, eff, changed="Y"):
        no = f"{num}의{branch}" if branch else str(num)
        head = f"제{num}조의{branch}" if branch else f"제{num}조"
        conn.execute(
            "INSERT INTO st_articles (statute_id, article_no, article_no_num, article_branch,"
            " title, article_text, article_changed, article_eff_date) VALUES (?,?,?,?,?,?,?,?)",
            (sid, no, num, branch, title, f"{head}({title}) {title} 본문.", changed, eff))

    def full_text(sid, eff, titles, changed="Y"):
        # Contiguous numbering satisfies the full-snapshot density check.
        for num in range(1, 41):
            article(sid, num, 0, titles.get(num, f"조문{num}"), eff, changed)

    version(OLD_FULL, LAW, "19970101", "연혁", "제정")
    full_text(OLD_FULL, "19970101", {32: "옛 예고"})
    version(OLD_DELTA, LAW, "20070701", "연혁", "일부개정")
    article(OLD_DELTA, 32, 2, "옛 통지", "20070701")
    version(REVISION, LAW, "20080101", "연혁", "전부개정")
    full_text(REVISION, "20080101", {26: "예고", 27: "통지", 32: "구제명령"})
    version(SNAPSHOT, LAW, "20251001", None, None)
    full_text(SNAPSHOT, "20251001",
              {26: "예고", 27: "통지", 32: "구제명령", 40: "개정 전"}, changed=None)
    version(NEW_DELTA, LAW, "20260801", "현행", "타법개정")
    article(NEW_DELTA, 40, 0, "개정 후", "20260801")

    # A current row containing only one amendment cannot establish membership.
    version(201, SPARSE_LAW, "20000101", "연혁", "제정", name="부분법")
    for num in range(1, 11):
        article(201, num, 0, f"원문{num}", "20000101")
    version(202, SPARSE_LAW, "20200101", "현행", "일부개정", name="부분법")
    article(202, 5, 0, "원문5(개정)", "20200101")

    # A 49-article delta is larger than the original 40-article full text,
    # but omits article 31 precisely because that article was left unchanged.
    version(301, LARGE_LAW, "19890101", "연혁", "제정", name="큰개정법")
    full_text(301, "19890101", {31: "안 고친 조문"})
    version(302, LARGE_LAW, "19950101", "연혁", "일부개정", name="큰개정법")
    for num in range(41, 51):
        article(302, num, 0, f"신설{num}", "19950101")
    version(303, LARGE_LAW, "20040701", "연혁", "일부개정", name="큰개정법")
    for num in range(1, 51):
        if num != 31:
            article(303, num, 0, f"고친{num}", "20040701")
    version(LARGE_SNAPSHOT, LARGE_LAW, "20250101", None, None, name="큰개정법")
    for num in range(1, 51):
        article(LARGE_SNAPSHOT, num, 0, "안 고친 조문" if num == 31 else f"고친{num}",
                "20250101", changed=None)

    # One delayed article already appears in a promulgated amendment. Another
    # future article reuses a vanished number, and a third number is entirely new.
    version(401, PENDING_LAW, "19900101", "연혁", "제정", name="예정법")
    full_text(401, "19900101", {})
    article(401, 10, 2, "옛 제도", "19900101")
    version(402, PENDING_LAW, "20240601", "연혁", "일부개정", name="예정법")
    article(402, 41, 2, "새 제도", "20240601")
    version(PENDING_SNAPSHOT, PENDING_LAW, "20250101", None, None, name="예정법")
    full_text(PENDING_SNAPSHOT, "20250101", {}, changed=None)
    version(404, PENDING_LAW, "20991231", "시행예정", "일부개정", name="예정법")
    article(404, 41, 2, "새 제도", "20991231")
    article(404, 10, 2, "다른 새 제도", "20991231")
    article(404, 50, 0, "처음 생기는 조문", "20991231")
    conn.commit()
    yield conn
    conn.close()


def _lookup(conn, articles, offense_date=None, sid=SNAPSHOT):
    resp = statutes._statute_lookup_impl(conn, None, (False, sid), articles, 10,
                                         offense_date=offense_date)
    return resp, statutes._format_response_md({"status": "ok", **resp})


def _numbers(resp):
    return [(a["no_num"], a.get("branch") or 0) for a in resp["articles"]]


def test_current_lookup_excludes_branches_removed_by_a_full_revision(corpus):
    resp, md = _lookup(corpus, ["32"])
    assert _numbers(resp) == [(32, 0)]
    assert "구제명령" in md
    assert "옛 통지" not in md


def test_a_vanished_number_explains_how_to_find_its_old_text(corpus):
    resp, md = _lookup(corpus, ["32의2"])
    assert resp["articles"] == []
    assert resp["missing"] == ["32-2"]
    assert resp["vanished"] == {"32-2": "20070701"}
    assert "2007-07-01판까지" in md and "현행에는 없는 옛 번호" in md
    assert "조문 개요" in md and "offense_date" in md
    assert "현행·연혁본 어디에도" not in md
    assert "제32조의2" not in md.split("근처 조문:")[1]


def test_an_amendment_after_the_snapshot_still_wins(corpus):
    resp, _ = _lookup(corpus, ["40"])
    assert [a["title"] for a in resp["articles"]] == ["개정 후"]


def test_dated_lookup_preserves_the_pre_revision_article(corpus):
    before, _ = _lookup(corpus, ["32"], offense_date="2007-08-01")
    after, md_after = _lookup(corpus, ["32"], offense_date="2015-01-01")
    gone, md_gone = _lookup(corpus, ["32의2"], offense_date="2015-01-01")
    assert _numbers(before) == [(32, 0), (32, 2)]
    assert [a["title"] for a in before["articles"]] == ["옛 예고", "옛 통지"]
    assert _numbers(after) == [(32, 0)]
    assert "구제명령" in md_after
    assert gone["vanished"] == {"32-2": "20070701"}
    assert "2015-01-01 시점에는 없는 옛 번호" in md_gone
    assert "같은 offense_date 로 조문 개요" in md_gone and "statute_id 만으로" not in md_gone


def test_dated_outline_lists_only_articles_in_force_then(corpus):
    resp, _ = _lookup(corpus, None, offense_date="2015-01-01")
    keys = set(_numbers(resp))
    assert (32, 2) not in keys
    assert {(26, 0), (27, 0), (32, 0)} <= keys


@pytest.mark.parametrize("change_kind", ["일부개정", "타법개정"])
def test_a_large_partial_amendment_keeps_unchanged_articles(corpus, change_kind):
    corpus.execute("UPDATE st_statutes SET change_kind=? WHERE id=303", (change_kind,))
    resp, md = _lookup(corpus, ["31"], offense_date="2004-07-15", sid=LARGE_SNAPSHOT)
    outline, _ = _lookup(corpus, None, offense_date="2004-07-15", sid=LARGE_SNAPSHOT)
    assert [a["title"] for a in resp["articles"]] == ["안 고친 조문"]
    assert "vanished" not in resp and "없는 옛 번호" not in md
    assert (31, 0) in _numbers(outline)


def test_a_sparse_base_keeps_the_historical_walk(corpus):
    assert statutes._articles_in_force(corpus, SPARSE_LAW, "20261001") is None
    resp, _ = _lookup(corpus, ["3"], sid=202)
    changed, _ = _lookup(corpus, ["5"], sid=202)
    assert [a["title"] for a in resp["articles"]] == ["원문3"]
    assert [a["title"] for a in changed["articles"]] == ["원문5(개정)"]


def test_a_delayed_article_is_pending_and_readable_on_its_effective_date(corpus):
    resp, md = _lookup(corpus, ["41의2"], sid=PENDING_SNAPSHOT)
    then, _ = _lookup(corpus, ["41의2"], offense_date="2099-12-31", sid=PENDING_SNAPSHOT)
    assert resp["articles"] == [] and resp["missing"] == ["41-2"]
    assert resp["pending"] == {"41-2": "20991231"}
    assert "vanished" not in resp
    assert "41-2(2099-12-31 시행 예정): 아직 시행 전인 조문입니다" in md
    assert "옛 번호" not in md and "어디에도" not in md
    assert [a["title"] for a in then["articles"]] == ["새 제도"]


def test_a_reused_number_reports_both_the_old_and_pending_article(corpus):
    reused, md_reused = _lookup(corpus, ["10의2"], sid=PENDING_SNAPSHOT)
    fresh, md_fresh = _lookup(corpus, ["50"], sid=PENDING_SNAPSHOT)
    early, md_early = _lookup(corpus, ["41의2"], offense_date="2020-01-01", sid=PENDING_SNAPSHOT)
    assert reused["vanished"] == {"10-2": "19900101"}
    assert reused["pending"] == {"10-2": "20991231"}
    assert "10-2(1990-01-01판까지): 현행에는 없는 옛 번호" in md_reused
    assert "10-2(2099-12-31 시행 예정): 아직 시행 전인 조문" in md_reused
    assert fresh["pending"] == {"50": "20991231"} and "vanished" not in fresh
    assert "어디에도" not in md_fresh
    assert early["pending"] == {"41-2": "20991231"}
    assert "2020-01-01 시점에는 시행 전인 조문입니다" in md_early
