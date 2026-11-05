"""Picking a statute version, dated-lookup notes and the court-name filter.

Versions are told apart by **date and load shape**, not by label ('시행예정',
'연혁'): a label flips only when the collector next runs, after the effective
date, and an old history delta row (an mst but an empty serial_id) holds only
the articles its amendment changed. The corpus below is that shape in small:
enactment (full text) -> old delta -> version loaded whole (serial_id) -> large
old delta -> full version past its effective date but still labelled
'시행예정' -> version not yet in force -> version with no date yet (9999).
"""
from __future__ import annotations

import asyncio
import importlib
import sqlite3
from types import SimpleNamespace

statutes = importlib.import_module("lawful_mcp.tools.statutes")

# The package exports functions under the module names, so modules come through import_module.
precedent_search = importlib.import_module("lawful_mcp.tools.precedent_search")
dive_module = importlib.import_module("lawful_mcp.tools.precedent_dive")

LAW = "005555"
_ENACT, _DELTA_2010, _WHOLE_2015, _BIG_DELTA_2016, _LAGGED_2026 = 1, 2, 3, 4, 5
_PENDING, _UNDATED = 6, 7
_N = 40


def _corpus():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE st_statutes (
            id INTEGER PRIMARY KEY, law_id TEXT, name TEXT, short_name TEXT,
            kind TEXT, issuing_agency TEXT, promulgation_date TEXT,
            effective_date TEXT, history_status TEXT, change_kind TEXT, mst TEXT, serial_id TEXT
        );
        CREATE TABLE st_articles (
            id INTEGER PRIMARY KEY, statute_id INTEGER, article_no TEXT,
            article_no_num INTEGER, article_branch INTEGER, title TEXT,
            article_text TEXT, article_changed TEXT, article_eff_date TEXT
        );
        CREATE TABLE st_backfill_log (
            law_name TEXT PRIMARY KEY, status TEXT, law_id TEXT, mst TEXT,
            statute_id INTEGER, n_articles INTEGER, cite_count INTEGER,
            note TEXT, done_at TEXT);
    """)

    def version(sid, eff, status, kind, mst, serial=None):
        conn.execute(
            "INSERT INTO st_statutes (id, law_id, name, kind, effective_date, history_status,"
            " change_kind, mst, serial_id) VALUES (?, ?, '판본법', '법률', ?, ?, ?, ?, ?)",
            (sid, LAW, eff, status, kind, mst, serial))

    def article(sid, num, title, eff, changed="Y", text=None):
        conn.execute(
            "INSERT INTO st_articles (statute_id, article_no, article_no_num, article_branch,"
            " title, article_text, article_changed, article_eff_date) VALUES (?,?,?,0,?,?,?,?)",
            (sid, str(num), num, title, text or f"제{num}조({title}) {title} 본문.", changed, eff))

    def full_text(sid, eff, titles, extra=(), changed="Y"):
        for num in range(1, _N + 1):
            article(sid, num, titles.get(num, f"조문{num}"), eff, changed)
        for num in extra:
            article(sid, num, titles.get(num, f"조문{num}"), eff, changed)

    version(_ENACT, "20000101", "연혁", "제정", "100")
    # An indented chapter heading row — it shares the next article's number (30).
    article(_ENACT, 30, None, "20000101", text="      제3장 벌칙")
    full_text(_ENACT, "20000101", {}, changed=None)   # baseline articles with no amendment mark (like 형법 제1조)
    # Old delta: a row amending article 5 only (an mst, no serial_id).
    version(_DELTA_2010, "20100101", "연혁", "일부개정", "200")
    article(_DELTA_2010, 5, "조문5(2010)", "20100101")
    # A version loaded whole (serial_id) — adds article 41. Labelled '연혁'.
    version(_WHOLE_2015, "20150101", "연혁", "일부개정", "300", serial="300")
    full_text(_WHOLE_2015, "20150101", {5: "조문5(2010)", 41: "신설41"}, extra=(41,))
    # A large old delta — more articles than the full version (44 > 41), but only
    # those its amendment touched. Article 44 is a deferred article printed early
    # under the promulgating version's date (소득세법 제164조의4).
    version(_BIG_DELTA_2016, "20160101", "연혁", "일부개정", "400")
    for num in [*range(1, 31), *range(31, _N + 1), 41, 42, 43, 44]:
        if num in (42, 43):
            continue
        article(_BIG_DELTA_2016, num, f"고친{num}" if num != 44 else "미뤄진44", "20160101")
    # A full version past its effective date, still labelled '시행예정'.
    version(_LAGGED_2026, "20260101", "시행예정", "일부개정", "500", serial="500")
    full_text(_LAGGED_2026, "20260101", {5: "조문5(2026)", 41: "신설41"}, extra=(41,), changed="N")
    # A version not yet in force (where article 44 really starts) and one with no date yet.
    version(_PENDING, "20991231", "시행예정", "일부개정", "600", serial="600")
    article(_PENDING, 44, "미뤄진44", "20991231")
    version(_UNDATED, "99991231", "시행예정", "일부개정", "700", serial="700")
    article(_UNDATED, 45, "날짜없는45", "99991231")
    conn.commit()
    return conn


def _group(conn):
    return conn.execute("SELECT id, effective_date, history_status, change_kind, mst, serial_id "
                        "FROM st_statutes WHERE law_id=?", (LAW,)).fetchall()


def _lookup(conn, articles, offense_date=None, sid=_LAGGED_2026):
    resp = statutes._statute_lookup_impl(conn, None, (False, sid), articles, 10, offense_date=offense_date)
    return resp, statutes._format_response_md({"status": "ok", **resp})


def test_current_version_is_decided_by_date_not_by_the_lagging_label():
    """A version past its effective date is current even while still labelled '시행예정'.
    Choosing by label drops a law's latest amendment whole (124 versions,
    형사소송법's 2026-10-02 amendment among them, went out as not yet in force)."""
    conn = _corpus()
    try:
        chosen = statutes._pick_current_version(conn, _group(conn), statutes._today_iso())
        resp, md = _lookup(conn, ["5"])
    finally:
        conn.close()
    assert chosen["id"] == _LAGGED_2026
    assert [a["title"] for a in resp["articles"]] == ["조문5(2026)"]


def test_the_latest_full_text_beats_a_bigger_old_delta():
    """The baseline is the latest full text, not the row with the most articles.
    Taking a large old delta as the baseline hides every article that
    amendment did not touch."""
    conn = _corpus()
    try:
        chosen = statutes._pick_current_version(conn, _group(conn), "20170101")
    finally:
        conn.close()
    assert chosen["id"] == _WHOLE_2015


def test_point_in_time_outline_lists_never_amended_articles_and_skips_headings():
    """A dated outline lists every article alive then, not only amended ones
    (amended-only gave 헌법 zero articles). An indented chapter heading row
    does not take an article's place."""
    conn = _corpus()
    try:
        resp = statutes._outline_statute(conn, _WHOLE_2015, offense_date="2012-01-01")
    finally:
        conn.close()
    nums = [a["no_num"] for a in resp["articles"]]
    assert nums == list(range(1, _N + 1))
    assert next(a for a in resp["articles"] if a["no_num"] == 30)["title"] == "조문30"
    assert next(a for a in resp["articles"] if a["no_num"] == 5)["title"] == "조문5(2010)"


def test_a_date_before_the_earliest_version_says_so_and_serves_the_current_text():
    conn = _corpus()
    try:
        resp, md = _lookup(conn, ["5"], offense_date="1990-01-01")
    finally:
        conn.close()
    assert resp["no_version_at"] == {"date": "19900101", "earliest": "20000101"}
    assert "시점 조회 불가" in md and "가장 이른 판본 2000-01-01 시행" in md
    assert [a["title"] for a in resp["articles"]] == ["조문5(2026)"]


def test_a_number_added_later_is_not_called_nowhere_or_pending():
    """A number added after the asked date and in force today is "added in that
    version". "Nowhere" sends the model away from today's article, and
    "pending" would be false."""
    conn = _corpus()
    try:
        resp, md = _lookup(conn, ["41"], offense_date="2012-01-01")
    finally:
        conn.close()
    assert resp["added"] == {"41": "20150101"}
    assert "pending" not in resp
    assert "41(2015-01-01판부터): 2012-01-01 시점에는 아직 없던 번호" in md
    assert "어디에도" not in md


def test_a_deferred_article_printed_early_in_an_old_delta_stays_pending():
    """Old delta rows print a deferred article under the promulgating version's
    date (소득세법 제164조의4: a 2025-01-01 row, in force 2027-01-01). A number
    not alive today is pending, not "added in that version"."""
    conn = _corpus()
    try:
        resp, md = _lookup(conn, ["44"], offense_date="2012-01-01")
    finally:
        conn.close()
    assert "added" not in resp
    assert resp["pending"] == {"44": "20991231"}
    assert "44(2099-12-31 시행 예정)" in md


def test_an_undated_pending_version_is_not_given_as_a_date_to_look_up():
    conn = _corpus()
    try:
        resp, md = _lookup(conn, ["45"])
    finally:
        conn.close()
    assert resp["pending"] == {"45": "99991231"}
    assert "9999" not in md
    assert "45: 시행일이 정해지지 않은 개정본에만 있는 번호입니다" in md


def test_an_unreadable_offense_date_is_refused_not_answered_for_today():
    """'2019. 3. 8.' and '2013년 7월 30일' are read; a date still unreadable is
    refused rather than answered with current text."""
    ctx = SimpleNamespace(deps=SimpleNamespace(), usage=None)
    md = statutes.statute_lookup(ctx, statute_id=1, articles=["1"], offense_date="작년 봄")
    assert md.startswith("## status: bad_offense_date")
    assert statutes.to_iso_date("2019. 3. 8.") == "20190308"
    assert statutes.to_iso_date("2013년 7월 30일") == "20130730"


def test_short_words_say_the_body_index_cannot_see_them():
    msg = statutes._short_words_message("절도", [])
    assert msg and "'절도'" in msg and "세 글자부터" in msg
    assert statutes._short_words_message("민법", [{"name": "민법"}]) is None
    assert statutes._short_words_message("부당해고 구제", []) is None


def _courts(names):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE prec_cases (id INTEGER PRIMARY KEY, court_name TEXT)")
    conn.executemany("INSERT INTO prec_cases (court_name) VALUES (?)", [(n,) for n in names])
    return conn


def test_court_name_matches_every_spelling_of_one_court():
    """The corpus writes one court several ways: '서울고등법원' also takes the
    '서울고법' rows, and a branch named with its main court also takes rows
    carrying the branch alone. A LIKE filter missed the 4,394 '서울고법' rows."""
    conn = _courts(["서울고등법원", "서울고법", "서울행정법원", "수원지방법원 성남지원", "성남지원", "부산지방법원"])
    try:
        high, note = precedent_search._check_court_name(conn, "서울고등법원")
        branch, _ = precedent_search._check_court_name(conn, "수원지방법원 성남지원")
        short, _ = precedent_search._check_court_name(conn, "서울고법")
    finally:
        conn.close()
    assert high == ("서울고등법원", "서울고법") and note is None
    assert short == high
    assert branch == ("성남지원", "수원지방법원 성남지원")



def test_court_names_are_read_from_the_slim_copy_not_the_body_table():
    """Court names come from the body-less prec_meta. DISTINCT over prec_cases
    reads every judgment body, and the first court-filtered search after a
    restart took 59s. (The two tables carry the same names in a real corpus;
    they differ here on purpose, to tell the source apart.)"""
    conn = _courts(["본문표에만있는법원"])
    conn.execute("CREATE TABLE prec_meta (id INTEGER PRIMARY KEY, court_name TEXT)")
    conn.execute("INSERT INTO prec_meta (court_name) VALUES ('서울고법')")
    try:
        names, _ = precedent_search._check_court_name(conn, "서울고등법원")
        body_only, _ = precedent_search._check_court_name(conn, "본문표에만있는법원")
    finally:
        conn.close()
    assert names == ("서울고법",) and body_only is None


def test_an_unknown_court_name_drops_the_filter_with_a_note():
    """A name matching no court drops the filter with a note rather than
    returning a silent zero, as an unknown court_level does."""
    conn = _courts(["서울고등법원"])
    try:
        names, note = precedent_search._check_court_name(conn, "화성법원")
    finally:
        conn.close()
    assert names is None and "화성법원" in note


def test_an_unreadable_case_number_is_named():
    ctx = SimpleNamespace(deps=SimpleNamespace(), usage=None)
    md = precedent_search.precedent_search(ctx, case_number="2016두")
    assert "case_number '2016두' 에서 사건번호를 읽지 못했습니다" in md
    assert "case_number 인자에 넣으세요" not in md


def test_a_dive_provider_error_answers_unread_instead_of_failing_the_turn(monkeypatch):
    """A provider error (429, 5xx) on the summary request reads as "not read",
    like a timeout — it does not fail the turn."""

    class FakeConnection:
        def execute(self, _sql, _params):
            return self

        def fetchone(self):
            return {"case_number": "2024다1", "case_name": "사건", "court_name": "대법원",
                    "court_level": "대법원", "year": 2024, "reference_statute": "민법",
                    "content_md": "공개 판결문 본문"}

        def close(self):
            pass

    class FailingDiveAgent:
        async def run(self, prompt, *, usage=None):
            raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(dive_module, "open_db", lambda: FakeConnection())
    ctx = SimpleNamespace(deps=SimpleNamespace(recent_calls=[], dive_subagent=FailingDiveAgent()), usage=None)
    md = asyncio.run(dive_module.precedent_dive(ctx, case_id=1, question="쟁점"))
    assert md.startswith("## status: error")
    assert "본문 요약 요청이 실패했습니다(RuntimeError)" in md and "읽지 못한 것으로" in md


def _search_corpus():
    """For the search window: a law with many versions, a law with one, and a law repealed in 2009."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE st_statutes (
            id INTEGER PRIMARY KEY, law_id TEXT, name TEXT, short_name TEXT,
            kind TEXT, issuing_agency TEXT, promulgation_date TEXT,
            effective_date TEXT, history_status TEXT, change_kind TEXT, mst TEXT, serial_id TEXT
        );
        CREATE TABLE st_articles (
            id INTEGER PRIMARY KEY, statute_id INTEGER, article_no TEXT,
            article_no_num INTEGER, article_branch INTEGER, title TEXT,
            article_text TEXT, article_changed TEXT, article_eff_date TEXT
        );
        CREATE VIRTUAL TABLE st_articles_fts USING fts5(
            title, article_text, content='st_articles', content_rowid='id', tokenize='trigram');
        CREATE TABLE st_backfill_log (
            law_name TEXT PRIMARY KEY, status TEXT, law_id TEXT, mst TEXT,
            statute_id INTEGER, n_articles INTEGER, cite_count INTEGER,
            note TEXT, done_at TEXT);
    """)
    rows = [(i, "A", "근로기준법", f"{1900 + i:04d}0101", "연혁", "일부개정", str(i), str(i)) for i in range(1, 121)]
    rows.append((500, "B", "근로복지기본법", "20200101", "현행", "제정", "500", "500"))
    rows.append((600, "C", "증권거래법", "19970101", "연혁", "제정", "600", "600"))
    rows.append((601, "C", "증권거래법", "20090204", "연혁", "폐지", "601", "601"))
    # An old law only in the backfill register, with no repeal version and no successor.
    rows.append((700, "D", "근로임시조치령", "19470101", None, "제정", None, None))
    conn.executemany(
        "INSERT INTO st_statutes (id, law_id, name, kind, effective_date, history_status, change_kind,"
        " mst, serial_id) VALUES (?, ?, ?, '법률', ?, ?, ?, ?, ?)", rows)
    conn.executemany(
        "INSERT INTO st_articles (statute_id, article_no, article_no_num, article_branch, title,"
        " article_text) VALUES (?, '1', 1, 0, '목적', '제1조(목적) 이 법의 목적.')",
        [(r[0],) for r in rows])
    conn.execute("INSERT INTO st_backfill_log (law_name, status, law_id, statute_id) "
                 "VALUES ('근로임시조치령', 'ok', 'D', 700)")
    conn.execute("INSERT INTO st_articles_fts(st_articles_fts) VALUES('rebuild')")
    conn.commit()
    return conn


def test_the_search_window_holds_laws_not_version_rows():
    """Cutting the window by version rows lets a law with many versions fill it
    and push other laws out (238 of 583 recorded searches changed). The window
    counts laws."""
    conn = _search_corpus()
    try:
        names = [m["name"] for m in statutes._search_statutes(conn, "근로", 10)]
    finally:
        conn.close()
    # The shorter-named 근로기준법's 120 version rows come first, so an 80-row window left 근로복지기본법 out.
    assert names == ["근로기준법", "근로복지기본법"]


def test_a_dated_search_keeps_the_repealed_law_the_query_names():
    """A dated search drops laws already repealed by then, but keeps the one the
    query names, marked repealed — '증권거래법' as of 2015-03-01 used to return
    nothing."""
    conn = _search_corpus()
    try:
        named = statutes._search_statutes(conn, "증권거래법", 10, offense_date="2015-03-01")
        current = statutes._search_statutes(conn, "증권거래법", 10)
    finally:
        conn.close()
    assert [m["name"] for m in named] == ["증권거래법"]
    assert named[0]["is_repealed"]
    assert current == []


def test_a_current_search_drops_dead_laws_unless_the_query_names_them():
    """An old law marked '폐지된 법령' with no repeal version stays in a current
    search only when the query names it — the law-level window filled its free
    slots with such laws (replaying 653 recorded calls: lines marked repealed
    went 261 -> 371)."""
    conn = _search_corpus()
    try:
        brushed = [m["name"] for m in statutes._search_statutes(conn, "근로", 10)]
        named = statutes._search_statutes(conn, "근로임시조치령", 10)
    finally:
        conn.close()
    assert "근로임시조치령" not in brushed
    assert named[0]["name"] == "근로임시조치령" and named[0]["is_repealed"]
