"""Ranking contracts for statute search.

Unlike the smoke tests, these do not read data/fixture.db: a collision between
two names is easier to state outright than to find in a sample. No network and
no API keys either.
"""
from __future__ import annotations

import sqlite3

from legal_search_mcp.tools import statutes


# ---------- a query that names a law ----------
#
# Real queries read 「law name + topic」, which switched exact matching off and
# left token coverage as the only measure — under which an administrative rule
# long enough to carry the topic words in its own name outranked the law the
# query pointed at. Over fourteen days of traffic, 322 of the 1,768 searches
# naming a law put an unrelated document first, and the model then fetched
# articles by that id, so the wrong law's text reached the answer. The corpus
# below is that collision, reduced.


def _naming_corpus():
    """A minimum corpus of colliding names: on token coverage the rule wins."""
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
        CREATE VIRTUAL TABLE st_articles_fts USING fts5(
            title, article_text, content='st_articles', content_rowid='id',
            tokenize='trigram');
        CREATE TABLE st_notices (
            id INTEGER PRIMARY KEY, serial_id TEXT, notice_id TEXT, name TEXT,
            kind TEXT, issuing_agency TEXT, notice_no TEXT, issued_date TEXT,
            effective_date TEXT, category TEXT, has_articles INTEGER,
            has_text_content INTEGER, history_status TEXT, body_source TEXT
        );
        CREATE TABLE st_notice_articles (
            id INTEGER PRIMARY KEY, notice_id INTEGER, article_seq INTEGER,
            article_no INTEGER, article_no_str TEXT, article_text TEXT
        );
        CREATE VIRTUAL TABLE st_notice_articles_fts USING fts5(
            article_text, content='st_notice_articles', content_rowid='id',
            tokenize='trigram');
    """)
    laws = [
        # Reachable only by its abbreviation: the full name holds neither
        # '신용카드' nor '발급'.
        (1, "000001", "신용정보의 이용 및 보호에 관한 법률", "신용정보법"),
        # A short name inside a longer one — length comparison decides.
        (2, "000002", "형법", None),
        (3, "000003", "군형법", None),
    ]
    for sid, lid, name, short in laws:
        conn.execute(
            "INSERT INTO st_statutes (id, law_id, name, short_name, kind,"
            " effective_date, history_status) VALUES (?,?,?,?,'법률','20200101','현행')",
            (sid, lid, name, short))
        conn.execute(
            "INSERT INTO st_articles (statute_id, article_no, article_no_num,"
            " article_branch, title, article_text) VALUES (?,'제1조',1,0,'목적',?)",
            (sid, f"제1조(목적) {name}의 목적을 정한다."))
    conn.execute("INSERT INTO st_articles_fts(st_articles_fts) VALUES('rebuild')")
    # The rule whose own name carries every topic word: on coverage it wins.
    conn.execute(
        "INSERT INTO st_notices (id, notice_id, name, kind, issuing_agency, category,"
        " has_articles, has_text_content, history_status) VALUES"
        " (1, 'N1', '신용카드 발급 주민등록번호 처리 지침', '훈령', '금융위원회',"
        " 'article_form', 1, 1, '현행')")
    conn.execute(
        "INSERT INTO st_notice_articles (id, notice_id, article_seq, article_no,"
        " article_no_str, article_text) VALUES"
        " (1, 1, 1, 1000, '1', '제1조(목적) 신용카드 발급 시 주민등록번호 처리를 정한다.')")
    conn.execute("INSERT INTO st_notice_articles_fts(st_notice_articles_fts)"
                 " VALUES('rebuild')")
    conn.commit()
    return conn


def test_named_statute_wins_over_topic_matching_notice():
    """A law the query named outranks a rule that merely shares topic words.

    Called through `_statute_lookup_impl` on purpose: if the search ranks it
    right and the merge (`_name_cover_key`) puts it back, what the model sees
    is still the wrong first result. This checks both places at once.
    """
    conn = _naming_corpus()
    try:
        out = statutes._statute_lookup_impl(
            conn, "신용정보법 신용카드 발급 주민등록번호", None, None, 5)
    finally:
        conn.close()
    names = [m.get("name") for m in out["matches"]]
    assert names, "no candidates at all would make this test vacuous"
    assert names[0] == "신용정보의 이용 및 보호에 관한 법률", names
    # The rule does not disappear, it just sorts below — the interleave has no
    # fixed share.
    assert "신용카드 발급 주민등록번호 처리 지침" in names


def test_longest_contained_name_wins():
    """'군형법 제92조' contains both '형법' and '군형법'; the answer is the longer."""
    conn = _naming_corpus()
    try:
        gun = [m["name"] for m in statutes._search_statutes(conn, "군형법 제92조", 5)]
        hyeong = [m["name"] for m in statutes._search_statutes(conn, "형법 제347조 사기", 5)]
    finally:
        conn.close()
    assert gun[0] == "군형법", gun
    assert hyeong[0] == "형법", hyeong


def _key(name, query):
    q_norm, tokens = statutes._statute_name_tokens(query)
    return statutes._name_cover_key(
        name, q_norm, tokens, frozenset(statutes._query_name_spans(query)))


def test_merge_key_matches_sql_ranking():
    """The merge key and the SQL sort are one measure — apart, the merge undoes
    the search."""
    q = "신용정보법 신용카드 발급 주민등록번호"
    named = _key("신용정보의 이용 및 보호에 관한 법률", q)
    by_short = _key("신용정보법", q)
    topical = _key("신용카드 발급 주민등록번호 처리 지침", q)
    assert by_short < topical, "a name the query contains outranks topic coverage"
    assert min(named, by_short) < topical
    # An exact match is containment's maximum: first without a branch of its own.
    assert _key("형법", "형법") < _key("군형법", "형법")
    # Interpuncts come out here as they do in SQL (`_name_norm_sql`).
    assert _key("초ㆍ중등교육법", "초중등교육법 학교운영위원회")[0] == 0


def test_name_match_does_not_cross_word_boundaries():
    """'손해배상 법률' contains 「상법」 by letters alone — a coincidence across a
    word boundary is not the query naming a law.

    Without this branch the top result for a damages question is 「상법」.
    """
    spans = statutes._query_name_spans("손해배상 법률 상담 절차")
    assert "상법" not in spans, spans
    assert _key("상법", "손해배상 법률 상담 절차")[0] != 0
    assert "민법" not in statutes._query_name_spans("국민법률구조공단 지원 대상")
    # What does stand on a boundary is still a candidate: '형법' inside
    # '군형법 제92조' is one, and length decides between them.
    assert "형법" in statutes._query_name_spans("형법 제347조 사기")
    assert {"군형법", "형법"} <= set(statutes._query_name_spans("군형법 제92조"))
    # An everyday name reaches the candidates expanded to its full form.
    assert "대한민국헌법" in statutes._query_name_spans("헌법")
    # Values in the alias table keep their interpuncts. Left in, they never
    # equal the normalised name in SQL and the everyday name quietly misses.
    assert "총포도검화약류등의안전관리에관한법률" in \
        statutes._query_name_spans("총포도검화약류등단속법")
