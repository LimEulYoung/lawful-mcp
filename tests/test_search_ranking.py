"""Ranking and parsing contracts for the two corpus search tools.

Unlike the smoke tests, these do not read data/fixture.db: a name collision
and a malformed holding are easier to state outright than to find in a sample.
No network and no API keys either.
"""
from __future__ import annotations

import re
import sqlite3

from lawful_mcp.tools import statutes
# Imported by name: `tools/__init__` exports the tool function under the same
# name as its module, so `from ... import precedent_search` gives the function.
from lawful_mcp.tools.precedent_search import (
    HOLDING_MAX_CHARS,
    _format_response_md,
    _holdings_items,
    _pick_holding,
    _set_holding,
)


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


# ---------- holdings ----------
#
# A holding is the court's own summary of the issue, and 81,488 cases carry
# one (91.7% of the Supreme Court's). It is in no search index, so it is only
# ever used to *choose* among results already retrieved. Truncating one by
# character count halves a legal proposition, and taking the first item alone
# answers a different question than the one asked 59% of the time (over 3,015
# production cases the best-matching item was the first 40.7%, the second
# 39.3%, the third or later 17.1%).


def test_holdings_split_uses_br_not_the_numbering():
    """`<br/>` owns the boundary; the labels only check it.

    A long item arrives broken across lines (1,206 cases). Counting a fragment
    that starts without a label as a new item halves a proposition.
    """
    items = _holdings_items(
        "[1] 첫째 쟁점이다<br/>[2] 둘째 쟁점인데 길어서<br/>줄이 넘어간 뒷부분이다")
    assert len(items) == 2, items
    assert items[0] == "[1] 첫째 쟁점이다"
    assert items[1] == "[2] 둘째 쟁점인데 길어서 줄이 넘어간 뒷부분이다"


def test_holdings_split_is_lossless():
    """Rejoined, the items read as the original — losing text while dividing it
    is worse than not showing it."""
    raw = "［1］ 전각 대괄호도 같은 표기다<br/>［2］ 둘째 항이다"
    items = _holdings_items(raw)
    assert len(items) == 2, items
    joined = re.sub(r"\s+", "", " ".join(items))
    original = re.sub(r"\s+", "", raw.replace("<br/>", " ")
                      .replace("［", "[").replace("］", "]"))
    assert joined == original


def test_holdings_inline_ganada_does_not_split_on_sentence_ending():
    """'가. … 나. …' also arrives on one line. The '다.' ending a sentence is
    not an item.

    This is why the search follows the expected order. Reading a sentence-final
    '다.' as an item head splits one item at every sentence.
    """
    items = _holdings_items(
        "가. 첫째 쟁점에 관하여 그렇게 판단하였다. 나. 둘째 쟁점의 판단 기준")
    assert len(items) == 2, items
    assert items[0] == "가. 첫째 쟁점에 관하여 그렇게 판단하였다."
    assert items[1] == "나. 둘째 쟁점의 판단 기준"
    # Some cases run them together, as in '(소극)다. …'.
    assert len(_holdings_items(
        "가. 첫째 쟁점 나. 둘째 쟁점(소극)다. 셋째 쟁점")) == 3
    # No labels at all means one item (59.7% of the corpus).
    assert _holdings_items("단일 쟁점만 있는 판시사항이다.") == \
        ["단일 쟁점만 있는 판시사항이다."]
    assert _holdings_items("") == []


def test_holdings_keeps_the_shared_preamble():
    """Whatever precedes the first item survives.

    Constitutional Court holdings sometimes open with a preamble shared by
    every item ('1. … 가. … 나. …'). Dropped, the items left have nothing to
    say what they are about; it showed up as three cases losing text in the
    full-corpus check.
    """
    items = _holdings_items(
        "1. 이 사건 심판대상 조항 중 가. 첫째 부분 나. 둘째 부분에 관한 판단")
    assert items[0].startswith("1. 이 사건 심판대상 조항 중 "), items
    assert len(items) == 2
    # Rejoined they read as the original — the property that rules out loss.
    assert re.sub(r"\s+", "", " ".join(items)) == \
        re.sub(r"\s+", "", "1. 이 사건 심판대상 조항 중 가. 첫째 부분 나. 둘째 부분에 관한 판단")


def test_holding_pick_follows_the_query_not_the_order():
    """The item the query points at, rather than the first one."""
    raw = "[1] 소멸시효 기산점에 관한 법리<br/>[2] 유치권 피담보채권의 범위"
    assert _pick_holding(raw, ["유치권", "피담보채권"]).startswith("[2]")
    assert _pick_holding(raw, ["소멸시효"]).startswith("[1]")
    # Matching no item at all (2.9% of searches) leaves the first as
    # representative.
    assert _pick_holding(raw, ["전혀무관한말"]).startswith("[1]")
    assert _pick_holding("", ["아무거나"]) == ""


def test_holding_is_capped_and_says_so():
    """A trailing `…` reports the cut, and the cap covers the polluted rows —
    the 36 cases holding a whole judgment in the column."""
    long_item = "[1] " + ("가" * 900)
    out = _pick_holding(long_item, [])
    assert len(out) == HOLDING_MAX_CHARS + 1, len(out)
    assert out.endswith("…")
    # An item inside the cap is untouched, which is most of them.
    short = "[1] 짧은 쟁점"
    assert _pick_holding(short, []) == short


def test_holding_absent_when_the_case_has_none():
    """A case with no holding gets no field — an empty line would still cost
    its place."""
    m = {"_holdings": "", "id": 1}
    _set_holding(m, ["아무거나"])
    assert "holding" not in m
    rendered = _format_response_md({
        "status": "ok",
        "matches": [{"id": 1, "case_number": "2020다1", "case_name": "손해배상",
                     "court_name": "대법원", "court_level": "대법원", "year": 2020,
                     "holding": "[1] 법원이 쓴 쟁점 요약", "preview": "본문 발췌",
                     "preview_provenance": "original_text_excerpt"},
                    {"id": 2, "case_number": "2020고단1", "case_name": "사기",
                     "court_name": "서울중앙지방법원", "court_level": "1심", "year": 2020,
                     "preview": "본문 발췌", "preview_provenance": "original_text_excerpt"}],
    })
    assert "  holding: [1] 법원이 쓴 쟁점 요약" in rendered
    assert rendered.count("holding:") == 1, rendered
