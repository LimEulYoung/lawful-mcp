"""statute_lookup on the sample database: matched articles in the list, partial matches, one call to an article,
search inside a statute, version dates, addenda and repealed laws."""

import sqlite3

from lawful_mcp.harness_local.deps import HarnessDeps
from lawful_mcp.harness_local.tools import statutes


class Ctx:
    deps = HarnessDeps(dive_subagent=None)


def lookup(**args):
    return statutes.statute_lookup(Ctx(), **args)


def test_a_keyword_search_shows_the_article_it_matched():
    out = lookup(query="개인정보 보호법 주민등록번호")
    assert "## mode: list" in out and "맞은 조문: 제24조의2(주민등록번호 처리의 제한)" in out


def test_without_an_article_holding_every_word_the_search_takes_the_most_and_says_so():
    out = lookup(query="근로기준법 해고예고 서면통지 사용자")
    assert "'해고예고'·'서면통지'·'사용자'가 모두 든 조문이 없어, 그중 2개가 함께 든 조문으로 찾았습니다" in out
    assert "맞은 조문: 제27조(해고사유 등의 서면통지)" in out


def test_a_query_naming_one_statute_with_article_numbers_reaches_the_text():
    for out in (lookup(query="근로기준법 제26조"), lookup(query="근로기준법", articles=["26"])):
        assert "## mode: detail" in out and "query 가 쓴 「근로기준법」" in out and "제26조(해고의 예고)" in out
    listed = lookup(query="해고 예고", articles=["26"])
    assert "## mode: list" in listed and "query 가 법령 하나를 이름으로 대지 않아" in listed


def test_statute_id_with_a_query_finds_that_statutes_articles():
    out = lookup(statute_id=468, query="영토")
    assert "## mode: detail" in out and "제3조 대한민국의 영토는" in out and "query 낱말('영토')" in out
    other = lookup(statute_id=468, query="근로기준법 제26조")
    assert "statute_id=468(「대한민국헌법」)는 query 가 쓴 법령이 아니어서" in other and "제26조(해고의 예고)" in other


def test_a_current_lookup_states_the_version_its_text_is_from():
    assert "- 판본: 현행 문언(" in lookup(statute_id=468, articles=["1"])
    assert "- 판본: 현행 조문 목록(" in lookup(statute_id=675)
    dated = lookup(statute_id=468, articles=["1"], offense_date="2000-01-01")
    assert "시점 조회: 2000-01-01" in dated and "판본: 현행" not in dated


def test_a_repealed_law_in_the_list_is_marked_repealed_not_pointed_to_an_old_version():
    match = {"statute_id": 1, "law_id": "S", "name": "증권거래법", "is_repealed": True,
             "current": {"id": 2, "name": "증권거래법", "effective_date": "19620115", "renamed": False}}
    out = statutes._format_response_md({"status": "ok", "mode": "list", "matches": [match]})
    assert "폐지된 법령 — 현행 후속본이 없습니다" in out and "현행 아님" not in out


def test_addenda_are_looked_up_where_the_corpus_has_them():
    assert statutes._articles_in_query("민법 제390조 제482조의2") == ["390", "482의2"]
    assert statutes._articles_in_query("어떤 법 부칙 제51조") == ["부칙 51"]
    assert statutes._addenda_specs(["부칙", "부칙 제3조의2", "13"]) == [(None, None), (3, 2)]
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE st_addenda (law_id TEXT, promul_date TEXT, promul_no TEXT, article_no INTEGER,
                                 article_branch INTEGER, title TEXT, article_text TEXT);
        INSERT INTO st_addenda VALUES ('L', '20100101', '100', 1, 0, '시행일', '제1조(시행일) 공포한 날부터 시행한다.'),
            ('L', '20100101', '100', 2, 0, '경과조치', '제2조(경과조치) 종전의 규정에 따른다.'),
            ('L', '20200101', '200', 2, 0, '적용례', '제2조(적용례) 이 법 시행 후부터 적용한다.');
    """)
    listed = statutes._addenda(conn, "L", [(None, None)], "20261103")
    assert [g["date"] for g in listed["addenda_list"]] == ["20200101", "20100101"]
    old = statutes._addenda(conn, "L", [(2, None)], "20150101")["addenda"]
    assert [a["label"] for a in old] == ["제2조(경과조치)"]
    assert "코퍼스에 없습니다" in statutes._addenda(conn, "X", [(2, None)], "20261103")["addenda_note"]
    # lawful-mcp: the bundled sample ships its laws' addenda (the research sample has none), so the list comes
    # back; '부칙' is still not called an excluded unit.
    out = lookup(statute_id=468, articles=["1", "부칙"])
    assert "## 부칙 목록" in out and "제외했습니다" not in out
    assert "별표1 는 제외했습니다" in lookup(statute_id=468, articles=["1", "별표1"])
