"""statute_search and statute_read on the sample database: matched articles in the list, partial matches, one call to
articles, articles past the cap or joined in one string, arguments kept apart, version dates, addenda and repealed
laws."""

import os
import sqlite3

from lawful_mcp.harness_local.deps import HarnessDeps
from lawful_mcp.harness_local.tools import statutes


class Ctx:
    deps = HarnessDeps(dive_subagent=None)


def search(**args):
    return statutes.statute_search(Ctx(), **args)


def read(**args):
    return statutes.statute_read(Ctx(), **args)


def test_a_keyword_search_shows_the_article_it_matched():
    out = search(query="개인정보 보호법 주민등록번호")
    assert "## mode: list" in out and "맞은 조문: 제24조의2(주민등록번호 처리의 제한)" in out


def test_without_an_article_holding_every_word_the_search_takes_the_most_and_says_so():
    out = search(query="근로기준법 해고예고 서면통지 사용자")
    assert "'해고예고'·'서면통지'·'사용자'가 모두 든 조문이 없어, 그중 2개가 함께 든 조문으로 찾았습니다" in out
    assert "맞은 조문: 제27조(해고사유 등의 서면통지)" in out


def test_a_query_naming_one_statute_with_article_numbers_reaches_the_text():
    out = search(query="근로기준법 제26조")
    assert "## mode: detail" in out and "query 가 쓴 「근로기준법」" in out and "제26조(해고의 예고)" in out
    both = search(query="근로기준법 제26조 제27조")
    assert "제26조(해고의 예고)" in both and "제27조(해고사유 등의 서면통지)" in both


def test_articles_past_the_eighth_in_a_query_are_named():
    out = search(query="민법 " + " ".join(f"제{n}조" for n in range(750, 761)))
    assert [line[4:7] for line in out.splitlines() if line.startswith("### ")] == [str(n) for n in range(750, 758)]
    assert ("query 의 조문 11개 중 앞 8개만 조회했습니다(호출당 최대 8개). 나머지(758, 759, 760)는 "
            "statute_read(statute_id=584, articles=[…]) 로 부르세요.") in out


def test_articles_joined_in_one_string_are_all_read():
    joined = read(statute_id=578, articles="185,188")
    assert "### 185 일반교통방해" in joined and "### 188 교통방해치사상" in joined
    assert joined == read(statute_id=578, articles=["185", "188"])
    mixed = read(statute_id=578, articles=["제185조, 제188조", "250"])
    assert all(f"### {n} " in mixed for n in (185, 188, 250))
    # A JSON list behind a stray '=' (seen in production) and quoted parts read as the same articles.
    assert read(statute_id=578, articles='=["185", "188"]') == joined
    assert read(statute_id=578, articles="'185','188'") == joined
    # Not every part an article (a paragraph): left as one token, as before.
    assert statutes._split_joined_articles(["750조 제1항, 제2항"]) == ["750조 제1항, 제2항"]
    assert statutes._split_joined_articles(["347의2, 부칙 3, 별표 5"]) == ["347의2", "부칙 3", "별표 5"]
    assert statutes._split_joined_articles(["1, 부칙 제3조의2"]) == ["1", "부칙 제3조의2"]
    assert statutes._split_joined_articles([",", " , "]) == [",", " , "]


def test_search_and_read_take_no_argument_of_the_other():
    from pydantic_ai.tools import Tool

    params = {f.__name__: set(Tool(f).function_schema.json_schema["properties"])
              for f in (statutes.statute_search, statutes.statute_read)}
    assert params == {"statute_search": {"query", "offense_date"},
                      "statute_read": {"statute_id", "articles", "offense_date"}}
    assert search() == ("## status: missing_input\n- message: query(법령명·키워드)가 필요합니다. 받은 statute_id 의 "
                        "본문은 statute_read 로 보세요.")
    assert read() == ("## status: missing_input\n- message: statute_id(statute_search 가 준 식별자)가 필요합니다. "
                      "법령을 모르면 statute_search 로 찾으세요.")
    named = read(statute_id="근로기준법", articles=["26"])
    assert named.startswith("## status: bad_statute_id") and "statute_search 로 다시 찾으세요" in named


def test_a_very_long_outline_comes_as_its_table_of_contents():
    toc = read(statute_id=584)
    assert "## articles" not in toc and len(toc) < 5000
    assert ("## 목차 — 조문 1193개는 목록이 길어 편·장·절과 조문 범위만 보입니다. 본문은 articles로(최대 8개), "
            "조문 찾기는 statute_search(query='민법 낱말')로 부르세요.\n- 제1편 총칙 (제1조~제184조)\n"
            "  - 제2장 인 (제3조~제30조)\n    - 제2절 주소 (제18조~제21조)\n") in toc
    assert "\n  - 제7장 소멸시효 (제162조~제184조)\n" in toc and "\n      - 제5관 면제 (제506조)\n" in toc
    # Shorter outlines keep every article title; a law without 편 starts its contents at 장.
    assert "## articles (132)\n- 1: 목적\n" in read(statute_id=675)


def test_many_branch_articles_come_as_a_list_and_a_few_in_full():
    many = read(statute_id=584, articles=["959"])
    assert "### 959 위임규정의 준용" in many and "### 959의2" not in many
    assert '## 제959조의 가지 조문 19개 — 많아서 제목과 본문 앞부분만 보입니다. 본문은 articles로 부르세요(예: ["959의2"])' in many
    assert "\n- 959의2: 한정후견의 개시 — 가정법원의 한정후견개시의 심판이 있는 경우에는 그 심판을 받은 사람의 한…\n" in many
    few = read(statute_id=675, articles=["76"])
    assert "### 76의2" in few and "### 76의3" in few and "가지 조문" not in few


def test_a_current_lookup_states_the_version_its_text_is_from():
    assert "- 판본: 현행 문언(" in read(statute_id=468, articles=["1"])
    assert "- 판본: 현행 조문 목록(" in read(statute_id=675)
    dated = read(statute_id=468, articles=["1"], offense_date="2000-01-01")
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
    out = read(statute_id=468, articles=["1", "부칙"])
    assert "## 부칙 목록" in out and "제외했습니다" not in out
    mixed = read(statute_id=468, articles=["1", "별표1"])
    assert "별표1 는 제외했습니다. 별표·별지·서식은 조문으로 오지 않습니다" in mixed
    assert "별표는 그것만 따로 부르면(예: articles=['별표 5']) 제목과 파일 주소가 옵니다." in mixed
    # No hint says there is no annexed table: the API reads an annexed table's file address with web_fetch.
    assert read(statute_id=468, articles=["별표 9"]) == (
        "## status: bad_articles\n- message: 요청한 번호의 별표가 이 법령의 별표 목록에 없습니다 — statute_id 만 넣으면 "
        "별표 목록(제목과 파일 주소)이 함께 옵니다.\n- statute_id: 468\n- articles: ['별표 9']")


def test_article_version_queries_start_from_the_laws_versions():
    """The per-article version queries walk the law's versions, not the article number across every law. Built from
    the sample database's own table and index definitions but without its ANALYZE statistics, as the production
    corpus has none: the planner then started from the article-number index (article 1: 28,307 rows of 1.67M), and a
    search inside a statute ran that for every article of the law."""
    sample = sqlite3.connect(f"file:{os.environ['HARNESS_DB']}?mode=ro", uri=True)
    ddl = [r[0] for r in sample.execute(
        "SELECT sql FROM sqlite_master WHERE tbl_name IN ('st_statutes', 'st_articles') AND sql IS NOT NULL "
        "AND type IN ('table', 'index') ORDER BY type DESC")]
    sample.close()
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    for sql in ddl:
        conn.execute(sql)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(st_statutes)")]
    for sid, law_id, eff in ((1, "L", "20200101"), (2, "L", "20210101"), (3, "M", "20200101"), (4, "L", "29990101")):
        row = {c: None for c in cols} | {"id": sid, "law_id": law_id, "name": law_id, "effective_date": eff}
        conn.execute(f"INSERT INTO st_statutes ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})", list(row.values()))
        conn.execute("INSERT INTO st_articles (statute_id, article_no, article_no_num, article_branch, title, article_text,"
                     " article_eff_date) VALUES (?, '3', 3, 0, ?, ?, ?)", (sid, f"제목{sid}", f"제3조 글{sid}", eff))
    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    assert statutes._get_historic_article(conn, "L", 3, 0, "20250101")["article_text"] == "제3조 글2"
    assert statutes._pending_article(conn, "L", "20250101", 3, 0)["effective_date"] == "29990101"
    assert statutes._added_article(conn, "L", "20200601", 3, None) == "20210101"
    conn.set_trace_callback(None)
    plans = [[r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + q)] for q in seen if "st_articles" in q]
    assert len(plans) == 3
    for plan in plans:
        assert plan[0].startswith("SEARCH s ") and "idx_st_articles_no" not in " ".join(plan), plan
