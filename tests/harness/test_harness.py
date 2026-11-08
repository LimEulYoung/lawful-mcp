"""The database tools and the Agent API pieces on the sample database — no network, no model calls."""

import asyncio
import importlib
import inspect
import re
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from lawful_mcp.harness_local import agent_api
from lawful_mcp.harness_local.agents import TOOLS
from lawful_mcp.harness_local.deps import HarnessDeps

TOOL = {t.__name__: t for t in TOOLS}


class Ctx:
    deps = HarnessDeps(dive_subagent=None)


def call(name, **args):
    out = TOOL[name](Ctx(), **args)
    return asyncio.run(out) if inspect.isawaitable(out) else out


def test_tools_are_the_four_database_tools():
    assert sorted(TOOL) == [
        "precedent_dive",
        "precedent_search",
        "sentencing_analysis",
        "statute_lookup",
    ]


@pytest.mark.parametrize(
    "name, args",
    [
        ("statute_lookup", {"query": "형법", "articles": ["347"]}),
        ("precedent_search", {"query": "불법행위 손해배상 책임"}),
        ("precedent_search", {"query": "사기죄 기망행위", "court_level": "대법원"}),
        ("sentencing_analysis", {"charges": ["사기"]}),
        ("sentencing_analysis", {"charges": ["사기", "폭행"]}),
    ],
)
def test_database_tools_answer(name, args):
    assert call(name, **args).startswith("## status: ok")


def test_search_lists_judgments():
    assert "- id: " in call("precedent_search", query="불법행위 손해배상 책임")


def test_unknown_judgment_is_missing_without_a_model_call():
    assert call("precedent_dive", case_id=999999999, question="법리").startswith("## status: missing")


def test_dive_failure_is_reported_without_sharing_the_main_usage():
    """A failed summary request leaves the judgment unread instead of failing the item; the sub-agent has its own usage."""
    seen = []

    class Failing:
        async def run(self, prompt, **kwargs):
            seen.append(kwargs)
            raise RuntimeError("HTTP 429")

    class FailingCtx:
        deps = HarnessDeps(dive_subagent=Failing())

    out = asyncio.run(TOOL["precedent_dive"](FailingCtx(), case_id=205021, question="법리"))
    assert out.startswith("## status: error") and "읽지 못한 것으로" in out and seen == [{}]


def test_today_hint():
    now = datetime(2026, 10, 31, 14, tzinfo=ZoneInfo("Asia/Seoul"))
    assert agent_api.today_hint(now) == "\n\n[오늘: 2026년 10월 31일 (토) 오후]"


def test_no_text_points_to_web_tools():
    """The harness has no web tools, so no tool text may send the model to one."""
    from lawful_mcp.harness_local.tools import statutes

    texts = [statutes._UNSUPPORTED_UNIT_HINT] + [t.__doc__ or "" for t in TOOLS]
    assert not [t for t in texts if "web_search" in t or "web_fetch" in t]


def test_same_probation_text_keeps_the_quadrant_of_its_table():
    from lawful_mcp.harness_local.deps import open_db
    from lawful_mcp.harness_local.tools.sentencing_guideline import read_catalog

    with open_db() as conn:
        catalog = read_catalog(conn, 39)
    variants = {f.group: f for f in catalog["probation"] if f.text == "처벌불원 또는 실질적 피해 회복(공탁 포함)"}
    for group, section in ((("일반", "긍정"), "표 1"), (("주요", "긍정"), "표 3")):
        assert variants[group].note.startswith(section)


def test_sentencing_takes_only_charges_and_a_date():
    from pydantic_ai import Tool

    tool = Tool(TOOL["sentencing_analysis"])
    schema = tool.tool_def.parameters_json_schema
    assert set(schema["properties"]) == {"charges", "offense_date"}   # the comparables' year is the offense's
    assert schema["properties"]["charges"] == {**schema["properties"]["charges"], "type": "array",
                                               "items": {"type": "string"}}
    validate = tool.function_schema.validator.validate_python
    for sent in ("사기", ["사기"], '["사기"]'):
        assert validate({"charges": sent})["charges"] == ["사기"]
    assert validate({"charges": '["사기", "폭행"]'})["charges"] == ["사기", "폭행"]
    assert call("sentencing_analysis", charges=["사기", "사기", " "]) == call("sentencing_analysis", charges=["사기"])
    assert call("sentencing_analysis", charges=[]).startswith("## status: missing_input")


def test_several_charges_are_one_response():
    out = call("sentencing_analysis", charges=["도로교통법위반(음주운전)", "도로교통법위반(무면허운전)"])
    assert out.startswith("## status: ok") and out.count("## status:") == 1
    assert out.count("## 양형 판단 규칙") == 1 and "6. 죄가 여럿이면" in out and "상상적 경합" in out
    assert "## 죄: 도로교통법위반(음주운전)" in out and "## 죄: 도로교통법위반(무면허운전)" in out
    assert "## 양형기준 — 교통범죄 — 위 도로교통법위반(음주운전) 항목과 같다" in out  # one guideline, shown once
    assert out.count("## 권고 형량범위 — 범죄유형별") == 1
    assert "## 묶음별 (1심)" in out and "- 도로교통법위반(음주운전)만:" in out
    for gone in ("- id=", "statute_choice", "branch_key", "guideline_type", "다시 불러"):
        assert gone not in out, gone


def test_an_unknown_charge_does_not_hide_a_known_one():
    out = call("sentencing_analysis", charges=["사기", "없는죄명"])
    assert out.startswith("## status: ok") and "## 죄: 없는죄명" in out and "공식 죄명(대검 죄명표)에 없는 표기" in out
    assert "## 죄: 사기" in out and "## 권고 형량범위" in out


def test_each_guideline_of_a_charge_is_shown():
    # 상표법위반: §232 ① (perjury) and §230 (IP infringement) fall under different guidelines.
    out = call("sentencing_analysis", charges=["상표법위반"])
    heads = re.findall(r"(?m)^## 양형기준 — (.+)$", out)
    assert len(heads) == 2 and any("위증" in h for h in heads) and any("지식재산" in h for h in heads)
    assert out.count("## 권고 형량범위 — 범죄유형별") == 2


def test_candidates_the_table_cannot_settle_are_named_with_the_reason():
    out = call("sentencing_analysis", charges=["상습절도"])
    assert re.search(r"(?m)^  · 원범죄 형법 §331의2.*: 법정형을 정하지 못했다 — 매핑표에 원범죄\(형법 §331의2\)의 법정형이 없다$", out)
    alone = call("sentencing_analysis", charges=["아동·청소년의성보호에관한법률위반(신고의무자성범죄)"])
    assert "자기 법정형이 없는 가중 규정이다" in alone and "기본 범죄를 죄명 목록에 함께 넣고" in alone
    for gone in ("branch_key", "reference_choice", "재호출"):
        assert gone not in out + alone, gone


def test_a_candidate_without_a_range_is_unsettled_not_a_row_without_imprisonment():
    # 성폭력처벌법 §7 ④: 「형법」 제299조의 죄는 제1항부터 제3항까지의 예에 따라 — the table holds no range of its own.
    out = call("sentencing_analysis", charges=["성폭력범죄의처벌등에관한특례법위반(13세미만미성년자강간)"])
    assert "자유형 — · 벌금 —" not in out
    assert re.search(r"(?m)^  · .*제299조.*: 법정형을 정하지 못했다 — 매핑표에 형의 범위가 없다", out)
    assert ": 자유형 120월 이상 · 무기" in out   # a candidate with a range stays a row; no stated maximum, none shown


def test_two_spellings_of_one_charge_are_read_once():
    out = call("sentencing_analysis", charges=["사기", "사기죄"])
    assert out.count("## 죄:") == 1 and "- 같은 죄로 읽은 이름: 사기죄 → 사기" in out
    both = call("sentencing_analysis", charges=["사기", "사기미수"])   # the suffix makes it another charge
    assert "## 죄: 사기\n" in both and "## 죄: 사기미수" in both and "같은 죄로 읽은" not in both


def test_a_habitual_offence_raises_both_bounds_and_says_it_is_applied():
    # 대법원 2016도18194: 상습특수상해 = 특수상해(1년 이상 10년 이하) × 1.5 at both bounds.
    out = call("sentencing_analysis", charges=["상습특수상해"])
    assert "  · 원범죄 형법 §258의2 ① — 특수상해, 특수존속상해: (원범죄의 형 ×1.5 반영) 자유형 18~180월" in out
    assert "이미 반영된 것이니 다시 가중하지 않는다" in out


def test_a_source_article_is_split_where_its_paragraphs_differ():
    # 형법 §257: ① 상해 7년 이하, ② 존속상해 10년 이하 — the first row alone gave 상습존속상해 the 상해 range.
    out = call("sentencing_analysis", charges=["상습존속상해"])
    assert "  · 원범죄 형법 §257 ① — 상해: (원범죄의 형 ×1.5 반영) 자유형 126월 이하 · 벌금 15,000,000원 이하" in out
    assert "  · 원범죄 형법 §257 ② — 존속상해: (원범죄의 형 ×1.5 반영) 자유형 180월 이하" in out
    # 형법 §301의2: 살해 and 치사 share the article and no paragraph tells them apart — named, not guessed.
    minor = call("sentencing_analysis", charges=["미성년자의제강간"])
    assert re.search(r"(?m)^  · 원범죄 형법 §301의2.*: 법정형을 정하지 못했다 — 원범죄의 죄마다 법정형이 다르다\(강간등살인, "
                     r"강간치사, 강제추행치사\)", minor)


def test_a_suffix_is_named_not_applied():
    out = call("sentencing_analysis", charges=["사기방조"])
    assert "죄명의 '방조': 아래 법정형은 정범의 형이다 — 방조는 형법 §32 ② 필요적 감경" in out
    assert "자동 반영" not in out


def test_the_rules_keep_what_answers_use_and_state_it_once():
    out = call("sentencing_analysis", charges=["사기"])
    rules = out[out.index("## 양형 판단 규칙"):out.index("## 죄: 사기")]
    for text in ("형종 선택", "벌금을 고르면 양형기준을 적용하지", "이미 반영된 것이니 다시 가중하지 않는다", "장기 ×2",
                 "사후적 경합범이다", "×1.5까지", "합한 것을 넘지 못한다", "단기 중 가장 높은 것", "사형·무기면 그 형으로",
                 "병과한다", "작량감경", "개수로 상쇄", "1/2\n     넓힌다", "처단형의 상한·하한을 따른다", "§62 ①",
                 "주요사유를 일반사유보다 중하게", "상상적 경합, §40", "§37 후단", "1/2과 1/3", "금액을 합산해 유형을",
                 "상한이 없으면(「N월 이상」) 30년"):
        assert text in rules, text
    for gone in ("특수교사", "84도2890", "2016도18194", "순서를 바꿔도"):
        assert gone not in rules, gone
    assert len(rules) < 2_000


def test_catalog_keeps_table_labels_without_type_claims():
    from lawful_mcp.harness_local.deps import open_db
    from lawful_mcp.harness_local.tools.sentencing_guideline import sections

    with open_db() as conn:
        lines = "\n".join(b for _h, b in sections(conn, 42)).split("\n")
    headings = [line for line in lines if line.startswith("[")]
    assert any("가. 자본시장의 공정성" in h for h in headings)
    assert not any("적용:" in h for h in headings)


def test_a_guideline_is_shown_whole_and_once():
    from lawful_mcp.harness_local.deps import open_db
    from lawful_mcp.harness_local.tools.sentencing_guideline import sections

    sa = importlib.import_module("lawful_mcp.harness_local.tools.sentencing_analysis")
    with open_db() as conn:
        cat = conn.execute("SELECT category_id FROM sg_probation_factors WHERE pole='일반' LIMIT 1").fetchone()[0]
        whole = dict(sections(conn, cat))
        pointer = sa._guideline_sections(conn, [(cat, "## 양형기준 — X", "첫째")])
    assert "[일반/" in whole[next(h for h in whole if h.startswith("## 집행유예 참작사유"))]
    assert whole[next(h for h in whole if h.startswith("## 일반양형인자"))].startswith("[")
    assert not any("생략" in b for b in whole.values())
    assert pointer == [("## 양형기준 — X — 위 첫째 항목과 같다", "")]   # a guideline shown once stays a pointer


def test_section_letters_match_what_the_tool_counts():
    from lawful_mcp.harness_local.deps import open_db
    from lawful_mcp.harness_local.tools.sentencing_guideline import read_catalog, section_tags

    with open_db() as conn:
        for (cat,) in conn.execute("SELECT DISTINCT category_id FROM sg_subtypes").fetchall():
            leaves = {r[0] for r in conn.execute(
                "SELECT id FROM sg_subtypes WHERE category_id=? AND type_criterion IS NOT NULL", (cat,))}
            tags = section_tags(conn, cat)
            special = [f for f in read_catalog(conn, cat)["guideline"] if f.special]
            for f in special:
                if f.note:
                    assert set(f.note.split("·")) == {tags[t] for t in f.types}, (cat, f.text)
                    assert f.types == {t for t in leaves if tags.get(t) in f.note.split("·")}, (cat, f.text)
                else:
                    assert f.types == leaves, (cat, f.text)
            for leaf in leaves:                      # every type keeps factors to choose from
                assert any(leaf in f.types for f in special), (cat, leaf)


def test_a_court_name_no_court_has_is_dropped_with_a_note():
    out = call("precedent_search", query="사기 편취", court_name="경기고등")
    assert "court_name '경기고등' 에 맞는 법원이 코퍼스에 없어 무시함" in out and "## matches: (없음)" not in out
    assert "무시함" not in call("precedent_search", query="사기 편취", court_name="서울")


def test_a_court_name_takes_the_courts_short_and_spaced_spellings():
    ps = importlib.import_module("lawful_mcp.harness_local.tools.precedent_search")
    for name in ("의정부지방법원", "의정부지법"):   # sample.db: 의정부지방법원 2025나204080
        out = call("precedent_search", case_number="2025나204080", court_name=name)
        assert "id: 234954" in out and "조건에 맞지 않습니다" not in out, name
    conn = sqlite3.connect(":memory:")
    conn.executescript("""CREATE TABLE prec_meta(court_name TEXT);
      INSERT INTO prec_meta VALUES('서울고등법원'),('수원지방법원'),('수원지방법원 성남지원'),('대법원');""")
    match = ps._COURT_MATCH.format(col="court_name")

    def names(name):
        key = ps._check_court_name(conn, name)[0]
        return {r[0] for r in conn.execute(f"SELECT court_name FROM prec_meta WHERE {match}", (key,))} if key else None

    assert names("서울고등법원") == names("서울고법") == names("서울 고등") == names("서울고등법원(춘천)") == {"서울고등법원"}
    assert names("수원지방법원 성남지원") == names("수원지법 성남지원") == names("성남지원") == {"수원지방법원 성남지원"}
    assert names("수원") == {"수원지방법원", "수원지방법원 성남지원"}
    assert names("대법원") == {"대법원"} and names("경기고등") is None


def test_a_statute_search_of_only_short_words_says_the_text_was_not_searched():
    # The article index is trigram: two-character words match no text, so the list comes from names alone.
    assert "'절도': 두 글자 이하 낱말은 조문 본문에서 찾지 못합니다" in call("statute_lookup", query="절도")
    assert "'간통': 두 글자 이하" in call("statute_lookup", query="형법 간통")
    for query in ("민법", "상습범", "국세기본법 가산세"):  # a statute name, or a word the text index can match
        assert "두 글자 이하" not in call("statute_lookup", query=query), query


def test_a_statute_search_whose_words_no_article_holds_says_the_list_is_by_name():
    by_name = "조문 본문에서 맞는 조문을 찾지 못해, 법령 이름으로만 찾았습니다."
    out = call("statute_lookup", query="민법 없는낱말이다")
    assert by_name in out and "- 584 민법" in out
    assert by_name in call("statute_lookup", query="없는낱말이다 없는말이에요")  # an empty list says it too
    assert "- message:" not in call("statute_lookup", query="상습범")  # found in the text


def test_source_crimes_are_listed_in_the_penalty_table():
    # The briefing lists every source crime in its table, without the first candidate's choice line.
    brief = call("sentencing_analysis", charges=["폭행치사"])
    assert "  · 원범죄 형법 §259" in brief and "## 선택 원범죄" not in brief and "statute_choice" not in brief


def test_section_letters_in_the_briefing():
    briefing = call("sentencing_analysis", charges=["상해"])
    # sample.db keeps the guideline's numbering (「01 일반적인 상해」) until it takes the display patch.
    assert re.search(r'- \[A\] "(?:\d+\.? )?일반적인 상해 1유형"', briefing)
    assert re.search(r'- \[B\] "(?:\d+\.? )?특수상해·누범상해 1유형"', briefing)
    assert "- [B] 비난할만한 범행 동기" in briefing and "- id=" not in briefing


def test_offense_dates_in_korean_forms_and_unreadable_dates():
    from lawful_mcp.harness_local.tools._coerce import to_iso_date

    assert {to_iso_date(d) for d in ("2019. 3. 8.", "2019.3.8.", "2019년 3월 8일", "2019-03-08")} == {"20190308"}
    for name, args in (
        ("statute_lookup", {"statute_id": 468, "articles": ["1"]}),
        ("sentencing_analysis", {"charges": ["사기"]}),
    ):
        assert call(name, offense_date="작년 봄", **args).startswith("## status: bad_offense_date")


def _versions(*rows):
    """(id, effective_date, change_kind, serial_id) → version rows; serial_id is set on a version loaded whole."""
    keys = ("id", "effective_date", "change_kind", "serial_id")
    return [dict(zip(keys, r, strict=True)) for r in rows]


def test_current_and_point_in_time_versions_are_the_latest_whole_text():
    from lawful_mcp.harness_local.tools.statutes import _pick_current_version

    # The latest whole text in force by the date, whatever its label; a row holding only the articles its amendment
    # changed is no candidate, nor a version not yet in force.
    rows = _versions(
        (1, "20250101", "일부개정", "11"),
        (2, "20251001", "일부개정", "12"),
        (3, "20251101", "일부개정", ""),
        (4, "20270101", "일부개정", "14"),
    )
    assert _pick_current_version(rows, "20251201")["id"] == 2
    # Enactments and full amendments are whole texts; a version loaded whole counts however small; a
    # repeal-and-enactment is a new whole text.
    rows = _versions(
        (1, "19620101", "제정", ""),
        (2, "19920901", "전부개정", ""),
        (3, "19990401", "폐지제정", ""),
        (4, "20050101", "일부개정", "123"),
        (5, "20100101", "일부개정", ""),
        (6, "20250101", "일부개정", "124"),
    )
    assert _pick_current_version(rows, "20000101")["id"] == 3
    assert _pick_current_version(rows, "20200101")["id"] == 4
    # A repeal is passed over for the version in force just before it.
    rows = _versions((1, "19800101", "제정", ""), (2, "19900101", "폐지", "21"))
    assert _pick_current_version(rows, "20200101")["id"] == 1


def test_a_repeal_ends_a_law_until_it_is_enacted_again():
    from lawful_mcp.harness_local.tools.statutes import _is_repealed_at

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE st_statutes (law_id TEXT, effective_date TEXT, change_kind TEXT);
        CREATE TABLE st_lapsed (law_id TEXT, lapsed_on TEXT);
        INSERT INTO st_statutes VALUES ('A', '19610417', '제정'), ('A', '19631217', '폐지'), ('A', '19980101', '타법개정'),
            ('B', '19500412', '폐지제정'), ('B', '19580125', '타법폐지'), ('B', '19600623', '제정'),
            ('B', '19610428', '일부개정'), ('C', '19120401', '제정');
        INSERT INTO st_lapsed VALUES ('C', '');
    """)
    # An amendment listed after the repeal does not bring the law back; a new enactment does.
    assert _is_repealed_at(conn, "A", None)["effective_date"] == "19631217"
    assert _is_repealed_at(conn, "A", "19700101")["change_kind"] == "폐지"
    assert _is_repealed_at(conn, "A", "19620101") is None
    assert _is_repealed_at(conn, "B", "19590101")["change_kind"] == "타법폐지"
    assert _is_repealed_at(conn, "B", "19620101") is None
    # A law that lost effect on a day the corpus does not know is gone today, with no date; earlier it is not known.
    assert _is_repealed_at(conn, "C", None) == {"effective_date": "", "change_kind": "효력 상실"}
    assert _is_repealed_at(conn, "C", "19300101") is None


def test_search_shows_the_version_in_force_ranked_by_its_law_and_keeps_a_named_repealed_law():
    from lawful_mcp.harness_local.tools.statutes import _drop_repealed, _fold_versions, _rank_name_rows

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE st_articles (statute_id INTEGER);
        CREATE TABLE st_lapsed (law_id TEXT, lapsed_on TEXT);
        CREATE TABLE statute_aliases (alias TEXT, statute_name TEXT, how TEXT);
        CREATE TABLE st_statutes (law_id TEXT, effective_date TEXT, change_kind TEXT);
        INSERT INTO st_statutes VALUES ('H', '19780131', '제정'), ('H', '20030701', '타법개정'),
            ('H', '20031130', '전부개정'), ('S', '19620115', '제정'), ('S', '20090204', '폐지'),
            ('E', '19620115', '제정'), ('E', '20090204', '폐지'), ('M', '19610101', '제정'),
            ('M', '19870701', '폐지'), ('L', '20090204', '제정');
    """)
    keys = ("id", "law_id", "name", "effective_date", "change_kind", "name_in_query", "fts_hits")

    def rows(*rs):
        return [dict(zip(keys, r, strict=True), short_name=None, name_cover=1, serial_id="1") for r in rs]

    # Asked by its old name, a renamed law shows the version in force, ranked by the name the query named.
    renamed = rows((1, "H", "주택건설촉진법", "19780131", "제정", 7, 0), (2, "H", "주택건설촉진법", "20030701", "타법개정", 7, 0),
                   (3, "H", "주택법", "20031130", "전부개정", 0, 2))
    (now,) = _fold_versions(conn, renamed, None)
    assert now["id"] == 3 and now["name_in_query"] == 7 and now["matched_name"] == "주택건설촉진법"
    (then,) = _fold_versions(conn, renamed, "19900101")
    assert then["id"] == 1 and then["fts_hits"] == 2
    other = rows((20, "T", "지하철도건설촉진법", "19790417", "제정", 0, 0))[0]
    assert _rank_name_rows(conn, [other, now], "주택건설촉진법", 10)[0]["id"] == 3
    # A dated or current search keeps the repealed law holding the longest name the query names and drops the ones
    # it only brushes or whose name a law in force holds.
    found = rows((10, "S", "증권거래법", "20090204", "폐지", 5, 0), (11, "E", "증권거래법 시행령", "20090204", "폐지", 0, 0),
                 (12, "L", "자본시장법", "20090204", "제정", 0, 0), (13, "M", "시장법", "19870701", "폐지", 3, 0))
    assert [r["id"] for r in _drop_repealed(conn, found, "20150301", 5)] == [10, 12]
    assert [r["id"] for r in _drop_repealed(conn, found, None, 5)] == [10, 12]
    namesake = found + rows((14, "H", "증권거래법", "20031130", "전부개정", 5, 0))
    assert [r["id"] for r in _drop_repealed(conn, namesake, "20150301", 5)] == [12, 14]


def test_a_dated_lookup_tells_numbers_added_later_from_numbers_that_never_existed():
    from lawful_mcp.harness_local.tools.statutes import _added_article, _format_response_md

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE st_statutes (id INTEGER, law_id TEXT, effective_date TEXT);
        CREATE TABLE st_articles (statute_id INTEGER, article_no_num INTEGER, article_branch INTEGER, article_text TEXT,
                                  article_eff_date TEXT);
        INSERT INTO st_statutes VALUES (1, 'V', '20000101'), (2, 'V', '20130701'), (3, 'V', '29990101');
        INSERT INTO st_articles VALUES (1, 17, 0, '제17조의2(대손세액공제) …', '20000101'),
            (2, NULL, NULL, '        제2절 공제', '20130701'), (2, 39, 0, '제39조(공제하지 아니하는 매입세액) …', '20130701'),
            (3, 80, 0, '제80조(시행 전) …', '29990101');
    """)
    assert _added_article(conn, "V", "20121116", 39, None) == "20130701"
    assert _added_article(conn, "V", "20121116", 80, None) is None  # not in force yet: the pending rule's case
    resp = {"status": "ok", "mode": "detail", "statute": {"id": 2, "name": "부가가치세법", "kind": "법률"},
            "offense_date": "20121116", "articles": [], "missing": ["39", "41"], "added": {"39": "20130701"}}
    out = _format_response_md(resp)
    assert "- 39(2013-07-01판부터): 2012-11-16 시점에는 아직 없던 번호입니다" in out
    assert "- 41: 현행·연혁본 어디에도 없는 번호입니다." in out


def test_an_article_in_force_since_the_date_is_added_even_when_a_pending_version_amends_it():
    # 행정규제기본법 §8-2 came in 2018 and a pending version amends it; it was reported pending (2026-11-04 pilot).
    import os

    from lawful_mcp.harness_local.tools.statutes import _detail_statute_at_date, _format_response_md

    src = sqlite3.connect(f"file:{os.environ['HARNESS_DB']}?mode=ro", uri=True)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    for (sql,) in src.execute("SELECT sql FROM sqlite_master WHERE name IN ('st_statutes', 'st_articles', 'st_lapsed')"):
        conn.execute(sql)
    # Versions 1 and 2 hold the whole text (a serial number); 3 and 4 hold only the articles they change.
    for sid, eff, serial in ((1, "20100101", "1"), (2, "20180101", "2"), (3, "20990101", ""), (4, "99990101", "")):
        conn.execute("INSERT INTO st_statutes (id, law_id, name, kind, effective_date, serial_id) "
                     "VALUES (?, 'L', '시험법', '법률', ?, ?)", (sid, eff, serial))
    for sid, num, branch, eff in ((1, 1, 0, "20100101"), (2, 1, 0, "20100101"), (2, 8, 2, "20180101"),
                                  (3, 8, 2, "20990101"), (3, 9, 0, "20990101"), (4, 10, 0, "99990101")):
        conn.execute("INSERT INTO st_articles (statute_id, article_no_num, article_branch, title, article_text, "
                     "article_eff_date) VALUES (?, ?, ?, '제목', '본문', ?)", (sid, num, branch, eff))
    meta = {"id": 2, "law_id": "L", "name": "시험법", "kind": "법률"}
    resp = _detail_statute_at_date(conn, meta, [(8, 2), (9, None), (10, None)], "20150101", asked_iso="20150101")
    assert resp["added"] == {"8-2": "20180101"} and set(resp["pending"]) == {"9", "10"}
    out = _format_response_md({"status": "ok", **resp})
    assert "- 8-2(2018-01-01판부터): 2015-01-01 시점에는 아직 없던 번호" in out
    assert "- 9(2099-01-01 시행 예정): 2015-01-01 시점에는 시행 전인 조문" in out
    assert "- 10: 시행일이 정해지지 않은 개정본에만 있는 번호입니다" in out and "9999" not in out


def test_an_unreadable_case_number_is_named():
    out = call("precedent_search", case_number="2016두")
    assert "case_number '2016두' 에서 사건번호를 읽지 못했습니다" in out and "인자에 넣으세요" not in out
    assert "case_number '2016두' 에서 사건번호를 읽지 못했습니다(예: 2010다89012) — 빼고 query 로만 찾았습니다" in call(
        "precedent_search", query="사기 편취", case_number="2016두")


def test_point_in_time_outline_lists_unamended_articles():
    def numbers(out):
        return {n for n in re.findall(r"^- (\d+):", out, re.M) if n != "0"}

    then = call("statute_lookup", statute_id=468, offense_date="2000.1.1")
    assert "시점 조회: 2000-01-01" in then and numbers(then) == numbers(call("statute_lookup", statute_id=468))


