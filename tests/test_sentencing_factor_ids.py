"""Sentencing factors are picked by id; `sentencing_factors` owns the list, the checks and the classification.

What this file holds in place:

  1. The caller sends **only ids** from the response lists; the classification
     (aggravating, mitigating, major, general) is read from the guideline
     table. When factors came as sentence dicts, an unknown key or shape was
     dropped silently and 「기본영역」 came back with nothing counted.
  2. An id that cannot be counted returns without a calculation
     (`bad_factor_id`) — better than a wrong number. Special factors count
     only within the chosen type's **section**.
  3. One candidate unable to use the ids does not cancel another candidate's
     calculation (drink-driving ① and ③; 문화재보호법위반 arson and damage).
  4. The retired dict arguments are accepted by the MCP server without
     appearing in its schema, and answered with how to send ids.

Runs on the bundled sample, which carries the guideline tables in full.
"""
from __future__ import annotations

import importlib
import json
import re
import sqlite3
from types import SimpleNamespace

import pytest

from lawful_mcp import tools
from lawful_mcp.deps import build_deps, open_db
from lawful_mcp.tools._coerce import to_iso_date
from lawful_mcp.tools.sentencing_analysis import _trim
from lawful_mcp.tools.sentencing_factors import format_catalog, read_catalog, resolve_ids, section_tags


def call(**args):
    return tools.sentencing_analysis(SimpleNamespace(deps=build_deps(None), usage=None), **args)


def _catalog_of(charge: str):
    conn = open_db()
    try:
        cat = conn.execute("SELECT sg_category_id FROM charge_legal_map WHERE charge_key=?", (charge,)).fetchone()[0]
        return read_catalog(conn, cat)
    finally:
        conn.close()


def _catalog(category_id: int):
    conn = open_db()
    try:
        return read_catalog(conn, category_id)
    finally:
        conn.close()


def _special(charge: str, text: str):
    return next(f for f in _catalog_of(charge)["guideline_factor_ids"] if f.key and f.text == text)


@pytest.fixture
def obstruction_factors():
    return _catalog_of("공무집행방해")


def test_ids_use_corpus_groups_and_count_duplicates_once(obstruction_factors):
    mild = "폭행·협박·위계 또는 공무방해의 정도가 경미한 경우"
    gf = next(f.id for f in obstruction_factors["guideline_factor_ids"] if f.text == mild)
    pf = [f.id for f in obstruction_factors["probation_factor_ids"]
          if f.text in (mild, "형사처벌 전력 없음", "진지한 반성")]
    args = {"charge": "공무집행방해", "guideline_type": "공무집행방해 1유형", "sentence_months": 8}
    out = call(**args, guideline_factor_ids=[gf], probation_factor_ids=pf)
    repeated = call(**args, guideline_factor_ids=[gf, gf], probation_factor_ids=pf * 2)
    assert out == repeated and out.startswith("## status: ok")
    assert "- 영역: 감경" in out and "- 자유형: 1~8월" in out
    assert "- major: positive=2 / negative=0" in out
    assert "- general: positive=1 / negative=0" in out
    assert "형사처벌 전력 없음 → [major / positive]" in out
    briefing = call(charge="공무집행방해")
    assert re.search(rf"id={gf}( \[[A-Z·]+\])?: {re.escape(mild)}", briefing)
    for i in pf:
        assert f"id={i}:" in briefing or f"id={i} [" in briefing


@pytest.mark.parametrize("field", ["guideline_factor_ids", "probation_factor_ids"])
def test_invalid_ids_never_produce_a_calculation(field):
    out = call(charge="공무집행방해", guideline_type="공무집행방해 1유형", sentence_months=8,
               **{field: [999999999]})
    assert out.startswith("## status: bad_factor_id")
    assert field in out and "선택 가능한 항목:" in out
    assert "## stage: calc" not in out and "## 권고형" not in out


def test_ids_are_checked_against_the_category_and_say_what_to_send(obstruction_factors):
    args = {"charge": "공무집행방해", "guideline_type": "공무집행방해 1유형"}
    general = next(f.id for f in obstruction_factors["guideline_factor_ids"] if f.key is None)
    out = call(**args, guideline_factor_ids=[general])
    assert out.startswith("## status: bad_factor_id") and "일반양형인자" in out
    probation = obstruction_factors["probation_factor_ids"][0].id
    out = call(**args, guideline_factor_ids=[probation])
    assert out.startswith("## status: bad_factor_id") and "`probation_factor_ids` 목록의 id" in out
    special = next(f.id for f in obstruction_factors["guideline_factor_ids"] if f.key)
    out = call(charge="사기", guideline_type="일반사기 1유형", guideline_factor_ids=[special])
    assert out.startswith("## status: bad_factor_id") and "목록에 없는 id" in out


@pytest.mark.parametrize(("charge", "guideline_type", "text"), [
    ("공무집행방해", "공무집행방해 2유형", "경합범 아닌 반복적 범행"),
    ("사기", "일반사기 2유형", "범죄수익을 의도적으로 은닉한 경우"),
])
def test_types_after_the_first_count_their_section_factors(charge, guideline_type, text):
    # The corpus hangs a section's factor table on its first type only; the later types use it too.
    factor = next(f for f in _catalog_of(charge)["guideline_factor_ids"] if f.text == text)
    out = call(charge=charge, guideline_type=guideline_type, guideline_factor_ids=[factor.id], sentence_months=12)
    assert out.startswith("## status: ok") and "- 영역: 가중" in out
    assert f"id={factor.id}: {text} → [특별 / 행위 / 가중]" in out


def test_ids_skip_candidates_outside_the_guideline():
    # Drink-driving: §148의2 ① (repeat offence) is outside the guideline, ③ inside —
    # one candidate that cannot use the ids does not refuse the others.
    factor = next(f for f in _catalog(7)["probation_factor_ids"]
                  if f.text == "운전으로 인한 도로교통상의 위험이 매우 높은 경우")
    out = call(charge="도로교통법위반(음주운전)", guideline_type="음주·무면허운전 3유형",
               probation_factor_ids=[factor.id], sentence_months=10)
    assert "bad_factor_id" not in out
    assert "인자 id는 이 조항에 적용하지 않았다 — 양형기준 적용 대상이 아니다" in out
    assert f"probation_factor_ids id={factor.id}:" in out and "- major: positive=0 / negative=1" in out


def test_empty_selection_and_missing_type(obstruction_factors):
    args = {"charge": "공무집행방해", "guideline_type": "공무집행방해 1유형", "sentence_months": 8}
    baseline = call(**args)
    for empty in (None, []):
        assert call(**args, guideline_factor_ids=empty, probation_factor_ids=empty) == baseline
    factor_id = next(f.id for f in obstruction_factors["guideline_factor_ids"] if f.key)
    out = call(charge="공무집행방해", guideline_factor_ids=[factor_id])
    assert out.startswith("## status: bad_factor_id") and "guideline_type" in out


def test_ids_of_one_category_keep_the_candidates_that_count_them():
    # 문화재보호법위반 spans arson (§94) and damage (§92·93); both categories have 「일반적 기준 1유형」.
    briefing = call(charge="문화재보호법위반")
    factor_id = int(re.search(r"^- id=(\d+):", briefing.split("## 특별양형인자")[1], re.M)[1])
    out = call(charge="문화재보호법위반", guideline_type="일반적 기준 1유형",
               guideline_factor_ids=[factor_id], sentence_months=12)
    assert not out.startswith("## status: bad_factor_id") and "## status: ok" in out
    assert f"guideline_factor_ids id={factor_id}:" in out
    assert "이 조항은 계산하지 않았다 — 보낸 인자 id가 이 조항의 양형기준 목록에 없다" in out
    typo = call(charge="문화재보호법위반", guideline_type="일반적 기준 1유형", guideline_factor_ids=[999999999])
    assert typo.startswith("## status: bad_factor_id")


def test_same_probation_text_keeps_the_quadrant_of_its_table():
    catalog = _catalog(39)
    variants = {f.key: f for f in catalog["probation_factor_ids"]
                if f.text == "처벌불원 또는 실질적 피해 회복(공탁 포함)"}
    for key, table in (("general_positive", "표 1"), ("major_positive", "표 3")):
        assert variants[key].note.startswith(table)
        _, classified, _ = resolve_ids(catalog, None, [variants[key].id])
        assert list(classified) == [key]


def test_schema_only_accepts_integer_id_arrays():
    from pydantic import ValidationError
    from pydantic_ai import Tool

    tool = Tool(tools.sentencing_analysis)
    properties = tool.tool_def.parameters_json_schema["properties"]
    assert "guideline_factors" not in properties and "probation_factors" not in properties
    for field in ("guideline_factor_ids", "probation_factor_ids"):
        assert {"type": "array", "items": {"type": "integer"}} in properties[field]["anyOf"]
        for bad in ([{"item": "형사처벌 전력 없음"}], {"major_positive": []}, "70", '{"item": 70}', "[",
                    ["7a"], ["1.5"], [True], [1.5]):
            with pytest.raises(ValidationError):
                tool.function_schema.validator.validate_python({"charge": "공무집행방해", field: bad})
        direct = tool.function_schema.validator.validate_python({"charge": "공무집행방해", field: [70]})
        for stringified in ("[70]", ["70"], '["70"]', [" 70 "]):
            assert tool.function_schema.validator.validate_python(
                {"charge": "공무집행방해", field: stringified}) == direct


@pytest.mark.parametrize("guideline", [[148], [148, 146]])
def test_serialized_id_calls_match_array_calls(guideline):
    from pydantic_ai import Tool

    tool = Tool(tools.sentencing_analysis)
    args = {"charge": "공무집행방해", "offense_date": "2020.7.24", "act_count": "2",
            "guideline_type": "공무집행방해 1유형", "guideline_factor_ids": guideline,
            "probation_factor_ids": [61, 64, 66]}
    encoded = {k: json.dumps(v) if k.endswith("factor_ids") else v for k, v in args.items()}
    decoded = tool.function_schema.validator.validate_python(encoded)
    assert decoded["guideline_factor_ids"] == guideline
    assert call(**decoded) == call(**args)
    # Direct calls (MCP) skip schema validation, so the tool unwraps the same shapes itself.
    for form in (json.dumps, lambda ids: [str(i) for i in ids]):
        direct = {k: form(v) if k.endswith("factor_ids") else v for k, v in args.items()}
        assert call(**direct) == call(**args)


def test_catalog_keeps_table_labels_without_type_claims():
    conn = open_db()
    try:
        lines = format_catalog(read_catalog(conn, 42))
    finally:
        conn.close()
    headings = [line for line in lines if line.startswith("[")]
    assert any("가. 자본시장의 공정성" in h for h in headings)
    assert not any("적용:" in h for h in headings)


def test_budget_folds_never_drop_selectable_ids():
    def sections():
        return [
            ("## 특별양형인자", "[특별/행위/감경]\n- id=1: 사유"),
            ("## 일반양형인자", "- 참고 사유"),
            ("## 집행유예 참작사유", "[major/positive]\n- id=2: 사유\n[general/positive]\n- id=3: 사유"),
        ]

    room_only = sections()
    assert _trim(room_only, 1, 10_000) == ["일반양형인자"]
    assert "id=1" in room_only[0][1] and "id=3" in room_only[2][1]
    over_budget = sections()
    assert _trim(over_budget, 1, 1) == ["일반양형인자", "집행유예 일반참작사유"]
    assert "id=1" in over_budget[0][1] and "id=2" in over_budget[2][1]
    assert "id=3" not in over_budget[2][1] and "[general/" not in over_budget[2][1]


def test_special_factors_count_only_in_their_section():
    # A 폭행범죄 type-1 case counted the 상해 section's 「공무집행방해의 경우」 and its band moved.
    other = _special("폭행", "공무집행방해의 경우")
    own = _special("폭행", "폭행의 정도가 경미한 경우(1, 6, 7유형)")
    args = {"charge": "폭행", "guideline_type": "폭행범죄 1유형", "sentence_months": 6}
    out = call(**args, guideline_factor_ids=[own.id, other.id])
    assert out.startswith("## status: bad_factor_id") and f"{other.id}(다른 절" in out
    assert f"{own.id}=" in out and f"{other.id}=" not in out          # only this section's factors are offered
    out = call(**args, guideline_factor_ids=[own.id])
    assert out.startswith("## status: ok") and "- 영역: 감경" in out
    # Similar wording in two sections: only the chosen type's section counts.
    special = _special("상해", "비난할만한 범행 동기")
    general = _special("상해", "비난할만한 범행 동기(4유형 제외)")
    args = {"charge": "상해", "guideline_type": "일반적인 상해 1유형", "sentence_months": 6}
    assert call(**args, guideline_factor_ids=[special.id]).startswith("## status: bad_factor_id")
    assert call(**args, guideline_factor_ids=[general.id]).startswith("## status: ok")


def test_section_letters_match_what_the_tool_counts():
    conn = open_db()
    try:
        for (cat,) in conn.execute("SELECT DISTINCT category_id FROM sg_subtypes").fetchall():
            leaves = {r[0] for r in conn.execute(
                "SELECT id FROM sg_subtypes WHERE category_id=? AND type_criterion IS NOT NULL", (cat,))}
            tags = section_tags(conn, cat)
            special = [f for f in read_catalog(conn, cat)["guideline_factor_ids"] if f.key]
            for f in special:
                if f.note:
                    assert set(f.note.split("·")) == {tags[t] for t in f.types}, (cat, f.id)
                    assert f.types == {t for t in leaves if tags.get(t) in f.note.split("·")}, (cat, f.id)
                else:
                    assert f.types == leaves, (cat, f.id)
            for leaf in leaves:                      # every type has special factors to pick
                assert any(leaf in f.types for f in special), (cat, leaf)
    finally:
        conn.close()


def test_section_letters_in_the_briefing_and_copied_names():
    briefing = call(charge="상해")
    assert '- [A] "일반적인 상해 1유형"' in briefing and '- [B] "특수상해·누범상해 1유형"' in briefing
    special = _special("상해", "비난할만한 범행 동기")
    assert f"- id={special.id} [B]: 비난할만한 범행 동기" in briefing
    out = call(charge="상해", guideline_type='[A] "일반적인 상해 1유형"', sentence_months=6)
    assert out.startswith("## status: ok") and "## 양형기준: 일반적인 상해 1유형" in out


def test_a_branch_reference_is_not_reported_as_ignored():
    # The briefing lists every base offence; the line for the first candidate's choice is left out.
    brief = call(charge="폭행치사")
    assert 'reference_choice="형법§259"' in brief and "## 선택 원범죄" not in brief and "미적용" not in brief
    chosen = call(charge="폭행치사", reference_choice="형법§259")
    assert "## 선택 원범죄: 형법 §259" in chosen and "미적용" not in chosen


def test_unreadable_offense_date_is_returned_not_answered_for_today():
    assert {to_iso_date(d) for d in ("2019. 3. 8.", "2019.3.8.", "2019년 3월 8일", "2019-03-08")} == {"20190308"}
    assert call(charge="사기", offense_date="작년 봄").startswith("## status: bad_offense_date")


def test_article_links_use_the_law_id_not_the_row_id():
    from lawful_mcp.config import case_url_base
    from lawful_mcp.tools.compute_sentencing_range import sentencing_candidates
    from lawful_mcp.tools.sentencing_analysis import _articles_section

    cands = sentencing_candidates("사기")
    sid = next(c.statute_id for c in cands if c.statute_id)
    conn = open_db()
    try:
        law_id = conn.execute("SELECT law_id FROM st_statutes WHERE id=?", (sid,)).fetchone()[0]
    finally:
        conn.close()
    ctx = SimpleNamespace(deps=build_deps(None), usage=None)
    links = "\n".join(_articles_section(ctx, cands, 0))     # no room — everything becomes a link
    assert f"{case_url_base()}/statutes/{law_id}/" in links
    if f"{sid:06d}" != law_id:
        assert f"/statutes/{sid:06d}/" not in links


def test_current_date_ignores_versions_not_yet_in_force():
    engine = importlib.import_module("lawful_mcp.tools.compute_sentencing_range")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE st_statutes (id INTEGER, law_id TEXT, effective_date TEXT, mst TEXT);
        CREATE TABLE st_articles (statute_id INTEGER, article_no_num INTEGER, article_branch INTEGER,
                                  article_text TEXT, article_changed TEXT);
        INSERT INTO st_statutes VALUES (1, 'L', '20200101', NULL), (2, 'L', '29991231', NULL);
        INSERT INTO st_articles VALUES (1, 1, NULL, '제1장 총칙', 'Y'), (1, 1, NULL, '제1조 현행', NULL),
                                       (2, 1, NULL, '제1조 시행 전', 'Y');
    """)
    row = {"statute_id": 1, "article_no_num": 1, "article_branch": None}
    got = engine._lookup_historic_article(conn, row, "20210101")
    assert got["current_effective_date"] == "20200101" and not got["is_outdated"] and got["text"] == "제1조 현행"


def test_fine_paragraphs_read_the_article_not_its_heading_whatever_the_branch_field():
    engine = importlib.import_module("lawful_mcp.tools.compute_sentencing_range")
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE st_articles (id INTEGER PRIMARY KEY, statute_id INTEGER, article_no_num INTEGER,
                                  article_branch INTEGER, article_text TEXT);
        INSERT INTO st_articles (statute_id, article_no_num, article_branch, article_text) VALUES
            (1, 136, NULL, '제7장 공무방해에 관한 죄'),
            (1, 136, NULL, '제136조(공무집행방해) ① … 5년 이하의 징역 또는 1천만원 이하의 벌금에 처한다.'),
            (2, 7, 0, '제2장 아동ㆍ청소년대상 성범죄의 처벌'),
            (2, 7, 0, '제7조(강간) ① … 무기징역 또는 5년 이상의 유기징역에 처한다. ③ … 1천만원 이상 3천만원 이하의 벌금에 처한다.'),
            (2, 7, 2, '제7조의2(예비, 음모) … 3년 이하의 징역에 처한다.');
    """)
    assert engine._extract_fine_paragraphs(conn, 1, 136, None) == ["① … 5년 이하의 징역 또는 1천만원 이하의 벌금에 처한다."]
    assert engine._extract_fine_paragraphs(conn, 2, 7, 0) == ["③ … 1천만원 이상 3천만원 이하의 벌금에 처한다."]
    assert engine._extract_fine_paragraphs(conn, 2, 7, 2) == []


def test_mcp_hides_the_retired_dict_args_and_answers_how_to_send_ids(monkeypatch):
    import anyio

    from lawful_mcp import server

    schema = next(t for t in anyio.run(server.mcp.list_tools) if t.name == "sentencing_analysis")
    props = schema.model_dump(by_alias=True, exclude_none=True)["inputSchema"]["properties"]
    assert "guideline_factors" not in props and "probation_factors" not in props
    assert {"type": "array", "items": {"type": "integer"}} in props["guideline_factor_ids"]["anyOf"]

    seen = {}

    def _stub(ctx, **kw):
        seen.update(kw)
        return "## status: ok\n"

    monkeypatch.setattr(server._t, "sentencing_analysis", _stub)

    def text(args):
        return anyio.run(lambda: server.mcp.call_tool("sentencing_analysis", args)).content[0].text

    for old in ({"guideline_factors": {"special_act_mitigators": ["진지한 반성"]}},
                {"probation_factors": [{"item": "형사처벌 전력 없음"}]}):
        out = text({"charge": "사기", "guideline_type": "일반사기 1유형", **old})
        assert out.startswith("## status: bad_factor_id") and "guideline_factor_ids" in out
        assert not seen, "a retired argument must not reach the tool"
    assert text({"charge": "사기", "guideline_factor_ids": "[148]", "probation_factor_ids": ["61"]})
    assert seen["guideline_factor_ids"] == [148] and seen["probation_factor_ids"] == [61]
    assert "guideline_factors" not in seen
