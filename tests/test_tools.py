"""Tool smoke tests against the bundled sample corpus.

No network and no API keys: every test here runs on data/fixture.db alone.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from lawful_mcp import tools
from lawful_mcp.tools import statutes
from lawful_mcp.config import corpus_db_path
from lawful_mcp.deps import build_deps, open_db


@pytest.fixture
def ctx():
    """Shuttle context, same shape the server passes to the tools."""
    return SimpleNamespace(deps=build_deps(None), usage=None)


def test_fixture_is_present_and_readable():
    assert corpus_db_path().exists(), "sample corpus missing: data/fixture.db"
    conn = open_db()
    try:
        assert conn.execute("SELECT COUNT(*) FROM prec_cases").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM st_articles").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM sg_ranges").fetchone()[0] > 0
    finally:
        conn.close()


def test_precedent_search_by_keyword(ctx):
    out = tools.precedent_search(ctx, query="손해배상 계약 해지")
    assert "## matches" in out or "status" in out
    assert "id:" in out


def test_precedent_search_finds_two_syllable_charge(ctx):
    """Two-syllable charges are why the morpheme index exists.

    A trigram index cannot produce a trigram from a two-character token, so
    this query returns nothing without the morphological FTS table.
    """
    out = tools.precedent_search(ctx, query="사기")
    assert "id:" in out


def test_precedent_search_shows_a_holding_from_the_sample(ctx):
    """The court's own issue summary reaches the response.

    The sample reserves a share of its cases for ones that carry a holding
    (`scripts/build_sample_db.py`) precisely so this path runs against real
    corpus text rather than only against constructed input.
    """
    conn = open_db()
    try:
        row = conn.execute(
            "SELECT case_number FROM prec_cases "
            "WHERE COALESCE(TRIM(holdings), '') <> '' LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert row, "the sample carries no holdings; rebuild it with --holdings-share"
    out = tools.precedent_search(ctx, case_number=row[0])
    assert "  holding: " in out, out


def test_precedent_search_requires_an_argument(ctx):
    out = tools.precedent_search(ctx)
    assert "status" in out


def test_statute_lookup_by_name(ctx):
    out = tools.statute_lookup(ctx, query="형법")
    assert "형법" in out


def test_statute_lookup_prefers_the_law_the_query_names(ctx):
    """Against the real corpus, not a constructed collision.

    A short name sits inside a longer one, so '군형법 제92조' names both and the
    answer is the longer. An official abbreviation counts as naming the law
    too — the search used to miss those entirely, answering 「할부거래법」 with
    「증권거래법」.
    """
    def first(query):
        """Name of the top hit, out of `- <id> <name> (<kind>)`."""
        for line in tools.statute_lookup(ctx, query=query).splitlines():
            if line.startswith("- "):
                return line[2:].split(" ", 1)[1].rsplit(" (", 1)[0]
        return None

    assert first("군형법 제92조") == "군형법"
    assert first("형법 제347조 사기") == "형법"
    assert first("신용정보법 신용카드 발급") == "신용정보의 이용 및 보호에 관한 법률"
    # A coincidence across a word boundary is not the query naming a law:
    # '손해배상 법률' contains 「상법」 by letters alone.
    assert first("손해배상 법률 상담 절차") != "상법"


def test_statute_search_previews_the_article_the_query_matched(ctx):
    """The preview is a choice among the law's articles, not a fixed one.

    Article 1 states a purpose in every act, so it neither answers the query
    nor tells two laws apart. The field has no consumer in this build — the
    response markdown does not print it — but production renders it, and the
    contract is shared.
    """
    conn = open_db()
    try:
        by_name = {
            m["name"]: m["preview"]
            for m in statutes._search_statutes(conn, "개인정보 주민등록번호", 5)
        }
    finally:
        conn.close()
    preview = by_name.get("개인정보 보호법", "")
    assert preview.startswith("제24조의2"), preview


def test_statute_preview_says_when_it_was_cut():
    """A cut preview ends in an ellipsis; a short one is left alone."""
    assert statutes._clip_preview("제1조(목적) 짧다") == "제1조(목적) 짧다"
    long = "가" * (statutes.SEARCH_PREVIEW_MAX_CHARS + 50)
    clipped = statutes._clip_preview(long)
    assert clipped.endswith("…")
    assert len(clipped) == statutes.SEARCH_PREVIEW_MAX_CHARS + 1


def test_statute_lookup_article_text(ctx):
    """Quick-access ids in the tool description must resolve in the corpus."""
    out = tools.statute_lookup(ctx, statute_id=578, articles=["347"])
    assert "347" in out
    assert "사기" in out


def test_statute_lookup_outline_without_articles(ctx):
    out = tools.statute_lookup(ctx, statute_id=584)
    assert "민법" in out


# --- one law, one name -------------------------------------------------------
#
# A law renamed by a wholesale amendment keeps its law_id, so the old-name row
# and the current one are the same law. Asking by the old id used to get the
# current name from an outline and the old name from a detail — while the
# detail's body was current text. One id looked like two laws.
RENAMED_OLD_ID = 31554        # 청소년의 성보호에 관한 법률 (2008)
RENAMED_NOW = "아동ㆍ청소년의 성보호에 관한 법률"


def _header_name(out: str) -> str:
    """Law name out of '## statute: <name> (id=N) — ...'."""
    for line in out.splitlines():
        if line.startswith("## statute: "):
            return line[len("## statute: "):].split(" (id=", 1)[0]
    return ""


def test_statute_lookup_answers_with_one_name_per_id(ctx):
    outline = tools.statute_lookup(ctx, statute_id=RENAMED_OLD_ID)
    detail = tools.statute_lookup(ctx, statute_id=RENAMED_OLD_ID, articles=["8"])
    assert _header_name(outline) == _header_name(detail) == RENAMED_NOW
    # And the name the caller asked under is answered for, not dropped.
    for out in (outline, detail):
        assert "옛 이름" in out
        assert "청소년의 성보호에 관한 법률" in out


def test_statute_outline_numbers_survive_a_round_trip(ctx):
    """An outline may only print article numbers that can be asked for.

    A branch is already inside `article_no` in some load generations, and
    appending it again produced '16의2-2' — a number that does not exist, so
    a model that read it back got a miss.
    """
    out = tools.statute_lookup(ctx, statute_id=32228)          # 형사소송법
    numbers = [
        line[2:].split(":", 1)[0]
        for line in out.splitlines()
        if line.startswith("- ") and ":" in line
    ]
    assert not [n for n in numbers if re.fullmatch(r"\d+의\d+-\d+", n)]
    branched = [n for n in numbers if "의" in n]
    assert branched, "fixture no longer carries a branch inside article_no"
    back = tools.statute_lookup(ctx, statute_id=32228, articles=[branched[0]])
    assert "## missing" not in back
    assert "text_kind" in back


def test_a_current_lookup_is_not_reported_as_a_dated_one(ctx):
    """`offense_date` is filled in internally for current lookups too, so only
    a date the caller actually passed may reach the response."""
    current = tools.statute_lookup(ctx, statute_id=578, articles=["347"])
    assert "시점 조회" not in current
    dated = tools.statute_lookup(
        ctx, statute_id=578, articles=["347"], offense_date="2015-06-01")
    assert "시점 조회: 2015-06-01" in dated


def test_search_marks_a_name_that_is_no_longer_current():
    """A result list has to say which of its names are still in use.

    The current-edition verdict reads every edition of the law, not just the
    ones the query matched: a law matched under its old name alone would
    otherwise be alone in its group and so count as current itself.
    """
    conn = open_db()
    try:
        rows = conn.execute(
            "SELECT id, law_id, name, effective_date FROM st_statutes "
            "WHERE id IN (?, ?)", (RENAMED_OLD_ID, 737)).fetchall()
        refs = statutes._current_refs(conn, rows)
    finally:
        conn.close()
    assert 737 not in refs, "the current edition needs no marker"
    ref = refs[RENAMED_OLD_ID]
    assert ref["id"] == 737 and ref["renamed"] is True

    md = statutes._format_response_md({
        "status": "ok", "mode": "list",
        "matches": [{"statute_id": RENAMED_OLD_ID, "law_id": "002044",
                     "name": "청소년의 성보호에 관한 법률", "kind": "법률",
                     "effective_date": "20080229", "current": ref,
                     "is_repealed": False}],
    })
    assert "2008-02-29 시행" in md
    assert "현행 아님" in md and "statute_id=737" in md
    assert "폐지" not in md, "a rename must never be reported as a repeal"


def test_a_rename_is_not_reported_as_a_repeal():
    """`st_backfill_log` records superseded editions loaded after the fact, not
    repeals: 278 of its 738 rows are renames whose successor is alive. A repeal
    is a row in it with no later edition in force — a pending amendment is not
    one. The sample carries no such table, so build one over it.
    """
    conn = open_db()
    try:
        conn.execute("CREATE TEMP TABLE st_backfill_log "
                     "(statute_id INTEGER, status TEXT)")
        conn.executemany("INSERT INTO st_backfill_log VALUES (?, 'ok')",
                         [(RENAMED_OLD_ID,), (20276,)])
        sids = [RENAMED_OLD_ID, 20276]
        assert statutes._backfilled_sids(conn, sids) == {RENAMED_OLD_ID, 20276}
        # 31554 has later editions in force; 20276's only successor is a
        # 시행예정 row, which does not count.
        assert statutes._repealed_sids(conn, sids) == {20276}
    finally:
        conn.close()


def test_the_repeal_verdict_fails_soft_without_its_table():
    """The bundled sample predates `st_backfill_log`, so search must still run
    and simply label nothing repealed."""
    conn = open_db()
    try:
        assert statutes._backfilled_sids(conn, [578, 584]) == set()
        assert statutes._repealed_sids(conn, [578, 584]) == set()
        assert statutes._search_statutes(conn, "형법", 3)
    finally:
        conn.close()


def _a_rule_id(ctx) -> int:
    """An administrative rule that the search actually surfaces."""
    conn = open_db()
    try:
        return conn.execute(
            "SELECT n.id FROM st_notices n JOIN st_notice_articles a ON a.notice_id = n.id "
            "GROUP BY n.id ORDER BY COUNT(a.id) DESC LIMIT 1"
        ).fetchone()[0]
    finally:
        conn.close()


def test_statute_lookup_rule_id_needs_its_prefix(ctx):
    """The prefix is what separates the two id spaces, which overlap.

    Fed back verbatim the rule opens; stripped to a bare integer it must not
    quietly resolve to whatever statute holds that number.
    """
    rid = _a_rule_id(ctx)
    prefixed = tools.statute_lookup(ctx, statute_id=f"admrul-{rid}")
    assert "## status: ok" in prefixed
    assert f"admrul-{rid}" in prefixed

    bare = tools.statute_lookup(ctx, statute_id=rid)
    assert "## status: ok" not in bare


def test_statute_lookup_search_hands_back_usable_rule_ids(ctx):
    """Whatever the list mode prints as an id has to work as an input."""
    out = tools.statute_lookup(ctx, query="규정")
    ids = [
        line.split()[1] for line in out.splitlines()
        if line.startswith("- ") and line[2:].split()[:1]
    ]
    rule_ids = [i for i in ids if i.startswith("admrul-")]
    assert rule_ids, f"no administrative rule in the results:\n{out}"
    assert "## status: ok" in tools.statute_lookup(ctx, statute_id=rule_ids[0])


def test_statute_lookup_accepts_a_pasted_url(ctx):
    """Models paste page urls back; case and host must not matter."""
    rid = _a_rule_id(ctx)
    out = tools.statute_lookup(ctx, statute_id=f"https://example.test/statutes/ADMRUL-{rid}")
    assert "## status: ok" in out


def test_statute_lookup_says_so_when_the_id_is_unreadable(ctx):
    """A malformed id gets its own status, not a generic missing_input."""
    out = tools.statute_lookup(ctx, statute_id="근로기준법")
    assert "bad_statute_id" in out


def test_statute_lookup_over_the_article_cap_answers_the_first_eight(ctx):
    """Above the cap the call proceeds and reports what it deferred."""
    out = tools.statute_lookup(
        ctx, statute_id=578, articles=[str(n) for n in range(250, 262)]
    )
    assert "## status: ok" in out
    assert "too_many_articles" not in out
    assert "261" in out          # named as deferred in the message


def test_statute_lookup_refuses_a_table_as_an_article(ctx):
    """'별표5' must not become 제5조 and come back labelled as official text.

    Reading the number off a structural unit answers a question nobody asked
    and calls it the source, which is worse than finding nothing.
    """
    out = tools.statute_lookup(ctx, statute_id=578, articles=["별표5"])
    assert "## status: bad_articles" in out
    assert "별표" in out                # says which word was refused
    assert "제5조" not in out


def test_statute_lookup_names_the_units_it_dropped(ctx):
    """A mixed call reads the articles and still reports what it cut."""
    out = tools.statute_lookup(ctx, statute_id=578, articles=["347", "별표5"])
    assert "## status: ok" in out
    assert "별표5 는 제외했습니다" in out


def test_statute_outline_never_prints_a_null_article_number(ctx):
    """Structural headings carry no article number; `str(None)` would leak."""
    out = tools.statute_lookup(ctx, statute_id=f"admrul-{_a_rule_id(ctx)}")
    assert "None" not in out


def test_sentence_statistics_returns_candidates(ctx):
    """A charge string yields charge_id candidates, not statistics."""
    out = tools.sentence_statistics(ctx, charges="절도")
    assert "candidates" in out or "charge_id" in out


def test_compute_sentencing_range_lookup_stage(ctx):
    out = tools.compute_sentencing_range(ctx, charge="사기")
    assert "법정형" in out or "status" in out


def test_compute_sentencing_range_refuses_a_numeric_charge(ctx):
    """An article number or charge_id in the charge slot gets its own status.

    Read as an ordinary not_found it means "no guideline for this offence",
    and callers repeat the number instead of sending a name.
    """
    out = tools.compute_sentencing_range(ctx, charge="[299,298,297]")
    assert "charge_numeric" in out
    assert "not_found" not in out


def test_compute_sentencing_range_strips_a_leading_number_tag(ctx):
    """"[1]상해" is a judgment's provision tag carried along with the name."""
    out = tools.compute_sentencing_range(ctx, charge="[1]상해")
    assert "## status: ok" in out
    assert "## charge: 상해" in out


def test_compute_sentencing_range_keeps_a_bare_number_numeric(ctx):
    """Stripping the tag must not empty the string and lose the numeric path."""
    assert "charge_numeric" in tools.compute_sentencing_range(ctx, charge="[123]")


def test_compute_sentencing_range_splits_a_list_of_charges(ctx):
    """A list is reported per charge rather than joined into one lookup key.

    Joining makes a key that cannot exist, and the empty result reads as
    "no guideline for these offences".
    """
    out = tools.compute_sentencing_range(ctx, charge=["주거침입", "퇴거불응"])
    assert "## status: multiple_charges" in out
    assert "not_found" not in out
    assert "- 주거침입: 매칭 ok" in out
    assert "- 퇴거불응: 매칭 ok" in out


def test_compute_sentencing_range_strips_wrapping_quotes(ctx):
    """A charge sent as a string literal still matches."""
    out = tools.compute_sentencing_range(ctx, charge='"협박"')
    assert "## status: ok" in out
    assert "## charge: 협박" in out


def test_compute_sentencing_range_keeps_quoted_numbers_numeric(ctx):
    """Stripping the quotes must not let a bare number reach the lookup."""
    assert "charge_numeric" in tools.compute_sentencing_range(ctx, charge='"298"')


def test_citation_urls_follow_configured_base(ctx, monkeypatch):
    monkeypatch.setenv("CASE_URL_BASE", "https://example.test")
    out = tools.precedent_search(ctx, query="손해배상")
    assert "https://example.test/cases/" in out
    assert "lawful.crow-tit.com" not in out


def test_precedent_dive_without_model_is_not_registered(monkeypatch):
    """The dive tool is left out when no model is configured, not broken."""
    from lawful_mcp.deps import build_dive_subagent

    monkeypatch.delenv("DIVE_API_KEY", raising=False)
    assert build_dive_subagent() is None


def test_mcp_layer_lets_a_json_looking_charge_through():
    """The validation layer must not reject what it just parsed for us.

    A JSON-looking string argument is json.loads-ed before the tool runs, so
    `charge='[299,298,297]'` becomes a list. A narrow annotation turned that into
    a validation error and the charge_numeric hint never reached the caller —
    over MCP only, since the tool itself takes the argument wide.
    """
    import anyio

    from lawful_mcp import server

    async def call(value):
        return await server.mcp.call_tool(
            "compute_sentencing_range", {"charge": value})

    assert "charge_numeric" in str(anyio.run(call, "[299,298,297]"))
    # Shapes that already worked must keep working.
    assert "charge_numeric" in str(anyio.run(call, "298"))
    assert "charge_numeric" in str(anyio.run(call, "297의2"))


def test_sibling_string_arguments_stay_narrow():
    """The asymmetry above is a decision, not an oversight — keep it visible."""
    import anyio

    from lawful_mcp import server

    schemas = {
        t.name: t.model_dump(by_alias=True, exclude_none=True)["inputSchema"]
        for t in anyio.run(server.mcp.list_tools)
    }

    def types(tool, field):
        prop = schemas[tool]["properties"][field]
        if "anyOf" in prop:
            return {b["type"] for b in prop["anyOf"] if "type" in b}
        return {prop["type"]} if "type" in prop else set()

    assert "array" in types("compute_sentencing_range", "charge")
    for tool, field in (
        ("precedent_search", "query"),
        ("sentence_statistics", "charges"),
        ("compute_sentencing_range", "statute_choice"),
    ):
        assert "array" not in types(tool, field), f"{tool}.{field}"


def test_current_protocol_revision_is_served_not_rejected():
    """Modern clients (e.g. claude.ai) send protocol revision 2026-07-28 with _meta."""
    import anyio
    import httpx
    from mcp_types import (
        CLIENT_CAPABILITIES_META_KEY,
        CLIENT_INFO_META_KEY,
        PROTOCOL_VERSION_META_KEY,
    )
    from mcp_types.version import SUPPORTED_PROTOCOL_VERSIONS

    modern = "2026-07-28"
    assert modern in SUPPORTED_PROTOCOL_VERSIONS, f"SDK does not support {modern}"

    meta = {
        PROTOCOL_VERSION_META_KEY: modern,
        CLIENT_CAPABILITIES_META_KEY: {},
        CLIENT_INFO_META_KEY: {"name": "test-client", "version": "1"},
    }

    from lawful_mcp.server import _streamable_app, mcp

    async def run():
        app = _streamable_app()
        async with mcp.session_manager.run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
                r = await client.post(
                    "/mcp",
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "MCP-Protocol-Version": modern,
                        "mcp-method": "tools/list",
                    },
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": meta}},
                )
                assert r.status_code == 200, f"{modern} rejected: {r.text[:200]}"
                tools = r.json()["result"]["tools"]
                tool_names = {t["name"] for t in tools}
                assert "precedent_search" in tool_names
                assert "inputSchema" in tools[0]
                assert "readOnlyHint" in tools[0]["annotations"]

    anyio.run(run)

