"""Tool smoke tests against the bundled sample corpus.

No network and no API keys: every test here runs on data/fixture.db alone.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from legal_search_mcp import tools
from legal_search_mcp.config import corpus_db_path
from legal_search_mcp.deps import build_deps, open_db


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


def test_statute_lookup_article_text(ctx):
    """Quick-access ids in the tool description must resolve in the corpus."""
    out = tools.statute_lookup(ctx, statute_id=578, articles=["347"])
    assert "347" in out
    assert "사기" in out


def test_statute_lookup_outline_without_articles(ctx):
    out = tools.statute_lookup(ctx, statute_id=584)
    assert "민법" in out


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


def test_citation_urls_follow_configured_base(ctx, monkeypatch):
    monkeypatch.setenv("CASE_URL_BASE", "https://example.test")
    out = tools.precedent_search(ctx, query="손해배상")
    assert "https://example.test/cases/" in out
    assert "lawful.crow-tit.com" not in out


def test_precedent_dive_without_model_is_not_registered(monkeypatch):
    """The dive tool is left out when no model is configured, not broken."""
    from legal_search_mcp.deps import build_dive_subagent

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

    from legal_search_mcp import server

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

    from legal_search_mcp import server

    schemas = {t.name: t.inputSchema for t in anyio.run(server.mcp.list_tools)}

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
