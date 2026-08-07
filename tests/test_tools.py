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


def test_statute_lookup_article_text(ctx):
    """Quick-access ids in the tool description must resolve in the corpus."""
    out = tools.statute_lookup(ctx, statute_id=578, articles=["347"])
    assert "347" in out
    assert "사기" in out


def test_statute_lookup_outline_without_articles(ctx):
    out = tools.statute_lookup(ctx, statute_id=584)
    assert "민법" in out


def test_sentence_statistics_returns_candidates(ctx):
    """A charge string yields charge_id candidates, not statistics."""
    out = tools.sentence_statistics(ctx, charges="절도")
    assert "candidates" in out or "charge_id" in out


def test_compute_sentencing_range_lookup_stage(ctx):
    out = tools.compute_sentencing_range(ctx, charge="사기")
    assert "법정형" in out or "status" in out


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
