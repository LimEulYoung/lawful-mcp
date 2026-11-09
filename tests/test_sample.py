"""What the bundled sample (data/fixture.db) guarantees beyond the harness tests.

The sample is built by ``scripts/build_sample_db.py``. These tests pin the parts a smaller or differently
drawn sample would silently lose: its size, the data each tool path needs, and the statute versions it
leaves out.
"""
from __future__ import annotations

import os

import pytest

from lawful_mcp.harness_local.agents import TOOLS
from lawful_mcp.harness_local.deps import open_db

TOOL = {f.__name__: f for f in TOOLS}
GITHUB_WARNING_BYTES = 50 * 1024 * 1024


class Ctx:
    deps = None


def call(name, **args):
    return TOOL[name](Ctx(), **args)


def test_the_sample_stays_under_githubs_file_size_warning():
    assert os.path.getsize(os.environ["HARNESS_DB"]) < GITHUB_WARNING_BYTES


def test_the_sample_has_every_table_the_harness_reads():
    conn = open_db()
    try:
        for table in ("prec_cases", "prec_case_numbers", "prec_meta", "prec_defendants", "prec_sentences",
                      "st_statutes", "st_articles", "st_addenda", "st_byeolpyo", "st_lapsed", "st_notices",
                      "st_treaties", "statute_aliases", "sg_ranges", "charge_legal_map", "clm_versions"):
            conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchall()
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_no_sampled_law_reads_as_empty_on_any_later_date():
    """The tool picks the current version by date. A whole text dated after the shipped one but shipped
    without its articles would take over once its date passed and serve an empty law."""
    statutes = __import__("lawful_mcp.harness_local.tools.statutes", fromlist=["_articles_on"])
    conn = open_db()
    try:
        law_ids = [r[0] for r in conn.execute(
            "SELECT DISTINCT s.law_id FROM st_statutes s JOIN st_articles a ON a.statute_id=s.id")]
        assert len(law_ids) >= 20
        for law_id in law_ids:
            for when in (statutes._today_iso(), "29991231"):
                assert statutes._articles_on(conn, law_id, when), (law_id, when)
    finally:
        conn.close()


def test_two_syllable_charges_are_found():
    """A trigram index cannot hold a two-character word; the morpheme index is why '사기' is found."""
    assert "- id: " in call("precedent_search", query="사기")


def test_a_holding_from_the_sample_reaches_the_response():
    """The sample reserves cases that carry the court's own holdings so this path runs on real text."""
    conn = open_db()
    try:
        row = conn.execute("SELECT case_number FROM prec_cases WHERE TRIM(holdings) <> '' LIMIT 1").fetchone()
    finally:
        conn.close()
    assert row, "the sample carries no holdings; rebuild it with --holdings-share"
    assert "  holding: " in call("precedent_search", case_number=row[0])


def test_search_syntax_typed_by_a_caller_does_not_break_the_tools():
    out = call("precedent_search", query="사기 AND")
    assert out.startswith("## status: ok") and "- id: " in out
    assert call("statute_search", query="개인정보 AND (보호법 OR)").startswith("## status: ok")


def test_addenda_of_a_sampled_law():
    out = call("statute_read", statute_id=584, articles=["부칙 1"])
    assert "## 부칙 (" in out and "제1조(시행일)" in out


def test_the_sample_answers_a_partner_query_with_its_treaties():
    out = call("statute_search", query="베트남 조세조약")
    first = next(line for line in out.split("## matches", 1)[1].splitlines() if line.startswith("- "))
    assert first.startswith("- trty-1307 ")
    detail = call("statute_read", statute_id="trty-1307", articles=["4"])
    assert "text_kind: 공식 조약 원문" in detail and "- trty-15611 · " in detail
    assert "다자 조세조약" in detail                               # multilateral conventions listed beside
    unsplit = call("statute_read", statute_id="trty-17985")      # a document not split into articles
    assert "## body" in unsplit and "## 첨부 원문: " in unsplit


@pytest.mark.parametrize("case_id", [205021, 234954])
def test_the_cases_the_harness_tests_read_by_id_are_in_the_sample(case_id):
    conn = open_db()
    try:
        assert conn.execute("SELECT 1 FROM prec_cases WHERE id=?", (case_id,)).fetchone()
    finally:
        conn.close()
