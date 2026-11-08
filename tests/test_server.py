"""The public wiring around the copied harness: settings, tool registration and the MCP argument layer.

The harness tools themselves are covered by ``tests/harness`` (the research harness's own tests). These
tests pin what this repository adds: which settings reach the harness, which tools a client sees and
with what description, and how arguments arrive through the MCP layer.
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path

import anyio
import pytest

from lawful_mcp import config
from lawful_mcp.harness_local.agents import TOOLS

HARNESS = {f.__name__: f for f in TOOLS}
HARNESS_DIR = Path(config.__file__).resolve().parent / "harness_local"
SETTINGS = ("CORPUS_DB", "HARNESS_DB", "DIVE_API_KEY", "DIVE_BASE_URL", "DIVE_MODEL", "DIVE_TIMEOUT",
            "JEV_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture
def fresh_server(monkeypatch):
    """Import ``lawful_mcp.server`` anew under the given environment; everything is restored afterwards."""
    dive = importlib.import_module("lawful_mcp.harness_local.tools.precedent_dive")
    monkeypatch.setattr(dive, "DIVE_TIMEOUT_S", dive.DIVE_TIMEOUT_S)
    corpus = os.environ["HARNESS_DB"]
    for name in SETTINGS:  # setenv first, so teardown restores the original state even if absent
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("HARNESS_DB", corpus)

    def load(**env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        sys.modules.pop("lawful_mcp.server", None)
        return importlib.import_module("lawful_mcp.server")

    yield load
    sys.modules.pop("lawful_mcp.server", None)


def _tools(server):
    return {t.name: t for t in anyio.run(server.mcp.list_tools)}


def _schema(tool):
    return tool.model_dump(by_alias=True, exclude_none=True)["inputSchema"]


def _types(prop):
    branches = prop.get("anyOf", [prop])
    return {b["type"] for b in branches if "type" in b}


# ---------- settings ----------

def test_corpus_db_wins_over_the_harness_name_and_defaults_to_the_sample(monkeypatch):
    monkeypatch.delenv("CORPUS_DB", raising=False)
    monkeypatch.delenv("HARNESS_DB", raising=False)
    assert config.corpus_db_path() == config.DEFAULT_CORPUS_DB
    monkeypatch.setenv("HARNESS_DB", "/h.db")
    assert str(config.corpus_db_path()) == "/h.db"
    monkeypatch.setenv("CORPUS_DB", "/c.db")
    config.export_corpus_path()
    assert os.environ["HARNESS_DB"] == "/c.db"


def test_the_jev_key_is_the_only_switch_for_issue_reordering(fresh_server):
    fresh_server(OPENROUTER_API_KEY="exported-for-something-else")
    assert os.environ["OPENROUTER_API_KEY"] == ""
    fresh_server(JEV_API_KEY="jev-key")
    assert os.environ["OPENROUTER_API_KEY"] == "jev-key"


def test_issue_without_a_key_is_the_plain_search():
    """Fail-open: with no key the reordering step is skipped and the response is the search without an issue."""
    search = HARNESS["precedent_search"]

    class Ctx:
        deps = None

    plain = search(Ctx(), query="음주운전 무면허운전")
    assert plain.startswith("## status: ok")
    assert search(Ctx(), query="음주운전 무면허운전", issue="음주운전과 무면허운전은 실체적 경합인가") == plain


def test_dive_timeout_reaches_the_harness(fresh_server):
    dive = importlib.import_module("lawful_mcp.harness_local.tools.precedent_dive")
    fresh_server(DIVE_TIMEOUT="42")
    assert dive.DIVE_TIMEOUT_S == 42.0
    fresh_server(DIVE_TIMEOUT="soon")
    assert dive.DIVE_TIMEOUT_S == config.DEFAULT_DIVE_TIMEOUT_S


# ---------- registration ----------

def test_without_a_dive_model_three_tools_are_registered(fresh_server):
    assert sorted(_tools(fresh_server())) == ["precedent_search", "sentencing_analysis", "statute_lookup"]


def test_with_a_dive_model_all_four_harness_tools_are_registered(fresh_server):
    server = fresh_server(DIVE_API_KEY="k", DIVE_BASE_URL="https://example.test/v1", DIVE_MODEL="m")
    assert sorted(_tools(server)) == sorted(HARNESS)


def test_descriptions_are_the_harness_docstrings_and_every_hint_is_explicit(fresh_server):
    server = fresh_server(DIVE_API_KEY="k", DIVE_BASE_URL="https://example.test/v1", DIVE_MODEL="m")
    for name, tool in _tools(server).items():
        assert tool.description == inspect.getdoc(HARNESS[name]), name
        hints = tool.annotations
        assert hints.read_only_hint is True and hints.destructive_hint is False, name
        assert hints.idempotent_hint is not None and hints.open_world_hint is not None, name


def test_wrapper_arguments_are_the_harness_arguments(fresh_server):
    server = fresh_server(DIVE_API_KEY="k", DIVE_BASE_URL="https://example.test/v1", DIVE_MODEL="m")
    for name, tool in _tools(server).items():
        params = [p for p in inspect.signature(HARNESS[name]).parameters if p != "ctx"]
        assert list(_schema(tool)["properties"]) == params, name


def test_model_facing_text_names_no_tool_this_server_lacks():
    """The server registers no web tools; a response pointing the model at one sends it looking for a call it
    cannot make."""
    for path in HARNESS_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "web_fetch" not in text and "web_search" not in text, path


# ---------- the MCP argument layer ----------

def test_charges_is_a_required_list_and_the_other_strings_stay_narrow(fresh_server):
    tools = _tools(fresh_server())
    schema = _schema(tools["sentencing_analysis"])
    assert _types(schema["properties"]["charges"]) == {"array"}
    assert schema["required"] == ["charges"]
    for tool, field in (("precedent_search", "query"), ("precedent_search", "issue"),
                        ("precedent_search", "case_number"), ("precedent_search", "court_name"),
                        ("statute_lookup", "query"), ("sentencing_analysis", "offense_date")):
        assert "array" not in _types(_schema(tools[tool])["properties"][field]), f"{tool}.{field}"


def test_charges_arrive_as_a_list_however_the_client_sends_them(fresh_server):
    """The SDK json-decodes a string argument unless its annotation is exactly ``str`` (``pre_parse_json``),
    and the harness's validator folds a bare string into a one-charge list."""
    server = fresh_server()

    def text(args):
        return str(anyio.run(server.mcp.call_tool, "sentencing_analysis", args))

    as_list = text({"charges": ["사기", "폭행"]})
    assert "## status: ok" in as_list
    assert text({"charges": '["사기", "폭행"]'}) == as_list
    assert text({"charges": "사기"}) == text({"charges": ["사기"]})


def test_a_list_given_for_a_search_query_is_refused_not_searched(fresh_server):
    """Joining it would quietly run a search for '음주운전, 무면허' — there is no answer prepared for a list."""
    from mcp.server.mcpserver.exceptions import ToolError

    server = fresh_server()
    with pytest.raises(ToolError, match="valid string"):
        anyio.run(server.mcp.call_tool, "precedent_search", {"query": ["음주운전", "무면허"]})


def test_current_protocol_revision_is_served_not_rejected(fresh_server):
    """Modern clients (e.g. claude.ai) send protocol revision 2026-07-28 with _meta."""
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
    server = fresh_server()

    async def run():
        app = server._streamable_app()
        async with server.mcp.session_manager.run():
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
                assert "precedent_search" in {t["name"] for t in tools}
                assert "readOnlyHint" in tools[0]["annotations"]

    anyio.run(run)
