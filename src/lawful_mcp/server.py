"""MCP application — five read-only corpus tools over stdio or HTTP.

No authentication and no metering: this server is meant to run next to the
client that uses it, against a corpus you hold. (The hosted deployment puts
those concerns in a separate layer, which is not part of this package.)

Tool descriptions are the harness tools' own docstrings, written in Korean —
the same text the hosted service gives a model. They say only how to use each
tool; what may be quoted and which url to cite ride on every response, so a
client with no system prompt of ours can still use the tools correctly.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import os
import sys
import threading
from types import SimpleNamespace

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from . import config
from .deps import build_deps, build_dive_subagent
from .harness_local.agents import TOOLS
from .harness_local.tools.sentencing_analysis import ChargesArg, OptStrArg

logger = logging.getLogger("lawful-mcp")

_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
_PORT = os.environ.get("MCP_PORT", "8100")
# Extra Host headers to accept, comma separated. The SDK's DNS-rebinding
# guard checks the Host header, so a reverse proxy's public name must be
# listed here or requests through it are rejected.
_EXTRA_HOSTS = [
    h.strip() for h in os.environ.get("MCP_EXTRA_HOSTS", "").split(",") if h.strip()
]

_INSTRUCTIONS = (
    "한국 법률 검색 도구 모음 — 판례·법령·행정규칙(고시·훈령·예규)·양형기준 corpus 기반. "
    "판례 검색(precedent_search) 후 필요할 때 본문 확인(precedent_dive — 법리는 판결문 원문 그대로), "
    "법령·행정규칙 찾기(statute_search)와 조문 본문(statute_read), 양형(sentencing_analysis — "
    "기준·인자·실선고 분포를 한 응답에). 판례 출처 표기는 도구가 반환한 url만 사용."
)

mcp = MCPServer("lawful-mcp", instructions=_INSTRUCTIONS)


def _reclaim_root_logging() -> None:
    """Reclaim root logger hijacked by MCP SDK's MCPServer.

    MCPServer(...) calls basicConfig(level=INFO, handlers=[RichHandler])
    internally. RichHandler wraps at 80 columns in non-TTY pipes and lets
    libraries like httpx log URL queries at INFO. We replace it with a
    standard stream handler.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.__class__.__module__.startswith("rich"):
            root.removeHandler(handler)
    if not root.handlers:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(stream)
    root.setLevel(logging.INFO)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


_reclaim_root_logging()


def _streamable_app():
    """Build the streamable HTTP ASGI app for the MCP endpoint.

    mcp 2.x passes transport security and JSON response flags to
    streamable_http_app() rather than the MCPServer constructor.
    """
    return mcp.streamable_http_app(
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(
            allowed_hosts=[
                "127.0.0.1",
                f"127.0.0.1:{_PORT}",
                "localhost",
                f"localhost:{_PORT}",
                *_EXTRA_HOSTS,
            ],
            allowed_origins=[f"https://{h}" for h in _EXTRA_HOSTS],
        ),
    )

# The harness reads its own variable names; map the public settings onto them
# before any tool runs. The dive bound is a module constant read at call time.
config.export_corpus_path()
config.export_jev_key()
importlib.import_module(".harness_local.tools.precedent_dive", __package__).DIVE_TIMEOUT_S = (
    config.dive_timeout_s()
)

# One sub-agent for the process: building a model client per call would pay
# connection setup on every dive. None when unconfigured, which is why the
# dive tool is registered conditionally.
_dive_subagent = build_dive_subagent()

# The tool list is the harness's (`harness_local.agents.TOOLS`). The wrappers
# below repeat each tool's arguments because the MCP layer builds its schema
# from the wrapper's signature; the names and arguments must stay those of the
# harness tools.
_T = {f.__name__: f for f in TOOLS}


def _ctx() -> SimpleNamespace:
    """Shuttle context for a tool call.

    The tools take a pydantic-ai ``RunContext`` but only read ``.deps`` (the
    dive sub-agent keeps its own usage), so a namespace is enough. Fresh per
    call.
    """
    return SimpleNamespace(deps=build_deps(_dive_subagent), usage=None)


def _tool(fn, *, title: str, **hints):
    """Register a tool: description = the harness tool's docstring, plus behaviour annotations.

    Every tool here is read-only, so all four hints are set explicitly: the
    MCP default for ``destructiveHint`` is true, and a client that sees no
    annotation has to assume the worst.
    """
    return mcp.tool(description=inspect.getdoc(fn), annotations=ToolAnnotations(title=title, **hints))


@_tool(
    _T["precedent_search"],
    title="판례 검색",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
def precedent_search(
    query: str | None = None,
    issue: str | None = None,
    case_number: str | None = None,
    court_level: str | None = None,
    court_name: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> str:
    return _T["precedent_search"](
        _ctx(),
        query=query,
        issue=issue,
        case_number=case_number,
        court_level=court_level,
        court_name=court_name,
        year_from=year_from,
        year_to=year_to,
    )


@_tool(
    _T["statute_search"],
    title="법령 찾기",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
def statute_search(
    query: str | None = None,
    offense_date: str | None = None,
) -> str:
    return _T["statute_search"](_ctx(), query=query, offense_date=offense_date)


@_tool(
    _T["statute_read"],
    title="법령 본문",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
def statute_read(
    statute_id: int | str | None = None,
    articles: list[str] | None = None,
    offense_date: str | None = None,
) -> str:
    return _T["statute_read"](
        _ctx(), statute_id=statute_id, articles=articles, offense_date=offense_date
    )


@_tool(
    _T["sentencing_analysis"],
    title="양형 분석",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
def sentencing_analysis(
    # The harness's own argument types. The MCP layer json-decodes a string
    # argument before validation unless its annotation is exactly ``str``
    # (SDK ``pre_parse_json``), so a list sent as a JSON string arrives as a
    # list; the validators fold a bare string into a one-charge list and a
    # list given for ``offense_date`` into a string.
    charges: ChargesArg,
    offense_date: OptStrArg = None,
) -> str:
    return _T["sentencing_analysis"](_ctx(), charges=charges, offense_date=offense_date)


if _dive_subagent is not None:

    @_tool(
        _T["precedent_dive"],
        title="판례 본문 분석",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
    async def precedent_dive(case_id: int, question: str) -> str:
        return await _T["precedent_dive"](_ctx(), case_id=case_id, question=question)


def build_app():
    """Starlette app serving the MCP endpoint at ``/mcp``."""
    import contextlib

    from starlette.applications import Starlette

    http_app = _streamable_app()

    @contextlib.asynccontextmanager
    async def _lifespan(_app):
        async with mcp.session_manager.run():
            yield

    return Starlette(routes=[*http_app.routes], lifespan=_lifespan)


def _warmup() -> None:
    """Load the morphological analyser and warm the FTS page cache.

    First call after start would otherwise pay both. Failure is not fatal:
    the work happens at call time anyway.
    """
    try:
        _T["precedent_search"](_ctx(), query="손해배상 계약 해지")
        logger.info("warmup done")
    except Exception as e:  # noqa: BLE001
        logger.warning("warmup skipped: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lawful MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="stdio for a local client (default), http for a network endpoint",
    )
    parser.add_argument("--host", default=_HOST)
    parser.add_argument("--port", type=int, default=int(_PORT))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("corpus: %s", config.corpus_db_path())
    if config.dive_config() is None:
        logger.info("precedent_dive disabled (set DIVE_API_KEY, DIVE_BASE_URL, DIVE_MODEL)")
    if config.jev_api_key() is None:
        logger.info("issue reordering off (set JEV_API_KEY); results keep their BM25 order")

    if args.transport == "stdio":
        mcp.run()
        return

    threading.Thread(target=_warmup, daemon=True).start()
    import uvicorn

    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
