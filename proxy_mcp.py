"""playmcp-proxy — PlayMCP in KC 발급 Endpoint 용 얇은 패스스루 MCP 서버.

공모전 규정상 등록 URL 은 반드시 "PlayMCP in KC 에서 발급한 Endpoint" 여야 하므로,
KC 컨테이너에는 이 프록시만 띄우고 실제 도구 실행은 기존 운영 MCP 서버
(`https://mcp.crow-tit.com/mcp`, Lightsail 박스 = harness.db ~14GB·DeepSeek 키 보유)로
그대로 포워딩한다. 즉 무거운 자원(코퍼스·키·harness_local)은 컨테이너에 전혀 넣지 않는다.

동작:
  - list_tools  → 업스트림에 connect 해 도구 목록(스키마·설명 포함)을 그대로 반환.
  - call_tool   → 업스트림에 같은 인자로 위임하고 콘텐츠를 그대로 반환.
  업스트림 연결은 요청마다 새로 열고 닫는다(stateless — 트래픽이 낮고 단순·견고).

env:
  MCP_UPSTREAM_URL  업스트림 MCP 엔드포인트 (기본 https://mcp.crow-tit.com/mcp)
  MCP_UPSTREAM_KEY  업스트림이 잠겨 있을 때(Bearer) — 현재 오픈 모드라 보통 불필요
  PORT              listen 포트 (PlayMCP/KC 가 주입; 기본 8080)
  HOST              listen 호스트 (기본 0.0.0.0)
"""
from __future__ import annotations

import contextlib
import logging
import os

import mcp.types as types
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

logger = logging.getLogger("playmcp-proxy")

UPSTREAM_URL = os.environ.get("MCP_UPSTREAM_URL", "https://mcp.crow-tit.com/mcp")
UPSTREAM_KEY = os.environ.get("MCP_UPSTREAM_KEY", "")
_HEADERS = {"Authorization": f"Bearer {UPSTREAM_KEY}"} if UPSTREAM_KEY else None


@contextlib.asynccontextmanager
async def _upstream():
    """업스트림 MCP 와 초기화된 세션 1개를 연다(요청 단위, 자동 정리)."""
    async with streamablehttp_client(UPSTREAM_URL, headers=_HEADERS) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


server: Server = Server("lawful-legal")


@server.list_tools()
async def _list_tools() -> list[types.Tool]:
    async with _upstream() as session:
        return (await session.list_tools()).tools


@server.call_tool()
async def _call_tool(name: str, arguments: dict | None):
    async with _upstream() as session:
        result = await session.call_tool(name, arguments or {})
    if result.isError:
        # 콘텐츠를 그대로 메시지로 올려 도구 오류로 전파(프록시가 삼키지 않음).
        text = "; ".join(
            getattr(c, "text", "") for c in result.content if getattr(c, "text", "")
        )
        raise RuntimeError(text or f"upstream tool '{name}' failed")
    return result.content


# ---------- streamable HTTP app ----------

_session_manager = StreamableHTTPSessionManager(
    app=server, json_response=True, stateless=True,
)


async def _handle_mcp(scope, receive, send):
    await _session_manager.handle_request(scope, receive, send)


async def _health(_request):
    return JSONResponse({"status": "ok", "upstream": UPSTREAM_URL})


@contextlib.asynccontextmanager
async def _lifespan(_app):
    async with _session_manager.run():
        logger.info("proxy ready → upstream %s", UPSTREAM_URL)
        yield


# 경로에 관계없이 MCP 프로토콜을 처리하도록 catch-all Mount("/") 로 둔다 —
# PlayMCP 가 발급하는 Endpoint 의 경로 구성에 영향받지 않게 하기 위함.
# (/healthz 는 단순 상태확인용으로 먼저 매칭)
app = Starlette(
    routes=[
        Route("/healthz", _health),
        Mount("/", app=_handle_mcp),
    ],
    lifespan=_lifespan,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
