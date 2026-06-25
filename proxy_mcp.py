"""lawful-mcp — 한국 법률 리서치·문서작성 MCP 서버 (streamable HTTP).

판례 검색·본문 질의, 법령·고시 조문 조회, 양형 분포·양형기준 계산, 법률 문서 생성
도구를 MCP 프로토콜로 노출한다. 도구 실행은 백엔드 엔드포인트에 위임한다.

env:
  MCP_UPSTREAM_URL  도구 실행 백엔드 엔드포인트
  MCP_UPSTREAM_KEY  백엔드 인증 키(Bearer) — 필요 시
  PORT              listen 포트 (기본 8080)
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

logger = logging.getLogger("lawful-mcp")

UPSTREAM_URL = os.environ.get("MCP_UPSTREAM_URL", "https://mcp.crow-tit.com/mcp")
UPSTREAM_KEY = os.environ.get("MCP_UPSTREAM_KEY", "")
_HEADERS = {"Authorization": f"Bearer {UPSTREAM_KEY}"} if UPSTREAM_KEY else None


@contextlib.asynccontextmanager
async def _upstream():
    """백엔드와 초기화된 세션 1개를 연다(요청 단위, 자동 정리)."""
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
        # 콘텐츠를 그대로 메시지로 올려 도구 오류로 전파한다.
        text = "; ".join(
            getattr(c, "text", "") for c in result.content if getattr(c, "text", "")
        )
        raise RuntimeError(text or f"upstream tool '{name}' failed")
    # 도구가 outputSchema 를 광고하면 structuredContent 까지 함께 돌려줘야
    # 출력 검증을 통과한다(텍스트만 반환 시 'no structured output' 에러).
    if result.structuredContent is not None:
        return result.content, result.structuredContent
    return result.content


# ---------- streamable HTTP app ----------

_session_manager = StreamableHTTPSessionManager(
    app=server, json_response=True, stateless=True,
)


async def _handle_mcp(scope, receive, send):
    await _session_manager.handle_request(scope, receive, send)


async def _health(_request):
    return JSONResponse({"status": "ok"})


@contextlib.asynccontextmanager
async def _lifespan(_app):
    async with _session_manager.run():
        logger.info("lawful-mcp ready")
        yield


# 경로에 관계없이 MCP 프로토콜을 처리하도록 catch-all Mount("/") 로 둔다.
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
