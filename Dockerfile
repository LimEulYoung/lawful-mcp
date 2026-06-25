# PlayMCP in KC 용 얇은 패스스루 MCP 프록시.
# 무거운 자원(harness.db·키)은 담지 않는다 — 실제 실행은 업스트림(mcp.crow-tit.com)이 처리.
FROM python:3.12-slim

WORKDIR /app

# 의존성은 MCP SDK + ASGI 런타임뿐. 슬림하게 유지.
RUN pip install --no-cache-dir \
    "mcp>=1.9" \
    "uvicorn>=0.30" \
    "starlette>=0.37" \
    "httpx>=0.27"

COPY proxy_mcp.py .

# PlayMCP/KC 가 PORT 를 주입하면 그 값을 사용(없으면 8080).
ENV PORT=8080 HOST=0.0.0.0
EXPOSE 8080

CMD ["python", "proxy_mcp.py"]
