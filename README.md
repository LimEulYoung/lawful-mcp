# lawful-mcp

한국 법률 리서치·문서작성 MCP 서버 (streamable HTTP).

판례 검색·본문 질의, 법령·고시 조문 조회, 양형 분포·양형기준 계산, 법률 문서 생성(docx·PDF)
도구를 MCP 로 제공한다.

## 실행

```bash
pip install "mcp>=1.9" uvicorn starlette httpx
python proxy_mcp.py            # http://localhost:8080/
curl localhost:8080/healthz    # 상태 확인
```

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MCP_UPSTREAM_URL` | (내장 기본값) | 도구 실행 백엔드 엔드포인트 |
| `MCP_UPSTREAM_KEY` | (없음) | 백엔드 인증 키(Bearer), 필요 시 |
| `PORT` / `HOST` | `8080` / `0.0.0.0` | listen 주소 |

## Docker

```bash
docker build -t lawful-mcp .
docker run -p 8080:8080 lawful-mcp
```
