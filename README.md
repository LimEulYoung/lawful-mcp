# playmcp-proxy

PlayMCP in KC(공모전) 발급 Endpoint 용 **얇은 패스스루 MCP 프록시**.

공모전 규정상 등록 URL 은 반드시 *PlayMCP in KC 에서 발급한 Endpoint* 여야 한다.
하지만 본 프로젝트의 진짜 MCP 서버(`server/mcp_server.py`)는 ~14GB `harness.db`,
DeepSeek/Tavily 등 `.env` 키, `harness_local` 라이브러리에 의존해 격리 컨테이너에 통째로
넣을 수 없다. 그래서 KC 컨테이너에는 이 프록시만 올리고, 실제 도구 실행은 이미 운영 중인
업스트림(`https://mcp.crow-tit.com/mcp`, Lightsail 박스)으로 그대로 포워딩한다.

```
PlayMCP 발급 URL ──▶ [KC 컨테이너: proxy_mcp.py] ──▶ https://mcp.crow-tit.com/mcp
   (공모전 요건)        list_tools / call_tool 패스스루        (harness.db·키 보유)
```

도구 목록·스키마·설명·결과는 업스트림에서 그대로 가져와 미러링하므로, 업스트림에 도구를
추가/수정해도 이 프록시는 손댈 필요 없다.

## 등록 방법 (Git 소스 빌드)

PlayMCP in KC → "+ 새 MCP 서버 등록" → **Git 소스 빌드**:

- **Git URL**: 이 레포 주소
- **브랜치/ref**: `main`
- **Dockerfile 경로**: `playmcp-proxy/Dockerfile`
- **PAT**: 레포가 private 이면 입력(GitHub: Settings → Developer settings → Personal access tokens), public 이면 비움

Status 가 `Active` 가 되면 상세에서 **Endpoint URL** 을 복사해 PlayMCP 에 등록한다.

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MCP_UPSTREAM_URL` | `https://mcp.crow-tit.com/mcp` | 업스트림 MCP 엔드포인트 |
| `MCP_UPSTREAM_KEY` | (없음) | 업스트림을 `MCP_API_KEY` 로 잠갔을 때 Bearer 키. 현재 오픈 모드라 보통 불필요 |
| `PORT` / `HOST` | `8080` / `0.0.0.0` | listen 주소(KC 가 `PORT` 주입 시 그 값 사용) |

## 로컬 테스트

```bash
pip install "mcp>=1.9" uvicorn starlette httpx
python proxy_mcp.py            # http://localhost:8080/  (catch-all, MCP streamable HTTP)
curl localhost:8080/healthz    # {"status":"ok","upstream":"..."}
```
