# Legal Search MCP

[English](README.md) | 한국어

한국 판례와 법령, 양형 데이터를 [MCP](https://modelcontextprotocol.io) 도구로
제공합니다. AI 클라이언트를 연결해 판례를 찾고, 특정 날짜의 조문을 확인하고,
판결문을 요약하고, 법정형과 양형기준 권고형 범위를 계산할 수 있습니다.

읽기만 하는 도구 네 개입니다.

| 도구 | 하는 일 |
|---|---|
| `precedent_search` | 사실관계·죄명·법원·연도·사건번호로 판결 찾기 |
| `precedent_dive` | 판결문 한 건을 읽고 물어본 것에 답하기 |
| `statute_lookup` | 법령·행정규칙 조문, 현행 또는 특정 시점 |
| `sentencing_analysis` | 죄명별 법정형·양형기준·인자·실선고 분포 조회, 입력한 인자로 형량 범위 계산 |

무료 법률AI 서비스 [로풀 (Lawful)](https://lawful.crow-tit.com)에서 사용하는 MCP
도구를 공개한 저장소입니다. 웹앱과 호스티드 Agent API는 별도로 관리합니다.
연구용 프로토타입과 벤치마크는
[`legal_mcp`](https://github.com/LimEulYoung/legal_mcp)에 보관합니다([논문](#논문)).

## 빠른 시작 — 호스티드

법령 개정 이력을 포함한 전체 코퍼스를 호스티드 서비스에서 제공합니다.
호출 횟수 제한은 없으며, 연결 방법은 세 가지입니다.

**Claude·ChatGPT 웹 — 코드 없이.** 커스텀 커넥터에 아래 주소를 넣고 OAuth
로그인만 하면 도구 4종이 붙습니다. API 키도 필요 없습니다.

```
https://mcp.crow-tit.com/mcp
```

메뉴 클릭 순서까지 적은 단계별 안내: [crow-tit.com/docs#mcp](https://crow-tit.com/docs#mcp)

**Claude Code** — 한 줄이면 됩니다. 키는
[console.crow-tit.com](https://console.crow-tit.com)에서 무료 발급:

```bash
claude mcp add --transport http lawful https://mcp.crow-tit.com/mcp \
  --header "Authorization: Bearer ct_..."
```

**HTTP MCP 설정을 지원하는 클라이언트** — 같은 키를 사용합니다. 설정 형식은
클라이언트마다 다르며, `mcpServers`의 HTTP 항목을 받는 경우 아래처럼 설정합니다.

```json
{
  "mcpServers": {
    "lawful": {
      "url": "https://mcp.crow-tit.com/mcp",
      "headers": { "Authorization": "Bearer ct_..." }
    }
  }
}
```

## 빠른 시작 — 직접 띄우기

**Python 3.11 이상**이 필요합니다. 가상환경에 설치하면 API 키 없이 동봉된 샘플로
실행할 수 있습니다.

```bash
git clone https://github.com/LimEulYoung/lawful-mcp
cd lawful-mcp
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

로컬 stdio 클라이언트에는 아래 경로를 클론한 디렉터리의 절대 경로로 바꿔 등록합니다.
서버는 클라이언트가 실행합니다.

```json
{
  "mcpServers": {
    "lawful-mcp": {
      "command": "/absolute/path/to/lawful-mcp/.venv/bin/lawful-mcp"
    }
  }
}
```

HTTP로 연결하려면 다음 명령으로 실행합니다. 접속 주소는 `http://127.0.0.1:8100/mcp`입니다.

```bash
.venv/bin/lawful-mcp --transport http --port 8100
```

직접 실행하는 서버에는 인증과 사용량 계측이 없습니다. HTTP 주소를 공유하는 경우
접근 제어는 배포 환경에서 구성합니다.

### 설정과 데이터 흐름

설정 목록은 [`.env.example`](.env.example)에 있습니다. 서버는 프로세스의 환경변수를
읽으며 **`.env` 파일을 자동으로 읽지 않습니다.** 실행하는 셸에서 값을 export하거나
MCP 클라이언트의 `env` 설정으로 전달하세요.

기본 설정에서는 도구 세 개가 로컬에서 검색·계산합니다. `precedent_dive`를 추가하면
선택한 공개 판결문과 사건 메타데이터, 입력한 질문을 설정한 모델 엔드포인트로 보냅니다.
질문에는 해당 공개 판례에서 확인할 쟁점만 넣으세요.

```bash
export DIVE_API_KEY=...
export DIVE_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible endpoint
export DIVE_MODEL=...
```

세 값을 모두 설정해야 `precedent_dive`가 등록됩니다. 설정하지 않아도 나머지 셋은
동작합니다. 설정을 바꾼 뒤에는 서버를 다시 시작하세요.

선택 기능인 dense 검색(`USE_DENSE=1`)은 검색어를 임베딩 엔드포인트로 보내며,
재정렬을 켜면 검색어와 후보 판례 발췌를 재정렬 엔드포인트에도 보냅니다.
`dense` 추가 의존성과 코퍼스 벡터가 필요하고, 동봉 샘플에는 벡터가 없습니다.

### 설치 확인

테스트는 동봉 샘플과 별도의 재현 데이터를 사용하므로 네트워크나 모델 키가 필요 없습니다.

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```

## 코퍼스

도구는 SQLite 파일 하나를 읽습니다. 동봉 샘플의 크기는 약 48MiB입니다.

| | 샘플 (`data/fixture.db`) | 호스티드 |
|---|---|---|
| 판결 | 700 | 220,000+ |
| 법령 | 주요 27개, 현행 조문 | 전체, 개정 이력까지 |
| 행정규칙 | 20 | 전체 |
| 양형기준 | 전량 | 전량 |
| 죄명 체계 | 전량 | 전량 |

샘플은 로컬 체험과 테스트에 사용할 수 있습니다. 전체 코퍼스는 호스티드 서비스에서
이용하거나, 직접 관리하는 호환 DB를 `CORPUS_DB`로 지정하세요.

법령 조회는 코퍼스에 해당 정보가 있으면 폐지와 유효기간 만료 등의 효력 상실을 구분합니다.
현재 샘플의 효력 상실 명부는 비어 있으며, 만료와 재시행 판정은 별도의 테스트 데이터로
검증합니다.

기본 검색은 트라이그램과 [Kiwi](https://github.com/bab2min/kiwipiepy) 형태소 인덱스의
순위를 RRF(reciprocal rank fusion)로 합칩니다. 형태소 인덱스 덕분에 `사기`, `절도`,
`폭행`처럼 두 글자인 죄명도 검색됩니다.

판결문은 저작권법 제7조에 따라 저작권 보호를 받지 않습니다. 데이터에는 법원,
사건번호 같은 출처 정보를 그대로 남겨 뒀습니다.

### 샘플 직접 만들기

[`scripts/build_sample_db.py`](scripts/build_sample_db.py)는 기존 호환 코퍼스에서
샘플을 추출하고 검색 인덱스를 재생성합니다. 스키마 레퍼런스이기도 하며,
호스티드 코퍼스를 다운로드하는 도구는 아닙니다.

```bash
.venv/bin/python scripts/build_sample_db.py --source /path/to/corpus.db \
    --dest my_sample.db --cases 5000 --statutes all
```

## 논문

검색 방식과 도구 설계가 실제로 통하는지는 변호사시험 문제로 검증했습니다.

> **Agentic RAG for Legal Question Answering in Civil Law: Evidence From the Korean Bar Examination**
> Eul Young Lim and Jihun Park
> *IEEE Access*, vol. 14, pp. 124441–124458, 2026.
> [doi:10.1109/ACCESS.2026.3722717](https://doi.org/10.1109/ACCESS.2026.3722717) — 오픈 액세스

벤치마크 코드와 문항, 모델별 결과는
[`legal_mcp`](https://github.com/LimEulYoung/legal_mcp)에 정리해 뒀습니다. 연구에
쓰신다면 논문을 인용해 주세요([`CITATION.cff`](CITATION.cff)).

## 이런 것도 있습니다

- **로풀 Agent API** — 질문 하나 넣으면 근거가 달린 답이 돌아옵니다.
  [crow-tit.com](https://crow-tit.com)
- **로풀 (Lawful)** — 누구나 무료로 쓰는 법률AI입니다. 챗, 검색, 노무 계산기, 법률
  문서 작성. [lawful.crow-tit.com](https://lawful.crow-tit.com)

## 알아 두실 점

공개된 법률 자료를 찾아 주는 검색 도구이지 법률 자문이 아닙니다. 돌려드리는 건
원자료와 통계고, 그게 특정 사건에서 어떤 의미인지 따지는 일은 변호사 몫입니다.

개발은 비공개 저장소에서 하고 여기에는 한 번에 묶어서 반영합니다. 이슈는 언제든 남겨
주세요. 다만 PR은 버튼으로 머지하지 않고 손으로 반영하는 경우가 많습니다.

보안 취약점은 [SECURITY.md](SECURITY.md)의 비공개 제보 경로로 알려 주세요.
코드는 [MIT 라이선스](LICENSE)입니다.
