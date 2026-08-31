# 로풀 MCP (lawful-mcp)

[English](README.md) | 한국어

한국 판례와 법령, 양형 데이터를 [MCP](https://modelcontextprotocol.io) 도구로
만들었습니다. AI 클라이언트에 붙여 두면 모델이 알아서 자료를 찾습니다. 특정 날짜에
시행 중이던 조문을 확인하고, 사실관계만 던져도 비슷한 판결을 찾아옵니다. 판결문을
읽고 물어본 것에 답하고, 양형 범위도 법원이 계산하는 순서 그대로 뽑아냅니다.

읽기만 하는 도구 다섯 개입니다.

| 도구 | 하는 일 |
|---|---|
| `precedent_search` | 사실관계·죄명·법원·연도·사건번호로 판결 찾기 |
| `precedent_dive` | 판결문 한 건을 읽고 물어본 것에 답하기 |
| `statute_lookup` | 법령·행정규칙 조문, 현행 또는 특정 시점 |
| `sentence_statistics` | 죄명별 1심 선고형이 실제로 어떻게 나왔는지 |
| `compute_sentencing_range` | 법정형 → 처단형 → 양형기준 권고형 → 검증 |

무료 법률AI 서비스 [로풀 (Lawful)](https://lawful.crow-tit.com)이 실제로 돌리고 있는
시스템입니다. 연구용 프로토타입 [`legal_mcp`](https://github.com/LimEulYoung/legal_mcp)에서
출발했고, 어떻게 만들었는지는 [논문](#논문)에 적었습니다.

## 빠른 시작 — 호스티드

판결 22만 건, 개정 이력이 붙은 법령 전체, 행정규칙, 양형기준을 전부 저희 서버에서
서비스합니다. 호출 횟수 제한은 없습니다. 붙는 길은 셋입니다.

**Claude·ChatGPT 웹 — 코드 없이.** 커스텀 커넥터에 아래 주소를 넣고 OAuth
로그인만 하면 도구 5종이 붙습니다. API 키도 필요 없습니다.

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

**그 밖의 MCP 클라이언트**(Claude Desktop, Cursor 등) — JSON 설정, 키는 같은 것:

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

샘플 코퍼스가 같이 들어 있어서 클론하면 바로 돌아갑니다.

```bash
git clone https://github.com/LimEulYoung/lawful-mcp
cd lawful-mcp
pip install -e .
lawful-mcp                 # stdio
lawful-mcp --transport http --port 8100
```

클라이언트에는 이렇게 등록합니다.

```json
{
  "mcpServers": {
    "lawful-mcp": {
      "command": "lawful-mcp"
    }
  }
}
```

설정은 전부 환경변수로 합니다. 목록은 [`.env.example`](.env.example)에 있습니다. 꼭
넣어야 하는 값은 없고, 아무것도 지정하지 않으면 같이 들어 있는 코퍼스를 씁니다.

### dive 도구에는 모델이 필요합니다

다섯 중 넷은 DB만 읽습니다. `precedent_dive` 하나만 공개된 판결문을 언어 모델에
넘겨 답을 뽑아내기 때문에 엔드포인트가 있어야 합니다.

```bash
export DIVE_API_KEY=...
export DIVE_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible endpoint
export DIVE_MODEL=...
```

세 값이 없으면 이 도구만 빠지고 나머지 넷은 그대로 돌아갑니다.

## 코퍼스

도구가 읽는 건 SQLite 파일 하나입니다. 저장소에 들어 있는 파일은 전체가 아니라
일부만 떼어 낸 샘플입니다.

| | 샘플 (`data/fixture.db`) | 호스티드 |
|---|---|---|
| 판결 | 800 | 220,000+ |
| 법령 | 주요 27개, 현행 조문 | 전체, 개정 이력까지 |
| 행정규칙 | 20 | 전체 |
| 양형기준 | 전량 | 전량 |
| 죄명 체계 | 전량 | 전량 |

샘플만으로도 도구를 전부 써 보고 테스트까지 돌릴 수 있습니다. 실제 작업에는 호스티드
코퍼스를 쓰시는 편이 낫습니다.

검색은 렉시컬 인덱스 두 개를 섞어 씁니다. 하나는 글자 단위 트라이그램 FTS, 다른
하나는 [Kiwi](https://github.com/bab2min/kiwipiepy)로 만든 형태소 FTS고, 둘의 순위를
RRF(reciprocal rank fusion)로 합칩니다. 형태소 인덱스는 속도 때문에 넣은 게 아닙니다.
죄명은 `사기`, `절도`, `폭행`처럼 두 글자짜리가 많은데, 두 글자에서는 트라이그램이
아예 만들어지지 않습니다. 형태소 인덱스가 없으면 이런 검색어는 결과가 하나도 안
나옵니다.

판결문은 저작권법 제7조에 따라 저작권 보호를 받지 않습니다. 데이터에는 법원,
사건번호 같은 출처 정보를 그대로 남겨 뒀습니다.

### 샘플 직접 만들기

`scripts/build_sample_db.py`는 전체 코퍼스에서 샘플을 잘라내는 스크립트이면서, 스키마
설명서 노릇도 합니다. 도구가 어떤 테이블을 읽는지, 전문 인덱스를 어떻게 만드는지,
법령 버전을 어떤 규칙으로 고르는지가 이 파일에 다 있습니다.

```bash
python scripts/build_sample_db.py --source /path/to/corpus.db \
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

MIT 라이선스입니다.
