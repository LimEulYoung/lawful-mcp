# Legal Search MCP

[English](README.md) | 한국어

한국 판례·법령·양형 데이터를 [MCP](https://modelcontextprotocol.io) 도구로 엽니다.
AI 클라이언트를 붙여 두면 모델이 특정 시점의 법령 조문을 찾고, 사건의 사실관계로
판결을 검색하고, 판결문 한 건을 읽어 질문에 답하고, 법원이 하는 방식대로 양형
범위를 계산할 수 있습니다.

조회 전용 도구 다섯 가지입니다.

| 도구 | 하는 일 |
|---|---|
| `precedent_search` | 사실관계·죄명·법원·연도·사건번호로 판결 검색 |
| `precedent_dive` | 판결문 한 건을 읽고 그에 대한 질문에 답변 |
| `statute_lookup` | 법령·행정규칙 조문, 현행 또는 특정 시점 기준 |
| `sentence_statistics` | 죄명별 1심 선고형 분포(실측) |
| `compute_sentencing_range` | 법정형 → 처단형 → 양형기준 권고형 → 검증 |

무료 법률AI 서비스 [로풀 (Lawful)](https://lawful.crow-tit.com)을 움직이는 운영
시스템이며, 연구 프로토타입 [`legal_mcp`](https://github.com/LimEulYoung/legal_mcp)의
후속입니다 — [논문](#논문)을 보세요.

## 바로 쓰기 — 호스티드

전체 코퍼스(판결 22만 건 이상, 개정 이력이 붙은 법령 전체, 행정규칙, 양형기준)는
저희가 서빙합니다. [console.crow-tit.com](https://console.crow-tit.com)에서 무료
키를 받아 아래를 추가하세요.

```json
{
  "mcpServers": {
    "legal-search": {
      "type": "http",
      "url": "https://mcp.crow-tit.com/mcp",
      "headers": { "Authorization": "Bearer YOUR_KEY" }
    }
  }
}
```

Claude Desktop, Claude Code, Cursor를 비롯한 모든 MCP 클라이언트에서 동작합니다.
호출 제한은 없습니다.

## 바로 쓰기 — 직접 호스팅

샘플 코퍼스가 함께 들어 있어 클론하면 그대로 실행됩니다.

```bash
git clone https://github.com/LimEulYoung/legal-search-mcp
cd legal-search-mcp
pip install -e .
legal-search-mcp                 # stdio
legal-search-mcp --transport http --port 8100
```

클라이언트에 등록합니다.

```json
{
  "mcpServers": {
    "legal-search": {
      "command": "legal-search-mcp"
    }
  }
}
```

설정은 전부 환경변수입니다 — [`.env.example`](.env.example)을 보세요. 필수 항목은
없으며, 기본값은 동봉된 코퍼스를 씁니다.

### dive 도구에는 모델이 필요합니다

다섯 중 넷은 순수한 DB 조회입니다. `precedent_dive`는 공개된 판결문 한 건을 언어
모델에 보내 답을 뽑아내므로 엔드포인트가 필요합니다.

```bash
export DIVE_API_KEY=...
export DIVE_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible endpoint
export DIVE_MODEL=...
```

셋이 없으면 이 도구만 등록되지 않고 나머지 넷은 평소대로 동작합니다.

## 코퍼스

도구는 SQLite 파일 하나를 읽습니다. 이 저장소에 담긴 것은 전체가 아니라 한정된
샘플입니다.

| | 샘플 (`data/fixture.db`) | 호스티드 |
|---|---|---|
| 판결 | 800 | 220,000+ |
| 법령 | 주요 27개, 현행 조문 | 전체, 개정 이력 포함 |
| 행정규칙 | 20 | 전체 |
| 양형기준 | 전량 | 전량 |
| 죄명 체계 | 전량 | 전량 |

샘플만으로도 모든 도구를 돌려 보고 테스트를 통과시킬 수 있습니다. 실제 작업에는
호스티드 코퍼스를 쓰세요.

검색은 렉시컬 인덱스 둘을 씁니다. 문자 트라이그램 FTS와
[Kiwi](https://github.com/bab2min/kiwipiepy)로 만든 형태소 FTS를 RRF(reciprocal
rank fusion)로 합칩니다. 형태소 인덱스는 최적화가 아닙니다 — 한국어 죄명은 `사기`,
`절도`, `폭행`처럼 두 글자인 경우가 많고, 두 글자에서는 트라이그램이 만들어지지
않아 형태소 인덱스 없이는 그런 질의가 아무것도 반환하지 못합니다.

판결문은 저작권법 제7조의 비보호 저작물입니다. 데이터에는 법원·사건번호 같은 출처
식별자를 보존해 두었습니다.

### 직접 만들기

`scripts/build_sample_db.py`는 전체 코퍼스에서 샘플을 잘라 내며, 동시에 스키마
레퍼런스이기도 합니다 — 도구가 어떤 테이블을 읽는지, 전문 인덱스를 어떻게 만드는지,
법령 버전이 어떻게 해석되는지가 여기 있습니다.

```bash
python scripts/build_sample_db.py --source /path/to/corpus.db \
    --dest my_sample.db --cases 5000 --statutes all
```

## 논문

검색과 도구 사용 설계를 변호사시험으로 평가했습니다.

> **Agentic RAG for Legal Question Answering in Civil Law: Evidence From the Korean Bar Examination**
> Eul Young Lim and Jihun Park
> *IEEE Access*, vol. 14, pp. 124441–124458, 2026.
> [doi:10.1109/ACCESS.2026.3722717](https://doi.org/10.1109/ACCESS.2026.3722717) — 오픈 액세스

벤치마크 코드·문항·모델별 결과는
[`legal_mcp`](https://github.com/LimEulYoung/legal_mcp)에 아카이브되어 있습니다.
연구에 이 도구를 쓰신다면 논문을 인용해 주세요([`CITATION.cff`](CITATION.cff)).

## 함께 제공되는 것

- **Legal Search API** — 질문 하나를 넣으면 근거가 붙은 답이 나옵니다. Anthropic
  Messages 형식입니다. [crow-tit.com](https://crow-tit.com)
- **로풀 (Lawful)** — 무료 소비자 서비스, 법률AI: 챗, 검색, 노무 계산기, 법률 문서
  작성. [lawful.crow-tit.com](https://lawful.crow-tit.com)

## 알아 두실 것

이 도구는 공개된 법률 자료를 찾아 주는 검색 도구이지 법률 자문이 아닙니다. 돌려주는
것은 원자료와 통계이고, 그것이 특정 사안에서 무엇을 뜻하는지 판단하는 일은 변호사의
몫입니다.

개발은 비공개 저장소에서 이뤄지고 이곳에는 묶음으로 반영됩니다. 이슈는 환영하지만
PR은 버튼이 아니라 손으로 병합될 수 있습니다.

MIT 라이선스입니다.
