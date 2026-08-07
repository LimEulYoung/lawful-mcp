"""도구 — RunContext[HarnessDeps] 시그니처.

정의된 코퍼스 도구 5종:
  compute_sentencing_range (결정적, 통합 양형 — 법정형·처단형·권고형·선고검증·집유)
  statute_lookup          (결정적 SQL, 법령·고시 본문)
  precedent_search        (FTS + charge LIKE + dense RRF + 2단계 rerank)
  precedent_dive          (선택한 공개 판례 본문을 sub-agent로 추출 요약)
  sentence_statistics     (동일 죄목 양형 분포 + stratified 예시)

웹 채팅과 MCP의 기본 runtime은 5종을 모두 등록한다.

폐지: `sentencing_lookup` (이전 fork) → `compute_sentencing_range` 통합 흡수.
sentencing.py 파일은 `_factors_for` 헬퍼 의존 (extract_factors_batch.py,
build_sentencing.py) 으로 *보존*, 도구로는 export 안 함.
"""
from .compute_sentencing_range import compute_sentencing_range
from .precedent_dive import precedent_dive
from .precedent_search import precedent_search
from .sentence_statistics import sentence_statistics
from .statutes import statute_lookup

__all__ = [
    "compute_sentencing_range",
    "statute_lookup",
    "precedent_search",
    "precedent_dive",
    "sentence_statistics",
]
