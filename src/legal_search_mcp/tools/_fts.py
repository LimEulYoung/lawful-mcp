"""FTS5 query sanitize — 도구 공용 (chapter 8 §8.0 참조).

한국어(자모·완성형) + CJK 한자 + 영숫자 + 공백만 유지, 나머지 모두 공백 치환.
LLM이 query에 점·쉼표·괄호 등 punctuation을 넣어도 FTS5 syntax error 방지.

precedent_search.py·statutes.py 양쪽에서 import. 이전엔 precedent_search만 sanitize
적용했고 statute_lookup은 누락되어 v6 평가에서 syntax error 2건 발생.
"""
from __future__ import annotations

import re

# `\w`는 default Unicode mode에서 한글·한자·영숫자·언더스코어 매칭 (언더스코어는 FTS5에 무해).
_FTS_KEEP = re.compile(r'[^\w\s]', re.UNICODE)


def safe_fts_query(q: str) -> str:
    """FTS5 syntax 위반 char 공백 치환 + 다중 공백 정리."""
    cleaned = _FTS_KEEP.sub(" ", q)
    return " ".join(cleaned.split())
