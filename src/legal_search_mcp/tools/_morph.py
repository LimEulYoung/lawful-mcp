"""Kiwi 형태소 분석기 — 도구 공용 인스턴스 하나.

`precedent_search`(형태소 FTS)와 `statutes`(법령명 토큰화)가 둘 다 Kiwi 를 쓴다.
⚠ **프로세스당 하나여야 한다.** 종전에는 두 모듈이 각자 `@lru_cache` 로 감싼 자기
`_kiwi()` 를 들고 있었고, `lru_cache` 는 함수마다 따로라 **인스턴스가 둘 떴다**
(실측 2026-07-30: 두 번째가 RSS **+246MB** · 0.49s). 그리고 `lawful`·`lawful-mcp`
두 프로세스가 다 싣는다.

⚠ 종전 `precedent_search` 주석은 "Kiwi 는 statutes 도구에서도 lazy 로드돼 프로세스에
이미 떠 있어 추가 비용 거의 없음"이라고 **적어 두고 바로 아래에서 두 번째를 만들었다.**
전제를 적었으면 실물로 한 번은 밟아야 한다.

지연 import 다 — kiwipiepy 미설치 환경에서도 두 도구 모듈 자체는 import 되고, 토큰화
호출에서만 필요하다(호출부는 예외를 잡아 빈 토큰으로 폴백한다).
"""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def kiwi():
    """형태소 분석기 — **프로세스당 1회**(~0.5s, RSS +246MB)."""
    from kiwipiepy import Kiwi
    return Kiwi()
