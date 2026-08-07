"""죄명 정규화 — DB charge_norm + 평가 set 비교 + LLM 입력 모두 동일 form.

정규화가 통일하는 노이즈(같은 죄명의 표기 흔들림):
- 공백 차이: '야생생물 보호 및 관리에 관한 법률 위반' vs 붙임표기 — 공백 제거.
- Unicode dot variants: '아동·청소년...'(U+00B7), '아동․...'(U+2024), '아동ㆍ...'(U+318D)이
  같은 죄명인데 다른 row로 분산 (263건이 4-way split → sample 신뢰도 ↓).

⚠ **괄호 부속표시는 보존한다**(2026-07-09 수정). 특별법 죄명은 괄호 안이 곧 실체
죄명이라(폭력행위등처벌에관한법률위반(공동재물손괴등) ≠ (공동상해) ≠ (공동공갈) — 법정형·
양형이 전혀 다름) 괄호를 지우면 87종 subtype 이 한 버킷으로 뭉쳐 `sentence_statistics` 의
단독범 통계가 잡탕이 된다(공동재물손괴 통계에 상해·공갈이 94% 섞이는 결함). 괄호를 지우면
`prec_defendants.n_charges`(= 피고인별 DISTINCT charge_norm 수) 도 과소계상돼 다죄 피고인이
단독범 풀에 새는 경로가 열린다. 그래서 괄호를 살리고 punctuation/공백만 통일한다 —
결과적으로 `charge_key` 와 동일 form(아래).
"""
from __future__ import annotations

import re

_CHARGE_NORM_WS = re.compile(r"\s+")
# 한국 죄명에서 발견된 dot 변형 → middle dot(U+00B7)으로 통일
_DOT_VARIANTS = str.maketrans({"․": "·", "ㆍ": "·", "∙": "·"})


def norm_charge(s: str) -> str:
    """canonical form: 공백 제거 + dot variants 통일 (괄호 부속표시 **보존**).

    괄호 안 실체 죄명을 살리므로 특별법 subtype 이 서로 다른 버킷으로 유지된다
    (모듈 독스트링 참조). `charge_key` 와 동일 결과 — 두 이름은 호출부 의도 표기용.
    """
    if not s:
        return ""
    s = _CHARGE_NORM_WS.sub("", s)
    return s.translate(_DOT_VARIANTS)


def charge_key(s: str) -> str:
    """charge_legal_map lookup 키 — `norm_charge` 와 동일(공백 제거 + dot 통일 + 괄호 보존).

    매핑 테이블은 식별성 보존 필요(도로교통법위반(음주운전) ≠ (무면허운전) ≠ 무접미)라
    괄호를 살린다. norm_charge 가 2026-07-09 괄호 보존으로 수렴하며 form 이 같아졌다.
    """
    if not s:
        return ""
    s = _CHARGE_NORM_WS.sub("", s)
    return s.translate(_DOT_VARIANTS)
