"""LLM이 도구 인자에 잘못된 type을 보내도 SQLite InterfaceError 안 나도록 coerce.

LLM이 `court_level=["대법원"]` 또는 `case_id="123"` 같이 type 위반 호출 빈번 — 도구
함수 진입부에서 한 번 정규화. None은 None 보존. *silent로 두지 않고* stderr에
경고 한 줄을 남겨 LLM 행동 이슈를 사후에 추적 가능하게 함 (chapter 8 §6.3 패턴).
"""
from __future__ import annotations

import json
import sys
from typing import Any


def _warn(arg: Any, target: str) -> None:
    print(
        f"[coerce] WARN: tool arg type-violated → {target} (got {type(arg).__name__}: {arg!r:.80s})",
        file=sys.stderr,
    )


# LLM이 null 의도로 보내는 가짜 string 표기. coerce 단계에서 None으로 정규화 —
# 실제 case ID 2025고단1373: sentencing_lookup({category:"None"}) 52회 반복 (52회
# missing → 같은 호출 → 무한 루프). 이 변환으로 mode 0 (48 카테고리 list) 진입.
_STR_NULL_TOKENS = frozenset({"none", "null", "nil", "undefined", "n/a", "na"})


def coerce_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in _STR_NULL_TOKENS:
            return None
        return s
    if isinstance(v, (list, tuple)):
        _warn(v, "str")
        return ", ".join(str(x) for x in v) if v else None
    _warn(v, "str")
    return str(v)


def coerce_int(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(v, float):
        _warn(v, "int")
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            _warn(v, "int")
            return int(s)
        _warn(v, "int")
        return None
    if isinstance(v, (list, tuple)) and v:
        _warn(v, "int")
        return coerce_int(v[0])
    _warn(v, "int")
    return None


def coerce_list(v: Any) -> list | None:
    """LLM이 배열 인자를 JSON 문자열('["15"]' / '[{...}]')로 이중 인코딩해 보내는 경우(MiMo 등)
    방어. 도구 함수 타입을 str 도 받도록 넓힌 뒤 진입부에서 이걸로 정규화한다.
    list/tuple → list · JSON 배열 문자열 → 파싱한 list · 그 외 스칼라/문자열 → 단일 원소 list ·
    None·빈·null토큰 → None."""
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in _STR_NULL_TOKENS:
            return None
        if s[0] == "[":
            try:
                parsed = json.loads(s)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                _warn(v, "list")
                return parsed
        _warn(v, "list")
        return [s]
    _warn(v, "list")
    return [v]


def coerce_dict(v: Any) -> dict | None:
    """dict 인자 방어 — LLM이 dict를 JSON 문자열('{...}')로 이중 인코딩하거나 엉뚱한
    스칼라/리스트로 보내는 경우(MiMo 등). dict → 그대로 · JSON 오브젝트 문자열 → 파싱한 dict ·
    그 외(리스트·스칼라·파싱실패) → None(경고). None·빈·null토큰 → None.
    downstream 이 `.get(...)` 로 소비하는 dict 인자(guideline_factors·probation_factors)에
    비-dict 이 들어와 AttributeError 나는 것을 차단."""
    if v is None:
        return None
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in _STR_NULL_TOKENS:
            return None
        if s[0] == "{":
            try:
                parsed = json.loads(s)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                _warn(v, "dict")
                return parsed
        _warn(v, "dict")
        return None
    _warn(v, "dict")
    return None


def coerce_dict_list(v: Any) -> list[dict] | None:
    """dict 리스트 인자 방어(statutory_modifications 등) — coerce_list 로 리스트화한 뒤
    dict 원소만 남긴다. LLM이 배열 원소를 스칼라로 보내면(예: '["48"]' → ["48"])
    그 원소를 버려 downstream 의 `elem.get(...)` AttributeError 를 원천 차단한다.
    비-dict 원소는 경고 후 제외. 남는 dict 가 없으면(빈·전부 garbage·None) None 을
    반환 — 호출측이 '미지정'으로 보고 상위 stage(lookup)에 머물러 LLM 이 enum 안내를
    다시 받고 self-correct 한다(빈 리스트로 처단형 stage 를 조용히 진입시키지 않음)."""
    items = coerce_list(v)
    if not items:
        return None
    out = [x for x in items if isinstance(x, dict)]
    dropped = len(items) - len(out)
    if dropped:
        _warn(v, f"list[dict] ({dropped}개 비-dict 원소 제외)")
    return out or None


def to_iso_date(s: str | None) -> str | None:
    """'2013-7-30' / '2013.7.30' / '20130730' → '20130730'. 잘못된 입력은 None.

    LLM 이 준 날짜 문자열을 도구가 쓰는 한 형태로 만드는 자리이므로 이 모듈이 갖는다.
    ⚠ 종전에는 `statutes` 와 `compute_sentencing_range` 가 **각자 같은 함수**를 갖고 있었고
    (docstring 한 줄만 달랐다) 둘 다 `offense_date` 를 이것으로 정규화했다(2026-07-30 합침).
    """
    if not s:
        return None
    s = str(s).strip().replace('-', '.').replace('/', '.')
    if '.' in s:
        parts = s.split('.')
        if len(parts) != 3:
            return None
        try:
            y, m, d = (int(p) for p in parts)
        except ValueError:
            return None
        return f'{y:04d}{m:02d}{d:02d}'
    digits = ''.join(ch for ch in s if ch.isdigit())
    if len(digits) == 8:
        return digits
    return None
