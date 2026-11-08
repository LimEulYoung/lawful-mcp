"""Coerce tool arguments the LLM sent with the wrong type, so they never cause a SQLite InterfaceError.

LLMs often make type-violating calls such as `court_level=["대법원"]` or `case_id="123"` — normalize
once at tool-function entry. None stays None. Each coercion writes one warning line to stderr.
"""
from __future__ import annotations

import json
import sys
from typing import Any


def _warn(arg: Any, target: str) -> None:
    print(f"[coerce] WARN: tool arg type-violated → {target} (got {type(arg).__name__}: {arg!r:.80s})",
          file=sys.stderr)


# Strings an LLM sends when it means null ('None', 'null' …) read as None, not as a value to look up.
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
        # An array wrapped in a JSON string ('[2]') reads as the array [2] (some models send arrays this way).
        if s.startswith("["):
            try:
                parsed = json.loads(s)
            except (ValueError, RecursionError):
                parsed = None
            if isinstance(parsed, list):
                _warn(v, "int")
                return coerce_int(parsed[0]) if parsed else None
        _warn(v, "int")
        return None
    if isinstance(v, (list, tuple)) and v:
        _warn(v, "int")
        return coerce_int(v[0])
    _warn(v, "int")
    return None


def coerce_list(v: Any) -> list | None:
    """Guard against the LLM double-encoding an array argument as a JSON string ('["15"]' / '[{...}]') or sending
    a scalar where an array goes. Tool functions accept a wide type and normalize with this at entry.
    list/tuple → list · JSON array string → parsed list · any other scalar/string → one-element list ·
    None/empty/null token → None."""
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
            # An unquoted array such as '[9의2, 10, 16]' is not JSON: split it on commas.
            inner = s.strip("[] ")
            if "," in inner and "{" not in inner:
                _warn(v, "list")
                return [t.strip().strip("'\"") for t in inner.split(",") if t.strip()]
        _warn(v, "list")
        return [s]
    _warn(v, "list")
    return [v]


def to_iso_date(s: str | None) -> str | None:
    """'2013-7-30' / '2013.7.30' / '2013. 7. 30.' / '2013년 7월 30일' / '20130730' → '20130730'. Invalid input → None.
    The one place an LLM-supplied date becomes the form the tools use."""
    if not s:
        return None
    s = str(s).strip().replace('년', '.').replace('월', '.').replace('일', '')
    s = s.replace('-', '.').replace('/', '.').rstrip('. ')
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


BAD_DATE_MESSAGE = "offense_date 를 날짜로 읽지 못했습니다 — '2013.7.30' 또는 '20130730' 형식으로 다시 호출하세요."


def unreadable_date(v: Any) -> bool:
    """True when a date argument was given but `to_iso_date` cannot read it (the tools then say so instead of
    answering for today)."""
    s = coerce_str(v)
    return bool(s) and to_iso_date(s) is None
