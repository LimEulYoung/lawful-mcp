"""Normalise tool arguments a model got the type wrong on.

Models routinely send a list where a string belongs (``court_level=["대법원"]``)
or a string where an integer belongs (``case_id="123"``). Left alone these
surface as an opaque SQLite ``InterfaceError`` deep in a query, so each tool
normalises its arguments once on entry. ``None`` stays ``None``.

Coercion is not silent: every conversion writes one line to stderr, so a
model that keeps getting a type wrong is visible in the logs rather than
hidden behind a result that happens to work.
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


# Strings a model sends when it means null. Without this mapping the literal
# "None" is a value: the tool searches for it, finds nothing, and the model
# repeats the same call — observed looping 52 times. Mapping them to None
# instead lets the tool take its no-argument path and answer usefully.
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
    """Normalise a list argument, including one double-encoded as JSON.

    Some models send ``'["15"]'`` — a JSON array inside a string — so tools
    widen the parameter type to accept a string and normalise here.
    list/tuple stays a list, a JSON array string is parsed, any other scalar
    becomes a one-element list, and empty or null-ish input becomes None.
    """
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
    """Normalise a dict argument, including one double-encoded as JSON.

    A dict passes through, a JSON object string is parsed, and anything else
    becomes None with a warning. Callers consume these with ``.get(...)``
    (``guideline_factors``, ``probation_factors``), so letting a non-dict
    through would raise AttributeError further down.
    """
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
    """Normalise a list-of-dicts argument, dropping elements that are not dicts.

    When a model sends scalars where objects belong (``'["48"]'``), those
    elements are discarded with a warning rather than reaching a caller that
    would do ``elem.get(...)`` on them.

    An empty result returns None, not an empty list. The difference matters:
    None reads as "not supplied", so the caller stays at the earlier stage and
    re-offers the enum for the model to correct itself, whereas an empty list
    would silently advance a stage with no selections made.
    """
    items = coerce_list(v)
    if not items:
        return None
    out = [x for x in items if isinstance(x, dict)]
    dropped = len(items) - len(out)
    if dropped:
        _warn(v, f"list[dict] ({dropped} non-dict element(s) dropped)")
    return out or None


def to_iso_date(s: str | None) -> str | None:
    """'2013-7-30' / '2013.7.30' / '20130730' -> '20130730'; None if unparseable.

    Both ``statute_lookup`` and ``compute_sentencing_range`` normalise their
    ``offense_date`` through this one function, so a date means the same thing
    to both.
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
