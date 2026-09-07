"""Resolve charge name strings to official charge names for sentencing tools.

Resolves charge input to entries from the official charge schedule (Supreme
Prosecutors' Office Directive No. 1516, Appendices 1–5), loaded in `official_charges`.
Judgments and indictments use this official nomenclature (~90% of judgment charge strings match exactly).

Naming rules from the directive:
  - Criminal Code offences have unparenthesized bare names (e.g. 살인, 사기, 특수상해).
    Attempt, preparation, and conspiracy are offences only if explicitly listed.
  - Appendices 3, 4, and 5 special acts (25 acts including Act on the Aggravated
    Punishment of Specific Crimes, Act on the Protection of Children and Juveniles
    against Sexual Offenses, Road Traffic Act, etc.) follow the form:
    ``<StatuteName>위반(<SubstantiveOffence>)``.
  - All other statutes use the single bare form ``<StatuteName>위반`` without parentheses.
  - Solicitation and aiding/abetting (교사·방조) append to the base charge (e.g. 사기방조).

Resolution order (compared after stripping dots, commas, and whitespace):
  1. Exact match against official charge names (including historical designations).
  2. Alias mapping (`charge_aliases`: colloquial terms, old terms, typos, omitted parentheses).
  3. Bare parenthesized forms for Criminal Code / Military Criminal Code:
     ``형법위반(X)`` / ``군형법위반(X)`` -> resolved as ``X``.
  4. Accomplice suffixes (교사, 방조) -> derived from valid parent charge.
     Attempt suffix (미수) -> resolved to parent for special acts (attempt is not separately listed).
  5. Stripping noise: characters after closing parenthesis, tail noise words (혐의, 사건, 범죄 등),
     leading '구' for repealed statute prefixes.
  6. Statute parsing: resolve abbreviations, former names, and concatenated ``<Law><Offence>`` forms.
  7. Parenthesis-only or prefix queries: returned as candidates (`statute_exact`).
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, replace
from typing import Callable, TypeVar

from .._charge import norm_charge
from .statutes import _STATUTE_ALIASES, _strip_dots

# Special-act charge pattern: <statute>위반(<substantive>). Bare codes (Criminal Code) are handled separately.
_SPLIT_RE = re.compile(r"^(.+?)위반(?:\((.+)\))?$")
# Bare code pattern: Criminal Code / Military Criminal Code wrapped in 위반(X).
_BARE_PAREN_RE = re.compile(r"^(?:형법|군형법)위반\((.+?)\)(.*)$")
_BARE_CODES = ("형법", "군형법")
_COMMA_CHARS = (",", "，", "、")
_SUFFIX_MODS = ("미수", "교사", "방조")
# Tail words commonly appended to charge names by users/models.
_TAIL_NOISE = ("피의사건", "공소사실", "혐의", "사건", "관련", "부분", "범죄", "으로", "죄", "건")
_PREFIX_CAND_MIN = 3
_ACCOMPLICE = ("교사", "방조")
_MAX_DEPTH = 4

# Specific acts split where punishment provisions transferred to a distinct successor act.
_SPLIT_LAW_ALIASES = {
    "성폭력범죄의처벌및피해자보호등에관한법률": "성폭력범죄의처벌등에관한특례법",
    "해사안전법": "해상교통안전법",
}


def norm_cmp(s: str | None) -> str:
    """Normalize for string comparison: strip whitespace, unified middle dots, commas, and full-width parens."""
    out = _strip_dots(norm_charge(s or "")).replace("（", "(").replace("）", ")")
    for c in _COMMA_CHARS:
        out = out.replace(c, "")
    return out


def split_charge(q: str) -> tuple[str | None, str | None]:
    """Split ``<law>위반(<substantive>)`` into (law, substantive). Returns (None, None) if not matching."""
    m = _SPLIT_RE.match(q)
    return (m.group(1), m.group(2)) if m else (None, None)


# Leading bracketed numbers or date tags (e.g., "[1]상해", "[2024.07] 식품위생법위반").
_BRACKET_NUM_PREFIX_RE = re.compile(r"^\[\d{1,4}(?:[.\-/]\d{1,2}){0,2}\]")
# Wrapping quote pairs.
_WRAP_QUOTE_PAIRS = {
    '"': '"', "'": "'", "`": "`", "＂": "＂", "＇": "＇",
    "“": "”", "‘": "’", "「": "」", "『": "』",
}
# Numeric charge tokens (article numbers, IDs).
_NUMERIC_CHARGE_TOKEN_RE = re.compile(r"^\d{1,5}(?:(?:의|-)\d{1,3})?$")
_NUMERIC_CHARGE_SEP_RE = re.compile(r"[\s\[\]()'\"‚,，·;/]+")
# Article reference patterns like "제347조".
ARTICLE_REF_RE = re.compile(r"제\s*(\d{1,4})\s*조")


def strip_charge_decorations(raw: str) -> tuple[str, str | None, str | None]:
    """Strip leading bracket tags and matching outer quotes if remainder is non-empty."""
    bracket_prefix = None
    quotes_stripped = None
    for _ in range(4):
        m = _BRACKET_NUM_PREFIX_RE.match(raw)
        if m and m.end() < len(raw):
            bracket_prefix = (bracket_prefix or "") + m.group(0)
            raw = raw[m.end():]
            continue
        closer = _WRAP_QUOTE_PAIRS.get(raw[:1])
        if closer and len(raw) > 2 and raw.endswith(closer):
            quotes_stripped = (quotes_stripped or "") + raw[0] + closer
            raw = raw[1:-1]
            continue
        break
    return raw, bracket_prefix, quotes_stripped


def numeric_charge_tokens(charge: str) -> list[str] | None:
    """Return numeric tokens if the input consists purely of article/ID numbers, else None."""
    parts = [p for p in _NUMERIC_CHARGE_SEP_RE.split(charge) if p]
    if not parts or not all(_NUMERIC_CHARGE_TOKEN_RE.match(p) for p in parts):
        return None
    return list(dict.fromkeys(parts))[:8]


class LawNames:
    """Law name dictionary built from `st_statutes` (kind='법률')."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.display: dict[str, str] = {}
        self.current: set[str] = set()
        self.repealed: set[str] = set()
        self.alias: dict[str, str] = {}
        try:
            rows = conn.execute(
                "SELECT law_id, name, short_name, history_status, effective_date, change_kind "
                "FROM st_statutes WHERE kind='법률'"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        today = _dt.date.today().strftime("%Y%m%d")
        latest: dict[str, tuple[str, str, str, bool]] = {}
        earliest: dict[str, tuple[str, str, str, bool]] = {}
        names_by_id: dict[str, set[str]] = {}
        for law_id, name, short, status, eff, kind in rows:
            n = norm_cmp(name)
            if not n:
                continue
            self.display.setdefault(n, name)
            names_by_id.setdefault(law_id, set()).add(n)
            e = str(eff or "")
            rec = (e, n, norm_cmp(short), bool(kind and "폐지" in str(kind) and "제정" not in str(kind)))
            if earliest.get(law_id) is None or e < earliest[law_id][0]:
                earliest[law_id] = rec
            if status == "시행예정" or e > today:
                continue
            if latest.get(law_id) is None or e >= latest[law_id][0]:
                latest[law_id] = rec
        for law_id, rec in earliest.items():
            latest.setdefault(law_id, rec)
        self.current = {n for _e, n, _s, _r in latest.values()}
        self.repealed = {n for _e, n, _s, r in latest.values() if r} - {n for _e, n, _s, r in latest.values() if not r}
        ordered = sorted(latest.items(), key=lambda kv: (kv[1][3], -int(kv[1][0] or 0)))
        for law_id, (_e, n, short, _rep) in ordered:
            for old in names_by_id.get(law_id, ()):
                if old != n and old not in self.current:
                    self.alias.setdefault(old, n)
            if short and short != n and short not in self.current:
                self.alias.setdefault(short, n)
        for a, v in _STATUTE_ALIASES.items():
            self.alias.setdefault(norm_cmp(a), norm_cmp(v))
        for a, v in _SPLIT_LAW_ALIASES.items():
            self.alias[norm_cmp(a)] = norm_cmp(v)

    def is_law(self, n: str) -> bool:
        return n in self.display

    def is_repealed(self, n: str) -> bool:
        return n in self.repealed

    def show(self, n: str) -> str:
        return self.display.get(n) or n


@dataclass(frozen=True)
class Entry:
    name: str
    law: str | None
    article: str | None
    hang: int | None
    source: str
    parent: str | None
    status: str
    successor: str | None


def _entry_split(e: Entry) -> tuple[str | None, str | None]:
    if not e.law or norm_cmp(e.law) in _BARE_CODES:
        return None, None
    return split_charge(norm_cmp(e.name))


@dataclass(frozen=True)
class Resolution:
    """Resolution result. kind: exact | statute_exact | none."""

    kind: str
    key: str | None = None
    candidates: tuple[str, ...] = ()
    alias_from: tuple[str, str] | None = None
    statute: str | None = None
    sub: str | None = None
    law_repealed: bool = False
    suffix_stripped: str | None = None
    derived: tuple[str, str] | None = None
    status: str = "현행"
    successor: str | None = None


class OfficialCharges:
    """Charge resolution index backed by official_charges, charge_aliases, and st_statutes."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.laws = law_names(conn)
        self.entries: dict[str, Entry] = {}
        self.by_cmp: dict[str, str] = {}
        self.by_law: dict[str, list[str]] = {}
        self.law_display: dict[str, str] = {}
        self.by_sub: dict[str, list[str]] = {}
        self.children: dict[str, list[str]] = {}
        self.alias: dict[str, tuple[str, str]] = {}
        try:
            rows = conn.execute(
                "SELECT name, law, article, hang, source, parent, status, successor FROM official_charges "
                "ORDER BY CASE status WHEN '구법' THEN 1 ELSE 0 END, rowid").fetchall()
        except sqlite3.OperationalError:
            rows = []
        for name, law, article, hang, source, parent, status, successor in rows:
            e = Entry(name, law, article, hang, source, parent, status or "현행", successor)
            self.entries[name] = e
            n = norm_cmp(name)
            self.by_cmp.setdefault(n, name)
            st, sub = _entry_split(e)
            if st:
                self.by_law.setdefault(st, []).append(name)
                self.law_display.setdefault(st, split_charge(norm_charge(name))[0] or st)
                if sub:
                    self.by_sub.setdefault(sub, []).append(name)
            if parent:
                self.children.setdefault(parent, []).append(name)
        self.special_laws = {st for st, names in self.by_law.items() if any("(" in x for x in names)}
        try:
            arows = conn.execute("SELECT alias, official_name, how FROM charge_aliases").fetchall()
        except sqlite3.OperationalError:
            arows = []
        for a, target, how in arows:
            self.alias.setdefault(norm_cmp(a), (target, how or ""))
        self.available = bool(rows)

    def resolve(self, charge: str) -> Resolution:
        q = norm_cmp(charge)
        if not q:
            return Resolution("none")
        return self._resolve(q, None, 0)

    def resolve_exact(self, charge: str) -> Resolution | None:
        r = self.resolve(charge)
        return r if r.kind == "exact" else None

    def _exact(self, name: str, alias_from: tuple[str, str] | None, **extra) -> Resolution:
        e = self.entries.get(name)
        st, sub = _entry_split(e) if e else split_charge(norm_cmp(name))
        statute = (self.law_display.get(st) or (e.law if e and e.law else st)) if st else None
        cands: tuple[str, ...] = ()
        if st and not sub and st in self.special_laws:
            cands = tuple(x for x in self.by_law[st] if x != name)
        return Resolution("exact", key=name, alias_from=alias_from, statute=statute, sub=sub, candidates=cands,
                          status=e.status if e else "현행", successor=e.successor if e else None, **extra)

    def _resolve(self, q: str, alias_from: tuple[str, str] | None, depth: int) -> Resolution:
        if depth > _MAX_DEPTH:
            return Resolution("none")
        # 1. Exact match
        if q in self.by_cmp:
            return self._exact(self.by_cmp[q], alias_from)
        # 2. Alias mapping
        if q in self.alias:
            target, how = self.alias[q]
            af = alias_from or (q, target)
            if how == "법률":
                law = split_charge(norm_cmp(target))[0] or norm_cmp(target)
                law = law if law in self.by_law else self.laws.alias.get(law, law)
                if law in self.special_laws:
                    names = list(self.by_law[law])
                    return Resolution("statute_exact", candidates=tuple(names),
                                      statute=self.law_display.get(law) or self.laws.show(law), alias_from=af)
                return self._resolve_law(law, None, af, depth + 1)
            return self._resolve(norm_cmp(target), af, depth + 1)
        st, sub = split_charge(q)
        # 3. Bare parenthesized forms for Criminal Code / Military Criminal Code
        m = _BARE_PAREN_RE.match(q)
        if m:
            inner = m.group(1) + m.group(2)
            r = self._resolve(inner, alias_from or (q, inner), depth + 1)
            if r.kind != "none" or m.group(2) in _SUFFIX_MODS:
                return r
        # 4. Accomplice suffixes (교사, 방조)
        for suf in _ACCOMPLICE:
            if q.endswith(suf) and len(q) > len(suf):
                parent = self._resolve(q[: -len(suf)], None, depth + 1)
                if parent.kind == "exact":
                    key = parent.key + suf
                    af = alias_from or ((q, key) if parent.alias_from else None)
                    return replace(parent, key=key, derived=(parent.key, suf), candidates=(), alias_from=af)
                if parent.kind == "statute_exact":
                    return replace(parent, alias_from=alias_from or parent.alias_from)
        # Attempt suffix (미수)
        if q.endswith("미수") and len(q) > 2:
            parent = self._resolve(q[:-2], None, depth + 1)
            if parent.kind == "exact" and parent.statute is not None:
                return replace(parent, suffix_stripped="미수", alias_from=alias_from or parent.alias_from)
        # 5. Stripping noise
        if ")" in q and not q.endswith(")") and q[q.rindex(")") + 1:] not in _SUFFIX_MODS:
            head = q[: q.rindex(")") + 1]
            r = self._resolve(head, alias_from, depth + 1)
            if r.kind == "exact":
                return self._retag(r, q, alias_from)
        for tail in _TAIL_NOISE:
            if q.endswith(tail) and len(q) > len(tail) + 1:
                r = self._resolve(q[: -len(tail)], alias_from, depth + 1)
                if r.kind != "none":
                    return self._retag(r, q, alias_from)
                break
        if st and q.startswith("구") and len(q) > 3:
            r = self._resolve(q[1:], alias_from, depth + 1)
            if r.kind != "none":
                return self._retag(r, q, alias_from)
        # 6. Law parsing
        if st is None:
            head = self._split_head_law(q)
            if head:
                full, tail = head
                return self._resolve_law(full, tail or None, alias_from, depth + 1, glued=q)
            # 7. Parenthesis-only or prefix queries
            if q in self.by_sub:
                return Resolution("statute_exact", candidates=tuple(self.by_sub[q]), alias_from=alias_from)
            cands = self._prefix_candidates(q)
            if cands:
                return Resolution("statute_exact", candidates=cands, alias_from=alias_from)
            return Resolution("none")
        return self._resolve_law(st, sub, alias_from, depth + 1)

    def _resolve_law(self, st: str, sub: str | None, alias_from: tuple[str, str] | None, depth: int,
                     glued: str | None = None) -> Resolution:
        full = st if st in self.by_law else self.laws.alias.get(st, st)
        rebuilt = full + "위반" + (f"({sub})" if sub else "")
        input_disp = glued or (st + "위반" + (f"({sub})" if sub else ""))
        if rebuilt in self.by_cmp:
            name = self.by_cmp[rebuilt]
            return self._exact(name, alias_from or ((input_disp, name) if (full != st or glued) else None))
        if full in self.special_laws:
            names = list(self.by_law[full])
            statute = self.law_display.get(full) or self.laws.show(full)
            return Resolution("statute_exact", candidates=tuple(names), statute=statute, sub=sub,
                              alias_from=alias_from or ((st, statute) if full != st else None))
        if self.laws.is_law(full) and full not in _BARE_CODES:
            disp = norm_charge(self.laws.show(full))
            name = disp + "위반"
            af = alias_from or ((input_disp, name) if (full != st or sub or glued) else None)
            return Resolution("exact", key=name, alias_from=af, statute=disp, sub=None,
                              law_repealed=self.laws.is_repealed(full))
        if full in self.by_law:
            names = list(self.by_law[full])
            return Resolution("statute_exact", candidates=tuple(names), statute=self.law_display.get(full) or full,
                              sub=sub, alias_from=alias_from)
        return Resolution("none", statute=st, sub=sub)

    @staticmethod
    def _retag(r: Resolution, q: str, alias_from: tuple[str, str] | None) -> Resolution:
        if alias_from:
            return replace(r, alias_from=alias_from)
        return replace(r, alias_from=(q, r.key) if r.key else r.alias_from)

    def _prefix_candidates(self, q: str) -> tuple[str, ...]:
        if len(q) < _PREFIX_CAND_MIN:
            return ()
        out = [name for n, name in self.by_cmp.items() if n != q and n.startswith(q)]
        for sub_name, names in self.by_sub.items():
            if sub_name != q and sub_name.startswith(q):
                out.extend(names)
        seen: set[str] = set()
        uniq: list[str] = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return tuple(uniq)

    def _split_head_law(self, q: str) -> tuple[str, str] | None:
        full = q if q in self.by_law else self.laws.alias.get(q, q)
        if full not in _BARE_CODES and (full in self.by_law or self.laws.is_law(full)):
            return full, ""
        for i in range(len(q) - 1, 1, -1):
            head, tail = q[:i], q[i:]
            if "(" in tail or ")" in tail:
                continue
            if tail.startswith("위반"):
                tail = tail[2:]
                if not tail:
                    continue
            full = head if head in self.by_law else self.laws.alias.get(head, head)
            if full in self.special_laws:
                return full, tail
            if head.endswith(("법", "법률")) and full not in _BARE_CODES and self.laws.is_law(full):
                return full, tail
        return None

    def related(self, name: str) -> list[str]:
        e = self.entries.get(name)
        out: list[str] = []
        if e and e.parent:
            out.append(e.parent)
            out.extend(self.children.get(e.parent, []))
        out.extend(self.children.get(name, []))
        n = norm_cmp(name)
        st, sub = _entry_split(e) if e else split_charge(n)
        if st is None:
            out.extend(self.by_sub.get(n, []))
        else:
            out.extend(self.by_law.get(st, []))
        if e and e.successor:
            out.append(e.successor)
        seen: set[str] = {name}
        uniq = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq


def warm(conn: sqlite3.Connection) -> int:
    return len(official_index(conn).entries)


_T = TypeVar("_T")
_CACHE: dict[tuple[str, str], tuple[int, object]] = {}
_LOCK = threading.RLock()


def db_cached(conn: sqlite3.Connection, name: str, build: Callable[[sqlite3.Connection], _T]) -> _T:
    try:
        path = conn.execute("PRAGMA database_list").fetchone()[2] or ""
    except Exception:
        path = ""
    if not path:
        return build(conn)
    stamp = _stamp(conn, path)
    with _LOCK:
        hit = _CACHE.get((path, name))
        if hit and hit[0] == stamp:
            return hit[1]  # type: ignore[return-value]
        obj = build(conn)
        _CACHE[(path, name)] = (stamp, obj)
        return obj


def _stamp(conn: sqlite3.Connection, path: str) -> tuple[str, object]:
    try:
        row = conn.execute("SELECT value FROM official_charges_meta WHERE key='loaded_at'").fetchone()
        if row and row[0]:
            return ("loaded_at", row[0])
    except sqlite3.OperationalError:
        pass
    try:
        return ("mtime", os.stat(path).st_mtime_ns)
    except OSError:
        return ("mtime", 0)


def law_names(conn: sqlite3.Connection) -> LawNames:
    return db_cached(conn, "laws", LawNames)


def official_index(conn: sqlite3.Connection) -> OfficialCharges:
    return db_cached(conn, "official", OfficialCharges)
