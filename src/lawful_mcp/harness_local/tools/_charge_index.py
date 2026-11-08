"""Resolve charge strings to official charge names for both sentencing tools.

The canonical source is `official_charges`, expanded from appendices 1-5 of the prosecution's charge-naming
regulation. Criminal Act and Military Criminal Act charges use bare names. Appendix 3-5 statutes use
"<statute>위반(<offense>)"; other statutes use only "<statute>위반". Solicitation and aiding follow the charge
name. Appendix 3-5 and other statutes do not separately mark attempts.

All comparisons remove dots, commas and whitespace. Resolution proceeds in this order:
  1. Exact official name, including historical entries and their successors.
  2. An explicit `charge_aliases` entry, resolved again by step 1.
  3. Bare-code wrappers: "형법위반(살인)미수" resolves as "살인미수".
  4. Solicitation/aiding suffixes with an official parent. Attempts under appendix 3-5 and other laws resolve
     through their parent; a bare-code attempt absent from the official table remains unresolved.
  5. Remove decoration: trailing text after a parenthesis, a "죄" tail (사기죄) and the historical-law prefix
     "구". Run this after suffix resolution so accessories and attempts retain their meaning.
  6. Resolve statute abbreviations, old names and split-law successors through `LawNames`. A concatenated
     statute and offense is split only when its head is an appendix statute or a known name ending in
     "법"/"법률". A whole statute name resolves to its bare violation name.
     An official result is `exact`; an appendix statute with an unknown parenthetical is `statute_exact`,
     with its official charges as candidates. Other known statutes use their bare violation name and report
     the discarded parenthetical through `alias_from`. Unknown statutes remain unresolved.
  7. A parenthetical-only form (음주운전) produces its statutes' names as candidates, never an automatic
     selection. Even a unique statute match may be an aggravated special-law offense rather than the intended basic
     code offense.

Names without a matching mapping row can still be exact official names. The sentencing tools decide which
candidates have mapping data. Nothing is chosen by resemblance: an unrelated charge must never supply another
offense's penalty. A name none of the steps reads gets `similar` names — current official names sharing the most
two-character pieces — shown as candidates for the model to choose from.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, replace

from .._charge import norm_charge
from . import statutes
from .statutes import _is_repealed_at, _strip_dots

# Special-law charges use "<statute>위반(<offense>)"; bare Criminal Act charges do not match.
# Some Criminal Act and Military Criminal Act charges also end in "위반" (e.g. "명령위반").
# Classify table entries by their `law` column (_entry_split), or a charge prefix would become a statute name.
_SPLIT_RE = re.compile(r"^(.+?)위반(?:\((.+)\))?$")
# In "형법위반(X)<suffix>" or "군형법위반(X)<suffix>", X plus the optional suffix is the bare charge name.
_BARE_PAREN_RE = re.compile(r"^(?:형법|군형법)위반\((.+?)\)(.*)$")
_BARE_CODES = ("형법", "군형법")     # Appendices 1 and 2 use bare names: X in "X법위반(Y)" is a statute; only Y can be a charge under these codes.
_COMMA_CHARS = (",", "，", "、")     # Separators interchangeable with the middle dot; paired with _strip_dots.
_SUFFIX_MODS = ("미수", "교사", "방조")
_ACCOMPLICE = ("교사", "방조")
# A similar name below this Dice score shares only incidental pieces (청소 → 아동·청소년…, 무기징역 → 근무기피목적위계).
_SIMILAR_MIN = 0.25
_MAX_DEPTH = 4


def norm_cmp(s: str | None) -> str:
    """Normalize for comparison: normalize spaces/dots, remove dots and commas, and unfold fullwidth parentheses.

    Commas and middle dots are interchangeable in statute names. Removing commas creates no additional collisions
    in the source charge and statute tables.
    """
    out = _strip_dots(norm_charge(s or "")).replace("（", "(").replace("）", ")")
    for c in _COMMA_CHARS:
        out = out.replace(c, "")
    return out


def _grams(s: str) -> set[str]:
    """The two-character pieces of a name, parentheses aside."""
    s = norm_cmp(s).replace("(", "").replace(")", "")
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def split_charge(q: str) -> tuple[str | None, str | None]:
    """Split "<statute>위반(<offense>)", or return (None, None). For queries; table entries use _entry_split."""
    m = _SPLIT_RE.match(q)
    return (m.group(1), m.group(2)) if m else (None, None)


# Numeric tokens cover article/type/list indices, including article branches such as "297의2".
_NUMERIC_CHARGE_TOKEN_RE = re.compile(r"^\d{1,5}(?:(?:의|-)\d{1,3})?$")
_NUMERIC_CHARGE_SEP_RE = re.compile(r"[\s\[\]()'\"‚,，·;/]+")
# An article reference, such as "형법 제347조", is not a charge name; no official name has this form.
ARTICLE_REF_RE = re.compile(r"제\s*(\d{1,4})\s*조")


def numeric_charge_tokens(charge: str) -> list[str] | None:
    """Return the tokens if charge is entirely numeric; any Korean or Latin letter makes it a charge name."""
    parts = [p for p in _NUMERIC_CHARGE_SEP_RE.split(charge) if p]
    if not parts or not all(_NUMERIC_CHARGE_TOKEN_RE.match(p) for p in parts):
        return None
    return list(dict.fromkeys(parts))[:8]


class LawNames:
    """Resolve statute existence, current names, abbreviations and renames from st_statutes.

    The current name comes from the latest in-force version per law_id, not its history_status label: older
    rows can lack labels and newly loaded deltas can carry them. Future versions do not rename a law yet.
    Abbreviations, old names and split successors outside st_statutes come from `statute_aliases`.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.display: dict[str, str] = {}   # normalized name -> first display spelling
        self.current: set[str] = set()      # latest name per law_id
        self.repealed: set[str] = set()     # latest names whose in-force version is repealed
        self.alias: dict[str, str] = {}     # normalized abbreviation/old name -> normalized current name
        rows = conn.execute("SELECT law_id, name, short_name, effective_date FROM st_statutes "
                            "WHERE kind='법률'").fetchall()
        today = statutes._today_iso()
        latest: dict[str, tuple[str, str, str, bool]] = {}   # law_id → (eff, name, short, repealed)
        earliest: dict[str, tuple[str, str, str, bool]] = {}
        names_by_id: dict[str, set[str]] = {}
        for law_id, name, short, eff in rows:
            n = norm_cmp(name)
            if not n:
                continue
            self.display.setdefault(n, name)
            names_by_id.setdefault(law_id, set()).add(n)
            e = str(eff or "")
            rec = (e, n, norm_cmp(short), False)
            if earliest.get(law_id) is None or e < earliest[law_id][0]:
                earliest[law_id] = rec
            if e > today:   # by date: the 시행예정 label can lag behind the effective date
                continue
            if latest.get(law_id) is None or e >= latest[law_id][0]:
                latest[law_id] = rec
        for law_id, rec in earliest.items():      # Retain a name even if only future versions have been loaded.
            latest.setdefault(law_id, rec)
        # Repealed by the statute tool's one test: an amendment listed after a repeal does not revive a law, and the
        # laws in `st_lapsed` are gone.
        latest = {lid: (e, n, s, _is_repealed_at(conn, lid, today) is not None) for lid, (e, n, s, _r) in latest.items()}
        self.current = {n for _e, n, _s, _r in latest.values()}
        self.repealed = {n for _e, n, _s, r in latest.values() if r} - {n for _e, n, _s, r in latest.values() if not r}
        # Prefer active laws and newer in-force versions when several laws share an abbreviation.
        ordered = sorted(latest.items(), key=lambda kv: (kv[1][3], -int(kv[1][0] or 0)))
        for law_id, (_e, n, short, _rep) in ordered:
            for old in names_by_id.get(law_id, ()):
                if old != n and old not in self.current:
                    self.alias.setdefault(old, n)
            # An abbreviation that is another law's current name must not replace that real name.
            # An abbreviation equal to an old name may still be replaced, as with "부정경쟁방지법".
            if short and short != n and short not in self.current:
                self.alias.setdefault(short, n)
        # `statute_aliases`: common names, old names and hanja spellings fill gaps; a split statute's successor
        # ('분리 승계': the 2010 sexual-offense split, the 2024 maritime-safety split) overrides its lineage — the
        # penalties went to a law outside it.
        for a, v, how in conn.execute("SELECT alias, statute_name, how FROM statute_aliases"):
            if how == "분리 승계":
                self.alias[norm_cmp(a)] = norm_cmp(v)
            else:
                self.alias.setdefault(norm_cmp(a), norm_cmp(v))

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
    parent: str | None
    status: str            # current | historical
    successor: str | None


def _entry_split(e: Entry) -> tuple[str | None, str | None]:
    """Return a table entry's (statute, parenthetical).

    Criminal Act and Military Criminal Act entries use bare names, including names ending in "위반", and return
    (None, None). Historical entries without a statute return the same pair.
    """
    if not e.law or norm_cmp(e.law) in _BARE_CODES:
        return None, None
    return split_charge(norm_cmp(e.name))


@dataclass(frozen=True)
class Resolution:
    """Charge resolution result: exact | statute_exact | none."""

    kind: str
    key: str | None = None                      # exact: official charge name, verbatim
    candidates: tuple[str, ...] = ()            # statute_exact: official names under this law, current before historical
                                                # For an exact bare appendix-law name, its detailed charges are also candidates.
    alias_from: tuple[str, str] | None = None   # (input, resolved spelling), when an alias, abbreviation, rename or parenthetical changed
    statute: str | None = None                  # statute display name without "위반"; None for bare code charges
    sub: str | None = None                      # input parenthetical
    law_repealed: bool = False                  # bare "<statute>위반" for a repealed law
    suffix_stripped: str | None = None          # resolved through the parent after peeling an attempt suffix
    derived: tuple[str, str] | None = None      # (official parent name, suffix) for a solicitation/aiding derivative
    status: str = "현행"                        # exact: current | historical
    successor: str | None = None                # successor charge for a historical name


class OfficialCharges:
    """One resolution index per database, built from official_charges, charge_aliases and st_statutes."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.laws = LawNames(conn)
        self.entries: dict[str, Entry] = {}
        self.by_cmp: dict[str, str] = {}            # norm_cmp(name) -> name
        self.by_law: dict[str, list[str]] = {}      # norm_cmp(statute) -> names under it, current first
        self.law_display: dict[str, str] = {}       # norm_cmp(statute) -> display spelling, preserving dots
        self.by_sub: dict[str, list[str]] = {}      # norm_cmp(parenthetical) -> names
        self.children: dict[str, list[str]] = {}    # parent -> derivative names
        self.alias: dict[str, tuple[str, str]] = {} # norm_cmp(alias) -> (target, how)
        rows = conn.execute("SELECT name, law, article, hang, parent, status, successor FROM official_charges "
                            "ORDER BY CASE status WHEN '구법' THEN 1 ELSE 0 END, rowid").fetchall()
        for name, law, article, hang, parent, status, successor in rows:
            e = Entry(name, law, article, hang, parent, status or "현행", successor)
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
        for a, target, how in conn.execute("SELECT alias, official_name, how FROM charge_aliases"):
            self.alias.setdefault(norm_cmp(a), (target, how or ""))
        # The pool of `similar`: current names, and "<statute>위반" — the regulation's name for the charges of a
        # statute outside the appendices — for current statutes the table does not list.
        pool = [e.name for e in self.entries.values() if e.status == "현행"]
        pool += [norm_charge(self.laws.show(n)) + "위반" for n in sorted(self.laws.current - self.laws.repealed)
                 if n not in self.by_law and n not in _BARE_CODES]
        self.grams = [(n, _grams(n)) for n in dict.fromkeys(pool)]

    # ---- Resolution ----
    def resolve(self, charge: str) -> Resolution:
        q = norm_cmp(charge)
        if not q:
            return Resolution("none")
        return self._resolve(q, None, 0)

    def resolve_exact(self, charge: str) -> Resolution | None:
        """Return only an exact match."""
        r = self.resolve(charge)
        return r if r.kind == "exact" else None

    def _exact(self, name: str, alias_from: tuple[str, str] | None, **extra) -> Resolution:
        e = self.entries.get(name)
        st, sub = _entry_split(e) if e else split_charge(norm_cmp(name))
        statute = (self.law_display.get(st) or (e.law if e and e.law else st)) if st else None
        cands: tuple[str, ...] = ()
        if st and not sub and st in self.special_laws:
            # A bare "<statute>위반" may itself be an appendix entry; include its detailed charges too.
            cands = tuple(x for x in self.by_law[st] if x != name)
        return Resolution("exact", key=name, alias_from=alias_from, statute=statute, sub=sub, candidates=cands,
                          status=e.status if e else "현행", successor=e.successor if e else None, **extra)

    def similar(self, charge: str, k: int = 8) -> tuple[str, ...]:
        """Up to k current official names sharing the most two-character pieces with `charge` (Dice) — those holding
        all of its pieces first (위험운전 → 특정범죄가중처벌등에관한법률위반(위험운전치사)) — as candidates for an unknown
        spelling (직무집행방해 → 공무집행방해, 강도상습 → 상습강도), never chosen here. "형법위반" gets none: the Criminal
        and Military Criminal Acts' charges have no "위반" form."""
        q = norm_cmp(charge)
        if not q or q in {c + "위반" for c in _BARE_CODES}:
            return ()
        g = _grams(q)
        scored = [(g <= h, 2 * len(g & h) / (len(g) + len(h)), n) for n, h in self.grams if g & h]
        scored = sorted((x for x in scored if x[1] >= _SIMILAR_MIN), key=lambda x: (not x[0], -x[1], len(x[2]), x[2]))
        return tuple(n for _all, _score, n in scored[:k])

    def _resolve(self, q: str, alias_from: tuple[str, str] | None, depth: int) -> Resolution:
        if depth > _MAX_DEPTH:
            return Resolution("none")
        # 1. Exact match.
        if q in self.by_cmp:
            return self._exact(self.by_cmp[q], alias_from)
        # 2. Alias table.
        if q in self.alias:
            target, how = self.alias[q]
            af = alias_from or (q, target)
            if how == "법률":
                law = split_charge(norm_cmp(target))[0] or norm_cmp(target)
                law = law if law in self.by_law else self.laws.alias.get(law, law)
                if law in self.special_laws:      # An umbrella term such as "성폭력" identifies a statute family, not one charge.
                    names = list(self.by_law[law])
                    return Resolution("statute_exact", candidates=tuple(names),
                                      statute=self.law_display.get(law) or self.laws.show(law), alias_from=af)
                return self._resolve_law(law, None, af)
            return self._resolve(norm_cmp(target), af, depth + 1)
        st, sub = split_charge(q)
        # 3. Bare-code wrapper: X plus the suffix is the charge name (rules 1 and 2 of the source regulation).
        m = _BARE_PAREN_RE.match(q)
        if m:
            # For bare code charges, "형법위반(살인)미수" means "살인미수". Preserve nonexistent combinations
            # as unresolved: later tail stripping must not turn an unpunishable attempt into the completed offense.
            inner = m.group(1) + m.group(2)
            r = self._resolve(inner, alias_from or (q, inner), depth + 1)
            if r.kind != "none" or m.group(2) in _SUFFIX_MODS:
                return r
        # 4. Solicitation/aiding derivatives require an official parent and run before parenthetical tail stripping.
        # Otherwise stripping the suffix would mix accessories into the principal-offense pool.
        for suf in _ACCOMPLICE:
            if q.endswith(suf) and len(q) > len(suf):
                parent = self._resolve(q[: -len(suf)], None, depth + 1)
                if parent.kind == "exact":
                    key = parent.key + suf
                    af = alias_from or ((q, key) if parent.alias_from else None)   # Include the suffix in the resolved spelling.
                    return replace(parent, key=key, derived=(parent.key, suf), candidates=(), alias_from=af)
                if parent.kind == "statute_exact":
                    return replace(parent, alias_from=alias_from or parent.alias_from)   # The statute is known but its parenthetical is not.
        # Appendix 3-5 and other laws do not mark attempts; code attempts must be listed explicitly.
        if q.endswith("미수") and len(q) > 2:
            parent = self._resolve(q[:-2], None, depth + 1)
            if parent.kind == "exact" and parent.statute is not None:
                return replace(parent, suffix_stripped="미수", alias_from=alias_from or parent.alias_from)
        # 5. Strip only decoration absent from official notation, after suffix resolution in step 4.
        # Attempt, solicitation and aiding are parts of the charge; retain them after a parenthesis.
        # Removing a rejected suffix here would incorrectly resolve an unpunishable attempt.
        # (a) Other text after the closing parenthesis is decoration; appendix 3-5 names end at that parenthesis.
        if ")" in q and not q.endswith(")") and q[q.rindex(")") + 1:] not in _SUFFIX_MODS:
            head = q[: q.rindex(")") + 1]
            r = self._resolve(head, alias_from, depth + 1)
            if r.kind == "exact":
                return self._retag(r, q, alias_from)
        # (b) A "죄" tail: 사기죄 → 사기. "…범죄" is a family name and stays.
        if q.endswith("죄") and not q.endswith("범죄") and len(q) > 2:
            r = self._resolve(q[:-1], alias_from, depth + 1)
            if r.kind != "none":
                return self._retag(r, q, alias_from)
        # (c) Strip the historical-law prefix "구" only from a form containing a statute part.
        if st and q.startswith("구") and len(q) > 3:
            r = self._resolve(q[1:], alias_from, depth + 1)
            if r.kind != "none":
                return self._retag(r, q, alias_from)
        # 6. Resolve the statute part.
        if st is None:
            head = self._split_head_law(q)
            if head:
                full, tail = head
                return self._resolve_law(full, tail or None, alias_from, glued=q)
            # 7. A parenthetical-only form yields candidates, even when all belong to one statute.
            if q in self.by_sub:
                return Resolution("statute_exact", candidates=tuple(self.by_sub[q]), alias_from=alias_from)
            return Resolution("none")
        return self._resolve_law(st, sub, alias_from)

    def _resolve_law(self, st: str, sub: str | None, alias_from: tuple[str, str] | None,
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
            # Other statutes have one official "<statute>위반" name; drop the parenthetical and report the change.
            # Bare codes do not enter here: their official names are the bare appendix 1-2 entries.
            disp = norm_charge(self.laws.show(full))
            name = disp + "위반"
            af = alias_from or ((input_disp, name) if (full != st or sub or glued) else None)
            return Resolution("exact", key=name, alias_from=af, statute=disp, sub=None,
                              law_repealed=self.laws.is_repealed(full))
        if full in self.by_law:     # A historical-only statute has only the names listed under it.
            names = list(self.by_law[full])
            return Resolution("statute_exact", candidates=tuple(names), statute=self.law_display.get(full) or full,
                              sub=sub, alias_from=alias_from)
        return Resolution("none", statute=st, sub=sub)

    @staticmethod
    def _retag(r: Resolution, q: str, alias_from: tuple[str, str] | None) -> Resolution:
        """Report the original input and final official name after stripping decoration.

        An intermediate spelling would make the guidance disagree with the actual mapping key. Keep an outer
        alias_from if one already exists because it is closer to the original input.
        """
        if alias_from:
            return replace(r, alias_from=alias_from)
        return replace(r, alias_from=(q, r.key) if r.key else r.alias_from)

    def _split_head_law(self, q: str) -> tuple[str, str] | None:
        """Split a known statute head and offense tail, preferring longer statute names.

        A whole known name or abbreviation returns (normalized current name, ''). A nonempty tail is accepted only
        for appendix statutes or known statute names ending in "법"/"법률", not loose colloquial prefixes.
        A tail starting with "위반" supplies a missing parenthetical, as in "도로교통법위반음주운전".
        """
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

    # ---- Related-name discovery ----
    def related(self, name: str) -> list[str]:
        """List related derivatives, siblings, special-law counterparts, charges under the same law and successors.

        Do not truncate before the tool counts and sorts the associated samples; a display limit belongs after sorting.
        """
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


# One index per database file: the tools read a database that does not change while the process runs. An in-memory
# database (a test fixture) is read each time.
_INDEX: dict[str, OfficialCharges] = {}


def official_index(conn: sqlite3.Connection) -> OfficialCharges:
    path = conn.execute("PRAGMA database_list").fetchone()[2]
    if not path:
        return OfficialCharges(conn)
    if path not in _INDEX:
        _INDEX[path] = OfficialCharges(conn)
    return _INDEX[path]
