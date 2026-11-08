"""statutory_penalty — every provision a charge can rest on, with its statutory penalty. Data only, no text.

Source: `charge_legal_map` (the sentencing mapping table — official charge name → the provisions punishing it, with
the penalty of each provision, of each of its branches, or of the source crime it aggravates) and `clm_versions`
(earlier wordings by effective date). `read_charge` returns a `Reading`: how the name was read, and one `Provision`
per provision, branch or source crime — with its penalty, or the reason the table cannot give one. It computes
nothing for a case: `sentencing_analysis` shows the provisions and the model picks by the facts.

A bound the article does not state is empty, as the article words it (「10년 이하」, 「1년 이상」); the general part's
floors and cap (imprisonment 1 month to 30 years, §42; a fine from 50,000 won, §45) are the rules' to state.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, replace

from .._charge import norm_charge
from ._charge_index import ARTICLE_REF_RE, Resolution, norm_cmp, numeric_charge_tokens, official_index
from ._coerce import to_iso_date

# ---------------------------------------------------------------- what is returned


@dataclass(frozen=True)
class Penalty:
    imp_min: int | None = None     # months
    imp_max: int | None = None
    fine_min: int | None = None    # won
    fine_max: int | None = None
    fine_formula: dict | None = field(default=None, hash=False)   # a fine by formula: base_label, low/high_mult
    life: bool = False
    death: bool = False
    kinds: tuple[str, ...] = ()    # the article's penalty kinds (imprisonment, fine, life, death), also without amounts


@dataclass(frozen=True)
class Provision:
    statute: str                   # as the table writes it: "형법", "도로교통법"
    article_num: int
    branch: int = 0                # 의N
    paragraph: str | None = None   # "②"
    where: str = ""                # the branch condition or the source crime, "" for the provision itself
    name: str = ""                 # the mapping row's charge name (tells apart rows of one article)
    penalty: Penalty | None = None
    times: float | None = None     # the source crime's penalty × this is `penalty` (already applied)
    reason: str = ""               # why the table gives no penalty
    category_id: int | None = None
    statute_id: int | None = None

    @property
    def article(self) -> str:      # "형법 §257 ②"
        return article_label({"statute_name": self.statute, "article_no_num": self.article_num,
                              "article_branch": self.branch, "paragraph": self.paragraph})

    @property
    def article_no(self) -> str:   # for statute_lookup: "347", "148의2"
        return f"{self.article_num}" + (f"의{self.branch}" if self.branch else "")


@dataclass
class Reading:
    asked: str
    kind: str                      # ok | candidates | no_guideline | unknown | numeric
    key: str = ""                  # the name the table was read with (suffix removed)
    suffixes: tuple[str, ...] = ()     # 미수 · 교사 · 방조 read off the end of the name
    notes: list[str] = field(default_factory=list)   # how the name was read
    provisions: list[Provision] = field(default_factory=list)
    resolution: Resolution | None = None   # the official-name reading of `key` (what the name is, when not ok);
                                           # kind "similar" for an unknown spelling's look-alike names
    official: Resolution | None = None     # the input read as an official charge name; when exact, it keys the
                                           # actual sentences, else its candidates are the names judgments use
    tokens: list[str] = field(default_factory=list)  # numbers given instead of a name


# ---------------------------------------------------------------- the Criminal Act's limits on a changed penalty
_CAP_50 = 600              # §42 proviso: an aggravated term up to 50 years; §55 ①: a commuted death or life sentence
_DEATH_COMMUTED_MIN = 240  # §55 ① 1: death mitigated → life or 20–50 years
_LIFE_COMMUTED_MIN = 120   # §55 ① 2: life mitigated → 10–50 years

# ---------------------------------------------------------------- reading the name
SUFFIXES = ("미수", "교사", "방조")


@dataclass(frozen=True)
class _Name:
    key: str                         # lookup key, suffixes removed
    raw: str                         # input normalized, suffixes kept
    suffixes: tuple[str, ...]
    notes: tuple[str, ...]


def _read_name(conn: sqlite3.Connection, charge: str) -> _Name:
    """The input as an official name (abbreviations, dots, colloquial forms resolved), with 미수 · 교사 · 방조 peeled
    off its end when the rest is a charge.

    ⚠ Suffixes are peeled before the exact match: 살인미수 is an official name, and matching it first would lose the
    note that the penalty is the completed crime's (the table holds the completed crime's rows only). 교사 · 방조 that are part of an official
    name (간첩방조, 자살방조) stay."""
    raw = norm_charge(charge or "")
    index = official_index(conn)
    notes = []

    def canonical(s: str):
        r = index.resolve_exact(s)
        return (r.key, r.alias_from) if r is not None else None

    key, alias, suffixes = raw, None, []
    for _round in SUFFIXES:   # suffixes stack: 사기미수방조
        for suf in SUFFIXES:
            if not key.endswith(suf) or len(key) <= len(suf):
                continue
            if suf != "미수":
                whole = index.resolve_exact(key)
                if whole is not None and whole.derived is None and norm_cmp(whole.key) == norm_cmp(key):
                    continue
            hit = canonical(key[: -len(suf)])
            if hit is not None:
                key, alias = hit[0], hit[1] or alias
                suffixes.insert(0, suf)
                break
        else:
            break
    if not suffixes:
        hit = canonical(raw)
        if hit is not None:
            key = raw = hit[0]
            alias = hit[1]
    if alias:
        notes.append(f"죄명 표기 해석 {alias[0]} → {alias[1]}")
    return _Name(key, raw, tuple(suffixes), tuple(notes))


# ---------------------------------------------------------------- mapping rows of a name
_SELECT_ROW = (
    "SELECT id, charge_key, sg_category_id, statute_id, statute_name, article_no_num, article_branch, paragraph, "
    "sentence_kind_options, stat_imp_min_months, stat_imp_max_months, stat_fine_min_won, stat_fine_max_won, "
    "has_life, has_death, has_conditional_branch, branch_options, reference_mode, reference_multiplier, "
    "reference_articles, fine_formula FROM charge_legal_map")


def _table_keys(conn: sqlite3.Connection, officials: list[str]) -> dict[str, list[str]]:
    """Official name → its mapping keys: its own name first, plus the keys `official_guideline_map` links to it
    (the guideline's umbrella names: 강간등살인, …)."""
    out = {o: [o] for o in officials}
    for o, k in conn.execute(f"SELECT official_name, charge_key FROM official_guideline_map WHERE official_name IN "
                             f"({','.join('?' * len(officials))})", officials):
        out[o].append(k)
    return out


def rows_of(conn: sqlite3.Connection, officials: list[str]) -> list[tuple[str, sqlite3.Row]]:
    """(name, row) for the names' mapping rows — its own rows, then its linked keys' rows; by category, then article,
    within a key."""
    keys_by = _table_keys(conn, officials)
    keys = list(dict.fromkeys(k for ks in keys_by.values() for k in ks))
    if not keys:
        return []
    by_key: dict[str, list[sqlite3.Row]] = {}
    for r in conn.execute(_SELECT_ROW + f" WHERE charge_key IN ({','.join('?' * len(keys))}) ORDER BY sg_category_id, "
                          "statute_id, article_no_num, article_branch, paragraph", keys):
        by_key.setdefault(r["charge_key"], []).append(r)
    return [(o, r) for o in officials for k in keys_by[o] for r in by_key.get(k, [])]


# ---------------------------------------------------------------- the version in force on the offense date
def _version(conn: sqlite3.Connection, row: sqlite3.Row, offense_iso: str | None) -> dict | str:
    """The row with the wording in force on `offense_iso` (its latest `clm_versions` entry on or before the date), or
    why there is none: that wording has no such crime at this article — a version row with no penalty marks it (the
    provision was added or the article renumbered later: 특수강요 §324 ② before 2016, 결혼약취 §288 ① before 2013) — or
    the table holds no wording for that date."""
    payload = dict(row)
    if not offense_iso:
        return payload
    v = conn.execute("SELECT * FROM clm_versions WHERE clm_id=? AND effective_date<=? ORDER BY effective_date DESC "
                     "LIMIT 1", (row["id"], offense_iso)).fetchone()
    if v is None:
        return "매핑표에 행위 시점 문언의 법정형이 없다 — 그때 문언은 근거 조문(offense_date)으로 확인한다"
    if _empty_version(v):
        return ("행위 시점의 문언에는 이 조·항에 이 죄가 없다(뒤에 신설되거나 조문이 바뀌었다) — "
                "그때 적용할 조항은 근거 조문으로 확인한다")
    for col in ("sentence_kind_options", "stat_imp_min_months", "stat_imp_max_months", "stat_fine_min_won",
                "stat_fine_max_won", "has_life", "has_death", "has_conditional_branch", "branch_options",
                "reference_mode", "reference_multiplier", "reference_articles", "fine_formula"):
        payload[col] = v[col]
    return payload


def _empty_version(v: sqlite3.Row) -> bool:
    """A version row with no penalty at all — the wording of its date has no such crime at this article
    (`match_reason` says why)."""
    return (json.loads(v["sentence_kind_options"]) == [] and v["stat_imp_min_months"] is None
            and v["stat_imp_max_months"] is None and v["stat_fine_min_won"] is None and v["stat_fine_max_won"] is None
            and not v["has_life"] and not v["has_death"] and not v["has_conditional_branch"]
            and json.loads(v["branch_options"]) == [] and v["reference_mode"] is None and not v["fine_formula"])


# ---------------------------------------------------------------- penalties
def _from(d, *, branch: bool = False) -> Penalty:
    """A row (or a branch option) as a Penalty. A branch option's keys drop the `stat_` prefix."""
    p = "" if branch else "stat_"
    get = d.get if isinstance(d, dict) else (lambda k, _d=None: d[k] if k in d.keys() else _d)
    kinds = get("sentence_kind_options")
    kinds = json.loads(kinds or "[]") if isinstance(kinds, str) or kinds is None else list(kinds)
    formula = get("fine_formula")
    formula = json.loads(formula) if isinstance(formula, str) and formula else (formula or None)
    return Penalty(imp_min=get(f"{p}imp_min_months"), imp_max=get(f"{p}imp_max_months"),
                   fine_min=get(f"{p}fine_min_won"), fine_max=get(f"{p}fine_max_won"), fine_formula=formula,
                   life=bool(get("has_life")), death=bool(get("has_death")), kinds=tuple(kinds))


def _scaled(formula: dict | None, k: float) -> dict | None:
    if not formula:
        return formula
    return {**formula, **{m: formula[m] * k for m in ("low_mult", "high_mult") if formula.get(m) is not None}}


def _mitigated(p: Penalty) -> Penalty:
    """Criminal Act §55 ① applied to every kind: death → life or 20–50 years, life → 10–50 years, a fixed term and
    a fine halved at both bounds (a fine formula's multiples likewise)."""
    mins, maxs = [], []
    if p.death:
        mins.append(_DEATH_COMMUTED_MIN)
        maxs.append(_CAP_50)
    if p.life:
        mins.append(_LIFE_COMMUTED_MIN)
        maxs.append(_CAP_50)
    if p.imp_min is not None:
        mins.append(max(p.imp_min // 2, 1))
    if p.imp_max is not None:
        maxs.append(p.imp_max // 2)
    kinds = [k for k in p.kinds if k not in ("life", "death")]
    if (mins or maxs) and "imprisonment" not in kinds:
        kinds.insert(0, "imprisonment")
    return replace(p, imp_min=min(mins) if mins else None, imp_max=max(maxs) if maxs else None,
                   fine_min=None if p.fine_min is None else p.fine_min // 2,
                   fine_max=None if p.fine_max is None else p.fine_max // 2, fine_formula=_scaled(p.fine_formula, 0.5),
                   life=p.death, death=False, kinds=tuple(kinds + (["life"] if p.death else [])))


def _multiplied(p: Penalty, k: float) -> Penalty:
    """The source crime's penalty under the provision's multiplier. 「그 죄에 정한 형의 …까지 가중」 raises the minimum
    as well as the maximum (대법원 2017. 6. 29. 선고 2016도18194), imprisonment within the §42 proviso's cap;
    ×0.5 (「본죄의 2분의 1을 감경」) is statutory mitigation."""
    if k < 1:
        return _mitigated(p)

    def up(v, cap=None):
        return None if v is None else min(int(v * k), cap) if cap else int(v * k)
    return replace(p, imp_min=up(p.imp_min, _CAP_50), imp_max=up(p.imp_max, _CAP_50), fine_min=up(p.fine_min),
                   fine_max=up(p.fine_max), fine_formula=_scaled(p.fine_formula, k))


# ---------------------------------------------------------------- source crimes
_CIRCLED = {str(n): c for n, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮", start=1)}


def article_label(d) -> str:
    """"형법 §257 ②" for a row or a source-crime reference."""
    get = d.get if isinstance(d, dict) else (lambda k: d[k])
    branch, para = get("article_branch") or 0, get("paragraph") or ""
    return (f"{get('statute_name') or '?'} §{get('article_no_num')}" + (f"의{branch}" if branch else "")
            + (f" {_CIRCLED.get(para, para)}" if para else ""))


def _source_rows(conn: sqlite3.Connection, ref: dict) -> list[sqlite3.Row]:
    """A source-crime reference's rows — the data names them (`rows`: mapping-row ids)."""
    ids = ref["rows"]
    return conn.execute(_SELECT_ROW + f" WHERE id IN ({','.join('?' * len(ids))}) ORDER BY id", ids).fetchall() \
        if ids else []


def _same_penalty(rows: list[sqlite3.Row]) -> bool:
    def key(r):
        p = _from(r)
        return (p.imp_min, p.imp_max, p.fine_min, p.fine_max, p.life, p.death, r["fine_formula"],
                r["branch_options"] if r["has_conditional_branch"] else None)
    return len({key(r) for r in rows}) <= 1


def _branches(d) -> list[tuple[str, Penalty]]:
    """(condition, penalty) per branch of the article; a branch without a fine formula takes the row's."""
    formula = json.loads(d["fine_formula"]) if d["fine_formula"] else None
    out = []
    for o in json.loads(d["branch_options"]):
        p = _from(o, branch=True)
        out.append((o["cond"], p if p.fine_formula or not formula else replace(p, fine_formula=formula)))
    return out


def _penalties(conn: sqlite3.Connection, row: dict) -> list[dict]:
    """The provision's penalties: one per branch of its article, one per source crime it applies, or its own."""
    mode = row["reference_mode"]
    if row["has_conditional_branch"]:
        return [{"where": cond, "penalty": p} for cond, p in _branches(row)]
    if mode not in ("가중", "준용", "공동가중", "분기"):
        return [{"penalty": _from(row)}]
    refs = json.loads(row["reference_articles"])
    if len(refs) == 1 and refs[0].get("modifier_kind"):
        note = f" ({refs[0]['note']})" if refs[0].get("note") else ""
        return [{"reason": f"자기 법정형이 없는 가중 규정이다{note} — 기본 범죄를 죄명 목록에 함께 넣고, 그 형을 "
                           "규칙 3 처단형의 본조 가중으로 가중한다"}]
    if not refs:
        return [{"reason": "매핑표에 원범죄가 적혀 있지 않다"}]
    times = row["reference_multiplier"]
    out = []
    for ref in refs:
        where = f"원범죄 {article_label(ref)}" + (f" — {ref['note']}" if ref["note"] else "")
        rows = _source_rows(conn, ref)
        if not rows:
            out.append({"where": where, "reason": f"매핑표에 원범죄({article_label(ref)})의 법정형이 없다"})
        elif not _same_penalty(rows):
            names = ", ".join(dict.fromkeys(r["charge_key"] for r in rows))
            out.append({"where": where, "reason": f"원범죄의 죄마다 법정형이 다르다({names}) — 해당 죄의 형에 ×{times:g}"})
        elif rows[0]["has_conditional_branch"]:   # the source crime's items differ (아동복지법 §71 ①)
            out += [{"where": f"{where} [{cond}]", "times": times, "penalty": _multiplied(p, times)}
                    for cond, p in _branches(rows[0])]
        else:
            out.append({"where": where, "times": times, "penalty": _multiplied(_from(rows[0]), times)})
    return out


def _provisions(conn: sqlite3.Connection, row: sqlite3.Row, offense_iso: str | None) -> list[Provision]:
    payload = _version(conn, row, offense_iso)
    base = dict(statute=row["statute_name"], article_num=row["article_no_num"],
                branch=row["article_branch"], paragraph=row["paragraph"],
                category_id=row["sg_category_id"], statute_id=row["statute_id"], name=row["charge_key"])
    if isinstance(payload, str):
        return [Provision(**base, reason=payload)]
    return [Provision(**base, **p) for p in _penalties(conn, payload)]


# ---------------------------------------------------------------- the entry
def read_charge(conn: sqlite3.Connection, charge: str, offense_date: str | None = None) -> Reading:
    """One input name: how it was read, its official name (for the actual sentences), and its provisions — or, when
    the table has none, what the name is (a statute's charges or names like it to choose from, an official name
    without a guideline, or an unknown spelling)."""
    raw = norm_charge(charge or "")
    tokens = numeric_charge_tokens(charge) or numeric_charge_tokens(raw)
    if tokens is None and (ref := ARTICLE_REF_RE.search(charge or "")):
        tokens = [ref.group(1)]
    if tokens is not None:
        return Reading(charge, "numeric", tokens=tokens)
    index = official_index(conn)
    name = _read_name(conn, charge)
    res = index.resolve(raw)   # the input as written: its official name keys the actual sentences
    # The table is read with the suffix-free name; when that has no rows, with its official spelling.
    res_key = res if norm_cmp(name.key) == norm_cmp(raw) else index.resolve(name.key)
    rows = rows_of(conn, [name.key])
    if not rows and res_key.kind == "exact" and res_key.key != name.key:
        rows = rows_of(conn, [res_key.key])
    reading = Reading(charge, "ok", name.key, name.suffixes, list(name.notes), resolution=res_key, official=res)
    if rows:
        offense_iso = to_iso_date(offense_date)
        reading.provisions = [p for _o, r in rows for p in _provisions(conn, r, offense_iso)]
    elif res_key.candidates:
        reading.kind = "candidates"
    elif res_key.kind == "exact":
        reading.kind = "no_guideline"
    elif similar := index.similar(name.key):   # an unknown spelling: names like it to choose from
        reading.kind, reading.resolution = "candidates", Resolution("similar", candidates=similar)
    else:
        reading.kind = "unknown"
    if res_key.alias_from and not any(n.startswith("죄명 표기 해석") for n in reading.notes):
        reading.notes.append(f"죄명 표기 해석 {res_key.alias_from[0]} → {res_key.alias_from[1]}")
    return reading


def guideline_articles(conn: sqlite3.Connection, names: list[str]) -> dict[str, list[str]]:
    """Official names → the articles of their mapping rows under a guideline (the names that have one)."""
    out: dict[str, list[str]] = {}
    for name, row in rows_of(conn, names):
        if row["sg_category_id"] is None:   # outside the guidelines' scope
            continue
        arts = out.setdefault(name, [])
        if article_label(row) not in arts:
            arts.append(article_label(row))
    return out
