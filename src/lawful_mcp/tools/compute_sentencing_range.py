"""Sentencing arithmetic, from a charge to a verified sentence.

Korean sentencing runs through four stages, and this tool advances through
them as the caller supplies more: give it a charge and it returns the
statutory range; add the statutory adjustments and it applies them in the
order the Criminal Act prescribes; add the guideline leaf and the factors
the court found and it returns the recommended range; add a proposed
sentence and it checks that sentence against the range and against the
conditions for suspending it.

**The tool computes; it does not decide.** Which adjustments apply, which
guideline leaf a case falls under, which factors are present — those are
findings, and the tool asks for them rather than inferring them. Where a
charge maps to more than one provision, it returns the options and waits.

There is no state between calls: each call carries the whole picture. That
is why an unresolved question comes back as a list of choices instead of a
guess — a guess would be indistinguishable from a finding in the answer.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BeforeValidator
from pydantic_ai import RunContext

from .._charge import charge_key as _charge_key_normalize
from ..deps import HarnessDeps, open_db
from ..eval.recommended_range import (
    AppliedFactor,
    RecommendedRange,
    determine_range,
    in_range,
    within_range_position,
)
from ._coerce import (
    coerce_dict, coerce_dict_list, coerce_int, coerce_list, coerce_str, to_iso_date)
from ._dedup import dedup_guard

_log = logging.getLogger(__name__)


def _rid(row) -> str:
    """Row id for a log line, tolerant of rows from different queries."""
    try:
        return str(row["id"])
    except (IndexError, KeyError, TypeError):
        return "?"


# Reference modes that legitimately fall through to direct handling.
# An unknown mode reaching the same branch would emit an unadjusted range
# that looks entirely normal — no aggravation applied, no error raised — so
# unknown values are marked in the trace instead.
_DIRECT_FALLTHROUGH_MODES = {None, "정보", "누범가중"}


def _b(x) -> str:
    """Format a bound, showing an unspecified one as '?' rather than 'None'."""
    return "?" if x is None else str(x)


# ---------- Formatting helpers for the model ----------
# The reader of tool responses is an LLM, which echoes parts of the output
# (month numbers, enum keys) back in subsequent calls. Three invariants hold:
# 1. Quantities use a single unit (months canonical, no years in numeric ranges
#    to prevent arithmetic errors on round-trip).
# 2. Consistent field labels across all stages (different names confuse the model).
# 3. No raw Python representations or English enum tokens (e.g. ['imprisonment']
#    or applied=True) leaked into responses.
# Open bounds use '이상'/'이하' instead of '?'.
_KIND_LABEL = {"imprisonment": "자유형", "fine": "벌금", "life": "무기", "death": "사형"}
_KIND_ORDER = ("imprisonment", "fine", "life", "death")
_MODIFIER_LABEL = {"is_attempted": "미수", "is_accessory": "방조", "is_solicitor": "교사"}


def _span_months(lo: int | None, hi: int | None) -> str:
    """Format a month range: `60~360월`, `180월 이상`, `12월 이하`, or `미상`."""
    if lo is None and hi is None:
        return "미상"
    if hi is None:
        return f"{lo}월 이상"
    if lo is None:
        return f"{hi}월 이하"
    return f"{lo}~{hi}월"


def _span_won(lo: int | None, hi: int | None) -> str:
    """Format a won range: `0~50,000,000원`, `5,000,000원 이하`, or `미상`."""
    if lo is None and hi is None:
        return "미상"
    if hi is None:
        return f"{lo:,}원 이상"
    if lo is None:
        return f"{hi:,}원 이하"
    return f"{lo:,}~{hi:,}원"


def _kinds_line(
    kind_options: list[str] | None,
    *,
    has_imp: bool,
    has_fine: bool,
    has_life: bool,
    has_death: bool,
) -> str | None:
    """Format `- 형종: 자유형 · 벌금 · 무기 · 사형` combining mapping and actual flags."""
    present = set(kind_options or [])
    if has_imp:
        present.add("imprisonment")
    if has_fine:
        present.add("fine")
    if has_life:
        present.add("life")
    if has_death:
        present.add("death")
    if not present:
        return None
    ordered = [k for k in _KIND_ORDER if k in present] + sorted(present - set(_KIND_ORDER))
    return "- 형종: " + " · ".join(_KIND_LABEL.get(k, k) for k in ordered)


def _source_label(src: str) -> str:
    """Convert EffectivePenalty.source provenance token to a Korean label."""
    if src == "direct":
        return "본조 직접"
    if src.startswith("direct_"):
        return f"본조 직접 ({src[len('direct_'):]})"
    if src.startswith("reference_"):
        mode, _, target = src[len("reference_"):].partition(":")
        return f"{mode} 참조 → {target}" if target else f"{mode} 참조"
    if src.startswith("branch:"):
        return f"본조 분기 {_branch_key_label(src[len('branch:'):])}"
    if src.startswith("versioned:"):
        return f"행위시 시점본 (시행 {src[len('versioned:'):]})"
    return src


def _rec_range_line(rec) -> str:
    """One-line recommended range; None bound means open. Both None means uncalculated."""
    if rec.min_months is None and rec.max_months is None:
        note = " (사형·무기형)" if rec.has_life else ""
        return f"- 범위: 권고 형량범위 없음 (상·하한 미산출){note}"
    return ("- 범위: " + _span_months(rec.min_months, rec.max_months)
            + (" · 무기 가능" if rec.has_life else ""))

# ---------- charge_key normalize + suffix split ----------

# Offences with no sentencing guideline: point at the statute lookup.
_NOT_FOUND_HINT = (
    "양형기준 비등재 가능성. 법정형 조회는 `statute_lookup` 도구 사용."
)

# Attempt, solicitation and aiding have no rows of their own — they are
# modifiers on a completed offence. A charge naming one is split so the
# parent offence can be looked up and the modifier applied.
_SUFFIX_MODIFIERS: list[tuple[str, str]] = [
    ("미수", "is_attempted"),
    ("교사", "is_solicitor"),
    ("방조", "is_accessory"),
]


# A leading bracketed number, as in "[1]상해". Judgments list their applied
# provisions as "[1] 형법 §257 / [2] …", and the tag travels with the offence
# name when a caller copies it across. No charge key in the mapping starts
# with '[', so stripping it cannot cost a legitimate input.
# ⚠ Strip only when something remains: a bare number ("[123]") belongs to
# `_numeric_charge_tokens`, and emptying the string here would lose that path.
_BRACKET_NUM_PREFIX_RE = re.compile(r"^\[\d{1,4}\]")

# Wrapping quotes ('"협박"'), where a caller passes the offence name as if it
# were a string literal. One such input came in three times running without
# ever recovering: the quotes survived normalisation and the exact match was
# missed. None of the 853 keys in the mapping holds a quote character, so —
# as with the bracket tag — stripping cannot cost a legitimate input.
# ⚠ One layer at a time, only when both ends pair up and something remains.
# A lone quote, or one inside the string, is left alone.
_WRAP_QUOTE_PAIRS = {
    '"': '"', "'": "'", "`": "`", "＂": "＂", "＇": "＇",
    "“": "”", "‘": "’", "「": "」", "『": "』",
}


@dataclass(frozen=True)
class NormalizedCharge:
    """A normalised charge key plus any modifier split off its tail."""

    key: str                        # 정규화된 lookup 키 (suffix 제거 가능)
    raw_key: str                    # charge_key() + 선행 번호 딱지·감싼 인용부호 제거 (suffix 미제거)
    modifiers: dict[str, bool]      # {is_attempted, is_accessory, is_solicitor}
    suffix_split_applied: bool      # 자동 분리 발생 여부 (응답 trace)
    bracket_prefix_stripped: str | None = None  # 제거된 딱지("[9]") — 응답 note 용
    quotes_stripped: str | None = None          # 제거된 감싼 인용부호('""') — 응답 note 용


def _normalize_charge(
    conn: sqlite3.Connection,
    charge: str,
    *,
    is_attempted: bool = False,
    is_accessory: bool = False,
    is_solicitor: bool = False,
) -> NormalizedCharge:
    """Split a charge into a lookup key and its modifiers.

    Rules:
      1. `charge_key()` normalises spacing and dots, keeping parentheses.
      1-b. A leading bracketed number ('[1]상해') and wrapping quotes
         ('"협박"') are stripped, but only when something remains after them.
         Either can wrap the other ('[1] "협박"'), so they alternate.
      2. A suffix (미수 / 교사 / 방조) at the *end* of the charge key splits off
         and sets a modifier — but only when the parent charge, the key without
         it, is itself in the mapping.
      3. A modifier the caller stated (is_attempted and friends) overrides: the
         two are OR-ed, never replaced.
    """
    raw = _charge_key_normalize(charge or "")
    bracket_prefix = None
    quotes_stripped = None
    for _ in range(4):  # one layer each in the observed inputs; the cap is slack
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
    modifiers: dict[str, bool] = {
        "is_attempted": bool(is_attempted),
        "is_accessory": bool(is_accessory),
        "is_solicitor": bool(is_solicitor),
    }

    key = raw
    suffix_split = False
    for suf, flag in _SUFFIX_MODIFIERS:
        if not key.endswith(suf) or len(key) <= len(suf):
            continue
        parent = key[: -len(suf)]
        # Split only when the parent offence actually exists; otherwise the
        # suffix is part of the offence name, not a modifier.
        hit = conn.execute(
            "SELECT 1 FROM charge_legal_map WHERE charge_key=? LIMIT 1",
            (parent,),
        ).fetchone()
        if hit is not None:
            key = parent
            modifiers[flag] = True
            suffix_split = True
            break

    return NormalizedCharge(
        key=key,
        raw_key=raw,
        modifiers=modifiers,
        suffix_split_applied=suffix_split,
        bracket_prefix_stripped=bracket_prefix,
        quotes_stripped=quotes_stripped,
    )


# ---------- DB lookup ----------

_SELECT_ROW = (
    "SELECT id, charge_key, md_source_name, sg_category_id, "
    "statute_id, statute_name, article_no_num, article_branch, paragraph, "
    "sentence_kind_options, stat_imp_min_months, stat_imp_max_months, "
    "stat_fine_min_won, stat_fine_max_won, has_life, has_death, "
    "has_conditional_branch, branch_options, "
    "reference_mode, reference_multiplier, reference_articles, "
    "is_alias, alias_of, "
    # Two columns in this table are not read here. The `source` carried on a
    # penalty is provenance built at runtime, not that column.
    "also_in_categories, "
    "sanity_warnings, fine_formula, act_descriptor "
    "FROM charge_legal_map"
)


@dataclass
class LookupResult:
    # status: exact | exact_cross_cat | exact_same_cat_multi_row
    #         | exact_wrong_category | fuzzy_candidates | not_found
    status: str
    rows: list[sqlite3.Row] = field(default_factory=list)
    candidates: list[sqlite3.Row] = field(default_factory=list)


def _lookup_charge(
    conn: sqlite3.Connection,
    key: str,
    sg_category_id: int | None = None,
) -> LookupResult:
    """Look up a charge: exact first, then substring candidates on a miss.

    One charge key can exist under several sentencing categories. With no
    category given, every matching row comes back for the caller to choose
    between; the tool does not pick, because a wrong pick is invisible in the
    answer.

    With a category given but no row under it, an exact match in a *different*
    category is looked up separately, so "right charge, wrong category" is
    reported as that rather than as a fuzzy substring guess.
    """
    if not key:
        return LookupResult(status="not_found")

    # 1. Exact match, within the given category if one was specified.
    if sg_category_id is not None:
        rows = conn.execute(
            _SELECT_ROW + " WHERE charge_key=? AND sg_category_id=? "
            "ORDER BY statute_id, article_no_num, article_branch, paragraph",
            (key, sg_category_id),
        ).fetchall()
        if rows:
            if len(rows) == 1:
                return LookupResult(status="exact", rows=list(rows))
            # One category can still hold several provisions for a charge —
            # assault is in both the Criminal Act and the special act.
            return LookupResult(status="exact_same_cat_multi_row", rows=list(rows))

        # Not in that category: is it exact in another?
        other = conn.execute(
            _SELECT_ROW + " WHERE charge_key=? ORDER BY sg_category_id",
            (key,),
        ).fetchall()
        if other:
            return LookupResult(status="exact_wrong_category", rows=list(other))
    else:
        rows = conn.execute(
            _SELECT_ROW + " WHERE charge_key=? "
            "ORDER BY sg_category_id, statute_id, article_no_num, article_branch, paragraph",
            (key,),
        ).fetchall()
        if rows:
            if len(rows) == 1:
                return LookupResult(status="exact", rows=list(rows))
            cats = set(r["sg_category_id"] for r in rows)
            if len(cats) == 1:
                # Several rows in one category: the ambiguity is which
                # provision, not which category.
                return LookupResult(status="exact_same_cat_multi_row", rows=list(rows))
            return LookupResult(status="exact_cross_cat", rows=list(rows))

    # 2. substring fuzzy — '%KEY%' OR KEY LIKE '%cand%'
    fuzzy = conn.execute(
        _SELECT_ROW + " WHERE charge_key LIKE ? OR ? LIKE '%' || charge_key || '%' "
        "ORDER BY sg_category_id, charge_key LIMIT 10",
        (f"%{key}%", key),
    ).fetchall()

    if fuzzy:
        return LookupResult(status="fuzzy_candidates", candidates=list(fuzzy))

    return LookupResult(status="not_found")


def _resolve_alias(conn: sqlite3.Connection, row: sqlite3.Row) -> sqlite3.Row:
    """Follow an alias to its primary row when the alias carries no penalty.

    Of the 882 rows, some aliases carry their own figures — 공용서류손상 and
    공용서류무효 hold the same 0-84 month range — and some are null, depending
    on the primary instead (폭력행위등처벌에관한법률위반(상습공갈)). Only the
    null ones follow through.
    """
    if not row["is_alias"] or row["alias_of"] is None:
        return row
    if row["stat_imp_min_months"] is not None or row["stat_imp_max_months"] is not None:
        return row  # alias 자체에 stats 있음 — 그대로
    if row["reference_mode"] is not None or row["has_conditional_branch"]:
        return row  # 분기/참조 자체 정의 있음
    primary = conn.execute(
        _SELECT_ROW + " WHERE charge_key=? AND is_alias=0 LIMIT 1",
        (row["alias_of"],),
    ).fetchone()
    return primary if primary is not None else row


# ---------- resolving a provision to a statutory range ----------

# ---------- ceilings from the general part of the Criminal Act ----------
# These are constants in code, not data read from the corpus, and that is
# deliberate. They are universal rules rather than facts about a particular
# offence, and deriving them from the statute text would be fragile in three
# ways: the ceiling is stated in prose, not as a number; the date it took
# effect is not in the article text at all but in version metadata; and
# binding every calculation to the presence and exact wording of one row
# means the day that row changes, every sentence quietly comes out wrong.
# Which version applies is still derived from the offence date.
_IMP_CAP_MONTHS                = 360   # §42① 유기징역 단독 상한 (30년)
_IMP_CAP_MONTHS_OLD            = 180   # §42① 구법 단독 상한 (15년, <2010.10.16)
_AGGRAVATED_IMP_CAP_MONTHS     = 600   # §42② 가중 상한 (50년)
_AGGRAVATED_IMP_CAP_MONTHS_OLD = 300   # §42② 구법 가중 상한 (25년, <2010.10.16)
_DEATH_COMMUTE_MIN_MONTHS      = 240   # §55①1호 사형 감경 시 유기 하한 (20년)
_LIFE_COMMUTE_MIN_MONTHS       = 120   # §55①2호 무기 감경 시 유기 하한 (10년)
_ART42_REFORM_ISO = "20101016"         # §42 개정 시행일 (행위시법주의 §1① 경계)
# Ceiling under the current text -> (old law, new law), for offence-date correction.
_ART42_CAPS = {
    _IMP_CAP_MONTHS:            (_IMP_CAP_MONTHS_OLD, _IMP_CAP_MONTHS),
    _AGGRAVATED_IMP_CAP_MONTHS: (_AGGRAVATED_IMP_CAP_MONTHS_OLD, _AGGRAVATED_IMP_CAP_MONTHS),
}


@dataclass
class EffectivePenalty:
    """The statutory range this tool reports, after resolution.

    `source` is a one-line account of how the range was decided, shown in the
    response trace; `trace` holds the multi-line version — which branch was
    taken, which multipliers applied.

    `fine_formula` describes a fine stated as a formula rather than an amount,
    for rows whose numeric fine bounds are NULL::

      {"base_label": str, "low_mult": float, "high_mult": float,
       "is_optional": bool}

    Mitigation and aggravation then apply to the formula's multipliers, there
    being no amount to apply them to.
    """

    imp_min_months: int | None
    imp_max_months: int | None
    fine_min_won: int | None
    fine_max_won: int | None
    has_life: bool
    has_death: bool
    sentence_kind_options: list[str]
    source: str
    trace: list[str] = field(default_factory=list)
    fine_formula: dict | None = None


@dataclass
class PendingResolution:
    """Resolution stopped short: the caller has to choose.

    kind ∈ {branch, branch_invalid, reference, reference_missing, reference_ambiguous,
            modifier_directive}
    """

    kind: str
    options: list[dict] = field(default_factory=list)
    message: str = ""


def _penalty_from_row(row: sqlite3.Row, *, source: str = "direct") -> EffectivePenalty:
    """Build a penalty from a mapping row's statutory bounds."""
    try:
        kinds = json.loads(row["sentence_kind_options"] or "[]")
    except (TypeError, json.JSONDecodeError) as e:
        kinds = []
        if isinstance(e, json.JSONDecodeError):
            _log.warning("⚠ DATA: sentence_kind_options JSON 손상 (id=%s) — 형종 누락", _rid(row))
    # Some fines are expressed as a formula over a base amount rather than
    # as fixed bounds — a multiple of the sum involved, for instance.
    formula: dict | None = None
    try:
        raw = row["fine_formula"]
        if raw:
            formula = json.loads(raw)
    except (TypeError, json.JSONDecodeError, IndexError) as e:
        formula = None
        if isinstance(e, json.JSONDecodeError):
            _log.warning("⚠ DATA: fine_formula JSON 손상 (id=%s) — 벌금식 누락", _rid(row))
    return EffectivePenalty(
        imp_min_months=row["stat_imp_min_months"],
        imp_max_months=row["stat_imp_max_months"],
        fine_min_won=row["stat_fine_min_won"],
        fine_max_won=row["stat_fine_max_won"],
        has_life=bool(row["has_life"]),
        has_death=bool(row["has_death"]),
        sentence_kind_options=list(kinds),
        source=source,
        fine_formula=formula,
    )


def _penalty_inline(p: EffectivePenalty) -> str:
    """One-line summary for ambiguous choice lines."""
    parts: list[str] = []
    if p.imp_min_months is not None or p.imp_max_months is not None:
        parts.append("자유형 " + _span_months(p.imp_min_months, p.imp_max_months))
    if p.fine_min_won is not None or p.fine_max_won is not None:
        parts.append("벌금 " + _span_won(p.fine_min_won, p.fine_max_won))
    if p.has_life:
        parts.append("무기")
    if p.has_death:
        parts.append("사형")
    return " · ".join(parts) if parts else "정량 미상 (원범죄 호별 분기 등)"


# ---------- Branch key notation and matching ----------

_CIRCLED_DIGITS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def _parse_branch_key(key: object) -> tuple[int | None, str]:
    """Parse branch key into (paragraph_num, remainder).

    The remainder strips whitespace, '제', '호', and delimiters.
    Paragraph markers match circled digits, '1항', or '1-' (when followed by a tail).
    """
    s = re.sub(r"\s+", "", str(key or ""))
    para: int | None = None
    if s and s[0] in _CIRCLED_DIGITS:
        para, s = _CIRCLED_DIGITS.index(s[0]) + 1, s[1:]
    else:
        m = re.match(r"^제?(\d+)항(.*)$", s) or re.match(r"^(\d+)[-_·.](.+)$", s)
        if m:
            para, s = int(m.group(1)), m.group(2)
    rest = re.sub(r"[-_·.]", "", s)
    rest = re.sub(r"제(?=\d)", "", rest)
    rest = re.sub(r"(\d)(호|항)$", r"\1", rest)
    return para, rest


def _branch_key_label(key: object) -> str:
    """ASCII representation for menus and trace lines: ① -> 1, ①2 -> 1-2, ③-1 -> 3-1, 1호 -> 1."""
    para, rest = _parse_branch_key(key)
    if para is not None and rest:
        return f"{para}-{rest}"
    if para is not None:
        return str(para)
    return rest or str(key)


def _match_branch_option(branch_key: str, opts: list[dict]) -> dict | None:
    """Match caller's branch_key against menu options. Returns None if ambiguous or not found."""
    want = _parse_branch_key(branch_key)
    parsed = [(o, _parse_branch_key(o.get("key"))) for o in opts]
    exact = [o for o, p in parsed if p == want]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    para, rest = want
    if para is None and rest.isdigit():
        bare = [o for o, p in parsed if p == (int(rest), "")]
        if len(bare) == 1:
            return bare[0]
    menu_paras = {p[0] for _, p in parsed}
    if rest and (menu_paras == {None} or (para is None and len(menu_paras) == 1)):
        by_rest = [o for o, p in parsed if p[1] == rest]
        if len(by_rest) == 1:
            return by_rest[0]
    return None


def _penalty_from_branch(opt: dict, *, branch_key: str) -> EffectivePenalty:
    """Build a penalty from one branch option.

    An option that names its own sentence kinds is taken at its word. Deriving
    them from the figures instead went wrong in both directions. A fine stated
    as a formula has NULL bounds, so the fine option was dropped from the
    branch entirely; and a union across branches handed a fine option to
    branches offering imprisonment alone (청소년성보호법 §7 ① 강간,
    공직선거법 §230 ⑤).

    A branch's own `fine_formula` propagates to the penalty, since branches of
    one provision can state different formulas.

    With nothing named, the legacy path derives the kinds from the figures:
    NULL means the option is not offered.
    """
    if "sentence_kind_options" in opt:
        kinds = list(opt["sentence_kind_options"])
    else:
        kinds = []
        if opt.get("imp_min_months") is not None or opt.get("imp_max_months") is not None:
            kinds.append("imprisonment")
        if opt.get("fine_min_won") is not None or opt.get("fine_max_won") is not None:
            kinds.append("fine")
        if opt.get("has_life"):
            kinds.append("life")
        if opt.get("has_death"):
            kinds.append("death")
    return EffectivePenalty(
        imp_min_months=opt.get("imp_min_months"),
        imp_max_months=opt.get("imp_max_months"),
        fine_min_won=opt.get("fine_min_won"),
        fine_max_won=opt.get("fine_max_won"),
        has_life=bool(opt.get("has_life")),
        has_death=bool(opt.get("has_death")),
        sentence_kind_options=kinds,
        source=f"branch:{branch_key}",
        fine_formula=opt.get("fine_formula"),
    )


def _apply_multiplier(
    base: EffectivePenalty, multiplier: float
) -> EffectivePenalty:
    """Apply an aggravation or reference multiplier to the upper bound.

    형법 §42 ② caps an aggravated term at 50 years. Life and death flags carry
    through untouched — aggravation never creates one. A fine ceiling
    multiplier comes from wording like 형법 §55 ④ "다액의 2분의 1", which is a
    1.5. The multiplier applies as given; rounding is integer multiplication,
    with a ceiling where one is needed.
    """
    if multiplier == 1.0:
        return EffectivePenalty(
            imp_min_months=base.imp_min_months,
            imp_max_months=base.imp_max_months,
            fine_min_won=base.fine_min_won,
            fine_max_won=base.fine_max_won,
            has_life=base.has_life,
            has_death=base.has_death,
            sentence_kind_options=list(base.sentence_kind_options),
            source=base.source,
            trace=list(base.trace),
        )

    new_imp_max = base.imp_max_months
    if new_imp_max is not None:
        new_imp_max = int(new_imp_max * multiplier)
        new_imp_max = min(new_imp_max, _AGGRAVATED_IMP_CAP_MONTHS)

    new_fine_max = base.fine_max_won
    if new_fine_max is not None:
        new_fine_max = int(new_fine_max * multiplier)

    return EffectivePenalty(
        imp_min_months=base.imp_min_months,
        imp_max_months=new_imp_max,
        fine_min_won=base.fine_min_won,
        fine_max_won=new_fine_max,
        has_life=base.has_life,
        has_death=base.has_death,
        sentence_kind_options=list(base.sentence_kind_options),
        source=base.source,
        trace=list(base.trace),
    )


# ---------- matching a referenced provision ----------

# Accepts a provision written any of the usual ways: with or without a
# space, with or without a branch or a paragraph.
_REF_CHOICE_RE = re.compile(
    r"^\s*(?P<name>[^§\s]+?)\s*§?\s*"
    r"(?P<art>\d+)(?:의(?P<branch>\d+))?"
    r"(?:\s*(?P<para>[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]+|\d+))?\s*$"
)


def _parse_statute_choice(choice: str, rows: list[sqlite3.Row]) -> sqlite3.Row | None:
    """Match the caller's chosen provision against the candidate rows.

    The same form `_parse_reference_choice` accepts ('형법§257',
    '폭력행위등처벌에관한법률§2③'). Matching is on the space-normalised statute
    name and the article number, then optionally the branch and the paragraph;
    the most specific row wins.
    """
    if not choice or not rows:
        return None
    m = _REF_CHOICE_RE.match(choice)
    if not m:
        return None
    name = m.group("name").replace(" ", "")
    art = int(m.group("art"))
    branch = int(m.group("branch")) if m.group("branch") else 0
    para = (m.group("para") or "").strip() or None

    def score_row(row: sqlite3.Row) -> int:
        rn = (row["statute_name"] or "").replace(" ", "")
        if rn != name:
            return -1
        if row["article_no_num"] != art:
            return -1
        s = 1
        if (row["article_branch"] or 0) == branch:
            s += 1
        rp = row["paragraph"] or None
        if para is None and rp is None:
            s += 1
        elif para is not None and rp == para:
            s += 2
        return s

    scored = [(score_row(r), r) for r in rows]
    scored = [(s, r) for s, r in scored if s > 0]
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def _format_statute_choice_form(row: sqlite3.Row) -> str:
    """A row -> the choice form the response shows ('형법§257')."""
    name = (row["statute_name"] or "").replace(" ", "")
    art = row["article_no_num"]
    branch_str = f"의{row['article_branch']}" if row["article_branch"] else ""
    para_str = f" {row['paragraph']}" if row["paragraph"] else ""
    return f"{name}§{art}{branch_str}{para_str}".strip()


def _parse_reference_choice(choice: str, refs: list[dict]) -> dict | None:
    """Match the caller's chosen reference against the candidates.

    In order of precedence: the normalised statute name and the article number
    must agree; a matching branch beats that; a matching paragraph beats both.
    """
    if not choice or not refs:
        return None
    m = _REF_CHOICE_RE.match(choice)
    if not m:
        return None
    name = m.group("name").replace(" ", "")
    art = int(m.group("art"))
    branch = int(m.group("branch")) if m.group("branch") else 0
    para = (m.group("para") or "").strip() or None

    def score(ref: dict) -> int:
        rn = (ref.get("statute_name") or "").replace(" ", "")
        if rn != name:
            return -1
        if ref.get("article_no_num") != art:
            return -1
        s = 1
        if (ref.get("article_branch") or 0) == branch:
            s += 1
        rp = ref.get("paragraph") or None
        if para is None and rp is None:
            s += 1
        elif para is not None and rp == para:
            s += 2
        return s

    scored = [(score(r), r) for r in refs]
    scored = [(s, r) for s, r in scored if s > 0]
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def _auto_match_reference(charge_key: str, refs: list[dict]) -> dict | None:
    """Auto-select only when there is exactly one option.

    One option is not a choice, so taking it is deterministic. Two or more
    require reading the facts, which is a finding rather than a computation.

    Matching on the wording of the options was tried and removed. It works —
    habitual extortion does resolve to extortion — but it also fires on
    resemblances that are not identities, and the tool cannot tell the two
    apart. Asking costs a round trip; guessing wrong is invisible.
    """
    if not refs:
        return None
    if len(refs) == 1:
        return refs[0]
    return None


# ---------- ref → parent row fetch ----------

# Paragraphs are written as circled numerals in one table and as plain
# digits in another, so lookups try both forms.
_PARA_DIGIT_TO_CIRCLED = {
    "1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤",
    "6": "⑥", "7": "⑦", "8": "⑧", "9": "⑨", "10": "⑩",
    "11": "⑪", "12": "⑫", "13": "⑬", "14": "⑭", "15": "⑮",
}


def _normalize_paragraph(para: str | None) -> list[str | None]:
    """Paragraph spellings to try, in order."""
    if para is None or para == "":
        return [None]
    p = para.strip()
    candidates: list[str | None] = [p]
    if p in _PARA_DIGIT_TO_CIRCLED:
        candidates.append(_PARA_DIGIT_TO_CIRCLED[p])
    # And the other direction.
    for d, c in _PARA_DIGIT_TO_CIRCLED.items():
        if p == c:
            candidates.append(d)
            break
    return candidates


def _lookup_by_article(
    conn: sqlite3.Connection, ref: dict
) -> sqlite3.Row | None:
    """Resolve a reference to the row for the underlying offence.

    Matches the unique key (statute_id, article_no_num, article_branch,
    paragraph). A paragraph written as a digit ('1') and as a circled numeral
    ('①') are both tried, the two spellings coexisting in the data.

    The statute name matches after space normalisation, which absorbs the
    difference between how a mapping row and a reference write the same act
    ('특정경제범죄가중처벌등에관한법률' against '특정경제범죄 가중처벌 등에 관한
    법률').
    """
    name = (ref.get("statute_name") or "").strip()
    art = ref.get("article_no_num")
    branch = ref.get("article_branch") or 0
    para = ref.get("paragraph")

    if art is None or not name:
        return None

    name_norm = name.replace(" ", "")

    # Try both paragraph spellings.
    for cand_para in _normalize_paragraph(para):
        if cand_para is not None:
            rows = conn.execute(
                _SELECT_ROW
                + " WHERE REPLACE(statute_name, ' ', '')=? AND article_no_num=? AND article_branch=? "
                "AND paragraph=? AND is_alias=0 LIMIT 1",
                (name_norm, art, branch, cand_para),
            ).fetchall()
            if rows:
                return rows[0]

    # A paragraph was named but matched nothing: fall back to the article,
    # which covers the case where the mapping applies to the article whole.
    rows = conn.execute(
        _SELECT_ROW
        + " WHERE REPLACE(statute_name, ' ', '')=? AND article_no_num=? AND article_branch=? "
        "AND is_alias=0 "
        "ORDER BY CASE WHEN paragraph IS NULL THEN 0 ELSE 1 END, paragraph "
        "LIMIT 1",
        (name_norm, art, branch),
    ).fetchall()
    return rows[0] if rows else None


# ---------- payload resolver ----------

def _resolve_payload(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    branch_key: str | None = None,
    reference_choice: str | None = None,
) -> tuple[EffectivePenalty | None, PendingResolution | None]:
    """Resolve a row to a statutory range, or to the question blocking it.

    Returns ``(penalty, None)`` when resolution completed, or ``(None,
    pending)`` when the caller must supply a branch_key or a reference_choice
    first.
    """
    rm = row["reference_mode"]

    # A version row that already carries bounds has had the referenced
    # offence resolved as of that date. Looking the reference up again would
    # fetch today's bounds and overwrite the historical ones — undoing
    # exactly what made the row correct.
    if rm == "준용":
        has_quant = (
            row["stat_imp_min_months"] is not None
            or row["stat_imp_max_months"] is not None
            or row["stat_fine_min_won"] is not None
            or row["stat_fine_max_won"] is not None
            or row["has_life"]
            or row["has_death"]
        )
        if has_quant:
            rm = None

    # 1. The provision itself branches: different bounds per limb.
    if row["has_conditional_branch"]:
        try:
            opts = json.loads(row["branch_options"] or "[]")
        except (TypeError, json.JSONDecodeError):
            opts = []
        if branch_key is None:
            return None, PendingResolution(
                kind="branch",
                options=opts,
                message="branch_key 인자로 옵션 선택 필요.",
            )
        chosen = _match_branch_option(branch_key, opts)
        if chosen is None:
            return None, PendingResolution(
                kind="branch_invalid",
                options=opts,
                message=(f"branch_key={branch_key!r} 는 아래 목록에 없습니다 — "
                         "목록의 키를 문자열 하나로 그대로 넣으세요."),
            )
        penalty = _penalty_from_branch(chosen, branch_key=chosen.get("key", branch_key))
        # Each branch states its own sentence kinds, so they are taken as
        # written rather than unioned with the row's — unioning gave a fine
        # option to branches that carry only imprisonment. A fine formula is
        # often shared at row level, so a branch without one falls back.
        if penalty.fine_formula is None:
            try:
                raw_formula = row["fine_formula"]
                if raw_formula:
                    penalty.fine_formula = json.loads(raw_formula)
            except (TypeError, json.JSONDecodeError, IndexError):
                pass
        penalty.trace.append(
            f"## branch_key 선택: {_branch_key_label(chosen.get('key', branch_key))} — "
            f"{chosen.get('cond', '')[:80]}"
        )
        return penalty, None

    # 2. The provision aggravates or adopts another offence's penalty.
    if rm in ("가중", "준용", "공동가중"):
        try:
            refs = json.loads(row["reference_articles"] or "[]")
        except (TypeError, json.JSONDecodeError):
            refs = []

        # An open-ended aggravation: the provision aggravates *any* offence
        # of some class — a mandatory reporter committing any sexual offence,
        # say — so there is no list of underlying offences to enumerate. The
        # row therefore carries a directive rather than a reference, and the
        # caller is told to look up the actual offence and attach the
        # modifier. No new arithmetic: it reuses the ordinary aggravation.
        if len(refs) == 1 and isinstance(refs[0], dict) and refs[0].get("modifier_kind"):
            return None, PendingResolution(
                kind="modifier_directive",
                options=refs,
                message=refs[0].get("note", ""),
            )

        chosen_ref: dict | None = None
        auto_matched = False
        if reference_choice:
            chosen_ref = _parse_reference_choice(reference_choice, refs)
            if chosen_ref is None:
                return None, PendingResolution(
                    kind="reference_invalid",
                    options=refs,
                    message=(
                        f"reference_choice={reference_choice!r} 파싱 또는 매칭 실패. "
                        "options 중 statute§article form 으로 재호출."
                    ),
                )
        else:
            chosen_ref = _auto_match_reference(row["charge_key"], refs)
            auto_matched = chosen_ref is not None

        if chosen_ref is None:
            return None, PendingResolution(
                kind="reference",
                options=refs,
                message=(
                    f"reference_mode={rm} — 원범죄 자동 매칭 실패. "
                    "reference_choice 인자로 명시 선택."
                ),
            )

        parent = _lookup_by_article(conn, chosen_ref)
        if parent is None:
            return None, PendingResolution(
                kind="reference_missing",
                options=[chosen_ref],
                message=(
                    f"매칭된 원범죄 row 가 매핑 테이블에 없음 — "
                    f"{chosen_ref.get('statute_name')}§{chosen_ref.get('article_no_num')}."
                ),
            )

        mult = row["reference_multiplier"]
        mult_missing = mult is None and rm in ("가중", "공동가중")
        if mult_missing:
            _log.warning("⚠ DATA: %s row (id=%s) reference_multiplier 누락 — ×1.0 미적용 위험",
                         rm, _rid(row))
        mult = mult or 1.0
        match_kind = "auto" if auto_matched else "explicit"
        ref_label = (
            f"{chosen_ref.get('statute_name')} §{chosen_ref.get('article_no_num')}"
        )
        if chosen_ref.get("article_branch"):
            ref_label += f"의{chosen_ref['article_branch']}"
        if chosen_ref.get("paragraph"):
            ref_label += f" {chosen_ref['paragraph']}"
        note = chosen_ref.get("note", "")

        # When the referenced offence itself branches per subparagraph, each
        # branch has its own penalty, and the aggravation applies to the one
        # actually committed. Collapsing them into an envelope would import
        # the harshest branch's ceiling into every case — so the branch is
        # asked for here exactly as it would be for a direct provision.
        branch_note = ""
        if parent["has_conditional_branch"]:
            try:
                p_opts = json.loads(parent["branch_options"] or "[]")
            except (TypeError, json.JSONDecodeError):
                p_opts = []
            if branch_key is None:
                return None, PendingResolution(
                    kind="branch",
                    options=p_opts,
                    message=(
                        f"원범죄 {ref_label} 이 호별 분기 — 행위가 해당하는 호를 branch_key "
                        f"로 선택하면 그 호의 법정형에 ×{mult} ({rm}) 적용. 호마다 형 상이."
                    ),
                )
            chosen_b = next((o for o in p_opts if o.get("key") == branch_key), None)
            if chosen_b is None:
                return None, PendingResolution(
                    kind="branch_invalid",
                    options=p_opts,
                    message=f"branch_key={branch_key!r} 는 원범죄 {ref_label} 의 호가 아님.",
                )
            base = _penalty_from_branch(chosen_b, branch_key=branch_key)
            branch_note = f" [{branch_key} 호: {chosen_b.get('cond', '')[:40]}]"
        else:
            base = _penalty_from_row(parent, source=f"reference_{rm}")
        adjusted = _apply_multiplier(base, mult)

        adjusted.source = f"reference_{rm}:{ref_label}{branch_note}"
        adjusted.trace = [
            f"## reference_mode: {rm} (multiplier={mult})",
            f"## 원범죄 ({match_kind} match): {ref_label}"
            + (f" — {note}" if note else ""),
            f"## 원형: {_penalty_inline(base)}{branch_note}",
        ]
        if mult_missing:
            adjusted.trace.append(
                "⚠ DATA: reference_multiplier 누락 — 가중 미적용(×1.0). 데이터 검증 필요.")
        return adjusted, None

    # 3. The provision adopts another's penalty unchanged, no multiplier.
    if rm == "분기":
        try:
            refs = json.loads(row["reference_articles"] or "[]")
        except (TypeError, json.JSONDecodeError):
            refs = []
        chosen_ref: dict | None = None
        auto_matched = False
        if reference_choice:
            chosen_ref = _parse_reference_choice(reference_choice, refs)
            if chosen_ref is None:
                return None, PendingResolution(
                    kind="reference_invalid",
                    options=refs,
                    message=(
                        f"reference_choice={reference_choice!r} 매칭 실패. "
                        "options 중 statute§article form 으로 재호출."
                    ),
                )
        else:
            # Which of the referenced offences applies depends on what was
            # done, which only the facts say. Matching on wording tends to
            # land on the harshest option, so anything beyond a single
            # candidate goes back to the caller.
            if len(refs) == 1:
                chosen_ref = refs[0]
                auto_matched = True
        if chosen_ref is None:
            return None, PendingResolution(
                kind="reference",
                options=refs,
                message="reference_mode=분기 — reference_choice 인자로 명시 선택.",
            )

        parent = _lookup_by_article(conn, chosen_ref)
        if parent is None:
            return None, PendingResolution(
                kind="reference_missing",
                options=[chosen_ref],
                message=(
                    f"매칭된 원범죄 row 가 매핑 테이블에 없음 — "
                    f"{chosen_ref.get('statute_name')}§{chosen_ref.get('article_no_num')}."
                ),
            )

        penalty = _penalty_from_row(parent, source="reference_분기")
        ref_label = (
            f"{chosen_ref.get('statute_name')} §{chosen_ref.get('article_no_num')}"
        )
        if chosen_ref.get("article_branch"):
            ref_label += f"의{chosen_ref['article_branch']}"
        note = chosen_ref.get("note", "")
        penalty.trace = [
            "## reference_mode: 분기 (전2조의 예에 의함)",
            f"## 선택 원범죄: {ref_label}"
            + (f" — {note}" if note else ""),
        ]
        return penalty, None

    # 4. Nothing to resolve: use the row's own bounds.
    penalty = _penalty_from_row(row, source="direct" if rm in (None, "정보") else f"direct_{rm}")
    if rm not in _DIRECT_FALLTHROUGH_MODES:
        # An unrecognised mode landing here would emit unadjusted bounds
        # that look correct, so mark it in the trace.
        msg = (f"⚠ DATA: 미지 reference_mode={rm!r} (id={_rid(row)}) — 알려진 값 아님. "
               "direct 처리했으나 가중/준용 누락 가능, 데이터 검증 필요.")
        penalty.trace.append(msg)
        _log.warning(msg)
    return penalty, None


# ---------- the processed range: statutory adjustments ----------

# The order the Criminal Act prescribes for applying adjustments. Order
# matters: the same set applied in a different sequence gives a different
# range, because each one operates on the result of the last.
_MOD_ORDER: dict[str, int] = {
    "본조_가중": 1,
    "특수교사방조_가중": 2,
    "누범_가중": 3,
    "법률상_필요감경": 4,
    "법률상_임의감경": 4,    # 법률상 감경은 함께 처리 (§56 ④)
    "경합범_가중": 5,        # §56 ⑤ — §37 전단·§38 ① 2호. L1·L3 에서 act_count/additional_charges 로 트리거.
    "작량감경": 6,           # §56 ⑥
}

# enum kind → (imp_max_mult, fine_max_mult, fine_min_mult, imp_min_mult).
# Repeat-offence aggravation doubles the custodial maximum and leaves fines
# alone. Multiple-offence aggravation raises the heaviest offence's maximum
# by half, again leaving fines and the lower bound untouched. Other
# adjustments move custodial terms and fines together.
_MOD_MULT: dict[str, tuple[float, float, float, float]] = {
    "본조_가중":         (1.5, 1.5, 1.0, 1.0),  # 형법 §42 ② cap 50년
    "특수교사방조_가중": (1.5, 1.5, 1.0, 1.0),
    "누범_가중":         (2.0, 1.0, 1.0, 1.0),  # §35 — 자유형 장기만
    "법률상_필요감경":   (0.5, 0.5, 0.5, 0.5),  # §55 ①
    "법률상_임의감경":   (0.5, 0.5, 0.5, 0.5),
    "경합범_가중":       (1.5, 1.0, 1.0, 1.0),  # §38 ① 2호 — 자유형 장기 1/2 가중. cap §42 ② 50년.
    "작량감경":          (0.5, 0.5, 0.5, 0.5),
}


def _kind_is_mit(kind: str) -> bool:
    return kind in ("법률상_필요감경", "법률상_임의감경", "작량감경")


def _kind_is_agg(kind: str) -> bool:
    return kind in ("본조_가중", "특수교사방조_가중", "누범_가중", "경합범_가중")


@dataclass
class ProcessedPenalty:
    """The range after statutory adjustments have been applied."""

    imp_min_months: int | None
    imp_max_months: int | None
    fine_min_won: int | None
    fine_max_won: int | None
    has_life: bool
    has_death: bool
    sentence_kind_options: list[str]
    trace: list[str] = field(default_factory=list)
    fine_formula: dict | None = None


def _expand_implicit_modifications(
    norm: NormalizedCharge, mods: list[dict] | None, act_count: int = 1
) -> list[dict]:
    """Fold implied adjustments into the list the caller supplied.

    Rules:
      - is_attempted (미수, §25) adds 법률상_임의감경, unless the caller already
        listed one. It is added with **applied=False**: §25 ② reads "감경할 수
        있다", so whether to mitigate is a judgment call and not the tool's.
        The trace shows it as available while the figures stay at the
        completed-offence range — no automatic halving. A caller stating
        applied=True gets it applied.
      - is_accessory (방조, §32) adds 법률상_필요감경 with applied=True. §32 ②
        reads "감경한다", which is mandatory, so applying it is correct.
      - is_solicitor (교사, §31 ①) carries the same punishment as the
        principal: no adjustment, trace only.
      - act_count >= 2 adds 경합범_가중 under §37, unless the caller listed one
        already. applied=False overrides it, which practice occasionally calls
        for. A 포괄일죄 — one continuous course of conduct — is act_count=1.

    Deriving the concurrence adjustment from act_count is deliberate, and
    removing it was tried and reverted: without it a model filled the tool's
    schema in the prompt's aggravator format instead, omitting `kind` and
    inventing `statutes` keys — all three retries of one case failed on the
    schema. The automatic trigger stays until prompt and schema agree.
    """
    out = list(mods or [])
    has_attempt = any("미수" in (m.get("type", "") or "") for m in out)
    has_access = any("방조" in (m.get("type", "") or "") for m in out)
    has_concurrence = any(m.get("kind") == "경합범_가중" for m in out)

    if norm.modifiers.get("is_attempted") and not has_attempt:
        out.append({
            "kind": "법률상_임의감경",
            "type": "미수 (§25)",
            "basis": "형법 §25 ②",
            # Discretionary: whether an attempt is mitigated is the court's
            # to decide, so it is offered rather than applied.
            "applied": False,
            "source": "auto_from_suffix",
        })
    if norm.modifiers.get("is_accessory") and not has_access:
        out.append({
            "kind": "법률상_필요감경",
            "type": "방조 (§32)",
            "basis": "형법 §32 ②",
            "applied": True,
            "source": "auto_from_suffix",
        })
    if act_count >= 2 and not has_concurrence:
        out.append({
            "kind": "경합범_가중",
            "type": f"동종 §37 전단 (act_count={act_count})",
            "basis": "형법 §37 전단, §38 ① 2호",
            "applied": True,
            "source": "auto_from_act_count",
        })
    return out


def _apply_single_modification(
    penalty: ProcessedPenalty, mod: dict,
    agg_cap_months: int = _AGGRAVATED_IMP_CAP_MONTHS,
) -> tuple[ProcessedPenalty, str]:
    """Apply one adjustment, returning the new range and a trace line.

    The caller guarantees the kind exists in `_MOD_MULT`; the INVALID guard in
    `_apply_statutory_modifications` is what does that.

    Where the fine is a formula rather than an amount, the multiplier applies
    to the formula's low_mult and high_mult. No amount is computed — the base
    a formula multiplies is not an input to this tool — so only the multipliers
    move.
    """
    kind = mod.get("kind", "")
    type_label = mod.get("type", "")
    imp_max_m, fine_max_m, fine_min_m, imp_min_m = _MOD_MULT[kind]

    new_imp_min = penalty.imp_min_months
    new_imp_max = penalty.imp_max_months
    new_fine_min = penalty.fine_min_won
    new_fine_max = penalty.fine_max_won
    new_has_life = penalty.has_life
    new_has_death = penalty.has_death
    new_fine_formula = dict(penalty.fine_formula) if penalty.fine_formula else None

    if _kind_is_agg(kind):
        # Aggravation raises the ceiling only, subject to the overall cap.
        if new_imp_max is not None:
            new_imp_max = min(int(new_imp_max * imp_max_m), agg_cap_months)
        if new_fine_max is not None:
            new_fine_max = int(new_fine_max * fine_max_m)
        # For a formula-based fine, only the upper multiplier moves.
        if new_fine_formula and "high_mult" in new_fine_formula:
            new_fine_formula["high_mult"] = new_fine_formula["high_mult"] * fine_max_m
    elif _kind_is_mit(kind):
        # Mitigation applies to each sentence kind independently, and the
        # processed range is the union of the results. Where death, life and
        # a term of years are all available, mitigating only one of them
        # discards the others — and the intersection with the guideline
        # range then comes out empty, which reads as "no lawful sentence"
        # when in fact there are several.
        #   death -> life, or 240-600 months
        #   life  -> 120-600 months
        #   term  -> halved
        cand_mins: list[int] = []
        cand_maxs: list[int] = []
        out_has_life = False
        out_has_death = False
        if new_has_death:
            # Death mitigated: life, or 20 to 50 years.
            out_has_life = True
            cand_mins.append(_DEATH_COMMUTE_MIN_MONTHS)
            cand_maxs.append(_AGGRAVATED_IMP_CAP_MONTHS)
        if new_has_life:
            # Life mitigated: 10 to 50 years; life itself is no longer available.
            cand_mins.append(_LIFE_COMMUTE_MIN_MONTHS)
            cand_maxs.append(_AGGRAVATED_IMP_CAP_MONTHS)
        if new_imp_min is not None or new_imp_max is not None:
            # A term of years is halved.
            if new_imp_min is not None:
                cand_mins.append(max(int(new_imp_min * imp_min_m), 1))
            if new_imp_max is not None:
                cand_maxs.append(int(new_imp_max * imp_max_m))
        new_imp_min = min(cand_mins) if cand_mins else None
        new_imp_max = max(cand_maxs) if cand_maxs else None
        new_has_life = out_has_life
        new_has_death = out_has_death
        if new_fine_min is not None:
            new_fine_min = int(new_fine_min * fine_min_m)
        if new_fine_max is not None:
            new_fine_max = int(new_fine_max * fine_max_m)
        # Mitigating a formula-based fine. Strictly the Act halves only the
        # maximum, but both multipliers are scaled the same way the fixed
        # bounds are, so the two representations agree.
        if new_fine_formula:
            if "high_mult" in new_fine_formula:
                new_fine_formula["high_mult"] = new_fine_formula["high_mult"] * fine_max_m
            if "low_mult" in new_fine_formula:
                new_fine_formula["low_mult"] = new_fine_formula["low_mult"] * fine_min_m

    new_penalty = ProcessedPenalty(
        imp_min_months=new_imp_min,
        imp_max_months=new_imp_max,
        fine_min_won=new_fine_min,
        fine_max_won=new_fine_max,
        has_life=new_has_life,
        has_death=new_has_death,
        sentence_kind_options=list(penalty.sentence_kind_options),
        trace=list(penalty.trace),
        fine_formula=new_fine_formula,
    )

    src = mod.get("source", "")
    src_suffix = f" [{src}]" if src else ""
    imp_str = (
        "자유형 " + _span_months(new_imp_min, new_imp_max)
        if (new_imp_min is not None or new_imp_max is not None) else "자유형 -"
    )
    # Show the fine in the trace only where it actually moved.
    if new_fine_max is not None:
        fine_str = f"벌금 {new_fine_max:,}원 이하"
    elif new_fine_formula:
        fine_str = f"벌금 {_format_fine_formula(new_fine_formula)}"
    else:
        fine_str = ""
    bits = [imp_str]
    if fine_str:
        bits.append(fine_str)
    if new_has_life:
        bits.append("무기")
    if new_has_death:
        bits.append("사형")
    suffix = " · ".join(bits)
    trace_line = f"- {kind} / {type_label}: 적용{src_suffix} → {suffix}"
    return new_penalty, trace_line


def _format_fine_formula(formula: dict) -> str:
    """One-line rendering of a formula-based fine.

    {"base_label": "부가세액", "low_mult": 2, "high_mult": 5}
        -> "부가세액의 2배 ~ 5배"
    {"base_label": "이득액", "low_mult": 0, "high_mult": 1, "is_optional": True}
        -> "이득액 이하 (임의 병과)"
    """
    base = formula.get("base_label", "?")
    low = formula.get("low_mult")
    high = formula.get("high_mult")
    optional_suffix = " (임의 병과)" if formula.get("is_optional") else ""

    def _fmt(v: float | int) -> str:
        if v is None:
            return "?"
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        return f"{v}"

    if low is None and high is None:
        return f"{base}{optional_suffix}"
    if low == high:
        return f"{base}의 {_fmt(low)}배{optional_suffix}"
    if low == 0:
        return f"{base}의 {_fmt(high)}배 이하{optional_suffix}"
    return f"{base}의 {_fmt(low)}배 ~ {_fmt(high)}배{optional_suffix}"


def _apply_statutory_modifications(
    base: EffectivePenalty,
    norm: NormalizedCharge,
    mods: list[dict] | None,
    act_count: int = 1,
    offense_iso: str | None = None,
) -> ProcessedPenalty:
    """Statutory range plus adjustments -> the processed range.

    형법 §56 순서 (본조가중 → 특수교사방조 → 누범 → 법률상감경 → 경합범가중 → 작량감경).
    Only adjustments marked as applied change the range; the rest appear in
    the trace so the caller can see they were considered. Multiple offences
    add their aggravation automatically, unless the caller stated it.

    ``offense_iso`` selects which ceiling on a term of years applies: the
    lower one for offences before the amendment took effect, the higher one
    after. Absent a date, the current ceiling is used.
    """
    # Correct the post-aggravation ceiling for the law as it stood.
    agg_cap = (_AGGRAVATED_IMP_CAP_MONTHS_OLD
               if (offense_iso and offense_iso < _ART42_REFORM_ISO)
               else _AGGRAVATED_IMP_CAP_MONTHS)
    penalty = ProcessedPenalty(
        imp_min_months=base.imp_min_months,
        imp_max_months=base.imp_max_months,
        fine_min_won=base.fine_min_won,
        fine_max_won=base.fine_max_won,
        has_life=base.has_life,
        has_death=base.has_death,
        sentence_kind_options=list(base.sentence_kind_options),
        trace=[],
        fine_formula=dict(base.fine_formula) if base.fine_formula else None,
    )

    all_mods = _expand_implicit_modifications(norm, mods, act_count=act_count)
    if not all_mods:
        penalty.trace.append("- (적용 가중·감경 없음)")
        return penalty

    # Sort into the order the Act prescribes.
    ordered = sorted(all_mods, key=lambda m: _MOD_ORDER.get(m.get("kind", ""), 99))

    seen: set[tuple[str, str]] = set()
    for mod in ordered:
        kind = mod.get("kind", "")
        type_label = mod.get("type", "")
        # Reject unknown kinds rather than skipping them silently.
        if kind not in _MOD_MULT:
            penalty.trace.append(
                f"- {kind!r} / {type_label}: 허용되지 않는 kind — 무시 "
                f"(허용: {sorted(_MOD_MULT)})"
            )
            continue
        # Reject a repeated (kind, type): the same ground supplied twice
        # would compound, halving the range twice over.
        key = (kind, type_label)
        if key in seen:
            penalty.trace.append(f"- {kind} / {type_label}: 중복 — 무시")
            continue
        seen.add(key)
        # A ground named at all is normally a ground applied, so an absent
        # flag counts as applied — the alternative silently drops it. An
        # explicit flag must be true; anything else, including an ambiguous
        # value, does not apply.
        if "applied" not in mod:
            mod = {**mod, "applied": True}  # normalize for trace
        applied_val = mod["applied"]
        if applied_val is not True:
            penalty.trace.append(
                f"- {kind} / {type_label}: applied={applied_val} → 미적용 (주장만 기록)"
            )
            continue
        penalty, line = _apply_single_modification(penalty, mod, agg_cap_months=agg_cap)
        penalty.trace.append(line)

    return penalty


def _penalty_kv_lines(
    imp_lo: int | None,
    imp_hi: int | None,
    fine_lo: int | None,
    fine_hi: int | None,
    fine_formula: dict | None,
    has_life: bool,
    has_death: bool,
    kind_options: list[str] | None,
) -> list[str]:
    """Shared formatting for statutory and processed penalty blocks."""
    out: list[str] = []
    has_imp = imp_lo is not None or imp_hi is not None
    has_fine_num = fine_lo is not None or fine_hi is not None
    kinds = _kinds_line(
        kind_options,
        has_imp=has_imp,
        has_fine=has_fine_num or bool(fine_formula),
        has_life=has_life,
        has_death=has_death,
    )
    if kinds:
        out.append(kinds)
    if has_imp:
        out.append(f"- 자유형: {_span_months(imp_lo, imp_hi)}")
    if has_fine_num:
        if fine_lo is None:
            out.append(f"- 벌금: {_span_won(50_000, fine_hi)} (하한은 형법 §45 최저액)")
        else:
            out.append(f"- 벌금: {_span_won(fine_lo, fine_hi)}")
    elif fine_formula:
        out.append(f"- 벌금: {_format_fine_formula(fine_formula)}")
    return out


def _format_processed_penalty_lines(p: ProcessedPenalty) -> list[str]:
    return _penalty_kv_lines(
        p.imp_min_months,
        p.imp_max_months,
        p.fine_min_won,
        p.fine_max_won,
        p.fine_formula,
        p.has_life,
        p.has_death,
        p.sentence_kind_options,
    )


# ---------- the recommended range: guideline leaf and factors ----------


def _convert_factors_to_applied(
    guideline_factors: dict | None,
) -> list[AppliedFactor]:
    """Turn the caller's factor lists into applied-factor records.

    Input shape::

      {
        "special_act_aggravators": ["text", ...],
        "special_act_mitigators":  [...],
        "special_actor_aggravators": [...],
        "special_actor_mitigators":  [...],
      }

    The text passes through as the caller wrote it and is not verified: how
    accurately a factor was classified is measured against gold labels in
    evaluation, not here. This counts by the (kind, direction) the dict key
    implies, and the text is for the trace.
    """
    if not guideline_factors:
        return []
    out: list[AppliedFactor] = []
    mapping = [
        ("special_act_aggravators", "행위", "가중"),
        ("special_act_mitigators", "행위", "감경"),
        ("special_actor_aggravators", "행위자_기타", "가중"),
        ("special_actor_mitigators", "행위자_기타", "감경"),
    ]
    for key, kind, direction in mapping:
        items = guideline_factors.get(key) or []
        if not isinstance(items, list):
            continue
        for t in items:
            out.append(AppliedFactor(
                scope="특별", kind=kind, direction=direction, text=str(t),
            ))
    return out


def _extract_fine_paragraphs(
    conn: sqlite3.Connection,
    statute_id: int | None,
    article_no_num: int,
    article_branch: int | None,
) -> list[str]:
    """Extract the paragraphs of the provision that mention a fine, verbatim.

    Where a fine is a formula rather than an amount — 특가법 §8의2 sets it at
    two to five times the tax evaded — the numeric bounds are NULL, and the
    statutory range reaches the caller only as the provision's own words. The
    split follows the standard paragraph marks (① ② ③). The text is the
    article as loaded, not a summary of it.

    Which multiple within that range applies is a judgment call: the
    guidelines define no fine recommendation for these categories. The tool
    states the statutory range and stops there.
    """
    if statute_id is None:
        return []
    if article_branch and article_branch != 0:
        row = conn.execute(
            "SELECT article_text FROM st_articles "
            "WHERE statute_id=? AND article_no_num=? AND article_branch=?",
            (statute_id, article_no_num, article_branch),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT article_text FROM st_articles "
            "WHERE statute_id=? AND article_no_num=? AND article_branch IS NULL",
            (statute_id, article_no_num),
        ).fetchone()
    if row is None or not row[0]:
        return []
    text = row[0]
    paragraphs = re.split(r"(?=[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])", text)
    return [p.strip() for p in paragraphs if p.strip() and "벌금" in p]


def _get_statute_floor(processed: "ProcessedPenalty") -> int | None:
    """The processed minimum, which floors the guideline range.

    The guidelines say that where a recommendation falls outside the processed
    range, the processed range governs. So the floor is the *processed*
    minimum — after 자수, 미수, 심신미약, 작량감경 and the rest — not the
    statutory one.

    Using the statutory floor was wrong in a measurable way. For a murder with
    자수 the processed range is [30, 600] while the statutory floor is 60, and
    forcing 60 deleted the [42, 60) part of a recommended [42, 144]: the very
    outcome the rule above forbids.

    None where the processed range carries no term of imprisonment, a
    fine-only leaf, which tells `determine_range` to skip the correction.
    """
    return processed.imp_min_months


def _intersect_with_processed(
    rec: RecommendedRange | None,
    processed: ProcessedPenalty,
) -> tuple[int | None, int | None]:
    """Where the processed and recommended ranges overlap: what may be imposed.

    With no recommendation the processed range stands as it is, and a
    recommendation carrying life is unbounded above.
    """
    if rec is None:
        return processed.imp_min_months, processed.imp_max_months

    proc_min = processed.imp_min_months or 0
    proc_max = processed.imp_max_months  # None 가능

    rec_min = rec.min_months or 0
    rec_max = rec.max_months  # None 가능 (unbounded)

    lo = max(proc_min, rec_min)
    if proc_max is None and rec_max is None:
        hi = None
    elif proc_max is None:
        hi = rec_max
    elif rec_max is None:
        hi = proc_max
    else:
        hi = min(proc_max, rec_max)
    return lo, hi


# ---------- verifying a proposed sentence ----------


def _verify_sentence(
    sentence_months: int | None,
    fine_amount: int | None,
    processed: ProcessedPenalty,
    rec: RecommendedRange | None,
    intersect: tuple[int | None, int | None],
) -> list[str]:
    """Check a proposed sentence against the ranges and locate it within them."""
    lines: list[str] = []
    if sentence_months is None and fine_amount is None:
        lines.append("- 선고형 미지정")
        return lines

    if sentence_months is not None:
        lines.append(f"- 선고형(자유형): {sentence_months}월")
        lo, hi = intersect
        in_intersect = True
        if lo is not None and sentence_months < lo:
            in_intersect = False
        if hi is not None and sentence_months > hi:
            in_intersect = False
        lines.append(f"- 선고 가능 범위({_span_months(lo, hi)}) 안: {in_intersect}")

        if rec is not None:
            ok = in_range(sentence_months, rec)
            pos = within_range_position(sentence_months, rec)
            lines.append(f"- 권고 영역({_span_months(rec.min_months, rec.max_months)}) 안: {ok}")
            if pos is not None:
                lines.append(f"- 영역 내 위치: {pos:.3f}")

        # Against the processed range.
        p_lo = processed.imp_min_months or 0
        p_hi = processed.imp_max_months
        in_proc = sentence_months >= p_lo and (p_hi is None or sentence_months <= p_hi)
        lines.append(f"- 처단형({_span_months(p_lo, p_hi)}) 안: {in_proc}")

    if fine_amount is not None:
        lines.append(f"- 선고형(벌금): {fine_amount:,}원")
        f_lo = processed.fine_min_won or 0
        f_hi = processed.fine_max_won
        in_fine = fine_amount >= f_lo and (f_hi is None or fine_amount <= f_hi)
        lines.append(f"- 벌금 처단형({_span_won(f_lo, f_hi)}) 안: {in_fine}")
    return lines


# ---------- suspended sentences ----------
# A sentence can be suspended if it is three years or less of imprisonment,
# or a fine at or below the statutory threshold. Eligibility is arithmetic;
# whether to suspend is not, and the guidelines answer it with a rule over
# four groups of factors — see the recommendation function below.
_PROBATION_IMP_CAP_MONTHS = 36
_PROBATION_FINE_CAP_WON = 5_000_000


def _probation_recommendation(
    sentence_months: int | None,
    fine_amount: int | None,
    processed: ProcessedPenalty,
    probation_factors: dict | None,
) -> list[str]:
    """Suspension: eligibility, then the guideline rule over the factor groups.

    `probation_factors`::

      {"major_positive": [...], "major_negative": [...],
       "general_positive": [...], "general_negative": [...]}

    Eligibility turns on the sentence actually imposed, not on the range:
      - imprisonment of 36 months or less   -> available (§62 ①)
      - a fine of 5,000,000 won or less     -> available (§62 ①, from
        2018-01-07)
      - both imposed, each within its cap   -> §62 ② allows suspending one or
        both
    Life and death appear in the processed range as options rather than as a
    sentence imposed, and do not enter this.
    """
    lines: list[str] = []

    has_imp = sentence_months is not None
    has_fine = fine_amount is not None
    if not has_imp and not has_fine:
        lines.append("- 집행유예 판단: 선고형(sentence_months·fine_amount) 미지정 — 건너뜀")
        return lines

    imp_ok = has_imp and sentence_months <= _PROBATION_IMP_CAP_MONTHS
    fine_ok = has_fine and fine_amount <= _PROBATION_FINE_CAP_WON

    parts: list[str] = []
    if has_imp:
        parts.append(
            f"자유형 {sentence_months}월 ≤ 36월" if imp_ok
            else f"자유형 {sentence_months}월 > 36월 상한"
        )
    if has_fine:
        parts.append(
            f"벌금 {fine_amount:,}원 ≤ 5,000,000원 (§62 ① 2018.1.7+)" if fine_ok
            else f"벌금 {fine_amount:,}원 > 5,000,000원 상한"
        )

    eligible = imp_ok or fine_ok
    status_text = "가능" if eligible else "불가"
    lines.append(f"- 집행유예: {status_text} ({'; '.join(parts)})")

    if has_imp and has_fine and imp_ok and fine_ok:
        lines.append("- 병과: 형의 일부(자유형 또는 벌금)만 집행유예 가능 (§62 ②)")

    if not eligible:
        return lines

    if not probation_factors:
        lines.append("- probation_factors 미지정 — 4분면 룰 적용 불가")
        return lines

    mp = len(probation_factors.get("major_positive") or [])
    mn = len(probation_factors.get("major_negative") or [])
    gp = len(probation_factors.get("general_positive") or [])
    gn = len(probation_factors.get("general_negative") or [])
    lines.append(f"- major: positive={mp} / negative={mn}")
    lines.append(f"- general: positive={gp} / negative={gn}")

    # Two major factors either way, or a margin of two, decides it. When
    # both rules fire, or neither does, the ordinary factors are compared
    # and the decision is left to discretion.
    rule1 = (mp >= 2) or (mp - mn >= 2)
    rule2 = (mn >= 2) or (mn - mp >= 2)
    if rule1 and rule2:
        verdict = (
            f"집유 *재량* (룰 3: 룰 1·2 동시 충족 — major +{mp}/-{mn}, 판사 재량)"
        )
    elif rule1:
        verdict = f"집유 권고 (룰 1: major +{mp}/-{mn})"
    elif rule2:
        verdict = f"집유 불권고 (룰 2: major +{mp}/-{mn})"
    else:
        # Neither major rule applies: compare the ordinary factors.
        if gp > gn:
            verdict = f"집유 권고 (룰 3: general +{gp}/-{gn})"
        elif gp < gn:
            verdict = f"집유 불권고 (룰 3: general +{gp}/-{gn})"
        else:
            verdict = (
                f"집유 *경계* (룰 3: major +{mp}/-{mn} general +{gp}/-{gn} — 판사 재량)"
            )
    lines.append(f"- 권고: {verdict}")
    return lines


# ---------- markdown-KV responses ----------

def _format_modifiers(mods: dict[str, bool]) -> str | None:
    flags = [k for k, v in mods.items() if v]
    if not flags:
        return None
    return " · ".join(_MODIFIER_LABEL.get(k, k) for k in flags)


def _format_penalty(penalty: EffectivePenalty) -> list[str]:
    """Render a penalty as markdown lines (same format as processed penalty block)."""
    return _penalty_kv_lines(
        penalty.imp_min_months,
        penalty.imp_max_months,
        penalty.fine_min_won,
        penalty.fine_max_won,
        penalty.fine_formula,
        penalty.has_life,
        penalty.has_death,
        penalty.sentence_kind_options,
    )


def _format_article(row: sqlite3.Row) -> str:
    name = row["statute_name"] or "?"
    art = row["article_no_num"]
    branch = row["article_branch"] or 0
    para = row["paragraph"] or ""
    art_s = f"§{art}" + (f"의{branch}" if branch else "")
    if para:
        art_s += f" {para}"
    return f"{name} {art_s}"


def _format_branch_option(o: dict) -> str:
    key = _branch_key_label(o.get("key", "?"))
    cond = (o.get("cond") or "").strip()
    parts: list[str] = []
    imp_lo, imp_hi = o.get("imp_min_months"), o.get("imp_max_months")
    if imp_lo is not None or imp_hi is not None:
        parts.append("자유형 " + _span_months(imp_lo, imp_hi))
    f_lo, f_hi = o.get("fine_min_won"), o.get("fine_max_won")
    if f_lo is not None or f_hi is not None:
        parts.append("벌금 " + _span_won(f_lo, f_hi))
    if o.get("has_life"):
        parts.append("무기")
    if o.get("has_death"):
        parts.append("사형")
    suffix = (" — " + " · ".join(parts)) if parts else ""
    return f"- {key}: {cond}{suffix}"


def _format_reference_option(ref: dict) -> str:
    rname = ref.get("statute_name", "?")
    art = ref.get("article_no_num")
    br = ref.get("article_branch") or 0
    para = ref.get("paragraph") or ""
    note = ref.get("note") or ""
    art_s = f"§{art}" + (f"의{br}" if br else "")
    if para:
        art_s += f" {para}"
    tail = f" — {note}" if note else ""
    return f"- {rname} {art_s}{tail}  (choice form: \"{rname}§{art}\")"


def _format_stage_header(
    norm: NormalizedCharge, row: sqlite3.Row, payload_row: sqlite3.Row, stage: str
) -> list[str]:
    """Response header shared by every stage."""
    lines: list[str] = [
        "## status: ok",
        f"## stage: {stage}",
        f"## charge: {norm.raw_key}",
    ]
    if norm.key != norm.raw_key:
        lines.append(f"- 정규화 죄명: {norm.key}")
    mod_str = _format_modifiers(norm.modifiers)
    if mod_str:
        lines.append(f"- 적용: {mod_str}")
    if norm.suffix_split_applied:
        lines.append("- note: 죄명 접미(미수·교사·방조)를 분리해 자동 반영")
    if norm.bracket_prefix_stripped:
        lines.append(
            f"- note: 선행 번호 {norm.bracket_prefix_stripped} 제거 → {norm.raw_key}"
            " — charge 는 죄명 문자열만"
        )
    if norm.quotes_stripped:
        lines.append(
            f"- note: 감싼 인용부호 {norm.quotes_stripped} 제거 → {norm.raw_key}"
            " — charge 는 죄명 문자열만"
        )
    if row["is_alias"] and row["alias_of"]:
        lines.append(f"- 별칭 원본: {row['alias_of']}")

    lines.append(f"## 본조: {_format_article(payload_row)}")
    if payload_row["md_source_name"] and payload_row["md_source_name"] != norm.raw_key:
        lines.append(f"- 매핑 원본 죄명: {payload_row['md_source_name']}")
    lines.append(f"- sg_category_id: {payload_row['sg_category_id']}")
    return lines


def _list_leaves_for_category(
    conn: sqlite3.Connection, sg_category_id: int
) -> list[sqlite3.Row]:
    """Guideline leaves available under this category.

    A leaf is an `sg_subtypes` row with a non-null `type_criterion`; parent
    group nodes are excluded, leaving the ends of the guideline tree. The
    lookup stage returns these so the caller can fill `guideline_leaf_id` on
    its next call.
    """
    return conn.execute(
        "SELECT id, section_no, name, type_no, type_criterion FROM sg_subtypes "
        "WHERE category_id=? AND type_criterion IS NOT NULL "
        "ORDER BY id",
        (sg_category_id,),
    ).fetchall()


# ---------- Sentencing guideline leaf naming, keys, and matching ----------
#
# The name printed by lookup and the name resolved by `guideline_type` are
# produced by the same function (`_leaf_label`) to prevent mismatches.
# Type numbers are matched only via 「N유형」 tokens, never plain substring
# search, preventing false positives from numbers in criterion text (e.g.
# '1억 원 이상, 5억 원 미만' matching 1유형). Ambiguous matches ask for clarification.
#
# Hangul enumeration prefixes (가., 나.) are preserved because they separate
# top-level groups (rape vs indecent act, arrest vs abuse) without which identical
# names and criteria collide. Numeric prefixes (01., 01 ) are stripped.
# Any remaining label collisions are resolved by appending parenthesized criteria
# in `_leaf_keys`.
_LEAF_ENUM_RE = re.compile(r"^\d{1,2}(?:[.)]\s*|\s+)")
_TYPE_TOKEN_RE = re.compile(r"(\d{1,2})\s*유형")
_ORDINAL_TYPE_RE = re.compile(r"제\s*(\d{1,2})\s*유형")
_QUOTE_TRANS = str.maketrans("", "", "\"'“”‘’`「」")
_TRAILING_PAREN_RE = re.compile(r"^(.*?)\s*[\(（]([^()（）]*)[\)）]\s*$")
_RAW_FOOTNOTE_RE = re.compile(r"[₀-₉]")


def _nospace(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _leaf_clean_name(name: str | None) -> str:
    return _LEAF_ENUM_RE.sub("", (name or "").strip()).strip()


def _leaf_label(row: sqlite3.Row) -> str:
    """Canonical name for print and match: 'SubtypeName N유형' (or just SubtypeName if no type_no)."""
    t_no = row["type_no"] if "type_no" in row.keys() else None
    name = _leaf_clean_name(row["name"])
    return f"{name} {t_no}유형" if t_no is not None else name


def _leaf_criterion(row: sqlite3.Row) -> str:
    return (row["type_criterion"] or "").strip()


def _leaf_keys(leaves: list[sqlite3.Row]) -> dict[int, str]:
    """Map leaf id to the key string. If duplicate labels exist in category, append criterion."""
    counts: dict[str, int] = {}
    for lf in leaves:
        label = _leaf_label(lf)
        counts[label] = counts.get(label, 0) + 1
    keys: dict[int, str] = {}
    for lf in leaves:
        label = _leaf_label(lf)
        crit = _leaf_criterion(lf)
        keys[lf["id"]] = f"{label} ({crit})" if counts[label] > 1 and crit else label
    return keys


def _format_months_kr(m: int | None) -> str:
    if m is None:
        return ""
    if m < 12:
        return f"{m}월"
    years, rem = divmod(m, 12)
    return f"{years}년" if rem == 0 else f"{years}년{rem}월"


def _compact_raw_range(raw: str) -> str:
    """Compact sg_ranges.raw_text: '6월 ~ 1년6월' -> '6월~1년6월'."""
    s = raw.replace("<br>", ", ")
    s = _RAW_FOOTNOTE_RE.sub("", s)
    s = re.sub(r"\s*[~\-–]\s*", "~", s)
    return re.sub(r"\s+", " ", s).strip(" ,")


def _range_text(r: sqlite3.Row) -> str:
    """Display format for one cell in sg_ranges. Prefers raw_text."""
    raw = r["raw_text"] if "raw_text" in r.keys() else None
    if raw:
        return _compact_raw_range(raw)
    lo, hi = r["min_months"], r["max_months"]
    if lo is None and hi is None:
        return "상·하한 미산출"
    if lo is None:
        return f"~{_format_months_kr(hi)}"
    if hi is None:
        return f"{_format_months_kr(lo)} 이상"
    return f"{_format_months_kr(lo)}~{_format_months_kr(hi)}"


def _format_leaf_candidates(conn: sqlite3.Connection, leaves: list[sqlite3.Row]) -> list[str]:
    """Format leaf candidate enum in lookup stage."""
    if not leaves:
        return []
    keys = _leaf_keys(leaves)
    lines = ["## 대법원 양형기준 권고형 범위 (정밀 계산 시 `guideline_type` 에 따옴표 안 명칭 그대로)"]
    for lf in leaves:
        key = keys[lf["id"]]
        crit = _leaf_criterion(lf)
        crit_str = f" ({crit})" if crit and not key.endswith(f"({crit})") else ""
        try:
            rows = conn.execute(
                "SELECT * FROM sg_ranges WHERE subtype_id=?", (lf["id"],)
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        r_map = {r["level"]: r for r in rows}
        parts = [f"{lvl} {_range_text(r_map[lvl])}" for lvl in ("기본", "감경", "가중") if lvl in r_map]
        desc = " · ".join(parts) if parts else "권고형 없음"
        lines.append(f'- "{key}"{crit_str}: {desc}')
    return lines


def _format_guide_notes(notes: list[str] | None) -> list[str]:
    """Format notes regarding guideline type resolution."""
    if not notes:
        return []
    return ["## 유형 지정 안내", *notes]


def _get_leaf_label(conn: sqlite3.Connection, leaf_id: int) -> str | None:
    try:
        row = conn.execute(
            "SELECT name, type_no, type_criterion FROM sg_subtypes WHERE id=?",
            (leaf_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    crit = _leaf_criterion(row)
    return _leaf_label(row) + (f" ({crit})" if crit else "")


def _normalize_type_query(q: str) -> str:
    q = (q or "").translate(_QUOTE_TRANS).strip()
    q = q.split(":", 1)[0].strip()
    return _ORDINAL_TYPE_RE.sub(lambda m: f"{m.group(1)}유형", q)


def _split_trailing_paren(q: str) -> tuple[str, str]:
    m = _TRAILING_PAREN_RE.match(q)
    return (m.group(1).strip(), m.group(2).strip()) if m else (q, "")


def _resolve_leaf(
    conn: sqlite3.Connection,
    sg_category_id: int,
    guideline_leaf_id: int | None = None,
    guideline_type: str | None = None,
) -> tuple[int | None, list[str]]:
    """Resolve `guideline_leaf_id` (compat) or `guideline_type` to a leaf ID.

    Returns (resolved_id, guide_notes).
    """
    notes: list[str] = []
    try:
        leaves = conn.execute(
            "SELECT id, name, type_no, type_criterion FROM sg_subtypes "
            "WHERE category_id=? AND type_criterion IS NOT NULL ORDER BY id",
            (sg_category_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return None, notes
    ids = {lf["id"] for lf in leaves}

    if guideline_leaf_id is not None:
        if guideline_leaf_id in ids:
            return guideline_leaf_id, notes
        notes.append(
            f"- guideline_leaf_id={guideline_leaf_id} 는 이 죄명의 유형이 아닙니다 — "
            "guideline_type 에 위 목록의 명칭을 넣어 재호출하세요."
        )

    q = _normalize_type_query(guideline_type or "")
    if not q or not leaves:
        return None, notes
    keys = _leaf_keys(leaves)
    qn = _nospace(q)

    # 1. Exact printed key match (including parenthesized criterion for collided labels)
    hit = [lf for lf in leaves if _nospace(keys[lf["id"]]) == qn]
    # 2. Exact label match
    if not hit:
        hit = [lf for lf in leaves if _nospace(_leaf_label(lf)) == qn]
    # 3. Structural match: type number token + name substring, narrowed by criterion
    if not hit:
        label_part, crit_part = _split_trailing_paren(q)
        tokens = list(_TYPE_TOKEN_RE.finditer(label_part))
        if tokens:
            want_no: int | None = int(tokens[-1].group(1))
            name_part = _nospace(_TYPE_TOKEN_RE.sub("", label_part))
        elif label_part.isdigit():
            want_no, name_part = int(label_part), ""
        else:
            want_no, name_part = None, _nospace(label_part)
        if want_no is not None or name_part:
            cands = []
            for lf in leaves:
                if want_no is not None and lf["type_no"] != want_no:
                    continue
                cn = _nospace(_leaf_clean_name(lf["name"]))
                if name_part and not (name_part in cn or cn in name_part):
                    continue
                cands.append(lf)
            if crit_part and len(cands) > 1:
                cp = _nospace(crit_part)
                narrowed = [
                    lf for lf in cands
                    if cp == _nospace(_leaf_criterion(lf))
                    or cp in _nospace(_leaf_criterion(lf))
                    or _nospace(_leaf_criterion(lf)) in cp
                ]
                if narrowed:
                    cands = narrowed
            hit = cands
    # 4. Criterion-only text match (e.g. '보통 동기 살인')
    if not hit and len(qn) >= 2:
        hit = [lf for lf in leaves if qn == _nospace(_leaf_criterion(lf))]
        if not hit:
            hit = [lf for lf in leaves if qn in _nospace(_leaf_criterion(lf))]
    # 5. Legacy leaf id integer
    if not hit and q.isdigit() and int(q) in ids:
        return int(q), notes

    if len(hit) == 1:
        return hit[0]["id"], notes
    if len(hit) > 1:
        shown = ", ".join(f'"{keys[lf["id"]]}"' for lf in hit[:6])
        notes.append(
            f'- guideline_type="{guideline_type}" 에 맞는 유형이 {len(hit)}개입니다 — '
            f"{shown} 중 하나를 그대로 넣어 재호출하세요."
        )
        return None, notes
    if len(leaves) == 1:
        notes.append(
            f'- guideline_type="{guideline_type}" 은 목록에 없지만 이 죄명의 유형이 하나뿐이라 '
            "그것으로 계산합니다."
        )
        return leaves[0]["id"], notes
    notes.append(
        f'- guideline_type="{guideline_type}" 은 위 목록에 없습니다 — '
        "목록의 따옴표 안 명칭을 그대로 넣어 재호출하세요."
    )
    return None, notes


def _list_factors_for_category(
    conn: sqlite3.Connection, sg_category_id: int
) -> list[sqlite3.Row]:
    """Factors available under this category.

    The union of the factors attached to each leaf. Within one category the
    leaves carry nearly the same factors: the type criterion is what
    distinguishes them, while the factors are shared.

    The lookup stage returns these so the caller can fill `guideline_factors`
    on its next call and cite the general factors in its reasoning.
    """
    return conn.execute(
        "SELECT DISTINCT scope, kind, direction, text FROM sg_factors "
        "WHERE category_id=? "
        "ORDER BY scope DESC, kind, direction, text",
        (sg_category_id,),
    ).fetchall()


# Factors are grouped by scope (special or ordinary), kind (conduct or
# offender) and direction (aggravating or mitigating). The four lists the
# caller supplies mirror these groups.
def _format_factor_enum(factors: list[sqlite3.Row]) -> list[str]:
    if not factors:
        return []
    # Group by scope, kind and direction.
    from collections import defaultdict
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for r in factors:
        groups[(r["scope"], r["kind"], r["direction"])].append(r["text"])

    # Special factors first: they choose the band. Ordinary factors only
    # move the sentence within it.
    special = [k for k in groups if k[0] == "특별"]
    general = [k for k in groups if k[0] == "일반"]
    n_special = sum(len(groups[k]) for k in special)
    n_general = sum(len(groups[k]) for k in general)

    # Group heading -> the key the caller passes it back under.
    _GF_KEY = {
        ("특별", "행위", "가중"):       "special_act_aggravators",
        ("특별", "행위", "감경"):       "special_act_mitigators",
        ("특별", "행위_공통", "가중"):  "special_act_aggravators",
        ("특별", "행위_공통", "감경"):  "special_act_mitigators",
        ("특별", "행위_미수", "가중"):  "special_act_aggravators",
        ("특별", "행위_미수", "감경"):  "special_act_mitigators",
        ("특별", "행위자_기타", "가중"): "special_actor_aggravators",
        ("특별", "행위자_기타", "감경"): "special_actor_mitigators",
    }

    lines: list[str] = []
    if special:
        lines.append(
            f"## 양형기준 특별인자 enum ({n_special}개 — `guideline_factors` 선택용)"
        )
        lines.append(
            "- schema: {special_act_aggravators: [text...], special_act_mitigators: [...], "
            "special_actor_aggravators: [...], special_actor_mitigators: [...]}"
        )
        lines.append(
            "- 각 그룹 헤더 옆 *key* 에 해당 list 에 text 그대로 넣어 호출."
        )
        for key in sorted(special):
            scope, kind, direction = key
            gf_key = _GF_KEY.get(key, "?")
            lines.append(f"[{scope}/{kind}/{direction}] → {gf_key}")
            for t in groups[key]:
                lines.append(f"- {t}")
    if general:
        lines.append(
            f"## 양형기준 일반인자 enum ({n_general}개 — 영역 결정 무영향, "
            "선고형 위치 결정 시 reasoning 인용)"
        )
        for key in sorted(general):
            scope, kind, direction = key
            lines.append(f"[{scope}/{kind}/{direction}]")
            for t in groups[key]:
                lines.append(f"- {t}")
    return lines


def _list_probation_factors_for_category(
    conn: sqlite3.Connection, sg_category_id: int
) -> list[sqlite3.Row]:
    """Suspension factors available under this category.

    A union across the category, grouped by section number and source note.
    That grouping keeps the separate four-quadrant tables some categories
    publish per sub-section — 증권·금융 among them — from collapsing into one.

    The lookup stage returns these so the caller can fill the four
    `probation_factors` lists without inventing entries, and can tell which
    sub-section a factor came from.
    """
    return conn.execute(
        """
        SELECT pole, direction, text, section_no, source_note
        FROM sg_probation_factors
        WHERE category_id=?
        ORDER BY pole DESC, direction, section_no, source_note, text
        """,
        (sg_category_id,),
    ).fetchall()


# pole ∈ {major, general}, direction ∈ {positive, negative}.
# Mirrors the four lists the caller passes back; the rule that consumes
# them is in the recommendation function above.
def _format_probation_factor_enum(rows: list[sqlite3.Row]) -> list[str]:
    if not rows:
        return []
    from collections import defaultdict

    groups: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for r in rows:
        key = (
            r["pole"],
            r["direction"],
            r["section_no"] or "",
            r["source_note"] or "",
        )
        groups[key].append(r["text"])

    order_pole = {"major": 0, "general": 1}
    order_dir = {"positive": 0, "negative": 1}

    # Group heading -> the key the caller passes it back under.
    _PF_KEY = {
        ("major", "positive"):   "major_positive",
        ("major", "negative"):   "major_negative",
        ("general", "positive"): "general_positive",
        ("general", "negative"): "general_negative",
    }

    lines: list[str] = [
        f"## 집행유예 4분면 enum ({len(rows)}개 — `probation_factors` 선택용)",
        "- schema: {major_positive: [text...], major_negative: [...], "
        "general_positive: [...], general_negative: [...]}",
        "- 각 그룹 헤더 옆 *key* 에 해당 list 에 text 그대로 넣어 호출.",
    ]
    for key in sorted(
        groups, key=lambda k: (order_pole[k[0]], order_dir[k[1]], k[2], k[3])
    ):
        pole, direction, sec, note = key
        pf_key = _PF_KEY.get((pole, direction), "?")
        head = f"[{pole}/{direction}] → {pf_key}"
        if sec:
            head += f"  (section={sec}"
            if note:
                head += f", {note}"
            head += ")"
        elif note:
            head += f"  ({note})"
        lines.append(head)
        for t in groups[key]:
            lines.append(f"- {t}")
    return lines


# The statutory grounds for adjustment. Universal, so this is a constant
# rather than a per-category lookup; the order and the multipliers live with
# the adjustment logic. Attempt, aiding and solicitation are added
# automatically from the charge name.
_STATUTORY_MOD_ENUM: list[str] = [
    "## 형법 §56 가중·감경 사유 enum (`statutory_modifications` 인자 선택용)",
    "[본조_가중] — 해당 본조 자체에 가중 규정 (상습범·특정범죄가중법 등)",
    "[특수교사방조_가중] — 특수교사·특수방조 (형법 §34 ②)",
    "[누범_가중] — 누범 (§35), 자유형 장기 2배",
    "[법률상_필요감경] — 의무 감경: 방조 (§32 ②), 농아자 (§11) 등",
    "[법률상_임의감경] — 재량 감경: 미수 (§25 ②), 중지미수 (§26), 자수 (§52), 심신미약 (§10 ②) 등",
    "[경합범_가중] — §37 전단·§38 ① 2호 (가장 무거운 죄의 장기 1/2 가중). 동종 다행위는 `act_count` 인자로 명시 (자동 적용). 명시 입력도 가능.",
    "[작량감경] — 정상참작 (§53, 단일 type)",
    "",
    "참고:",
    "- charge suffix '미수/방조/교사' 는 도구가 자동 분리 — 명시 안 해도 auto-add",
    "- 인자 schema: [{kind: <위 6 종>, type: '미수 (§25)' 등, basis: '형법 §25 ②', applied: bool}]",
    "- applied 키 누락 = 기본 적용 (default true). applied=true 명시도 적용.",
    "- applied=false 또는 null 명시 = *주장됐으나 부적용* skip (trace 만).",
    "- 잘못된 kind 또는 같은 (kind, type) 중복은 INVALID / DUPLICATE 로 자동 무시.",
]


def _format_modifier_enum() -> list[str]:
    return list(_STATUTORY_MOD_ENUM)


def _lookup_historic_article(
    conn: sqlite3.Connection, payload_row: sqlite3.Row, offense_iso: str
) -> dict | None:
    """Fetch the provision text as it stood at the offence date.

    The last version of the article before that date, within the same law_id.
    Looking only at rows marked as changed misses an article that was never
    amended — it exists solely in the baseline snapshot, where the changed flag
    is NULL — and reports it as not loaded. Changed rows are preferred with the
    baseline as fallback, and a changed row wins a tie on effective date.
    `statute_lookup` had the same defect.
    """
    statute_id = payload_row['statute_id'] if 'statute_id' in payload_row.keys() else None
    if not statute_id:
        return None
    art_no = payload_row['article_no_num']
    art_br = payload_row['article_branch'] or 0
    s = conn.execute(
        'SELECT law_id FROM st_statutes WHERE id=?', (statute_id,)
    ).fetchone()
    if not s or not s['law_id']:
        return None
    law_id = s['law_id']
    r = conn.execute(
        """SELECT s.effective_date, s.mst, a.article_text
           FROM st_articles a JOIN st_statutes s ON s.id=a.statute_id
           WHERE s.law_id=? AND a.article_no_num=? AND COALESCE(a.article_branch,0)=?
             AND s.effective_date <= ?
           ORDER BY s.effective_date DESC, (a.article_changed='Y') DESC LIMIT 1""",
        (law_id, art_no, art_br, offense_iso),
    ).fetchone()
    if not r:
        return None
    # Also the current version's date, to show both ends of the timeline.
    current = conn.execute(
        """SELECT effective_date FROM st_statutes WHERE law_id=?
           ORDER BY effective_date DESC LIMIT 1""",
        (law_id,),
    ).fetchone()
    current_eff = current['effective_date'] if current else None
    return {
        'effective_date': r['effective_date'],
        'mst': r['mst'],
        'text': r['article_text'],
        'current_effective_date': current_eff,
        'is_outdated': current_eff and r['effective_date'] != current_eff,
    }


# The ceiling on a term of years was raised by amendment. Which ceiling
# applies is decided by the date the amendment took effect, not the date it
# was promulgated, because the law in force at the time of the offence
# governs. The two ceilings and the boundary date are defined once, with the
# other general-part constants.


def _apply_art42_versioned_cap(
    payload: dict, current_max: int | None, offense_iso: str,
) -> str | None:
    """Correct a ceiling for the version of the general part then in force.

    A historical row bakes in the §42 cap as of *the article's own* effective
    date. But the §42 ① ceiling itself rose from 15 to 30 years on 2010-10-16,
    and that is an amendment to the general part, unrelated to the article: an
    article unchanged since before then keeps the old cap in every historical
    row. 살인 §250 has not changed since 1996, so its rows all hold max=180.
    The tool re-applies the cap in force at the offence date.

    An open-ended ceiling — "X년 이상의 징역", the article naming no upper bound
    — is recognised by the current row's ceiling being exactly the current §42
    cap (360 alone, 600 aggravated). An article stating its own ceiling
    ("15년 이하" = 180) matches neither value and is left alone, which makes a
    false positive impossible.

    The aggravated cap after 누범 or 경합 (`_AGGRAVATED_IMP_CAP_MONTHS`) is a
    separate constant, and its own dependence on the date is not handled here.
    """
    ver_max = payload.get("stat_imp_max_months")
    if ver_max is None or current_max not in _ART42_CAPS:
        return None
    old_cap, new_cap = _ART42_CAPS[current_max]
    # Only correct a ceiling that came from the general part in the first
    # place — recognisable because it equals one of the two general ceilings.
    # A provision that stated its own maximum at the time keeps it: reading
    # today's open-ended form back onto the past would inflate a one-year
    # maximum into thirty.
    if ver_max not in (old_cap, new_cap):
        return None
    target = old_cap if offense_iso < _ART42_REFORM_ISO else new_cap
    if ver_max == target:
        return None  # 이미 행위시 cap 과 일치
    payload["stat_imp_max_months"] = target
    era = "구법(~2010.10.15)" if offense_iso < _ART42_REFORM_ISO else "신법(2010.10.16~)"
    return (
        f"## §42① 상한 시점보정: {ver_max}→{target}월 "
        f"(행위 {offense_iso} {era}, 상한 개방형 본조)"
    )


def _clmv_version_is_empty(v: sqlite3.Row) -> bool:
    """Is this historical row empty of any penalty information?

    Overlaying an empty row on the current one blanks the statutory range and
    sends out an empty '?~?' — a bad row quietly breaking the logic. True makes
    `_get_versioned_payload` skip the overlay and keep the current figures.

    Only a row with no direct figures, no branches, no references and no
    formula counts as empty. A reference-only or branch-only row has NULL
    figures legitimately and is still overlaid.
    """
    try:
        kinds = json.loads(v["sentence_kind_options"] or "[]")
    except (TypeError, json.JSONDecodeError):
        kinds = []
    try:
        branch = json.loads(v["branch_options"] or "[]")
    except (TypeError, json.JSONDecodeError):
        branch = []
    has_formula = ("fine_formula" in v.keys()) and bool(v["fine_formula"])
    return (
        not kinds
        and v["stat_imp_min_months"] is None
        and v["stat_imp_max_months"] is None
        and v["stat_fine_min_won"] is None
        and v["stat_fine_max_won"] is None
        and not v["has_life"]
        and not v["has_death"]
        and not v["has_conditional_branch"]
        and not branch
        and v["reference_mode"] is None
        and not has_formula
    )


def _get_versioned_payload(
    conn: sqlite3.Connection, base_row: sqlite3.Row, offense_iso: str | None,
) -> tuple[dict, dict | None, str | None]:
    """Overlay historical bounds onto the current row for a given date.

    Overrides the numeric fields; with no match the base row stands.

    Returns (payload_dict, version_meta | None, art42_trace | None), where
    version_meta is {effective_date, matched_paragraph, match_confidence, mst}
    and art42_trace is the line describing a ceiling correction, present only
    when one happened.
    """
    base_dict = dict(base_row)
    if not offense_iso:
        return base_dict, None, None
    # Note the current ceiling before overriding, to tell an open-ended
    # maximum from one the provision states itself.
    current_max = base_dict.get("stat_imp_max_months")
    version_meta = None
    v = conn.execute(
        """SELECT * FROM clm_versions
           WHERE clm_id=? AND effective_date<=?
           ORDER BY effective_date DESC LIMIT 1""",
        (base_row['id'], offense_iso),
    ).fetchone()
    if v and _clmv_version_is_empty(v):
        # An empty historical row must not overwrite the current bounds —
        # that would erase the penalty entirely. Keep the current row, mark
        # it, and log.
        _log.warning(
            "⚠ DATA: clm_versions 빈 시점본 (clm_id=%s eff=%s) — override 건너뜀, "
            "현행 base 정량 유지", base_row['id'], v['effective_date'])
        version_meta = {
            'effective_date': v['effective_date'],
            'empty_version': True,
            'matched_paragraph': v['matched_paragraph'],
            'match_confidence': v['match_confidence'],
            'mst': v['mst'],
        }
    elif v:
        # Overlay the historical bounds.
        override_cols = [
            'sentence_kind_options', 'stat_imp_min_months', 'stat_imp_max_months',
            'stat_fine_min_won', 'stat_fine_max_won', 'has_life', 'has_death',
            'has_conditional_branch', 'branch_options',
            'reference_mode', 'reference_multiplier', 'reference_articles',
            'fine_formula',  # historical rows can carry their own formula
        ]
        for col in override_cols:
            if col in v.keys():
                base_dict[col] = v[col]
        version_meta = {
            'effective_date': v['effective_date'],
            'matched_paragraph': v['matched_paragraph'],
            'match_confidence': v['match_confidence'],
            'mst': v['mst'],
        }
    # Correct the ceiling whether or not a historical row was found.
    art42_trace = _apply_art42_versioned_cap(base_dict, current_max, offense_iso)
    return base_dict, version_meta, art42_trace


def _historic_appendix(
    conn: sqlite3.Connection, payload_row: sqlite3.Row, offense_date: str | None
) -> str:
    """Provision text, appended to every stage response after an exact match.

    형법 §1 ①: an offence and its punishment are governed by the law in force
    when it was committed. This tool's figures come from the current mapping,
    so the caller needs the text as it then stood to see a branch or a
    statutory range that has since changed.
    """
    offense_iso = to_iso_date(offense_date)
    if not offense_iso:
        return ""
    historic = _lookup_historic_article(conn, payload_row, offense_iso)
    lines = ["", f"## 행위시 조문 본문 (offense_date={offense_iso})"]
    if historic:
        lines.append(f"- 적용 시행본 시행일: {historic['effective_date']}")
        if historic.get('is_outdated'):
            lines.append(
                f"- ⚠ 현행본 시행일 ({historic['current_effective_date']}) 과 다름 "
                f"— 본조 분기·법정형 *시점별 차이* 가능. 행위시 본문 그대로 적용."
            )
        else:
            lines.append("- 현행본과 동일 — 시간 축 차이 없음.")
        lines.append("- 본문:")
        lines.append(historic['text'])
    else:
        lines.append("- 해당 법령의 행위시점 본문이 DB 미적재. 현행 정량 그대로 사용.")
    return "\n".join(lines)


def _manual_source(conn: sqlite3.Connection, sg_category_id: int) -> tuple[str, str] | None:
    """Link to the published guideline commentary for this offence group.

    A separate loader populates the column. An absent column or an empty value
    means None and the link is simply omitted: swallowing OperationalError
    makes that a question of whether the data is there rather than a branch, so
    the tool works the same in an installation that never loaded it.
    """
    try:
        r = conn.execute(
            "SELECT name, manual_url FROM sg_categories WHERE id=?", (sg_category_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if r and r["manual_url"]:
        return (r["name"], r["manual_url"])
    return None


def _format_lookup_response(
    conn: sqlite3.Connection,
    norm: NormalizedCharge,
    row: sqlite3.Row,
    payload_row: sqlite3.Row,
    penalty: EffectivePenalty,
    act_count: int = 1,
    guide_notes: list[str] | None = None,
) -> str:
    """The lookup stage, once the provision has resolved."""
    lines = _format_stage_header(norm, row, payload_row, "lookup")

    # Cite the commentary here: lookup is the first response in every flow.
    src = _manual_source(conn, payload_row["sg_category_id"])
    if src:
        lines.append(f"- 출처: [{src[0]} 양형기준 해설서(PDF)]({src[1]})")

    # How the provision resolved.
    if penalty.trace:
        lines.extend(penalty.trace)

    # The statutory range.
    pen_lines = _format_penalty(penalty)
    if pen_lines:
        lines.append("## 법정형")
        lines.extend(pen_lines)
    else:
        lines.append("## 법정형: 본조 직접 정량 없음")
    lines.append(f"- 정량 근거: {_source_label(penalty.source)}")

    # Guideline leaves to choose from on the next call.
    leaves = _list_leaves_for_category(conn, payload_row["sg_category_id"])
    lines.extend(_format_leaf_candidates(conn, leaves))
    lines.extend(_format_guide_notes(guide_notes))

    # Factors to choose from. Offered as the union across the category,
    # since they barely differ between leaves within one.
    factors = _list_factors_for_category(conn, payload_row["sg_category_id"])
    lines.extend(_format_factor_enum(factors))

    # Suspension factors to choose from.
    prob_factors = _list_probation_factors_for_category(
        conn, payload_row["sg_category_id"]
    )
    lines.extend(_format_probation_factor_enum(prob_factors))

    # Statutory adjustments to choose from; the same for every offence.
    lines.extend(_format_modifier_enum())

    # Flag that multiple-offence aggravation will apply.
    if act_count >= 2:
        lines.append("")
        lines.append(f"## 경합범 가중 안내 (act_count={act_count})")
        lines.append(
            "- §37 전단·§38 ① 2호 자동 적용 예정 (LLM 이 statutory_modifications 에 "
            "별도 명시 안 해도 됨). 처단형 stage 진입 시 trace 에 노출."
        )

    return "\n".join(lines)


def _format_processed_response(
    norm: NormalizedCharge,
    row: sqlite3.Row,
    payload_row: sqlite3.Row,
    penalty: EffectivePenalty,
    processed: ProcessedPenalty,
    guide_notes: list[str] | None = None,
) -> str:
    """The processed-range stage: statutory range, adjustments applied, result."""
    lines = _format_stage_header(norm, row, payload_row, "처단형")

    # Statutory range, after resolution.
    if penalty.trace:
        lines.append("## 법정형 산출 trace")
        lines.extend(penalty.trace)
    pen_lines = _format_penalty(penalty)
    if pen_lines:
        lines.append("## 법정형")
        lines.extend(pen_lines)
    lines.append(f"- 정량 근거: {_source_label(penalty.source)}")

    # How each adjustment moved it.
    lines.append("## 처단형 산출 (형법 §56 순서)")
    lines.extend(processed.trace)

    # The processed range.
    proc_lines = _format_processed_penalty_lines(processed)
    if proc_lines:
        lines.append("## 처단형")
        lines.extend(proc_lines)
    else:
        lines.append("## 처단형: 법정형 그대로")

    lines.extend(_format_guide_notes(guide_notes))
    return "\n".join(lines)


_INTERSECT_HDR = "## 선고 가능 범위 (처단형 ∩ 권고형)"


def _format_range_block(
    penalty: EffectivePenalty,
    processed: ProcessedPenalty,
    rec: RecommendedRange | None,
    leaf_id: int,
    floor: int | None,
    leaf_label: str = "",
) -> list[str]:
    """Shared body for recommended and final stages."""
    lines: list[str] = []
    lines.append("## 법정형")
    lines.extend(_format_penalty(penalty))
    lines.append(f"- 정량 근거: {_source_label(penalty.source)}")

    lines.append("## 처단형 산출 (형법 §56 순서)")
    lines.extend(processed.trace)
    proc_lines = _format_processed_penalty_lines(processed)
    if proc_lines:
        lines.append("## 처단형")
        lines.extend(proc_lines)
    else:
        lines.append("## 처단형: 법정형 그대로")

    if leaf_label:
        lines.append(f"## 양형기준: {leaf_label}")
    elif leaf_id is not None and leaf_id >= 0:
        lines.append(f"## 양형기준 유형 #{leaf_id}")
    else:
        lines.append("## 양형기준: 유형 미확정")
    if floor is not None:
        lines.append(f"- 처단형 하한: {floor}월 (권고 하한이 이보다 낮으면 여기로 올림 — 공통원칙 §02)")
    if rec is None:
        lines.append("- 권고형: 없음 (이 유형에는 권고 형량범위 표가 없음 — 벌금형 전용 등)")
    else:
        lines.append("## 권고형")
        lines.append(f"- 영역: {rec.level}")
        lines.append(_rec_range_line(rec))
        lines.append(f"- 특별조정: {'적용' if rec.is_special_adjusted else '없음'}")
        if rec.raw_text:
            lines.append(f"- 원문: {_compact_raw_range(rec.raw_text)}")
    return lines


def _format_recommended_response(
    norm: NormalizedCharge,
    row: sqlite3.Row,
    payload_row: sqlite3.Row,
    penalty: EffectivePenalty,
    processed: ProcessedPenalty,
    rec: RecommendedRange | None,
    intersect: tuple[int | None, int | None],
    leaf_id: int,
    floor: int | None,
    leaf_label: str = "",
    guide_notes: list[str] | None = None,
) -> str:
    """The guideline stage: processed range, recommended range, and their overlap."""
    lines = _format_stage_header(norm, row, payload_row, "권고형")
    lines.extend(_format_range_block(penalty, processed, rec, leaf_id, floor, leaf_label))

    # Their overlap: what may lawfully be imposed.
    lo, hi = intersect
    lines.append(_INTERSECT_HDR)
    if lo is not None and hi is not None and lo > hi:
        rec_span = _span_months(rec.min_months, rec.max_months) if rec else "미상"
        lines.append(
            f"- 자유형: 처단형 {_span_months(processed.imp_min_months, processed.imp_max_months)} 과 "
            f"권고형 {rec_span} 이 겹치지 않음"
        )
        lines.append(
            "- 양형기준 [공통원칙] §02: 권고가 처단형 벗어나면 *처단형* 우선."
        )
    else:
        lines.append(
            f"- 자유형: {_span_months(lo, hi)}"
            + (" — 권고 준수" if rec is not None else " — 권고 미적용")
        )

    lines.extend(_format_guide_notes(guide_notes))
    return "\n".join(lines)


def _format_final_response(
    norm: NormalizedCharge,
    row: sqlite3.Row,
    payload_row: sqlite3.Row,
    penalty: EffectivePenalty,
    processed: ProcessedPenalty,
    rec: RecommendedRange | None,
    intersect: tuple[int | None, int | None],
    leaf_id: int,
    floor: int | None,
    verify_lines: list[str],
    probation_lines: list[str],
    fine_paragraphs: list[str] | None = None,
    mit_applied: bool = False,
    leaf_label: str = "",
    guide_notes: list[str] | None = None,
) -> str:
    """The final stage: ranges, the proposed sentence checked, and suspension."""
    lines = _format_stage_header(norm, row, payload_row, "final")
    lines.extend(_format_range_block(penalty, processed, rec, leaf_id, floor, leaf_label))

    lines.append(_INTERSECT_HDR)
    import json as _json
    try:
        raw_kinds = _json.loads(payload_row["sentence_kind_options"] or "[]")
    except (TypeError, _json.JSONDecodeError):
        raw_kinds = []
    kinds_eff = set(processed.sentence_kind_options or []) | set(raw_kinds)

    lo, hi = intersect
    if "imprisonment" in kinds_eff:
        if lo is not None and hi is not None and lo > hi:
            lines.append("- 자유형: 처단형과 권고형이 겹치지 않음 → 처단형 우선 (공통원칙 §02)")
        elif lo is None and hi is None:
            lines.append("- 자유형: 범위 미정 (무기·사형 전속 등)")
        else:
            lines.append(f"- 자유형: {_span_months(lo, hi)}")

    if "fine" in kinds_eff or processed.fine_min_won or processed.fine_max_won or processed.fine_formula:
        f_lo = processed.fine_min_won
        f_hi = processed.fine_max_won
        if f_hi is not None or f_lo is not None:
            lines.append(f"- 벌금: {_span_won(f_lo if f_lo is not None else 0, f_hi)}")
        elif processed.fine_formula:
            lines.append(f"- 벌금: {_format_fine_formula(processed.fine_formula)}")
        elif fine_paragraphs:
            lines.append("- 벌금: 정량 미상 (조문 본문 — 식 기반 정량):")
            for p in fine_paragraphs:
                for line in p.splitlines():
                    lines.append(f"  > {line}")
        else:
            lines.append("- 벌금: 정량 미상 (법정형 벌금 정량 NULL — statute_lookup 조회 권장)")
        if mit_applied:
            note = "§55 ① 6호 다액 1/2 자동 반영됨" if processed.fine_formula else "벌금 다액 1/2 (§55 ① 6호)"
            lines.append(f"  ※ 감경 적용: {note}")
        lines.append("- 자유형·벌금 중 형종 선택은 재량 (둘 다 법정형 안)")

    # The proposed sentence, checked.
    lines.append("## 선고형 검증")
    lines.extend(verify_lines)

    # Suspension.
    lines.append("## 집행유예")
    lines.extend(probation_lines)

    lines.extend(_format_guide_notes(guide_notes))
    return "\n".join(lines)


def _format_pending_response(
    norm: NormalizedCharge,
    row: sqlite3.Row,
    payload_row: sqlite3.Row,
    pending: PendingResolution,
) -> str:
    """Resolution stopped: the response asking for what is missing."""
    status = {
        "branch": "needs_branch_key",
        "branch_invalid": "invalid_branch_key",
        "reference": "needs_reference_choice",
        "reference_invalid": "invalid_reference_choice",
        "reference_missing": "reference_target_missing",
        "modifier_directive": "needs_base_charge",
    }.get(pending.kind, f"pending_{pending.kind}")

    lines = [
        f"## status: {status}",
        "## stage: lookup",
        f"## charge: {norm.raw_key}",
        f"## 본조: {_format_article(payload_row)}",
        f"- sg_category_id: {payload_row['sg_category_id']}",
    ]
    if pending.message:
        lines.append(f"- message: {pending.message}")

    if pending.kind in ("branch", "branch_invalid"):
        lines.append(f"## branch_options ({len(pending.options)})")
        for o in pending.options:
            lines.append(_format_branch_option(o))
        first = _branch_key_label(pending.options[0].get("key")) if pending.options else "1"
        lines.append(f'## 호출: 위 목록의 키를 branch_key 에 문자열 하나로 넣어 재호출 (예: branch_key="{first}")')
    elif pending.kind in ("reference", "reference_invalid"):
        lines.append(f"## reference_articles ({len(pending.options)})")
        for ref in pending.options:
            lines.append(_format_reference_option(ref))
        rm = payload_row["reference_mode"]
        mult = payload_row["reference_multiplier"]
        if rm in ("가중", "준용", "공동가중"):
            lines.append(
                f"- reference_mode={rm}, multiplier={_b(mult)} — 선택 후 상한에 곱셈 적용"
            )
        lines.append("## 호출: reference_choice=\"형법§347\" 등 form 으로 재호출")
    elif pending.kind == "reference_missing":
        for ref in pending.options:
            lines.append(_format_reference_option(ref))
        lines.append(
            "- 매핑 테이블에 해당 조항 row 없음. statute_lookup 도구로 본문 조회 권고."
        )
    elif pending.kind == "modifier_directive":
        d = pending.options[0] if pending.options else {}
        mk = d.get("modifier_kind", "본조_가중")
        basis = d.get("basis", "")
        lines.append("## 가중 수식어 (modifier) — 독립 법정형 없음, base 죄에 부착")
        lines.append(f"- 이 죄명은 *{basis}* 가중 규정 — 자기 형량 없이 *어떤 base 죄든* 그 형을 가중.")
        lines.append("## 처리 절차 (2-step):")
        lines.append("  1) 실제 base 죄(행위에 해당하는 성범죄 등)를 charge 인자로 compute_sentencing_range 재호출")
        lines.append(
            f"  2) 그 호출에 statutory_modifications=[{{\"kind\": \"{mk}\", "
            f"\"type\": \"{basis}\", \"basis\": \"{basis}\", \"applied\": true}}] 추가"
        )
        lines.append(f"- 효과: base 형 상한 ×1.5 (½ 가중, {mk}), §42② 50년 cap. base 가 자유형이면 그 자유형 가중.")

    return "\n".join(lines)


def _penalty_brief(r: sqlite3.Row) -> str:
    """One-line penalty summary, so a caller can tell the candidates apart."""
    if r["has_conditional_branch"]:
        try:
            n = len(json.loads(r["branch_options"] or "[]"))
        except (TypeError, json.JSONDecodeError):
            n = 0
        return f"분기형 {n}옵션"
    rm = r["reference_mode"]
    if rm and rm != "정보":
        return rm
    parts = []
    a, b = r["stat_imp_min_months"], r["stat_imp_max_months"]
    if a is not None and b is not None:
        parts.append(f"징역 {a}~{b}월")
    elif a is not None:
        parts.append(f"징역 {a}월 이상")
    elif b is not None:
        parts.append(f"징역 {b}월 이하")
    if r["has_life"]:
        parts.append("무기")
    if r["has_death"]:
        parts.append("사형")
    fmax = r["stat_fine_max_won"]
    if fmax:
        parts.append(f"벌금≤{fmax // 10000:,}만원")
    return " · ".join(parts) or "정량 미상"


def _candidate_line(r: sqlite3.Row, with_cat: bool = False) -> str:
    """One candidate line: provision, penalty, conduct described, and how to
    choose it.

    `act_descriptor` is a conduct label derived from the corpus — from the text
    of the article's items and the titles of the provisions it references. It
    is shown so that a family of provisions cited by article number alone, as
    drug offences tend to be, can still be navigated by what the conduct was:
    수출입 against 매매 against 소지 against 사용. Absent, it is omitted. The
    `notes` column, being free prose, is not loaded at all, for the reason
    `_format_lookup_response` gives — conduct is described from verified fields
    or not at all.
    """
    cat = f"sg_category_id={r['sg_category_id']}  " if with_cat else ""
    act = (r["act_descriptor"] or "").strip()
    act_str = f"  행위={act}" if act else ""
    return (f"- {cat}본조={_format_article(r)}  [{_penalty_brief(r)}]{act_str}  "
            f"md_source={r['md_source_name']}  "
            f"statute_choice={_format_statute_choice_form(r)!r}")


def _format_cross_cat_response(
    norm: NormalizedCharge, rows: list[sqlite3.Row]
) -> str:
    cats = sorted({r["sg_category_id"] for r in rows})
    lines = [
        "## status: ambiguous_category",
        "## stage: lookup",
        f"## charge: {norm.raw_key}",
        f"- 동일 charge_key 가 {len(cats)} 카테고리에 존재 (총 {len(rows)} row). "
        "sg_category_id 명시 후 재호출. 같은 카테고리에 여러 row 면 statute_choice 도 명시.",
        "## candidates",
    ]
    for r in rows:
        lines.append(_candidate_line(r, with_cat=True))
    return "\n".join(lines)


def _format_same_cat_multi_row_response(
    norm: NormalizedCharge, rows: list[sqlite3.Row]
) -> str:
    """Several provisions within one category: which provision, not which
    category.

    상해 falls under 형법 §257 ① and 폭처법 §2 ③ (누범상해) alike, both in the
    same violent-offence category. The caller states which, having read the
    facts for the form the conduct took and whether there is a prior record.
    """
    cat = rows[0]["sg_category_id"]
    lines = [
        "## status: ambiguous_statute",
        "## stage: lookup",
        f"## charge: {norm.raw_key}",
        f"- 같은 sg_category_id={cat} 안에 {len(rows)} row 존재 (조항/항 모호). "
        "facts 의 행위·전과 보고 statute_choice 인자로 명시 후 재호출.",
        "## candidates",
    ]
    for r in rows:
        lines.append(_candidate_line(r))
    return "\n".join(lines)


def _format_wrong_cat_response(
    norm: NormalizedCharge,
    rows: list[sqlite3.Row],
    requested_cat: int,
) -> str:
    lines = [
        "## status: exact_wrong_category",
        "## stage: lookup",
        f"## charge: {norm.raw_key}",
        f"- 요청 sg_category_id={requested_cat} 에 매칭 row 없음. "
        f"다른 카테고리에 exact 매칭:",
        "## candidates",
    ]
    for r in rows:
        lines.append(
            f"- sg_category_id={r['sg_category_id']}  "
            f"본조={_format_article(r)}  "
            f"md_source={r['md_source_name']}"
        )
    return "\n".join(lines)


def _format_fuzzy_response(
    norm: NormalizedCharge, candidates: list[sqlite3.Row]
) -> str:
    lines = [
        "## status: not_found_with_candidates",
        "## stage: lookup",
        f"## charge: {norm.raw_key}",
        "- 정확한 매칭 없음. 유사 후보:",
    ]
    for c in candidates:
        lines.append(
            f"  - {c['charge_key']}  (sg_category_id={c['sg_category_id']}, "
            f"{_format_article(c)})"
        )
    lines.append("- facts 의 부속표시 확인 후 정확한 charge 로 재호출.")
    return "\n".join(lines)


def _format_invalid_statute_choice_response(
    norm: NormalizedCharge, rows: list[sqlite3.Row], statute_choice: str
) -> str:
    """Response when the chosen provision matches none of the candidates."""
    lines = [
        "## status: invalid_statute_choice",
        "## stage: lookup",
        f"## charge: {norm.raw_key}",
        f"- statute_choice={statute_choice!r} 파싱 또는 매칭 실패. "
        "candidates 중 statute_choice 값 그대로 재호출.",
        "## candidates",
    ]
    for r in rows:
        lines.append(
            f"- 본조={_format_article(r)}  "
            f"md_source={r['md_source_name']}  "
            f"statute_choice={_format_statute_choice_form(r)!r}"
        )
    return "\n".join(lines)


def _format_not_found_response(norm: NormalizedCharge) -> str:
    lines = [
        "## status: not_found",
        "## stage: lookup",
        f"## charge: {norm.raw_key}",
        f"- {_NOT_FOUND_HINT}",
        "- 양형기준 비등재 (48 카테고리 미등재) — 권고 적용 안 됨.",
    ]
    return "\n".join(lines)


# ---------- numeric charge (charge_numeric) ----------
#
# Much of the traffic that found nothing had an integer in `charge` rather
# than an offence name — a number an earlier tool had handed the caller: an
# article number from `statute_lookup` (charge=[299,298,297] straight after
# articles=['297'..'300']), or a charge_id from `sentence_statistics`
# (charge=[1155], 준강제추행, straight after its candidate list). Answering
# with the ordinary not_found ("no sentencing guideline") reads as "this
# offence has no guideline", and callers repeated the same number across
# turns — five in a row, observed. A distinct status breaks that loop.
#
# ⚠ The number is not resolved and carried forward. The id spaces overlap:
# 형법 §298 is 강제추행 while `sentence_statistics` charge_id 298 is 뇌물수수,
# so guessing produces a plausible wrong answer. Offering candidates by
# reverse article lookup would be possible, but comes second — only if this
# wording turns out not to work.

# Cap on a charge list, the same policy value as statute_lookup's "answer the
# first eight".
_MULTI_CHARGE_ITEM_CAP = 8


def _format_multiple_charges_response(
    conn: sqlite3.Connection, items: list[str]
) -> str:
    """A list of charges: report per-item matching rather than looking up the
    joined string, and send the caller back one charge at a time.

    Joining through `coerce_str` ("주거침입, 퇴거불응") makes a single key that
    cannot exist, and the call fell through to not_found. Of eighteen calls
    that produced nothing in one day, fourteen had this shape: the candidates
    held the right answer for each part, but nothing said one charge per call,
    so half of them repeated the list until they gave up. Saying which items
    match exactly removes that round trip, and closes the path from an empty
    result into a numeric sweep (charge=[1..10] eight times in a row, straight
    after one such empty result).
    """
    shown = items[:_MULTI_CHARGE_ITEM_CAP]
    lines = [
        "## status: multiple_charges",
        "## stage: lookup",
        f"## charge: {', '.join(shown)}",
        "- 이 도구는 한 호출에 죄명 하나만 계산합니다 — 아래 죄명별로 각각 호출하세요.",
        "- 리스트 인덱스(0, 1 등)가 아니라, charge='사기' 처럼 죄명 문자열로 재호출해야 합니다.",
        "- 여러 죄의 경합(§37)은 각 죄의 계산에서 statutory_modifications=['경합범_가중'] 로,",
        " 같은 죄 여러 행위는 act_count 로 반영합니다.",
        "## 죄명별 매칭",
    ]
    for item in shown:
        norm = _normalize_charge(conn, item)
        # Digits left after the quotes come off ('"298"') take the numeric
        # hint, exactly as they do on the single-charge path.
        if _numeric_charge_tokens(item) is not None or _numeric_charge_tokens(norm.key) is not None:
            lines.append(f"- {item}: 숫자 — 죄명 문자열을 넣으세요")
            continue
        result = _lookup_charge(conn, norm.key)
        # Re-call on raw_key, not norm.key: the latter has the suffix split
        # off, so '살인미수' would come back as '살인' and calling that loses
        # the attempt reduction.
        if result.status == "exact":
            lines.append(
                f"- {norm.key}: 매칭 ok — charge='{norm.raw_key}' 로 재호출"
                f" ({_format_article(result.rows[0])})")
        elif result.status in ("exact_cross_cat", "exact_same_cat_multi_row"):
            lines.append(
                f"- {norm.key}: 매칭 ok(조항·카테고리 후보 여러 개) —"
                f" charge='{norm.raw_key}' 로 재호출하면 후보를 안내합니다")
        elif result.status == "fuzzy_candidates":
            cands = ", ".join(c["charge_key"] for c in result.candidates[:3])
            lines.append(f"- {norm.key}: 정확 일치 없음 — 유사 후보: {cands}")
        else:
            lines.append(
                f"- {norm.key}: 양형기준 비등재(권고 없음) — 실선고 분포는"
                f" sentence_statistics(charges='{norm.key}') 로 확인 가능")
    if len(items) > len(shown):
        lines.append(
            f"- (나머지 {len(items) - len(shown)}개 생략: {', '.join(items[len(shown):])})")
    return "\n".join(lines)


_NUMERIC_CHARGE_TOKEN_RE = re.compile(r"^\d{1,5}(?:(?:의|-)\d{1,3})?$")
_NUMERIC_CHARGE_SEP_RE = re.compile(r"[\s\[\]()'\"‚,，·;/]+")


def _numeric_charge_tokens(charge: str) -> list[str] | None:
    """The tokens, if `charge` is nothing but numbers; None otherwise.

    One letter anywhere — Hangul or Latin — makes it an offence name and
    leaves it to the ordinary lookup. Branch forms like '297의2' count as
    numeric.
    """
    parts = [p for p in _NUMERIC_CHARGE_SEP_RE.split(charge) if p]
    if not parts or not all(_NUMERIC_CHARGE_TOKEN_RE.match(p) for p in parts):
        return None
    return list(dict.fromkeys(parts))[:8]


def _format_charge_numeric_response(charge: str, tokens: list[str]) -> str:
    # The correction and a way out, nothing else. Enumerating the ID spaces
    # ("not an article number, not a charge_id, not a leaf id") was removed:
    # naming IDs in a failure response reinforces the ID frame and invites the
    # numeric sweep it exists to stop. The docstring Args owns that lesson.
    lines = [
        "## status: charge_numeric",
        "## stage: lookup",
        f"## charge: {charge}",
        "- 숫자가 아니라 죄명 문자열을 넣으세요."
        " 예: charge='강제추행', charge='도로교통법위반(음주운전)'.",
    ]
    if tokens and all(t.isdigit() and int(t) < 10 for t in tokens):
        lines.append("- 리스트 인덱스(0, 1 등)는 죄명이 아닙니다. 앞서 안내된 charge='죄명' 문자열을 넣으세요.")
    elif len(tokens) > 1:
        lines.append("- 죄명은 호출당 하나입니다 — 여러 죄는 각각 호출하세요.")
    lines.append(
        "- 정확한 죄명 표기를 모르면 sentence_statistics(charges=키워드) 로 후보를 찾으세요."
    )
    return "\n".join(lines)


def _coerce_charge_arg(v: object) -> object:
    """Fold list-form charge arguments into strings before Pydantic validation (BeforeValidator for ChargeArg).

    The wire schema exposes only `string`, but when models send a list despite the schema,
    this folds single elements or JSON arrays before schema validation fails, allowing
    the tool body to guide the user via multiple_charges or charge_numeric guidance.
    """
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            item = v[0]
            if item is None or isinstance(item, str):
                return item
            return json.dumps(item, ensure_ascii=False)
        return json.dumps(list(v), ensure_ascii=False)
    return v


# Single canonical definition for the charge argument — used by both agents and MCP server.
# Appears on the wire as `{"type": "string"}` + required, while incoming lists are folded
# before validation by BeforeValidator.
#
# Empirical benchmark (17-parameter full schema):
# - string|array|null: produced garbage list [1,2], [{...}], null (5/8 failures)
# - string|array required: produced [1,2], [3,4,5] (2/8 failures)
# - string|array(items:string) required: produced [''] (2/8 failures)
# - string required: 0/8 failures (models call tool in parallel for multiple charges)
# The array branch was the primary cause of model confusion.
ChargeArg = Annotated[str, BeforeValidator(_coerce_charge_arg)]


def _coerce_optional_str_arg(v: object) -> object:
    """Fold list or numeric arguments into strings before Pydantic validation (BeforeValidator for OptStrArg).

    The wire schema is string|null, but values deviating from the schema are folded rather than
    failing schema validation. Lists fold to single items or comma-joined text. Numbers become str.
    None and strings are passed through untouched.
    """
    def text(x: object) -> str:
        if isinstance(x, str):
            return x
        if isinstance(x, (dict, list, tuple)):
            return json.dumps(x, ensure_ascii=False)
        return str(x)
    if isinstance(v, (list, tuple)):
        items = [x for x in v if x is not None]
        if not items:
            return None
        return text(items[0]) if len(items) == 1 else ", ".join(text(x) for x in items)
    if v is None or isinstance(v, str):
        return v
    return text(v)


# Wire schema for 5 optional string arguments (statute_choice, branch_key, reference_choice,
# guideline_type, offense_date): `anyOf[string, null]` with default null.
# An array branch caused models (e.g. Grok) to send [1] or [{...}]. Narrowing to string|null eliminates
# the garbage while BeforeValidator preserves robustness against clients sending arrays.
OptStrArg = Annotated[str | None, BeforeValidator(_coerce_optional_str_arg)]


# ---------- public tool ----------

@dedup_guard("compute_sentencing_range")
def compute_sentencing_range(
    ctx: RunContext[HarnessDeps],
    # charge is ChargeArg (above) — wire schema string and required, lists folded before validation.
    # The 5 optional string arguments use OptStrArg (above) — wire schema string|null, lists folded before validation.
    # Integer/dict parameters remain wide to accept loose model outputs and normalize via coerce_*.
    charge: ChargeArg,
    sg_category_id: int | str | list | None = None,
    statute_choice: OptStrArg = None,
    branch_key: OptStrArg = None,
    reference_choice: OptStrArg = None,
    is_attempted: bool = False,
    is_accessory: bool = False,
    is_solicitor: bool = False,
    # Subsequent stage parameters. Wide types normalized inside body.
    statutory_modifications: list | dict | str | None = None,
    guideline_leaf_id: int | str | list | None = None,
    guideline_type: OptStrArg = None,
    guideline_factors: dict | str | list | None = None,
    sentence_months: int | str | list | None = None,
    fine_amount: int | str | list | None = None,
    probation_factors: dict | str | list | None = None,
    act_count: int | str | list = 1,
    offense_date: OptStrArg = None,
) -> str:
    """통합 양형 도구 — 죄명에서 출발해 법정형 → 처단형 → 권고형(양형기준) → 선고 검증까지 단계별 계산.

    언제: 형량 판단 기준·범위 확인(공식 양형기준 '범위', 예측 아님). 실선고 분포는 sentence_statistics, 유사 판결은 precedent_search 교차 확인.

    규칙 — 인자를 채울수록 깊은 단계로 자동 진행:
    - charge 만 → **lookup**: 법정형, 양형기준 권고형 범위, 가중·감경 인자 enum.
    - + statutory_modifications → **처단형**: 형법 §56 순서로 가중·감경 적용 trace.
    - + guideline_type(또는 guideline_leaf_id)·guideline_factors → **권고형**: 특별인자로 감경·기본·가중 영역 결정.
    - + sentence_months/fine_amount(+probation_factors) → **final**: 선고형 검증 + 집행유예 4분면.
    - 후속 단계 값은 이전 단계 응답 enum 의 key 만 사용하고 추측하지 마세요.
    - 이 도구는 호출 사이 상태를 보존하지 않습니다. 재호출 시 charge와 확정 인자를 모두 반복하고 새 인자를 추가하세요.

    응답: markdown-KV(`## section` + `- key: value`). `출처`(양형기준 해설서 PDF 링크)가 있으면 함께 제시하세요.

    Args:
      charge: 판결문 form 죄명 (예: 살인, 도로교통법위반(음주운전)). 정규화는 도구 내부에서 처리.
        **한 호출에 하나** — 여러 죄는 죄명별로 각각 호출. 리스트로 오면 계산하지 않고
        죄명별 매칭 현황으로 유도한다(multiple_charges). 경합 사안은 죄명별로 각각 계산.
        **숫자·ID 불가** — 조문 번호나 charge_id 가 아닌 한국어 죄명 문자열을 넣으세요.
        숫자가 오면 계산하지 않고 재호출을 유도한다(charge_numeric).
      offense_date: 행위 일자 (예: '2013.7.30', '2013-07-30', '20130730'). 지정 시
        행위시 조문 본문·시점본 정량 반영 — 형법 §1 ① "범죄의 성립과 처벌은 행위시의
        법률에 의한다" 원칙. 미지정 시 현행 기준.
      sg_category_id: 동일 charge_key 가 복수 카테고리일 때 명시 (ambiguous_category 응답이 후보 제공).
      statute_choice: 같은 sg_category 안에 여러 row (ambiguous_statute 응답) 일 때
        구체 조항 명시 (예: "형법§257", "폭력행위등처벌에관한법률§2③").
      branch_key: 분기 조항의 선택지 — needs_branch_key 응답 목록의 키를 문자열 하나로 그대로
        (예: "1", "3-1"). 항 기호 표기("①", "③-1")도 같은 뜻으로 받는다.
      reference_choice: 가중·준용·분기 row 의 원범죄 명시 (예: "형법§347") — 응답이 안내.
      is_attempted: 미수 명시. 죄명 접미("살인미수" 등)로도 자동 인식되며, 접미와 이 플래그
        중 하나만 있어도 적용됩니다(OR).
      is_accessory: 방조 명시 — 인식 규칙은 is_attempted 와 동일.
      is_solicitor: 교사 명시 — 인식 규칙은 is_attempted 와 동일.
      statutory_modifications: 형법 §56 가중·감경 사유 list — lookup 응답의 enum에서 선택.
        지정 시 처단형 단계 진입.
      guideline_type: 양형기준 권고형 유형 명칭 — lookup 응답 목록의 **따옴표 안 명칭 그대로**
        (예: "일반사기 1유형", "일반적인 상해 1유형"; 같은 이름이 둘이면 목록이 괄호 기준까지 붙여
        준다). 지정 시 권고형 단계 진입. 축약·번호만도 하나로 정해지면 통하고, 여럿에 걸리면
        계산 대신 후보를 되돌려준다.
      guideline_leaf_id: (하위 호환) 양형기준 leaf id. guideline_type 권장.
      guideline_factors: 특별 가중·감경 인자 dict — lookup 응답의 인자 enum에서 선택.
      sentence_months: 검증할 선고형(자유형, 월 단위) — 지정 시 final 단계(선고 가능 범위 검증).
      fine_amount: 검증할 선고형(벌금, 원 단위) — sentence_months 와 같은 final 단계 진입.
      probation_factors: 집행유예 4분면 인자 dict — lookup 응답의 enum에서 선택.
      act_count: 같은 charge 의 별개 행위 수 (동종 다행위). >=2 면 §37 전단·§38 ① 2호
        경합범 가중을 처단형에 자동 적용 (자유형 장기 1/2 가중, §42 ② 50년 cap).
        statutory_modifications 에 경합범_가중 항목을 명시하면 그 명시가 우선.
        포괄일죄·영업범처럼 판결문이 §37 을 명시하지 않는 유형은 1로 두세요.
    """
    # Normalise everything the wide signature let through.
    # A list of two or more charges is caught before `coerce_str` joins it:
    # the joined string is a single key that cannot exist, and the call falls
    # through to not_found (see `_format_multiple_charges_response`).
    charge_items = None
    if isinstance(charge, str) and charge.lstrip()[:1] == "[":
        parsed = coerce_list(charge)   # a list written as text ('["주거침입","퇴거불응"]')
        if parsed is not None:
            charge = parsed
    list_notes: list[str] = []  # Discarded items from charge list, reported back to caller
    if isinstance(charge, (list, tuple)):
        deduped = list(dict.fromkeys(
            s for s in (coerce_str(x) for x in charge) if s))
        non_numeric = [s for s in deduped if _numeric_charge_tokens(s) is None]
        numeric_items = [s for s in deduped if _numeric_charge_tokens(s) is not None]

        if len(non_numeric) == 1 and len(numeric_items) >= 1:
            # Single non-numeric charge combined with numbers: use the charge and discard numbers.
            # Do not guess that the number is a guideline type — report what was discarded.
            charge = non_numeric[0]
            list_notes.append(
                f"- charge 리스트의 숫자 {', '.join(numeric_items)} 은 무시했습니다 — "
                "죄명은 문자열 하나. 유형은 guideline_type 으로 지정하세요."
            )
        elif len(deduped) >= 2:
            charge_items = deduped
        elif deduped:
            charge = deduped[0]  # one left after dedup — use it, don't join
    charge = coerce_str(charge)
    sg_category_id = coerce_int(sg_category_id)
    statute_choice = coerce_str(statute_choice)
    branch_key = coerce_str(branch_key)
    reference_choice = coerce_str(reference_choice)
    statutory_modifications = coerce_dict_list(statutory_modifications)  # only dict elements
    guideline_leaf_id = coerce_int(guideline_leaf_id)
    guideline_type = coerce_str(guideline_type)
    guideline_factors = coerce_dict(guideline_factors)
    sentence_months = coerce_int(sentence_months)
    fine_amount = coerce_int(fine_amount)
    probation_factors = coerce_dict(probation_factors)
    offense_date = coerce_str(offense_date)
    act_count = coerce_int(act_count) or 1  # scalar/array/None defense — default 1

    if not charge or charge.strip() in ("{}", "None", "null", "NoneType"):
        return (
            "## status: missing_input\n"
            "- charge 인자 필요. 판결문 form 죄명 (예: 살인, 도로교통법위반(음주운전))."
        )

    numeric_tokens = _numeric_charge_tokens(charge)
    if numeric_tokens is not None:
        return _format_charge_numeric_response(charge, numeric_tokens)

    conn = open_db()
    try:
        if charge_items is not None:
            return _format_multiple_charges_response(conn, charge_items)
        norm = _normalize_charge(
            conn,
            charge,
            is_attempted=is_attempted,
            is_accessory=is_accessory,
            is_solicitor=is_solicitor,
        )
        # Stripping the tag and the quotes can leave nothing but digits
        # ('"298"'). Letting that through as an unlisted charge revives the
        # "no guideline for this offence" misreading that charge_numeric cuts,
        # so it goes to the same hint.
        residual_numeric = _numeric_charge_tokens(norm.key)
        if residual_numeric is not None:
            return _format_charge_numeric_response(charge, residual_numeric)
        result = _lookup_charge(conn, norm.key, sg_category_id=sg_category_id)

        # A stated choice resolves an otherwise ambiguous match.
        if statute_choice and result.status in (
            "exact_cross_cat", "exact_same_cat_multi_row"
        ):
            chosen = _parse_statute_choice(statute_choice, result.rows)
            if chosen is None:
                return _format_invalid_statute_choice_response(
                    norm, result.rows, statute_choice
                )
            result = LookupResult(status="exact", rows=[chosen])

        if result.status == "exact":
            row = result.rows[0]
            payload_row = _resolve_alias(conn, row)
            # With an offence date, overlay the bounds in force then.
            offense_iso = to_iso_date(offense_date)
            payload_row, version_meta, art42_trace = _get_versioned_payload(
                conn, payload_row, offense_iso
            )
            penalty, pending = _resolve_payload(
                conn,
                payload_row,
                branch_key=branch_key,
                reference_choice=reference_choice,
            )
            # Record that the historical bounds were used.
            if version_meta and penalty:
                if version_meta.get('empty_version'):
                    # Empty historical row: say that current bounds were used.
                    penalty.trace.insert(0,
                        f"## ⚠ 시점본 미적재: clm_versions {version_meta['effective_date']} "
                        f"행 정량 없음 — offense_date={offense_iso} 행위시 정량 미상, "
                        f"현행 base 정량 사용 (데이터 검증 필요)")
                else:
                    penalty.trace.insert(0,
                        f"## 적용 시점본: {version_meta['effective_date']} "
                        f"(matched=§{version_meta['matched_paragraph'] or '?'}, "
                        f"confidence={version_meta['match_confidence']}) "
                        f"— offense_date={offense_iso} 행위시 정량")
                    penalty.source = f"versioned:{version_meta['effective_date']}"
            # Record any correction to the general-part ceiling.
            if art42_trace and penalty:
                penalty.trace.insert(0, art42_trace)
            # Provision text, appended to every stage response.
            appendix = _historic_appendix(conn, payload_row, offense_date)

            if pending is not None:
                return _format_pending_response(norm, row, payload_row, pending) + appendix

            # The stage advances as arguments arrive. Multiple offences
            # aggravate the processed range but do not by themselves
            # advance the stage: that would skip the lookup response the
            # caller needs in order to fill in the next arguments.
            #
            # A modifier in the charge name — "attempted murder" — does not
            # advance the stage either. It used to: the response skipped the
            # lookup stage, so the caller never saw the factor lists, and it
            # invented argument names on the next call which the tool then
            # ignored. The mitigation an attempt implies is still applied
            # once the stage is reached properly.
            resolved_leaf_id, leaf_notes = _resolve_leaf(
                conn, payload_row["sg_category_id"], guideline_leaf_id, guideline_type
            )
            guide_notes = list_notes + leaf_notes
            leaf_label = _get_leaf_label(conn, resolved_leaf_id) if resolved_leaf_id else ""
            needs_processed = statutory_modifications is not None
            needs_recommended = resolved_leaf_id is not None
            needs_final = sentence_months is not None or fine_amount is not None

            if not (needs_processed or needs_recommended or needs_final):
                return _format_lookup_response(
                    conn, norm, row, payload_row, penalty, act_count=act_count,
                    guide_notes=guide_notes,
                ) + appendix

            # Always computed: the later stages build on it.
            processed = _apply_statutory_modifications(
                penalty, norm, statutory_modifications, act_count=act_count,
                offense_iso=offense_iso,
            )

            if not (needs_recommended or needs_final):
                return _format_processed_response(
                    norm, row, payload_row, penalty, processed, guide_notes=guide_notes,
                ) + appendix

            # The guideline recommendation.
            factors = _convert_factors_to_applied(guideline_factors)
            floor = _get_statute_floor(processed)
            rec = determine_range(
                conn, resolved_leaf_id, factors,
                legal_floor_months=floor,
                is_attempted=norm.modifiers.get("is_attempted", False),
            ) if resolved_leaf_id is not None else None
            intersect = _intersect_with_processed(rec, processed)

            if not needs_final:
                return _format_recommended_response(
                    norm,
                    row,
                    payload_row,
                    penalty,
                    processed,
                    rec,
                    intersect,
                    resolved_leaf_id,
                    floor,
                    leaf_label=leaf_label,
                    guide_notes=guide_notes,
                ) + appendix

            # final stage
            verify_lines = _verify_sentence(
                sentence_months, fine_amount, processed, rec, intersect
            )
            probation_lines = _probation_recommendation(
                sentence_months, fine_amount, processed, probation_factors
            )
            # A fine expressed only in the provision text: quote it.
            fine_paragraphs: list[str] = []
            if processed.fine_min_won is None and processed.fine_max_won is None:
                fine_paragraphs = _extract_fine_paragraphs(
                    conn,
                    payload_row["statute_id"],
                    payload_row["article_no_num"],
                    payload_row["article_branch"],
                )
            # Whether mitigation applied, noted beside the fine.
            mit_applied = any(
                _kind_is_mit(m.get("kind", "")) and m.get("applied", True)
                for m in (statutory_modifications or [])
            )
            return _format_final_response(
                norm,
                row,
                payload_row,
                penalty,
                processed,
                rec,
                intersect,
                resolved_leaf_id or -1,
                floor,
                verify_lines,
                probation_lines,
                fine_paragraphs=fine_paragraphs,
                mit_applied=mit_applied,
                leaf_label=leaf_label,
                guide_notes=guide_notes,
            ) + appendix

        if result.status == "exact_cross_cat":
            return _format_cross_cat_response(norm, result.rows)

        if result.status == "exact_same_cat_multi_row":
            return _format_same_cat_multi_row_response(norm, result.rows)

        if result.status == "exact_wrong_category":
            return _format_wrong_cat_response(norm, result.rows, sg_category_id)

        if result.status == "fuzzy_candidates":
            return _format_fuzzy_response(norm, result.candidates)

        return _format_not_found_response(norm)
    finally:
        conn.close()


# Worked examples. Run this module directly to exercise every resolution
# path against the corpus:
#   python -m lawful_mcp.tools.compute_sentencing_range

if __name__ == "__main__":
    from collections import deque
    from types import SimpleNamespace

    fake_deps = SimpleNamespace(recent_calls=deque(maxlen=10))
    fake_ctx = SimpleNamespace(deps=fake_deps)

    cases = [
        # Exact match, ordinary case.
        {"charge": "살인"},
        # Modifier split off the charge name.
        {"charge": "살인미수"},
        # Aggravation resolving to a single underlying offence.
        {"charge": "상습공갈"},
        # Reference with several candidates: the tool asks.
        {"charge": "위조공문서행사"},
        # Same, with the choice supplied.
        {"charge": "위조공문서행사", "reference_choice": "형법§225"},
        # Branching reference with candidates.
        {"charge": "준강도"},
        # Same, with the choice supplied.
        {"charge": "준강도", "reference_choice": "형법§333"},
        # A provision that branches internally.
        {"charge": "도로교통법위반(음주운전)"},
        # branch_options + branch_key
        {"charge": "도로교통법위반(음주운전)", "branch_key": "③2"},
        # branch_options + invalid branch_key
        {"charge": "도로교통법위반(음주운전)", "branch_key": "③9"},
        # Repeat-offence aggravation on a branching provision.
        {"charge": "폭력행위등처벌에관한법률위반(공갈)"},
        # Same, with the branch supplied.
        {"charge": "폭력행위등처벌에관한법률위반(공갈)", "branch_key": "③3"},
        # alias
        {"charge": "공용서류손상"},
        # cross-cat
        {"charge": "강도상해"},
        # Charge found in another category, resolved by naming it.
        {"charge": "강도상해", "sg_category_id": 26},
        # fuzzy
        {"charge": "도로교통법위반"},
        # not_found
        {"charge": "음악산업진흥에관한법률위반"},
        # Processed range: discretionary mitigation for surrender.
        {
            "charge": "살인",
            "statutory_modifications": [
                {
                    "kind": "법률상_임의감경",
                    "type": "자수 (§52)",
                    "basis": "형법 §52 ①",
                    "applied": True,
                },
            ],
        },
        # Repeat-offence aggravation and mitigation together, in order.
        {
            "charge": "절도",
            "statutory_modifications": [
                {"kind": "누범_가중", "type": "누범 (§35)", "basis": "형법 §35", "applied": True},
                {"kind": "법률상_임의감경", "type": "자수 (§52)", "basis": "형법 §52 ①", "applied": True},
            ],
        },
        # Attempt, taken from the charge name.
        {"charge": "살인미수"},
        # An adjustment supplied but marked as not applied.
        {
            "charge": "상습공갈",
            "statutory_modifications": [
                {
                    "kind": "작량감경",
                    "type": "작량감경 (§53)",
                    "basis": "형법 §53",
                    "applied": True,
                },
            ],
        },
        # Recommended range from a guideline leaf and its factors.
        {
            "charge": "살인",
            "guideline_leaf_id": 216,
            "guideline_factors": {
                "special_act_aggravators": [],
                "special_act_mitigators": ["피해자 유발"],
                "special_actor_aggravators": [],
                "special_actor_mitigators": ["자수", "처벌불원"],
            },
        },
        # Recommended range on top of a mitigated processed range.
        {
            "charge": "살인",
            "statutory_modifications": [
                {
                    "kind": "법률상_임의감경",
                    "type": "자수 (§52)",
                    "basis": "형법 §52 ①",
                    "applied": True,
                },
            ],
            "guideline_leaf_id": 216,
            "guideline_factors": {
                "special_act_mitigators": ["피해자 유발"],
                "special_actor_mitigators": ["자수", "처벌불원"],
            },
        },
        # Final stage: a proposed sentence, checked, with suspension.
        {
            "charge": "살인",
            "guideline_leaf_id": 216,
            "guideline_factors": {
                "special_act_mitigators": ["피해자 유발"],
                "special_actor_mitigators": ["자수", "처벌불원"],
            },
            "sentence_months": 84,
            "probation_factors": {
                "major_positive": ["진지한 반성"],
                "major_negative": [],
                "general_positive": [],
                "general_negative": [],
            },
        },
        # Eligible for suspension, decided on the ordinary factors.
        {
            "charge": "절도",
            "guideline_leaf_id": 216,  # 임의 (절도 leaf 모름 — None 으로도 OK)
            "sentence_months": 24,
            "probation_factors": {
                "major_positive": [],
                "major_negative": [],
                "general_positive": ["피해 회복"],
                "general_negative": [],
            },
        },
        # empty
        {"charge": ""},
    ]
    for case in cases:
        print(f"\n{'='*60}\nCASE: {case}\n{'='*60}")
        print(compute_sentencing_range(fake_ctx, **case))
