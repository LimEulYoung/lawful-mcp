"""compute_sentencing_range — 통합 양형 도구.

§09 §3.1 의 단일 진입점. 단계별 *선택* 입력으로 4 응답 stage 자동 분기:

  lookup   — charge only           → 법정형 + leaf 후보 + enum
  처단형    — + statutory_modifications → + 형법 §56 trace
  권고형    — + leaf_id + factors  → + 누진 룰 적용 영역
  final    — + sentence_months + probation_factors → + 선고 검증 + 집유

본 파일은 *점진 구현*:
  Phase A — lookup spine (charge_key normalize + DB lookup + fuzzy)
  Phase B — reference_mode 6 종 처리 + branch_options runtime resolution
  Phase C — 처단형 / 권고형 / 선고형 / 집유 (후속)
  Phase D — 평가셋 50 case spot-check (후속)

도구 데이터 source:
  data/databases/harness.db / charge_legal_map (882 row, Phase 3 적재)
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field

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
from ._coerce import coerce_dict, coerce_dict_list, coerce_int, coerce_str, to_iso_date
from ._dedup import dedup_guard

_log = logging.getLogger(__name__)


def _rid(row) -> str:
    """row id 안전 추출 (다양한 SELECT 출처 대비) — DATA 경고 로그용."""
    try:
        return str(row["id"])
    except (IndexError, KeyError, TypeError):
        return "?"


# reference_mode 의 알려진 전체 집합 + catch-all(_resolve §4)의 *정당한* fall-through 값.
# 미지 값이 catch-all 로 조용히 떨어지면 가중/준용이 미적용된 raw 정량이 에러 없이
# 나간다 → _resolve 에서 미지 값을 trace 에 명시 표시한다(L1 견고성).
_DIRECT_FALLTHROUGH_MODES = {None, "정보", "누범가중"}


def _b(x) -> str:
    """None-safe 정량 표시 — trace·검증·표시 출력에 'None' 문자열 누출 방지. None→'?'."""
    return "?" if x is None else str(x)


def _rec_range_line(rec) -> str:
    """권고 형량범위 한 줄. RecommendedRange.min/max=None 은 '하한/상한 없음'(open bound).
    *둘 다* None 이면 무경계(산출 불가) — 예: 사형·무기 전용 시점본에 leaf 적용 · leaf 범위
    부적용. 이때 `?~?월` 누출(R 배터리 leak) 대신 명시. 단일 open bound 는 기존대로 '?'."""
    if rec.min_months is None and rec.max_months is None:
        note = " (사형·무기형)" if rec.has_life else ""
        return f"- 범위: 권고 형량범위 없음 (상·하한 미산출){note}"
    return (f"- 범위: {_b(rec.min_months)}~{_b(rec.max_months)}월"
            + (" (life)" if rec.has_life else ""))

# ---------- charge_key normalize + suffix split ----------

# 양형기준 미등재 — 본 도구는 lookup spine 만 책임. statute_lookup 으로 안내.
_NOT_FOUND_HINT = (
    "양형기준 비등재 가능성. 법정형 조회는 `statute_lookup` 도구 사용."
)

# 미수/교사/방조: 매핑 row 없음 (882 row 전수 확인). LLM 이 형태 그대로
# 호출하면 *suffix 분리* 후 parent row 매칭 시도.
_SUFFIX_MODIFIERS: list[tuple[str, str]] = [
    ("미수", "is_attempted"),
    ("교사", "is_solicitor"),
    ("방조", "is_accessory"),
]


@dataclass(frozen=True)
class NormalizedCharge:
    """charge_key 정규화 + suffix 분리 결과."""

    key: str                        # 정규화된 lookup 키 (suffix 제거 가능)
    raw_key: str                    # charge_key() 만 적용 (suffix 미제거)
    modifiers: dict[str, bool]      # {is_attempted, is_accessory, is_solicitor}
    suffix_split_applied: bool      # 자동 분리 발생 여부 (응답 trace)


def _normalize_charge(
    conn: sqlite3.Connection,
    charge: str,
    *,
    is_attempted: bool = False,
    is_accessory: bool = False,
    is_solicitor: bool = False,
) -> NormalizedCharge:
    """LLM 입력 charge 를 매핑 lookup 키 + modifier 로 분해.

    Rules:
      1. `charge_key()` 로 1차 정규화 (공백·dot 통일, 괄호 보존)
      2. suffix (미수/교사/방조) 가 *charge_key 마지막* 이고 parent (suffix 제거 형태)
         가 매핑에 존재하면 → 분리 + modifier 설정.
      3. 호출자가 명시한 modifier (is_attempted 등) 는 *override* — 자동 분리와 OR.
    """
    raw = _charge_key_normalize(charge or "")
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
        # parent 가 매핑 테이블에 정확히 존재할 때만 분리. 없으면 원본 유지.
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
    # db_hit_count·source 컬럼은 도구가 읽지 않는 dead shadow (적재 차단 — 철학 A).
    # Penalty.source 는 _penalty_from_row 의 source 인자(provenance)지 DB컬럼 아님.
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
    """매핑 lookup — exact 우선, miss 시 substring fuzzy 후보 노출.

    Cross-cat (같은 charge_key, 다른 sg_category_id) 의 경우 sg_category_id 미지정
    이면 *모든 cat row* 반환해 *호출자가 명시 선택* (auto-매칭 금지 원칙).

    sg_category_id 명시됐는데 그 cat 에 없으면 *다른 cat 에 exact* 있는지 별도
    확인 — fuzzy substring 매칭과 *exact-but-wrong-cat* 을 구분 노출.
    """
    if not key:
        return LookupResult(status="not_found")

    # 1. exact match (cat 명시되면 cat 안에서)
    if sg_category_id is not None:
        rows = conn.execute(
            _SELECT_ROW + " WHERE charge_key=? AND sg_category_id=? "
            "ORDER BY statute_id, article_no_num, article_branch, paragraph",
            (key, sg_category_id),
        ).fetchall()
        if rows:
            if len(rows) == 1:
                return LookupResult(status="exact", rows=list(rows))
            # 같은 cat 안에 여러 row (상해 = 형법§257 / 폭처법§2③ 등)
            return LookupResult(status="exact_same_cat_multi_row", rows=list(rows))

        # cat 안 miss → 다른 cat 에 exact 있는지 확인
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
                # 같은 sg_category 안에 여러 row — 카테고리 모호 아님, *조항* 모호
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
    """alias row 가 stat null 이면 alias_of primary 로 fetch.

    882 row 중 일부 alias 는 stats 채워져 있고 (e.g. 공용서류손상→공용서류무효: imp 0~84
    동일), 일부는 null (e.g. 폭력행위등처벌에관한법률위반(상습공갈) — primary 에 의존).
    null 인 경우만 primary follow.
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


# ---------- Phase B — payload resolver (reference / branch) ----------

# ── 형법 총칙(總則) 상한·환산 상수 (§42·§55) ────────────────────────────
# 모든 죄에 적용되는 *보편 규칙* — 특정 죄명·category·비율 같은 데이터(철학 C 의
# 하드코딩 대상)가 아니라 형법총칙 그 자체의 일반 룰이므로 명명 상수로 코드에 둔다.
# DB(st_articles §42) 파생은 *거부*: ① cap 이 한국어 산문("유기는 1개월 이상 30년
# 이하…가중하는 때에는 50년까지")이라 파싱이 fragile ② 시행일(2010.10.16)은 §42 본문에
# 없음(본문엔 '개정 2010.4.15', 시행일은 statute 버전 메타) ③ 매 계산이 의존하는 보편
# 상수를 특정 DB 행 존재·형식에 묶으면 그 행이 바뀔 때 *조용히 틀림* → 철학 B 역행.
# 시점 의존성(구법/신법)은 offense_date 로 이미 파생(_apply_art42_versioned_cap, agg_cap).
_IMP_CAP_MONTHS                = 360   # §42① 유기징역 단독 상한 (30년)
_IMP_CAP_MONTHS_OLD            = 180   # §42① 구법 단독 상한 (15년, <2010.10.16)
_AGGRAVATED_IMP_CAP_MONTHS     = 600   # §42② 가중 상한 (50년)
_AGGRAVATED_IMP_CAP_MONTHS_OLD = 300   # §42② 구법 가중 상한 (25년, <2010.10.16)
_DEATH_COMMUTE_MIN_MONTHS      = 240   # §55①1호 사형 감경 시 유기 하한 (20년)
_LIFE_COMMUTE_MIN_MONTHS       = 120   # §55①2호 무기 감경 시 유기 하한 (10년)
_ART42_REFORM_ISO = "20101016"         # §42 개정 시행일 (행위시법주의 §1① 경계)
# 현행 §42 cap → (구법, 신법) 환산표 (시점보정 _apply_art42_versioned_cap 용).
_ART42_CAPS = {
    _IMP_CAP_MONTHS:            (_IMP_CAP_MONTHS_OLD, _IMP_CAP_MONTHS),
    _AGGRAVATED_IMP_CAP_MONTHS: (_AGGRAVATED_IMP_CAP_MONTHS_OLD, _AGGRAVATED_IMP_CAP_MONTHS),
}


@dataclass
class EffectivePenalty:
    """resolve 후 도구가 노출하는 *최종* 법정형.

    `source` 는 어떻게 결정됐는지 한 줄 설명 — 응답 trace 에 표시.
    `trace` 는 resolution 과정 (가중 적용 / branch 선택 등) 의 다중행 설명.
    `fine_formula` (M15) 는 식 기반 fine 정량 NULL row 의 *식 명세*. dict:
      {"base_label": str, "low_mult": float, "high_mult": float, "is_optional": bool}
    감경·가중 multiplier 가 정량 NULL 대신 *공식의 low/high_mult* 에 적용됨.
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
    """resolve 가 *미완* 인 상태 — LLM 의 추가 인자 필요.

    kind ∈ {branch, branch_invalid, reference, reference_missing, reference_ambiguous,
            modifier_directive}
    """

    kind: str
    options: list[dict] = field(default_factory=list)
    message: str = ""


def _penalty_from_row(row: sqlite3.Row, *, source: str = "direct") -> EffectivePenalty:
    """매핑 row 의 stat_* 필드 → EffectivePenalty."""
    try:
        kinds = json.loads(row["sentence_kind_options"] or "[]")
    except (TypeError, json.JSONDecodeError) as e:
        kinds = []
        if isinstance(e, json.JSONDecodeError):
            _log.warning("⚠ DATA: sentence_kind_options JSON 손상 (id=%s) — 형종 누락", _rid(row))
    # M15: fine_formula JSON 컬럼 — 식 기반 row 의 base_label + low/high_mult
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
    """EffectivePenalty → 한 줄 요약 (trace 용). None-safe: 미명시 정량은 '?',
    정량 없는 형종은 생략. M36: 준용/가중 trace 의 `fine {None:,}` 크래시 +
    `imp None~None월` 누출 해소."""
    parts: list[str] = []
    if p.imp_min_months is not None or p.imp_max_months is not None:
        lo = "?" if p.imp_min_months is None else f"{p.imp_min_months}"
        hi = "?" if p.imp_max_months is None else f"{p.imp_max_months}"
        parts.append(f"imp {lo}~{hi}월")
    if p.fine_min_won is not None or p.fine_max_won is not None:
        lo = "?" if p.fine_min_won is None else f"{p.fine_min_won:,}"
        hi = "?" if p.fine_max_won is None else f"{p.fine_max_won:,}"
        parts.append(f"fine {lo}~{hi}원")
    if p.has_life:
        parts.append("life")
    if p.has_death:
        parts.append("death")
    return ", ".join(parts) if parts else "정량 미상 (원범죄 호별 분기 등)"


def _penalty_from_branch(opt: dict, *, branch_key: str) -> EffectivePenalty:
    """branch_options 의 한 옵션 dict → EffectivePenalty.

    M14: opt 에 sentence_kind_options 가 명시되어 있으면 우선 채택. 정량 NULL 인 식 기반
    fine 옵션 (특가법 §8의2 등 B 그룹) 이 *옵션 자체 제거* 되던 부정확 해소. 동시에 A 그룹
    (일부 branch 가 imp only — 청소년성보호법 §7 ① 강간, 공직선거법 §230 ⑤ 등) 에서
    fine 옵션이 *잘못 부여* 되던 union 보정 (구 M13) 부작용도 해소.

    M15: opt 에 fine_formula 가 있으면 EffectivePenalty 에 전파. branch 마다 fine 식이
    다른 경우 (특가법 §6 등) 대비.

    명시 없는 경우 (legacy fallback): 정량 기반 자동 산정 — NULL = 옵션 없음 단순 룰.
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
    """가중 (1.5) / 준용 (1.0) multiplier 를 *상한* 에 적용.

    형법 §42 ② 가중 50년 cap. 무기·사형 flag 는 그대로 보존 (가중으로 추가 생성 X).
    벌금 상한은 형법 §55 ④ "다액의 2분의 1" → multiplier 1.5 같은 형식. 본 함수는
    multiplier 그대로 적용하고 라운딩은 정수 곱 (필요 시 ceiling).
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


# ---------- reference auto-match + reference_choice 파싱 ----------

# "형법§347" / "형법 §347" / "형법 §347 ②" / "형법§347의2" 모두 매칭.
# statute_name (lazy match, 마지막 § 까지) + article_no (필수) + (의 branch) + (paragraph)
_REF_CHOICE_RE = re.compile(
    r"^\s*(?P<name>[^§\s]+?)\s*§?\s*"
    r"(?P<art>\d+)(?:의(?P<branch>\d+))?"
    r"(?:\s*(?P<para>[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]+|\d+))?\s*$"
)


def _parse_statute_choice(choice: str, rows: list[sqlite3.Row]) -> sqlite3.Row | None:
    """LLM 이 명시한 statute_choice 문자열 → rows 중 매칭 row.

    `_parse_reference_choice` 와 같은 form ("형법§257", "폭력행위등처벌에관한법률§2③" 등).
    `statute_name` 공백 정규화 + `article_no_num` + (optional) `article_branch` + (optional)
    `paragraph` 매칭. 가장 specific 한 row 반환.
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
    """row → choice form ("형법§257", "폭력행위등처벌에관한법률§2③" 등) — 응답 노출용."""
    name = (row["statute_name"] or "").replace(" ", "")
    art = row["article_no_num"]
    branch_str = f"의{row['article_branch']}" if row["article_branch"] else ""
    para_str = f" {row['paragraph']}" if row["paragraph"] else ""
    return f"{name}§{art}{branch_str}{para_str}".strip()


def _parse_reference_choice(choice: str, refs: list[dict]) -> dict | None:
    """LLM 이 명시한 reference_choice 문자열 → refs 중 매칭 ref dict.

    매칭 규칙:
      1. statute_name 정확 일치 (정규화 후) + article_no_num 일치 → 우선
      2. + article_branch 일치 → 더 우선
      3. + paragraph 일치 → 가장 우선
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
    """옵션이 *단 1 개* 일 때만 자동 선택 — 그 외엔 LLM 판단 영역 (needs_choice).

    chapter §1.4 의 원칙 — *도구는 결정론, LLM 은 판단*:
      - 옵션 1 개: 선택지 없음 → 결정론적 (도구 자동)
      - 옵션 ≥ 2: 행위·사실관계 추론 필요 → LLM 결정

    이전 버전의 substring note 매칭 (예: "아동학대" → "학대" §273 자동) 은
    *행위 추론을 도구가 가로채는 over-fit* 위험이라 차단. *상습공갈 → 공갈* 같은
    적절한 패턴까지 needs_choice 가 되지만, *원칙 일관성* + *under/over-fit 원천
    차단* 측면에서 trade-off 수용.
    """
    if not refs:
        return None
    if len(refs) == 1:
        return refs[0]
    return None


# ---------- ref → parent row fetch ----------

# paragraph normalization: 매핑 row 는 circled (①②③) 사용. reference_articles 일부 row
# 는 digit ("1", "2") 형태. lookup 시 양쪽 다 시도.
_PARA_DIGIT_TO_CIRCLED = {
    "1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤",
    "6": "⑥", "7": "⑦", "8": "⑧", "9": "⑨", "10": "⑩",
    "11": "⑪", "12": "⑫", "13": "⑬", "14": "⑭", "15": "⑮",
}


def _normalize_paragraph(para: str | None) -> list[str | None]:
    """paragraph 후보 list — 매칭 시 try 순서."""
    if para is None or para == "":
        return [None]
    p = para.strip()
    candidates: list[str | None] = [p]
    if p in _PARA_DIGIT_TO_CIRCLED:
        candidates.append(_PARA_DIGIT_TO_CIRCLED[p])
    # 반대 방향: circled → digit
    for d, c in _PARA_DIGIT_TO_CIRCLED.items():
        if p == c:
            candidates.append(d)
            break
    return candidates


def _lookup_by_article(
    conn: sqlite3.Connection, ref: dict
) -> sqlite3.Row | None:
    """reference_articles entry → charge_legal_map 의 *원범죄* row.

    UNIQUE 키 (statute_id, article_no_num, article_branch, paragraph) 매칭. paragraph 가
    digit("1") vs circled("①") 형식 차이 있으면 양쪽 다 시도.

    statute_name 은 *공백 정규화 후* 매칭 — 매핑 row 와 reference_articles 의 표기 차이
    (예: "특정경제범죄가중처벌등에관한법률" vs "특정경제범죄 가중처벌 등에 관한 법률") 흡수.
    """
    name = (ref.get("statute_name") or "").strip()
    art = ref.get("article_no_num")
    branch = ref.get("article_branch") or 0
    para = ref.get("paragraph")

    if art is None or not name:
        return None

    name_norm = name.replace(" ", "")

    # paragraph 후보 양쪽 (digit + circled) 모두 시도
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

    # paragraph 명시됐는데 다 miss → paragraph 무시하고 article 매칭
    # (예: refs paragraph="1" 인데 매핑 row 는 paragraph=NULL 인 케이스 — 조 전체 적용)
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
    """row + 인자 → 최종 EffectivePenalty 또는 PendingResolution.

    Returns:
      (penalty, None) — resolution 완료
      (None, pending) — LLM 의 추가 인자 (branch_key / reference_choice) 필요
    """
    rm = row["reference_mode"]

    # M30 — 시점본 정량 채워졌으면 준용 분기 skip.
    # clm_versions 의 준용 row 가 *원범죄 시점본 정량* 을 직접 추출해 채움 (Gemini).
    # 도구가 다시 *원범죄 현행 정량* lookup 하면 시점본 swap 효과 무효화 → 현행 덮어씀.
    # 정량이 채워졌다는 건 *원범죄 시점본 적용 완료* 의미 — 그대로 사용.
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

    # 1. branch_options (has_conditional_branch=1) — 본문 직접 정량 분기
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
        chosen = next((o for o in opts if o.get("key") == branch_key), None)
        if chosen is None:
            return None, PendingResolution(
                kind="branch_invalid",
                options=opts,
                message=f"branch_key={branch_key!r} 는 유효하지 않음.",
            )
        penalty = _penalty_from_branch(chosen, branch_key=branch_key)
        # M14: 구 M13 의 raw row union 보정 제거. branch_options 각 옵션이 자체
        # sentence_kind_options 명시 (M14 patch) 하므로 union 불필요. A 그룹 (일부 branch
        # 만 imp only) 에 raw 의 fine kind 가 잘못 부여되던 부작용도 동시 해소.
        # M15: branch 자체에 fine_formula 없으면 row top-level fine_formula 로 fallback
        # (특가법 §8의2 ① 처럼 row 단위 식 공유 case 다수).
        if penalty.fine_formula is None:
            try:
                raw_formula = row["fine_formula"]
                if raw_formula:
                    penalty.fine_formula = json.loads(raw_formula)
            except (TypeError, json.JSONDecodeError, IndexError):
                pass
        penalty.trace.append(
            f"## branch_key 선택: {branch_key} — {chosen.get('cond', '')[:80]}"
        )
        return penalty, None

    # 2. reference_mode 가중 / 준용 / 공동가중 — 원범죄 + multiplier
    if rm in ("가중", "준용", "공동가중"):
        try:
            refs = json.loads(row["reference_articles"] or "[]")
        except (TypeError, json.JSONDecodeError):
            refs = []

        # 가중 *수식어* (modifier-directive) — underlying base 죄가 *open*(여러 법·무한정)이라
        # 목록화 불가능한 가중 규정 (예: 아청법 §18 신고의무자 성범죄 = 어떤 성범죄든 ½ 가중).
        # reference_articles 에 article 대신 {"modifier_kind": ...} directive 1건만 적재 →
        # 도구는 base 죄를 강제 목록화하지 않고, LLM 이 *실제 base 성범죄* 를 따로 조회한 뒤
        # statutory_modifications 로 그 modifier 를 부착하도록 안내 (철학 §1.4 — 도구=메커니즘,
        # LLM=판단). 기존 본조_가중 modifier 재사용 → 신규 계산 로직 0.
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

        # M37(d) — 원범죄가 *호별 분기* row (예: 아동복지법 §71① 각 호) 면 호마다 법정형이
        # 다르다 (§72 상습범 = "범한 그 호의 죄에 정한 형" 의 가중). 합집합 envelope 로
        # 뭉뚱그리면 상한 과대(매매 호의 15년이 성적학대 사건에도 적용)·선고가능범위 천장
        # 오류. 직접 분기 row 와 *동일하게* branch_key 로 해당 호를 받아 그 호의 형에 가중.
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

    # 3. reference_mode 분기 — "전2조의 예" — multiplier 없음, ref 그대로
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
            # 분기 mode 는 *행위 형태* 가 facts 에 따라 다름 (예: 준강간 → 강간 / 유사강간 /
            # 강제추행 중 facts 에서 추출). substring note 매칭은 *최대 형* 으로 over-fit
            # 위험 — 옵션 1 개일 때만 자동, 그 외엔 항상 LLM 선택.
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

    # 4. NULL / 정보 / 누범가중(branch=0) → row stats 직접
    penalty = _penalty_from_row(row, source="direct" if rm in (None, "정보") else f"direct_{rm}")
    if rm not in _DIRECT_FALLTHROUGH_MODES:
        # 위 분기 어디에도 안 걸린 *미지* reference_mode. 조용히 direct 처리하면
        # 가중/준용 미적용 raw 정량이 에러 없이 나간다 → 명시적으로 표시(L1).
        msg = (f"⚠ DATA: 미지 reference_mode={rm!r} (id={_rid(row)}) — 알려진 값 아님. "
               "direct 처리했으나 가중/준용 누락 가능, 데이터 검증 필요.")
        penalty.trace.append(msg)
        _log.warning(msg)
    return penalty, None


# ---------- Phase C1 — 처단형 (statutory_modifications 적용) ----------

# 형법 §56 가중·감경 순서. enum kind → 순서 (낮을수록 먼저).
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
# 누범가중 (§35) 은 *자유형 장기 2배* — 벌금 미변경. 다른 가중·감경은 자유형·벌금 동시.
# 경합범가중 (§37·§38) 은 *자유형 장기 1.5배* (가장 무거운 죄의 장기 1/2 가중) — 벌금·하한 그대로.
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
    """처단형 — statutory_modifications 적용 후."""

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
    """suffix·act_count 로 암묵적으로 의미되는 modifiers 를 statutory_modifications 에 자동 반영.

    Rules:
      - is_attempted (미수, §25): "법률상_임의감경" — 단, statutory_modifications 에
        이미 미수 항목 있으면 추가하지 않음. **applied=False (M37)**: §25 ② 는 *재량*
        감경("감경할 수 있다") 이라 감경 적용 여부는 *판사·LLM 판단* (gold 영역, R2·
        챕터 §4.1). 도구는 trace 에 *적용 가능* 으로만 노출하고 처단형 정량은
        감경 전(기수) 기준 유지 — auto-halving 금지. 호출자가 applied=True 명시하면 적용.
      - is_accessory (방조, §32): "법률상_필요감경" — §32 ② "감경한다" *필요적* 이라
        applied=True 자동 적용이 정당. 호출자 명시 우선.
      - is_solicitor (교사, §31 ①): 동일 처벌 — 감경 없음. trace 만.
      - act_count >= 2: "경합범_가중" 동종 §37 전단 — L1 (동종 다행위). 호출자가
        statutory_modifications 에 이미 경합범_가중 항목 있으면 추가하지 않음
        (수동 명시 우선). applied=False 로 override 가능 (실무에 가끔 있는
        §37 미적용 case).

    M27 시도 후 revert — auto_from_act_count 제거 시 LLM 이 prompt 의 aggravators
    형식을 도구 schema 로 옮길 때 *kind 누락·statutes invalid 키* 사용 회귀
    (case 41 3/3 retry 모두 schema 잘못). prompt↔도구 schema mismatch 영역 해소
    전엔 자동 trigger 유지. 챕터 R2 룰 단서로 *포괄일죄 act_count=1* 패턴 명문화.
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
            "applied": False,  # M37 — §25 재량감경: 적용 여부는 LLM/판사 판단 (auto-halving 금지)
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
    """1 modification 적용 후 새 penalty + trace line.

    호출 전 `_MOD_MULT` 에 kind 존재 보장 (_apply_statutory_modifications 의
    INVALID 가드).

    M15: 정량 NULL + fine_formula 있는 경우 (특가법 §8의2 등 식 기반 fine), formula
    의 low_mult/high_mult 에 multiplier 자동 적용 — 정량 산출은 안 함 (base 가
    public input 아님), 식의 *배수* 만 변동 표기.
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
        # 가중: 상한 multiplier, 하한 그대로. cap 600개월.
        if new_imp_max is not None:
            new_imp_max = min(int(new_imp_max * imp_max_m), agg_cap_months)
        if new_fine_max is not None:
            new_fine_max = int(new_fine_max * fine_max_m)
        # M15: fine_formula 의 high_mult 만 가중 (상한 multiplier)
        if new_fine_formula and "high_mult" in new_fine_formula:
            new_fine_formula["high_mult"] = new_fine_formula["high_mult"] * fine_max_m
    elif _kind_is_mit(kind):
        # 형법 §55 ① 각 형종 *독립* 감경 후 합집합. 처단형은 모든 형종 옵션의
        # 합집합으로 정의 (양형기준 [공통원칙] §02). 사형/무기/유기가 동시 존재할
        # 때 한 형종만 감경하면 다른 형종 감경 결과를 잃어버려 처단형 ∩ 권고 = ∅
        # false negative 가 발생.
        #   사형 → 무기 + 240~600월   (§55 ① 1호)
        #   무기 → 120~600월          (§55 ① 2호)
        #   유기 → 1/2                 (§55 ① 3호)
        cand_mins: list[int] = []
        cand_maxs: list[int] = []
        out_has_life = False
        out_has_death = False
        if new_has_death:
            # 사형 감경 (§55①1호): 무기 + 유기 20년~50년
            out_has_life = True
            cand_mins.append(_DEATH_COMMUTE_MIN_MONTHS)
            cand_maxs.append(_AGGRAVATED_IMP_CAP_MONTHS)
        if new_has_life:
            # 무기 감경 (§55①2호): 유기 10년~50년 (life 자체는 사라짐)
            cand_mins.append(_LIFE_COMMUTE_MIN_MONTHS)
            cand_maxs.append(_AGGRAVATED_IMP_CAP_MONTHS)
        if new_imp_min is not None or new_imp_max is not None:
            # 유기 감경: 1/2
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
        # M15: fine_formula 감경 — §55 ① 6호 (벌금 다액 1/2) 등. 도구 정책: low_mult 도
        # _MOD_MULT 의 fine_min_m 와 동일 비율 적용 (정량 감경과 일관). 엄격히 §55 ① 6호
        # 는 *다액* 만 1/2 이지만 도구의 정량 감경 룰과 일관성 우선.
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

    applied = mod.get("applied")
    src = mod.get("source", "")
    src_suffix = f" [{src}]" if src else ""
    imp_str = (
        f"imp {new_imp_min}~{new_imp_max}월" if new_imp_max is not None else "imp -"
    )
    # 처단형 trace 의 fine 표시는 *정량 변화 명시* case (결정론 추적용) 또는
    # *fine_formula 의 배수 변동* (M15 — 식 기반 row).
    if new_fine_max is not None:
        fine_str = f"fine {new_fine_max:,}원"
    elif new_fine_formula:
        fine_str = f"fine {_format_fine_formula(new_fine_formula)}"
    else:
        fine_str = ""
    life_str = ", life" if new_has_life else ""
    death_str = ", death" if new_has_death else ""
    bits = [imp_str]
    if fine_str:
        bits.append(fine_str)
    suffix = ", ".join(bits) + life_str + death_str
    trace_line = (
        f"- {kind} / {type_label} (applied={applied}{src_suffix}) → {suffix}"
    )
    return new_penalty, trace_line


def _format_fine_formula(formula: dict) -> str:
    """fine_formula dict → 한 줄 표기. M15.

    예: {"base_label": "부가세액", "low_mult": 2, "high_mult": 5}
        → "부가세액의 2배 ~ 5배"
        {"base_label": "이득액", "low_mult": 0, "high_mult": 1, "is_optional": True}
        → "이득액 이하 (임의 병과)"
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
    """base 법정형 + statutory_modifications → 처단형.

    형법 §56 순서 (본조가중 → 특수교사방조 → 누범 → 법률상감경 → 경합범가중 → 작량감경).
    `applied=True` 인 mod 만 실제 적용. `applied=False` 또는 누락은 trace 만 노출.
    `act_count >= 2` 면 §37 전단 경합범 가중을 자동 prepend (수동 명시 우선).

    `offense_iso`: 행위시점 (YYYYMMDD). M37 — 가중 후 §42 ② 유기징역 cap 시점보정:
    구법(<2010.10.16) 행위는 25년(300월), 신법은 50년(600월). 미지정 시 신법 cap.
    """
    # M37 — 가중 후 §42 ② cap 시점보정.
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

    # §56 순서 정렬
    ordered = sorted(all_mods, key=lambda m: _MOD_ORDER.get(m.get("kind", ""), 99))

    seen: set[tuple[str, str]] = set()
    for mod in ordered:
        kind = mod.get("kind", "")
        type_label = mod.get("type", "")
        # Fix B: unknown kind 차단 — LLM 오타·환각 식별. 6 종만 허용.
        if kind not in _MOD_MULT:
            penalty.trace.append(
                f"- {kind!r} / {type_label} (applied={mod.get('applied')}) "
                f"→ INVALID kind (무시; 허용: {sorted(_MOD_MULT)})"
            )
            continue
        # Fix C: 같은 (kind, type) 중복 차단 — LLM 환각으로 동일 사유 2번 입력 시
        # cumulative 곱셈 (예: 자수 × 2 = 1/4) 방지. 다른 type 의 같은 kind 는 OK.
        key = (kind, type_label)
        if key in seen:
            penalty.trace.append(
                f"- {kind} / {type_label} → DUPLICATE (무시)"
            )
            continue
        seen.add(key)
        # Fix D: applied default True — 판결문이 *명시한* 사유 = 보통 적용. LLM 이
        # 키 누락하면 적용으로 처리 (silent skip false-negative 방지).
        # 키 명시 시 True 만 적용; False/null/기타 모두 skip — *명시적 모호*도 보수적.
        if "applied" not in mod:
            mod = {**mod, "applied": True}  # normalize for trace
        applied_val = mod["applied"]
        if applied_val is not True:
            penalty.trace.append(
                f"- {kind} / {type_label} (applied={applied_val}) → skip"
            )
            continue
        penalty, line = _apply_single_modification(penalty, mod, agg_cap_months=agg_cap)
        penalty.trace.append(line)

    return penalty


def _format_processed_penalty_lines(p: ProcessedPenalty) -> list[str]:
    out: list[str] = []
    if p.imp_min_months is not None or p.imp_max_months is not None:
        lo = "?" if p.imp_min_months is None else f"{p.imp_min_months}"
        hi = "?" if p.imp_max_months is None else f"{p.imp_max_months}"
        out.append(f"- imprisonment: {lo}~{hi}월")
    if p.fine_min_won is not None or p.fine_max_won is not None:
        # 조문에 fine 하한 명시 없으면 형법 §45 default (벌금 5만원 이상) — M8 polish
        lo = f"{p.fine_min_won:,}" if p.fine_min_won is not None else "50,000 (§45 default)"
        hi = "?" if p.fine_max_won is None else f"{p.fine_max_won:,}"
        out.append(f"- fine: {lo}~{hi}원")
    elif p.fine_formula:
        # M15: 정량 NULL + fine_formula → 식 기반 표기
        out.append(f"- fine: {_format_fine_formula(p.fine_formula)}")
    if p.has_life:
        out.append("- life: 가능")
    if p.has_death:
        out.append("- death: 가능")
    return out


# ---------- Phase C2 — 권고형 (leaf + factors → recommended range) ----------


def _convert_factors_to_applied(
    guideline_factors: dict | None,
) -> list[AppliedFactor]:
    """LLM 의 4 list dict → recommended_range.AppliedFactor 리스트.

    Input shape (chapter 09 §3.1):
      {
        "special_act_aggravators": ["text", ...],
        "special_act_mitigators":  [...],
        "special_actor_aggravators": [...],
        "special_actor_mitigators":  [...],
      }

    text 는 LLM 입력 그대로 통과. verify 없음 — LLM 의 분류 정확도는 평가 단계의
    gold label 비교로 측정 (chapter 09 §7 의 A1). 도구는 *dict key 가 의도한
    (kind, direction)* 으로 카운트만 — text 내용은 trace 용.
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
    """조문 본문에서 '벌금' 포함 paragraph 발췌 — 결정론 (법조문 원문 그대로).

    M11: fine 정량이 NULL 인 row (특가법 §8의2 처럼 *식 기반* 정량 — 부가세 2~5배 등)
    에서 LLM 에 *법정 범위* 를 결정론적으로 전달. paragraph split 룰은 한국 조문의
    표준 표기 (① ② ③ ...). 추출 텍스트는 *st_articles 원문* — Phase 2 Gemini 가공이 아님.

    배수 선택 (2~5배 중 어느 배수) 은 *판사 재량* (양형위 [조세범죄] 등 cat 에 fine 권고
    미정의) — 도구는 *법정 범위* 만 전달, 위치 결정은 LLM/판사 정성 추론.
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
    """처단형 imp_min — 양형위 [공통원칙] §02 의 *권고 floor* 보정 입력.

    양형위 §02: *"권고가 처단형을 벗어나면 처단형이 기준"* — floor 는 *처단형 floor*
    이지 *법정형 floor* 가 아님. 즉 자수·미수·심신미약·작량감경 등 법정·작량 감경
    적용 *후* 의 처단형 하한이 권고 영역의 절대 floor.

    이전 (M6) — `charge_legal_map.stat_imp_min_months` (법정형 floor) 사용. 자수 적용
    살인 case 에서 처단형 [30, 600] 인데 법정형 floor 60 강제 → 권고 raw [42, 144] 의
    [42, 60) 부분이 사라지는 양형위 §02 위반 발견.
    M7 — `processed.imp_min_months` (감경 적용 후 처단형) 사용.

    처단형 imp 없는 (fine 전용 leaf) case 는 None — determine_range 가 보정 skip.
    """
    return processed.imp_min_months


def _intersect_with_processed(
    rec: RecommendedRange | None,
    processed: ProcessedPenalty,
) -> tuple[int | None, int | None]:
    """처단형 ∩ 권고형 — intersection 으로 *선고 가능 범위*.

    rec 가 None 이면 (processed.min, processed.max) 그대로.
    rec.has_life 면 max=None (제한 없음) 으로 처리.
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


# ---------- Phase C3 — 선고형 검증 ----------


def _verify_sentence(
    sentence_months: int | None,
    fine_amount: int | None,
    processed: ProcessedPenalty,
    rec: RecommendedRange | None,
    intersect: tuple[int | None, int | None],
) -> list[str]:
    """선고형 (sentence_months, fine_amount) 검증 + 위치 계산."""
    lines: list[str] = []
    if sentence_months is None and fine_amount is None:
        lines.append("- 선고형 미지정")
        return lines

    if sentence_months is not None:
        lines.append(f"- sentence_months={sentence_months}")
        lo, hi = intersect
        in_intersect = True
        if lo is not None and sentence_months < lo:
            in_intersect = False
        if hi is not None and sentence_months > hi:
            in_intersect = False
        lines.append(f"- 처단형 ∩ 권고 [{_b(lo)}, {_b(hi)}] 안: {in_intersect}")

        if rec is not None:
            ok = in_range(sentence_months, rec)
            pos = within_range_position(sentence_months, rec)
            lines.append(f"- 권고 영역 [{_b(rec.min_months)}, {_b(rec.max_months)}] 안: {ok}")
            if pos is not None:
                lines.append(f"- 영역 내 위치: {pos:.3f}")

        # 처단형 검증
        p_lo = processed.imp_min_months or 0
        p_hi = processed.imp_max_months
        in_proc = sentence_months >= p_lo and (p_hi is None or sentence_months <= p_hi)
        lines.append(f"- 처단형 [{_b(p_lo)}, {_b(p_hi)}] 안: {in_proc}")

    if fine_amount is not None:
        lines.append(f"- fine_amount={fine_amount:,}원")
        f_lo = processed.fine_min_won or 0
        f_hi = processed.fine_max_won
        in_fine = fine_amount >= f_lo and (f_hi is None or fine_amount <= f_hi)
        # M9: f_hi None 시 TypeError 방지 (특가법 §8의2 등 부가세 식 기반 fine 정량 NULL row)
        f_hi_disp = f"{f_hi:,}" if f_hi is not None else "무제한"
        lines.append(f"- 벌금 처단형 [{f_lo:,}, {f_hi_disp}] 안: {in_fine}")
    return lines


# ---------- Phase C4 — 집행유예 권고 ----------

# 형법 §62 ①: 3년 이하 징역·금고 또는 *500만원 이하 벌금* (2018.1.7 개정 시행).
# §62 ②: 형 병과 시 형의 일부 집유 가능.
# 본 도구는 sentence_months ≤ 36 OR fine_amount ≤ 5_000_000 일 때 eligibility=True.
# 4분면 룰은 양형기준 [공통원칙] §05 (형종 무관).
_PROBATION_IMP_CAP_MONTHS = 36
_PROBATION_FINE_CAP_WON = 5_000_000


def _probation_recommendation(
    sentence_months: int | None,
    fine_amount: int | None,
    processed: ProcessedPenalty,
    probation_factors: dict | None,
) -> list[str]:
    """집유 권고 — eligibility + 4분면 룰.

    `probation_factors` (chapter 09 §3.1):
      {"major_positive": [...], "major_negative": [...],
       "general_positive": [...], "general_negative": [...]}

    Eligibility 는 *실제 선고형* 기준:
      - imp ≤ 36월              → 적용 가능 (§62 ①)
      - fine ≤ 500만원         → 적용 가능 (§62 ① 2018.1.7+)
      - imp+fine 병과 + 둘 다 cap 안 → §62 ② 일부/전부 집유 가능
      - 무기·사형 옵션은 *처단형 노출* 이지 실제 선고형 아님 — 미반영.
    """
    lines: list[str] = []

    has_imp = sentence_months is not None
    has_fine = fine_amount is not None
    if not has_imp and not has_fine:
        lines.append("- 집유 판단: sentence_months/fine_amount 둘 다 미지정 — skip")
        return lines

    imp_ok = has_imp and sentence_months <= _PROBATION_IMP_CAP_MONTHS
    fine_ok = has_fine and fine_amount <= _PROBATION_FINE_CAP_WON

    parts: list[str] = []
    if has_imp:
        parts.append(
            f"imp {sentence_months}월 ≤ 36월" if imp_ok
            else f"imp {sentence_months}월 > 36월 cap"
        )
    if has_fine:
        parts.append(
            f"fine {fine_amount:,}원 ≤ 5,000,000원 (§62 ① 2018.1.7+)" if fine_ok
            else f"fine {fine_amount:,}원 > 5,000,000원 cap"
        )

    eligible = imp_ok or fine_ok
    status_text = "적용 가능" if eligible else "부적용"
    lines.append(f"- eligibility: {status_text} ({'; '.join(parts)})")

    if has_imp and has_fine and imp_ok and fine_ok:
        lines.append("- 병과: §62 ② 형의 일부 (imp 또는 fine) 집유 가능")

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

    # 양형기준 [공통원칙] §05 4분면 룰 (chapter 09 §2.4):
    # 룰 1: major_positive ≥ 2 OR (mp − mn) ≥ 2 → 집유 *권고*
    # 룰 2: major_negative ≥ 2 OR (mn − mp) ≥ 2 → 집유 *불권고*
    # 둘 다 충족 시 → 룰 3 (재량) fallback
    # 둘 다 불충족 시 → 룰 3 (general 비교 + 재량)
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
        # 룰 3: general 비교 (major 룰 1·2 모두 불충족 시)
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


# ---------- markdown-KV 응답 ----------

def _format_modifiers(mods: dict[str, bool]) -> str | None:
    flags = [k for k, v in mods.items() if v]
    if not flags:
        return None
    return ", ".join(flags)


def _format_penalty(penalty: EffectivePenalty) -> list[str]:
    """EffectivePenalty → markdown 행 list."""
    out: list[str] = []
    if penalty.sentence_kind_options:
        out.append(f"- sentence_kind_options: {penalty.sentence_kind_options}")

    if penalty.imp_min_months is not None or penalty.imp_max_months is not None:
        lo_s = "?" if penalty.imp_min_months is None else f"{penalty.imp_min_months}"
        hi_s = "?" if penalty.imp_max_months is None else f"{penalty.imp_max_months}"
        out.append(f"- imprisonment: {lo_s}~{hi_s}월")

    if penalty.fine_min_won is not None or penalty.fine_max_won is not None:
        # 조문에 fine 하한 명시 없으면 형법 §45 default (벌금 5만원 이상) — M8 polish
        lo_s = f"{penalty.fine_min_won:,}" if penalty.fine_min_won is not None else "50,000 (§45 default)"
        hi_s = "?" if penalty.fine_max_won is None else f"{penalty.fine_max_won:,}"
        out.append(f"- fine: {lo_s}~{hi_s}원")
    elif penalty.fine_formula:
        # M15: 정량 NULL + fine_formula → 식 기반 표기
        out.append(f"- fine: {_format_fine_formula(penalty.fine_formula)}")

    if penalty.has_life:
        out.append("- life: 가능")
    if penalty.has_death:
        out.append("- death: 가능")
    return out


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
    key = o.get("key", "?")
    cond = (o.get("cond") or "").strip()
    parts: list[str] = []
    imp_lo, imp_hi = o.get("imp_min_months"), o.get("imp_max_months")
    if imp_lo is not None or imp_hi is not None:
        parts.append(f"imp={_b(imp_lo)}~{_b(imp_hi)}월")
    f_lo, f_hi = o.get("fine_min_won"), o.get("fine_max_won")
    if f_lo is not None or f_hi is not None:
        fl = "?" if f_lo is None else f"{f_lo:,}"
        fh = "?" if f_hi is None else f"{f_hi:,}"
        parts.append(f"fine={fl}~{fh}원")
    if o.get("has_life"):
        parts.append("life")
    if o.get("has_death"):
        parts.append("death")
    suffix = (" " + ", ".join(parts)) if parts else ""
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
    """공통 헤더 (status / stage / charge / 본조 / sg_category_id)."""
    lines: list[str] = [
        "## status: ok",
        f"## stage: {stage}",
        f"## charge: {norm.raw_key}",
    ]
    if norm.key != norm.raw_key:
        lines.append(f"- normalized_key: {norm.key}")
    mod_str = _format_modifiers(norm.modifiers)
    if mod_str:
        lines.append(f"- modifiers: {mod_str}")
    if norm.suffix_split_applied:
        lines.append("- note: suffix(미수/교사/방조) 자동 분리 — modifier 자동 set")
    if row["is_alias"] and row["alias_of"]:
        lines.append(f"- alias_of: {row['alias_of']}")

    lines.append(f"## 본조: {_format_article(payload_row)}")
    if payload_row["md_source_name"] and payload_row["md_source_name"] != norm.raw_key:
        lines.append(f"- md_source_name: {payload_row['md_source_name']}")
    lines.append(f"- sg_category_id: {payload_row['sg_category_id']}")
    return lines


def _list_leaves_for_category(
    conn: sqlite3.Connection, sg_category_id: int
) -> list[sqlite3.Row]:
    """sg_category 의 leaf 후보 list.

    leaf = `sg_subtypes` 중 `type_criterion IS NOT NULL` row. parent group 노드는
    제외 (양형기준 trie 의 *말단* 만). chapter 09 §3.1 의 lookup stage 응답에
    노출되어 LLM 이 *다음 호출* 에 `guideline_leaf_id` 인자를 채우는 enum.
    """
    return conn.execute(
        "SELECT id, section_no, name, type_criterion FROM sg_subtypes "
        "WHERE category_id=? AND type_criterion IS NOT NULL "
        "ORDER BY id",
        (sg_category_id,),
    ).fetchall()


def _format_leaf_candidates(leaves: list[sqlite3.Row]) -> list[str]:
    if not leaves:
        return []
    lines = [f"## 양형기준 leaf 후보 ({len(leaves)}개)"]
    for lf in leaves:
        name = (lf["name"] or "?").strip()
        crit = (lf["type_criterion"] or "").strip()
        lines.append(f"- id={lf['id']} [{name}] {crit}")
    return lines


def _list_factors_for_category(
    conn: sqlite3.Connection, sg_category_id: int
) -> list[sqlite3.Row]:
    """sg_category 의 union factor list.

    leaf 별 attach 된 factor 를 카테고리 단위로 합집합. 같은 cat 안에서 leaf 별
    factor 가 거의 동일 (유형 차이는 *type_criterion* 으로 분기, factor 는 공유).

    chapter 09 §3.1 의 lookup stage 응답에 노출되어 LLM 이 *다음 호출* 에
    `guideline_factors` 인자 (특별만) + reasoning 의 일반인자 인용 enum.
    """
    return conn.execute(
        "SELECT DISTINCT scope, kind, direction, text FROM sg_factors "
        "WHERE category_id=? "
        "ORDER BY scope DESC, kind, direction, text",
        (sg_category_id,),
    ).fetchall()


# scope ∈ {특별, 일반}, kind ∈ {행위, 행위_공통, 행위_미수, 행위자_기타}, direction ∈ {가중, 감경}.
# chapter 09 §3.1 의 `guideline_factors` 4 list 와 대응 — 특별 / 행위 계열·행위자 계열 × 가중·감경.
def _format_factor_enum(factors: list[sqlite3.Row]) -> list[str]:
    if not factors:
        return []
    # scope / kind / direction 으로 그룹화
    from collections import defaultdict
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for r in factors:
        groups[(r["scope"], r["kind"], r["direction"])].append(r["text"])

    # 특별 (영역 결정) 먼저, 일반 (위치 결정) 나중
    special = [k for k in groups if k[0] == "특별"]
    general = [k for k in groups if k[0] == "일반"]
    n_special = sum(len(groups[k]) for k in special)
    n_general = sum(len(groups[k]) for k in general)

    # 그룹 헤더 → guideline_factors dict key 매핑 (chapter 09 §3.1).
    # _convert_factors_to_applied 의 입력 schema.
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
    """sg_category 의 집유 4분면 사유 list (sg_probation_factors).

    카테고리 단위 union. section_no (subtype 분할) + source_note (sub_label / 메타)
    가 grouping 키 — 같은 cat 안에서 sub-section 마다 다른 4분면 사유 보존
    (42 증권금융처럼 sub-section 별 표가 분리된 경우).

    chapter 09 §3.1 의 lookup stage 응답에 노출되어 LLM 의 `probation_factors`
    인자 (4 list: major_positive / major_negative / general_positive / general_negative)
    채울 때 환각 차단 + sub-section 식별 단서.
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
# chapter 09 §3.1 의 `probation_factors` 4 list 와 대응.
# 룰은 `_probation_recommendation` 참고 (양형위 [공통원칙] §05).
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

    # 그룹 헤더 → probation_factors dict key 매핑.
    # _probation_recommendation 의 입력 schema (chapter 09 §3.1).
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


# 형법 §56 가중·감경 사유 enum — cat 별 변동 X. 정적 상수.
# _MOD_ORDER / _MOD_MULT 가 source-of-truth (compute_sentencing_range.py:653~).
# 미수·방조·교사 는 charge suffix 자동 분리로 auto-add (`_expand_implicit_modifications`).
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
    """charge_legal_map row 의 statute (law_id) + article 로 행위시 본문 lookup.

    같은 law_id 의 *행위시점 이전의 최후 버전* 조문을 찾는다. M37(f) — 종전엔
    `article_changed='Y'`(변경분) 만 봐 *한 번도 개정 안 된* 조문(baseline 스냅샷
    changed=None 에만 존재)을 '미적재' 로 놓쳤다 (statute_lookup 과 동일 결함).
    변경분 우선 + baseline fallback (동일 effective_date 면 'Y' 우선).
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
    # 현행본 effective_date 도 (시간 축 비대칭 표시용)
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


# 형법 §42 ① 유기징역 상한 — 개정 2010.4.15, *시행 2010.10.16*.
# 행위시법주의(§1①)는 *시행일* 기준 → 경계 = 20101016. 단독 15년(180월)→30년(360월),
# 가중 25년(300월)→50년(600월). _ART42_REFORM_ISO / _ART42_CAPS 는 총칙 상수 블록(line
# ~252)서 정의 — 단일 출처.


def _apply_art42_versioned_cap(
    payload: dict, current_max: int | None, offense_iso: str,
) -> str | None:
    """§42 ① 유기징역 상한의 *시점 의존성* 보정 (M33).

    clm_versions 시점본 정량은 *본조 시행일* 기준 §42 cap 을 박아둔다 (build_clm_
    versions prompt line 185). 그러나 §42 ① 상한 자체가 2010.10.16 에 15→30년 으로
    바뀌었고, 이는 *본조와 무관한 총칙 개정* 이라 본조가 그 이전 시행본에 멈춰 있으면
    시점본에 전파되지 않는다 (예: 살인 §250 은 1996 이후 불변 → 시점본 max=180 고정).
    → 도구가 행위시점 §42 cap 으로 재보정.

    *상한 개방형* ("X년 이상의 징역" — 본조 상한 미명시) 판별 = 현행 charge_legal_map
    상한 (current_max) 이 현행 §42 cap (360 단독 / 600 가중) 인 row. 본조가 상한을
    직접 박은 row (예: "15년 이하" = 180) 는 current_max ∉ {360,600} 이라 건드리지
    않는다 → 오탐 0. (누범·경합 가중 후 cap = _AGGRAVATED_IMP_CAP_MONTHS 는 별개 —
    그 시점 의존성은 미해결 사각지대, §3 참조.)
    """
    ver_max = payload.get("stat_imp_max_months")
    if ver_max is None or current_max not in _ART42_CAPS:
        return None
    old_cap, new_cap = _ART42_CAPS[current_max]
    # M37 — 시점본 상한이 두 §42 cap 값 중 하나일 때만 *개방형 시점본* (상한 = §42 총칙
    # cap). 시점본이 *구체적 닫힌 상한* (예: 성착취물 §11⑤ 2013년 '1년 이하' max=12,
    # 후에 '1년 이상' 개방형으로 개정) 이면 본조가 그 시점엔 상한을 직접 박은 것 →
    # 현행 max(360)만 보고 과거를 360 으로 부풀리면 안 됨. 개방형 판별을 *시점본* 으로.
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
    """clm_versions 시점본 행이 *아무 법정형 정보도 없는* 퇴화행인지.

    빈 행으로 base 정량을 override 하면 법정형이 통째로 NULL 로 덮여 '?~?' 빈 범위가
    나간다 (철학 B 위반: 나쁜 DB 행이 로직을 조용히 깨뜨림). True 면 _get_versioned_
    payload 가 override 를 건너뛰고 현행 base 정량을 유지한다.

    *직접 정량·분기·참조·식(formula) 어느 것도 없을 때만* 퇴화로 본다 — 참조전용·
    분기전용 시점본(직접 정량 NULL 이 정상)은 override 대상으로 보존.
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
    """offense_iso 지정 시 clm_versions 에서 시점본 정량 row 가져와서 base_row 의
    정량 필드 override. 매치 없으면 base_row 그대로.

    return: (payload_dict, version_meta|None, art42_trace|None)
      version_meta = {effective_date, matched_paragraph, match_confidence, mst}
      art42_trace  = §42① 상한 시점보정 trace 줄 (보정 발생 시만, M33)
    """
    base_dict = dict(base_row)
    if not offense_iso:
        return base_dict, None, None
    # 현행 §42 cap (override 전) — 상한 개방형 판별 기준 (M33)
    current_max = base_dict.get("stat_imp_max_months")
    version_meta = None
    v = conn.execute(
        """SELECT * FROM clm_versions
           WHERE clm_id=? AND effective_date<=?
           ORDER BY effective_date DESC LIMIT 1""",
        (base_row['id'], offense_iso),
    ).fetchone()
    if v and _clmv_version_is_empty(v):
        # 빈/퇴화 시점본 — override 하면 base 정량이 통째로 NULL 로 덮여 법정형이
        # 사라진다. 현행 base 유지 + empty 표시(다운스트림 trace) + log 경고 (L1 견고성).
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
        # 시점본 정량으로 override
        override_cols = [
            'sentence_kind_options', 'stat_imp_min_months', 'stat_imp_max_months',
            'stat_fine_min_won', 'stat_fine_max_won', 'has_life', 'has_death',
            'has_conditional_branch', 'branch_options',
            'reference_mode', 'reference_multiplier', 'reference_articles',
            'fine_formula',  # M31 — 시점본 fine_formula override (clm_versions 에 적재)
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
    # M33 — §42① 유기징역 상한 시점보정 (시점본 유무 무관 — 미적재 row 도 보정)
    art42_trace = _apply_art42_versioned_cap(base_dict, current_max, offense_iso)
    return base_dict, version_meta, art42_trace


def _historic_appendix(
    conn: sqlite3.Connection, payload_row: sqlite3.Row, offense_date: str | None
) -> str:
    """exact-match 후 모든 stage 응답 끝에 append 되는 *행위시 조문 본문* 단락.

    형법 §1 ① "범죄의 성립과 처벌은 행위시의 법률에 의한다" 원칙.
    도구 정량은 *현행* charge_legal_map 기준이라 LLM 이 행위시 본문 보고
    *시점별 본조 분기·법정형 차이* 정확 인식 가능.
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
    """sg_categories.manual_url — 범죄군별 양형기준 해설서 PDF(자체 호스팅) 출처.

    값은 scripts/build_sg_manuals.py 가 적재. 컬럼이 없거나 값이 비면 None — 링크 생략이
    if 분기가 아니라 데이터 유무로 결정되도록 OperationalError 를 흡수한다(미적재 환경에서도
    도구는 정상 동작).
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
) -> str:
    """resolve 완료된 lookup stage 응답.

    Args:
      conn: leaf 후보 enum fetch 용.
      row: 원래 lookup hit row (alias 분기 trace 용).
      payload_row: alias 해소 후 metadata 소스 (본조, sg_category_id).
      penalty: resolve 후 최종 법정형.

    `notes` / `penalty_notes` DB 컬럼은 *Opus 자연어 요약* 이라 도구가 아예 적재
    (`_SELECT_ROW`)하지 않는다 — "응답 노출 금지" 원칙을 *구조적으로* 보장 (자유 prose
    를 기능 출력으로 piggyback 하지 않음). 본 도구는 *정량* (선고형 범위) 만 책임.
    절차·부가처분 정보 (반의사불벌, 병과, 몰수, 신상등록 등) 가 필요하면 호출자가
    `statute_lookup` 으로 본조 원문을 조회.
    """
    lines = _format_stage_header(norm, row, payload_row, "lookup")

    # 출처 — 범죄군 양형기준 해설서 PDF (manual_url 적재 시에만, lookup = 모든 흐름의 첫 응답)
    src = _manual_source(conn, payload_row["sg_category_id"])
    if src:
        lines.append(f"- 출처: [{src[0]} 양형기준 해설서(PDF)]({src[1]})")

    # resolve trace (reference / branch 적용 내역)
    if penalty.trace:
        lines.extend(penalty.trace)

    # 최종 법정형
    pen_lines = _format_penalty(penalty)
    if pen_lines:
        lines.append("## 법정형 (effective)")
        lines.extend(pen_lines)
    else:
        lines.append("## 법정형: 본조 직접 정량 없음")
    lines.append(f"- source: {penalty.source}")

    # 양형기준 leaf 후보 enum — LLM 의 후속 `guideline_leaf_id` 선택용.
    # chapter 09 §3.1 line 245 의 lookup 응답 spec.
    leaves = _list_leaves_for_category(conn, payload_row["sg_category_id"])
    lines.extend(_format_leaf_candidates(leaves))

    # 양형기준 특별/일반 factor enum — LLM 의 `guideline_factors` 선택용.
    # leaf 별 factor 는 카테고리 안에서 거의 동일하므로 *union* 노출.
    factors = _list_factors_for_category(conn, payload_row["sg_category_id"])
    lines.extend(_format_factor_enum(factors))

    # 양형기준 집유 4분면 enum — LLM 의 `probation_factors` 인자 선택용.
    prob_factors = _list_probation_factors_for_category(
        conn, payload_row["sg_category_id"]
    )
    lines.extend(_format_probation_factor_enum(prob_factors))

    # 형법 §56 가중·감경 사유 enum — LLM 의 `statutory_modifications` 인자 선택용.
    # cat 별 변동 X — 정적 상수. chapter 09 §3.1 의 lookup 응답 4 enum 마지막.
    lines.extend(_format_modifier_enum())

    # 경합범 가중 안내 — act_count >= 2 면 §37 전단 자동 적용 예정.
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
) -> str:
    """처단형 stage — 법정형 + statutory_modifications 적용 trace + 처단형."""
    lines = _format_stage_header(norm, row, payload_row, "처단형")

    # 법정형 (resolve 후)
    if penalty.trace:
        lines.extend(penalty.trace)
    pen_lines = _format_penalty(penalty)
    if pen_lines:
        lines.append("## 법정형 (effective)")
        lines.extend(pen_lines)
    lines.append(f"- source: {penalty.source}")

    # 처단형 trace
    lines.append("## 처단형 — 형법 §56 순서 적용")
    lines.extend(processed.trace)

    # 처단형 결과
    proc_lines = _format_processed_penalty_lines(processed)
    if proc_lines:
        lines.append("## 처단형 (final)")
        lines.extend(proc_lines)
    else:
        lines.append("## 처단형: 법정형 그대로")

    return "\n".join(lines)


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
) -> str:
    """권고형 stage — 처단형 + 양형기준 권고 + 처단형 ∩ 권고."""
    lines = _format_stage_header(norm, row, payload_row, "권고형")

    # 법정형 + 처단형 요약
    lines.append("## 법정형 (effective)")
    lines.extend(_format_penalty(penalty))
    lines.append(f"- source: {penalty.source}")

    lines.append("## 처단형 — 형법 §56 trace")
    lines.extend(processed.trace)
    lines.append("## 처단형 (final)")
    lines.extend(_format_processed_penalty_lines(processed))

    # 양형기준 권고
    lines.append(f"## 양형기준 leaf_id: {leaf_id}")
    if floor is not None:
        lines.append(f"- 처단형 floor: {floor}월 (공통원칙 §02 보정)")
    if rec is None:
        lines.append("- 권고형: leaf 에 sg_ranges row 없음 (벌금형 전용 등)")
    else:
        lines.append("## 권고형")
        lines.append(f"- 영역: {rec.level}")
        lines.append(_rec_range_line(rec))
        lines.append(f"- 특별조정 적용: {rec.is_special_adjusted}")
        if rec.raw_text:
            lines.append(f"- raw: {rec.raw_text}")

    # 처단형 ∩ 권고
    lo, hi = intersect
    lines.append("## 선고 가능 범위 (처단형 ∩ 권고)")
    if lo is not None and hi is not None and lo > hi:
        lines.append(
            f"- 처단형 [{_b(processed.imp_min_months)}, {_b(processed.imp_max_months)}] "
            f"∩ 권고 [{_b(rec.min_months) if rec else '?'}, {_b(rec.max_months) if rec else '?'}] "
            f"= ∅ (overlap 없음)"
        )
        lines.append(
            "- 양형기준 [공통원칙] §02: 권고가 처단형 벗어나면 *처단형* 우선."
        )
    else:
        lines.append(
            f"- [{_b(lo)}, {_b(hi)}]월"
            + (" — 권고 준수" if rec is not None else " — 권고 미적용")
        )

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
) -> str:
    """final stage — 권고형 + 선고형 검증 + 집유."""
    lines = _format_stage_header(norm, row, payload_row, "final")

    # 법정형
    lines.append("## 법정형 (effective)")
    lines.extend(_format_penalty(penalty))
    lines.append(f"- source: {penalty.source}")

    # 처단형
    lines.append("## 처단형 trace")
    lines.extend(processed.trace)
    lines.append("## 처단형 (final)")
    lines.extend(_format_processed_penalty_lines(processed))

    # 권고형
    lines.append(f"## 양형기준 leaf_id: {leaf_id}")
    if floor is not None:
        lines.append(f"- floor: {floor}월")
    if rec is None:
        lines.append("- 권고형: 미적용 (sg_ranges 없음)")
    else:
        lines.append("## 권고형")
        lines.append(f"- 영역: {rec.level}")
        lines.append(_rec_range_line(rec))
        lines.append(f"- 특별조정: {rec.is_special_adjusted}")

    lines.append("## 선고 가능 범위")
    import json as _json
    try:
        raw_kinds = _json.loads(payload_row["sentence_kind_options"] or "[]")
    except (TypeError, _json.JSONDecodeError):
        raw_kinds = []
    # 형종 union (처단형 처리 후 ∪ 매핑 raw). 형종일 때만 해당 줄 출력 — 누출 방지.
    kinds_eff = set(processed.sentence_kind_options or []) | set(raw_kinds)

    # 실형 — imprisonment 형종일 때만. lo/hi None 이면 `[None,None]월` 누출 방지(R 견고성).
    lo, hi = intersect
    if "imprisonment" in kinds_eff:
        if lo is not None and hi is not None and lo > hi:
            lines.append(
                "- imprisonment: 처단형 ∩ 권고 = ∅ (처단형 우선, [공통원칙] §02)"
            )
        elif lo is None and hi is None:
            lines.append("- imprisonment: 유기징역 범위 미정 (무기·사형 전속 등)")
        else:
            lo_s = "?" if lo is None else str(lo)
            hi_s = "?" if hi is None else str(hi)
            lines.append(f"- imprisonment: [{lo_s}, {hi_s}]월")

    # 벌금 — 처단형 fine 옵션 또는 매핑 row 의 sentence_kind_options 에 fine 있으면 표시
    # M11: fine 정량 NULL 인 row (특가법 §8의2 등 식 기반) 도 fine 옵션 보존 (raw 매핑 기준)
    fine_kind_present = "fine" in kinds_eff
    if fine_kind_present:
        f_lo = processed.fine_min_won
        f_hi = processed.fine_max_won
        if f_hi is not None or f_lo is not None:
            # 정량 명시 — 기존 표시
            f_lo_d = f_lo if f_lo is not None else 0
            f_lo_str = f"{f_lo_d:,}" if f_lo_d > 0 else "0"
            f_hi_str = f"{f_hi:,}" if f_hi is not None else "?"
            lines.append(f"- fine: [{f_lo_str}, {f_hi_str}]원")
        elif processed.fine_formula:
            # M15: fine_formula 우선 — 감경·가중 multiplier 자동 적용된 배수 표기
            lines.append(f"- fine: {_format_fine_formula(processed.fine_formula)}")
        elif fine_paragraphs:
            # M11 fallback: 정량 NULL + fine_formula 미명세 → 조문 본문 식 인용
            lines.append("- fine: 정량 미상 (조문 본문 — 식 기반 정량):")
            for p in fine_paragraphs:
                # 멀티라인 paragraph 들여쓰기
                for line in p.splitlines():
                    lines.append(f"  > {line}")
        else:
            lines.append("- fine: 정량 미상 (법정형 fine 정량 NULL — statute_lookup 조회 권장)")
        # M13: 감경 적용 시 §55 ① 6호 (벌금 다액 1/2) 명시 — LLM 결정 직전 참고
        # M15: fine_formula 있으면 multiplier 자동 적용된 배수 표시로 이미 반영됨 → 메모만
        if mit_applied:
            note = "§55 ① 6호 다액 1/2 자동 반영됨" if processed.fine_formula else "벌금 다액 1/2 (§55 ① 6호)"
            lines.append(f"  ※ 감경 적용: {note}")
        lines.append("- (※ 법정형 안 imp/fine 둘 다 가능 — 형종 선택은 판사 재량)")

    # 선고 검증
    lines.append("## 선고형 검증")
    lines.extend(verify_lines)

    # 집유
    lines.append("## 집행유예")
    lines.extend(probation_lines)

    return "\n".join(lines)


def _format_pending_response(
    norm: NormalizedCharge,
    row: sqlite3.Row,
    payload_row: sqlite3.Row,
    pending: PendingResolution,
) -> str:
    """resolve 미완 상태 — LLM 의 추가 인자 요청 응답."""
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
        lines.append("## 호출: branch_key=\"...\" 인자 명시 후 재호출")
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
    """후보 disambiguation 용 정량 1줄 요약 (LLM 이 행위 규모로 조항 선택하도록)."""
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
    """ambiguous 후보 1줄 — 본조 + 정량요약 + 행위(act_descriptor) + md_source + statute_choice.

    `act_descriptor` 는 *substrate 파생* 행위 라벨(st_articles 호 본문/참조 조문 제목에서 도출,
    `migrate_act_descriptor.py`). md_source 가 조문 인용뿐인 행(약물류 등)의 *행위축* 항해를
    위해 노출 — 같은 조항군에서 수출입↔매매↔소지↔사용 구별. 없으면 무노출.
    `notes`(Opus 자유 prose)는 도구가 적재조차 안 함(`_format_lookup_response` 원칙) — 행위
    보강은 자유 prose 가 아닌 검증된 substrate 필드로만."""
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
    """같은 sg_category 안에 여러 row — *조항* 모호 (카테고리 모호 아님).

    예: `상해` 가 형법 §257 ① + 폭처법 §2 ③ (누범상해) 둘 다 같은 sg_cat=46 폭력범죄.
    LLM 이 facts 의 *행위 형태 / 전과* 보고 적절한 statute_choice 명시.
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
    """LLM 의 statute_choice 가 row 매칭 실패 시 응답."""
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


# ---------- public tool ----------

@dedup_guard("compute_sentencing_range")
def compute_sentencing_range(
    ctx: RunContext[HarnessDeps],
    # ⚠ 인자 타입은 *의도적으로 넓다*: MiMo 등 모델이 스칼라/배열/JSON문자열로 이중 인코딩해
    # 보내는 경우(예: sg_category_id=["48"], statutory_modifications='[{...}]')가 잦다.
    # pydantic-ai 스키마 검증은 함수 진입 *전*에 돌아 좁은 타입이면 검증 실패→retry 소진→
    # UnexpectedModelBehavior 로 턴 전체가 죽는다. 넓게 받아 진입부 coerce_* 로 정규화한다
    # (검증-후-정규화 단일 지점 — _coerce.py 철학). Args docstring 이 *의도한* 타입을 안내.
    charge: str | list | None = None,
    sg_category_id: int | str | list | None = None,
    statute_choice: str | list | None = None,
    branch_key: str | list | None = None,
    reference_choice: str | list | None = None,
    is_attempted: bool = False,
    is_accessory: bool = False,
    is_solicitor: bool = False,
    # 후속 stage 인자. 넓게 받은 뒤 아래 진입부에서 정규화한다.
    statutory_modifications: list | dict | str | None = None,
    guideline_leaf_id: int | str | list | None = None,
    guideline_factors: dict | str | list | None = None,
    sentence_months: int | str | list | None = None,
    fine_amount: int | str | list | None = None,
    probation_factors: dict | str | list | None = None,
    act_count: int | str | list = 1,
    offense_date: str | list | None = None,
) -> str:
    """통합 양형 도구 — 죄명에서 출발해 법정형 → 처단형 → 권고형(양형기준) → 선고 검증까지 단계별 계산.

    언제:
    - 형량이 화제가 되는 모든 국면 — 양형 판단의 기준 확인, 구형·변론 의견의 근거, 당사자의
      형량 범위 이해. 결과는 '예측'이 아니라 공식 양형기준이 정한 '범위'이며, 어떤 인자가
      범위를 움직이는지가 핵심 정보입니다.
    - 실제 선고 분포와의 대조는 sentence_statistics, 유사 사건의 실제 결과는
      precedent_search 로 교차 확인하세요.

    규칙 — 인자를 채울수록 깊은 단계로 자동 진행:
    - charge 만 → **lookup**: 법정형, 양형기준 leaf 후보, 가중·감경 인자 enum.
    - + statutory_modifications → **처단형**: 형법 §56 순서로 가중·감경 적용 trace.
    - + guideline_leaf_id·guideline_factors → **권고형**: 특별인자로 감경·기본·가중 영역 결정.
    - + sentence_months/fine_amount(+probation_factors) → **final**: 선고형 검증 + 집행유예 4분면.
    - 후속 단계 인자에 넣을 값(leaf id·인자 key 등)은 이전 단계 응답의 enum 이 제공합니다 —
      **enum 에 있는 key 만 사용하고 추측하지 마세요**.
    - 이 도구는 호출 사이 상태를 보존하지 않습니다. 재호출할 때마다 charge와 앞 단계에서 확정한
      선택·플래그·offense_date를 모두 반복하고, 새 단계 인자를 추가하세요.

    응답: markdown-KV(`## section` + `- key: value`). `출처`(양형기준 해설서 PDF 링크)가
    있으면 함께 제시하세요.

    Args:
      charge: 판결문 form 죄명 (예: 살인, 도로교통법위반(음주운전)). 정규화는 도구 내부에서 처리.
      offense_date: 행위 일자 (예: '2013.7.30', '2013-07-30', '20130730'). 지정 시
        행위시 조문 본문·시점본 정량 반영 — 형법 §1 ① "범죄의 성립과 처벌은 행위시의
        법률에 의한다" 원칙. 미지정 시 현행 기준.
      sg_category_id: 동일 charge_key 가 복수 카테고리일 때 명시 (ambiguous_category 응답이 후보 제공).
      statute_choice: 같은 sg_category 안에 여러 row (ambiguous_statute 응답) 일 때
        구체 조항 명시 (예: "형법§257", "폭력행위등처벌에관한법률§2③").
      branch_key: 분기형 row 의 옵션 key (예: "③2") — 응답이 후보를 제공.
      reference_choice: 가중·준용·분기 row 의 원범죄 명시 (예: "형법§347") — 응답이 안내.
      is_attempted: 미수 명시. 죄명 접미("살인미수" 등)로도 자동 인식되며, 접미와 이 플래그
        중 하나만 있어도 적용됩니다(OR).
      is_accessory: 방조 명시 — 인식 규칙은 is_attempted 와 동일.
      is_solicitor: 교사 명시 — 인식 규칙은 is_attempted 와 동일.
      statutory_modifications: 형법 §56 가중·감경 사유 list — lookup 응답의 enum에서 선택.
        지정 시 처단형 단계 진입.
      guideline_leaf_id: 양형기준 leaf id — lookup 응답의 후보에서 선택. 지정 시 권고형 단계 진입.
      guideline_factors: 특별 가중·감경 인자 dict — lookup 응답의 인자 enum에서 선택.
      sentence_months: 검증할 선고형(자유형, 월 단위) — 지정 시 final 단계(선고 가능 범위 검증).
      fine_amount: 검증할 선고형(벌금, 원 단위) — sentence_months 와 같은 final 단계 진입.
      probation_factors: 집행유예 4분면 인자 dict — lookup 응답의 enum에서 선택.
      act_count: 같은 charge 의 별개 행위 수 (동종 다행위). >=2 면 §37 전단·§38 ① 2호
        경합범 가중을 처단형에 자동 적용 (자유형 장기 1/2 가중, §42 ② 50년 cap).
        statutory_modifications 에 경합범_가중 항목을 명시하면 그 명시가 우선.
        포괄일죄·영업범처럼 판결문이 §37 을 명시하지 않는 유형은 1로 두세요.
    """
    # 진입부 정규화 — 넓은 스키마로 받은 값을 계산이 기대하는 타입으로 통일(위 시그니처 주석).
    charge = coerce_str(charge)
    sg_category_id = coerce_int(sg_category_id)
    statute_choice = coerce_str(statute_choice)
    branch_key = coerce_str(branch_key)
    reference_choice = coerce_str(reference_choice)
    statutory_modifications = coerce_dict_list(statutory_modifications)  # dict 원소만 — .get 크래시 차단
    guideline_leaf_id = coerce_int(guideline_leaf_id)
    guideline_factors = coerce_dict(guideline_factors)
    sentence_months = coerce_int(sentence_months)
    fine_amount = coerce_int(fine_amount)
    probation_factors = coerce_dict(probation_factors)
    offense_date = coerce_str(offense_date)
    act_count = coerce_int(act_count) or 1  # 스칼라/배열/None 방어 — 기본 1

    if not charge:
        return (
            "## status: missing_input\n"
            "- charge 인자 필요. 판결문 form 죄명 (예: 살인, 도로교통법위반(음주운전))."
        )

    conn = open_db()
    try:
        norm = _normalize_charge(
            conn,
            charge,
            is_attempted=is_attempted,
            is_accessory=is_accessory,
            is_solicitor=is_solicitor,
        )
        result = _lookup_charge(conn, norm.key, sg_category_id=sg_category_id)

        # statute_choice 명시 + (exact_cross_cat OR exact_same_cat_multi_row) → 해당 row 자동 선택
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
            # M28 Phase E — offense_date 지정 시 시점본 정량으로 swap (clm_versions)
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
            # 시점본 적용 trace — penalty.trace 에 prepend
            if version_meta and penalty:
                if version_meta.get('empty_version'):
                    # 빈 시점본 — override 건너뜀. 현행 base 정량 사용을 명시(오해 방지).
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
            # M33 — §42① 상한 시점보정 trace (version_meta 유무 무관)
            if art42_trace and penalty:
                penalty.trace.insert(0, art42_trace)
            # 행위시 본문 단락 — exact-match 후 모든 stage 응답 끝에 append.
            appendix = _historic_appendix(conn, payload_row, offense_date)

            if pending is not None:
                return _format_pending_response(norm, row, payload_row, pending) + appendix

            # stage 자동 분기 — 후속 인자 채워질수록 깊은 stage.
            # act_count >= 2 는 *처단형 계산 시* 자동 §37 가중 적용 — 그러나 stage
            # 트리거는 아님. lookup 응답을 깨지 않고, statutory_modifications/leaf_id
            # 가 채워질 때 비로소 처단형/권고형 stage 진입하면서 §37 가중도 반영됨.
            #
            # M23 — is_attempted/is_accessory suffix 자동 분리만으로 처단형 stage
            # 자동 진입하던 동작 제거. case 34 진단:
            #   `charge="살인미수"` 호출 → is_attempted=True auto-set → 종전엔 처단형
            #   stage 응답 (factor enum 미노출) → LLM 이 다음 호출에 schema key 추측 →
            #   `special_mitigating/aggravating` 같은 지어낸 key 입력 → 도구가 무시.
            # 변경 후: suffix 만으론 lookup 머무름. statutory_modifications 명시 시만
            # 처단형 진입. §25 자동 §법률상_임의감경 추가 (auto_from_suffix) 메커니즘은
            # _expand_implicit_modifications 안에 그대로 — 처단형 진입 시 trace 노출.
            needs_processed = statutory_modifications is not None
            needs_recommended = guideline_leaf_id is not None
            needs_final = sentence_months is not None or fine_amount is not None

            if not (needs_processed or needs_recommended or needs_final):
                return _format_lookup_response(
                    conn, norm, row, payload_row, penalty, act_count=act_count,
                ) + appendix

            # 처단형 계산 (어느 stage 든 필수 — 권고·final 도 처단형 의존)
            processed = _apply_statutory_modifications(
                penalty, norm, statutory_modifications, act_count=act_count,
                offense_iso=offense_iso,
            )

            if not (needs_recommended or needs_final):
                return _format_processed_response(
                    norm, row, payload_row, penalty, processed
                ) + appendix

            # 권고형 계산
            factors = _convert_factors_to_applied(guideline_factors)
            floor = _get_statute_floor(processed)
            rec = determine_range(
                conn, guideline_leaf_id, factors,
                legal_floor_months=floor,
                is_attempted=norm.modifiers.get("is_attempted", False),
            ) if guideline_leaf_id is not None else None
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
                    guideline_leaf_id,
                    floor,
                ) + appendix

            # final stage
            verify_lines = _verify_sentence(
                sentence_months, fine_amount, processed, rec, intersect
            )
            probation_lines = _probation_recommendation(
                sentence_months, fine_amount, processed, probation_factors
            )
            # M11: fine 정량 NULL + fine 옵션 보존된 row → 조문 본문 식 기반 룰 발췌
            fine_paragraphs: list[str] = []
            if processed.fine_min_won is None and processed.fine_max_won is None:
                fine_paragraphs = _extract_fine_paragraphs(
                    conn,
                    payload_row["statute_id"],
                    payload_row["article_no_num"],
                    payload_row["article_branch"],
                )
            # M13: 감경 적용 여부 — 선고 가능 범위 fine line 옆에 §55 ① 6호 명시용
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
                guideline_leaf_id or -1,
                floor,
                verify_lines,
                probation_lines,
                fine_paragraphs=fine_paragraphs,
                mit_applied=mit_applied,
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


# ---------- self-check ----------

if __name__ == "__main__":
    # 간단한 smoke test. 실제 RunContext 없이 직접 함수 호출 시뮬레이션.
    from collections import deque
    from types import SimpleNamespace

    fake_deps = SimpleNamespace(recent_calls=deque(maxlen=10))
    fake_ctx = SimpleNamespace(deps=fake_deps)

    cases = [
        # exact, 일반
        {"charge": "살인"},
        # suffix 자동 분리
        {"charge": "살인미수"},
        # 가중 — auto-match (상습공갈 → 공갈)
        {"charge": "상습공갈"},
        # 준용 — 옵션 list (위조공문서행사 → §225~§228 4 후보)
        {"charge": "위조공문서행사"},
        # 준용 + reference_choice 명시
        {"charge": "위조공문서행사", "reference_choice": "형법§225"},
        # 분기 — 옵션 list (준강도 → §333/§334)
        {"charge": "준강도"},
        # 분기 + reference_choice 명시
        {"charge": "준강도", "reference_choice": "형법§333"},
        # branch_options — 옵션 list (음주운전)
        {"charge": "도로교통법위반(음주운전)"},
        # branch_options + branch_key
        {"charge": "도로교통법위반(음주운전)", "branch_key": "③2"},
        # branch_options + invalid branch_key
        {"charge": "도로교통법위반(음주운전)", "branch_key": "③9"},
        # 누범가중 + branch=1 (폭처법 §2 ③)
        {"charge": "폭력행위등처벌에관한법률위반(공갈)"},
        # 누범가중 + branch_key 명시
        {"charge": "폭력행위등처벌에관한법률위반(공갈)", "branch_key": "③3"},
        # alias
        {"charge": "공용서류손상"},
        # cross-cat
        {"charge": "강도상해"},
        # cross-cat + cat 명시
        {"charge": "강도상해", "sg_category_id": 26},
        # fuzzy
        {"charge": "도로교통법위반"},
        # not_found
        {"charge": "음악산업진흥에관한법률위반"},
        # Phase C1: 처단형 — 자수 (법률상 임의감경 applied)
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
        # Phase C1: 누범 가중 (장기 2배) + 자수 (1/2 감경)
        {
            "charge": "절도",
            "statutory_modifications": [
                {"kind": "누범_가중", "type": "누범 (§35)", "basis": "형법 §35", "applied": True},
                {"kind": "법률상_임의감경", "type": "자수 (§52)", "basis": "형법 §52 ①", "applied": True},
            ],
        },
        # Phase C1: 미수 자동 적용 (살인미수 → §25 임의감경)
        {"charge": "살인미수"},
        # Phase C1: 본조_가중 (이미 reference 로 처리됐는데 추가 가중 가능 — 작량감경 with applied=False)
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
        # Phase C2: 권고형 — 살인 보통동기 leaf 216, 감경 factor 2개
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
        # Phase C2: 권고형 + 처단형 (자수 임의감경)
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
        # Phase C3+C4: final stage — sentence_months + probation_factors
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
        # Phase C4: 집유 적용 가능 (24월) + 4분면 일반 비교
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
