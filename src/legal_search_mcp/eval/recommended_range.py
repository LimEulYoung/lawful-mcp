"""양형기준 권고 형량범위 결정 함수.

양형위가 정한 누진 룰을 결정론적 함수로 구현. `harness/prompts.py:39-40` 의
자연어 룰을 코드화한 것. 평가에서 *gold range* 를 만들어 LLM 의 점 예측
과 비교 (`judge_in_range`, `within_range_position`).

룰 (양형기준 [공통원칙]):

1) 권고영역 결정
   - 행위 인자 (kind ∈ {행위, 행위_공통, 행위_미수}) 의 특별가중·특별감경 net
   - 행위자/기타 인자 (kind == 행위자_기타) 의 특별가중·특별감경 net
   - 둘 다 0 → 기본
   - 부호 일치 → 그 방향 (양수 가중, 음수 감경)
   - 부호 불일치 → 행위 인자 net 우월

2) 형량범위 특별 조정
   - 가중영역 + (특별가중 - 특별감경 ≥ 2) → 상한 1/2 가중
   - 감경영역 + (특별감경 - 특별가중 ≥ 2) → 하한 1/2 감경

일반양형인자는 영역 결정에 영향 없음 (선고형 결정 단계의 참작 사유).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Literal, Sequence

Level = Literal["감경", "기본", "가중"]
_ACT_KINDS = {"행위", "행위_공통", "행위_미수"}

# DEPRECATED (M6, 2026-05-22) — 카테고리 단위 법정형 floor 는 *대표 본조 근사* 라
# 카테고리 내 본조별 법정형 편차 큰 case (마약 §58~§61, 강도 §333~§337 등) 에 부정확.
# `compute_sentencing_range` 는 이제 `charge_legal_map.stat_imp_min_months` (본조별)
# 를 사용. 본 dict 는 호환성 유지용 빈 dict — 외부 script (`scripts/decompose_gap.py`)
# 에서 import 깨지지 않도록 두되, 실제 값은 비움.
LEGAL_FLOOR_MONTHS: dict[str, int] = {}


@dataclass(frozen=True)
class AppliedFactor:
    scope: Literal["특별", "일반"]
    kind: str  # 행위 / 행위_공통 / 행위_미수 / 행위자_기타
    direction: Literal["가중", "감경"]
    text: str = ""


@dataclass(frozen=True)
class RecommendedRange:
    level: Level
    min_months: int | None  # None: 하한 없음 (is_open_low)
    max_months: int | None  # None: 상한 없음 (is_unbounded_high or has_life)
    has_life: bool
    is_special_adjusted: bool  # 2차 특별조정 적용 여부
    raw_text: str


def _net(counts: dict[tuple[str, str], int], kinds: set[str]) -> int:
    """주어진 kind 집합 안에서 (특별가중 합) - (특별감경 합)."""
    agg = sum(counts.get((k, "가중"), 0) for k in kinds)
    mit = sum(counts.get((k, "감경"), 0) for k in kinds)
    return agg - mit


def _select_level(factors: Sequence[AppliedFactor]) -> Level:
    """1차: 권고영역 결정."""
    special = [f for f in factors if f.scope == "특별"]
    counts: dict[tuple[str, str], int] = {}
    for f in special:
        counts[(f.kind, f.direction)] = counts.get((f.kind, f.direction), 0) + 1

    act_net = _net(counts, _ACT_KINDS)
    actor_net = _net(counts, {"행위자_기타"})

    if act_net == 0 and actor_net == 0:
        return "기본"
    if act_net >= 0 and actor_net >= 0 and (act_net > 0 or actor_net > 0):
        return "가중"
    if act_net <= 0 and actor_net <= 0 and (act_net < 0 or actor_net < 0):
        return "감경"
    # 부호 불일치 → 행위 인자 우월
    if act_net > 0:
        return "가중"
    if act_net < 0:
        return "감경"
    return "기본"


def _special_adjusted(level: Level, factors: Sequence[AppliedFactor]) -> bool:
    """2차: 특별조정 적용 여부 (상한 1/2 가중 OR 하한 1/2 감경)."""
    sp_agg = sum(1 for f in factors if f.scope == "특별" and f.direction == "가중")
    sp_mit = sum(1 for f in factors if f.scope == "특별" and f.direction == "감경")
    if level == "가중" and sp_agg - sp_mit >= 2:
        return True
    if level == "감경" and sp_mit - sp_agg >= 2:
        return True
    return False


def _fetch_range_row(
    conn: sqlite3.Connection, leaf_id: int, level: Level
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT level, min_months, max_months, is_open_low, is_unbounded_high,"
        " has_life, raw_text FROM sg_ranges WHERE subtype_id=? AND level=?",
        (leaf_id, level),
    ).fetchone()


def determine_range(
    conn: sqlite3.Connection,
    leaf_id: int,
    factors: Sequence[AppliedFactor],
    legal_floor_months: int | None = None,
    is_attempted: bool = False,
) -> RecommendedRange | None:
    """leaf + 적용 factor → 양형위 권고 형량범위.

    `legal_floor_months`: 카테고리 법정형 하한. 권고 영역 하한이 floor 미만이면
    양형기준 [공통원칙] §02 ("권고가 처단형을 벗어나면 처단형이 기준") 에 따라
    floor 로 끌어올림. None 이면 보정 없음 (default).

    `is_attempted`: 미수 권고감경 룰은 sg_categories.attempt_recommend_discount
    (JSON {"low":[n,d],"high":[n,d]}) 에서 읽어 적용 — 현재 살인범죄군(cat 24)만
    하한 ×1/3·상한 ×2/3 (`24_살인범죄.md` line 29). 나머지 cat 은 NULL = 기수와 동일 권고.

    `sg_ranges` row 가 없는 leaf (벌금형 전용 등) 는 None.
    """
    level = _select_level(factors)
    base = _fetch_range_row(conn, leaf_id, level)
    if base is None:
        base = _fetch_range_row(conn, leaf_id, "기본")
        if base is None:
            return None
        level = "기본"

    lo, hi = base["min_months"], base["max_months"]
    has_life = bool(base["has_life"])
    unbounded_high = bool(base["is_unbounded_high"]) or has_life
    open_low = bool(base["is_open_low"])
    adj = _special_adjusted(level, factors)

    if adj:
        if level == "가중" and not unbounded_high and hi is not None:
            hi = hi + hi // 2
        elif level == "감경" and not open_low and lo is not None:
            lo = lo // 2

    # M22 — 미수 권고감경 (코드=로직, *대상군·비율은 DB*). 양형군별 룰을
    # sg_categories.attempt_recommend_discount(JSON)에서 읽는다 — 현재 살인범죄군
    # (cat 24)만 하한 ×1/3·상한 ×2/3 (24_살인범죄.md line 29):
    #   "살인미수범죄의 권고 형량범위는 하한을 1/3, 상한을 2/3로 각 감경. 단, '무기'는
    #    '20년 이상'으로, '무기 이상'은 '20년 이상, 무기'로 각 감경하여 적용."
    # 단순 imp 범위(닫힘·life=False·unbounded=False)만 적용. '무기'/'무기이상' 포함 row
    # 는 별도 보강 대상 — 아래 가드로 제외.
    if is_attempted and lo is not None and hi is not None and not has_life and not unbounded_high:
        disc_row = conn.execute(
            "SELECT c.attempt_recommend_discount FROM sg_subtypes s "
            "JOIN sg_categories c ON c.id = s.category_id WHERE s.id=?",
            (leaf_id,),
        ).fetchone()
        if disc_row and disc_row[0]:
            disc = json.loads(disc_row[0])
            ln, ld = disc["low"]
            hn, hd = disc["high"]
            lo = lo * ln // ld
            hi = hi * hn // hd

    final_min = None if open_low else lo
    final_max = None if unbounded_high else hi

    # 법정형 floor 적용 — 권고 영역이 floor 보다 아래로 내려가면 floor 가 강제 하한
    if legal_floor_months is not None:
        if final_min is None or final_min < legal_floor_months:
            final_min = legal_floor_months
        if final_max is not None and final_max < legal_floor_months:
            final_max = legal_floor_months

    return RecommendedRange(
        level=level,
        min_months=final_min,
        max_months=final_max,
        has_life=has_life,
        is_special_adjusted=adj,
        raw_text=base["raw_text"],
    )


def in_range(months: float, r: RecommendedRange) -> bool:
    """선고 형량(개월) 이 권고 영역 안인지."""
    if r.min_months is not None and months < r.min_months:
        return False
    if r.max_months is not None and months > r.max_months:
        return False
    return True


def within_range_position(months: float, r: RecommendedRange) -> float | None:
    """영역 안에서의 상대 위치 [0, 1]. 하한/상한 open 이면 None."""
    if r.min_months is None or r.max_months is None:
        return None
    span = r.max_months - r.min_months
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, (months - r.min_months) / span))


# ---------- self-check ----------

if __name__ == "__main__":
    from harness.deps import open_db

    conn = open_db()

    # Case 1: 살인 보통동기 (leaf 216), 2018고합281
    # 특별감경: 피해자 유발(강함) 1 (행위_공통/감경)
    # 특별가중: 0
    # → 감경영역 [84, 144], 특별조정 X
    # 실제 144개월 → in_range=True, position=1.0 (상한)
    살인_factors = [
        AppliedFactor("특별", "행위_공통", "감경", "피해자 유발(강함)"),
        AppliedFactor("일반", "행위자_기타", "감경", "진지한 반성"),
    ]
    r = determine_range(conn, 216, 살인_factors)
    print(f"살인 216: level={r.level} range=[{r.min_months}, {r.max_months}]"
          f" adj={r.is_special_adjusted}")
    assert r.level == "감경", r.level
    assert (r.min_months, r.max_months) == (84, 144), (r.min_months, r.max_months)
    assert not r.is_special_adjusted
    assert in_range(144, r)
    assert within_range_position(144, r) == 1.0

    # Case 2: 사기 일반/1억-5억 (leaf 188), 2017고단384
    # 특별가중: 반복 행위 + 심각한 피해 (행위) + 동종 누범 (행위자) = 3
    # 특별감경: 0
    # → 가중영역 [30, 72], 특별조정: 가중-감경=3 ≥ 2 → 상한 1/2 → [30, 108]
    # 실제 36개월 → in_range=True, position ~ 0.077 (하한 근처)
    사기_factors = [
        AppliedFactor("특별", "행위", "가중", "반복적 범행"),
        AppliedFactor("특별", "행위", "가중", "심각한 피해"),
        AppliedFactor("특별", "행위자_기타", "가중", "동종 누범"),
        AppliedFactor("일반", "행위자_기타", "감경", "진지한 반성"),
    ]
    r = determine_range(conn, 188, 사기_factors)
    print(f"사기 188: level={r.level} range=[{r.min_months}, {r.max_months}]"
          f" adj={r.is_special_adjusted}")
    assert r.level == "가중", r.level
    assert (r.min_months, r.max_months) == (30, 108), (r.min_months, r.max_months)
    assert r.is_special_adjusted
    assert in_range(36, r)
    pos = within_range_position(36, r)
    print(f"  pos(36) = {pos:.3f}")
    assert pos is not None and pos < 0.1

    # Case 3: 모두 0 → 기본
    r = determine_range(conn, 188, [])
    assert r.level == "기본"
    assert (r.min_months, r.max_months) == (12, 48), (r.min_months, r.max_months)

    # Case 4: 부호 불일치 (행위 우월). 사기 188 — 행위 특별가중 1 / 행위자 특별감경 1
    mixed = [
        AppliedFactor("특별", "행위", "가중", "반복적 범행"),
        AppliedFactor("특별", "행위자_기타", "감경", "처벌불원"),
    ]
    r = determine_range(conn, 188, mixed)
    # 행위 net=+1, 행위자 net=-1 → 행위 우월 → 가중
    assert r.level == "가중", r.level
    assert not r.is_special_adjusted  # 차이 0 < 2

    print("\nall self-checks passed.")
    conn.close()
