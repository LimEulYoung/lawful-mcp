"""Recommended sentencing range under the Sentencing Commission guidelines.

The guidelines state their rules in prose; this is that prose as a
deterministic function. Given the guideline leaf a case falls under and the
factors a court found, it returns the range the guidelines recommend.

The rules, from the guidelines' common principles:

1) Pick the band (mitigated / basic / aggravated)
   - Net special factors among conduct factors
     (kind in {행위, 행위_공통, 행위_미수})
   - Net special factors among offender and other factors (행위자_기타)
   - Both zero -> basic
   - Same sign -> that direction (positive aggravates, negative mitigates)
   - Opposite signs -> the conduct net wins

2) Special adjustment of the range
   - Aggravated band, and (special aggravating - special mitigating) >= 2
     -> raise the upper bound by half
   - Mitigated band, and (special mitigating - special aggravating) >= 2
     -> halve the lower bound

Ordinary factors do not move the band. They are weighed later, when the
court fixes the sentence within the range.

Terms in the data stay Korean because they are values in the corpus, not
labels chosen here: 특별/일반 (special/ordinary), 가중/감경 (aggravating/
mitigating), 행위/행위자_기타 (conduct / offender and other).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Literal, Sequence

Level = Literal["감경", "기본", "가중"]
_ACT_KINDS = {"행위", "행위_공통", "행위_미수"}

# Deprecated and intentionally empty. A statutory floor per guideline
# category was only ever an approximation from a representative article, and
# it is wrong wherever articles inside one category carry different statutory
# ranges (narcotics and robbery both do). The floor now comes per-article
# from the charge-to-penalty map. Kept as an empty dict so an outside import
# does not break.
LEGAL_FLOOR_MONTHS: dict[str, int] = {}


@dataclass(frozen=True)
class AppliedFactor:
    scope: Literal["특별", "일반"]      # special / ordinary
    kind: str                           # 행위 / 행위_공통 / 행위_미수 / 행위자_기타
    direction: Literal["가중", "감경"]  # aggravating / mitigating
    text: str = ""


@dataclass(frozen=True)
class RecommendedRange:
    level: Level
    min_months: int | None  # None means no lower bound
    max_months: int | None  # None means no upper bound, or life is available
    has_life: bool
    is_special_adjusted: bool  # whether the second-stage adjustment applied
    raw_text: str


def _net(counts: dict[tuple[str, str], int], kinds: set[str]) -> int:
    """(special aggravating) - (special mitigating) within a set of kinds."""
    agg = sum(counts.get((k, "가중"), 0) for k in kinds)
    mit = sum(counts.get((k, "감경"), 0) for k in kinds)
    return agg - mit


def _select_level(factors: Sequence[AppliedFactor]) -> Level:
    """Stage one: pick the band."""
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
    # Signs disagree: the conduct factors decide.
    if act_net > 0:
        return "가중"
    if act_net < 0:
        return "감경"
    return "기본"


def _special_adjusted(level: Level, factors: Sequence[AppliedFactor]) -> bool:
    """Stage two: does the range get the half-step adjustment?"""
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
    """Guideline leaf plus applied factors -> the recommended range.

    ``legal_floor_months``: the statutory minimum. Where the recommended
    range would fall below it, the guidelines defer to the statute, so the
    floor is raised. None disables the correction.

    ``is_attempted``: the discount for an attempt is data, not code. Which
    offence groups get one, and by how much, is read from
    ``sg_categories.attempt_recommend_discount``.

    Returns None for a leaf with no range row — a fine-only leaf, for
    instance.
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

    # Attempt discount. The ratio is stored per offence group as JSON
    # ({"low": [n, d], "high": [n, d]}), so adding a group is a data change.
    # Applied only to a plainly bounded term range: where the guideline text
    # reads "life" or "life or more", the discount is expressed in words
    # rather than a ratio, so those rows are excluded here.
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

    # The statutory floor wins over a recommendation that falls below it.
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
    """Is a sentence, in months, inside the recommended range?"""
    if r.min_months is not None and months < r.min_months:
        return False
    if r.max_months is not None and months > r.max_months:
        return False
    return True


def within_range_position(months: float, r: RecommendedRange) -> float | None:
    """Relative position in the range, 0 to 1. None if either end is open."""
    if r.min_months is None or r.max_months is None:
        return None
    span = r.max_months - r.min_months
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, (months - r.min_months) / span))


# Worked examples. Run this module directly to check the rules against the
# corpus: `python -m legal_search_mcp.eval.recommended_range`.

if __name__ == "__main__":
    from ..deps import open_db

    conn = open_db()

    # Murder, ordinary motive (leaf 216).
    # One special mitigating conduct factor, no aggravating ones
    # -> mitigated band [84, 144], no special adjustment.
    murder_factors = [
        AppliedFactor("특별", "행위_공통", "감경", "피해자 유발(강함)"),
        AppliedFactor("일반", "행위자_기타", "감경", "진지한 반성"),
    ]
    r = determine_range(conn, 216, murder_factors)
    print(f"murder (leaf 216): level={r.level} range=[{r.min_months}, {r.max_months}]"
          f" adjusted={r.is_special_adjusted}")
    assert r.level == "감경", r.level
    assert (r.min_months, r.max_months) == (84, 144), (r.min_months, r.max_months)
    assert not r.is_special_adjusted
    assert in_range(144, r)
    assert within_range_position(144, r) == 1.0

    # Fraud, 100M-500M won (leaf 188).
    # Three special aggravating factors, none mitigating -> aggravated band
    # [30, 72]; the difference is 3, so the upper bound rises by half to 108.
    fraud_factors = [
        AppliedFactor("특별", "행위", "가중", "반복적 범행"),
        AppliedFactor("특별", "행위", "가중", "심각한 피해"),
        AppliedFactor("특별", "행위자_기타", "가중", "동종 누범"),
        AppliedFactor("일반", "행위자_기타", "감경", "진지한 반성"),
    ]
    r = determine_range(conn, 188, fraud_factors)
    print(f"fraud (leaf 188): level={r.level} range=[{r.min_months}, {r.max_months}]"
          f" adjusted={r.is_special_adjusted}")
    assert r.level == "가중", r.level
    assert (r.min_months, r.max_months) == (30, 108), (r.min_months, r.max_months)
    assert r.is_special_adjusted
    assert in_range(36, r)
    pos = within_range_position(36, r)
    print(f"  position of 36 months = {pos:.3f}")
    assert pos is not None and pos < 0.1

    # No factors at all -> basic band.
    r = determine_range(conn, 188, [])
    assert r.level == "기본"
    assert (r.min_months, r.max_months) == (12, 48), (r.min_months, r.max_months)

    # Signs disagree: one aggravating conduct factor against one mitigating
    # offender factor. Conduct wins, and a difference of 0 is below the
    # threshold for the special adjustment.
    mixed = [
        AppliedFactor("특별", "행위", "가중", "반복적 범행"),
        AppliedFactor("특별", "행위자_기타", "감경", "처벌불원"),
    ]
    r = determine_range(conn, 188, mixed)
    assert r.level == "가중", r.level
    assert not r.is_special_adjusted

    print("\nall self-checks passed.")
    conn.close()
