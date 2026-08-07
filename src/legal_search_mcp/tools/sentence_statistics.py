"""sentence_statistics — 그 죄명 하나만 유죄인 피고인(다른 죄명 없음)의 1심 선고 분포 + 비교/사례.

풀 정의(v2, 2026-07-06 — 표본 설계 원칙: 애매하면 통계를 내지 말고 원자료를 보여준다):
  - **1심 전용**: `prec_sentences.instance='1심'` (추출 시 사건번호 고합/고단/고정 술어로
    스탬프). 소스(source) 불문 — scourt takeover 로 prec_cases.source 가 바뀌어도 표본 불변
    (구 STATS_SOURCES 소스 리스트 방식은 takeover 로 표본 65% 증발 — 폐지).
    2심(노)은 영구 제외: 감경 경향 + 파기자판만 선별 시 계통 편향.
  - **통계 = 단일 죄명 전용**: `prec_defendants.n_charges=1`(죄명 "종류" 1개 — 같은 죄명
    여러 건[동종경합]은 포함, 다른 죄명과의 경합[이종경합]은 제외: 형이 여러 죄를 포괄해 오염).
    서로 다른 죄명의 단독 관측분포를 합산·가중해 경합범 분포로 만들 수 없다. 형법 제38조의
    가중 상한은 법정 처단형 규칙일 뿐 경험적 선고분포의 결합 공식이 아니다.
  - n >= MIN_N_STATS(30): 통계(mean/std/집유율) + severity 11분위 comparable.
  - n <  MIN_N_STATS: **통계 미산출**(소표본 mean/std 가 anchor 로 오용되는 것 차단) →
    단일 죄명 개별 사례만 나열. 경합 사례는 하나의 형이 여러 죄를 포괄해 귀속할 수 없으므로
    표본 보충에도 쓰지 않는다.

평가셋(judgments_499 499건 + 원본 950건 cross-source) leak 는
`HarnessDeps.exclude_case_ids` frozenset 으로 차단 — `_fetch_samples` 가
`pd.case_id NOT IN (...)` 로 자동 제외(prod deps 엔 속성 없음 → 빈 set).

다중 피고인 case 는 *피고인 단위 row* 가 표본 — 한 case 의 A·B 가 양형 다르면 둘 다.
"""
from __future__ import annotations

import re
import sqlite3
import statistics
from typing import Any

from pydantic_ai import RunContext

from .._charge import norm_charge as _norm_charge
from ._coerce import coerce_int, coerce_list, coerce_str
from ..config import case_url_base
from ..deps import HarnessDeps, open_db
from ._dedup import dedup_guard

# ---------- 상수 ----------

STATS_INSTANCE = "1심"

# 통계 산출 최소 표본 — 미만이면 그리드 모드(통계 미산출)
MIN_N_STATS = 30

GRID_MAX = 15            # 단일 charge 입력 시 그리드 최대 행
GRID_MULTI_MAX = 5       # 다중 charge 입력 시 죄명당 그리드 최대 행
RELATED_MAX = 15         # family '관련 죄명' 표시 상한(n 내림차순; 강간·상해 등 큰 계열)
CATEGORY_MIN_POOLS = 5   # category 진입점 최소 pool(이하 잡동 category 는 미노출)

# 구어·속어·상위어 → 표준 질의(카테고리·base·family 로 재해석). 소규모 수기 시드(§6-2).
# 값은 다시 리졸버 체인을 타므로 category명/base_crime/family 키 중 무엇이든 가능.
_ALIAS = {
    "마약": "마약범죄", "마약류": "마약범죄", "마약사범": "마약범죄", "필로폰": "마약범죄",
    "히로뽕": "마약범죄", "대마": "마약범죄", "대마초": "마약범죄", "뽕": "마약범죄",
    "몰카": "카메라등이용촬영", "도촬": "카메라등이용촬영", "불법촬영": "카메라등이용촬영",
    "보이스피싱": "사기", "전화금융사기": "사기", "전세사기": "사기", "취업사기": "사기",
    "중고사기": "사기", "대포통장": "전자금융거래법위반", "성폭력": "성범죄",
    "음주운전": "도로교통법위반(음주운전)", "음주": "도로교통법위반(음주운전)",
    "뺑소니": "도주", "성매수": "성매매", "성구매": "성매매",
    # 3차 확충(2026-07-09 2세션) — 매치처 존재 검증분(canonical/base/charge_norm/family)
    "무면허운전": "도로교통법위반(무면허운전)", "음주측정거부": "도로교통법위반(음주측정거부)",
    "통매음": "통신매체이용음란",
    # ⚠ law_base(§Task3) 가 특별법 죄명을 법률명 base 로 흡수 → bare 죄명은 미해결.
    #   촬영물 협박/반포는 canonical(성폭력특례법(...)) 로 직행해야 resolve=pool.
    "몸캠": "성폭력범죄의처벌등에관한특례법위반(촬영물등이용협박)",
    "몸캠피싱": "성폭력범죄의처벌등에관한특례법위반(촬영물등이용협박)",
    "리벤지포르노": "성폭력범죄의처벌등에관한특례법위반(카메라등이용촬영물반포등)",
    "유사수신": "유사수신행위의규제에관한법률위반",
    "짝퉁": "상표법위반", "가품": "상표법위반", "위조상품": "상표법위반",
    "사설토토": "도박", "온라인도박": "도박", "인터넷도박": "도박",
}

_GUIDANCE_CONCURRENT = (
    "각 결과는 그 죄명만 유죄인 피고인의 관측분포입니다. 죄명별 분포를 합산·평균하거나 "
    "1.5배 하여 경합범의 선고분포로 만들지 마세요. 형법 제38조 제1항 제2호는 같은 종류의 "
    "형에서 가장 중한 죄의 장기(벌금은 다액)에 2분의 1까지 가중하되, 각 죄의 장기 또는 "
    "다액을 합산한 범위를 넘지 못하게 하는 법정 처단형 상한 규칙이며 통계 결합 공식이 아닙니다."
)
_GUIDANCE_RECALL = "비교 case 11분위가 필요하면 charges 를 죄명 하나로 재호출하세요."

# ---------- 정규화 ----------
# 공통 모듈 — eval.metrics + scripts/build_sentences.py도 동일 import.


# ---------- 통계 ----------

def _percentile(sorted_vals: list[int], p: float) -> int | None:
    """이미 정렬된 list에서 percentile 값 — R-7 linear interpolation (numpy 기본).

    표본 작을 때 `int(n*p)` 식은 결과가 상위로 한 칸씩 밀려 분포가 과대평가됨
    (예: n=10, p=0.9 → max). 양형 분포에 그대로 노출되면 LLM의 형량 추정이 위로 편향.
    """
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    pos = p * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return int(round(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac))


def _imprisonment_stats(rows: list[sqlite3.Row]) -> dict[str, Any]:
    """imprisonment 분포 통계 — deciles에 없는 형종별 mean/std/집유 정보만.

    분위수(p10~p90)는 severity-united deciles가 대체 (중복 토큰 제거).
    """
    months = [r["sentence_months"] for r in rows if r["sentence_months"] is not None]
    if not months:
        return {}
    mean = round(statistics.mean(months), 1)
    std = round(statistics.stdev(months), 1) if len(months) >= 2 else 0.0

    probation_count = sum(1 for r in rows if r["probation"])
    probation_months_vals = sorted(
        r["probation_months"] for r in rows
        if r["probation"] and r["probation_months"] is not None
    )

    return {
        "mean": mean,
        "std": std,
        "probation_ratio": round(probation_count / len(rows), 2),
        "probation_months_p50": _percentile(probation_months_vals, 0.5),
    }


def _fine_stats(rows: list[sqlite3.Row]) -> dict[str, Any]:
    """벌금 분포 통계 — deciles에 없는 mean만 (분위수는 deciles가 대체)."""
    amounts = [r["fine_amount"] for r in rows if r["fine_amount"] is not None]
    if not amounts:
        return {}
    return {"mean": int(statistics.mean(amounts))}


def _by_type(rows: list[sqlite3.Row]) -> dict[str, int]:
    out = {"imprisonment": 0, "fine": 0, "not_guilty": 0, "life": 0}
    for r in rows:
        st = r["sentence_type"]
        if st in out:
            out[st] += 1
    return out


# ---------- 비슷한 예시 (stratified) ----------

def _severity_key(row: sqlite3.Row) -> tuple[int, int]:
    """severity 순서: life > imprisonment > fine > not_guilty. 형종 내에선 형량 오름차순."""
    st = row["sentence_type"]
    if st == "life":
        return (3, 999_999)  # 무기징역/사형 — 가장 무거움
    if st == "imprisonment":
        return (2, row["sentence_months"] or 0)
    if st == "fine":
        # 벌금 1000만원 ≒ 징역 1개월 정도의 무게로 환산 (very rough)
        return (1, (row["fine_amount"] or 0) // 1_000_000)
    return (0, 0)  # not_guilty


def _stratified_sample(
    rows: list[sqlite3.Row],
    *,
    reference_year: int | None = None,
) -> list[tuple[int, sqlite3.Row]]:
    """severity 정렬 후 11분위(q=0/10/.../100) 절단점에서 row 1건씩 추출.

    각 q에 대해 ``idx = round((q/100) * (n-1))``의 row를 잡는다. 동률
    (severity_key 동일) row가 여러 개일 때 ``reference_year`` 가까운 row가
    secondary key로 우선 선택된다.

    row 수 ≤ 11이면 정렬 그대로 반환 + q는 idx 위치 비율로 추정. 분위 절단점에
    같은 case가 중복 등장하면 가장 낮은 q만 유지 (case 중복 토큰 절약).
    """
    if not rows:
        return []
    if reference_year is not None:
        sorted_rows = sorted(
            rows,
            key=lambda r: (
                _severity_key(r),
                abs((r["decision_year"] or 0) - reference_year),
            ),
        )
    else:
        sorted_rows = sorted(
            rows,
            key=lambda r: (_severity_key(r), -(r["decision_year"] or 0)),
        )
    n = len(sorted_rows)
    if n == 1:
        return [(50, sorted_rows[0])]
    if n <= 11:
        return [(round(i / (n - 1) * 100), r) for i, r in enumerate(sorted_rows)]

    out: list[tuple[int, sqlite3.Row]] = []
    seen: set[tuple[int, str]] = set()
    for q in range(0, 101, 10):
        idx = round((q / 100) * (n - 1))
        r = sorted_rows[idx]
        key = (r["case_id"], r["defendant_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append((q, r))
    return out


def _grid_sort(
    rows: list[sqlite3.Row], reference_year: int | None
) -> list[sqlite3.Row]:
    """그리드(사례 나열)용 정렬 — 기준연도 근접순, 없으면 최신순."""
    if reference_year is not None:
        return sorted(
            rows,
            key=lambda r: (
                abs((r["decision_year"] or 0) - reference_year),
                -(r["decision_year"] or 0),
            ),
        )
    return sorted(rows, key=lambda r: -(r["decision_year"] or 0))


# ---------- row 직렬화 ----------

REASON_MAX = 160  # sentencing_reason 표시 캡 — 히스토리 재생 절단(MAX_TOOL_OUTPUT) 여유 확보


def _sentence_str(row: sqlite3.Row) -> str:
    st = row["sentence_type"]
    if st == "imprisonment":
        s = f"징역 {row['sentence_months']}월"
        if row["probation"]:
            pm = row["probation_months"]
            s += f" 집유{pm}월" if pm is not None else " 집유"
        return s
    if st == "fine":
        amt = row["fine_amount"]
        return f"벌금 {amt:,}원" if amt is not None else "벌금"
    if st == "life":
        return "무기징역/사형"
    return "무죄"


def _short_case_no(case_number: str | None) -> str:
    """병합 사건번호(콤마 나열, ~100자)는 첫 번호 + 병합 수로 축약 — 전체는 url 페이지에."""
    parts = [p.strip() for p in (case_number or "").split(",") if p.strip()]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} 외 {len(parts) - 1}건 병합"


def _case_lines(
    row: sqlite3.Row,
    charges_by_def: dict[tuple[int, str], list[str]],
    *,
    concurrent: bool = False,
) -> list[str]:
    """comparable/grid 공용 case 본문 라인(들여쓰기 키:값).

    토큰 절약: case_id 별도 라인 없음(url 말미가 id), 사건번호·연도·피고인 = 한 줄,
    charges 라인은 경합(concurrent) 행에만 — 단독범 행은 질의 죄명과 항상 동일해 생략.
    """
    lines = [
        f"  case: {_short_case_no(row['case_number'])} ({row['decision_year']}) · def {row['defendant_id']}",
        f"  url: {case_url_base()}/cases/{row['case_id']}",
        f"  sentence: {_sentence_str(row)}",
    ]
    if concurrent:
        key = (row["case_id"], row["defendant_id"])
        chg = "/".join(charges_by_def.get(key, []))
        lines.append(f"  charges: {chg}")
        lines.append("  concurrent: true")
    reason = row["sentencing_reason"]
    if reason:
        if len(reason) > REASON_MAX:
            reason = reason[: REASON_MAX - 1] + "…"
        lines.append(f"  reason: {reason}")
    return lines


# ---------- markdown 직렬화 ----------

def _format_response_md(resp: dict[str, Any]) -> str:
    """응답 dict → markdown-KV 문자열. LLM 토큰 효율 ↑ (JSON 대비 ~25% 절감).

    구조: `## section`별 헤더 + `- key: value` 들여쓰기. charge_blocks 는 죄명당
    `## charge:` 헤더 하나. 사례(comparable/grid)는 `- q:`/`- case:` 블록 단위.
    """
    lines: list[str] = [f"## status: {resp.get('status', 'ok')}"]
    if resp.get("mode"):
        lines.append(f"## mode: {resp['mode']}")

    # 후보 목록(status=candidates) — 죄명 선택지, charge_id 로 재호출
    if resp.get("status") == "candidates":
        lines.append(f"## query: {resp.get('query', '')}")
        if resp.get("note"):
            lines.append(f"## {resp['note']}")
        cands = resp.get("candidates") or []
        related = resp.get("related") or []
        if not cands and not related:
            lines.append("- (해당 죄명 계열 후보 없음 — 다른 표기로 재시도)")
        if cands:
            lines.append(
                "## 후보 죄명 (사안에 맞는 charge_id 하나로 재호출 → 통계). "
                "n=단일 죄명 1심 표본수(30↑ 통계, 미만 그리드)")
            for c in cands:
                cat = f" [{c['category']}]" if c.get("category") else ""
                lines.append(
                    f"- charge_id={c['charge_id']} · n={c['n']}{cat} · {c['label']}")
        if related:
            lines.append(
                "## 관련 죄명 (같은 계열의 다른 죄 — 해당하면 charge_id 로 통계). "
                "직접 죄명이 아니면 이쪽을 확인")
            for c in related[:RELATED_MAX]:
                cat = f" [{c['category']}]" if c.get("category") else ""
                lines.append(
                    f"- charge_id={c['charge_id']} · n={c['n']}{cat} · {c['label']}")
            if len(related) > RELATED_MAX:
                lines.append(
                    f"- (외 {len(related) - RELATED_MAX}개 더 — 죄명을 좁혀 검색하세요)")
        return "\n".join(lines)

    for blk in resp.get("charge_blocks") or []:
        lines.append(f"## charge: {blk['charge']}")
        lines.append(f"- n: {blk['n']} (단일 죄명)")
        bt = blk.get("by_type")
        if bt:
            lines.append(
                "- by_type: " + " / ".join(f"{k} {v}" for k, v in bt.items() if v)
            )
        imp = blk.get("imprisonment")
        if imp:
            lines.append(
                "- imprisonment: "
                + " ".join(f"{k}={v}" for k, v in imp.items() if v is not None)
            )
        fine = blk.get("fine")
        if fine:
            lines.append(
                "- fine: " + " ".join(f"{k}={v}" for k, v in fine.items())
            )
        for note in blk.get("notes") or []:
            lines.append(f"- note: {note}")

        cc = blk.get("comparables") or []
        if cc:
            lines.append("## comparable_cases (severity-united 11분위, 단일 죄명 풀)")
            for q, row, cbd in cc:
                lines.append("\n".join([f"- q: {q}", *_case_lines(row, cbd)]))

        grid = blk.get("grid") or []
        if grid:
            header = f"## case_grid: {blk['charge']} (개별 사례 나열 — 통계 아님, 분포 일반화 금지"
            if any(conc for _r, _c, conc in grid):
                header += "; concurrent 행 = 경합범, 형은 전체 죄에 대한 하나의 선고"
            header += ")"
            lines.append(header)
            for row, cbd, concurrent in grid:
                body = _case_lines(row, cbd, concurrent=concurrent)
                lines.append("\n".join(["- " + body[0].lstrip(), *body[1:]]))

    if resp.get("guidance"):
        lines.append("## guidance")
        for g in resp["guidance"]:
            lines.append(f"- {g}")

    if resp.get("charges_normalized"):
        lines.append(f"## charges_normalized: {', '.join(resp['charges_normalized'])}")

    um = resp.get("unmatched_charges") or []
    if um:
        lines.append("## unmatched_charges")
        for u in um:
            sug = ", ".join(u.get("suggested") or []) or "(없음)"
            lines.append(f"- input: {u['input']}  suggested: {sug}")

    warnings = resp.get("warnings") or []
    if warnings:
        lines.append("## warnings")
        for w in warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)


# ---------- 메인 도구 ----------

def _coerce_single_str(x: Any) -> str | None:
    """스칼라 기대 — 정상 스칼라는 무경고. list/JSON배열 문자열이면 첫 값만(경합은 개별 호출)."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        for v in x:
            s = coerce_str(v)
            if s:
                return s
        return None
    if isinstance(x, str) and x.strip()[:1] == "[":   # 이중 인코딩된 배열 문자열 방어
        parsed = coerce_list(x)
        if isinstance(parsed, list):
            for v in parsed:
                s = coerce_str(v)
                if s:
                    return s
            return None
    return coerce_str(x)


def _coerce_single_int(x: Any) -> int | None:
    """스칼라 기대 — 정상 int 는 무경고. list/JSON배열 문자열이면 첫 정수만."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        for v in x:
            iv = coerce_int(v)
            if iv is not None:
                return iv
        return None
    if isinstance(x, str) and x.strip()[:1] == "[":   # 이중 인코딩된 배열 문자열 방어
        parsed = coerce_list(x)
        if isinstance(parsed, list):
            for v in parsed:
                iv = coerce_int(v)
                if iv is not None:
                    return iv
            return None
    return coerce_int(x)


@dedup_guard("sentence_statistics")
def sentence_statistics(
    ctx: RunContext[HarnessDeps],
    charges: str | None = None,
    charge_id: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    reference_year: int | None = None,
) -> str:
    """단일 죄명 1심 선고 분포 — charges 또는 charge_id 중 하나는 필수입니다. charges로
    후보를, charge_id로 표본 30↑ 통계와 비교판례(30 미만은 단독 개별 사례)를 반환합니다.

    언제:
    - 형량 전망·구형·양형 의견·사건 위치를 검토할 때 결론 전에 호출하세요.
    - compute_sentencing_range 의 공식 '범위'가 실무에서 어디 안착하는지 실데이터로 받칠 때.

    규칙:
    - charges 하나로 status=candidates를 받은 뒤 맞는 charge_id 하나로 재호출하세요.
    - 죄명·charge_id 는 한 번에 하나씩입니다 — 여러 죄명을 한 번에 넣지 마세요.
    - 이종경합은 통계·사례에서 제외합니다. 죄명별 분포를 합산·평균·1.5배해 경합범 분포로
      만들지 마세요. 형법 제38조 제1항 제2호는 가장 중한 죄 장기(벌금은 다액)의 1/2까지,
      각 죄 장기·다액 합계 이내로 가중하는 처단형 상한이지 통계 결합식이 아닙니다.
    - status=candidates는 통계가 아니며, low_n_grid 사례는 일반화하면 안 됩니다.

    응답: markdown-KV. 비교 사례는 반환 url만 인용 링크로 쓰고 집계 수치는 링크 없이 제시.

    Args:
      charges: 후보 검색용 죄명 하나. 구어·카테고리·법률명도 인식.
      charge_id: 후보에서 고른 pool 식별자 하나(int) — 통계 조회용. charges 와 택일.
      year_from: 판결 연도 범위 필터 시작.
      year_to: 판결 연도 범위 필터 끝. year_from 과 함께 또는 단독 사용.
      reference_year: 비교 case·그리드 선택의 연도 기준 — 가까운 사건 우선. None 이면 최근 우선.
    """
    q_raw = _coerce_single_str(charges)
    cid = _coerce_single_int(charge_id) if charge_id is not None else None
    if charge_id is not None and cid is None:
        return _format_response_md(
            {"status": "no_data", "warnings": ["charge_id 파싱 실패"]})
    if q_raw and cid is not None:
        return _format_response_md({
            "status": "invalid_input",
            "warnings": ["charges와 charge_id는 함께 쓰지 말고 둘 중 하나만 지정하세요."],
        })
    if cid is None and not q_raw:
        return _format_response_md({
            "status": "missing_input",
            "warnings": ["charges 또는 charge_id 중 하나는 필요합니다."],
        })
    qnorm = _norm_charge(q_raw) if q_raw else None
    if q_raw and not qnorm:
        return _format_response_md(
            {"status": "no_data", "warnings": ["charges 정규화 후 비어있음"]})

    conn = open_db()
    try:
        exclude_ids = getattr(ctx.deps, "exclude_case_ids", None) or frozenset()
        tax = _tax_available(conn)

        # ── charge_id 모드(조회): 특정 pool 통계 (하나) ──
        #    경합도 죄명별 단독 관측분포만 조회 — 리스트/자동선택·분포 결합 없음.
        if cid is not None:
            return _format_response_md(_blocks_from_pools(
                conn, [cid], year_from=year_from, year_to=year_to,
                exclude_ids=exclude_ids, reference_year=reference_year))

        # ── 텍스트 charges 모드 (하나) ──
        assert q_raw and qnorm

        # 접미 '죄' 제거 — '사기죄'→'사기'(일반 사용자 최빈 입력). 죄명(base/canonical/
        # charge_norm)은 '죄'로 끝나는 게 taxonomy 에 0건이라 안전. ⚠ category 는 전부
        # '…범죄'로 끝나므로 음의 lookbehind 로 보존('성범죄'→'성범' 파괴 방지, §6-2 진입점).
        qnorm = re.sub(r"(?<!범)죄$", "", qnorm) or qnorm

        # taxonomy 부재 → 레거시 exact-match 경로
        if not tax:
            return _format_response_md(_legacy_stats(
                conn, [qnorm], qnorm != q_raw, year_from=year_from,
                year_to=year_to, exclude_ids=exclude_ids,
                reference_year=reference_year))

        # ⓪ alias 재작성 (구어·상위어 → 표준 질의: category/base/family)
        qnorm = _ALIAS.get(qnorm, qnorm)

        # ① bare 법률명(괄호 없는 '…법위반') → 그 법률 subtype 후보 완전 나열
        law_c = _lawname_candidates(conn, qnorm, year_from, year_to, exclude_ids)
        if law_c:
            return _format_response_md(
                {"status": "candidates", "query": qnorm, "candidates": law_c,
                 "note": f"'{qnorm}' 법률의 죄명 후보(괄호 subtype)"})

        # ② 상위계열(family) — "사기" → 직접(base=사기) + 관련(컴퓨터등사용사기·보험사기…)
        if _is_family(conn, qnorm):
            direct = _candidates_for_base(
                conn, qnorm, year_from, year_to, exclude_ids)
            related = _family_related(
                conn, qnorm, year_from, year_to, exclude_ids)
            # dedup: 한 charge_id 가 base 불일치 행으로 양쪽에 잡히는 것 방지(직접 우선)
            direct_ids = {c["charge_id"] for c in direct}
            related = [c for c in related if c["charge_id"] not in direct_ids]
            if direct or related:
                return _format_response_md(
                    {"status": "candidates", "query": qnorm,
                     "candidates": direct, "related": related})

        # ③ taxonomy 해석: base/charge_norm/canonical
        kind, payload = _resolve_one(conn, qnorm)
        if kind == "candidates":
            cands = _candidates_for_base(
                conn, payload, year_from, year_to, exclude_ids)
            return _format_response_md(
                {"status": "candidates", "query": qnorm, "candidates": cands})
        if kind == "pool":
            return _format_response_md(_blocks_from_pools(
                conn, [payload], year_from=year_from, year_to=year_to,
                exclude_ids=exclude_ids, reference_year=reference_year))

        # ④ category 명 → 그 계열 대표 pool 후보(표본 상위)
        if _is_category(conn, qnorm):
            cats = _category_candidates(
                conn, qnorm, year_from, year_to, exclude_ids)
            if cats:
                return _format_response_md(
                    {"status": "candidates", "query": qnorm, "candidates": cats,
                     "note": f"'{qnorm}' 계열 대표 죄명(표본 상위) — 좁혀 재검색 가능"})

        # ⑤ 미매치 → suggestions
        return _format_response_md(
            {"status": "no_data",
             "unmatched_charges": [
                 {"input": qnorm,
                  "suggested": _suggest_canonical_charges(conn, qnorm, 5)}],
             "warnings": ["정확 매치 0건 — unmatched_charges.suggested 로 재호출"]})
    finally:
        conn.close()


# ---------- charge match validation ----------

def _subtype_siblings(
    conn: sqlite3.Connection, charge: str, limit: int = 4
) -> list[str]:
    """bare 법률명 하위의 괄호 subtype 죄명 top N(빈도 ↓순) — 좁혀 재호출 힌트용.

    특별법은 괄호 안이 실체 죄명이라(폭력행위등처벌에관한법률위반(공동재물손괴등) ≠ (공동상해))
    법률명만 매칭되면 subtype 별 양형이 뒤섞인 것처럼 오해될 수 있어, 존재하는 subtype 을
    보여 특정 죄명으로 재호출하도록 유도한다. subtype 이 3종 미만이면(진짜 단일 죄명) 빈 list.
    """
    rows = conn.execute(
        "SELECT charge_norm, COUNT(*) cnt FROM prec_defendant_charges "
        "WHERE charge_norm LIKE ? || '(%' "
        "GROUP BY charge_norm ORDER BY cnt DESC LIMIT ?",
        [charge, limit],
    ).fetchall()
    total = conn.execute(
        "SELECT COUNT(DISTINCT charge_norm) c FROM prec_defendant_charges "
        "WHERE charge_norm LIKE ? || '(%'",
        [charge],
    ).fetchone()["c"]
    return [r["charge_norm"] for r in rows] if total >= 3 else []


def _suggest_canonical_charges(
    conn: sqlite3.Connection, charge: str, limit: int = 5
) -> list[str]:
    """비슷한 정형 charge_norm top N (빈도 ↓순) — 후보로 LLM self-correct 유도.

    2단계: ① **접두 매치 우선** — 특별법 죄명이 괄호 subtype 을 갖도록 정규화(2026-07-09)
    되면서, 모델이 subtype 없는 bare 법률명(예: '폭력행위등처벌에관한법률위반')을 넘기면
    정확매치가 실패한다. 이때 그 법률명으로 시작하는 실제 subtype(공동재물손괴등/공동상해/…)
    을 우선 반환해 좁혀 재호출하게 한다. ② 접두 후보가 limit 미달이면 3-gram substring 매치로
    보충(의역 표기 '청소년유해약물등판매' → '청소년보호법위반' 류 교정).
    """
    if not charge or len(charge) < 3:
        return []

    out: list[str] = []
    seen: set[str] = set()

    # ① 접두 매치 — 법률명 → 괄호 subtype 나열 (빈도 ↓순).
    #   input 전체로 먼저(bare 법률명 케이스), 이어서 괄호 앞 법률명만으로(틀린 subtype
    #   케이스: '폭처법(공동재물손괴)' → '등' 누락 등 → 법률명 subtype 형제 제시).
    prefixes = [charge]
    if "(" in charge:
        base = charge.split("(", 1)[0]
        if len(base) >= 3 and base != charge:
            prefixes.append(base)
    for pref in prefixes:
        if len(out) >= limit:
            break
        prefix_rows = conn.execute(
            "SELECT charge_norm, COUNT(*) cnt FROM prec_defendant_charges "
            "WHERE charge_norm LIKE ? || '%' AND charge_norm != ? "
            "GROUP BY charge_norm ORDER BY cnt DESC LIMIT ?",
            [pref, charge, limit],
        ).fetchall()
        for r in prefix_rows:
            if r["charge_norm"] not in seen:
                seen.add(r["charge_norm"])
                out.append(r["charge_norm"])
    if len(out) >= limit:
        return out[:limit]

    # ② 3-gram substring 보충
    grams = list({charge[i : i + 3] for i in range(len(charge) - 2)})
    placeholders = " OR ".join(["charge_norm LIKE ?"] * len(grams))
    params = [f"%{g}%" for g in grams]
    rows = conn.execute(
        f"SELECT charge_norm, COUNT(*) cnt FROM prec_defendant_charges "
        f"WHERE ({placeholders}) AND charge_norm != ? "
        f"GROUP BY charge_norm ORDER BY cnt DESC LIMIT ?",
        [*params, charge, limit],
    ).fetchall()
    for r in rows:
        if r["charge_norm"] not in seen:
            seen.add(r["charge_norm"])
            out.append(r["charge_norm"])
    return out[:limit]


def _check_charges_match(
    conn: sqlite3.Connection, normalized: list[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    """각 정규화 charge에 대해 DB 정확 매치 확인 → (matched, unmatched_with_suggested)."""
    matched: list[str] = []
    unmatched: list[dict[str, Any]] = []
    for c in normalized:
        cnt = conn.execute(
            "SELECT 1 FROM prec_defendant_charges WHERE charge_norm = ? LIMIT 1",
            (c,),
        ).fetchone()
        if cnt:
            matched.append(c)
        else:
            unmatched.append(
                {"input": c, "suggested": _suggest_canonical_charges(conn, c, limit=5)}
            )
    return matched, unmatched


# ---------- DB query helpers ----------

def _fetch_samples(
    conn: sqlite3.Connection,
    charge_norms: str | list[str],
    year_from: int | None,
    year_to: int | None,
    exclude_case_ids: frozenset[int] | None = None,
) -> list[sqlite3.Row]:
    """피고인 단위 **단독범** row 조회 — instance='1심' + n_charges=1 고정.

    charge_norms 는 단일 문자열 또는 **charge_norm 리스트**(taxonomy pool — 오타·구법
    변형 병합). 통계와 low-n 그리드 모두 이 helper만 사용하며 경합 사례 보충 경로는 없다.
    DISTINCT 로 pool 변형 중복 피고인을 제거한다.
    """
    norms = [charge_norms] if isinstance(charge_norms, str) else list(charge_norms)
    if not norms:
        return []
    marks = ",".join("?" * len(norms))
    sql = f"""
        SELECT DISTINCT
            pd.case_id, pd.defendant_id, pd.n_charges,
            pd.sentence_type, pd.sentence_months, pd.fine_amount,
            pd.probation, pd.probation_months, pd.sentencing_reason,
            pc.case_number, pc.court_name, pc.decision_year
        FROM prec_defendant_charges pdc
        JOIN prec_defendants pd
          ON pd.case_id = pdc.case_id AND pd.defendant_id = pdc.defendant_id
        JOIN prec_sentences ps ON ps.case_id = pd.case_id
        JOIN prec_cases pc ON pc.id = pd.case_id
        WHERE pdc.charge_norm IN ({marks})
          AND ps.instance = ?
          AND pd.n_charges = 1
          AND pd.sentence_type IN ('imprisonment','fine','not_guilty','life')
    """
    params: list[Any] = [*norms, STATS_INSTANCE]

    if year_from is not None:
        sql += " AND pc.decision_year >= ?"
        params.append(year_from)
    if year_to is not None:
        sql += " AND pc.decision_year <= ?"
        params.append(year_to)
    if exclude_case_ids:
        marks = ",".join("?" * len(exclude_case_ids))
        sql += f" AND pd.case_id NOT IN ({marks})"
        params.extend(exclude_case_ids)

    return conn.execute(sql, params).fetchall()


# ---------- charge_taxonomy (pool 식별자 + query 해석) ----------

def _tax_available(conn: sqlite3.Connection) -> bool:
    """charge_taxonomy 테이블 존재 여부 — 부재 시 레거시 exact-match 경로."""
    try:
        conn.execute("SELECT 1 FROM charge_taxonomy LIMIT 1")
        return True
    except sqlite3.OperationalError:
        return False


def _pool_label(canonical: str | None, modality: str | None, role: str | None) -> str:
    lbl = canonical or "?"
    extra = [x for x in ((modality if modality and modality != "기수" else None),
                         (role if role and role != "정범" else None)) if x]
    return lbl + ("·" + "·".join(extra) if extra else "")


def _pool_norms(conn: sqlite3.Connection, charge_id: int) -> dict[str, Any] | None:
    """charge_id → {charge_id, label, category, norms[]} (오타·변형 병합된 charge_norm 들)."""
    rows = conn.execute(
        "SELECT charge_norm, canonical_id, modality, role, category "
        "FROM charge_taxonomy WHERE charge_id=? AND is_noise=0", (charge_id,)).fetchall()
    if not rows:
        return None
    return {"charge_id": charge_id,
            "label": _pool_label(rows[0]["canonical_id"], rows[0]["modality"], rows[0]["role"]),
            "category": rows[0]["category"],
            "norms": [r["charge_norm"] for r in rows]}


def _candidates_for_base(
    conn: sqlite3.Connection, base: str,
    year_from: int | None, year_to: int | None, exclude: frozenset[int],
) -> list[dict[str, Any]]:
    """base_crime 패밀리의 pool 후보 [{charge_id,label,category,n}] — 단독범 n 내림차순."""
    ids = [r["charge_id"] for r in conn.execute(
        "SELECT DISTINCT charge_id FROM charge_taxonomy "
        "WHERE base_crime=? AND is_noise=0 AND charge_id IS NOT NULL", (base,))]
    out = []
    for cid in ids:
        p = _pool_norms(conn, cid)
        if not p:
            continue
        n = len(_fetch_samples(conn, p["norms"], year_from, year_to, exclude))
        if n == 0:
            continue  # 표본 0건 pool 은 후보에서 제외 — 클릭해도 통계 없음(노이즈)
        out.append({"charge_id": cid, "label": p["label"], "category": p["category"], "n": n})
    out.sort(key=lambda x: -x["n"])
    return out


def _resolve_one(conn: sqlite3.Connection, qnorm: str) -> tuple[str, Any]:
    """텍스트 죄명 → ('candidates', base_crime) | ('pool', charge_id) | ('none', None).

    base_crime 이 2+ pool 을 가지면(사기·강도 등 패밀리) 후보 제시, 특정 죄명(charge_norm/
    canonical 정확매치)이면 그 pool 로 직행.
    """
    base_ids = [r["charge_id"] for r in conn.execute(
        "SELECT DISTINCT charge_id FROM charge_taxonomy "
        "WHERE base_crime=? AND is_noise=0 AND charge_id IS NOT NULL", (qnorm,))]
    if len(base_ids) >= 2:
        return ("candidates", qnorm)
    row = conn.execute(
        "SELECT charge_id FROM charge_taxonomy "
        "WHERE charge_norm=? AND is_noise=0 AND charge_id IS NOT NULL", (qnorm,)).fetchone()
    if row:
        return ("pool", row["charge_id"])
    canon_rows = conn.execute(
        "SELECT DISTINCT charge_id, base_crime FROM charge_taxonomy "
        "WHERE canonical_id=? AND is_noise=0 AND charge_id IS NOT NULL", (qnorm,)).fetchall()
    if len(canon_rows) == 1:
        return ("pool", canon_rows[0]["charge_id"])
    if len(canon_rows) > 1:
        return ("candidates", canon_rows[0]["base_crime"])
    if len(base_ids) == 1:
        return ("pool", base_ids[0])
    return ("none", None)


# ---------- 상위계열(family) — 발견성 레이어 (§6-3) ----------
# "사기" → 직접(base=사기) + 관련(컴퓨터등사용사기·보험사기…). canonical_id 로 정의된
# charge_family 테이블 참조(build_charge_families.py). 통계 pool 은 분리 유지, 발견성만 보강.

def _is_family(conn: sqlite3.Connection, qnorm: str) -> bool:
    """qnorm 이 상위계열 키인지 — charge_family 부재 시 graceful False(family 미적재 환경)."""
    try:
        return conn.execute(
            "SELECT 1 FROM charge_family WHERE family=? LIMIT 1", (qnorm,)
        ).fetchone() is not None
    except sqlite3.OperationalError:
        return False


def _family_related(
    conn: sqlite3.Connection, family: str,
    year_from: int | None, year_to: int | None, exclude: frozenset[int],
) -> list[dict[str, Any]]:
    """family 소속 pool 중 base_crime≠family 인 것(관련 죄명) [{charge_id,label,category,n}].

    base==family 는 직접(_candidates_for_base)이 담당 → 여기선 제외(중복 방지). n=0 제외.
    """
    canons = [r["canonical_id"] for r in conn.execute(
        "SELECT canonical_id FROM charge_family WHERE family=?", (family,))]
    if not canons:
        return []
    marks = ",".join("?" * len(canons))
    ids = [r["charge_id"] for r in conn.execute(
        f"SELECT DISTINCT charge_id FROM charge_taxonomy "
        f"WHERE canonical_id IN ({marks}) AND base_crime != ? "
        f"AND is_noise=0 AND charge_id IS NOT NULL", [*canons, family])]
    out = []
    for cid in ids:
        p = _pool_norms(conn, cid)
        if not p:
            continue
        n = len(_fetch_samples(conn, p["norms"], year_from, year_to, exclude))
        if n == 0:
            continue
        out.append({"charge_id": cid, "label": p["label"], "category": p["category"], "n": n})
    out.sort(key=lambda x: -x["n"])
    return out


# ---------- alias / category / bare 법률명 진입점 (§6-1·6-2) ----------

def _pool_counts(
    conn: sqlite3.Connection, charge_ids: list[int],
    yf: int | None, yt: int | None, exclude: frozenset[int],
) -> dict[int, int]:
    """charge_id 리스트 → {charge_id: 단독범 1심 n} 일괄(1쿼리) — 대량 pool 집합(category)용."""
    ids = list(dict.fromkeys(charge_ids))
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    sql = (
        "SELECT t.charge_id, COUNT(DISTINCT pd.case_id||'-'||pd.defendant_id) n "
        "FROM charge_taxonomy t "
        "JOIN prec_defendant_charges pdc ON pdc.charge_norm=t.charge_norm "
        "JOIN prec_defendants pd ON pd.case_id=pdc.case_id "
        "  AND pd.defendant_id=pdc.defendant_id AND pd.n_charges=1 "
        "JOIN prec_sentences ps ON ps.case_id=pd.case_id AND ps.instance=? "
        "JOIN prec_cases pc ON pc.id=pd.case_id "
        f"WHERE t.charge_id IN ({marks}) AND t.is_noise=0")
    params: list[Any] = [STATS_INSTANCE, *ids]
    if yf is not None:
        sql += " AND pc.decision_year>=?"
        params.append(yf)
    if yt is not None:
        sql += " AND pc.decision_year<=?"
        params.append(yt)
    if exclude:
        sql += f" AND pd.case_id NOT IN ({','.join('?' * len(exclude))})"
        params.extend(exclude)
    sql += " GROUP BY t.charge_id"
    return {r["charge_id"]: r["n"] for r in conn.execute(sql, params)}


def _pool_meta(conn: sqlite3.Connection, charge_ids: list[int]) -> dict[int, tuple[str, str | None]]:
    """charge_id → (label, category) 일괄(1쿼리)."""
    ids = list(dict.fromkeys(charge_ids))
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    out: dict[int, tuple[str, str | None]] = {}
    for r in conn.execute(
        f"SELECT charge_id, canonical_id, modality, role, category "
        f"FROM charge_taxonomy WHERE charge_id IN ({marks}) AND is_noise=0 "
        f"GROUP BY charge_id", ids):
        out[r["charge_id"]] = (
            _pool_label(r["canonical_id"], r["modality"], r["role"]), r["category"])
    return out


def _cands_batch(
    conn: sqlite3.Connection, ids: list[int],
    yf: int | None, yt: int | None, exclude: frozenset[int], *, cap: int,
) -> list[dict[str, Any]]:
    """charge_id 리스트 → 후보(n>0·n내림차순·cap). 일괄 카운트(category/bare 법률명 대량용)."""
    counts = _pool_counts(conn, ids, yf, yt, exclude)
    meta = _pool_meta(conn, ids)
    out = []
    for cid in ids:
        n = counts.get(cid, 0)
        if n == 0:
            continue
        label, cat = meta.get(cid, ("?", None))
        out.append({"charge_id": cid, "label": label, "category": cat, "n": n})
    out.sort(key=lambda x: -x["n"])
    return out[:cap]


def _is_category(conn: sqlite3.Connection, q: str) -> bool:
    """q 가 category 명인지(잡동 '기타' 제외, 최소 pool 이상). 부재 graceful False."""
    if q == "기타":
        return False
    try:
        r = conn.execute(
            "SELECT COUNT(DISTINCT charge_id) c FROM charge_taxonomy "
            "WHERE category=? AND is_noise=0 AND charge_id IS NOT NULL", (q,)).fetchone()
    except sqlite3.OperationalError:
        return False
    return bool(r) and r["c"] >= CATEGORY_MIN_POOLS


def _category_candidates(
    conn: sqlite3.Connection, q: str,
    yf: int | None, yt: int | None, exclude: frozenset[int],
) -> list[dict[str, Any]]:
    """category 명 → 그 계열 대표 pool 후보(표본 상위 RELATED_MAX)."""
    ids = [r["charge_id"] for r in conn.execute(
        "SELECT DISTINCT charge_id FROM charge_taxonomy "
        "WHERE category=? AND is_noise=0 AND charge_id IS NOT NULL", (q,))]
    return _cands_batch(conn, ids, yf, yt, exclude, cap=RELATED_MAX)


def _lawname_candidates(
    conn: sqlite3.Connection, q: str,
    yf: int | None, yt: int | None, exclude: frozenset[int],
) -> list[dict[str, Any]]:
    """bare 법률명(괄호 없는 '…법위반') → 그 법률의 괄호 subtype 후보(완전 나열, cap).

    성폭력특례법 등은 subtype 마다 base 가 흔들려 base 매치로는 일부만 잡힘 → canonical
    prefix 로 전체 subtype 을 후보로. 단일 죄명 법률명(subtype<2)은 미발동(정상 경로 위임).
    """
    if "(" in q or not (q.endswith("법위반") or q.endswith("법률위반")):
        return []
    ids = [r["charge_id"] for r in conn.execute(
        "SELECT DISTINCT charge_id FROM charge_taxonomy "
        "WHERE canonical_id LIKE ? || '%' AND is_noise=0 AND charge_id IS NOT NULL", (q,))]
    if len(ids) < 2:
        return []
    return _cands_batch(conn, ids, yf, yt, exclude, cap=RELATED_MAX)


# ---------- 블록/응답 빌더 (charge_id·텍스트·레거시 공용) ----------

def _build_block(
    conn: sqlite3.Connection, label: str, norms: list[str], *,
    year_from: int | None, year_to: int | None, exclude_ids: frozenset[int],
    multi: bool, reference_year: int | None,
    warnings: list[str],
) -> tuple[dict[str, Any], bool, bool]:
    """pool(norms) → 통계/그리드 블록. (blk, is_stats, is_grid) 반환."""
    grid_cap = GRID_MULTI_MAX if multi else GRID_MAX
    singles = _fetch_samples(conn, norms, year_from, year_to, exclude_ids)
    blk: dict[str, Any] = {"charge": label, "n": len(singles)}
    is_stats = is_grid = False
    if len(singles) >= MIN_N_STATS:
        is_stats = True
        bt = _by_type(singles)
        blk["by_type"] = bt
        blk["imprisonment"] = _imprisonment_stats(
            [r for r in singles if r["sentence_type"] == "imprisonment"]) or None
        blk["fine"] = _fine_stats(
            [r for r in singles if r["sentence_type"] == "fine"]) or None
        ng = bt.get("not_guilty", 0)
        if ng / len(singles) > 0.3:
            warnings.append(
                f"{label}: 무죄 비율 {ng}/{len(singles)} "
                f"({100 * ng / len(singles):.0f}%) — 죄목 적용 자체에 다툼 많음")
        if not multi:
            comparable = _stratified_sample(singles, reference_year=reference_year)
            blk["comparables"] = [(q, r, {}) for q, r in comparable]
    else:
        is_grid = True
        blk["by_type"] = _by_type(singles) if singles else None
        grid_rows: list[tuple[sqlite3.Row, bool]] = [
            (r, False) for r in _grid_sort(singles, reference_year)[:grid_cap]]
        if grid_rows:
            cbd: dict[tuple[int, str], list[str]] = {}
            blk["grid"] = [(r, cbd, conc) for r, conc in grid_rows]
            blk["notes"] = [
                f"단일 죄명 표본 {len(singles)}건 < {MIN_N_STATS} — 통계 생략, 단독 사례만 나열"
            ]
        else:
            blk["notes"] = ["단일 죄명 표본 0건 — 경합 사례로 보충하지 않음"]
    return blk, is_stats, is_grid


def _finish_dict(
    blocks: list[dict[str, Any]], multi: bool,
    any_stats: bool, any_grid: bool, warnings: list[str],
) -> dict[str, Any]:
    if not any(b["n"] or b.get("grid") for b in blocks):
        return {"status": "no_data", "charge_blocks": blocks,
                "warnings": (warnings or []) + ["매칭 case 없음"]}
    resp: dict[str, Any] = {"status": "ok" if any_stats else "low_n_grid",
                            "charge_blocks": blocks}
    if multi:
        resp["mode"] = "multi — 죄명별 단일 죄명 관측분포 (서로 결합 금지)"
        resp["guidance"] = [_GUIDANCE_CONCURRENT, _GUIDANCE_RECALL]
    elif any_grid:
        resp["mode"] = "low_n — 통계 대신 개별 사례 그리드"
    if warnings:
        resp["warnings"] = warnings
    return resp


def _blocks_from_pools(
    conn: sqlite3.Connection, charge_ids: list[int], *,
    year_from: int | None, year_to: int | None, exclude_ids: frozenset[int],
    reference_year: int | None,
) -> dict[str, Any]:
    """charge_id 리스트 → pool 별 통계 블록 → 응답 dict."""
    pools = [p for p in (_pool_norms(conn, i) for i in dict.fromkeys(charge_ids)) if p]
    if not pools:
        return {"status": "no_data", "warnings": ["해당 charge_id pool 없음"]}
    multi = len(pools) > 1
    blocks: list[dict[str, Any]] = []
    warnings: list[str] = []
    any_stats = any_grid = False
    for p in pools:
        blk, st, gr = _build_block(
            conn, p["label"], p["norms"], year_from=year_from, year_to=year_to,
            exclude_ids=exclude_ids, multi=multi, reference_year=reference_year,
            warnings=warnings)
        any_stats |= st
        any_grid |= gr
        blocks.append(blk)
    return _finish_dict(blocks, multi, any_stats, any_grid, warnings)


def _legacy_stats(
    conn: sqlite3.Connection, normalized: list[str], norm_changed: bool, *,
    year_from: int | None, year_to: int | None, exclude_ids: frozenset[int],
    reference_year: int | None,
) -> dict[str, Any]:
    """taxonomy 부재 시 — 기존 exact charge_norm 매치 경로(+subtype 힌트)."""
    matched, unmatched = _check_charges_match(conn, normalized)
    if not matched:
        return {"status": "no_data", "unmatched_charges": unmatched,
                "warnings": ["입력 charges 정확 매치 0건. suggested 표기로 재호출."]}
    multi = len(matched) > 1
    blocks: list[dict[str, Any]] = []
    warnings: list[str] = []
    any_stats = any_grid = False
    for ch in matched:
        blk, st, gr = _build_block(
            conn, ch, [ch], year_from=year_from, year_to=year_to,
            exclude_ids=exclude_ids, multi=multi, reference_year=reference_year,
            warnings=warnings)
        if "(" not in ch:
            hint = _subtype_siblings(conn, ch)
            if hint:
                blk.setdefault("notes", []).append(
                    f"'{ch}' 는 법률명만으로 매칭됐습니다 — 이 특별법은 괄호 subtype 별 "
                    f"양형이 크게 다릅니다(예: {', '.join(hint)}). 특정 죄명이 대상이면 "
                    "그 subtype 으로 재호출하세요.")
        any_stats |= st
        any_grid |= gr
        blocks.append(blk)
    resp = _finish_dict(blocks, multi, any_stats, any_grid, warnings)
    if norm_changed:
        resp["charges_normalized"] = matched
    if unmatched:
        resp["unmatched_charges"] = unmatched
    return resp
