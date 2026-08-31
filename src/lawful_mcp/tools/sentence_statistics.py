"""Observed first-instance sentencing distribution for a single charge.

What a court actually imposed, from the corpus, for defendants convicted of
one charge and nothing else. The sampling rules exist to keep that sentence
attributable:

- **First instance only.** Appellate decisions skew lenient and only reach
  the corpus selectively, which would bias the distribution systematically.
- **One charge type per defendant.** Several counts of the same charge stay
  in; a defendant also convicted of a different charge is excluded, because
  the single sentence imposed covers all of them and cannot be attributed to
  one. This is also why distributions for different charges must not be
  added, averaged or scaled into a combined-offence distribution: the
  Criminal Act's aggravation rule is a ceiling on the processed range, not a
  formula for combining observed sentences.
- **Below MIN_N_STATS, no statistics are computed at all.** A mean over a
  handful of cases reads as an anchor whether or not it deserves to, so a
  small sample returns the individual cases instead and lets the reader see
  what it is made of.

The unit of observation is the defendant, not the case: where codefendants
in one judgment received different sentences, each is a sample.

``HarnessDeps`` may optionally carry an ``exclude_case_ids`` frozenset, which
is subtracted from every sample query — useful when the corpus overlaps an
evaluation set. Absent, nothing is excluded.
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

# ---------- constants ----------

STATS_INSTANCE = "1심"

# Below this many samples the tool reports individual cases instead of
# computing a distribution.
MIN_N_STATS = 30

GRID_MAX = 15            # case rows to list for a single charge
GRID_MULTI_MAX = 5       # case rows per charge when several are listed
RELATED_MAX = 15         # related charges to surface, largest sample first
CATEGORY_MIN_POOLS = 5   # a category with fewer pools is not offered as an entry point

# Everyday words for offences, mapped to what the corpus calls them. People
# ask about "voice phishing" or "hidden camera"; judgments say fraud and
# unlawful filming. Values re-enter the resolver chain, so a mapping may
# point at a category, a base crime or a charge name.
_ALIAS = {
    "마약": "마약범죄", "마약류": "마약범죄", "마약사범": "마약범죄", "필로폰": "마약범죄",
    "히로뽕": "마약범죄", "대마": "마약범죄", "대마초": "마약범죄", "뽕": "마약범죄",
    "몰카": "카메라등이용촬영", "도촬": "카메라등이용촬영", "불법촬영": "카메라등이용촬영",
    "보이스피싱": "사기", "전화금융사기": "사기", "전세사기": "사기", "취업사기": "사기",
    "중고사기": "사기", "대포통장": "전자금융거래법위반", "성폭력": "성범죄",
    "음주운전": "도로교통법위반(음주운전)", "음주": "도로교통법위반(음주운전)",
    "뺑소니": "도주", "성매수": "성매매", "성구매": "성매매",
    # Each mapping below was checked to resolve to something in the corpus.
    "무면허운전": "도로교통법위반(무면허운전)", "음주측정거부": "도로교통법위반(음주측정거부)",
    "통매음": "통신매체이용음란",
    # Special-act offences resolve under their statute name, so the bare
    # everyday word has to point at the full canonical charge or it will not
    # reach a sample pool.
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

# ---------- normalisation ----------
# Shared with the indexer that builds the sentencing tables, so a charge is
# keyed the same way at write time and at read time.


# ---------- statistics ----------

def _percentile(sorted_vals: list[int], p: float) -> int | None:
    """Percentile of an already-sorted list, R-7 linear interpolation.

    The naive `int(n*p)` index rounds up on small samples — at n=10, the 90th
    percentile lands on the maximum — which inflates the whole distribution.
    Read as a sentencing range, that bias points one way: too severe.
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
    """Custodial statistics: mean, spread and suspension rate.

    Percentiles are left out because the severity-ordered deciles already
    carry the shape of the distribution.
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
    """Fine statistics: the mean only, since the deciles carry the shape."""
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


# ---------- comparable cases, spread across the distribution ----------

def _severity_key(row: sqlite3.Row) -> tuple[int, int]:
    """Severity order: life > custodial > fine > acquittal, then by amount."""
    st = row["sentence_type"]
    if st == "life":
        return (3, 999_999)  # 무기징역/사형 — 가장 무거움
    if st == "imprisonment":
        return (2, row["sentence_months"] or 0)
    if st == "fine":
        # Order fines alongside custodial sentences on one severity scale.
        # The exchange rate is rough by nature: it exists to sort a mixed
        # list, not to claim a fine equals so many months.
        return (1, (row["fine_amount"] or 0) // 1_000_000)
    return (0, 0)  # not_guilty


def _stratified_sample(
    rows: list[sqlite3.Row],
    *,
    reference_year: int | None = None,
) -> list[tuple[int, sqlite3.Row]]:
    """Pick one case at each decile of the severity-sorted samples.

    Eleven cutpoints (0, 10, ... 100), each taking the row at
    ``round((q/100) * (n-1))``. Where several cases tie on severity, the one
    nearest ``reference_year`` wins, so the examples sit near the period the
    caller asked about.

    With eleven rows or fewer, every row is returned and its percentile is
    inferred from position. A case landing on two cutpoints is shown once,
    at the lower one.
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
    """Order cases for listing: nearest the reference year, else most recent."""
    if reference_year is not None:
        return sorted(
            rows,
            key=lambda r: (
                abs((r["decision_year"] or 0) - reference_year),
                -(r["decision_year"] or 0),
            ),
        )
    return sorted(rows, key=lambda r: -(r["decision_year"] or 0))


# ---------- row serialisation ----------

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
    """Abbreviate a consolidated case number to the first plus a count.

    Consolidated matters list every joined number and run to about a hundred
    characters; the full list is one click away on the linked page.
    """
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
    """Render one case as indented key: value lines.

    Compact on purpose: the id is already the tail of the url, and the charge
    line appears only for multi-charge rows — for single-charge rows it would
    repeat the charge that was queried.
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


# ---------- markdown serialisation ----------

def _format_response_md(resp: dict[str, Any]) -> str:
    """Response dict -> markdown-KV string.

    `## section` headers with `- key: value` lines under them. Measured about
    25% cheaper in tokens than the equivalent JSON, which is why the tools
    answer in this shape rather than a serialised object.
    """
    lines: list[str] = [f"## status: {resp.get('status', 'ok')}"]
    if resp.get("mode"):
        lines.append(f"## mode: {resp['mode']}")

    # Candidate list: not statistics, but the charges the query could mean.
    # The caller picks one and calls back with its id.
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


# ---------- the tool ----------

def _coerce_single_str(x: Any) -> str | None:
    """Take a scalar. A list yields its first element: charges are queried
    one at a time, never combined."""
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
    """Take a scalar integer; a list yields its first element."""
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

        # By id: the distribution for exactly one pool. Even for a
        # multi-charge matter this returns single-charge distributions only;
        # it never picks charges for the caller or combines distributions.
        if cid is not None:
            return _format_response_md(_blocks_from_pools(
                conn, [cid], year_from=year_from, year_to=year_to,
                exclude_ids=exclude_ids, reference_year=reference_year))

        # By text: resolve what the words refer to.
        assert q_raw and qnorm

        # Strip the trailing 죄 ("the offence of"), which people add and
        # judgments do not: 사기죄 -> 사기. No charge name in the corpus ends
        # that way, so nothing legitimate is damaged. Category names do end
        # in 범죄, hence the negative lookbehind — without it 성범죄 (sexual
        # offences) would be mangled into a word that matches nothing.
        qnorm = re.sub(r"(?<!범)죄$", "", qnorm) or qnorm

        # No taxonomy in this corpus: fall back to exact matching.
        if not tax:
            return _format_response_md(_legacy_stats(
                conn, [qnorm], qnorm != q_raw, year_from=year_from,
                year_to=year_to, exclude_ids=exclude_ids,
                reference_year=reference_year))

        # 0. Rewrite everyday words into corpus vocabulary.
        qnorm = _ALIAS.get(qnorm, qnorm)

        # 1. A statute name with no qualifier ("violation of the Road
        #    Traffic Act") lists every offence under that statute.
        law_c = _lawname_candidates(conn, qnorm, year_from, year_to, exclude_ids)
        if law_c:
            return _format_response_md(
                {"status": "candidates", "query": qnorm, "candidates": law_c,
                 "note": f"'{qnorm}' 법률의 죄명 후보(괄호 subtype)"})

        # 2. A family name returns direct matches plus its relatives:
        #    "fraud" also surfaces computer fraud, insurance fraud, and so on.
        if _is_family(conn, qnorm):
            direct = _candidates_for_base(
                conn, qnorm, year_from, year_to, exclude_ids)
            related = _family_related(
                conn, qnorm, year_from, year_to, exclude_ids)
            # One charge can match both lists; keep the direct hit.
            direct_ids = {c["charge_id"] for c in direct}
            related = [c for c in related if c["charge_id"] not in direct_ids]
            if direct or related:
                return _format_response_md(
                    {"status": "candidates", "query": qnorm,
                     "candidates": direct, "related": related})

        # 3. Resolve through the taxonomy: base crime, charge name, canonical.
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

        # 4. A category name returns its best-sampled pools.
        if _is_category(conn, qnorm):
            cats = _category_candidates(
                conn, qnorm, year_from, year_to, exclude_ids)
            if cats:
                return _format_response_md(
                    {"status": "candidates", "query": qnorm, "candidates": cats,
                     "note": f"'{qnorm}' 계열 대표 죄명(표본 상위) — 좁혀 재검색 가능"})

        # 5. Nothing matched: suggest near misses rather than return empty.
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
    """Offences under a bare statute name, most frequent first.

    In a special act the parenthesised qualifier is the offence, and they
    sentence differently. Matching on the statute name alone would present
    them as one population, so the caller is shown the qualifiers that exist
    and asked to pick. Returns nothing when the statute really does define
    only a couple of offences.
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
    """Charge names close to the query, most frequent first.

    Two passes. Prefix matches come first: a caller who passes a statute name
    without its qualifier gets the real offences under it rather than an
    empty result. Trigram substring matches then top the list up, which
    catches a paraphrase of an offence rather than its formal name.
    """
    if not charge or len(charge) < 3:
        return []

    out: list[str] = []
    seen: set[str] = set()

    # 1. Prefix match, most frequent first. Try the whole input, then just
    #    the statute name before the parenthesis — a caller who gets the
    #    qualifier slightly wrong still gets the sibling offences under that
    #    statute rather than nothing.
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

    # 2. Fill out with trigram substring matches.
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
    """Split charges into those the corpus knows and those it can only suggest for."""
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
    """Single-charge, first-instance defendant rows — the sample, defined once.

    ``charge_norms`` may be one name or a taxonomy pool: the spelling
    variants and superseded forms that mean the same offence. Both the
    statistics path and the small-sample listing come through here, so there
    is no route by which a multi-charge defendant enters either.
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


# ---------- charge taxonomy: pool identity and query resolution ----------

def _tax_available(conn: sqlite3.Connection) -> bool:
    """Does this corpus carry the taxonomy? Without it, exact matching only."""
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
    """Pool id -> its label, category, and every charge name that maps to it."""
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
    """Pools under one base crime, largest sample first."""
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
    """Resolve a charge name to a pool, to a set of candidates, or to nothing.

    A base crime covering more than one pool returns candidates for the
    caller to choose between; an exact charge name goes straight to its pool.
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


# ---------- families: a discovery layer ----------
# Asking about "fraud" should surface computer fraud and insurance fraud too.
# Families only widen what the caller is shown; the sample pools stay
# separate, so nothing here merges distributions.

def _is_family(conn: sqlite3.Connection, qnorm: str) -> bool:
    """Is this a family name? False when the corpus has no family table."""
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
    """Relatives of a family: its pools other than the direct matches.

    The direct ones are listed separately, so excluding them here keeps a
    charge from appearing twice. Pools with no samples are dropped.
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


# ---------- entry points: alias, category, bare statute name ----------

def _pool_counts(
    conn: sqlite3.Connection, charge_ids: list[int],
    yf: int | None, yt: int | None, exclude: frozenset[int],
) -> dict[int, int]:
    """Sample count per pool, in one query — a category can hold many pools."""
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
    """Label and category per pool, in one query."""
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
    """Pools with at least one sample, largest first, capped."""
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
    """Is this a category name? The catch-all category is not an entry point,
    and a category too small to be useful is not offered either."""
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
    """Category -> its best-sampled pools."""
    ids = [r["charge_id"] for r in conn.execute(
        "SELECT DISTINCT charge_id FROM charge_taxonomy "
        "WHERE category=? AND is_noise=0 AND charge_id IS NOT NULL", (q,))]
    return _cands_batch(conn, ids, yf, yt, exclude, cap=RELATED_MAX)


def _lawname_candidates(
    conn: sqlite3.Connection, q: str,
    yf: int | None, yt: int | None, exclude: frozenset[int],
) -> list[dict[str, Any]]:
    """A statute name with no qualifier -> every offence under that statute.

    Base-crime matching only finds some of them, because the base varies
    between qualifiers within one act; matching on the canonical prefix finds
    all. Statutes defining a single offence fall through to the normal path.
    """
    if "(" in q or not (q.endswith("법위반") or q.endswith("법률위반")):
        return []
    ids = [r["charge_id"] for r in conn.execute(
        "SELECT DISTINCT charge_id FROM charge_taxonomy "
        "WHERE canonical_id LIKE ? || '%' AND is_noise=0 AND charge_id IS NOT NULL", (q,))]
    if len(ids) < 2:
        return []
    return _cands_batch(conn, ids, yf, yt, exclude, cap=RELATED_MAX)


# ---------- response builders, shared by every resolution path ----------

def _build_block(
    conn: sqlite3.Connection, label: str, norms: list[str], *,
    year_from: int | None, year_to: int | None, exclude_ids: frozenset[int],
    multi: bool, reference_year: int | None,
    warnings: list[str],
) -> tuple[dict[str, Any], bool, bool]:
    """Build one pool's block: statistics if the sample allows, else cases."""
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
    """Assemble the response from one block per pool."""
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
    """Fallback for a corpus without the taxonomy: exact matching only."""
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
