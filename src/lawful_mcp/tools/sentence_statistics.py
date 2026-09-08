"""First-instance sentencing distribution for a single charge (no concurrent offences).

Offence keys are official charge names (Supreme Prosecutors' Office Directive No. 1516,
resolved via `_charge_index`). Judgment charge strings are deterministically mapped
to official names via `charge_norm_map`.
"""
from __future__ import annotations

import sqlite3
import statistics
from typing import Any

from pydantic_ai import RunContext

from .. import config
from .._charge import norm_charge as _norm_charge
from ._charge_index import ARTICLE_REF_RE, Resolution, numeric_charge_tokens, official_index, strip_charge_decorations
from ._coerce import coerce_list, coerce_str
from ..deps import HarnessDeps, open_db

# Constants
STATS_INSTANCE = "1심"
MIN_N_STATS = 30
GRID_MAX = 15
RELATED_MAX = 15


def _percentile(sorted_vals: list[int], p: float) -> int | None:
    """Linear interpolation percentile (R-7)."""
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


def _fine_share(rows: list[sqlite3.Row]) -> tuple[int, int] | None:
    """(Number of sentences, fine %). Denominator excludes not guilty."""
    dec = [r for r in rows if r["sentence_type"] in ("imprisonment", "fine", "life")]
    if not dec:
        return None
    return len(dec), round(sum(1 for r in dec if r["sentence_type"] == "fine") / len(dec) * 100)


def _fine_trend(pool: list[sqlite3.Row]) -> dict[str, Any] | None:
    """Fine ratio trend over time — overall, recent half, and recorded years.

    Choice of sentence kind (imprisonment vs fine) shifts over time, but filtering
    by year collapses sample sizes since median count per charge is 2.
    Instead of exposing year filter arguments, we report the trend over the full pool.
    """
    dated = sorted((r for r in pool if r["decision_year"]), key=lambda r: r["decision_year"])
    if not dated:
        return None
    out: dict[str, Any] = {"years": (dated[0]["decision_year"], dated[-1]["decision_year"])}
    whole = _fine_share(dated)
    if whole:
        out["all"] = whole
    half = dated[len(dated) // 2:]
    recent = _fine_share(half)
    if recent and len(dated) >= MIN_N_STATS:
        out["recent"] = (*recent, half[0]["decision_year"], half[-1]["decision_year"])
    return out


def _severity_key(row: sqlite3.Row) -> tuple[int, int]:
    st = row["sentence_type"]
    if st == "life":
        return (3, 999_999)
    if st == "imprisonment":
        return (2, row["sentence_months"] or 0)
    if st == "fine":
        return (1, (row["fine_amount"] or 0) // 1_000_000)
    return (0, 0)


def _stratified_sample(
    rows: list[sqlite3.Row],
    *,
    reference_year: int | None = None,
) -> list[tuple[int, sqlite3.Row]]:
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
    if reference_year is not None:
        return sorted(
            rows,
            key=lambda r: (
                abs((r["decision_year"] or 0) - reference_year),
                -(r["decision_year"] or 0),
            ),
        )
    return sorted(rows, key=lambda r: -(r["decision_year"] or 0))


REASON_MAX = 160


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
    parts = [p.strip() for p in (case_number or "").split(",") if p.strip()]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} 외 {len(parts) - 1}건 병합"


def _case_lines(row: sqlite3.Row) -> list[str]:
    lines = [
        f"  case: {_short_case_no(row['case_number'])} ({row['decision_year']}) · def {row['defendant_id']}",
        f"  url: {config.case_url_base()}/cases/{row['case_id']}",
        f"  sentence: {_sentence_str(row)}",
    ]
    reason = row["sentencing_reason"]
    if reason:
        if len(reason) > REASON_MAX:
            reason = reason[: REASON_MAX - 1] + "…"
        lines.append(f"  reason: {reason}")
    return lines


def _format_response_md(resp: dict[str, Any]) -> str:
    lines: list[str] = [f"## status: {resp.get('status', 'ok')}"]
    if resp.get("mode"):
        lines.append(f"## mode: {resp['mode']}")
    if resp.get("query_resolved"):
        lines.append(f"## note: 죄명 표기 해석 {resp['query_resolved']} — 재호출도 이 표기로")
    for note in resp.get("notes_top") or []:
        lines.append(f"## note: {note}")

    def name_line(c: dict[str, Any]) -> str:
        tail = ""
        if c.get("status") == "구법":
            tail = f" (구법 표기{' → 현행 ' + c['successor'] if c.get('successor') else ''})"
        return f"- n={c['n']} · {c['label']}{tail}"

    if resp.get("status") == "candidates":
        lines.append(f"## query: {resp.get('query', '')}")
        if resp.get("note"):
            lines.append(f"## {resp['note']}")
        cands = resp.get("candidates") or []
        lines.append(
            "## 후보 죄명 (charges=<이름> 그대로 재호출 → 통계). "
            "n=단일 죄명 1심 표본수(30↑ 통계, 미만 그리드)")
        for c in cands[:RELATED_MAX]:
            lines.append(name_line(c))
        if len(cands) > RELATED_MAX:
            lines.append(f"- (외 {len(cands) - RELATED_MAX}개 더 — 죄명을 좁혀 검색하세요)")
        for w in resp.get("warnings") or []:
            lines.append(f"- note: {w}")
        return "\n".join(lines)

    for blk in resp.get("charge_blocks") or []:
        lines.append(f"## charge: {blk['charge']}")
        lines.append(f"- n: {blk['n']} (단일 죄명)")
        tr = blk.get("fine_trend")
        if tr:
            if tr.get("all"):
                n_all, p_all = tr["all"]
                share = f"- 벌금 비율: 전체 {p_all}% (n={n_all})"
                if tr.get("recent"):
                    n_r, p_r, y0, y1 = tr["recent"]
                    share += f" · 최근 절반 {p_r}% ({y0}~{y1}, n={n_r})"
                lines.append(share)
            lines.append(f"- 수록 연도: {tr['years'][0]}~{tr['years'][1]}")
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
            for q, row in cc:
                lines.append("\n".join([f"- q: {q}", *_case_lines(row)]))

        grid = blk.get("grid") or []
        if grid:
            lines.append(f"## case_grid: {blk['charge']} (개별 사례 나열 — 통계 아님, 분포 일반화 금지)")
            for row in grid:
                body = _case_lines(row)
                lines.append("\n".join(["- " + body[0].lstrip(), *body[1:]]))

    related = resp.get("related") or []
    if related:
        lines.append(
            "## 관련 죄명 (같은 죄의 미수·상습·특수형, 같은 법률의 다른 죄, 이 이름을 품은 특별법 죄 — "
            "해당하면 charges 로 통계)")
        for c in related[:RELATED_MAX]:
            lines.append(name_line(c))
        if len(related) > RELATED_MAX:
            lines.append(f"- (외 {len(related) - RELATED_MAX}개 더)")

    um = resp.get("unmatched_charges") or []
    if um:
        lines.append("## unmatched_charges")
        for u in um:
            lines.append(f"- input: {u['input']}")

    warnings = resp.get("warnings") or []
    if warnings:
        lines.append("## warnings")
        for w in warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)


def _coerce_single_str(x: Any) -> str | None:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        for v in x:
            s = coerce_str(v)
            if s:
                return s
        return None
    if isinstance(x, str) and x.strip()[:1] == "[":
        parsed = coerce_list(x)
        if isinstance(parsed, list):
            for v in parsed:
                s = coerce_str(v)
                if s:
                    return s
            return None
    return coerce_str(x)


def sentence_statistics(
    ctx: RunContext[HarnessDeps],
    charges: str | None = None,
    reference_year: int | None = None,
) -> str:
    """단일 죄명 1심 선고 분포 — charges(죄명 하나)가 필수입니다. 죄명이 공식 표기면 곧바로 통계
    (표본 30↑ 통계+비교판례 / 미만 개별 사례)를, 법률명만 주면 그 법률의 죄명 후보를 반환합니다.

    언제:
    - 형량 전망·구형·양형 의견·사건 위치를 검토할 때 결론 전에 호출하세요.
    - compute_sentencing_range 의 공식 '범위'가 실무에서 어디 안착하는지 실데이터로 받칠 때.

    규칙:
    - 죄명은 판결문·공소장 표기(대검 죄명표)로 하나씩: 형법은 '특수상해'처럼, 특별법은
      '도로교통법위반(음주운전)'처럼. 약칭·구어(정보통신망법·몰카·보이스피싱)도 풀어 줍니다.
    - status=candidates 는 통계가 아니라 선택지입니다 — 후보의 이름 하나를 charges 에 그대로 넣어 재호출.
    - 이종경합은 통계·사례에서 제외합니다. 죄명별 분포를 합산·평균·1.5배해 경합범 분포로
      만들지 마세요. 형법 제38조 제1항 제2호는 가장 중한 죄 장기(벌금은 다액)의 1/2까지,
      각 죄 장기·다액 합계 이내로 가중하는 처단형 상한이지 통계 결합식이 아닙니다.
    - low_n_grid 사례는 일반화하면 안 됩니다.

    응답: markdown-KV. 비교 사례는 반환 url만 인용 링크로 쓰고 집계 수치는 링크 없이 제시.

    Args:
      charges: 죄명 하나(공식 표기·약칭·구어) 또는 법률명('스토킹처벌법위반' → 그 법률의 죄명 후보).
      reference_year: 비교 case·그리드 선택의 연도 기준 — 가까운 사건 우선. None 이면 최근 우선.
        연도로 **좁히는** 인자는 없습니다 — 풀이 죄명당 중앙값 2건이라 연도로 자르면 통계가
        사라집니다. 시간 변화는 응답의 `벌금 비율`(전체·최근 절반)과 `수록 연도`가 싣습니다.
    """
    q_raw = _coerce_single_str(charges)
    if not q_raw:
        return _format_response_md({
            "status": "missing_input",
            "warnings": ["charges(죄명 하나)가 필요합니다."],
        })
    qnorm = _norm_charge(q_raw)
    qnorm, _bracket, _quotes = strip_charge_decorations(qnorm)
    if numeric_charge_tokens(qnorm) is not None or ARTICLE_REF_RE.search(qnorm):
        return _format_response_md({
            "status": "charge_numeric",
            "warnings": [
                "charges 에는 죄명 문자열을 넣으세요(예: charges='강제추행'). "
                "조문 번호·식별자 숫자로는 통계를 찾지 않습니다.",
            ],
        })
    if not qnorm:
        return _format_response_md(
            {"status": "no_data", "warnings": ["charges 정규화 후 비어있음"]})

    conn = open_db()
    try:
        exclude_ids = getattr(ctx.deps, "exclude_case_ids", None) or frozenset()
        index = official_index(conn)
        if not index.available:
            return _format_response_md({
                "status": "unavailable",
                "warnings": ["공식 죄명 표(official_charges)가 이 DB 에 적재되지 않았습니다 — "
                             "build_sample_db.py 를 돌린 뒤 다시 호출하세요."]})
        res = index.resolve(qnorm)
        if res.kind == "exact":
            resp = _stats_for_name(conn, index, res,
                                   exclude_ids=exclude_ids, reference_year=reference_year)
        elif res.kind == "statute_exact":
            resp = _law_candidates(conn, index, res, qnorm,
                                   exclude_ids=exclude_ids)
        else:
            resp = {"status": "no_data", "unmatched_charges": [{"input": qnorm}],
                    "warnings": [f"'{qnorm}' 은 공식 죄명(대검 죄명표)에 없는 표기입니다 — 판결문 죄명 표기"
                                 "(예: 특수상해, 도로교통법위반(음주운전))로 재호출하고, 법률명만 알면"
                                 " charges='<법률명>위반' 으로 그 법률의 죄명 후보를 받으세요."]}
        if res.alias_from:
            resp["query_resolved"] = f"{res.alias_from[0]} → {res.alias_from[1]}"
        return _format_response_md(resp)
    finally:
        conn.close()


def _pool_norms(conn: sqlite3.Connection, name: str) -> list[str]:
    try:
        rows = conn.execute("SELECT charge_norm FROM charge_norm_map WHERE official_name=?", (name,)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    norms = [r[0] for r in rows]
    if name not in norms:
        norms.append(name)
    return norms


def _solo_counts(
    conn: sqlite3.Connection, names: list[str], exclude: frozenset[int],
) -> dict[str, int]:
    if not names:
        return {}
    owner: dict[str, str] = {n: n for n in names}
    try:
        marks = ",".join("?" * len(names))
        for cn, off in conn.execute(
                f"SELECT charge_norm, official_name FROM charge_norm_map WHERE official_name IN ({marks})", names):
            owner.setdefault(cn, off)
    except sqlite3.OperationalError:
        pass
    norms = list(owner)
    marks = ",".join("?" * len(norms))
    sql = f"""
        SELECT pdc.charge_norm, COUNT(DISTINCT pd.case_id || '/' || pd.defendant_id)
        FROM prec_defendant_charges pdc
        JOIN prec_defendants pd ON pd.case_id = pdc.case_id AND pd.defendant_id = pdc.defendant_id
        JOIN prec_sentences ps ON ps.case_id = pd.case_id
        JOIN prec_cases pc ON pc.id = pd.case_id
        WHERE pdc.charge_norm IN ({marks}) AND ps.instance = ? AND pd.n_charges = 1
          AND pd.sentence_type IN ('imprisonment','fine','not_guilty','life')
    """
    params: list[Any] = [*norms, STATS_INSTANCE]
    if exclude:
        sql += f" AND pd.case_id NOT IN ({','.join('?' * len(exclude))})"
        params.extend(exclude)
    sql += " GROUP BY pdc.charge_norm"
    out: dict[str, int] = {}
    for cn, n in conn.execute(sql, params):
        out[owner[cn]] = out.get(owner[cn], 0) + n
    return out


def _name_items(index, names: list[str], counts: dict[str, int]) -> list[dict[str, Any]]:
    items = []
    for n in names:
        c = counts.get(n, 0)
        if c <= 0:
            continue
        e = index.entries.get(n)
        items.append({"label": n, "n": c, "status": e.status if e else "현행",
                      "successor": e.successor if e else None})
    items.sort(key=lambda x: -x["n"])
    return items


def _stats_for_name(
    conn: sqlite3.Connection, index, res: Resolution, *,
    exclude_ids: frozenset[int], reference_year: int | None,
) -> dict[str, Any]:
    name = res.key or ""
    warnings: list[str] = []
    blk, is_stats, is_grid = _build_block(
        conn, name, _pool_norms(conn, name),
        exclude_ids=exclude_ids, reference_year=reference_year, warnings=warnings)
    resp = _finish_dict([blk], is_stats, is_grid, warnings)
    notes_top: list[str] = []
    if res.status == "구법":
        notes_top.append(f"'{name}' 은 옛 죄명표 표기(판결 당시 표기)"
                         + (f" — 현행 죄명은 '{res.successor}'" if res.successor else ""))
    if res.suffix_stripped:
        notes_top.append(f"접미 '{res.suffix_stripped}' 는 이 법률의 죄명 표기에 붙지 않습니다 — 부모 죄명 '{name}' 의 풀")
    if res.law_repealed:
        notes_top.append(f"'{res.statute}' 은 폐지된 법률입니다")
    if res.statute and not res.sub and not res.derived and index.entries.get(name) is None:
        notes_top.append(f"'{res.statute}' 은 죄명표에 세부 죄명이 없는 법률이라 이 풀은 그 법률의 모든 조항 위반을"
                         " 함께 셉니다 — 조항별 분포가 아닙니다.")
    elif res.candidates and not res.derived:
        notes_top.append(f"'{name}' 은 괄호 없는 표기 — 세부 죄명이 정해진 사건(아래 관련 죄명)을 뺀 그 밖의 조항 위반 풀입니다.")
    if notes_top:
        resp["notes_top"] = notes_top
    related_names: list[str] = [res.derived[0]] if res.derived else []
    related_names.extend(index.related(name))
    related_names.extend(name + suf for suf in ("교사", "방조"))
    related_names = [x for x in dict.fromkeys(related_names) if x != name]
    if related_names:
        counts = _solo_counts(conn, related_names, exclude_ids)
        resp["related"] = _name_items(index, related_names, counts)
    if resp.get("status") == "no_data":
        resp.setdefault("warnings", []).append(
            f"'{name}' 은 공식 죄명이지만 단독범 1심 표본이 0건입니다 — 다른 표기로 다시 찾을 필요 없습니다."
            + (" 관련 죄명에 표본이 있습니다." if resp.get("related") else ""))
    return resp


def _law_candidates(
    conn: sqlite3.Connection, index, res: Resolution, qnorm: str, *,
    exclude_ids: frozenset[int],
) -> dict[str, Any]:
    names = list(res.candidates)
    counts = _solo_counts(conn, names, exclude_ids)
    items = _name_items(index, names, counts)
    statute = res.statute
    warnings: list[str] = []
    if statute:
        bare = statute.replace(" ", "") + "위반"
        if bare not in names:
            bare_n = _solo_counts(conn, [bare], exclude_ids).get(bare, 0)
            if bare_n:
                warnings.append(f"괄호 없는 '{bare}' 표기 {bare_n}건은 세부 죄명을 알 수 없어 제외했습니다.")
    if res.sub:
        head = f"'({res.sub})' 은 {statute or '이 법률'} 의 공식 죄명(대검 죄명표)이 아닙니다 — 공식 죄명 중 표본이 있는 것"
    elif statute:
        head = f"'{statute}' 의 공식 죄명(대검 죄명표) 중 단독범 1심 표본이 있는 것 — {len(items)}/{len(names)}"
    else:
        head = f"'{qnorm}' 을 부속 죄명으로 가진 법률의 죄명 — 표본이 있는 것"
    if not items:
        if res.sub:
            msg = f"{head} — 없습니다. 이 법률의 공식 죄명 {len(names)}개 모두 단독범 1심 표본 0건입니다."
        else:
            what = f"'{statute}' 의 공식 죄명" if statute else f"'{qnorm}' 을 부속 죄명으로 가진 공식 죄명"
            msg = f"{what} {len(names)}개 모두 단독범 1심 표본 0건입니다 — 다른 표기로 다시 찾을 필요 없습니다."
        return {"status": "no_data", "unmatched_charges": [{"input": qnorm}], "warnings": [msg] + warnings}
    return {"status": "candidates", "query": qnorm, "candidates": items, "note": head, "warnings": warnings}


def _fetch_samples(
    conn: sqlite3.Connection,
    charge_norms: str | list[str],
    exclude_case_ids: frozenset[int] | None = None,
) -> list[sqlite3.Row]:
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

    if exclude_case_ids:
        marks = ",".join("?" * len(exclude_case_ids))
        sql += f" AND pd.case_id NOT IN ({marks})"
        params.extend(exclude_case_ids)

    return conn.execute(sql, params).fetchall()


def _build_block(
    conn: sqlite3.Connection, label: str, norms: list[str], *,
    exclude_ids: frozenset[int], reference_year: int | None, warnings: list[str],
) -> tuple[dict[str, Any], bool, bool]:
    singles = _fetch_samples(conn, norms, exclude_ids)
    blk: dict[str, Any] = {"charge": label, "n": len(singles)}
    trend = _fine_trend(singles)
    if trend:
        blk["fine_trend"] = trend
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
        blk["comparables"] = _stratified_sample(singles, reference_year=reference_year)
    else:
        is_grid = True
        blk["by_type"] = _by_type(singles) if singles else None
        grid_rows = _grid_sort(singles, reference_year)[:GRID_MAX]
        if grid_rows:
            blk["grid"] = grid_rows
            blk["notes"] = [
                f"단일 죄명 표본 {len(singles)}건 < {MIN_N_STATS} — 통계 생략, 단독 사례만 나열"
            ]
        else:
            blk["notes"] = ["단일 죄명 표본 0건 — 경합 사례로 보충하지 않음"]
    return blk, is_stats, is_grid


def _finish_dict(
    blocks: list[dict[str, Any]], any_stats: bool, any_grid: bool, warnings: list[str],
) -> dict[str, Any]:
    if not any(b["n"] or b.get("grid") for b in blocks):
        return {"status": "no_data", "charge_blocks": blocks,
                "warnings": (warnings or []) + ["매칭 case 없음"]}
    resp: dict[str, Any] = {"status": "ok" if any_stats else "low_n_grid",
                            "charge_blocks": blocks}
    if any_grid:
        resp["mode"] = "low_n — 통계 대신 개별 사례 그리드"
    if warnings:
        resp["warnings"] = warnings
    return resp
