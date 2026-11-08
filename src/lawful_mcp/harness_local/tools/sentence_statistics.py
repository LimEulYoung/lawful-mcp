"""sentence_statistics — first-instance sentences of the convictions that contain the given charges, with comparable
or listed cases: `statistics`, sections for `sentencing_analysis` (the only way in; not a registered tool).

Charge key = official charge name (the official charge-name schedule of the Supreme Prosecutors' Office —
`_charge_index`), read by `sentencing_analysis`. `charge_norm_map` groups the charge spellings found in judgments
under official names (typos, former-law names, aliases; deterministic), and a charge's pool is that group.

Pool (sampling principle: when in doubt, show the raw data rather than statistics):
  - **First instance only**: the sentencing data (`prec_sentences`) holds first-instance judgments only (case-number
    codes `고합`/`고단`/`고정`). Appeals lean towards mitigation and are left out.
  - **Groups by charges**: a defendant's one sentence covers all of their charges, so convictions are grouped by how
    close their charges are to the given ones (`_actual_sentences`). The statistics come from the groups that hold
    every given charge, nearest first, until there are MIN_N_STATS convictions; per-charge distributions are never
    summed or weighted into a concurrent-offence distribution.
  - **Convictions only**: acquittals are counted beside them; the statistics, comparable cases and listed cases are
    convictions (acquittals at the bottom of a severity order would pull the middle down).
  - Old spellings keep their own pools and are named with their size: the charge's former names (`official_charges`
    status 구법 with this successor — an earlier law, possibly another statutory penalty) and spellings that cover
    several current charges (the bare '<statute>위반', a former umbrella name).
  - n >= MIN_N_STATS (30) convictions: sentence-type shares, medians with the middle half and the suspension rate,
    the same for the last ten years, and severity 11-quantile comparables. Fewer: **no statistics** (a small-sample
    figure would be misused as an anchor) — the convictions are listed instead.
  - Shown cases carry what their judgment noted, where the sentencing data v2 (`prec_sentences.raw_v2`) has it: five
    circumstances that split sentences, the size of the harm or danger and the guideline the judgment wrote. It
    replaces the reason summary on those cases.

Every query leaves out the judgments in `exclude`, the evaluation's blocked judgments (`exclude_case_ids`).

In multi-defendant cases the sample unit is the *per-defendant row* — if defendants A and B of one case were
sentenced differently, both count.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from typing import Any

from . import statutes as _statutes
from ._charge_index import Resolution, norm_cmp, official_index, split_charge

# ---------- constants ----------

# Minimum sample for computing statistics — below it, grid mode (no statistics)
MIN_N_STATS = 30

GRID_MAX = 15            # max grid rows
RELATED_MAX = 15         # display cap for candidate/related charges (by n, descending)

# ---------- columns beside the sentence ----------
#   term_kind            '징역' | '금고' (금고 has no compulsory labour; nearly all custodial sentences for negligence
#                        offences are 금고)
#   sentence_months_min  the minimum of an indeterminate juvenile sentence; sentence_months is its maximum
#   data_quality         excluded:*   not a judgment imposing a sentence (an order to reopen a case, etc.)
#                        unverified:* the stored term matches no sentence in the disposition (separate sentences
#                                     summed, etc.)

# Rows marked excluded or unverified are not this judgment's sentence and are left out of every count.
_QUALITY_WHERE = (
    " AND (pd.data_quality IS NULL"
    " OR pd.data_quality NOT LIKE 'excluded:%' AND pd.data_quality NOT LIKE 'unverified:%')"
)


# ---------- statistics ----------
# Statistics and comparable cases use convictions only; acquittals are counted beside them. Medians and quartiles are
# observed sentences (nearest rank), so they read as real sentences.

KIND_NAME = {"fine": "벌금", "life": "무기징역·사형"}
KIND_ORDER = ("징역", "금고", "벌금", "무기징역·사형")
RECENT_YEARS = 10


def _pick(sorted_vals: list[int], p: float) -> int:
    """The observed value at quantile p (nearest rank)."""
    return sorted_vals[round(p * (len(sorted_vals) - 1))]


def _recent_from() -> int:
    """First year of the last ten years, counted from the run date (statutes._today_iso, frozen by the evaluation)."""
    return int(_statutes._today_iso()[:4]) - RECENT_YEARS + 1


def _years(rows: list[sqlite3.Row]) -> str:
    years = [r["decision_year"] for r in rows if r["decision_year"]]
    if not years:
        return "연도 미상"
    return str(years[0]) if min(years) == max(years) else f"{min(years)}~{max(years)}"


def _kind(row: sqlite3.Row) -> str:
    """징역 and 금고 are different custodial sentences (금고 has no compulsory labour) and are counted apart."""
    if row["sentence_type"] == "imprisonment":
        return row["term_kind"] or "징역"
    return KIND_NAME.get(row["sentence_type"], row["sentence_type"])


def _kinds(rows: list[sqlite3.Row], shares: bool) -> str:
    """'징역 41 (91%) · 벌금 4 (9%)'; counts only when `shares` is false."""
    counts = Counter(_kind(r) for r in rows)
    order = [k for k in KIND_ORDER if counts[k]] + sorted(k for k in counts if k not in KIND_ORDER)
    return " · ".join(f"{k} {counts[k]}" + (f" ({round(100 * counts[k] / len(rows))}%)" if shares else "")
                      for k in order)


def _span(vals: list[int], unit: str) -> str:
    """'중앙값 8월 (가운데 50% 6~12월)' over sorted values."""
    lo, mid, hi = (_pick(vals, p) for p in (.25, .5, .75))
    middle = f"{lo:,}{unit}" if lo == hi else f"{lo:,}~{hi:,}{unit}"
    return f"중앙값 {mid:,}{unit} (가운데 50% {middle})"


def _medians(rows: list[sqlite3.Row], brief: bool = False) -> list[str]:
    """Median and middle half of each kind of sentence, with the suspended share of custodial sentences; `brief`
    gives the medians only."""
    out = []
    jail = [r for r in rows if r["sentence_type"] == "imprisonment"]
    months = sorted(r["sentence_months"] for r in jail if r["sentence_months"] is not None)
    if months:
        name = "징역" if {_kind(r) for r in jail} == {"징역"} else "자유형(징역·금고)"
        if brief:
            out.append(f"{name} 중앙값 {_pick(months, .5)}월")
        else:
            suspended = round(100 * sum(1 for r in jail if r["probation"]) / len(jail))
            line = f"{name} {_span(months, '월')} · 집행유예 {suspended}%"
            terms = sorted(r["probation_months"] for r in jail if r["probation"] and r["probation_months"] is not None)
            if terms:
                line += f" (유예기간 중앙값 {_pick(terms, .5)}월)"
            out.append(line)
    amounts = sorted(r["fine_amount"] for r in rows if r["sentence_type"] == "fine" and r["fine_amount"] is not None)
    if amounts:
        out.append(f"벌금 중앙값 {_pick(amounts, .5):,}원" if brief else f"벌금 {_span(amounts, '원')}")
    return out


def _recent_line(rows: list[sqlite3.Row], since: int) -> str:
    recent = [r for r in rows if (r["decision_year"] or 0) >= since]
    if not recent:
        return f"- 최근 {RECENT_YEARS}년({since}~): 유죄 판결 없음"
    full = len(recent) >= MIN_N_STATS
    return (f"- 최근 {RECENT_YEARS}년({since}~) 유죄 {len(recent)}건: {_kinds(recent, full)}"
            + "".join(" · " + m for m in (_medians(recent) if full else [])))


# ---------- comparable cases (stratified) ----------

def _severity_key(row: sqlite3.Row) -> tuple[int, int, int]:
    """Severity order: life > imprisonment > fine; within a kind by the actual term or amount, and a suspended term
    before an executed one of the same length."""
    st = row["sentence_type"]
    if st == "life":
        return (3, 0, 0)  # life imprisonment/death — the heaviest
    if st == "imprisonment":
        return (2, row["sentence_months"] or 0, 0 if row["probation"] else 1)
    return (1, row["fine_amount"] or 0, 0)  # fine


def _stratified_sample(
    rows: list[sqlite3.Row],
    *,
    reference_year: int | None = None,
) -> list[tuple[int, sqlite3.Row]]:
    """After sorting convictions by severity, takes one row at each of the 11 quantile cut points (q=0/10/.../100).

    For each q it takes the row at ``idx = round((q/100) * (n-1))``. When several rows tie (same severity), the row
    closest to ``reference_year`` (else the newest) comes first, then the lower case and defendant id.

    A judgment is shown once: if it appears at several cut points (several defendants or a tie), only the lowest q
    is kept.
    """
    if not rows:
        return []
    if reference_year is not None:
        sorted_rows = sorted(rows, key=lambda r: (_severity_key(r), abs((r["decision_year"] or 0) - reference_year),
                                                  r["case_id"], r["defendant_id"]))
    else:
        sorted_rows = sorted(rows, key=lambda r: (_severity_key(r), -(r["decision_year"] or 0),
                                                  r["case_id"], r["defendant_id"]))
    n = len(sorted_rows)
    out: list[tuple[int, sqlite3.Row]] = []
    seen: set[int] = set()
    for q in range(0, 101, 10):
        r = sorted_rows[round((q / 100) * (n - 1))]
        if r["case_id"] in seen:
            continue
        seen.add(r["case_id"])
        out.append((q, r))
    return out


def _grid_sort(
    rows: list[sqlite3.Row], reference_year: int | None
) -> list[sqlite3.Row]:
    """Sort for the grid (case listing) — closest to the reference year first, or newest first if there is none;
    then the lower case and defendant id."""
    if reference_year is not None:
        return sorted(
            rows,
            key=lambda r: (
                abs((r["decision_year"] or 0) - reference_year),
                -(r["decision_year"] or 0),
                r["case_id"],
                r["defendant_id"],
            ),
        )
    return sorted(rows, key=lambda r: (-(r["decision_year"] or 0), r["case_id"], r["defendant_id"]))


# ---------- row serialization ----------

REASON_MAX = 160  # display cap for sentencing_reason


def _sentence_str(row: sqlite3.Row) -> str:
    """A conviction's sentence; every stored prison term, fine and suspension carries its number."""
    st = row["sentence_type"]
    if st == "imprisonment":
        kind, months, low = row["term_kind"] or "징역", row["sentence_months"], row["sentence_months_min"]
        # An indeterminate juvenile sentence: the two numbers are one sentence.
        s = f"{kind} 장기 {months}월·단기 {low}월" if low is not None else f"{kind} {months}월"
        return s + (f" 집유{row['probation_months']}월" if row["probation"] else "")
    if st == "fine":
        return f"벌금 {row['fine_amount']:,}원"
    return "무기징역/사형"


def _short_case_no(case_number: str | None) -> str:
    """Shortens joined case numbers (comma-separated, ~100 chars) to the first number + the count of joined cases —
    the full list is on the url page."""
    parts = [p.strip() for p in (case_number or "").split(",") if p.strip()]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} 외 {len(parts) - 1}건 병합"


# ---------- what each shown judgment noted (sentencing data v2) ----------
# `prec_sentences.raw_v2` records per defendant five circumstances that split sentences — answered 예 / 아니오 / 없음
# (the judgment says nothing) — the size of the harm or danger in the judgment's words, and the sentencing guideline
# the judgment wrote. Each shown case carries them, so the model can see which cases resemble its own.
# A circumstance the judgment does not mention is left out: written as "없음" it would read as a denial.

CIRCUMSTANCES = (  # field, the judgment says yes, the judgment says no
    ("prior_same", "동종 전과 있음", "동종 전과 없음"),
    ("repeat_or_probation", "누범·집유 중 범행", "누범·집유 중 아님"),
    ("settlement", "합의·처벌불원", "합의·처벌불원 없음"),
    ("restitution", "피해 회복", "피해 회복 없음"),
    ("confession", "자백", "부인"),
)
NOTED_MAX = 200  # display cap for the size sentence and for the guideline the judgment wrote (97% and 91% whole)
NOTED_CAPTION = ("- 사정은 판결이 적은 형의 사정(말하지 않은 사정은 뺐다)과 피해·위험의 크기, 판결의 양형기준은 판결이 적은"
                 " 양형기준(영역 · 권고 범위 · 유형)이다.")


def _clip(text: str) -> str:
    return text if len(text) <= NOTED_MAX else text[: NOTED_MAX - 1] + "…"


def _noted(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> dict[tuple[int, str], list[str]]:
    """(case, defendant) → the lines on what the judgment noted, for `rows`; empty without v2 data."""
    if not rows:
        return {}
    ids = sorted({r["case_id"] for r in rows})
    found = conn.execute(f"SELECT case_id, raw_v2 FROM prec_sentences WHERE raw_v2 IS NOT NULL "
                         f"AND case_id IN ({','.join('?' * len(ids))})", ids).fetchall()
    out: dict[tuple[int, str], list[str]] = {}
    for case_id, raw in found:
        for d in json.loads(raw)["defendants"]:
            said = [yes if d[f] == "예" else no for f, yes, no in CIRCUMSTANCES if d[f] in ("예", "아니오")]
            if d["magnitude"]:
                said.append(f"크기 「{_clip(d['magnitude'])}」")
            g = d["guideline"]
            written = " · ".join(x for x in (g["zone"], g["range"], g["type"]) if x)
            lines = [f"  사정: {' · '.join(said)}"] if said else []
            if written:
                lines.append(f"  판결의 양형기준: {_clip(written)}")
            out[(case_id, str(d.get("id")))] = lines
    return out


def _case_lines(row: sqlite3.Row, noted: list[str] = (), others: frozenset[str] = frozenset()) -> list[str]:
    """Case body lines shared by comparables and the grid (indented key: value), with what the judgment noted.

    The url ends with the case id; other charges are listed only when the judgment convicted of charges beyond the
    given ones (`others`). `defendant_id` is not shown: it is an anonymized label that means something only inside
    its judgment (`피고인`, `A`, `甲`, `7`, …).
    """
    lines = [
        f"  사건: {_short_case_no(row['case_number'])} ({row['decision_year'] or '선고 연도 미상'})",
        f"  링크: https://lawful.crow-tit.com/cases/{row['case_id']}",
        *([f"  다른 죄: {' · '.join(sorted(others))}"] if others else []),
        f"  선고: {_sentence_str(row)}",
        *noted,
    ]
    # What the judgment noted takes the place of the reason summary.
    reason = None if noted else row["sentencing_reason"]
    if reason:
        if len(reason) > REASON_MAX:
            reason = reason[: REASON_MAX - 1] + "…"
        lines.append(f"  양형 이유: {reason}")
    return lines


# ---------- the pool: convictions that contain the given charges ----------
# A defendant's sentence covers all of their charges. Convictions are grouped by how close their charges are to the
# given ones — more of the given charges first, then fewer other charges — so the first group is "exactly these
# charges". Statistics come from the groups that hold every given charge, taken in that order until there are
# MIN_N_STATS convictions (whole groups); judgments holding only some of the given charges are counted on their own
# lines, never pooled into the statistics. For a defendant with several charges this predicts the kind of sentence
# and the term better than one charge's pool (leave-one-out over first-instance convictions: kind 0.51 → 0.60, term
# within ×1.5 0.26 → 0.37).

TIERS_SHOWN = 4  # groups holding every given charge, listed one line each


def _fetch_pool(
    conn: sqlite3.Connection, names: list[str], exclude: frozenset[int],
) -> list[tuple[sqlite3.Row, frozenset[str]]]:
    """First-instance defendants with at least one of `names` (official charge names), each with all of their
    charges as official names. Acquittals included; `exclude` judgments left out."""
    norms = list(dict.fromkeys(n for name in names for n in _pool_norms(conn, name)))
    if not norms:
        return []
    marks = ",".join("?" * len(norms))
    where = f"""
        FROM prec_defendant_charges pdc
        JOIN prec_defendants pd ON pd.case_id = pdc.case_id AND pd.defendant_id = pdc.defendant_id
        JOIN prec_cases pc ON pc.id = pd.case_id
        WHERE pdc.charge_norm IN ({marks})
          AND pd.sentence_type IN ('imprisonment','fine','not_guilty','life')""" + _QUALITY_WHERE
    params: list[Any] = list(norms)
    if exclude:
        where += f" AND pd.case_id NOT IN ({','.join('?' * len(exclude))})"
        params.extend(exclude)
    rows = conn.execute(f"""
        SELECT DISTINCT pd.case_id, pd.defendant_id, pd.sentence_type, pd.sentence_months, pd.fine_amount,
               pd.probation, pd.probation_months, pd.sentencing_reason, pc.case_number, pc.decision_year,
               pd.term_kind, pd.sentence_months_min
        {where}""", params).fetchall()
    official = dict(conn.execute("SELECT charge_norm, official_name FROM charge_norm_map"))
    charges: dict[tuple[int, str], set[str]] = defaultdict(set)
    for cid, did, norm in conn.execute(f"""
            SELECT x.case_id, x.defendant_id, x.charge_norm FROM prec_defendant_charges x
            WHERE (x.case_id, x.defendant_id) IN (SELECT DISTINCT pd.case_id, pd.defendant_id {where})""", params):
        charges[(cid, str(did))].add(official.get(norm) or norm)
    return [(r, frozenset(charges[(r["case_id"], str(r["defendant_id"]))])) for r in rows]


def _group_name(extra: int, given: int) -> str:
    """A group holding every given charge, by how many other charges it adds."""
    these = "이 죄" if given == 1 else "이 죄들"
    return f"{these}만" if not extra else f"{these} + 다른 죄 {extra}개"


def _guilty(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    return [r for r in rows if r["sentence_type"] != "not_guilty"]


def _group_line(label: str, rows: list[sqlite3.Row]) -> str:
    guilty = _guilty(rows)
    acquitted = len(rows) - len(guilty)
    line = f"- {label}: 유죄 {len(guilty)}건"
    if guilty:
        line += " · " + _kinds(guilty, True) + "".join(" · " + m for m in _medians(guilty, brief=True))
    return line + (f" · 무죄 {acquitted}건" if acquitted else "")


def _actual_sentences(
    conn: sqlite3.Connection, names: list[str], *, exclude: frozenset[int], reference_year: int | None,
) -> tuple[str, list[tuple[str, list[str]]]]:
    """(status, sections) — the groups, the statistics (or the listed cases) and the comparable cases for `names`."""
    given = set(names)
    head = f"## 묶음별 (1심) — 기준 죄명: {', '.join(names)}"
    pool = _fetch_pool(conn, names, exclude)
    if not pool:
        return "no_data", [(head, ["- 이 죄가 든 1심 판결이 없다."])]

    def key(charges: frozenset[str]) -> tuple[int, int]:
        return -len(given & charges), len(charges - given)

    groups: dict[tuple[int, int], list[tuple[sqlite3.Row, frozenset[str]]]] = defaultdict(list)
    for row, charges in pool:
        groups[key(charges)].append((row, charges))
    full = sorted(k for k in groups if -k[0] == len(given))   # groups holding every given charge
    lines = [_group_line(_group_name(k[1], len(given)), [r for r, _c in groups[k]]) for k in full[:TIERS_SHOWN]]
    if len(full) > TIERS_SHOWN:
        rest = [r for k in full[TIERS_SHOWN:] for r, _c in groups[k]]
        lines.append(_group_line(f"그보다 다른 죄가 많은 판결({len(full) - TIERS_SHOWN}묶음)", rest))
    if len(given) > 1:
        lines += [_group_line(f"{name}만", [r for r, c in pool if c == {name}]) for name in names]
    out = [(head, lines)]

    taken: list[tuple[sqlite3.Row, frozenset[str]]] = []
    used: list[tuple[int, int]] = []
    for k in full:
        if len(_guilty([r for r, _c in taken])) >= MIN_N_STATS:
            break
        taken += groups[k]
        used.append(k)
    guilty = [(r, c) for r, c in taken if r["sentence_type"] != "not_guilty"]
    if not guilty:   # judgments exist (acquittals, or only some of the charges) — listed above, nothing to count
        return "low_n_grid", out + [("## 통계 표본", ["- 입력한 죄가 모두 든 1심 유죄 판결이 없다."])]
    names_used = [_group_name(k[1], len(given)) for k in used]
    span = names_used[0] if len(names_used) == 1 else f"{names_used[0]} ~ {names_used[-1]}"
    rows = [r for r, _c in guilty]
    acquitted = len(taken) - len(guilty)
    sample = [f"- 유죄 {len(rows)}건" + (f" · 무죄 {acquitted}건(통계와 사례에서 뺐다)" if acquitted else "")
              + f" · 수록 기간 {_years(rows)}"]
    others = {(r["case_id"], r["defendant_id"]): c - given for r, c in guilty}
    if len(rows) < MIN_N_STATS:
        listed = [r for k in used for r in _grid_sort(_guilty([x for x, _c in groups[k]]), reference_year)][:GRID_MAX]
        sample += [f"- 형종: {_kinds(rows, False)}",
                   f"- 참고: 유죄 표본 {len(rows)}건 < {MIN_N_STATS} — 통계 없이 유죄 사례만 나열한다"]
        cases_head = "## 사례 나열 (유죄 판결 — 통계 아님, 분포로 일반화하지 않는다)"
        status, shown = "low_n_grid", [(None, r) for r in listed]
    else:
        sample += [f"- 형종: {_kinds(rows, True)}", *(f"- {m}" for m in _medians(rows)),
                   _recent_line(rows, _recent_from())]
        cases_head = "## 비교 판례 (통계 표본의 유죄 판결을 형이 가벼운 순으로 세운 11분위)"
        status, shown = "ok", _stratified_sample(rows, reference_year=reference_year)
    noted = _noted(conn, [r for _q, r in shown])
    cases = [NOTED_CAPTION] if noted else []
    for q, r in shown:
        body = _case_lines(r, noted.get((r["case_id"], str(r["defendant_id"])), ()),
                           others[(r["case_id"], r["defendant_id"])])
        cases.append("\n".join([f"- 자리: {q}%", *body]) if q is not None else "\n".join(["- " + body[0].lstrip(), *body[1:]]))
    return status, out + [(f"## 통계 표본 — {span}", sample), (cases_head, cases)]


# ---------- main ----------

READING = "## 실선고 통계 — 읽는 법", """아래는 전국 양형 통계가 아니라 이 코퍼스에 수록된 1심 판결문 표본의 분포다.
- 판결을 입력한 죄를 더 많이, 다른 죄를 더 적게 포함한 순으로 묶는다. 다른 죄가 함께 있는 판결의 형은 그 죄까지
  합친 하나의 형이다.
- 통계와 비교 판례는 입력한 죄가 모두 든 묶음을 앞에서부터 유죄 30건이 될 때까지 더한 표본에서 낸다. 모두 더해도
  30건이 안 되면 통계 없이 사례만 나열한다.
- 무죄는 건수만 적고, 통계·비교 판례·사례는 유죄 판결만으로 만들었다.
- 비교 판례는 유죄 판결을 형이 가벼운 순으로 세워 0·10·…·100% 자리에서 하나씩 고른 것이다.
- 규범이 정한 범위와 대조할 실무 참고값이지, 이 사건의 정답 분포가 아니다."""


def _name_notes(conn: sqlite3.Connection, index, res: Resolution, exclude: frozenset[int]) -> list[str]:
    """Notes on one resolved name's pool: an old or repealed spelling, a peeled suffix, a statute without detailed
    charges, and the old spellings that keep their own pools."""
    name = res.key or ""
    notes = []
    if res.status == "구법":
        notes.append(f"- 참고: '{name}' 은 옛 죄명표 표기(판결 당시 표기)"
                     + (f" — 현행 죄명은 '{res.successor}'" if res.successor else ""))
    if res.suffix_stripped:
        notes.append(f"- 참고: 접미 '{res.suffix_stripped}' 는 이 법률의 죄명 표기에 붙지 않습니다 — 부모 죄명 '{name}' 의 풀")
    if res.law_repealed:
        notes.append(f"- 참고: '{res.statute}' 은 폐지된 법률입니다")
    if res.statute and not res.sub and not res.derived and index.entries.get(name) is None:
        # Other statutes — the only charge name is 「<statute name>위반」 (item 6 of the Prosecutors' directive), so the
        # pool is every penal provision of that statute.
        notes.append(f"- 참고: '{res.statute}' 은 죄명표에 세부 죄명이 없는 법률이라 이 풀은 그 법률의 모든 조항 위반을"
                     " 함께 셉니다 — 조항별 분포가 아닙니다.")
    elif res.candidates and not res.derived:
        # A schedule statute name without parentheses (`도로교통법위반`): the schedule's 「그 밖의 조항」 ("other
        # provisions") spelling, which judgments also used before the statute's detailed charges existed.
        notes.append(f"- 참고: '{name}' 은 괄호 없는 표기 — 세부 죄명 없이 적힌 판결의 풀이다. 그 밖의 조항 위반과,"
                     " 세부 죄명이 생기기 전 판결이 섞여 있을 수 있다(세부 죄명은 아래 관련 죄명).")
    # Old spellings keep their own pools: an earlier law's name may carry another statutory penalty, and a name
    # covering several charges mixes them.
    former, umbrella = _former_names(index, name), _umbrella_names(conn, index, name)
    counts = conviction_counts(conn, former + umbrella, exclude)
    for member in former + umbrella:
        if counts.get(member):
            label, why = (("옛 죄명 표기", "옛 법의 같은 행위라 법정형이 다를 수 있어 따로 둔다.") if member in former
                          else ("세부 죄명을 나누지 않은 표기", "다른 세부 죄명이 섞여 있어 따로 둔다."))
            notes.append(f"- {label}: {member} 1심 유죄 {counts[member]}건 — {why}")
    return notes


RELATED = ("## 관련 죄명 (같은 죄의 미수·상습·특수형, 같은 법률의 다른 죄, 이 이름을 품은 특별법 죄 — "
           "해당하면 그 이름을 목록에 넣어 다시 부른다)")


def _related_lines(conn: sqlite3.Connection, index, names: list[str], resolutions: list[Resolution],
                   exclude: frozenset[int]) -> list[str]:
    """Related charges of the given names — the parent of a derived name, the same statute's charges, the incited and
    aided forms — with their first-instance conviction counts."""
    related: list[str] = []
    for res in resolutions:
        name = res.key or ""
        related += [res.derived[0]] if res.derived else []
        related += index.related(name) + [name + suf for suf in ("교사", "방조")]
    shown = set(names) | {m for name in names for m in _former_names(index, name) + _umbrella_names(conn, index, name)}
    related = [x for x in dict.fromkeys(related) if x not in shown]
    items = _name_items(index, related, conviction_counts(conn, related, exclude))
    lines = [_name_line(c) for c in items[:RELATED_MAX]]
    if len(items) > RELATED_MAX:
        lines.append(f"- (외 {len(items) - RELATED_MAX}개 더)")
    return lines


def _name_line(c: dict[str, Any]) -> str:
    tail = ""
    if c.get("status") == "구법":
        tail = f" (구법 표기{' → 현행 ' + c['successor'] if c.get('successor') else ''})"
    return f"- {c['label']}{tail} — 1심 유죄 {c['n']}건"


def statistics(
    conn: sqlite3.Connection, resolutions: list[Resolution], *, exclude: frozenset[int] = frozenset(),
    reference_year: int | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    """(status, sections) of the actual sentences for official names already resolved (`Resolution.kind ==
    'exact'`): notes on each pool, how to read the numbers, the groups, the statistics or listed cases, and related
    charges. `reference_year` puts the cases closest to it first among equals."""
    index = official_index(conn)
    picked = list({r.key: r for r in resolutions if r.kind == "exact" and r.key}.values())
    names = [r.key for r in picked]
    notes = list(dict.fromkeys(note for r in picked for note in _name_notes(conn, index, r, exclude)))
    status, parts = _actual_sentences(conn, names, exclude=exclude, reference_year=reference_year)
    related = _related_lines(conn, index, names, picked, exclude)
    parts = [(READING[0], [READING[1], *notes]), *parts, *([(RELATED, related)] if related else [])]
    return status, [(h, "\n".join(lines)) for h, lines in parts]


# ---------- official charge name → pool ----------

def _pool_norms(conn: sqlite3.Connection, name: str) -> list[str]:
    """Official charge name → the judgment spellings grouped under that name (`charge_norm_map`, which maps every
    stored spelling, the name's own included)."""
    return [r[0] for r in conn.execute("SELECT charge_norm FROM charge_norm_map WHERE official_name=?", (name,))]


def _former_names(index, name: str) -> list[str]:
    """Former names the charge schedule gives `name` as their successor (status 구법). Their judgments punished the
    same act under an earlier law, possibly with another statutory penalty, so they are named beside the pool rather
    than pooled."""
    return [e.name for e in index.entries.values() if e.status == "구법" and e.successor == name]


def _umbrella_names(conn: sqlite3.Connection, index, name: str) -> list[str]:
    """Spellings that cover `name` together with other charges, so their judgments stay a separate pool: the bare
    '<statute>위반' of a parenthesized charge, and a former name without one successor whose schedule note lists this
    charge's parenthesis ('구 표기 — 현행 (도주치사)·(도주치상)')."""
    statute, _sub = split_charge(norm_cmp(name))
    paren = re.search(r"\(([^()]+)\)$", name)
    if not statute or not paren:
        return []
    out = []
    bare = index.by_cmp.get(norm_cmp(statute + "위반"))
    if bare and bare != name:
        out.append(bare)
    rows = conn.execute("SELECT name, note FROM official_charges WHERE status='구법' AND successor IS NULL").fetchall()
    out += [n for n, note in rows
            if f"({paren[1]})" in (note or "") and split_charge(norm_cmp(n))[0] == statute and n not in out]
    return out


def conviction_counts(conn: sqlite3.Connection, names: list[str], exclude: frozenset[int]) -> dict[str, int]:
    """First-instance convictions that contain each official charge name — for the candidate and related lists."""
    if not names:
        return {}
    owner = dict(conn.execute(f"SELECT charge_norm, official_name FROM charge_norm_map WHERE official_name IN "
                              f"({','.join('?' * len(names))})", names).fetchall())
    if not owner:
        return {}
    norms = list(owner)
    sql = f"""
        SELECT pdc.charge_norm, pd.case_id, pd.defendant_id
        FROM prec_defendant_charges pdc
        JOIN prec_defendants pd ON pd.case_id = pdc.case_id AND pd.defendant_id = pdc.defendant_id
        WHERE pdc.charge_norm IN ({','.join('?' * len(norms))})
          AND pd.sentence_type IN ('imprisonment','fine','life')
    """ + _QUALITY_WHERE
    params: list[Any] = list(norms)
    if exclude:
        sql += f" AND pd.case_id NOT IN ({','.join('?' * len(exclude))})"
        params.extend(exclude)
    seen: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for cn, cid, did in conn.execute(sql, params):
        seen[owner[cn]].add((cid, str(did)))   # a defendant counts once per name, whatever the spellings
    return {n: len(v) for n, v in seen.items()}


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
