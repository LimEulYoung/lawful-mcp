"""The one sentencing surface: a briefing on a charge, or the arithmetic for a case.

Given a charge alone this returns **everything the judgement needs, in one
response** (the briefing). Given the findings as well, it computes *that case*
(the calculation). The two do not overlap — a calculation does not reprint the
guideline.

## Why one tool

``compute_sentencing_range`` (the ladder) and ``sentence_statistics`` (the
distribution) used to stand side by side, and on a charge whose provision or
branch was ambiguous the ladder put up a **gate** and asked. Measured: of 20
free-form drink-driving calls, 16 stopped at that first gate and answered
without ever seeing the guideline, and accuracy on the kind of sentence was
*lower* than with no tool at all (35% vs 60%). Thirty days of production traffic
looked the same — 45% of the calls to the calculator were not calculations but
questions put back to the caller.

So this tool **does not ask back**. Where several provisions or branches are in
play it walks all of them through ``sentencing_candidates`` and merges them into
one response. Where the findings arrive but the provision is still open, it
computes every candidate.

## Boundaries

- No corpus logic of its own. The statutory, processed and recommended ranges
  and the verification come from ``compute_sentencing_range`` (the engine, not a
  registered tool), the distribution from ``sentence_statistics``, the article
  text from ``statute_lookup``.
- Nothing that cannot be acted on. Argument schemas, the key beside a group
  heading, "call again with…" — the engine's ``protocol=False`` drops those
  **inside its formatters**, rather than this module scrubbing a finished
  response with a regex.
- Size is held by a budget. Truncation is a silent failure, so when the response
  runs over it folds itself and says what it folded (measured: article text is
  the largest cost by far — one financial-markets statute ran to 9,906
  characters).
"""
from __future__ import annotations

import re

from pydantic_ai import RunContext

from ..config import case_url_base
from ..deps import HarnessDeps
from ._coerce import coerce_dict, coerce_int, coerce_str
from .compute_sentencing_range import (
    _MOD_MULT,
    _MOD_ORDER,
    GUIDELINE_FACTOR_KEYS,
    PROBATION_FACTOR_KEYS,
    Candidate,
    ChargeArg,
    OptStrArg,
    compute_sentencing_range,
    sentencing_candidates,
)
from .sentence_statistics import sentence_statistics
from .statutes import statute_lookup

# Response budget, in characters. Over it, `_articles_section` and `_trim` fold.
#
# Measured over all 677 charges: longest 12,812 (아동학대처벌법), median 7,935,
# 11 charges over 12,000, none over 13,000 — **1.4% of headroom (188 characters)**.
# One more constant line in the response grows all 677 at once (two lines about
# repeat offences cost exactly +97 characters across 673 of them). Anyone adding
# a section should read that headroom first and re-measure after; whether it
# still fits is what `test_sentencing_analysis` checks.
# Staying under is `_trim`'s job, but trimming means **folding**. 21 of the 677
# actually folded (3%: 15 general suspension factors, 6 ordinary factors, 0
# conditions), so it is still the exception — if that share climbs noticeably,
# it is time to raise the budget.
BUDGET = 13_000

# ---------------------------------------------------------------- the sequence
# Source: Sentencing Commission, 「양형기준 해설」 — how the guidelines are applied,
# and how a processed range is derived under the general part of the Criminal Act.
#
# The commentary's narrative guidelines (the ▷ form) are **left out**: this corpus
# holds none of them, so telling a model they exist only makes room to invent a
# multiplier. The multiple-offence chapter is out for the same reason — the engine
# handles it under §38.
#
# **Do not drop "형종 선택".** The commentary's derivation runs 법정형 → (형종 선택)
# → 누범가중 → …, and without that step a model picks from the imprisonment table
# in front of it: 6 fine-to-imprisonment flips in one run. The point is not to
# invent a recommended fine (only 4 of the 48 offence groups set one) but to make
# a place for the fine already carried in the statutory range.
FLOW = """## 양형 판단 흐름
출처: 대법원 양형위원회 「양형기준 해설」 — 양형기준의 적용방법 · 형법 총칙 처단형 산출과정

법정형 → 형종 선택 → 처단형 → 권고 형량범위 → 선고형 → 집행유예 순으로 좁혀 간다.
앞 단계가 뒤 단계의 테두리이며, 뒤 단계는 앞 단계를 넘지 못한다.

1. 법정형 — 해당 조문이 정한 형. 선택형(자유형 또는 벌금)이면 둘 다 확인한다.
2. 형종 선택 — 법정형이 선택형을 둔 경우, 사건의 경중을 보고 자유형과 벌금 중 하나를 정한다.
   벌금형을 선택하면 아래 4의 양형기준은 적용되지 않는다 — 교통·선거·스토킹·동물보호법
   범죄만 벌금 기준을 따로 두고, 나머지 범죄군의 양형기준은 자유형을 택한 경우의 기준이다.
3. 처단형 — 선택한 형종에 형법 §56 순서로 법률상 가중·감경을 적용한 범위.
4. 권고 형량범위 — 자유형을 선택한 경우, 양형기준을 적용한다.
   4-1. 범죄유형 결정 — 유형 목록에서 사건이 속하는 유형 하나를 고른다.
   4-2. 권고영역 결정 — 특별양형인자를 비교해 감경·기본·가중 중 하나를 고른다.
        · 행위인자끼리 가중·감경을 개수로 상쇄하고, 행위자/기타인자끼리 따로 상쇄한다.
        · 남은 인자가 없으면 기본영역.
        · 남은 인자가 한 방향뿐이면 그 방향의 영역.
        · 가중·감경이 섞여 남으면 행위인자를 우월하게 본다 — 행위인자 쪽이 많거나
          동수이면 그 방향. 행위자/기타인자 쪽이 많은 경우는 개수로 정해지지 않으므로
          특별양형인자를 종합 평가해 정한다.
        · 일반양형인자는 영역 결정에 영향이 없다 — 5단계에서 쓴다.
   4-3. 특별조정 — 상쇄하고 남은 특별가중인자가 2개 이상이면 가중영역의 상한을 1/2 높이고,
        남은 특별감경인자가 2개 이상이면 감경영역의 하한을 1/2 낮춘다. 자유형에만 적용한다.
   4-4. 처단형 우선 — 권고 형량범위가 처단형 범위를 벗어나면 처단형의 상한·하한이 기준이 된다.
5. 선고형 — 자유형이면 위 범위 안에서, 벌금이면 처단형의 금액 범위 안에서, 특별·일반
   양형인자를 종합해 구체적 형을 정한다.
6. 집행유예 — 선고형이 징역·금고 3년 이하일 때 판단한다. 집행유예 주요참작사유를
   비교해 실형과 집행유예 중 하나를 정한다."""

# ---------------------------------------------------------------- deriving §56
# The names, order and effect of the §56 grounds are **read from the engine's own
# tables** (`_MOD_ORDER`, `_MOD_MULT`). Only the prose lives here, so a change to
# a multiplier moves this section with it.
_ORDER_NOTE = {
    "본조_가중": "본조 자체의 가중 규정 (상습범·특정범죄가중법 등)",
    "특수교사방조_가중": "특수교사·특수방조 (형법 §34 ②)",
    # The line that separates a repeat offence from a post-judgment concurrent
    # one. It is here **not to teach law but to make one of this tool's two
    # `kind` values choosable**: `누범_가중` (long term ×2) and `법률상_임의감경`
    # (×0.5) run opposite ways, and the fact that tells them apart — whether the
    # offence predates the earlier judgment — is always in the record. Measured
    # over 1,040 cases, 36 (3.5%) cite a post-judgment concurrent offence, and
    # **fraud alone accounts for 22/144 (15.3%)**. In one of them a model read the
    # prior conviction as an aggravating ground and picked the aggravated band
    # (12 months against an actual 2); the court treated the same conviction as a
    # §39 ① equity ground and imposed 2 months.
    # The **argument value** is what gets written down. 「사후적 경합범」 is not a
    # `kind`, and naming it alone gets a model to invent a value under that name.
    "누범_가중": "누범 (형법 §35 ① — 금고 이상의 형의 집행이 끝나거나 면제된 후 3년 내에 범한 죄)\n"
                  "   앞 판결 확정 전에 범한 죄는 누범이 아니라 사후적 경합범이고,"
                  " `법률상_임의감경` 에 해당한다",
    # §56 ④ treats statutory mitigation as one step, so the mandatory and the
    # discretionary share a line and the numbering does not collide.
    #
    # **§39 ① has to be here.** This list is the only table of mitigating grounds
    # a model sees, and without §39 ① two benchmark runs had the same case called
    # with an invented `kind="사후적_경합_감경"` (the engine rejected it, correctly).
    # Nothing was hallucinated: a real doctrine had no name in the list. §39 ① is
    # "may reduce or remit", i.e. discretionary statutory mitigation, so it is the
    # second of the two names. Remission this tool cannot express (it deals in
    # multipliers) — only the reduction is computed.
    "법률상_필요감경": "필요적 — 방조 (§32 ②), 청각·언어 장애인 (§11) 등"
                       " / 임의적 — 미수 (§25 ②), 중지미수 (§26), 자수 (§52), 심신미약 (§10 ②),"
                       " 사후적 경합범 (§39 ①) 등",
    "경합범_가중": "경합범 (§37 전단·§38 ① 2호). 같은 죄명의 별개 행위가 여럿이면 적용된다",
    "작량감경": "정상참작 감경 (형법 §53)",
}


def _effect(kind: str) -> str:
    """`_MOD_MULT`'s (imp max, fine max, fine min, imp min) as one readable line."""
    imp_max, fine_max, fine_min, imp_min = _MOD_MULT[kind]
    if imp_max == imp_min == fine_max == fine_min:          # every bound alike — mitigation
        return f"자유형·벌금의 상한과 하한 모두 ×{imp_max:g}"
    bits = [f"{label} ×{m:g}" for label, m in
            (("자유형 장기", imp_max), ("벌금 다액", fine_max)) if m != 1.0]
    return " · ".join(bits) + (" (하한 불변)" if fine_max != 1.0 else " (하한·벌금 불변)")


def _sec56() -> str:
    """The §56 order. **Each line begins with the `kind` value itself** — no prettier name.

    It used to print `kind.replace("_", " ")`, so 「본조 가중」, and folded the two
    statutory-mitigation values into one 「법률상 감경」 line. A model could put the
    underscore back (`누범_가중` came out right), but **a folded line cannot be
    recovered** — no value by that name exists. Across two benchmark runs, the
    same case had a model reaching for §39 ① as `사후적_경합_감경`, then
    `사후적_경합범`; both were rejected. Printing the argument value removes the
    gap (this tool's own rule: the values in a call are the ones the response
    printed).
    """
    lines = ["## 처단형 계산방법",
             "법정형에 아래 사유를 형법 §56 이 정한 순서대로 적용한 것이 처단형이다.",
             "적용할 사유가 정해지면 이 도구를 `statutory_modifications` 인자와 함께"
             " 다시 불러 계산 결과를 받는다 — 예: [{\"kind\": \"누범_가중\"}]."
             " kind 는 아래 줄머리의 값을 그대로 쓴다.", ""]
    for kind in sorted(_ORDER_NOTE, key=lambda k: (_MOD_ORDER[k], k)):
        # §56 ④ is one step, but **there are two argument values** — show both.
        names = ("법률상_필요감경 / 법률상_임의감경" if kind == "법률상_필요감경" else kind)
        lines += [f"{_MOD_ORDER[kind]}. {names} — {_ORDER_NOTE[kind]}",
                  f"   효과: {_effect(kind)}"]
    lines += ["", "- 가중 결과가 자유형 상한을 넘으면 형법 §42 ② 에 따라 50년으로 자른다"
                  " (2010.10.16 이전 행위는 25년).",
              "- 순서를 바꿔도 대개 결과는 같지만(배수 곱), 위 상한에 걸리면 달라진다."]
    return "\n".join(lines)


# ---------------------------------------------------------------- the caption
# A caption has to describe **what is actually there**. Below 30 samples the
# statistics tool returns individual cases instead of quantiles, and given only a
# statute name it returns candidates. Attach a reading that does not match and the
# caption is false where it stands — so it branches on the declared `## status:`
# rather than guessing from the text.
_STAT_HEAD = """## 실선고 통계 — 읽는 법
아래는 전국 양형 통계가 아니라 이 코퍼스에 수록된 1심 판결문 표본의 분포다.
- `by_type` 의 `not_guilty` 는 무죄 건수다. 형종별 평균(`mean`)은 그 형종을 받은 건들 안의
  평균이므로 무죄를 포함하지 않는다."""

_STAT_BY_STATUS = {
    "ok": "- 반면 아래 비교 판례의 분위는 무죄를 포함한 전체 표본을 줄 세운 것이라, 하위 분위에"
          " 무죄가\n  올 수 있다. 둘은 모수가 다르므로 섞어 읽지 않는다.",
    "low_n_grid": "- 이 죄명은 표본이 30건 미만이라 분위 통계 대신 개별 사례만 나열된다. 몇 건의"
                  " 사례일 뿐\n  분포가 아니므로, 여기서 평균이나 경향을 끌어내지 않는다.",
    "candidates": "- 이 이름으로는 표본을 찾지 못했고 아래는 같은 법률의 죄명 후보다. 분포가 아니다.",
}
_STAT_TAIL = "- 규범이 정한 범위와 대조할 실무 참고값이지, 이 사건의 정답 분포가 아니다."


def _stat_section(ctx, charge: str, reference_year: int | None) -> tuple[str, str]:
    """(status of the statistics tool, the statistics section). Caption matches content."""
    raw = sentence_statistics(ctx, charges=charge, reference_year=reference_year).strip()
    status, body = _status_of(raw), _embed(raw)
    caption = _STAT_BY_STATUS.get(status)
    head = [_STAT_HEAD, caption, _STAT_TAIL] if caption else [
        "## 실선고 통계", "- 이 죄명으로는 이 코퍼스에서 1심 선고 표본을 찾지 못했다."]
    return status, "\n".join([*head, "", body])


# ---------------------------------------------------------------- assembling
def _status_of(md: str) -> str:
    """The `## status:` token on the first line."""
    first = md.lstrip().split("\n", 1)[0]
    return first[len("## status:"):].strip() if first.startswith("## status:") else ""


def _split(md: str) -> list[tuple[str, str]]:
    """(heading, body) per '## ' heading. The first element is ('', preamble)."""
    parts = re.split(r"(?m)^(## .*)$", md)
    return [("", parts[0])] + [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]


def _kv(body: str) -> dict[str, str]:
    """Only the `- key: value` lines (the engine's markdown-KV contract)."""
    return dict(re.findall(r"(?m)^- ([^:\n]+): (.*)$", body))


_META = ("## status:", "## stage:", "## mode:")


def _embed(md: str) -> str:
    """Fold another tool's response into this one, dropping its meta headings.

    A response carries one status line. A second `## status: ok` in the middle of
    the body leaves a model unable to tell which one belongs to the response."""
    return _render([(h, b) for h, b in _split(md) if not h.startswith(_META)])


def _render(sections: list[tuple[str, str]]) -> str:
    out = []
    for header, body in sections:
        if header:
            out.append(header)
        body = body.strip("\n")
        if body:
            out.append(body)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def _index_of(sections: list[tuple[str, str]], prefix: str) -> int | None:
    for i, (header, _b) in enumerate(sections):
        if header.startswith(prefix):
            return i
    return None


def _only_articles(md: str, want: set[str]) -> str:
    """Keep just the articles asked for — `statute_lookup` answers `347` with
    `347의2` alongside, and 347의2 (computer fraud) is a different offence."""
    keys = {a.replace("의", "-") for a in want} | set(want)
    head = re.split(r"(?m)^### ", md)[0].strip()
    blocks = re.findall(r"(?ms)^### (\S+)(.*?)(?=^### |\Z)", md)
    kept = [f"### {n}{b}".rstrip() for n, b in blocks if n in keys]
    return (head + "\n" + "\n".join(kept)).strip() if kept else ""


def _articles_section(ctx, cands: list[Candidate], room: int) -> list[str]:
    """The provisions relied on: full text while the budget allows, then links.

    Article text is this response's largest cost (measured: 9,906 characters for
    one financial-markets statute, 7,013 for the election act). Carrying all of it
    puts one charge in six over the replay cap, so the order of spending is decided
    here — full text from the front, links once there is no room. A model that
    needs the body calls `statute_lookup`, which is still there.
    """
    by_statute: dict[int, list[str]] = {}
    for c in cands:
        if c.statute_id and c.article_no:
            arts = by_statute.setdefault(c.statute_id, [])
            if c.article_no not in arts:
                arts.append(c.article_no)
    out: list[str] = []
    listed: list[str] = []
    room -= 260                      # set aside the 「원문 생략」 preamble below
    for sid, arts in by_statute.items():
        # One request per statute — asking article by article repeats the law's
        # header that many times. When the set will not fit, carry at least the
        # first article: one provision in full beats links alone, because that
        # provision is what the arithmetic rests on.
        taken: list[str] = []
        for subset in (arts, arts[:1]):
            if room <= 400:
                break
            md = _embed(_only_articles(
                statute_lookup(ctx, statute_id=sid, articles=subset), set(subset)))
            if md and len(md) <= room:
                out.append(md)
                room -= len(md)
                taken = subset
                break
        listed += [f"- §{a}: {case_url_base()}/statutes/{sid:06d}/{a}"
                   for a in arts if a not in taken]
    if listed:
        out.append("## 근거 조문 — 원문 생략\n"
                   "지면이 모자라 링크만 싣는다. 본문이 필요하면 `statute_lookup` 으로 조회한다.\n"
                   + "\n".join(listed))
    return out


# One row of the statutory-range table: `- "조항" / "분기" — 조건 문구: 자유형 … · 벌금 …`
_ROW_RE = re.compile(r"(?P<head>- .*?) — (?P<cond>.*?): (?P<tail>자유형 .*)$")
_COND_MIN = 20          # below this a condition no longer distinguishes branches


def _trim(sections: list[tuple[str, str]], room: int) -> list[str]:
    """Fold, in this order, when over budget — and always say what was folded.

    1) the general quadrants of the suspension factors (the major ones decide first)
    2) the ordinary sentencing factors (no effect on the band — FLOW 4-2 says so)
    3) the conditions in the statutory-range table (**rows are never removed**)

    Without 3 nothing can be folded on a charge with many candidates: 아동학대처벌법
    has 117 of them, and that one table runs to 7,906 characters — 62% of the
    response. Folding 1 and 2 still left it over budget.
    **A row (a candidate) is never deleted.** 「여기 있는 것이 전부다」 is this
    section's promise, and deleting a candidate tells a model that branch does not
    exist — which is the whole ground for not asking back. Cut the condition text
    instead, and say it was cut.
    """
    folded: list[str] = []
    for prefix, what in (("## 집행유예 참작사유", "집행유예 일반참작사유"),
                         ("## 일반양형인자", "일반양형인자")):
        if sum(len(h) + len(b) for h, b in sections) <= room:
            break
        i = _index_of(sections, prefix)
        if i is None:
            continue
        header, body = sections[i]
        if prefix == "## 집행유예 참작사유":
            keep, dropped = [], 0
            general = False
            for line in body.splitlines():
                if line.startswith("[general/"):
                    general = True
                elif line.startswith("["):
                    general = False
                if general and line.startswith("- "):
                    dropped += 1
                    continue
                if general and line.startswith("["):
                    continue
                keep.append(line)
            keep.append(f"- (일반참작사유 {dropped}개는 지면상 생략 — 주요참작사유가 먼저 가른다)")
            sections[i] = (header, "\n".join(keep))
        elif prefix == "## 법정형 — 조항·분기별":
            over = sum(len(h) + len(b) for h, b in sections) - room
            keep, cut = [], 0
            for line in body.splitlines():
                m = _ROW_RE.match(line)
                # Cut only the overflow: divide it across the remaining rows and
                # leave anything already shorter alone (cutting short conditions
                # makes the table unreadable).
                if m and over > 0:
                    width = max(_COND_MIN, len(m["cond"]) - max(1, over // 20))
                    if width < len(m["cond"]):
                        over -= len(m["cond"]) - width
                        cut += 1
                        line = f'{m["head"]} — {m["cond"][:width]}…: {m["tail"]}'
                keep.append(line)
            if cut:
                keep.append(f"- (조건 문구 {cut}개를 지면상 줄였다 — 후보는 하나도 안 지웠다."
                            " 전문은 `statute_lookup` 으로 조문을 본다)")
            sections[i] = (header, "\n".join(keep))
        else:
            n = len([l for l in body.splitlines() if l.startswith("- ")])
            sections[i] = (header, f"- (인자 {n}개 생략 — 권고영역 결정에는 영향이 없다)")
        folded.append(what)
    return folded


# ---------------------------------------------------------------- composition
def _where(c: Candidate) -> str:
    """The arguments that call this candidate back, which also name it in the response."""
    return " ".join(x for x in (c.statute_choice, c.branch_key, c.reference_choice) if x) or c.article


def _merge_guide_notes(got: list[tuple[Candidate, str]]) -> tuple[str, str] | None:
    """Every candidate's 「유형 지정 안내」 in one section, each tagged with the branch
    it belongs to. None when there are none.

    This is the only section that differs when candidates are merged. The
    guideline, factor and suspension sections are identical down to the md5
    (which is why `_brief` takes the first candidate's), but **the scope note
    differs per provision and paragraph**. Measured on drink-driving: the two
    §148의2 ① branches (repeat within 10 years) fall outside the guideline's
    scope and the three ③ branches fall inside. Carrying only the first
    candidate's note reads as though 「이 조항의 제1항은 양형기준 적용 대상이
    아닙니다」 covers the whole response, and **the branches that do have a
    guideline lose it** — leaving the recommended-range table just above unusable.
    """
    by_note: dict[str, list[str]] = {}
    clean: list[str] = []
    for c, resp in got:
        block = _split(resp)
        i = _index_of(block, "## 유형 지정 안내")
        lines = [l for l in block[i][1].splitlines() if l.strip()] if i is not None else []
        if not lines:
            clean.append(_where(c))
        for line in lines:
            by_note.setdefault(line, []).append(_where(c))
    if not by_note:
        return None
    if not clean and len(by_note) == 1:
        # Same note on every candidate — tagging branches would invent a difference.
        return "## 유형 지정 안내", next(iter(by_note)).strip()
    out = ["- 안내는 갈래마다 다르다. 사실관계로 정해진 갈래의 줄만 읽는다."]
    for note, wheres in by_note.items():
        out.append(f"- [{' · '.join(wheres)}] {note.strip().removeprefix('- ')}")
    if clean:
        out.append(f"- [{' · '.join(clean)}] 양형기준 적용범위 안 —"
                   " 위 「권고 형량범위」에서 유형을 고른다.")
    return "## 유형 지정 안내", "\n".join(out)


def _brief(ctx, charge: str, cands: list[Candidate], narrow: dict,
           reference_year: int | None) -> str | None:
    """The briefing. Walks every candidate and merges them into one. None if all fail."""
    got: list[tuple[Candidate, str]] = []
    for c in cands:
        resp = compute_sentencing_range(
            ctx, charge=charge, statute_choice=c.statute_choice, branch_key=c.branch_key,
            reference_choice=c.reference_choice, protocol=False, **narrow)
        if _status_of(resp) == "ok":
            got.append((c, resp))
    if not got:
        return None

    sections = _split(got[0][1])
    # This response belongs to the briefing; leaving the engine's stage name sends
    # a model looking for the ladder.
    i = _index_of(sections, "## stage:")
    if i is not None:
        sections[i] = ("## stage: brief", sections[i][1])

    # Several provisions or branches: only the statutory range becomes a per-
    # candidate table. Measured: the five drink-driving branches have guideline,
    # factor and suspension sections identical down to the md5, and **only the
    # statutory range differs** — so the rest is carried once.
    if len(got) > 1:
        sections = [(h, b) for h, b in sections
                    if not h.startswith(("## branch_key 선택:", "## reference 선택:"))]
    li = _index_of(sections, "## 법정형")
    if len(got) > 1 and li is not None:
        by_statute: dict[str, list[str]] = {}
        for c, resp in got:
            block = _split(resp)
            bi = _index_of(block, "## 법정형")
            vals = _kv(block[bi][1]) if bi is not None else {}
            # Do **not** run the branch name together as `statute_choice + ' ' +
            # branch_key`. Run together, a model does not know where to break it
            # when feeding it back: across two benchmark runs, `도로교통법§148의2
            # ③ 3-2` came back as `statute_choice="도로교통법§148의2 ③ 2"` in two
            # different cases, and both failed to resolve — one round trip each.
            # **Group by provision.** One charge has 117 candidates
            # (아동학대처벌법), so repeating the statute name per row costs
            # thousands of characters on its own — that broke the budget, and the
            # conditions were already short (「상해」, 「재물손괴」) so cutting them
            # saved nothing. The repetition was the waste.
            # Put the argument **name** next to the value, so it is copied whole.
            sub = " ".join(f'{k}="{v}"' for k, v in
                           (("branch_key", c.branch_key),
                            ("reference_choice", c.reference_choice)) if v)
            by_statute.setdefault(c.statute_choice or c.article, []).append(
                f"  · {sub or '—'} — {c.label or '—'}: "
                f"자유형 {vals.get('자유형', '—')} · 벌금 {vals.get('벌금', '—')}")
        rows = [line for sc, subs in by_statute.items()
                for line in (f'- statute_choice="{sc}"', *subs)]
        sections[li] = ("## 법정형 — 조항·분기별",
                        "사실관계로 아래 중 하나가 정해진다. 여기 있는 것이 전부다.\n"
                        "좁히려면 조항 줄과 그 아래 · 줄의 인자를 **표기 그대로** 넣어 다시 부른다.\n"
                        + "\n".join(rows))

    if len(got) > 1:
        merged = _merge_guide_notes(got)
        ni = _index_of(sections, "## 유형 지정 안내")
        if merged is None:
            if ni is not None:
                del sections[ni]
        elif ni is not None:
            sections[ni] = merged
        else:                               # note absent from the first candidate only
            gi = _index_of(sections, "## 특별양형인자")
            sections.insert(gi if gi is not None else len(sections), merged)

    # The sequence goes first (before the statutory range); §56 sits between the
    # statutory range and the recommended one.
    gi = _index_of(sections, "## 권고 형량범위")
    if gi is not None:
        sections.insert(gi, ("", _sec56()))
    li = _index_of(sections, "## 법정형")
    if li is not None:
        sections.insert(li, ("", FLOW))

    stat_status, stat_md = _stat_section(ctx, charge, reference_year)
    room = BUDGET - len(_render(sections)) - len(stat_md)
    folded = _trim(sections, BUDGET - len(stat_md) - 1500)
    room = BUDGET - len(_render(sections)) - len(stat_md)
    articles = _articles_section(ctx, [c for c, _r in got], room)

    body = _render(sections + [("", a) for a in articles] + [("", stat_md)])
    if folded:
        body = body.replace("\n## 양형 판단 흐름",
                            f"\n- 지면 절약: {' · '.join(folded)}는 접었다.\n\n## 양형 판단 흐름", 1)
    return body


def _calc(ctx, charge: str, cands: list[Candidate], narrow: dict, calc: dict) -> str:
    """The calculation. The guideline is not reprinted — the briefing already gave it."""
    blocks: list[str] = []
    for c in cands:
        resp = compute_sentencing_range(
            ctx, charge=charge, statute_choice=c.statute_choice, branch_key=c.branch_key,
            reference_choice=c.reference_choice, protocol=False, **narrow, **calc)
        sections = _split(resp)
        i = _index_of(sections, "## stage:")
        if i is not None:
            sections[i] = ("## stage: calc", sections[i][1])
        if len(cands) > 1:
            sections.insert(1, (f"## 조항·분기: {_where(c)}", c.label))
        blocks.append(_render(sections[1:] if blocks else sections))
    tail = ("\n- 기준 전문(유형·양형인자·집행유예 참작사유)은 앞의 양형 분석 응답에 있다 —"
            " 같은 죄명으로 다시 부르지 않는다.\n")
    if len(cands) > 1:
        tail = ("\n- 조항·분기가 사실관계로 정해지지 않아 후보 전부를 계산했다."
                " 하나로 좁히려면 `statute_choice`·`branch_key` 를 준다." + tail)
    return "\n".join(blocks).rstrip() + "\n" + tail


_CALC_ARGS = ("statutory_modifications", "guideline_type", "guideline_factors",
              "probation_factors", "sentence_months", "fine_amount")

# The statuses meaning "the charge was not in the mapping table". **Only these**
# turn into `no_guideline` when a distribution exists — a name absent from the
# mapping but present in the sentencing sample is an official charge that the
# guideline does not cover.
#
# The ambiguity and bad-argument statuses (`ambiguous_statute`,
# `ambiguous_category`, `exact_wrong_category`, `invalid_statute_choice`) must
# **never** go in. There the charge resolved and only the provision is open,
# which has nothing to do with whether a guideline exists. Measured: a model gave
# `statute_choice='도로교통법§148의2 ③ 2'` (trailing junk), got `ambiguous_statute`,
# and because a distribution existed it went out painted as `no_guideline` — a
# status that counts as success and means 「이 죄명에는 양형기준이 없다」. A
# notation slip would have been tallied as a fact about corpus coverage.
_NOT_IN_CORPUS = ("not_found", "not_found_with_candidates")

# ---------------------------------------------------------------- factor keys
# The factor lists in a response are grouped under **group headings** like
# `[특별/행위/감경]`. Pass a heading as the dict key and the arithmetic still runs
# `status: ok` while not one factor is counted, because the engine only `.get()`s
# the keys it knows. Measured (사기, 일반사기 1유형, sentence fixed at 6 months):
#   {"special_act_mitigators": […]} + {"major_positive": […×2]} → 영역 감경 · 집유 권고
#   {"감경": […]}              + {"positive": […]}          → 영역 기본 · major 0/0
#   {"zzz": […]}               + {"qqq": […]}               → 영역 기본 · major 0/0
# The last two raised no warning and no change of status. So it hands them back
# **without computing** — better than a wrong number. The heading-to-key mapping
# is printed here and only here (a correct call's response is left alone).
_HEADER_KEY = {
    "guideline_factors": (("[특별/행위/가중]", "special_act_aggravators"),
                          ("[특별/행위/감경]", "special_act_mitigators"),
                          ("[특별/행위자_기타/가중]", "special_actor_aggravators"),
                          ("[특별/행위자_기타/감경]", "special_actor_mitigators")),
    "probation_factors": (("[major/positive]", "major_positive"),
                          ("[major/negative]", "major_negative"),
                          ("[general/positive]", "general_positive"),
                          ("[general/negative]", "general_negative")),
}
_VALID_KEYS = {"guideline_factors": GUIDELINE_FACTOR_KEYS,
               "probation_factors": PROBATION_FACTOR_KEYS}


def _bad_factor_keys(calc: dict) -> str | None:
    """The response to hand back if a key is unknown, else None. Normalizes in place."""
    bad: list[tuple[str, list[str]]] = []
    for arg in ("guideline_factors", "probation_factors"):
        if arg not in calc:
            continue
        d = coerce_dict(calc[arg])
        if d is None:                       # a list, a string — leave it to the engine
            continue
        calc[arg] = d
        unknown = [str(k) for k in d if k not in _VALID_KEYS[arg]]
        if unknown:
            bad.append((arg, unknown))
    if not bad:
        return None
    lines = ["## status: bad_factor_key"]
    for arg, unknown in bad:
        lines.append(f"- {arg} 에 모르는 key 가 있습니다: {', '.join(f'`{k}`' for k in unknown)}."
                     " 그대로 두면 그 인자가 하나도 안 세어진 채 계산되므로 계산하지 않았습니다.")
    lines.append("- 응답의 그룹 헤더에 대응하는 key 는 아래와 같습니다."
                 " 헤더 아래 항목의 **문장을 그대로** 해당 list 에 넣어 다시 부르세요.")
    for arg, _u in bad:
        lines.append(f"### {arg}")
        lines += [f"- {header} → {key}" for header, key in _HEADER_KEY[arg]]
    return "\n".join(lines) + "\n"


def sentencing_analysis(
    ctx: RunContext[HarnessDeps],
    charge: ChargeArg,
    statute_choice: OptStrArg = None,
    branch_key: OptStrArg = None,
    reference_choice: OptStrArg = None,
    sg_category_id: int | str | list | None = None,
    offense_date: OptStrArg = None,
    is_attempted: bool = False,
    is_accessory: bool = False,
    is_solicitor: bool = False,
    statutory_modifications: list | dict | str | None = None,
    guideline_type: OptStrArg = None,
    guideline_factors: dict | str | list | None = None,
    probation_factors: dict | str | list | None = None,
    sentence_months: int | str | list | None = None,
    fine_amount: int | str | list | None = None,
    act_count: int | str | list = 1,
    reference_year: int | str | list | None = None,
) -> str:
    """죄명 하나로 양형 판단에 필요한 것을 한 번에 — 법정형·처단형 계산방법·양형기준 권고 형량범위와
    양형인자·집행유예 참작사유·근거 조문·실선고 분포. 인자를 더하면 그 사건의 처단형·권고형·선고형을 계산한다.

    언제:
    - 형량·구형·양형 의견을 판단해야 하는 국면에서 **결론 전에** 죄명만으로 한 번 부른다.
    - 유형과 양형인자가 사실관계로 정해진 뒤, 같은 도구를 계산 인자와 함께 다시 부른다.
      두 번째 응답은 기준 본문을 반복하지 않고 계산 결과만 싣는다.
    - 판례 본문은 precedent_search·precedent_dive, 조문 전문은 statute_lookup 으로 이어간다.

    규칙:
    - 죄명은 판결문 표기로 하나만 (예: 사기, 도로교통법위반(음주운전)). 정규화는 도구가 한다.
    - 경합 사안은 죄명별로 각각 호출한다. 죄명별 분포를 합산·평균해 경합범 분포로 만들지 않는다.
    - 조항·분기가 여럿이면 후보의 정량이 응답에 전부 실린다 — 되묻지 않으니 사실관계로 골라 쓴다.
    - 계산 인자의 값은 응답에 실린 목록의 표기를 그대로 쓰고 추측하지 않는다.
    - 권고 형량범위는 공식 기준의 범위이지 선고 예측이 아니며, 실선고 분포는 관측일 뿐이다.

    응답: markdown. 판례·조문 url 이 있으면 출처로 그대로 인용한다.

    Args:
      charge: 판결문 표기의 죄명 하나. 법률명만 알면 '<법률명>위반' 으로 그 법률의 죄명 후보를 받는다.
      statute_choice: 조항 후보가 여럿일 때 하나로 좁히는 조항 (예: "형법§257").
      branch_key: 분기 조항의 선택지 키 (예: "1", "3-1"). 응답 목록의 키를 그대로 쓴다.
      reference_choice: 가중·준용 조항의 원범죄 (예: "형법§347").
      sg_category_id: 같은 죄명이 여러 양형기준 카테고리에 걸릴 때 하나로 좁히는 id.
      offense_date: 행위 일자 (예: '2013.7.30'). 지정하면 행위시 조문·정량으로 계산한다
        — 형법 §1 ① "범죄의 성립과 처벌은 행위시의 법률에 의한다". 미지정이면 현행 기준.
      is_attempted: 미수 명시. 죄명 접미('살인미수')로도 자동 인식된다.
      is_accessory: 방조 명시 — 인식 규칙은 is_attempted 와 같다.
      is_solicitor: 교사 명시 — 인식 규칙은 is_attempted 와 같다.
      statutory_modifications: 형법 §56 가중·감경 사유 list. 예:
        [{"kind": "누범_가중"}]. 지정하면 처단형을 계산한다.
      guideline_type: 양형기준 범죄유형의 명칭 — 응답 목록의 따옴표 안 표기 그대로
        (예: "일반사기 2유형"). 지정하면 권고 형량범위를 계산한다.
      guideline_factors: 특별양형인자 dict. key 는 응답의 그룹 헤더 순서와 같은 넷뿐 —
        special_act_aggravators · special_act_mitigators · special_actor_aggravators ·
        special_actor_mitigators. 값은 목록의 문장 그대로 넣는다.
      probation_factors: 집행유예 참작사유 dict. key 는 4분면과 같은 넷뿐 —
        major_positive · major_negative · general_positive · general_negative.
      sentence_months: 검증할 선고형(자유형, 월). 지정하면 선고 가능 범위와 대조한다.
      fine_amount: 검증할 선고형(벌금, 원). sentence_months 와 같은 단계로 들어간다.
      act_count: 같은 죄명의 별개 행위 수. 2 이상이면 경합범 가중(§37 전단)을 자동 적용한다.
      reference_year: 비교 판례를 고를 때 가까이 볼 연도. 표본을 연도로 좁히지는 않는다.
    """
    charge = coerce_str(charge)
    if not charge or charge.strip() in ("{}", "None", "null"):
        return ("## status: missing_input\n"
                "- charge 인자 필요. 판결문 표기의 죄명 하나 (예: 사기, 도로교통법위반(음주운전)).")

    narrow = dict(sg_category_id=sg_category_id, offense_date=offense_date,
                  is_attempted=is_attempted, is_accessory=is_accessory,
                  is_solicitor=is_solicitor)
    calc = {k: v for k, v in (("statutory_modifications", statutory_modifications),
                              ("guideline_type", guideline_type),
                              ("guideline_factors", guideline_factors),
                              ("probation_factors", probation_factors),
                              ("sentence_months", sentence_months),
                              ("fine_amount", fine_amount)) if v not in (None, "", [], {})}
    act_count = coerce_int(act_count) or 1
    if act_count >= 2:
        calc["act_count"] = act_count
    if (bad := _bad_factor_keys(calc)) is not None:
        return bad

    cands = sentencing_candidates(
        charge,
        sg_category_id=coerce_int(sg_category_id),
        statute_choice=coerce_str(statute_choice), branch_key=coerce_str(branch_key),
        reference_choice=coerce_str(reference_choice), offense_date=coerce_str(offense_date),
        is_attempted=is_attempted, is_accessory=is_accessory, is_solicitor=is_solicitor)

    if cands:
        body = (_calc(ctx, charge, cands, narrow, calc) if calc
                else _brief(ctx, charge, cands, narrow, coerce_int(reference_year)))
        if body is not None:
            return body

    # No guideline attaches. Use the engine's own guidance, and add the
    # distribution when there is one. (A charge that did not resolve at all and an
    # official charge the guideline does not cover part ways here: the former has
    # no distribution either, and the status carries that difference.)
    guide = compute_sentencing_range(ctx, charge=charge, protocol=False, **narrow)
    status = _status_of(guide)
    if status in ("charge_numeric", "missing_input", "multiple_charges"):
        return guide
    stat_status, stat_md = _stat_section(ctx, charge, coerce_int(reference_year))
    if status in _NOT_IN_CORPUS and stat_status in ("ok", "low_n_grid"):
        guide = guide.replace(f"## status: {status}", "## status: no_guideline", 1)
    return _render(_split(guide) + [("", stat_md)])
