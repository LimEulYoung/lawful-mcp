"""sentencing_analysis — the sentencing tool: what the record holds on the charges of a case, in one response.

For a list of charges it shows, without computing anything for the case:
  - the rules of the judgement (the decision flow, the adjusted statutory range, several charges) — once;
  - for each charge, how its name was read, its statutory penalty per provision, branch or source crime
    (`statutory_penalty`), and the guideline of each category those provisions fall under (`sentencing_guideline`);
  - the articles' text (`statute_lookup`);
  - the first-instance sentences of the judgments holding the charges, grouped by how close their charges are
    (`sentence_statistics`).
The model applies the rules to the facts. Each part is read as data and written here, once; nothing is parsed back.

The response is not shortened: over the charge sets of the research database's first-instance defendants it is
about 11,000 characters at the median for 2-6 charges and at most about 33,000.
"""
from __future__ import annotations

import json
import re
from typing import Annotated

from pydantic import BeforeValidator
from pydantic_ai import RunContext

from ..deps import HarnessDeps, open_db
from . import sentencing_guideline as guideline
from ._coerce import BAD_DATE_MESSAGE, coerce_list, coerce_str, to_iso_date, unreadable_date
from .sentence_statistics import conviction_counts, statistics
from . import statutes
from ._charge_index import official_index
from .statutory_penalty import Penalty, Provision, Reading, article_label, guideline_articles, read_charge

MAX_CHARGES = 6   # charges read per call; the rest are named as left out


# ---------------------------------------------------------------- the rules
# Sources: the Supreme Court Sentencing Commission's 「양형기준 해설」 (how the guidelines are applied, the rules for
# evaluating factors, multiple offences), the 2025 guideline book's [다수범죄 처리기준] and [집행유예 참작사유의
# 평가원칙], and the Criminal Act as in the snapshot. Only the rules that answers use are kept.
#
# ⚠ Rule 2 says that choosing a fine leaves the guideline out: without it the model picks from the imprisonment table
#   in front of it.
# ⚠ Rule 3 tells a repeat offence from a post-judgment concurrent offence and puts the latter under statutory
#   mitigation: without it the model invents a ground by that name.
# ⚠ A provision aggravating another offence's penalty (상습범, 특가법 …) is already applied in the penalty table's
#   「원범죄의 형 ×… 반영」 rows — rule 3 says so, or the model aggravates twice.
RULES = """## 양형 판단 규칙
법정형 → 형종 선택 → 처단형 → 권고 형량범위 → 선고형 → 집행유예 순으로 좁힌다. 처단형은 선고형이 넘을 수 없는
한계이고, 권고 형량범위는 구속력이 없는 기준이다(벗어나면 판결서에 양형의 이유를 적는다).
1. 법정형 — 아래 표의 형. 하한이 없으면(「N월 이하」·「N원 이하」) 징역·금고 1월(형법 §42), 벌금 5만 원(§45)이다.
   상한이 없으면(「N월 이상」) 30년(§42 — 2010. 10. 16. 전 행위는 15년)이고, 「원범죄의 형 ×k 반영」 줄이면 그 상한에
   ×k 한다(50년, 2010. 10. 16. 전에는 25년까지).
2. 형종 선택 — 선택형이면 사건의 경중으로 자유형과 벌금 중 하나를 고른다. 벌금을 고르면 양형기준을 적용하지
   않는다(교통·선거·스토킹·동물보호법 범죄만 벌금 기준을 둔다).
3. 처단형 — 고른 형에 형법 §56 순서로 가중·감경한다.
   · 본조 가중: 법정형 표의 줄에 「원범죄의 형 ×… 반영」이 있으면 이미 반영된 것이니 다시 가중하지 않는다.
   · 누범(§35 — 금고 이상 형의 집행을 마치거나 면제된 뒤 3년 안의 금고 이상 죄): 장기 ×2. 앞 판결 확정 전에 범한
     죄는 누범이 아니라 사후적 경합범이다(아래 6).
   · 법률상 감경(미수 §25 ②, 방조 §32 ②, 심신미약 §10 ②, 자수 §52, 사후적 경합범 §39 ① 등): 장기·단기와 벌금의
     상한·하한 ×½.
   · 경합범 가중(§38): 가장 무거운 죄의 장기·다액 ×1.5까지, 각 죄의 장기·다액을 합한 것을 넘지 못한다. 하한은 각
     죄의 단기 중 가장 높은 것. 가장 무거운 죄가 사형·무기면 그 형으로, 형의 종류가 다르면 병과한다.
   · 작량감경(§53): 법률상 감경과 같이 ×½.
4. 권고 형량범위 — 자유형을 고르면 유형 목록에서 유형 하나를 고르고, 특별양형인자로 영역(감경·기본·가중)을 정한다.
   · 행위인자끼리, 행위자·기타인자끼리 가중과 감경을 개수로 상쇄한다. 남은 인자가 없으면 기본영역, 한 방향뿐이면
     그 방향. 섞여 남으면 행위인자 쪽이 많거나 같을 때 그 방향, 아니면 특별양형인자를 종합해 정한다.
   · 상쇄하고 남은 특별가중인자가 2개 이상이면 가중영역 상한을, 특별감경인자가 2개 이상이면 감경영역 하한을 1/2
     넓힌다. 일반양형인자는 영역 결정에 쓰지 않는다.
   · 권고 범위가 처단형을 벗어나면 처단형의 상한·하한을 따른다.
5. 선고형 — 권고 범위(벌금이면 처단형) 안에서 특별·일반 양형인자를 종합해 정한다. 집행유예는 3년 이하 징역·금고나
   500만 원 이하 벌금일 때 할 수 있고(§62 ① — 금고 이상 판결 확정 뒤 집행 종료·면제 후 3년 안에 범한 죄는 안 된다),
   양형기준의 집행유예 참작사유로 판단하되 주요사유를 일반사유보다 중하게 본다.
6. 죄가 여럿이면 — 한 행위가 여러 죄(상상적 경합, §40)면 가장 무거운 죄의 형으로 처벌한다. 판결이 확정된 죄와 그
   전에 범한 죄(사후적 경합범, §37 후단)는 양형기준을 적용하지 않고 §39 ①로 정한다(감경·면제할 수 있다). 확정되지
   않은 여러 죄(§37 전단)는 처단형을 위 3의 경합범 가중으로, 권고 범위를 다수범죄 처리기준으로 정한다:
   · 기본범죄(형이 가장 중한 죄, 다른 죄의 권고 상한이 더 높으면 그 죄)의 상한에 다른 죄 상한의 1/2을 더한다
     (3개 이상이면 상한이 높은 두 죄의 1/2과 1/3). 하한은 각 죄의 하한 중 가장 높은 것.
   · 양형기준이 한 범죄로 다루는 것(사기에 딸린 문서 위조·행사 등)은 따로 세지 않는다.
   · 사기·횡령·배임처럼 이득액 등으로 유형을 나누는 죄가 여러 건이면 금액을 합산해 유형을 정한다(유형이 오르면
     하한을 감경한다)."""


# ---------------------------------------------------------------- arguments
def _coerce_optional_str_arg(v: object) -> object:
    """If a list, number or object arrives for an optional string argument, fold it into a string **before
    validation** (OptStrArg) instead of failing the call: a list becomes its element (several are comma-joined), a
    number str, a dict JSON. None and strings are left untouched."""
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


# The optional string argument (offense_date) goes out as `anyOf[string, null]`, default null. A schema that also
# allows an array draws arrays such as `[1]` and `[{"start": …}]` from models; whatever still arrives is folded above.
OptStrArg = Annotated[str | None, BeforeValidator(_coerce_optional_str_arg)]


def _charges_arg(v: object) -> object:
    """Fold what models send for a list of names into one: a bare string is one charge, a JSON array string is
    unpacked, non-string items become strings. Pydantic then checks list[str]."""
    if isinstance(v, str):
        parsed = coerce_list(v) if v.lstrip()[:1] == "[" else None
        v = parsed if isinstance(parsed, list) else [v]
    if isinstance(v, (list, tuple)):
        return [x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in v if x is not None]
    return v


ChargesArg = Annotated[list[str], BeforeValidator(_charges_arg)]


# ---------------------------------------------------------------- one charge
_SUFFIX_RULES = {
    "미수": "죄명의 '미수': 아래 법정형은 기수의 형이다 — 미수는 형법 §25 ② 임의적 감경(규칙 3 처단형 — 법률상 감경)",
    "교사": "죄명의 '교사': 아래 법정형은 정범의 형이다 — 교사는 형법 §31 ① 정범과 같은 형",
    "방조": "죄명의 '방조': 아래 법정형은 정범의 형이다 — 방조는 형법 §32 ② 필요적 감경(규칙 3 처단형 — 법률상 감경)",
}
_NO_RANGE = "매핑표에 형의 범위가 없다(다른 조·항의 예에 따르는 조항 등) — 조문 원문으로 정한다"


def _span(lo: int | None, hi: int | None, unit: str) -> str:
    def f(v):
        return f"{v:,}" if unit == "원" else f"{v}"
    if lo is None:
        return f"{f(hi)}{unit} 이하"
    if hi is None:
        return f"{f(lo)}{unit} 이상"
    return f"{f(lo)}~{f(hi)}{unit}"


def _formula(fm: dict) -> str:
    """{"base_label": "부가세액", "low_mult": 2, "high_mult": 5} → 「부가세액의 2배 ~ 5배」."""
    def n(v):
        return f"{int(v)}" if isinstance(v, float) and v.is_integer() else f"{v}"
    base, lo, hi = fm.get("base_label", "?"), fm.get("low_mult"), fm.get("high_mult")
    opt = " (임의 병과)" if fm.get("is_optional") else ""
    if lo is None and hi is None:
        return base + opt
    if lo == hi or hi is None:
        return f"{base}의 {n(lo)}배{opt}"
    if not lo:
        return f"{base}의 {n(hi)}배 이하{opt}"
    return f"{base}의 {n(lo)}배 ~ {n(hi)}배{opt}"


def _penalty_text(p: Penalty) -> str:
    """「자유형 12~60월 · 벌금 5,000,000~20,000,000원 · 무기」 — the kinds the provision has, each with its range."""
    parts = []
    if p.imp_min is not None or p.imp_max is not None:
        parts.append(f"자유형 {_span(p.imp_min, p.imp_max, '월')}")
    elif "imprisonment" in p.kinds:
        parts.append("자유형(기간은 조문 원문)")
    if p.fine_min is not None or p.fine_max is not None:
        parts.append(f"벌금 {_span(p.fine_min, p.fine_max, '원')}")
    elif p.fine_formula:
        parts.append(f"벌금 {_formula(p.fine_formula)}")
    elif "fine" in p.kinds:
        parts.append("벌금(금액은 조문 원문)")
    parts += [k for k, on in (("무기", p.life), ("사형", p.death)) if on]
    return " · ".join(parts)


def _row(p: Provision) -> str:
    """One provision's penalty, or why the table gives none."""
    text = _penalty_text(p.penalty) if p.penalty else ""
    if not text:
        return f"법정형을 정하지 못했다 — {p.reason or _NO_RANGE}"
    if p.times:
        text = (f"(원범죄의 형 ×{p.times:g} 반영) " if p.times != 1 else "(원범죄의 형과 같다) ") + text
    return text


def _penalty_section(provisions: list[Provision], offense_date: str | None) -> tuple[str, str]:
    """The statutory penalty of every provision, grouped by article; an article outside the guidelines says so."""
    by_article: dict[str, list[Provision]] = {}
    for p in provisions:
        by_article.setdefault(p.article, []).append(p)
    lines, count = [], 0
    for article, ps in by_article.items():
        # A provision the guidelines leave out (its mapping row has no category): 도로교통법 §148의2 ①, …
        head = f"- {article}" + (" — 양형기준 적용범위 밖" if all(p.category_id is None for p in ps) else "")
        # One line per distinct penalty: an article under two guidelines repeats its rows. Rows of one article
        # without a branch or source crime are told apart by their charge name when their penalties differ.
        named = len({_row(p) for p in ps if not p.where}) > 1
        rows = list(dict.fromkeys((p.where or (p.name if named else ""), _row(p)) for p in ps))
        count += len(rows)
        if len(rows) == 1 and not rows[0][0]:
            lines.append(f"{head}: {rows[0][1]}")
        else:
            lines += [head, *(f"  · {label}: {text}" if label else f"  · {text}" for label, text in rows)]
    if count > 1:
        lines.insert(0, "사실관계로 아래 중 하나가 정해진다. 여기 있는 것이 전부다.")
    if offense_date:
        iso = to_iso_date(offense_date) or ""
        lines.insert(0, f"행위 시점({iso[:4]}-{iso[4:6]}-{iso[6:]})에 시행된 문언의 형이다.")
    return "## 법정형", "\n".join(lines)


def _article_of(index, name: str) -> str:
    """The article an official name punishes under (대검 죄명표), or its statute for "<statute>위반"."""
    e = index.entries.get(name)
    if e is None or not e.law or not e.article:
        return e.law if e is not None and e.law else name.removesuffix("위반")
    return article_label({"statute_name": e.law, "article_no_num": e.article, "article_branch": 0,
                          "paragraph": str(e.hang) if e.hang else None})


def _candidates(conn, r: Reading, exclude: frozenset[int]) -> list[str]:
    """Names to choose from — a statute's charges, or official names spelled like an unknown one — each with its
    guideline articles (or its article) and first-instance convictions."""
    res = r.resolution
    names = list(res.candidates)
    arts = guideline_articles(conn, names)
    counts = conviction_counts(conn, names, exclude)
    if res.kind == "similar":   # all of them, most similar first
        shown = names
        head = (f"'{r.asked}' 은 공식 죄명(대검 죄명표)에 없는 표기다. 아래는 글자가 비슷한 공식 죄명이다 — 뜻이 다를 수 "
                "있으니 사실관계에 맞는 것이 있을 때만 그 이름을 넣어 다시 부르고, 없으면 판결문 죄명 표기(법률명만 알면 "
                "'<법률명>위반')로 다시 부른다.")
    else:                       # those with a guideline or judgments, the most convicted first
        shown = sorted((n for n in names if n in arts or counts.get(n)), key=lambda n: (-counts.get(n, 0), n not in arts))
        if res.kind == "exact" and res.statute:
            head = (f"'{res.key}' 은 죄명표의 괄호 없는 표기(그 밖의 조항 위반)라 그 이름의 양형기준이 없다. 아래 세부 죄명 중 "
                    "사실관계에 맞는 것을 죄명 목록에 넣어 다시 부른다.")
        elif res.sub and res.statute:
            head = (f"'({res.sub})' 은 {res.statute} 의 공식 죄명(대검 죄명표)이 아니다. 아래 공식 죄명 중 사실관계에 맞는 "
                    "것을 죄명 목록에 넣어 다시 부른다.")
        elif res.statute:
            head = f"법률명만으로는 죄가 정해지지 않는다. 아래 {res.statute} 의 공식 죄명 중 사실관계에 맞는 것을 넣어 다시 부른다."
        else:
            head = f"'{r.asked}' 은 아래 죄명들의 부속 죄명이다. 적용 법조에 맞는 것을 넣어 다시 부른다."
    lines = [f"- {head}"]
    if res.suffix_stripped:
        lines.append(f"- 접미 '{res.suffix_stripped}' 를 떼고 찾았다 — 미수·교사·방조는 아래 죄명 끝에 붙여 다시 부른다.")
    if not shown:
        return lines + [f"- 이 {len(names)}개 죄명은 모두 양형기준이 없고 1심 유죄 판결도 없다."]
    index = official_index(conn)
    lines.append("- 이름 — 양형기준 조항(없으면 조문) · 1심 유죄 판결 수")
    for n in shown[:20]:
        g = f"양형기준 {', '.join(arts[n])}" if n in arts else f"{_article_of(index, n)} · 양형기준 없음"
        lines.append(f"- {n} — {g} · {counts.get(n, 0)}건")
    if len(shown) > 20:
        lines.append(f"- (외 {len(shown) - 20}개)")
    return lines


def _guidance(conn, r: Reading, exclude: frozenset[int]) -> list[str]:
    """What a name without provisions is, and what to do."""
    if r.kind == "numeric":
        lines = ["- 숫자가 아니라 죄명 문자열을 넣는다. 예: charges=['강제추행', '도로교통법위반(음주운전)']."]
        if all(t.isdigit() and int(t) < 10 for t in r.tokens):
            lines.append("- 리스트 인덱스(0, 1 등)는 죄명이 아니다. 앞서 안내된 죄명 문자열을 넣는다.")
        return lines + ["- 정확한 죄명 표기를 모르면 '<법률명>위반' 으로 그 법률의 죄명 후보를 받는다."]
    if r.kind == "candidates":
        return _candidates(conn, r, exclude)
    res = r.resolution
    if r.kind == "no_guideline":
        if res.law_repealed:
            return [f"- '{res.key}' 의 법률은 폐지되었다 — 이 이름으로는 양형기준이 없다. 후속 법률의 현행 명칭은 "
                    "`statute_lookup` 으로 확인한다."]
        if res.status == "구법":
            return [f"- '{res.key}' 은 옛 죄명표 표기라 양형기준이 없다"
                    + (f" — 현행 죄명 '{res.successor}' 으로 다시 부른다." if res.successor
                       else " — 현행 죄명은 `statute_lookup` 으로 확인한다.")]
        return [f"- '{res.key}' 은 공식 죄명(대검 죄명표)이지만 양형기준(48개 범죄군)에 없다 — 권고형 없음. "
                "법정형·조문은 `statute_lookup` 으로 조회한다. 실선고는 아래에 싣는다."]
    return [f"- '{r.asked}' 은 공식 죄명(대검 죄명표)에 없는 표기다. 판결문 죄명 표기(예: 특수상해, "
            "도로교통법위반(음주운전))로 다시 부르고, 법률명만 알면 '<법률명>위반' 으로 그 법률의 죄명 후보를 받는다. "
            "형법·군형법의 죄명에는 '위반'이 붙지 않는다(형법위반 → 사기, 공무집행방해). "
            "죄가 여럿이면 죄명마다 목록의 원소로 나눠 넣는다.",
            "- 양형기준이 없는 죄일 수 있다. 법정형은 `statute_lookup` 으로 조회한다."]


def _charge_head(conn, r: Reading, offense_date: str | None, exclude: frozenset[int]) -> list[tuple[str, str]]:
    """One charge: how its name was read, and its penalty table — or, for a name without provisions, what it is."""
    head = [f"- {n}" for n in r.notes] + [f"- {_SUFFIX_RULES[s]}" for s in r.suffixes]
    if r.kind != "ok":
        return [(f"## 죄: {r.asked}", "\n".join(head + _guidance(conn, r, exclude)))]
    return [(f"## 죄: {r.asked}", "\n".join(head)), _penalty_section(r.provisions, offense_date)]


def _guidelines(conn, r: Reading, shown: dict[int, str]) -> list[tuple[int, str, str | None]]:
    """The guideline categories the charge's provisions fall under, as (category, heading, the earlier charge that
    already showed it or None) — a guideline is shown once."""
    categories: dict[int, list[str]] = {}
    for p in r.provisions:
        if p.category_id is not None and p.article not in categories.setdefault(p.category_id, []):
            categories[p.category_id].append(p.article)
    out = []
    for cat, articles in categories.items():
        title = f"## 양형기준 — {guideline.name(conn, cat)}" + (f" ({' · '.join(articles)})" if len(categories) > 1
                                                             else "")
        out.append((cat, title, shown.get(cat)))
        shown.setdefault(cat, r.asked)
    return out


def _guideline_sections(conn, guides: list[tuple[int, str, str | None]]) -> list[tuple[str, str]]:
    out = []
    for cat, title, same in guides:
        if same:
            out.append((f"{title} — 위 {same} 항목과 같다", ""))
        else:
            out += [(title, guideline.source(conn, cat)), *guideline.sections(conn, cat)]
    return out


def _not_counted(conn, readings: list[Reading], exclude: frozenset[int]) -> list[str]:
    """Names the actual sentences cannot take — not an official charge name. A name whose penalty was found (the
    table's own spelling, 마약류관리에관한법률위반(필로폰)) gets the official names judgments use for its statute."""
    lines = [f"- 실선고에 넣지 못한 이름(판결 죄명표의 공식 죄명이 아님): {', '.join(r.asked for r in readings)}"]
    for r in readings:
        names = list(r.official.candidates) if r.kind == "ok" and r.official else []
        counts = conviction_counts(conn, names, exclude) if names else {}
        top = sorted((n for n in names if counts.get(n)), key=lambda n: -counts[n])[:5]
        if top:
            lines.append(f"- {r.asked}: 판결이 쓰는 이름 — {' · '.join(f'{n} {counts[n]}건' for n in top)}. "
                         "그 이름을 목록에 넣으면 실선고가 나온다.")
    return lines


def _same(a: Reading, b: Reading) -> bool:
    """Two spellings of one charge (사기 · 사기죄): the same table key, suffixes and official name."""
    return (a.kind == b.kind == "ok" and (a.key, a.suffixes) == (b.key, b.suffixes)
            and (a.official.key if a.official else None) == (b.official.key if b.official else None))


def _render(sections: list[tuple[str, str]]) -> str:
    out = []
    for header, body in sections:
        if header:
            out.append(header)
        body = body.strip("\n")
        if body:
            out.append(body)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


# ---------------------------------------------------------------- the tool
def sentencing_analysis(
    ctx: RunContext[HarnessDeps],
    charges: ChargesArg,
    offense_date: OptStrArg = None,
) -> str:
    """양형 자료 — 사건의 죄명 목록을 받아 판단 규칙(처단형, 권고영역, 집행유예, 여러 죄), 죄마다의 법정형과
    양형기준(범죄유형별 권고 형량범위, 특별·일반 양형인자, 집행유예 참작사유)과 근거 조문, 그리고 그 죄들이 든
    1심 판결의 실제 선고와 비교 판례를 한 응답으로 돌려줍니다.

    - 사건의 죄를 모두 한 목록으로 넣습니다.

    Args:
      charges: 판결문 표기 죄명 목록(예: ["도로교통법위반(음주운전)", "도로교통법위반(무면허운전)"]). 미수·교사·방조는
        죄명 끝에 붙입니다(예: 사기미수). 법률명만 알면 '<법률명>위반'으로 그 법률의 죄명 후보를 받습니다.
      offense_date: 행위 일자(예: '2013.7.30'). 주면 그 시점의 조문·법정형을, 없으면 현행을 보여 줍니다.
    """
    asked = _charges_arg(charges) if charges is not None else []   # direct callers skip the schema validator
    names = list(dict.fromkeys(s for s in (coerce_str(x) for x in (asked if isinstance(asked, list) else [asked]))
                               if s and s.strip() not in ("{}", "None", "null")))
    if not names:
        return ("## status: missing_input\n"
                "- charges 인자 필요. 판결문 표기의 죄명 목록 (예: [\"사기\"], [\"도로교통법위반(음주운전)\"]).")
    offense_date = coerce_str(offense_date)
    if unreadable_date(offense_date):
        return f"## status: bad_offense_date\n- {BAD_DATE_MESSAGE}\n- offense_date: {offense_date}"
    offense_iso = to_iso_date(offense_date)
    exclude = getattr(getattr(ctx, "deps", None), "exclude_case_ids", None) or frozenset()

    conn = open_db()
    try:
        readings: list[Reading] = []
        same: list[str] = []
        for name in names[:MAX_CHARGES]:
            r = read_charge(conn, name, offense_date)
            first = next((x for x in readings if _same(x, r)), None)
            if first:
                same.append(f"{name} → {first.asked}")
            else:
                readings.append(r)
        # The governing articles of every charge, statute by statute, each once.
        wanted: dict[int, list[tuple[int, int]]] = {}
        for p in (p for r in readings for p in r.provisions if p.statute_id):
            if (p.article_num, p.branch) not in wanted.setdefault(p.statute_id, []):
                wanted[p.statute_id].append((p.article_num, p.branch))
        # A name still to be chosen (a statute's charges, names like an unknown one) gets its list only: no
        # statistics for the bare statute's mixed pool, and no rules unless another name is settled.
        settled = [r for r in readings if r.kind != "candidates"]
        officials = [r.official for r in settled if r.official and r.official.kind == "exact"]
        stat_status, stats = (statistics(conn, officials, exclude=exclude,
                                         reference_year=int(offense_iso[:4]) if offense_iso else None)
                              if officials else ("no_data", []))
        missing = [r for r in settled if r.kind != "numeric" and not (r.official and r.official.kind == "exact")]
        if missing:
            stats = [("## 실선고", "\n".join(_not_counted(conn, missing, exclude))), *stats]
        shown: dict[int, str] = {}
        heads = [_charge_head(conn, r, offense_date, exclude) for r in readings]
        guides = [_guidelines(conn, r, shown) for r in readings]

        kinds = [r.kind for r in readings]
        status = ("ok" if "ok" in kinds else "no_guideline" if stat_status in ("ok", "low_n_grid")
                  else "candidates" if "candidates" in kinds else "charge_numeric" if set(kinds) == {"numeric"}
                  else "not_found")
        head = [f"## status: {status}", f"## 죄명: {', '.join(r.asked for r in readings)}"]
        if same:
            head.append(f"- 같은 죄로 읽은 이름: {', '.join(same)}")
        if names[MAX_CHARGES:]:
            head.append(f"- 한 번에 죄명 {MAX_CHARGES}개까지 읽는다 — 빠진 죄명: {', '.join(names[MAX_CHARGES:])}")
        rules = [RULES] if status in ("ok", "no_guideline") else []
        out = [("", "\n".join(head)), *(("", r) for r in rules)]
        for charge, guide in zip(heads, guides):
            out += charge + _guideline_sections(conn, guide)
        for sid, articles in wanted.items():
            out += statutes.article_sections(conn, sid, articles, offense_date)
        return _render(out + stats)
    finally:
        conn.close()
