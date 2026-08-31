"""Read one judgment body in a sub-agent and answer a question about it.

A Korean judgment runs to tens of thousands of characters, and the caller
usually wants one fact out of it. Putting the body in the caller's context
spends that budget on text nobody reads twice, so the body goes to a
sub-agent instead and only its answer comes back. The sub-agent is injected
through ``HarnessDeps.dive_subagent``, so the model behind it is swappable.

Only ``content_md`` is sent. The stored summaries duplicate the body closely
enough that adding them cost context without improving answers.
"""
from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext

from ..config import case_url_base
from ..deps import HarnessDeps, open_db
from ..schemas import DIVE_SUMMARY_MAX_CHARS
from ._coerce import coerce_int, coerce_str
from ._dedup import dedup_guard


# Body truncation, head and tail. Judgment length is heavily skewed: p99 is
# 27K characters and p99.9 is 84K, but the longest run to 2.4M. A 60K
# threshold therefore passes 99.8% of judgments through untouched and only
# clips the extreme tail of the distribution.
#
# The tail half matters as much as the head: a Korean judgment states its
# disposition — the sentence actually imposed — at the very end, so a
# head-only clip would drop the one fact most questions are about.
DIVE_TEXT_HEAD = 40_000
DIVE_TEXT_TAIL = 20_000
DIVE_THRESHOLD = DIVE_TEXT_HEAD + DIVE_TEXT_TAIL
DIVE_QUESTION_MAX_CHARS = 1_000


def _clip_question(question: str) -> str:
    """Bound the question so a runaway prompt cannot crowd out the judgment."""
    q = question or ""
    if len(q) <= DIVE_QUESTION_MAX_CHARS:
        return q
    return q[: DIVE_QUESTION_MAX_CHARS - 1].rstrip() + "…"


def _cap_summary(summary: str) -> str:
    """Enforce the length contract at the boundary.

    The schema already caps it, but a substituted sub-agent may not go
    through that schema, so the response path checks again.
    """
    if len(summary) <= DIVE_SUMMARY_MAX_CHARS:
        return summary
    return summary[: DIVE_SUMMARY_MAX_CHARS - 1].rstrip() + "…"


def _truncate_for_dive(text: str) -> str:
    if len(text) <= DIVE_THRESHOLD:
        return text
    return text[:DIVE_TEXT_HEAD] + "\n\n[...중략...]\n\n" + text[-DIVE_TEXT_TAIL:]


def _build_body_sections(case: Any) -> tuple[list[str], list[str], bool]:
    """Case row -> (sections, sources_used, text_truncated)."""
    sections: list[str] = []
    sources: list[str] = []
    truncated = False

    body = (case["content_md"] or "").strip()
    if body:
        truncated = len(body) > DIVE_THRESHOLD
        sections.append(f"## 원문\n{_truncate_for_dive(body)}")
        sources.append("content_md")

    return sections, sources, truncated


def _format_response_md(resp: dict[str, Any]) -> str:
    """Response dict -> markdown-KV string, omitting null and false metadata."""
    status = resp.get("status", "ok")
    lines: list[str] = [f"## status: {status}"]
    if resp.get("message"):
        lines.append(f"- message: {resp['message']}")

    # Case metadata, emitted even for a miss so the caller can see which id
    # it asked about.
    case_id = resp.get("case_id")
    case_no = resp.get("case_number")
    case_name = resp.get("case_name")
    court_level = resp.get("court_level")
    year = resp.get("year")
    if case_id is not None or case_no or case_name:
        lines.append("## case")
        if case_id is not None:
            lines.append(f"- id: {case_id}")
            lines.append(f"- url: {case_url_base()}/cases/{case_id}")
        if case_no:
            lines.append(f"- case_no: {case_no}")
        if case_name:
            lines.append(f"- name: {case_name}")
        if court_level or year:
            lines.append(f"- court: {court_level or ''} {year or ''}".rstrip())
        if resp.get("reference_statute"):
            lines.append(f"- statute: {resp['reference_statute']}")
    # Only flagged when the body was actually clipped. It changes what a
    # negative answer means, so the caller has to be able to see it.
    if resp.get("text_truncated"):
        lines.append("## text_truncated: true")
        lines.append(f"- retained_text: 원문 앞 {DIVE_TEXT_HEAD}자 + 뒤 {DIVE_TEXT_TAIL}자")
    if resp.get("not_in_text"):
        lines.append("## not_in_text: true")
        if resp.get("text_truncated"):
            lines.append("- not_in_text_conclusive: false")
            lines.append("- note: 중간 원문이 생략되어 보존 구간에 없다는 뜻일 뿐, 전체 원문 부재를 확정하지 못함")
    if resp.get("summary"):
        # One marker, not four. The heading says the summary is generated and
        # cannot be quoted verbatim; repeating that in separate metadata
        # fields spent tokens saying the same thing again.
        lines.append("## summary — AI 생성 요약(직접인용 불가)")
        lines.append(resp["summary"])

    return "\n".join(lines)


@dedup_guard("precedent_dive")
async def precedent_dive(
    ctx: RunContext[HarnessDeps],
    case_id: int,
    question: str,
) -> str:
    """단건 판결·결정 본문 추출(sub-agent 위임) — search preview가 부족할 때 case id로
    호출하면 question에 답하는 300자 내외 생성 요약을 반환합니다(직접인용 아님).

    언제:
    - 한 case의 구체 판단 이유·결론·역할·피해규모 등이 preview에 없을 때.

    규칙:
    - question은 선택한 공개 판례 안에서 확인할 쟁점·항목만 쓰고 사용자 대화·첨부·문서·계정·
      세션 정보나 사적 사실을 넣지 마세요. 명백한 식별번호는 서버가 추가 삭제합니다.
    - text_truncated=true일 때 not_in_text는 보존된 앞·뒤 구간에 없다는 뜻일 뿐입니다.

    응답: markdown-KV. summary는 생성 요약이므로 직접인용하지 말고, 답에 쓴 판례는 반환
    url을 함께 제시하세요.

    Args:
      case_id: precedent_search 결과의 id 필드.
      question: 선택한 공개 판례 안에서 추출할 쟁점·항목.
    """
    case_id = coerce_int(case_id)
    question = coerce_str(question)
    if case_id is None or not question:
        return _format_response_md({"status": "missing", "case_id": case_id})
    question = _clip_question(question)

    conn = open_db()
    try:
        case = conn.execute(
            """
            SELECT case_number, case_name, court_name, court_level,
                   COALESCE(decision_year, case_year) AS year,
                   reference_statute,
                   content_md
            FROM prec_cases WHERE id = ?
            """,
            (case_id,),
        ).fetchone()
    finally:
        conn.close()

    if case is None:
        return _format_response_md({"status": "missing", "case_id": case_id})

    sections, _sources, truncated = _build_body_sections(case)
    if not sections:
        return _format_response_md({
            "status": "ok",
            "case_id": case_id,
            "case_number": case["case_number"],
            "summary": "관련 정보 없음 (본문 결손)",
            "not_in_text": True,
        })

    body_text = "\n\n".join(sections)
    user_prompt = (
        f"## 질문\n{question}\n\n"
        f"## 판결문 (사건번호: {case['case_number']}, "
        f"{case['court_name'] or '?'}, {case['year'] or '?'}년)\n\n"
        f"{body_text}"
    )

    result = await ctx.deps.dive_subagent.run(user_prompt, usage=ctx.usage)
    dive = result.output  # DiveResult

    return _format_response_md({
        "status": "ok",
        "case_id": case_id,
        "case_number": case["case_number"],
        "case_name": case["case_name"],
        "court_level": case["court_level"],
        "year": case["year"],
        "reference_statute": case["reference_statute"],
        "summary": _cap_summary(dive.summary),
        "not_in_text": dive.not_in_text,
        "text_truncated": truncated,
    })
