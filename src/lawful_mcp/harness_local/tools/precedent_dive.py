"""precedent_dive — hand one judgment's text to an isolated sub-agent.

The text is not put into the main LLM context; it goes to a sub-agent as its prompt. The sub-agent
model is injected through `HarnessDeps.dive_subagent`, so it can be swapped.

Only `content_md` is given to the sub-agent — `summary`/`generated_summary` are left out: they largely
duplicate the text, and Supreme Court summaries run to 12.5K characters.
"""
from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai import RunContext

from ..deps import HarnessDeps, open_db
from ._coerce import coerce_int, coerce_str
from .precedent_search import _statute_line, court_label


# Text truncation — the guard against outliers in the sub-agent's context (texts run to 1.3M characters). A Korean
# judgment puts the disposition (which states the sentence) at the end, so the tail is kept.
# The value follows the corpus, not the model: text length is p99 25K and p99.9 71K characters, so 60K keeps 99.9%
# of the texts whole. Revisit only for a dive model whose context window is below 60K tokens.
DIVE_TEXT_HEAD = 40_000
DIVE_TEXT_TAIL = 20_000
DIVE_THRESHOLD = DIVE_TEXT_HEAD + DIVE_TEXT_TAIL
DIVE_QUESTION_MAX_CHARS = 1_000

# Upper bound on waiting for the summary (seconds). The bound is the tool's: pydantic-ai's default HTTP client
# allows a 600 s read and the OpenAI SDK retries a timeout twice, so one dive could otherwise hang for 30 minutes.
# On timeout the judgment is returned as unread (`status: timeout`). Keep it out of the dive model's settings, so
# there is one bound.
DIVE_TIMEOUT_S = 15


def _clip_question(question: str) -> str:
    """Cap the extraction question's length, `…` included — keeps a runaway prompt from swallowing the
    sub-agent's input."""
    if len(question) <= DIVE_QUESTION_MAX_CHARS:
        return question
    return question[: DIVE_QUESTION_MAX_CHARS - 1].rstrip() + "…"


def _truncate_for_dive(text: str) -> str:
    if len(text) <= DIVE_THRESHOLD:
        return text
    return text[:DIVE_TEXT_HEAD] + "\n\n[...중략...]\n\n" + text[-DIVE_TEXT_TAIL:]


def _format_response_md(resp: dict[str, Any]) -> str:
    """precedent_dive response dict → markdown-KV string (null/false fields omitted)."""
    status = resp.get("status", "ok")
    lines: list[str] = [f"## status: {status}"]
    if resp.get("message"):
        lines.append(f"- message: {resp['message']}")

    # case metadata (shown even when status=missing or only case_id is present)
    case_id = resp.get("case_id")
    case_no = resp.get("case_number")
    case_name = resp.get("case_name")
    court = resp.get("court")
    if case_id is not None or case_no or case_name:
        lines.append("## case")
        if case_id is not None:
            lines.append(f"- id: {case_id}")
            lines.append(f"- url: https://lawful.crow-tit.com/cases/{case_id}")
        if case_no:
            lines.append(f"- case_no: {case_no}")
        if case_name:
            lines.append(f"- name: {case_name}")
        if court:
            lines.append(f"- court: {court}")
        if resp.get("reference_statute"):
            lines.append(f"- statute: {_statute_line(resp['reference_statute'])}")
    # body truncated flag — shown only when the text was cut (lets the LLM judge how reliable the dive is)
    if resp.get("text_truncated"):
        lines.append("## text_truncated: true")
        lines.append(f"- retained_text: 원문 앞 {DIVE_TEXT_HEAD}자 + 뒤 {DIVE_TEXT_TAIL}자")
    if resp.get("not_in_text"):
        lines.append("## not_in_text: true")
        if resp.get("text_truncated"):
            lines.append("- not_in_text_conclusive: false")
            lines.append("- note: 중간 원문이 생략되어 보존 구간에 없다는 뜻일 뿐, 전체 원문 부재를 확정하지 못함")
    if resp.get("summary"):
        # The result has two strands: judgment lines the sub-agent carried over verbatim as `> ` (holdings,
        # reasoning, citation notation) and a summary of the rest. The text_kind line marks only the `> ` lines as
        # original text, so the summary is not read as quotable. Code does not check the `> ` lines against the
        # judgment; the sub-agent's instructions carry that.
        lines.append("## 판결문 확인 결과")
        lines.append("- text_kind: '> ' 로 시작하는 줄은 판결문 원문, 나머지는 요약")
        lines.append(resp["summary"])

    return "\n".join(lines)


async def precedent_dive(
    ctx: RunContext[HarnessDeps],
    case_id: int,
    question: str,
) -> str:
    """판결문 읽기 — precedent_search 결과의 판결 하나를 본문에서 읽어, question에 답하는 내용을 1,000자
    안팎으로 돌려줍니다.

    Args:
      case_id: precedent_search 결과의 id.
      question: 이 판결에서 확인할 쟁점·항목.
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
            SELECT case_number, case_name, court_name, court_level, decision_date,
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
    court = court_label(case["court_name"], case["court_level"], case["decision_date"], case["year"])

    body = (case["content_md"] or "").strip()
    if not body:
        return _format_response_md({
            "status": "ok",
            "case_id": case_id,
            "case_number": case["case_number"],
            "court": court,
            "summary": "관련 정보 없음 (본문 결손)",
            "not_in_text": True,
        })

    user_prompt = (
        f"## 질문\n{question}\n\n"
        f"## 판결문 (사건번호: {case['case_number']}, {court or '?'})\n\n"
        f"## 원문\n{_truncate_for_dive(body)}"
    )

    # The sub-agent keeps its own usage, so its requests do not count toward the main model's request limit.
    failure = None
    try:
        result = await asyncio.wait_for(ctx.deps.dive_subagent.run(user_prompt), timeout=DIVE_TIMEOUT_S)
    except asyncio.TimeoutError:
        failure = ("timeout", f"{DIVE_TIMEOUT_S:.0f}초 안에 본문 요약이 오지 않았습니다")
    except Exception as e:  # noqa: BLE001 — provider errors (429, 5xx, malformed output) fail open like the timeout
        failure = ("error", f"본문 요약 요청이 실패했습니다({type(e).__name__})")
    if failure:
        # fail-open — only the summary failed to arrive, so **the tool did not die; the text went unread**.
        # Saying so in both status and message lets the model tell this apart from "read it and found nothing".
        return _format_response_md({
            "status": failure[0],
            "case_id": case_id,
            "case_number": case["case_number"],
            "case_name": case["case_name"],
            "court": court,
            "reference_statute": case["reference_statute"],
            "message": (f"{failure[1]} — 이 판례의 "
                        "본문은 읽지 못한 것으로 취급하고, 판례 검색 발췌로 답하거나 다른 "
                        "판례를 확인하세요."),
        })
    dive = result.output  # DiveResult; its validator caps the summary length

    return _format_response_md({
        "status": "ok",
        "case_id": case_id,
        "case_number": case["case_number"],
        "case_name": case["case_name"],
        "court": court,
        "reference_statute": case["reference_statute"],
        "summary": dive.summary,
        "not_in_text": dive.not_in_text,
        "text_truncated": len(body) > DIVE_THRESHOLD,
    })
