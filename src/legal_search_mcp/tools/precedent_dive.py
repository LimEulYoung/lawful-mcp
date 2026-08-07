"""precedent_dive — 단건 판례 본문 격리 sub-agent 위임.

본문을 메인 LLM 컨텍스트에 넣지 않고 sub-agent에 prompt로 전달. sub-agent 모델은
`HarnessDeps.dive_subagent`로 주입되어 swap 가능.

`content_md`만 sub-agent에 주입 — `summary`/`generated_summary`는 대법원 outlier
(max 17K chars)에서 token 한계 압박, 정보도 본문과 중복이 커서 제외.
"""
from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext

from ..config import case_url_base
from ..deps import HarnessDeps, open_db
from ..schemas import DIVE_SUMMARY_MAX_CHARS
from ._coerce import coerce_int, coerce_str
from ._dedup import dedup_guard


# 본문 truncation — sub-agent 컨텍스트 outlier 방어선 (max 2.4M chars 케이스 존재).
# content_md에만 적용. summaries는 짧아서 truncation 불필요.
# 한국 판결문은 주문(형량 명시)이 본문 끝에 있어 tail 필수.
# dive 모델 = DeepSeek V4 Flash (1M-token context) → 컨텍스트는 사실상 무제약.
# 실제 제약은 prefill 비용·레이턴시뿐이라 head 40K + tail 20K chars 로 설정:
# prec_cases 165K건 본문 길이 분포 p99=27K·p99.9=84K → 60K threshold 면 99.8% 무손실.
# worst prefill ~44K tokens (한국어 worst 1.36 chars/token, 1M 컨텍스트의 4% ·
# input $0.14/M → ~$0.006/call). 극단 outlier(최대 2.4M chars)만 head/tail 로 방어.
# (이전: Gemma max_model_len=16384 대응 head 12K + tail 6K. M31 DeepSeek 마이그레이션 후 상향.)
DIVE_TEXT_HEAD = 40_000
DIVE_TEXT_TAIL = 20_000
DIVE_THRESHOLD = DIVE_TEXT_HEAD + DIVE_TEXT_TAIL
DIVE_QUESTION_MAX_CHARS = 1_000

# question 은 선택된 공개 판례 내부에서 확인할 추출 항목이다. 무엇을 넣지 말아야 하는지는
# 이 도구의 docstring 이 모델에게 말한다 — 정규식 치환 층은 두지 않는다(2026-08-05 제품 결정).
# 길이 상한만 코드가 본다.


def _clip_question(question: str) -> str:
    """추출 질문 길이 상한 — 폭주한 프롬프트가 서브에이전트 입력을 삼키지 않게 한다."""
    q = question or ""
    if len(q) <= DIVE_QUESTION_MAX_CHARS:
        return q
    return q[: DIVE_QUESTION_MAX_CHARS - 1].rstrip() + "…"


def _cap_summary(summary: str) -> str:
    """구조화 스키마 밖의 대체 모델을 주입해도 응답 경계에서 길이 계약을 보장."""
    if len(summary) <= DIVE_SUMMARY_MAX_CHARS:
        return summary
    return summary[: DIVE_SUMMARY_MAX_CHARS - 1].rstrip() + "…"


def _truncate_for_dive(text: str) -> str:
    if len(text) <= DIVE_THRESHOLD:
        return text
    return text[:DIVE_TEXT_HEAD] + "\n\n[...중략...]\n\n" + text[-DIVE_TEXT_TAIL:]


def _build_body_sections(case: Any) -> tuple[list[str], list[str], bool]:
    """case row → (sections, sources_used, text_truncated).

    content_md만 주입 — summary/generated_summary는 outlier(대법원 max 17K)에서
    token 한계 압박 + content_md와 정보 중복이 커서 dive 정확도 향상이 미미했음.
    """
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
    """precedent_dive 응답 dict → markdown-KV 문자열 (usage·sources·null/false 메타 omit)."""
    status = resp.get("status", "ok")
    lines: list[str] = [f"## status: {status}"]
    if resp.get("message"):
        lines.append(f"- message: {resp['message']}")

    # case 메타 (status=missing이거나 case_id만 있을 때도 표시)
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
    # body truncated flag — 본문 잘렸을 때만 표시 (LLM이 dive 신뢰성 판단)
    if resp.get("text_truncated"):
        lines.append("## text_truncated: true")
        lines.append(f"- retained_text: 원문 앞 {DIVE_TEXT_HEAD}자 + 뒤 {DIVE_TEXT_TAIL}자")
    if resp.get("not_in_text"):
        lines.append("## not_in_text: true")
        if resp.get("text_truncated"):
            lines.append("- not_in_text_conclusive: false")
            lines.append("- note: 중간 원문이 생략되어 보존 구간에 없다는 뜻일 뿐, 전체 원문 부재를 확정하지 못함")
    if resp.get("summary"):
        # 종전에는 같은 사실을 네 번 적었다 — 절 제목 `(생성 추출 요약)` + `summary_provenance`
        # + `quote_eligible: false` + `quotation_status: not_verbatim`. 표지 하나로 합친다.
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
