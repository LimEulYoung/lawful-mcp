"""Structured output schema for the ``precedent_dive`` sub-agent."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# Hard cap on the extracted summary — not the target length.
#
# The target length is set by the length instruction in DIVE_PROMPT and by
# the Field description below; the sub-agent writes to those. This constant
# only fires when a model overruns them.
#
# Keep the two values apart. If the cap equalled the instructed length, every
# summary that ran slightly over would be clipped exactly at the end — losing
# the final sentence, which is usually the conclusion. A generous cap lets
# those overruns through intact and still stops a runaway generation.
#
# Changing this does not change typical summary length. To change that, edit
# DIVE_PROMPT and the Field description — and the tool descriptions that
# quote the length, so they do not age apart.
DIVE_SUMMARY_MAX_CHARS = 2400


class DiveResult(BaseModel):
    """Facts extracted from one judgment body in answer to a question."""

    summary: str = Field(
        max_length=DIVE_SUMMARY_MAX_CHARS,
        description="질문에 대한 생성 추출 요약, 500자 이내(직접인용 아님)",
    )
    not_in_text: bool = Field(description="원문에 답이 명시되어 있지 않으면 True")

    @field_validator("summary", mode="before")
    @classmethod
    def _cap_summary(cls, value):
        """Clip deterministically instead of forcing a structured-output retry."""
        if not isinstance(value, str) or len(value) <= DIVE_SUMMARY_MAX_CHARS:
            return value
        return value[: DIVE_SUMMARY_MAX_CHARS - 1].rstrip() + "…"
