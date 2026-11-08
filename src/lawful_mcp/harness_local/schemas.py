"""Output schemas.

  DiveResult — output of the precedent_dive sub-agent
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# **Hard cap** on the dive extraction summary — not the target length.
#
# ⚠ The target length is set by the length instruction in `DIVE_PROMPT` (1,000 characters) and the Field
# description below; the sub-agent writes to those. This constant is a safety net that only fires when
# the model overruns that instruction. **Do not set the two to the same value** — if they were equal, every
# summary slightly over the instruction would be clipped and lose its last sentence (usually the conclusion).
# With a generous cap such overruns survive intact, and the cap only stops runaway output.
#
# Changing this value does not change the typical summary length. To change that, edit `DIVE_PROMPT` and
# the Field description, and at the same time update the `precedent_dive` docstring.
DIVE_SUMMARY_MAX_CHARS = 2400


# DiveResult — output of the precedent_dive sub-agent
class DiveResult(BaseModel):
    """판결·결정문 본문에서 질문에 답하는 사실 추출."""

    summary: str = Field(
        max_length=DIVE_SUMMARY_MAX_CHARS,
        description="질문에 대한 추출 결과, 1,000자 이내. `>` 줄은 판결문 원문 그대로",
    )
    not_in_text: bool = Field(description="원문에 답이 명시되어 있지 않으면 True")

    @field_validator("summary", mode="before")
    @classmethod
    def _cap_summary(cls, value):
        """Cap the length deterministically when the model overruns the length instruction, instead of a structured-output retry."""
        if not isinstance(value, str) or len(value) <= DIVE_SUMMARY_MAX_CHARS:
            return value
        return value[: DIVE_SUMMARY_MAX_CHARS - 1].rstrip() + "…"
