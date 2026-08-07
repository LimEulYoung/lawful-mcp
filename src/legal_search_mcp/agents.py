"""Tool list and the dive sub-agent factory.

The model is not chosen here. ``deps.build_dive_subagent`` reads the
environment and passes a model in, so there is exactly one place that
decides which endpoint the sub-agent talks to.
"""
from __future__ import annotations

from pydantic_ai import Agent

from .prompts import DIVE_PROMPT
from .schemas import DiveResult
from .tools import (
    compute_sentencing_range,
    precedent_dive,
    precedent_search,
    sentence_statistics,
    statute_lookup,
)

TOOLS = [
    compute_sentencing_range,
    statute_lookup,
    precedent_search,
    precedent_dive,
    sentence_statistics,
]


def make_dive_subagent(model, settings) -> Agent:
    """Sub-agent that ``precedent_dive`` delegates to.

    No tools of its own — the judgment body arrives in the user prompt.
    ``model`` and ``settings`` are required: a default here would silently
    diverge from the configured endpoint.
    """
    return Agent(
        model=model,
        model_settings=settings,
        deps_type=None,
        output_type=DiveResult,
        system_prompt=DIVE_PROMPT,
    )
