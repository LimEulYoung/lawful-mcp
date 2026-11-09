"""Tool list + dive sub-agent.

  TOOLS              — the 5 tools the Agent API registers.
  make_dive_subagent — the sub-agent precedent_dive delegates to; the caller supplies the model and settings.
"""
from __future__ import annotations

from pydantic_ai import Agent

from .prompts import DIVE_PROMPT
from .schemas import DiveResult
from .tools.precedent_dive import precedent_dive
from .tools.precedent_search import precedent_search
from .tools.sentencing_analysis import sentencing_analysis
from .tools.statutes import statute_read, statute_search

TOOLS = [
    sentencing_analysis,
    statute_search,
    statute_read,
    precedent_search,
    precedent_dive,
]


# ---------- dive sub-agent factory ----------

def make_dive_subagent(model, settings) -> Agent:
    """Sub-agent that precedent_dive delegates to.

    0 tools — the judgment body is injected as the user prompt. model and settings are required, so the
    agent never runs on a default the caller did not choose.
    """
    return Agent(
        model=model,
        model_settings=settings,
        deps_type=None,
        output_type=DiveResult,
        system_prompt=DIVE_PROMPT,
    )
