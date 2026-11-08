"""Dependencies for the harness tools: the dive sub-agent, built from the environment.

The tools take a pydantic-ai ``RunContext`` and read ``ctx.deps``, a
``harness_local.deps.HarnessDeps`` with one slot, ``dive_subagent``. Each tool
opens its own read-only SQLite connection, so the database is not a
dependency.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config import dive_config
from .harness_local.deps import HarnessDeps

if TYPE_CHECKING:
    from pydantic_ai import Agent


def build_dive_subagent() -> "Agent | None":
    """Build the dive sub-agent from the environment, or None if unconfigured.

    Any OpenAI-compatible endpoint works. The agent has no tools of its own:
    the judgment body is injected into its prompt.
    """
    cfg = dive_config()
    if cfg is None:
        return None

    base_url, api_key, model = cfg
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider

    from .harness_local.agents import make_dive_subagent

    return make_dive_subagent(
        OpenAIChatModel(model, provider=OpenAIProvider(base_url=base_url, api_key=api_key)),
        # Only a generation ceiling: a summary that stops mid-sentence loses
        # the conclusion. Sampling parameters are left to the endpoint, whose
        # defaults differ by vendor.
        OpenAIChatModelSettings(max_tokens=16384),
    )


def build_deps(dive_subagent: "Agent | None" = None) -> HarnessDeps:
    """Fresh dependencies for one tool call.

    The dive sub-agent is passed in because building a model client per call
    would be wasteful.
    """
    return HarnessDeps(dive_subagent=dive_subagent)


__all__ = ["HarnessDeps", "build_deps", "build_dive_subagent"]
