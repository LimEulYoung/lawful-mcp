"""Dependencies injected into the corpus tools, plus the database factory.

The tools take a pydantic-ai ``RunContext``; ``HarnessDeps`` is what sits on
its ``.deps``. Three slots:

  embed / rerank   Retrieval clients, only touched when USE_DENSE=1. They are
                   lazy, so the default lexical path constructs neither.
  dive_subagent    Sub-agent that reads one judgment body and answers a
                   question about it. None when no model is configured.

The SQLite connection is deliberately *not* a dependency: each tool opens its
own via ``open_db()``. A single shared connection breaks when async tools run
in worker threads.
"""
from __future__ import annotations

import os
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .config import corpus_db_path

if TYPE_CHECKING:
    import httpx
    from openai import OpenAI
    from pydantic_ai import Agent


class LazyClient:
    """Proxy that builds the real client on first attribute access.

    With USE_DENSE=0 the retrieval clients are never touched, so nothing is
    constructed and no key is required; turning dense retrieval on builds
    them at first use.
    """

    __slots__ = ("_factory", "_obj")

    def __init__(self, factory):
        self._factory = factory
        self._obj = None

    def __getattr__(self, name):  # slots resolve normally; everything else delegates
        if self._obj is None:
            self._obj = self._factory()
        return getattr(self._obj, name)


@dataclass
class HarnessDeps:
    embed: "OpenAI | LazyClient"
    rerank: "httpx.Client | LazyClient"
    dive_subagent: "Agent | None"
    # Last (tool_name, args_json) pairs, read by the dedup guard to reject an
    # identical immediate repeat. Per-call, so a client's legitimate retry of
    # the same arguments is not blocked.
    recent_calls: deque[tuple[str, str]] = field(default_factory=lambda: deque(maxlen=10))


def open_db(
    path: str | os.PathLike[str] | None = None,
    *,
    read_only: bool = True,
) -> sqlite3.Connection:
    """Open the corpus database.

    Read-only by default and ``check_same_thread=False`` so async tool calls
    dispatched to worker threads can share it safely.
    """
    p = Path(path) if path else corpus_db_path()
    if not p.exists():
        raise FileNotFoundError(
            f"corpus database not found: {p}\n"
            "Set CORPUS_DB, or use the sample fixture in data/fixture.db."
        )

    if read_only:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(p, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    # sqlite-vec backs the dense vector index. It is optional: without it the
    # lexical path (the default) works, and only USE_DENSE=1 would fail.
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception:  # noqa: BLE001 - extension is optional
        pass

    return conn


def open_embed_client() -> "OpenAI":
    """OpenAI-compatible embedding client (only used when USE_DENSE=1)."""
    from openai import OpenAI

    base_url = os.environ.get("EMBED_BASE_URL", "https://integrate.api.nvidia.com/v1")
    api_key = os.environ.get("EMBED_API_KEY", "dummy")
    return OpenAI(base_url=base_url, api_key=api_key)


def open_rerank_client() -> "httpx.Client":
    """Rerank HTTP client (only used when USE_DENSE=1).

    The caller puts the model name in the request path, so this client's
    base_url is the host only.
    """
    import httpx

    base_url = os.environ.get("RERANK_BASE_URL", "https://ai.api.nvidia.com")
    api_key = os.environ.get("RERANK_API_KEY", "")
    headers = (
        {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        if api_key
        else {}
    )
    return httpx.Client(base_url=base_url, timeout=30.0, headers=headers)


def build_dive_subagent() -> "Agent | None":
    """Build the dive sub-agent from the environment, or None if unconfigured.

    Any OpenAI-compatible endpoint works. The agent has no tools of its own:
    the judgment body is injected into its prompt.
    """
    from .config import dive_config

    cfg = dive_config()
    if cfg is None:
        return None

    base_url, api_key, model = cfg
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider

    from .agents import make_dive_subagent

    return make_dive_subagent(
        OpenAIChatModel(model, provider=OpenAIProvider(base_url=base_url, api_key=api_key)),
        # Only a generation ceiling: a summary that stops mid-sentence loses
        # the conclusion. Sampling parameters are left to the endpoint, whose
        # defaults differ by vendor.
        OpenAIChatModelSettings(max_tokens=16384),
    )


def build_deps(dive_subagent: "Agent | None" = None) -> HarnessDeps:
    """Fresh dependencies for one tool call.

    Per-call, so the dedup guard's ``recent_calls`` does not accumulate
    across calls. The dive sub-agent is passed in because building a model
    client per call would be wasteful.
    """
    return HarnessDeps(
        embed=LazyClient(open_embed_client),
        rerank=LazyClient(open_rerank_client),
        dive_subagent=dive_subagent,
    )


__all__ = [
    "HarnessDeps",
    "LazyClient",
    "build_deps",
    "build_dive_subagent",
    "open_db",
    "open_embed_client",
    "open_rerank_client",
]
