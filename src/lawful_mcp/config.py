"""Corpus path, dive-model and Jev-key configuration.

``harness_local`` is copied unmodified from the research harness and reads two
variables of its own: ``HARNESS_DB`` (the corpus) and ``OPENROUTER_API_KEY``
(the ``issue`` reordering in ``precedent_search``). This module maps this
package's settings onto them, so the copy stays byte-identical and a
self-hosted server keeps one documented set of names.

The complete configuration reference is ``.env.example``. HTTP transport
settings live in ``server.py``. Values come from the process environment;
this package does not load .env.

Environment variables:
  CORPUS_DB          Path to the corpus SQLite file. Defaults to the sample
                     fixture shipped in this repository. The harness's own
                     name, HARNESS_DB, is accepted too; CORPUS_DB wins.
  DIVE_API_KEY       API key for the model behind ``precedent_dive``. Any
                     OpenAI-compatible endpoint works. When unset, the tool
                     is not registered and the other three still run.
  DIVE_BASE_URL      Base URL of that endpoint (OpenAI-compatible).
  DIVE_MODEL         Model name to request.
  DIVE_TIMEOUT       Seconds to wait before returning a timeout status with
                     case metadata and retry guidance. Defaults to 15.
  JEV_API_KEY        OpenRouter API key for reordering ``precedent_search``
                     results by its ``issue`` argument. When unset, ``issue``
                     is accepted and results keep their BM25 order.
  USE_JEV            Read by the harness itself: 0 keeps BM25 order even with
                     a key set.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repository root: src/lawful_mcp/config.py -> up three.
_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_CORPUS_DB = _ROOT / "data" / "fixture.db"

# Seconds the dive summary waits before the tool answers with the case
# metadata alone. 15s is 1.5x the measured range (6.6–10.2s) and is the
# harness's own bound; a slower self-hosted model may need more.
DEFAULT_DIVE_TIMEOUT_S = 15.0


def corpus_db_path() -> Path:
    """Path to the corpus database."""
    raw = os.environ.get("CORPUS_DB") or os.environ.get("HARNESS_DB")
    return Path(raw) if raw else DEFAULT_CORPUS_DB


def export_corpus_path() -> None:
    """Point the harness at the corpus: it opens ``HARNESS_DB`` on every call."""
    os.environ["HARNESS_DB"] = str(corpus_db_path())


def jev_api_key() -> str | None:
    """Key for the ``issue`` reordering in ``precedent_search``, or None.

    None is a supported state: the search runs and keeps its BM25 order.
    """
    return os.environ.get("JEV_API_KEY", "").strip() or None


def export_jev_key() -> None:
    """Hand ``JEV_API_KEY`` to the harness, which reads ``OPENROUTER_API_KEY``.

    The variable is cleared when ``JEV_API_KEY`` is unset: an OpenRouter key
    exported for something else must not start sending search results to
    OpenRouter. The server calls this; a library user who imports the harness
    directly sets ``OPENROUTER_API_KEY`` themselves.
    """
    os.environ["OPENROUTER_API_KEY"] = jev_api_key() or ""


def dive_config() -> tuple[str, str, str] | None:
    """(base_url, api_key, model) for the dive sub-agent, or None if unset.

    Returning None is a supported state, not an error: the server registers
    the other three tools and reports the dive tool as unavailable.
    """
    api_key = os.environ.get("DIVE_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = os.environ.get("DIVE_BASE_URL", "").strip()
    model = os.environ.get("DIVE_MODEL", "").strip()
    if not base_url or not model:
        return None
    return (base_url, api_key, model)


def dive_timeout_s() -> float:
    """Seconds ``precedent_dive`` waits for its summary before giving up.

    A malformed value falls back to the default rather than taking the server
    down.
    """
    try:
        value = float(os.environ.get("DIVE_TIMEOUT", "") or DEFAULT_DIVE_TIMEOUT_S)
    except ValueError:
        return DEFAULT_DIVE_TIMEOUT_S
    return value if value > 0 else DEFAULT_DIVE_TIMEOUT_S
