"""Environment wiring, in one place.

Everything this server reads from the environment is declared here so a
self-hoster can see the whole surface at once. Nothing else in the package
reads ``os.environ`` for configuration.

Environment variables:
  CORPUS_DB          Path to the corpus SQLite file. Defaults to the sample
                     fixture shipped in this repository.
  CASE_URL_BASE      Base URL used to build citation links in tool output.
                     Defaults to the hosted service, whose case and statute
                     ids match the published corpus.
  DIVE_API_KEY       API key for the model behind ``precedent_dive``. Any
                     OpenAI-compatible endpoint works. When unset, the tool
                     is not registered and the other four still run.
  DIVE_BASE_URL      Base URL of that endpoint (OpenAI-compatible).
  DIVE_MODEL         Model name to request.
  DIVE_TIMEOUT       Seconds to wait for that model before giving up and
                     returning the excerpt alone. Defaults to 15.
  USE_DENSE          Set to 1 to enable the embedding + rerank retrieval
                     path in ``precedent_search``. Off by default: the
                     lexical path (trigram + morpheme RRF) scored on par in
                     a known-item A/B and needs no external service.
  EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL / EMBED_DIM
  RERANK_BASE_URL / RERANK_API_KEY
                     Only read when USE_DENSE=1.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repository root: src/lawful_mcp/config.py -> up three.
_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_CORPUS_DB = _ROOT / "data" / "fixture.db"

# Seconds the dive summary waits before the tool answers with the excerpt
# alone. 15s is 1.5x the measured range (6.6–10.2s) and is a product choice:
# the call happens mid-turn, where being late is itself the harm.
DEFAULT_DIVE_TIMEOUT_S = 15.0

# Citation links in tool output point at the hosted corpus browser. Case and
# statute ids are the published corpus ids, so links resolve for the fixture
# and for the full hosted corpus alike. Point this elsewhere if you serve
# your own corpus under different ids.
DEFAULT_CASE_URL_BASE = "https://lawful.crow-tit.com"


def corpus_db_path() -> Path:
    """Path to the corpus database."""
    raw = os.environ.get("CORPUS_DB")
    return Path(raw) if raw else DEFAULT_CORPUS_DB


def case_url_base() -> str:
    """Base URL for citation links, without a trailing slash."""
    return os.environ.get("CASE_URL_BASE", DEFAULT_CASE_URL_BASE).rstrip("/")


def dive_config() -> tuple[str, str, str] | None:
    """(base_url, api_key, model) for the dive sub-agent, or None if unset.

    Returning None is a supported state, not an error: the server registers
    the other four tools and reports the dive tool as unavailable.
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

    Without a cap of our own the wait is the library's: pydantic-ai builds an
    httpx client with a 600-second read timeout and the OpenAI SDK retries a
    timeout twice, so one call could hang for up to 30 minutes. A malformed
    value falls back to the default rather than taking the server down.
    """
    try:
        value = float(os.environ.get("DIVE_TIMEOUT", "") or DEFAULT_DIVE_TIMEOUT_S)
    except ValueError:
        return DEFAULT_DIVE_TIMEOUT_S
    return value if value > 0 else DEFAULT_DIVE_TIMEOUT_S
