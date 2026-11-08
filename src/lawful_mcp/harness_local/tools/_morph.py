"""Kiwi morpheme analyzer — one instance per process, shared by `precedent_search` (morpheme FTS) and `statutes`
(statute-name tokens). An instance takes about 250 MB and half a second to load, and `lru_cache` is per function,
so the tools must not each wrap their own.

The import is deferred to the first call, so importing a tool module does not load the model.
"""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def kiwi():
    """Morpheme analyzer, loaded once per process."""
    from kiwipiepy import Kiwi
    return Kiwi()
