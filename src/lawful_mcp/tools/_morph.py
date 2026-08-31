"""One shared Kiwi analyser for the whole process.

Both search tools need morphological analysis — one to query the morpheme
index, the other to split law names — and the analyser is expensive: about
half a second to load and 246MB resident.

**It has to be one instance.** Caching it per module does not achieve that:
each module's cache is its own, so a second analyser loads the first time
the other tool runs. That is what happened here, and the comment in the
first module said the opposite — that the analyser would already be loaded,
so the cost was negligible. It was not measured until it was.

The import is deferred, so both tools import cleanly without kiwipiepy
installed; only tokenisation needs it, and callers fall back to empty tokens
when it is missing.
"""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def kiwi():
    """The analyser. Loads once per process (~0.5s, +246MB resident)."""
    from kiwipiepy import Kiwi
    return Kiwi()
