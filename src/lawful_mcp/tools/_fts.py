"""Make a free-text query safe to hand to FTS5.

Punctuation is syntax in an FTS5 MATCH expression, so a query that carries a
comma, a parenthesis or a quote — as natural-language queries from a model
routinely do — raises a syntax error instead of searching. Keep letters,
digits and whitespace; replace everything else with a space.

Every tool that builds a MATCH expression goes through here. Both search
tools use it: when only one of them did, the other raised syntax errors on
ordinary queries.
"""
from __future__ import annotations

import re

# In Unicode mode `\w` covers Hangul, Han characters, Latin letters, digits
# and underscore. Underscore is harmless to FTS5, so it can stay.
_FTS_KEEP = re.compile(r'[^\w\s]', re.UNICODE)


def safe_fts_query(q: str) -> str:
    """Replace FTS5 syntax characters with spaces and collapse runs of space."""
    cleaned = _FTS_KEEP.sub(" ", q)
    return " ".join(cleaned.split())
