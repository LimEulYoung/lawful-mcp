"""Make a free-text query safe to hand to FTS5.

Punctuation is syntax in an FTS5 MATCH expression, so a query that carries a
comma, a parenthesis or a quote — as natural-language queries from a model
routinely do — raises a syntax error instead of searching. Keep letters,
digits and whitespace; replace everything else with a space.

⚠ Stripping punctuation is not enough — the FTS5 boolean operators survive it.
Joining the remaining tokens by hand turns ``손해배상 AND 위자료`` into
``손해배상 OR AND OR 위자료`` and the expression dies with
``fts5: syntax error near "AND"`` (measured 2026-09-12: both search tools
raised an unhandled exception the moment a caller typed search syntax).
Build every MATCH expression through `query_tokens` + `quoted_and`/`quoted_or`
— never concatenate tokens yourself.

Every tool that builds a MATCH expression goes through here. Both search
tools use it: when only one of them did, the other raised syntax errors on
ordinary queries.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

# In Unicode mode `\w` covers Hangul, Han characters, Latin letters, digits
# and underscore. Underscore is harmless to FTS5, so it can stay.
_FTS_KEEP = re.compile(r'[^\w\s]', re.UNICODE)

# FTS5 boolean operators. The grammar only reads them as operators in upper
# case, but case is ignored here: in a Korean legal corpus 'or'/'and' mean
# nothing as search terms, and left in lower case they match English prose
# fragments by accident (the corpus index holds 1,257 'AND' and 499 'OR').
_FTS_OPERATORS = frozenset({"AND", "OR", "NOT", "NEAR"})


def safe_fts_query(q: str) -> str:
    """Replace FTS5 syntax characters with spaces and collapse runs of space."""
    cleaned = _FTS_KEEP.sub(" ", q or "")
    return " ".join(cleaned.split())


def query_tokens(q: str) -> list[str]:
    """Query -> FTS5 tokens, operators dropped and duplicates removed.

    Original order is kept. Feed the result to `quoted_and`/`quoted_or` to
    build a MATCH expression: dropping the operators is not enough on its own
    (they would survive as tokens), and quoting is what makes the expression
    valid by construction.
    """
    out: list[str] = []
    seen: set[str] = set()
    for t in safe_fts_query(q).split():
        if t.upper() in _FTS_OPERATORS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _quoted(tokens: Iterable[str], sep: str) -> str:
    # Tokens already went through `safe_fts_query`, so none of them can hold a
    # quote — the quoting cannot be escaped from the inside.
    return sep.join(f'"{t}"' for t in tokens if t)


def quoted_and(tokens: Sequence[str]) -> str:
    """Tokens -> quoted AND expression. Same meaning as the implicit
    whitespace AND, without the operator collision."""
    return _quoted(tokens, " ")


def quoted_or(tokens: Sequence[str]) -> str:
    """Tokens -> quoted OR expression, the input to bag-of-words BM25."""
    return _quoted(tokens, " OR ")
