"""FTS5 query building shared by the tools (`precedent_search`, `statutes`, `treaties`).

Keeps only Korean (jamo and precomposed syllables), CJK Han characters, alphanumerics and whitespace; everything
else becomes a space, so punctuation in a model's query cannot break FTS5 syntax.

⚠ Stripping punctuation is not enough: the FTS5 boolean operators (AND·OR·NOT·NEAR) remain, and
`"손해배상 AND 위자료"` joined as is fails with `fts5: syntax error near "AND"`. Use `query_tokens()` for an
operator-free token list and `quoted_and()`/`quoted_or()` for a MATCH expression — never join tokens by hand.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

# In the default Unicode mode `\w` matches Hangul, Han characters, alphanumerics and underscore
# (underscore is harmless to FTS5).
_FTS_KEEP = re.compile(r'[^\w\s]', re.UNICODE)
# FTS5 boolean operators. They are operators only in upper case, but are dropped in any case: in a Korean legal
# corpus 'or'/'and' mean nothing as search terms and would only match fragments of English text.
_FTS_OPERATORS = frozenset({"AND", "OR", "NOT", "NEAR"})


def safe_fts_query(q: str) -> str:
    """Replace characters that violate FTS5 syntax with spaces and collapse runs of whitespace."""
    cleaned = _FTS_KEEP.sub(" ", q or "")
    return " ".join(cleaned.split())


def query_tokens(q: str) -> list[str]:
    """Query → FTS5 token list. Operators removed, duplicates removed (original order kept).

    To build a MATCH expression, pass this list to `quoted_and`/`quoted_or`: only quoted, the expression is valid
    by construction.
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
    # Tokens have already passed `[^\w\s]`, so a quote character cannot occur inside a token.
    return sep.join(f'"{t}"' for t in tokens if t)


def quoted_and(tokens: Sequence[str]) -> str:
    """Tokens → quoted AND expression. Same meaning as the implicit AND (space-separated), but
    without operator collisions."""
    return _quoted(tokens, " ")


def quoted_or(tokens: Sequence[str]) -> str:
    """Tokens → quoted OR expression. The input to bag-of-words BM25 search."""
    return _quoted(tokens, " OR ")
