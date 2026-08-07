"""Canonical form for a charge name.

The same charge is written several ways across judgments, and the variants
have to collapse to one key or statistics for that charge split across
buckets:

- Spacing. A statute-derived charge appears both spaced and unspaced.
- Interpunct variants. The middle dot in a charge name occurs as U+00B7,
  U+2024, U+318D and U+2219 depending on the source. Left alone, one charge
  splits four ways.

**Parenthesised qualifiers are preserved.** In special-act charges the text
inside the parentheses is the substantive offence:
``폭력행위등처벌에관한법률위반(공동재물손괴등)`` and ``(공동상해)`` and
``(공동공갈)`` carry different statutory penalties and different sentencing.
Stripping the parentheses collapses dozens of distinct offences into one
bucket, so a "criminal damage" distribution silently fills with assault and
extortion cases. It also undercounts ``prec_defendants.n_charges`` — the
number of distinct charges per defendant — which is how multi-charge
defendants leak into the single-charge pool that ``sentence_statistics``
reports on.

So normalisation touches spacing and punctuation only. The result is the same
form as ``charge_key``; the two names mark caller intent, not different rules.
"""
from __future__ import annotations

import re

_CHARGE_NORM_WS = re.compile(r"\s+")
# Interpunct variants seen in charge names, unified to U+00B7.
_DOT_VARIANTS = str.maketrans({"․": "·", "ㆍ": "·", "∙": "·"})


def norm_charge(s: str) -> str:
    """Canonical form: strip spaces, unify interpuncts, keep parentheses.

    Keeping the parenthesised qualifier is what holds special-act offences in
    separate buckets — see the module docstring.
    """
    if not s:
        return ""
    s = _CHARGE_NORM_WS.sub("", s)
    return s.translate(_DOT_VARIANTS)


def charge_key(s: str) -> str:
    """Lookup key into the charge-to-penalty map. Same rules as ``norm_charge``.

    The mapping table has to tell offences apart —
    ``도로교통법위반(음주운전)`` (drink-driving) and ``(무면허운전)`` (unlicensed
    driving) and the bare form are three different entries — which is why the
    parentheses survive here too.
    """
    if not s:
        return ""
    s = _CHARGE_NORM_WS.sub("", s)
    return s.translate(_DOT_VARIANTS)
