"""Charge-name normalization — the DB's charge_norm (prec_defendant_charges), the charge_legal_map keys and the
resolver input all share this one form.

It removes whitespace and unifies the dot variants (U+2024, U+318D, U+2219 → U+00B7), which spell one charge in
different ways. Parenthesized qualifiers are kept: in special-act charges the text inside the parentheses is the
offence itself (`폭력행위등처벌에관한법률위반(공동재물손괴등)` ≠ `(공동상해)` ≠ `(공동공갈)`, each with its own statutory
penalty and sentences), so removing it would pool different crimes in `sentence_statistics`. The database stores
charge_norm in this form; a change here needs the data normalized again.
"""
from __future__ import annotations

import re

_CHARGE_NORM_WS = re.compile(r"\s+")
# Dot variants found in Korean charge names → unified to the middle dot (U+00B7)
_DOT_VARIANTS = str.maketrans({"․": "·", "ㆍ": "·", "∙": "·"})


def norm_charge(s: str) -> str:
    """Canonical form: strip whitespace + unify dot variants (parenthesized qualifiers are **kept**)."""
    if not s:
        return ""
    return _CHARGE_NORM_WS.sub("", s).translate(_DOT_VARIANTS)
