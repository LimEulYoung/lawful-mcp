"""README.md and README.ko.md are the same document in two languages.

A translation rots silently, and the part that hurts when it does is the part
a reader copies out: endpoints, install commands, environment variables. Those
all live in fenced code blocks, so the blocks are compared verbatim — a Korean
reader is the one least likely to cross-check the English. Section structure is
compared by heading level, which catches a section added on one side only.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^```.*?^```", re.M | re.S)


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_code_blocks_are_identical():
    en, ko = FENCE.findall(_read("README.md")), FENCE.findall(_read("README.ko.md"))
    assert en, "no code blocks found — the fence pattern stopped matching"
    assert en == ko


def test_section_structure_matches():
    levels = lambda text: [len(h) for h in re.findall(r"^(#+) ", text, re.M)]
    assert levels(_read("README.md")) == levels(_read("README.ko.md"))
