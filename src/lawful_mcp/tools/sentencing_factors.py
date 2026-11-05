"""Sentencing factors and suspension grounds, picked by database id.

The corpus owns the classification. Whether a ground fits the case is the
model's judgement; the tool reads its kind and direction (special factors) or
its quadrant (suspension grounds) from the guideline category it belongs to.
The wording is not verified — counting only is the intended design.

Special factors are checked against the **section** of the chosen type,
suspension grounds against the category.

- `sg_factors.subtype_id` attaches a section's table to the section's first
  type (147 of 149 sections) or to a section node above the types (two drug
  sections). So a row on a type covers the types sharing its parent, section
  number and name, and a row on an inner node covers the types below it — not
  just the one type it hangs on. Narrowed to that one type, 281 of 436 types
  could pick no special factor at all. Without the section check, 8 of 47
  benchmark picks were factors from another section of the same category, and
  one changed the band (a 폭행범죄 type-1 case counted the 상해 section's
  「공무집행방해의 경우」).
- `sg_probation_factors.section_no` is the order of the suspension table, not
  the type's section number (the second 교통범죄 table is drink and unlicensed
  driving, while section 02 is hit-and-run). The corpus has nothing linking a
  table to a type, so the table stays a label and the choice is the model's.
"""

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Annotated

from pydantic import BeforeValidator, StrictInt

_DIGITS = re.compile(r"\s*\d+\s*")


def _decode_ids(value):
    """Unwrap the two ways providers stringify an integer array, without widening the schema.

    A JSON layer around the whole array (`"[148, 146]"`, seen in the wild) and
    digit-string elements (`["148"]`). A schema error spends the tool's single
    retry, and the same call again fails the turn, so only these unambiguous
    shapes are fixed here. Anything else — booleans, decimals, objects —
    reaches the `StrictInt` check unchanged and is refused. Direct calls that
    skip schema validation (MCP, tests) pass through this at the top of
    `sentencing_analysis` too.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return value
    if isinstance(value, list):
        return [int(v) if isinstance(v, str) and _DIGITS.fullmatch(v) else v for v in value]
    return value


FactorIds = Annotated[list[StrictInt] | None, BeforeValidator(_decode_ids)]

_FIELDS = ("guideline_factor_ids", "probation_factor_ids")

# The retired argument names, which took factors as sentence dicts. The MCP
# server accepts them without putting them in the schema and answers with how
# to send ids (`retired_factor_args`). Not accepting them lets the SDK and
# pydantic drop the unknown argument silently and compute 「기본영역」 with no
# factors — better no number than a wrong one. The agent tool schema omits them.
RETIRED_FIELDS = ("guideline_factors", "probation_factors")
RETIRED_MESSAGE = (
    "guideline_factors·probation_factors(문장 dict)는 받지 않습니다 — 응답의 「특별양형인자」·"
    "「집행유예 참작사유」 목록에서 사건에 해당하는 사유의 id를 골라 guideline_factor_ids·"
    "probation_factor_ids 정수 배열로 guideline_type과 함께 보내세요(예: guideline_factor_ids=[3045]). "
    "가중·감경·주요·일반 분류는 도구가 기준표에서 읽습니다. 목록은 죄명만으로 부른 응답에 있습니다."
)


def retired_factor_args(**given) -> str | None:
    """The reply to return without computing when any retired dict argument arrived, else None."""
    sent = [k for k in RETIRED_FIELDS if given.get(k) not in (None, "", {}, [])]
    if not sent:
        return None
    return f"## status: bad_factor_id\n- {RETIRED_MESSAGE}\n- 받은 옛 인자: {', '.join(sent)}\n"


@dataclass(frozen=True)
class Factor:
    id: int
    text: str
    group: tuple[str, ...]
    key: str | None          # list name on the arithmetic side; None for ordinary factors (not selectable)
    note: str = ""
    aliases: frozenset[int] = frozenset()   # every row id of this ground — any of them means the same ground
    types: frozenset[int] | None = None     # leaf types where it counts; None = unchecked (suspension)


def _ids(csv: str) -> frozenset[int]:
    return frozenset(int(i) for i in csv.split(","))


def _sections(conn: sqlite3.Connection, category_id: int) -> tuple[list[int], dict[int, frozenset[int]]]:
    """The category's leaf types (guideline order) and, per node, the types its factor rows cover."""
    nodes = {r["id"]: r for r in conn.execute("SELECT * FROM sg_subtypes WHERE category_id=? ORDER BY id",
                                              (category_id,))}
    leaves = [i for i, n in nodes.items() if n["type_criterion"] is not None]

    def same_section(n):
        return (n["parent_subtype_id"], n["section_no"], n["name"])

    def below(node_id):
        out = []
        for leaf in leaves:
            n, seen = nodes[leaf], set()
            while n is not None and n["id"] not in seen:
                if n["id"] == node_id:
                    out.append(leaf)
                    break
                seen.add(n["id"])
                n = nodes.get(n["parent_subtype_id"])
        return out

    cover = {}
    for i, n in nodes.items():
        types = ([leaf for leaf in leaves if same_section(nodes[leaf]) == same_section(n)]
                 if n["type_criterion"] is not None else below(i))
        cover[i] = frozenset(types)
    return leaves, cover


def _tag(n: int) -> str:
    return chr(ord("A") + n) if n < 26 else f"Z{n - 25}"


def section_tags(conn: sqlite3.Connection, category_id: int | None) -> dict[int, str]:
    """Leaf type -> its section's letter (guideline order). Empty for a single-section category.

    The type list prints the letter before each type, and a special factor
    tied to a section carries that letter. Full section names are too long:
    sex offences have 9 sections, the names alone took 2,272 characters of
    headings, and 26 charges lost their statute text and 18 their suspension
    grounds to the budget.
    """
    if category_id is None:
        return {}
    leaves, cover = _sections(conn, category_id)
    groups = list(dict.fromkeys(cover[leaf] for leaf in leaves))
    if len(groups) < 2:
        return {}
    letter = {g: _tag(n) for n, g in enumerate(groups)}
    return {leaf: letter[cover[leaf]] for leaf in leaves}


def read_catalog(conn: sqlite3.Connection, category_id: int | None) -> dict[str, list[Factor]]:
    """The category's factor list — one entry per classification and wording.

    A sentence repeated across sections or suspension tables within one group
    counts the same everywhere, so it is listed once under its smallest row id
    and the other row ids stay valid. Most of a response used to be that
    repetition: 관세 has the most suspension rows, 100, but only 30 distinct
    entries. A suspension sentence present in some tables only carries those
    tables' labels, and a special factor present in some sections only carries
    their letters — that is how the model knows which table or section it
    belongs to.
    """
    if category_id is None:
        return {field: [] for field in _FIELDS}
    leaves, cover = _sections(conn, category_id)
    tags = section_tags(conn, category_id)
    rank = {leaf: n for n, leaf in enumerate(leaves)}
    merged: dict[tuple, dict] = {}
    for r in conn.execute("SELECT id, subtype_id, scope, kind, direction, text FROM sg_factors WHERE category_id=? "
                          "ORDER BY scope DESC, kind, direction, text, id", (category_id,)):
        m = merged.setdefault((r["scope"], r["kind"], r["direction"], r["text"]), {"ids": [], "types": set()})
        m["ids"].append(r["id"])
        m["types"] |= cover.get(r["subtype_id"], frozenset())
    guideline = []
    for (scope, kind_kr, direction_kr, text), m in merged.items():
        kind = {"행위": "act", "행위_공통": "act", "행위_미수": "act", "행위자_기타": "actor"}[kind_kr]
        direction = {"가중": "aggravators", "감경": "mitigators"}[direction_kr]
        key = f"special_{kind}_{direction}" if scope == "특별" else None
        everywhere = m["types"] == set(leaves)
        # Section letters go on special factors only — they are the only ones
        # picked, and the letter is what makes a 「(2유형)」 in the wording
        # read as the second type *of that section*.
        letters = sorted({tags[t] for t in m["types"] if t in tags})
        note = "" if everywhere or not key or not letters else "·".join(letters)
        guideline.append(Factor(min(m["ids"]), text, (scope, kind_kr, direction_kr), key, note,
                                frozenset(m["ids"]), frozenset(m["types"])))
    # Special factors: common grounds first, then section order. Ordinary factors by wording.
    order = {g: n for n, g in enumerate(dict.fromkeys(f.group for f in guideline))}
    first = {f.id: min((rank[t] for t in f.types), default=0) for f in guideline}
    guideline.sort(key=lambda f: (order[f.group], f.note != "", first[f.id], f.note, f.text) if f.key
                   else (order[f.group], False, 0, "", f.text))

    tables = [r[0] for r in conn.execute(
        "SELECT DISTINCT COALESCE(section_no, '') || char(31) || source_note FROM sg_probation_factors "
        "WHERE category_id=? ORDER BY 1", (category_id,))]

    def label(table: str) -> str:
        # `section_no` is the table's order within the guideline, not a type's
        # section number — written 「표 N」 so it does not read as the type
        # list's 「02. …」.
        section, note = table.split(chr(31))
        ordinal = f"표 {int(section)}" if section.isdigit() else (f"표 {section}" if section else "")
        return " · ".join(filter(None, (ordinal, note)))

    probation = []
    rows = conn.execute(
        "SELECT MIN(id) AS id, GROUP_CONCAT(id) AS ids, pole, direction, text, "
        "GROUP_CONCAT(COALESCE(section_no, '') || char(31) || source_note, char(30)) AS tables "
        "FROM sg_probation_factors WHERE category_id=? GROUP BY pole, direction, text "
        "ORDER BY CASE pole WHEN 'major' THEN 0 ELSE 1 END, CASE direction WHEN 'positive' THEN 0 ELSE 1 END, "
        "MIN(COALESCE(section_no, '') || char(31) || source_note), text",
        (category_id,),
    )
    for r in rows:
        present = sorted(set(r["tables"].split(chr(30))))
        note = "" if len(present) == len(tables) else " / ".join(label(t) for t in present)
        probation.append(Factor(r["id"], r["text"], (r["pole"], r["direction"]),
                                f"{r['pole']}_{r['direction']}", note, _ids(r["ids"])))
    order = {g: n for n, g in enumerate(dict.fromkeys(f.group for f in probation))}
    probation.sort(key=lambda f: (order[f.group], f.note != "", f.note, f.text))   # common sentences first
    return {"guideline_factor_ids": guideline, "probation_factor_ids": probation}


def format_catalog(catalog: dict[str, list[Factor]]) -> list[str]:
    """Keep the original group headings and print each selectable id beside its source text."""
    lines = []
    sections = (
        ("특별양형인자", "guideline_factor_ids", [f for f in catalog["guideline_factor_ids"] if f.key]),
        ("일반양형인자", "", [f for f in catalog["guideline_factor_ids"] if not f.key]),
        ("집행유예 참작사유", "probation_factor_ids", catalog["probation_factor_ids"]),
    )
    for title, field, factors in sections:
        if not factors:
            continue
        lines.append(f"## {title} ({len(factors)}개)")
        if field == "guideline_factor_ids":
            lines.append(f"- 해당 사유의 id를 `{field}` 정수 배열로 선택. 분류는 자동 적용."
                         " [A] 표시 사유는 유형 목록의 [A] 유형에만 센다.")
        elif field:
            lines.append(f"- 해당 사유의 id를 `{field}` 정수 배열로 선택. 분류는 자동 적용.")
        else:
            lines.append("- 권고영역 결정에는 쓰지 않고 선고형 위치를 판단할 때 참고.")
        # Suspension tables get a group heading per table; a special factor's
        # section letter goes on its own line — a heading per section repeated
        # the classification for every section (상해 +300 characters,
        # 강제추행 +440).
        inline = field == "guideline_factor_ids"
        groups: dict[tuple, list[Factor]] = {}
        for f in factors:
            groups.setdefault((f.group, "" if inline else f.note), []).append(f)
        for (group, note), members in groups.items():
            lines.append("[" + "/".join(group) + "]" + (f" ({note})" if note else ""))
            for f in members:
                tag = f" [{f.note}]" if inline and f.note else ""
                lines.append(f"- id={f.id}{tag}: {f.text}" if field else f"- {f.text}")
    return lines


def _why_not(factor_id: int, field: str, catalog: dict[str, list[Factor]]) -> str:
    """Why an id cannot be counted — phrased as what to send instead."""
    if field == "guideline_factor_ids":
        found = next((f for f in catalog[field] if factor_id in f.aliases), None)
        if found is not None and not found.key:
            return f"{factor_id}(일반양형인자 — 권고영역 계산에 쓰지 않으니 빼고 보낸다)"
        if found is not None:
            return f"{factor_id}(다른 절 [{found.note}]의 인자 — 이 유형의 절에는 없다)"
    other = _FIELDS[1 - _FIELDS.index(field)]
    if any(factor_id in f.aliases and f.key for f in catalog[other]):
        return f"{factor_id}(`{other}` 목록의 id — 그쪽으로 보낸다)"
    return f"{factor_id}(이 양형기준의 목록에 없는 id)"


def resolve_ids(
    catalog: dict[str, list[Factor]],
    guideline_ids: list[int] | None,
    probation_ids: list[int] | None,
    leaf_id: int | None = None,
) -> tuple[dict, dict, str]:
    """Return the arithmetic's arguments only after checking both selections.

    Any invalid id raises ValueError — nothing is computed. With ``leaf_id``,
    special factors count only for the types their section covers
    (`Factor.types`)."""
    groups, trace, errors = [], [], []
    for field, ids in zip(_FIELDS, (guideline_ids, probation_ids), strict=True):
        selected: dict[str, list[str]] = {}
        groups.append(selected)
        if ids is None:
            continue
        if not isinstance(ids, list) or any(type(i) is not int for i in ids):
            errors.append(f"- {field}: id의 정수 배열이 필요합니다.")
            continue
        allowed = [f for f in catalog[field]
                   if f.key and (leaf_id is None or f.types is None or leaf_id in f.types)]
        available = {i: f for f in allowed for i in f.aliases}
        invalid = [i for i in dict.fromkeys(ids) if i not in available]
        if invalid:
            errors.append(f"- {field}: 셀 수 없는 id: " + ", ".join(_why_not(i, field, catalog) for i in invalid))
            errors.append("- 선택 가능한 항목: " + "; ".join(f"{f.id}={f.text}" for f in allowed))
            continue
        for i in dict.fromkeys(ids):
            f = available[i]
            values = selected.setdefault(f.key, [])
            if f.text not in values:
                values.append(f.text)
                note = (f" [{f.note}]" if f.types is not None else f" ({f.note})") if f.note else ""
                trace.append(f"- {field} id={i}: {f.text} → [{' / '.join(f.group)}]{note}")
    if errors:
        raise ValueError("\n".join(errors) + "\n- 입력을 수정해 다시 호출하세요. 계산하지 않았습니다.")
    return *groups, ("\n## 적용 인자\n" + "\n".join(trace) + "\n" if trace else "")
