"""sentencing_guideline — one guideline category as the Sentencing Commission writes it. Data from `sg_*`, sections
of text for `sentencing_analysis`.

  name, source(conn, category_id) — the category's name and the link to its commentary
  sections(conn, category_id)     — its types with their recommended ranges, and its factors

The model judges which type and factors fit the case; each factor carries its classification (special or general,
act or actor, aggravating or mitigating; probation factors by weight and direction) and, where it belongs to only
some type sections or probation tables, which ones.

- `sg_factors.subtype_id` hangs a section's table on the section's first type (147 of 149 sections) or on the
  section node above its types (2 drug sections). So a row on a type covers the types sharing its parent, section
  number and name, and a row on an inner node covers the types below it. A special factor carries the letters of the
  sections it covers when it does not cover all of them, so the model does not take a factor from another section.
- `sg_probation_factors.section_no` numbers the probation tables in order; it is not the types' section number
  (in 교통범죄 the second table is 음주·무면허운전, while section 02 is 교통사고 후 도주). Nothing in the corpus maps
  a table to its types, so the table stays a label and the choice stays with the model.
"""

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Factor:
    text: str
    group: tuple[str, ...]
    special: bool            # special factors decide the band; general ones place the sentence within it
    note: str = ""           # the sections (special factors) or probation tables it belongs to, when not all
    types: frozenset[int] = frozenset()   # the leaf types its rows cover (special and general factors)


def _sections(conn: sqlite3.Connection, category_id: int) -> tuple[list[int], dict[int, frozenset[int]]]:
    """The category's leaf types in guideline order, and for every node the types its factor rows cover."""
    nodes = {r["id"]: r for r in conn.execute("SELECT * FROM sg_subtypes WHERE category_id=? ORDER BY id",
                                              (category_id,))}
    leaves = [i for i, n in nodes.items() if n["type_criterion"] is not None]

    def same_section(n):
        return (n["parent_subtype_id"], n["section_no"], n["name"])

    def below(node_id):
        out = []
        for leaf in leaves:
            n = nodes[leaf]
            while n is not None:
                if n["id"] == node_id:
                    out.append(leaf)
                    break
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


def section_tags(conn: sqlite3.Connection, category_id: int) -> dict[int, str]:
    """leaf type → its section's letter, in guideline order; empty when the category has one section.

    The type list prints the letter before each type and each section-bound special factor carries its letters.
    Full section names cost too much — 성범죄 has 9 sections and their names alone took 2,272 characters of
    headings.
    """
    leaves, cover = _sections(conn, category_id)
    groups = list(dict.fromkeys(cover[leaf] for leaf in leaves))
    if len(groups) < 2:
        return {}
    letter = {g: _tag(n) for n, g in enumerate(groups)}
    return {leaf: letter[cover[leaf]] for leaf in leaves}


def read_catalog(conn: sqlite3.Connection, category_id: int) -> dict[str, list[Factor]]:
    """The category's factors — `guideline` (special and general) and `probation` — one entry per classification
    and text.

    A text repeated across sections or probation tables in the same group is listed once: the category with the most
    probation rows (관세) has 100 rows but 30 distinct entries. A probation text that only some tables carry keeps
    those tables' labels, and a special factor that only some sections carry keeps those sections' letters, because
    they tell the model which table or section it belongs to.
    """
    leaves, cover = _sections(conn, category_id)
    tags = section_tags(conn, category_id)
    rank = {leaf: n for n, leaf in enumerate(leaves)}
    merged: dict[tuple, dict] = {}
    for r in conn.execute("SELECT id, subtype_id, scope, kind, direction, text FROM sg_factors WHERE category_id=? "
                          "ORDER BY scope DESC, kind, direction, text, id", (category_id,)):
        types = merged.setdefault((r["scope"], r["kind"], r["direction"], r["text"]), set())
        types |= cover.get(r["subtype_id"], frozenset())
    guideline = []
    for (scope, kind_kr, direction_kr, text), types in merged.items():
        special = scope == "특별"
        # Only special factors carry the section: they decide the band, and the letter is what lets the model read
        # 「(2유형)」 inside a text as the second type of *that* section.
        letters = sorted({tags[t] for t in types if t in tags})
        note = "" if types == set(leaves) or not special or not letters else "·".join(letters)
        guideline.append(Factor(text, (scope, kind_kr, direction_kr), special, note, frozenset(types)))
    # Special factors: shared ones first, then section by section. General factors keep the text order.
    order = {g: n for n, g in enumerate(dict.fromkeys(f.group for f in guideline))}
    guideline.sort(key=lambda f: (order[f.group], f.note != "", min((rank[t] for t in f.types), default=0),
                                  f.note, f.text) if f.special else (order[f.group], False, 0, "", f.text))

    tables = [r[0] for r in conn.execute(
        "SELECT DISTINCT COALESCE(section_no, '') || char(31) || source_note FROM sg_probation_factors "
        "WHERE category_id=? ORDER BY 1", (category_id,))]

    def label(table: str) -> str:
        # `section_no` is the table's place in the guideline, not a type section number — written 「표 N」 so it
        # does not read as the type list's 「02. …」.
        section, note = table.split(chr(31))
        ordinal = f"표 {int(section)}" if section else ""
        return " · ".join(filter(None, (ordinal, note)))

    probation = []
    rows = conn.execute(
        "SELECT pole, direction, text, "
        "GROUP_CONCAT(COALESCE(section_no, '') || char(31) || source_note, char(30)) AS tables "
        "FROM sg_probation_factors WHERE category_id=? GROUP BY pole, direction, text "
        "ORDER BY CASE pole WHEN '주요' THEN 0 ELSE 1 END, CASE direction WHEN '긍정' THEN 0 ELSE 1 END, "
        "MIN(COALESCE(section_no, '') || char(31) || source_note), text",
        (category_id,),
    )
    for r in rows:
        present = sorted(set(r["tables"].split(chr(30))))
        note = "" if len(present) == len(tables) else " / ".join(label(t) for t in present)
        probation.append(Factor(r["text"], (r["pole"], r["direction"]), False, note))
    order = {g: n for n, g in enumerate(dict.fromkeys(f.group for f in probation))}
    probation.sort(key=lambda f: (order[f.group], f.note != "", f.note, f.text))   # shared texts first
    return {"guideline": guideline, "probation": probation}


def _factor_lines(title: str, factors: list[Factor]) -> list[str]:
    """One factor section's lines, under the guideline's group headings."""
    lines = []
    if title == "특별양형인자" and any(f.note for f in factors):
        lines.append("- [A] 표시 사유는 유형 목록의 [A] 유형에만 센다.")
    # Probation tables head their own groups; a special factor's section letters ride on its line, so the
    # classification is not repeated for every section.
    inline = title == "특별양형인자"
    groups: dict[tuple, list[Factor]] = {}
    for f in factors:
        groups.setdefault((f.group, "" if inline else f.note), []).append(f)
    for (group, note), members in groups.items():
        lines.append("[" + "/".join(group) + "]" + (f" ({note})" if note else ""))
        lines += [f"- [{f.note}] {f.text}" if inline and f.note else f"- {f.text}" for f in members]
    return lines


def _factor_sections(catalog: dict[str, list[Factor]]) -> list[tuple[str, str]]:
    """The factor sections: special, general, probation."""
    out = []
    for title, members in (("특별양형인자", [f for f in catalog["guideline"] if f.special]),
                           ("일반양형인자", [f for f in catalog["guideline"] if not f.special]),
                           ("집행유예 참작사유", catalog["probation"])):
        if members:
            out.append((f"## {title} ({len(members)}개)", "\n".join(_factor_lines(title, members))))
    return out


# ---------------------------------------------------------------- the type table
def _type_table(conn: sqlite3.Connection, category_id: int) -> tuple[str, str]:
    """Each type — 「name N유형」 (criterion) — with its three recommended ranges, the section letter before the
    quoted name. Names, criteria and ranges are stored as shown: a type's name is unique in its category, and the
    Korean enumeration (가., 나.) stays — it separates parent groups (강간/강제추행, 체포감금/유기학대)."""
    leaves = conn.execute("SELECT id, name, type_no, type_criterion FROM sg_subtypes WHERE category_id=? "
                          "AND type_criterion IS NOT NULL ORDER BY id", (category_id,)).fetchall()
    tags = section_tags(conn, category_id)
    lines = []
    for lf in leaves:
        name = f"{lf['name']} {lf['type_no']}유형" if lf["type_no"] is not None else lf["name"]
        ranges = {r["level"]: r["raw_text"] for r in conn.execute(
            "SELECT level, raw_text FROM sg_ranges WHERE subtype_id=?", (lf["id"],))}
        desc = " · ".join(f"{lv} {ranges[lv]}" for lv in ("기본", "감경", "가중"))
        tag = f"[{tags[lf['id']]}] " if lf["id"] in tags else ""
        lines.append(f'- {tag}"{name}" ({lf["type_criterion"]}): {desc}')
    return "## 권고 형량범위 — 범죄유형별", "\n".join(lines)


# ---------------------------------------------------------------- the category
def name(conn: sqlite3.Connection, category_id: int) -> str:
    """The category's name as the guideline writes it."""
    return conn.execute("SELECT name FROM sg_categories WHERE id=?", (category_id,)).fetchone()["name"]


def source(conn: sqlite3.Connection, category_id: int) -> str:
    """The link to the category's commentary."""
    url = conn.execute("SELECT manual_url FROM sg_categories WHERE id=?", (category_id,)).fetchone()["manual_url"]
    return f"- 출처: [{name(conn, category_id)} 양형기준 해설서(PDF)]({url})"


def sections(conn: sqlite3.Connection, category_id: int) -> list[tuple[str, str]]:
    """The category's type table and factor sections."""
    return [_type_table(conn, category_id), *_factor_sections(read_catalog(conn, category_id))]
