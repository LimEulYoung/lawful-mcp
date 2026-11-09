"""precedent_search — judgment retrieval.

OR-mode BM25 over the trigram index and over the morpheme index, fused by RRF, with excerpt previews. When `issue` is
given, the candidate pool is reordered by Jev (the "issue reordering" section below).

A query under 2 characters is an input error. Words of 3+ characters are matched by the trigram index, 2-character
words (charge names such as '사기') by the morpheme index.
"""
from __future__ import annotations

import functools
import logging
import os
import re
import sqlite3
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Any, Sequence

import httpx
from pydantic_ai import RunContext

from .. import telemetry
from ..deps import HarnessDeps, open_db
from ._coerce import coerce_int, coerce_str
from ._fts import _FTS_OPERATORS, query_tokens as _query_tokens, quoted_or as _quoted_or
from ._morph import kiwi as _kiwi

# Retrieval constants (fixed; not exposed to the model).
# Results returned. In searches that returned 8, 7.5% of the judgments the model went on to cite or read were the
# 8th (7th–8th: 15.7%, 6th–8th: 24.1%); the 8th result makes the response about 14% longer.
LIMIT = 7
RRF_K = 60                  # Reciprocal Rank Fusion parameter
OVERSAMPLE = 5              # candidate pool = LIMIT * OVERSAMPLE, so that filtering still leaves LIMIT
FTS_OR_MAX_TOKENS = 40      # cap on OR-query tokens
SNIPPET_TOKENS = 256        # snippet() excerpt tokens (≈ characters): enough for the first points of a judgment
PREVIEW_FALLBACK_CHARS = 200   # summary head when no excerpt matched
EXCERPT_HALF = 60           # characters on each side of the match in an excerpt taken from the text
# Display cap for one item of the issues decided (and for the referenced-articles line). The items shown are 122
# characters at the median and 309 at p90 (882 items over 200 real queries); one runs to 18,524 (2011헌바100). At 150
# characters 38% would be cut, at 200 25%, at 300 11%, and 200→300 costs only 2% of the response.
HOLDING_MAX_CHARS = 300

# Judgment BM25 column weights, in the prec_cases_fts column order
# (case_name, content_md, summary, reference_statute, generated_summary).
# Unweighted, title hits lose to body frequency: on 40 title queries the title weight brings 28% more same-title
# judgments into the top 20 (208 → 266) with body-phrase recall unchanged (35/40). E.g. rank 1 for `음주운전` is
# 「도로교통법위반(음주운전)」 instead of 「공용서류손상·허위공문서작성…」.
# `generated_summary` (AI-written) is weighted lowest, so it does not outrank the court-written summary and articles.
BM25_WEIGHTS = (10.0, 1.0, 2.0, 2.0, 0.5)


def bm25_expr(table: str, weights: Sequence[float] = BM25_WEIGHTS) -> str:
    """`bm25(table, …)` with column weights, which must follow the table's column order (statute articles pass
    `statutes._ART_BM25_WEIGHTS`). No weights for a single-column index: `bm25(table)`."""
    return f"bm25({', '.join([table, *(str(float(w)) for w in weights)])})"


# ---------- direct routing by case number ----------
# prec_cases_fts does not index case_number, so a case number like "2010다89012" cannot be caught by FTS.
# The format is regular (year + case-type code + serial), so it is detected by regex and
# routed to a direct case_number lookup.
_CASE_NO_RE = re.compile(r"\d{2,4}[가-힣]{1,5}\d+")
# Case numbers in standard citation form: right after '선고' (decided) or right before '판결/결정'
# (judgment/decision). Korean legal citations take the form
# 'court + decision date + 선고 + case number + 판결/결정' (the decision date is sometimes omitted), so
# dates, court names and trailing words attach around the case number and break fullmatch. '선고/판결/결정'
# are not in the set of case-type codes (167 kinds, '다/두/노/헌마' …), so they cannot be mistaken for a code
# — a strong context → only tokens adjacent to them are safely split off (ordinary queries do not meet this
# adjacency condition and pass through).
_CASE_NO_CITED_RES = (
    re.compile(r"선고(\d{2,4}[가-힣]{1,5}\d+)"),            # '…선고 2010다89012 …'
    re.compile(r"(\d{2,4}[가-힣]{1,5}\d+)(?=판결|결정)"),   # '… 2018노1234 판결'
)


def _extract_case_number(query: str) -> str | None:
    """Routing key for direct case-number lookup (normalized case number) or None. Catches two forms:

    ① Standard citation form — even with court name, decision date and trailing words mixed in, as in
       '대법원 2010. 5. 26. 선고 2010다89012 판결', extract the case number right after '선고' or right before
       '판결/결정'. Embedded extraction happens only when this strong context is adjacent, so ordinary queries
       are not misread.
    ② Bare case number — the whole query is a case number, with a court in front ('대법원 2010다89012') or words
       after it ('2007다90760 전원합의체 판결') as people cite. Only a *full match* is accepted, to avoid misreading
       ordinary queries such as '민법 제750조'.
    """
    raw = (query or "").strip().replace(" ", "")
    for rx in _CASE_NO_CITED_RES:
        m = rx.search(raw)
        if m:
            return m.group(1)
    q = re.sub(r"^(대법원|헌법재판소|[가-힣]*(법원|고법|지법|지원))", "", raw)
    q = re.sub(r"(전원합의체|판결|결정|선고|사건|판례)+$", "", q)
    return q if _CASE_NO_RE.fullmatch(q) else None


def _case_number_ids(conn: sqlite3.Connection, cno: str, limit: int) -> list[int]:
    """Normalized case number → prec_cases ids (at most `limit`): the judgments carrying that number
    (`prec_case_numbers` has one row per number, so joined cases '2009다84608, 84615' are found by either), the
    ones filed under it alone first."""
    rows = conn.execute(
        "SELECT n.case_id FROM prec_case_numbers n JOIN prec_cases c ON c.id = n.case_id WHERE n.number = ? "
        "ORDER BY length(c.case_number), c.id LIMIT ?",
        (cno, limit),
    ).fetchall()
    return [r[0] for r in rows]


# ---------- court_level normalization + slot-misuse hints (fail-loud) ----------
# The prec_cases.court_level DB enum has only these 4 values. If the LLM passes a court *name* such as
# '고등법원' (high court) or '지방법원' (district court), the SQL filter returns 0 rows altogether and an empty
# result goes out *silently*. Alias mapping + ignoring unknown values with a hint prevents the silent zero.
_VALID_COURT_LEVELS = ("1심", "2심", "대법원", "헌재")
_COURT_LEVEL_ALIASES = {
    "고등법원": "2심", "고법": "2심", "항소심": "2심", "2심법원": "2심",
    "지방법원": "1심", "지법": "1심", "1심법원": "1심", "단독": "1심", "1심판결": "1심",
    "대법": "대법원", "대법원판결": "대법원",
    "헌법재판소": "헌재",
}


_LEVEL_NAMES = {"대법원": "대법원", "헌재": "헌법재판소"}


def court_label(name: str | None, level: str | None, date: str | None, year: Any) -> str:
    """'대법원 2012-02-16', '서울고등법원 2심 2012-02-16': the court, its level unless the name says it, and the
    decision date — the year when the date is unknown."""
    if level and _LEVEL_NAMES.get(level) == name:
        level = None
    return " ".join(str(x) for x in (name, level, date or year) if x)


def _normalize_court_level(cl: str | None) -> tuple[str | None, str | None]:
    """court_level → (DB enum|None, note|None). Valid values/aliases map to the enum; unknown values drop
    the filter (None) and come with a note (so the search does not silently fall to 0 results)."""
    if not cl:
        return None, None
    if cl in _VALID_COURT_LEVELS:
        return cl, None
    if cl in _COURT_LEVEL_ALIASES:
        return _COURT_LEVEL_ALIASES[cl], None
    return None, f"court_level '{cl}' 은 인식 못 해 무시함 — 유효값: 1심/2심/대법원/헌재"


_COURT_SHORT = re.compile(r"(고|행|지|가)법(?!원)")
_COURT_FULL = {"고": "고등법원", "행": "행정법원", "지": "지방법원", "가": "가정법원"}
# The court-name filter: the given part of a name, spaces aside ('수원지방법원 성남지원' · '수원지방법원성남지원').
_COURT_MATCH = "instr(REPLACE({col}, ' ', ''), ?) > 0"


def _check_court_name(conn: sqlite3.Connection, name: str | None) -> tuple[str | None, str | None]:
    """court_name → (the part of a court name to filter by | None, note | None).

    The model's spelling is read as the corpus writes court names: short forms spelled out ('서울고법' →
    '서울고등법원'), no spaces, no seat in brackets ('부산고등법원(창원)' → '부산고등법원'). A value that is part of no
    court name drops the filter with a note, as an unknown court_level does (the filter would otherwise silently
    return 0 results)."""
    if not name:
        return None, None
    key = _COURT_SHORT.sub(lambda m: _COURT_FULL[m.group(1)], re.sub(r"\s|\(.*?\)", "", name))
    if key and conn.execute(
        f"SELECT 1 FROM prec_meta WHERE {_COURT_MATCH.format(col='court_name')} LIMIT 1", (key,)
    ).fetchone():
        return key, None
    return None, (f"court_name '{name}' 에 맞는 법원이 코퍼스에 없어 무시함 — 법원 이름·지역 부분 일치"
                  "(예: '수원', '서울고등')")


def _case_no_in_query_hint(query: str | None) -> str | None:
    """Hint for a case number put in the query slot — steers it to the case_number slot."""
    if _CASE_NO_RE.fullmatch((query or "").replace(" ", "")):
        return ("query 가 사건번호 형식입니다 — 사건번호로 특정 판례를 찾으려면 "
                "case_number 인자에 넣으세요(query 는 사실관계 키워드 전용).")
    return None


def _filter_sql(
    col: str,
    court_level: str | None = None,
    court_name: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> tuple[list[str], list[Any]]:
    """SQL conditions and their arguments for the court level, court name and year filters. `col` prefixes the
    column names ('c.' or ''). The year is the decision year, else the filing year."""
    cond: list[str] = []
    args: list[Any] = []
    if court_level:
        cond.append(f"{col}court_level = ?"); args.append(court_level)
    if court_name:
        cond.append(_COURT_MATCH.format(col=f"{col}court_name")); args.append(court_name)
    if year_from is not None:
        cond.append(f"COALESCE({col}decision_year, {col}case_year) >= ?"); args.append(year_from)
    if year_to is not None:
        cond.append(f"COALESCE({col}decision_year, {col}case_year) <= ?"); args.append(year_to)
    return cond, args


# ---------- rankers ----------

def _or_match(query: str) -> str:
    """Trigram OR match expression over the query's words of 3+ characters (the trigram floor), deduplicated and
    capped. OR, not FTS5's implicit AND: on long, phrase-like queries every word must match at once and nothing
    does (known-item test, N=140: AND finds 0.11, OR 1.00).

    The words are quoted (`_quoted_or`): removing punctuation leaves 'AND'/'NOT', which FTS5 would read as
    operators."""
    toks = [w for w in _query_tokens(query) if len(w) >= 3][:FTS_OR_MAX_TOKENS]
    return _quoted_or(toks)


# Morpheme index (prec_cases_morph_fts): trigram cannot match words under 3 characters ('살인' murder, '사기' fraud),
# so a second index of Kiwi morphemes is fused in by RRF. The Kiwi instance is the one `_morph.kiwi()` shared with
# the statutes tool (each one costs about 250 MB).
_MORPH_KEEP_TAGS = ("NNG", "NNP", "NNB", "NR", "NP", "SL", "SN", "SH", "XR", "VV", "VA")


def _morphemes(query: str) -> list[str]:
    """The query's content morphemes, FTS5 operator words removed: Kiwi returns 'AND'/'OR'/'NOT' as content
    morphemes (SL), and the morpheme index holds them as rare words (high IDF) that would pull up unrelated
    English-prose judgments."""
    return [f for t in _kiwi().tokenize(query)
            if t.tag in _MORPH_KEEP_TAGS and (f := t.form.strip()) and f.upper() not in _FTS_OPERATORS]


def _morph_match(query: str) -> str:
    """Morpheme OR match expression (no length floor — covers the words trigram cannot match)."""
    return _quoted_or(list(dict.fromkeys(_morphemes(query)))[:FTS_OR_MAX_TOKENS])


def _rank(
    conn: sqlite3.Connection,
    table: str,
    weights: Sequence[float],
    match: str,
    limit: int,
    filters: dict[str, Any],
) -> list[int]:
    """BM25 ranking of one full-text index (`table`, with its column `weights`). match = `_or_match` or
    `_morph_match`.

    Filters (court level, court, year) are applied **in the SQL** — drawing the top pool first and filtering it
    afterwards would lose a sparse filter (a common keyword + '부산') entirely. Without filters there is no JOIN: an
    OR match on common words hits tens of thousands of rows, and looking each up makes the query 60 times slower
    (6 s against 0.09 s). The JOIN goes to prec_meta, the filter columns without the text (1 s, against 27 s on
    prec_cases)."""
    if not match:
        return []
    cond, args = _filter_sql("c.", **filters)
    if not cond:
        sql = f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY {bm25_expr(table, weights)} LIMIT ?"
    else:
        sql = (
            f"SELECT f.rowid FROM {table} f JOIN prec_meta c ON c.id = f.rowid "
            f"WHERE {' AND '.join([f'f.{table} MATCH ?', *cond])} ORDER BY {bm25_expr(f'f.{table}', weights)} LIMIT ?"
        )
    rows = conn.execute(sql, (match, *args, limit)).fetchall()
    return [r["rowid"] for r in rows]


# The source's line breaks in summary and reference_statute ('<br/>'; '<br>' in a few tables of the text). Those
# columns are in the full-text index, so the data keeps them and the output turns them into spaces.
_BR = re.compile(r"<br/?>")


def _strip_markup(text: str | None) -> str:
    """One line of plain text: line-break tags and runs of whitespace become one space.

    Whitespace is **collapsed** too: 30% of `reference_statute` values contain line breaks (shaped like
    `[1]\\n민법 제839조의2,\\n…`), which would break the one-line-per-field output."""
    return " ".join(_BR.sub(" ", text or "").split())


def _clip(text: str, limit: int) -> str:
    """`text`, or its first `limit` characters and a trailing `…` that says it was cut."""
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


# Images in the judgment text ('![](/case-img/19679/p1_1.png#w=50&px=525x81)' on its own line or mid-sentence, a few
# '<img src=…></img>'): a preview has nothing to show of them, so they are taken out before it is cut.
_IMAGE = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)\n*|<img\b[^>]*>(?:</img>)?")
# The piece left where a snippet starts or ends inside one ('…256/p2_2.png#w=19&px=252x74)', '![](/case-img/…').
_IMAGE_PIECE = re.compile(r"^…[\x21-\x7e]*[/#=&][\x21-\x7e]*\)|!\[[^\]\n]*(?:\]\([^)\n]*)?…?$")


def _drop_images(text: str) -> str:
    return _IMAGE.sub("", text)


def _fts_snippets(
    conn: sqlite3.Connection, ids: Sequence[int], match: str
) -> dict[int, tuple[str, str]]:
    """Per matched judgment: ``(excerpt, source kind)``.

    FTS's automatic column choice (-1) cannot tell generated_summary from the original text, so a generated
    summary would be mistaken for a direct quote of the judgment. Each column's marker is checked and the
    provenance is returned with the excerpt.
    """
    if not ids or not match:
        return {}
    ph = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""
        SELECT rowid,
               snippet(prec_cases_fts, 1, '⟦', '⟧', '…', {SNIPPET_TOKENS}) AS original_s,
               snippet(prec_cases_fts, 2, '⟦', '⟧', '…', {SNIPPET_TOKENS}) AS official_s,
               snippet(prec_cases_fts, 4, '⟦', '⟧', '…', {SNIPPET_TOKENS}) AS generated_s,
               snippet(prec_cases_fts, 3, '⟦', '⟧', '…', {SNIPPET_TOKENS}) AS statute_s
        FROM prec_cases_fts
        WHERE prec_cases_fts MATCH ? AND rowid IN ({ph})
        """,
        [match, *ids],
    ).fetchall()
    out: dict[int, tuple[str, str]] = {}
    # Prefer the original judgment/decision text; failing that, official summary → generated summary →
    # referenced-articles metadata.
    columns = (
        ("original_s", "original_text_excerpt"),
        ("official_s", "official_summary_excerpt"),
        ("generated_s", "generated_summary_excerpt"),
        ("statute_s", "reference_statute_metadata_excerpt"),
    )
    for row in rows:
        for key, source in columns:
            raw = _IMAGE_PIECE.sub("…", _drop_images(row[key] or ""))
            if "⟦" not in raw:  # no actual query hit in this column
                continue
            clean = _strip_markup(re.sub(r"[⟦⟧]", "", raw))
            if clean:
                out[row["rowid"]] = (clean, source)
                break
    return out


# ---------- excerpt at the match point (judgments the trigram snippet missed) ----------
# The trigram snippet centers a window on the match only for 3+ characters. For 2-character words
# ('살인' murder, '부당' unjust) the snippet is empty, so the match window is cut from the text directly to show
# *why it matched* instead of the head of the summary.


def _excerpt_around(text: str, terms: Sequence[str]) -> str | None:
    """Plain-text excerpt: EXCERPT_HALF characters on each side of the first occurrence of the *longest* of terms
    found in text, the earliest of equal length (None if there is none). One-character terms ('년', '가') are
    not anchors: they hit the judgment's heading first."""
    if not text or not terms:
        return None
    clean = _drop_images(_BR.sub(" ", text))
    found = [(-len(t), i, t) for t in terms if len(t) >= 2 for i in [clean.find(t)] if i != -1]
    if not found:
        return None
    _, pos, hit = min(found)
    start = max(0, pos - EXCERPT_HALF)
    end = min(len(clean), pos + len(hit) + EXCERPT_HALF)
    seg = " ".join(clean[start:end].split())
    if start > 0:
        seg = "…" + seg
    if end < len(clean):
        seg = seg + "…"
    return seg


def _preview_terms(query: str) -> list[str]:
    """Words to excerpt and pick holdings by — trigram words (3+ chars) and morphemes (incl. 2 chars), longest
    first (longer words are more distinctive, so they match first). Operator words are dropped, so an excerpt
    never anchors on the letters 'AND'."""
    toks = [w for w in _query_tokens(query) if len(w) >= 3] + _morphemes(query)
    return sorted(set(toks), key=len, reverse=True)


def _body_excerpts(
    conn: sqlite3.Connection,
    ids: Sequence[int],
    terms: Sequence[str],
) -> dict[int, tuple[str, str]]:
    """``{id: (excerpt, source kind)}`` — a match window from the official summary → referenced articles → head of
    text, for ids the trigram snippet missed. The AI generated_summary is excluded. content_md is cut at 50K."""
    if not ids or not terms:
        return {}
    ph = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""SELECT id, summary, reference_statute,
                   substr(COALESCE(content_md,''), 1, 50000) AS content_head
            FROM prec_cases WHERE id IN ({ph})""",
        list(ids),
    ).fetchall()
    out: dict[int, tuple[str, str]] = {}
    for r in rows:
        candidates = (
            (r["summary"] or "", "official_summary_excerpt"),
            (r["reference_statute"] or "", "reference_statute_metadata_excerpt"),
            (r["content_head"] or "", "original_text_excerpt"),
        )
        for text, source in candidates:
            seg = _excerpt_around(text, terms)
            if seg:
                out[r["id"]] = (seg, source)
                break
    return out


# ---------- issues decided (holdings) — split into items + pick the item the query points at ----------
# The issues decided are the court's own summary of the issues, shorter than the judgment text and closer to
# query vocabulary. They are in neither full-text index, so search does not find judgments by them; showing the
# item the query points at lets the model choose which judgment to read.
#
# They are split into items (`[1] … [2] …`), one legal proposition each, because cutting by character count
# halves a proposition and the first item alone is **the wrong issue 59% of the time** (over 3,015 real cases the
# item the query words hit most was item 1 40.7%, item 2 39.3%, item 3 or later 17.1%).
# Three kinds of item label. ⚠ Writing `[가-하]` swallows most Hangul syllables — the letters are listed
# one by one.
_GANADA = "가나다라마바사아자차카타파하"
_HOLD_HEAD = re.compile(r"^(?:\[\s*\d+\s*\]|[" + _GANADA + r"]\.|\d+\.)\s*\S")
_HOLD_INLINE_BRACKET = re.compile(r"(?<!\S)\[\s*(\d+)\s*\]")


def _inline_bracket_marks(text: str) -> list[int]:
    """Positions of `[1] … [2] …` item heads within one line. Empty list if fewer than 2 or out of order.

    ⚠ Accepted as items **only when ascending from 1**. Items often refer to other items
    ('[2] 위 [1]항의 소멸시효 …'), so cutting wherever a number appears would split an item in two at the
    reference. Same rule as the expected order in `_inline_ganada`."""
    marks = [(m.start(), int(m.group(1))) for m in _HOLD_INLINE_BRACKET.finditer(text)]
    if len(marks) < 2:
        return []
    if [n for _, n in marks] != list(range(1, len(marks) + 1)):
        return []
    return [pos for pos, _ in marks]


def _inline_ganada(text: str) -> list[str]:
    """Cut '가. … 나. … 다. …' run together on one line into items (versions without line breaks).

    Searched **only in the expected order** — otherwise the '다.' of '…하였다.' is caught as an item head.
    Treated as an item head only when the preceding character is a space, `)` or `]` (some versions write it
    attached, as in '(소극)다. …')."""
    starts: list[int] = []
    at = 0
    for ch in _GANADA:
        pos = text.find(ch + ".", at)
        while pos > 0 and text[pos - 1] not in " \t)]":   # mid-word is not an item head
            pos = text.find(ch + ".", pos + 1)
        if pos < 0:
            break
        starts.append(pos)
        at = pos + 2
    if len(starts) < 2:
        return []
    return _cut_at(text, starts)


def _cut_at(text: str, starts: list[int]) -> list[str]:
    """Cut at the item-head positions. **Text before the first head is not dropped** — some Constitutional
    Court issues decided open with a shared preamble, as in '1. … 가. … 나. …', and losing it leaves the
    remaining items without what they are about."""
    head = text[:starts[0]].strip()
    items = [text[a:b].strip() for a, b in zip(starts, starts[1:] + [len(text)])]
    if head:
        items[0] = head + " " + items[0]
    return items


def _holdings_items(text: str) -> list[str]:
    """Issues-decided text → **list of items** (in order of appearance). If it cannot be split, one whole item.

    Line breaks own the boundaries — they are present in 93% of the corpus and agree with the `[N]` count 99.3%
    of the time. Labels are used not as boundaries but for *verification*: a fragment that starts without a label
    is a long item split by a line break in the source, so it is reattached to the previous item (otherwise
    the proposition is halved). Only the order is needed, so the three numbering schemes (brackets, '가나다',
    digit-dot) are not read as numbers.

    Checked on 81,488 holdings: no characters lost, no empty items; where the label count and the item count
    disagree (2.44%), 98.9% are in-text references ('위 [1]항의 …') where the split is right."""
    segs = [" ".join(s.split()) for s in (text or "").split("\n")]
    segs = [s for s in segs if s]
    if not segs:
        return []
    if len(segs) == 1:
        # A single fragment may mean the labels came inline. ⚠ This branch must be checked **before** the
        # `_HOLD_HEAD` test — '가. … 나. …' is a label from the first character, so checked later it would fold
        # into 'a document with labels' and become one whole item.
        marks = _inline_bracket_marks(segs[0])
        if marks:
            return _cut_at(segs[0], marks)
        return _inline_ganada(segs[0]) or segs
    if any(_HOLD_HEAD.match(s) for s in segs):
        items: list[str] = []
        for seg in segs:
            if items and not _HOLD_HEAD.match(seg):
                items[-1] += " " + seg      # the same item, split by a line break
            else:
                items.append(seg)
        return items
    return segs


def _pick_holding(text: str, terms: Sequence[str]) -> str:
    """The one item of the issues decided that **the query words hit most** (first item on a tie or no
    match), cut at the length cap.

    The same rule as the preview showing 'why it matched' (`_excerpt_around`) — it does not judge the content
    but points at the match."""
    items = _holdings_items(text)
    if not items:
        return ""
    best = items[0]
    if terms and len(items) > 1:
        best = max(items, key=lambda it: sum(1 for t in terms if t in it))
    return _clip(best, HOLDING_MAX_CHARS)


# ---------- RRF fusion ----------

def _rrf_fuse(rankings: Sequence[Sequence[int]], *, limit: int) -> list[int]:
    """Reciprocal Rank Fusion. score(d) = Σ 1/(RRF_K + rank_r(d))."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ordered[:limit]]


# ---------- metadata hydration + filters ----------

def _hydrate(
    conn: sqlite3.Connection,
    case_ids: Sequence[int],
    court_level: str | None,
    year_from: int | None,
    year_to: int | None,
    cap: int | None = LIMIT,
    court_name: str | None = None,
) -> list[dict[str, Any]]:
    """Hydrate metadata in case_ids order. cap=None returns everything (for issue reordering).

    court_name: the part of a court name given by _check_court_name — '부산' → '부산지방법원'·'부산고등법원'…
      (Busan District/High Court). Orthogonal to court_level (instance).
    """
    if not case_ids:
        return []
    placeholders = ",".join("?" for _ in case_ids)
    cond, args = _filter_sql("", court_level, court_name, year_from, year_to)
    where = [f"id IN ({placeholders})", *cond]

    rows = conn.execute(
        f"""
        SELECT id, case_number, case_name, court_name, court_level, decision_date,
               COALESCE(decision_year, case_year) AS year, reference_statute, holdings,
               COALESCE(generated_summary, summary, content_md) AS display_summary,
               CASE
                   WHEN generated_summary IS NOT NULL THEN 'generated_summary_excerpt'
                   WHEN summary IS NOT NULL THEN 'official_summary_excerpt'
                   ELSE 'original_text_excerpt'
               END AS display_summary_source
        FROM prec_cases
        WHERE {' AND '.join(where)}
        """,
        [*case_ids, *args],
    ).fetchall()

    by_id = {r["id"]: r for r in rows}
    out: list[dict[str, Any]] = []
    for cid in case_ids:
        if cid not in by_id:
            continue
        r = by_id[cid]
        full_summary = _drop_images(r["display_summary"] or "").strip()
        out.append(
            {
                "id": r["id"],
                "case_number": r["case_number"],
                "case_name": r["case_name"],
                "court_name": r["court_name"],
                "court_level": r["court_level"],
                "decision_date": r["decision_date"],
                "year": r["year"],
                "reference_statute": r["reference_statute"],
                "_full_summary": full_summary,   # internal: preview fallback
                "_full_summary_source": r["display_summary_source"],
                "_holdings": r["holdings"],      # internal: `_pick_holding` picks the item
                "preview": "",
            }
        )
        if cap is not None and len(out) >= cap:
            break
    return out


# ---------- markdown serialization ----------

# provenance → the kind line the model reads. The kind itself says whether the text may be quoted: a kind whose
# name contains '원문' ("original text") may be quoted, any other may not — so never put '원문' in the name of a
# non-quotable kind. Only `original_text_excerpt` is original text.


def _statute_line(text: str) -> str:
    """Referenced articles on one line, cut at the holding cap with a trailing `…`."""
    return _clip(_strip_markup(text), HOLDING_MAX_CHARS)


_PREVIEW_KINDS = {
    "original_text_excerpt": "원문 발췌",
    "official_summary_excerpt": "공식 요약 발췌",
    "generated_summary_excerpt": "AI 요약 발췌",
    "reference_statute_metadata_excerpt": "참조조문 메타·판시 아님",
}


def _set_preview(match: dict[str, Any], text: str, source: str) -> None:
    """Always set the preview string and its provenance together, so the source cannot get lost."""
    match["preview"] = text
    match["preview_provenance"] = source


def _set_holding(match: dict[str, Any], terms: Sequence[str]) -> None:
    """Attach the issues-decided item the query points at. Judgments without any (most lower-court ones)
    get no such field at all."""
    holding = _pick_holding(match.get("_holdings") or "", terms)
    if holding:
        match["holding"] = holding


def _drop_preview_internals(match: dict[str, Any]) -> None:
    match.pop("_full_summary", None)
    match.pop("_full_summary_source", None)
    match.pop("_holdings", None)


def _format_response_md(resp: dict[str, Any]) -> str:
    """precedent_search response dict → markdown-KV string (better token efficiency)."""
    status = resp.get("status", "ok")
    lines: list[str] = [f"## status: {status}"]
    if "message" in resp:
        lines.append(f"- message: {resp['message']}")
    if resp.get("note"):
        lines.append(f"- note: {resp['note']}")
    matches = resp.get("matches") or []
    if matches:
        lines.append("## matches")
        for m in matches:
            block = [
                f"- id: {m['id']}",
                f"  url: https://lawful.crow-tit.com/cases/{m['id']}",
                f"  case_no: {m.get('case_number','')}",
                f"  case_name: {m.get('case_name','')}",
                f"  court: {court_label(m.get('court_name'), m.get('court_level'), m.get('decision_date'), m.get('year'))}",
            ]
            if m.get("reference_statute"):
                block.append(f"  statute: {_statute_line(m['reference_statute'])}")
            if m.get("holding"):
                # The court-written issues decided, marked as original text so the one quoting rule (quote only what
                # a tool marks '원문') covers them; a trailing `…` says the item was cut.
                block.append(f"  holding: {m['holding']}")
                block.append("  holding_kind: 판시사항 원문")
            if m.get("preview"):
                block.append(f"  preview: {m['preview']}")
                source = m.get("preview_provenance") or "unknown"
                # An unknown provenance is written under its own name — the name has no '원문', so the
                # positive-form rule automatically makes it non-quotable.
                block.append(f"  preview_kind: {_PREVIEW_KINDS.get(source, source)}")
            lines.append("\n".join(block))
    elif status == "ok":
        lines.append("## matches: (없음)")
    return "\n".join(lines)


# ---------- issue reordering (Jev) ----------
# The candidate pool found by query (LIMIT×OVERSAMPLE) is reordered against issue (the legal question the
# search is meant to answer). For each judgment, Jev (TypeSafe, via OpenRouter) is asked "did this judgment
# directly decide that question?", and LIMIT judgments come out in order of that probability. Without an issue,
# this stage does not exist.
#
# Measured offline on 855 KCL bar-exam statements over the same 35 candidates: top-7 hit rate BM25 75.0% → 82.5%,
# rank 1 44.4% → 52.9%. Given the query keywords instead, the same Jev gets 77.2%, indistinguishable from chance —
# the effect comes from **the question sentence**, not the reranker (bge and Qwen3 rerankers reach the same level
# when given the issue). 12% of the answers lie outside the candidate pool, which this stage cannot rescue —
# finding judgments is still the query's job.
#
# ⚠ Do not cut by a threshold — always LIMIT results. Cutting saves characters (−32% at threshold 0.5), but
#   the damage when the issue is misjudged grows, and there is no instrumentation to see it.
# ⚠ Do not mix with the BM25 rank (RRF). The overall hit rate is similar at 81.9%, but on fact-pattern items
#   it cuts the gain from 74.3% → 68.6%.
# ⚠ fail-open — if any key absence, error or timeout occurs, that search keeps BM25 order.
USE_JEV = os.environ.get("USE_JEV", "1") == "1"   # when off, BM25 order even if issue is given (read at import)
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
# Pin the release down to the date — it is an alpha endpoint, so a new release changes the scores. This is the
# release that was measured.
JEV_MODEL = "typesafe/jev-1.14-20261014"
# One search takes as long as the slowest of its parallel requests. Measured (2,988 searches): median 0.47 s,
# p99 1.22 s, max 1.91 s, and 0.3% of searches exceeded 1.5 s. A value that is not hit normally and sends the
# search back to BM25 only when upstream slows down.
JEV_BUDGET_S = 1.5
# Cap on Jev requests in flight at once. One search sends as many requests as it has candidates, so this is
# 3–4 searches' worth. Beyond it, requests queue, hit the budget and fall back to BM25 — the value keeps
# threads from multiplying under load.
JEV_MAX_INFLIGHT = 128
# The code checks only the length cap (what to write is the docstring's business). Measured issues: median
# 61 chars, max 121.
ISSUE_MAX_CHARS = 300
# Both questions are exactly the measured request (whether sending one question alone gives the same
# probabilities was not measured). Order is set by the three-way (`세갈래`) question's `관련_판단` ("directly
# decided") probability. At a fixed 7 results, the yes/no question hits the same (82.5 vs 82.6%), and should a
# threshold ever be revisited, the three-way one is the one that cuts evenly between 0.3 and 0.5.
_JEV_QUESTIONS = {
    "기본": {
        "type": "noul",
        "instructions": "`판례`는 `찾는_내용`과 관련 있는 판례인가?",
        "criteria": {"true": "찾는 내용과 관련 있는 판례", "false": "찾는 내용과 관련 없는 판례"},
    },
    "세갈래": {
        "type": "choice",
        "instructions": "`판례`는 `찾는_내용`과 어떤 관계인가?",
        "criteria": {
            "관련_판단": "`찾는_내용`의 법적 쟁점을 직접 판단한 판례",
            "다른_쟁점": "같은 법 분야이지만 다른 쟁점을 판단한 판례",
            "무관": "`찾는_내용`과 관련 없는 판례",
        },
    },
}

_log = logging.getLogger(__name__)


@functools.cache
def _jev_pool() -> tuple[httpx.Client, ThreadPoolExecutor]:
    """One set per process — the number of concurrent Jev requests is bounded here. Connections are reused
    only by searches that follow within 5 seconds (httpx default keepalive); one search's requests open their
    connections side by side, which adds only tens of ms (HTTP/2 on one connection is no faster)."""
    limits = httpx.Limits(max_connections=JEV_MAX_INFLIGHT, max_keepalive_connections=JEV_MAX_INFLIGHT)
    return (
        httpx.Client(timeout=JEV_BUDGET_S, limits=limits),
        ThreadPoolExecutor(max_workers=JEV_MAX_INFLIGHT, thread_name_prefix="jev"),
    )


def _jev_card(match: dict[str, Any]) -> str:
    """The judgment text sent to Jev = exactly the result block the model receives (the measured condition).
    Sending only the issues decided would score lower-court judgments, which have none, on their case name
    alone."""
    return "\n".join(_format_response_md({"status": "ok", "matches": [match]}).splitlines()[2:])


def _jev_relevance(client: httpx.Client, key: str, issue: str, card: str, record: dict[str, Any]) -> float:
    """p(관련_판단) for one card; `record` (a telemetry event) gets the request, the response and the time."""
    body = {"model": JEV_MODEL, "state": {"찾는_내용": issue, "판례": card}, "questions": _JEV_QUESTIONS}
    start = time.monotonic()
    record.update(status="started", request=body)
    try:
        resp = client.post(JEV_URL, headers={"Authorization": f"Bearer {key}"}, json=body)
        resp.raise_for_status()
        data = resp.json()
        score = float(data["answers"]["세갈래"]["probabilities"]["관련_판단"])
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        fee = usage.get("cost", data.get("cost"))
        fee = fee if type(fee) in (int, float) and 0 <= fee < float("inf") else None
        record.update(status="ok", response=data, reported_cost_usd=fee)
        return score
    except Exception as exc:
        record.update(status="error", error=type(exc).__name__)
        raise
    finally:
        record["seconds"] = round(time.monotonic() - start, 4)


def _jev_scores(issue: str, cards: Sequence[str]) -> tuple[list[float] | None, str]:
    """Per card, p(관련_판단) and a one-word outcome. If any is missing, returns without scores (fail-open)."""
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        return None, "no-key"
    client, pool = _jev_pool()
    events = [telemetry.emit({"kind": "jev_request", "model": JEV_MODEL, "endpoint": JEV_URL,
                              "status": "queued", "reported_cost_usd": None}) for _ in cards]
    futures = [pool.submit(_jev_relevance, client, key, issue, card, event) for card, event in zip(cards, events, strict=True)]
    # A single failure already settles the search on BM25 order, so the rest are not waited for.
    done, pending = wait(futures, timeout=JEV_BUDGET_S, return_when=FIRST_EXCEPTION)
    for f, event in zip(futures, events, strict=True):
        if f in pending and f.cancel():
            event["status"] = "cancelled_before_request"
    # Whether it is a status code or the response shape, only that search falls back to BM25.
    e = next((f.exception() for f in done if f.exception()), None)
    if e is not None:
        status = getattr(getattr(e, "response", None), "status_code", None)
        return None, f"error:{type(e).__name__}" + (f":{status}" if status else "")
    if pending:
        return None, "timeout"
    return [f.result() for f in futures], "ok"


def _order_by_issue(issue: str, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """matches (BM25 order) → by relevance to issue. Ties keep BM25 order.

    The log records only outcome, count and time — the issue sentence is never written."""
    t0 = time.monotonic()
    scores, outcome = _jev_scores(issue, [_jev_card(m) for m in matches])
    telemetry.emit({"kind": "jev_rerank", "model": JEV_MODEL, "candidates": len(matches),
                    "outcome": outcome, "fallback": scores is None,
                    "seconds": round(time.monotonic() - t0, 4)})
    _log.info("[issue-rerank] %s n=%d %.0fms", outcome, len(matches), (time.monotonic() - t0) * 1000)
    if scores is None:
        return matches
    order = sorted(range(len(matches)), key=lambda i: (-scores[i], i))
    return [matches[i] for i in order]


# ---------- public tool ----------

def precedent_search(
    ctx: RunContext[HarnessDeps],
    query: str | None = None,
    issue: str | None = None,
    case_number: str | None = None,
    court_level: str | None = None,
    court_name: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> str:
    """판례 검색 — 키워드나 사건번호로 판례(대법원·하급심·헌법재판소)를 찾아 최대 7건을 돌려줍니다.

    - query(키워드 검색)나 case_number(사건번호 조회) 중 하나는 꼭 넣습니다.
    - issue에 이 검색으로 답하려는 법적 질문을 한 문장으로 주면, 그 질문을 직접 판단한 판례가 앞에 옵니다.
    - 결과의 조문(statute)은 그 판결 선고 당시 판본의 번호이고, 판시사항(holding)은 query에 가장 가까운 항
      하나만 옵니다.
    - 판결문 본문의 판단 이유·법리는 결과의 id로 precedent_dive를 불러 읽습니다.

    Args:
      query: 사실관계·쟁점·죄명의 핵심 낱말 여러 개(사건번호 제외).
      issue: 결과 순서를 정할 법적 질문 한 문장(당사자·날짜·금액은 빼고).
      case_number: 사건번호(예: "2010다89012").
      court_level: '1심' | '2심' | '대법원' | '헌재'.
      court_name: 법원 이름 일부(예: "부산", "특허법원").
      year_from: 선고 연도 범위 시작(선고일이 없으면 접수 연도).
      year_to: 선고 연도 범위 끝.
    """
    query = coerce_str(query)
    issue = (coerce_str(issue) or "")[:ISSUE_MAX_CHARS].strip()
    case_number = coerce_str(case_number)
    court_level = coerce_str(court_level)
    court_level, court_level_note = _normalize_court_level(court_level)
    court_name = coerce_str(court_name)
    year_from = coerce_int(year_from)
    year_to = coerce_int(year_to)

    # Cross-required/minimum-length checks fail loud before opening the 14GB DB. The case-number string is
    # normalized here too, so a valid case_number takes the exact-lookup path even when a short query comes
    # with it.
    cno = _extract_case_number(case_number) if case_number else None
    # A case_number that holds no readable number ('2016두') is said so, instead of telling the model to put the
    # number in case_number.
    unread = f"case_number '{case_number}' 에서 사건번호를 읽지 못했습니다(예: 2010다89012)" if case_number and not cno else None
    if not cno and not query:
        return _format_response_md({
            "status": "error",
            "message": (
                f"{unread} — 번호를 고치거나 query(사실관계 키워드)로 검색하세요." if unread else
                "query(사실관계 키워드) 또는 case_number(사건번호) 중 하나는 필요합니다. "
                "사건번호가 있으면 case_number 인자에 넣으세요."
            ),
            "matches": [],
        })
    if not cno and query and len(query) < 2:
        return _format_response_md({
            "status": "error",
            "message": (
                f"query 길이 {len(query)}자 — 2자 이상 필요. "
                "사실관계 키워드 또는 죄명+맥락(예: '사기 피해 1억')으로 재호출하세요."
            ),
            "matches": [],
        })

    conn = open_db()
    try:
        court_name, court_name_note = _check_court_name(conn, court_name)
        # Direct case-number lookup (FTS cannot catch it — case_number is not indexed).
        # Separation of roles — case numbers are accepted only through the explicit case_number slot. The
        # docstring steers the LLM to recognize a case number and put it in this argument, and query is kept
        # purely for fact-pattern keyword search (no case number is extracted from the query text — so the
        # roles of the two paths do not blur). Even when it comes in citation form
        # ('대법원 … 선고 2010다89012 판결'), only case_number is normalized and looked up.
        cno_miss = None
        if cno:
            ids = _case_number_ids(conn, cno, LIMIT)
            matches = _hydrate(
                conn, ids, court_level, year_from, year_to,
                cap=LIMIT, court_name=court_name,
            )
            if matches:
                # Direct case-number lookup has no query words to choose by — the first item is the main issue.
                cno_terms = _preview_terms(query) if query else []
                for m in matches:
                    _set_preview(
                        m,
                        m["_full_summary"][:PREVIEW_FALLBACK_CHARS].replace("\n", " "),
                        m["_full_summary_source"],
                    )
                    _set_holding(m, cno_terms)
                    _drop_preview_internals(m)
                resp = {"status": "ok", "matches": matches}
                notes = [n for n in (court_level_note, court_name_note) if n]
                if notes:
                    resp["note"] = " / ".join(notes)
                return _format_response_md(resp)
            # There are two reasons for not finding it — they are written out separately so the model does not
            # conclude 'no such judgment exists'. The number may exist but be excluded by the court level /
            # court name / year conditions (e.g. a district court given for an appeal). If the number does not
            # exist at all, it is a wrong number or a judgment not in the corpus.
            cno_miss = (
                f"사건번호 '{case_number}' 판례는 있으나 지정한 심급·법원·연도 조건에 맞지 않습니다"
                if ids else
                f"사건번호 '{case_number}' 에 해당하는 판례를 찾지 못했습니다"
                "(번호가 틀렸거나 코퍼스에 수록되지 않은 판결)"
            )
            if not query:
                return _format_response_md({
                    "status": "ok",
                    "message": cno_miss + (
                        ". 조건을 빼고 다시 조회하세요." if ids else
                        ". 사건번호를 확인하거나 사실관계 키워드(query)로 재검색하세요."
                    ),
                    "matches": [],
                })
            # Otherwise (query present), fall back to the keyword search below (in case of format differences or
            # typos). The note says that those results are not the requested case — without it, the first result
            # would be read as that case.

        # ----- keyword search (query required) -----
        # Without a cno, query presence and minimum length were already validated before opening the DB.
        pool_size = LIMIT * OVERSAMPLE
        tri_match = _or_match(query)       # trigram: 3+ char partial matching, typo-tolerant
        morph_match = _morph_match(query)  # morpheme: word level (incl. 2-char '살인'·'사기')
        # Ordering is by relevance (BM25) alone; the filters apply inside both rankings (see `_rank`).
        filters = dict(court_level=court_level, court_name=court_name, year_from=year_from, year_to=year_to)
        tri_ids = _rank(conn, "prec_cases_fts", BM25_WEIGHTS, tri_match, pool_size, filters)
        # The morpheme index has one column, so it has no column weights.
        morph_ids = _rank(conn, "prec_cases_morph_fts", (), morph_match, pool_size, filters)
        # RRF-fuse the two rankings, trigram and text morpheme.
        rankings = [r for r in (tri_ids, morph_ids) if r]
        ids = _rrf_fuse(rankings, limit=pool_size) if rankings else []
        # With an issue, build cards for the whole pool, reorder, then cut to LIMIT (the "issue reordering"
        # section).
        reorder = bool(issue) and USE_JEV
        matches = _hydrate(
            conn, ids, court_level, year_from, year_to,
            cap=None if reorder else LIMIT, court_name=court_name,
        ) if ids else []
        # Excerpts prefer the trigram snippet (the matched phrase); morpheme-only matches fall back to the
        # head of the summary.
        snips = _fts_snippets(conn, [m["id"] for m in matches], tri_match) if (matches and tri_match) else {}
        # For trigram snippet misses (2-char morpheme-only), the match window is excerpted from the text
        # directly to show the LLM *why it matched* (instead of the head of the summary). Only the missed
        # ids are fetched, in one batch.
        missed = [m["id"] for m in matches if m["id"] not in snips]
        terms = _preview_terms(query)
        excerpts = _body_excerpts(conn, missed, terms) if missed else {}
        for m in matches:
            preview = snips.get(m["id"]) or excerpts.get(m["id"])
            if preview:
                text, source = preview
            else:
                text = m["_full_summary"][:PREVIEW_FALLBACK_CHARS].replace("\n", " ")
                source = m["_full_summary_source"]
            _set_preview(m, text, source)
            _set_holding(m, terms)
            _drop_preview_internals(m)
        if reorder and len(matches) > 1:
            matches = _order_by_issue(issue, matches)
        resp = {"status": "ok", "matches": matches[:LIMIT]}
        # fail-loud notes: invalid court_level or court_name (filter ignored) · a case-number lookup that missed ·
        # case-number slot misuse. Slot misuse is noted even when there are results — judgments **citing**
        # that number come up, so the first result is easily read as that case.
        notes = [court_level_note, court_name_note, unread and f"{unread} — 빼고 query 로만 찾았습니다"]
        if cno_miss:
            notes.append(f"{cno_miss} — 아래는 query 키워드로 찾은 판례입니다")
        notes.append(_case_no_in_query_hint(query))
        notes = [n for n in notes if n]
        if notes:
            resp["note"] = " / ".join(notes)
        return _format_response_md(resp)
    finally:
        conn.close()
