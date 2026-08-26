"""Retrieve judgments by case number or by the facts of a matter.

Two retrieval paths, selected by USE_DENSE:

- Lexical (the default). Two full-text indexes over the same judgments —
  character trigrams and Kiwi morphemes — each queried in OR mode and fused
  with reciprocal rank fusion. No external service is involved.
- Dense (opt-in). Adds an embedding KNN and a reranker, which need an
  embedding service configured.

The default is lexical because it measured at least as good: in a known-item
retrieval A/B (N=140) OR-mode BM25 matched dense on short queries and beat it
on richer ones, at a fraction of the latency.

OR mode is not incidental — it is the whole result. The same comparison run
with FTS5's implicit AND collapsed to 0.11 recall, because a long
natural-language query then demands that every token appear in one judgment.

Two indexes rather than one because neither covers Korean alone: trigrams
handle partial and misspelled words but cannot represent a two-character
term, and Korean charge names are frequently two characters (사기, 절도,
폭행). The morpheme index catches exactly those.
"""
from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Sequence

from pydantic_ai import RunContext

from ..config import case_url_base
from ..deps import HarnessDeps, open_db
from ._coerce import coerce_int, coerce_str
from ._dedup import dedup_guard
from ._morph import kiwi as _kiwi

EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/llama-nemotron-embed-1b-v2")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "nvidia/llama-nemotron-rerank-1b-v2")
# Matryoshka truncation width; must match the stored vector dimension.
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))

# Retrieval tuning. Not exposed as tool arguments: a caller has no basis to
# choose them, and they are not what a caller is reasoning about.
LIMIT = 7            # results returned. Recall at 10 is much better than at 5
                     #   for short queries, and BM25 ranking is coarse enough
                     #   that a few extra candidates hedge cheaply.
                     #   Eight until each result grew a holding line. At eight
                     #   the response is 20% longer; at seven it is 5%, which
                     #   is flat in practice (3,922 -> 4,132 characters over
                     #   150 production queries). What seven gives up is 7.5%
                     #   of the cases the model actually cited or dived into —
                     #   they were the eighth result — and six would give up
                     #   15.7%. Do not go to five: that loses 24.1%.
RRF_K = 60           # reciprocal rank fusion constant
OVERSAMPLE = 5       # candidate pool = LIMIT * OVERSAMPLE, so filters can bite
                     #   without leaving the result short
USE_RERANK = os.environ.get("USE_RERANK", "1") == "1"
# Dense retrieval, off by default. Measured against the lexical path on 140
# known-item queries: equal on short queries, worse on rich ones (0.80 vs
# 1.00), and it adds a per-call embedding round trip. Turn it on to reproduce
# the hybrid configuration; leave it off to run with no external service.
USE_DENSE = os.environ.get("USE_DENSE", "0") == "1"
FTS_OR_MAX_TOKENS = int(os.environ.get("FTS_OR_MAX_TOKENS", "40"))   # cap on OR-query tokens
SNIPPET_TOKENS = int(os.environ.get("SNIPPET_TOKENS", "256"))        # snippet width, roughly in
                                                                     #   characters; ~250 is enough
                                                                     #   to carry a holding, and in
                                                                     #   line with what search APIs
                                                                     #   return
PREVIEW_TOP_K = int(os.environ.get("PREVIEW_TOP_K", "3"))            # sentences kept in a preview
PREVIEW_MAX_CHARS = int(os.environ.get("PREVIEW_MAX_CHARS", "400"))  # preview cap; one sentence always survives
RERANK_INPUT_MAX_CHARS = int(os.environ.get("RERANK_INPUT_MAX_CHARS", "4000"))  # guard against outlier-length input
PREVIEW_FALLBACK_CHARS = 200                                         # cut used when reranking is off
# Display cap for one holding item. The cap exists for the tail: among the
# cases search returns, holdings run to 1,001 characters at p99 and 18,524 at
# the longest — rows where a full judgment was written into the `holdings`
# column, 36 of 81,488. Uncapped, one search could add 19,060 characters and
# quintuple the response.
#
# The value trades truncation against characters spent (882 selected items
# over 200 production queries: median 122, p90 309). At 150 characters 38% of
# items are cut, at 200 25%, at 300 11% — and 200 -> 300 costs 72 characters
# per search, 2% of the response. A legal proposition loses most of its worth
# cut in half, so that 2% buys truncation down to under a third. Past 400 the
# curve flattens: another 31 characters for 5%.
HOLDING_MAX_CHARS = 300

from ._fts import safe_fts_query as _safe_fts_query


# ---------- direct routing by case number ----------
# Case numbers are not in the full-text index, so neither search path can
# find one. The format is regular — year, case-type marker, serial — so it is
# detected by pattern and routed to a direct lookup instead.
_CASE_NO_RE = re.compile(r"\d{2,4}[가-힣]{1,5}\d+")
# A case number inside a full citation. Korean citations read "court +
# date + 선고 (decided) + case number + 판결 (judgment)", so the number
# arrives wrapped in text and cannot be matched whole.
#
# The words 선고, 판결 and 결정 make safe anchors: none of them appears in the
# 167 case-type markers, so a token adjacent to one is a case number rather
# than a coincidence. Only adjacency counts, which is why an ordinary query
# mentioning a judgment does not trip this.
_CASE_NO_CITED_RES = (
    re.compile(r"선고(\d{2,4}[가-힣]{1,5}\d+)"),            # '…선고 2010다89012 …'
    re.compile(r"(\d{2,4}[가-힣]{1,5}\d+)(?=판결|결정)"),   # '… 2018노1234 판결'
)


def _extract_case_number(query: str) -> str | None:
    """Extract a case number to route on, or None. Two forms are accepted.

    A full citation — "대법원 2010. 5. 26. 선고 2010다89012 판결" — yields the
    number next to its anchor word, and only there, so surrounding court
    names and dates do not matter and ordinary prose is not misread.

    A bare case number, possibly with a trailing word, must match the whole
    query. Anything less would catch statutory references like 민법 제750조,
    which have the same shape as a case number.
    """
    raw = (query or "").strip().replace(" ", "")
    for rx in _CASE_NO_CITED_RES:
        m = rx.search(raw)
        if m:
            return m.group(1)
    q = re.sub(r"(전원합의체|판결|결정|선고|사건|판례)$", "", raw)
    return q if _CASE_NO_RE.fullmatch(q) else None


# ---------- court level: normalise, and say so when it cannot ----------
# The column holds four values. A caller naturally passes a court's name
# instead ("High Court", "District Court"), which matches nothing and makes
# the whole search return empty with no indication why. Known names are
# mapped; anything else drops the filter and says it did.
_VALID_COURT_LEVELS = ("1심", "2심", "대법원", "헌재")
_COURT_LEVEL_ALIASES = {
    "고등법원": "2심", "고법": "2심", "항소심": "2심", "2심법원": "2심",
    "지방법원": "1심", "지법": "1심", "1심법원": "1심", "단독": "1심", "1심판결": "1심",
    "대법": "대법원", "대법원판결": "대법원",
    "헌법재판소": "헌재",
}


def _normalize_court_level(cl: str | None) -> tuple[str | None, str | None]:
    """court_level -> (stored value or None, note or None).

    An unrecognised value drops the filter and returns a note, rather than
    filtering everything away silently.
    """
    if not cl:
        return None, None
    if cl in _VALID_COURT_LEVELS:
        return cl, None
    if cl in _COURT_LEVEL_ALIASES:
        return _COURT_LEVEL_ALIASES[cl], None
    return None, f"court_level '{cl}' 은 인식 못 해 무시함 — 유효값: 1심/2심/대법원/헌재"


def _case_no_in_query_hint(query: str | None) -> str | None:
    """Hint for a caller who put a case number in the free-text slot."""
    if _CASE_NO_RE.fullmatch((query or "").replace(" ", "")):
        return ("query 가 사건번호 형식입니다 — 사건번호로 특정 판례를 찾으려면 "
                "case_number 인자에 넣으세요(query 는 사실관계 키워드 전용).")
    return None


# ---------- rankers ----------

def _fts_rank(conn: sqlite3.Connection, query: str, limit: int) -> list[int]:
    safe = _safe_fts_query(query)
    words = [w for w in safe.split() if len(w) >= 3]
    if not words:
        return []
    rows = conn.execute(
        """
        SELECT rowid FROM prec_cases_fts
        WHERE prec_cases_fts MATCH ?
        ORDER BY bm25(prec_cases_fts) LIMIT ?
        """,
        (safe, limit),
    ).fetchall()
    return [r["rowid"] for r in rows]


def _or_match(query: str) -> str:
    """Build an OR match expression from the query's distinctive tokens.

    FTS5 treats a bare multi-word query as AND, so a sentence-length query
    demands that every token appear in one judgment and returns nothing.
    Searching bag-of-words instead took known-item recall from 0.11 to 1.00
    over 140 queries. Tokens are at least three characters (the trigram
    floor), deduplicated and capped.
    """
    toks = [w for w in _safe_fts_query(query).split() if len(w) >= 3]
    seen: set[str] = set()
    toks = [t for t in toks if not (t in seen or seen.add(t))][:FTS_OR_MAX_TOKENS]
    return " OR ".join(toks)


_FILTER_TABLE: str | None = None


def _filter_table(conn: sqlite3.Connection) -> str:
    """Which table to join for filters.

    Judgment bodies dominate the corpus, so joining the main table pulls
    those pages through for rows that are only being filtered on. A slim
    metadata copy without the bodies is used when the corpus has one.
    있으면 그걸 쓴다 — 필터 검색 시 매치 수만 건을 prec_cases 에서 룩업하면 본문 페이지까지
    throttled 디스크에서 읽혀 폭증(실측 27s)하는데, prec_meta(수 MB, 캐시 상주)로 JOIN 하면 1s.
    Falls back to the main table when it does not. Resolved once per process.
    """
    global _FILTER_TABLE
    if _FILTER_TABLE is None:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prec_meta'"
        ).fetchone()
        _FILTER_TABLE = "prec_meta" if row else "prec_cases"
    return _FILTER_TABLE


def _fts_or_rank(
    conn: sqlite3.Connection,
    match: str,
    limit: int,
    *,
    court_level: str | None = None,
    court_name: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[int]:
    """BM25 ranking over the trigram index, in OR mode.

    필터(심급·법원/지역·연도)는 prec_cases JOIN 으로 **SQL 레벨에서** 적용한다 — 상위 pool 을
    먼저 뽑고 나중에 거르면 희소 필터(예: 흔한 키워드+'부산')가 전역 상위권 밖이라 통째 사라진다
    """
    if not match:
        return []
    cond = ["f.prec_cases_fts MATCH ?"]
    args: list[Any] = [match]
    if court_level:
        cond.append("c.court_level = ?"); args.append(court_level)
    if court_name:
        cond.append("c.court_name LIKE ?"); args.append(f"%{court_name}%")
    if year_from is not None:
        cond.append("COALESCE(c.decision_year, c.case_year) >= ?"); args.append(year_from)
    if year_to is not None:
        cond.append("COALESCE(c.decision_year, c.case_year) <= ?"); args.append(year_to)
    # Skip the join when nothing is being filtered. An OR match on common
    # tokens hits tens of thousands of rows, and looking every one of them up
    # to satisfy a join costs about 60x the FTS scan alone (5.97s versus
    # 0.09s measured). The join exists only for the filters, so it is added
    # only when there are filters.
    if len(cond) == 1:
        rows = conn.execute(
            "SELECT rowid FROM prec_cases_fts WHERE prec_cases_fts MATCH ? "
            "ORDER BY bm25(prec_cases_fts) LIMIT ?",
            (match, limit),
        ).fetchall()
        return [r["rowid"] for r in rows]
    # Join the slim metadata table, not the one carrying judgment bodies.
    sql = (
        f"SELECT f.rowid FROM prec_cases_fts f JOIN {_filter_table(conn)} c ON c.id = f.rowid "
        f"WHERE {' AND '.join(cond)} ORDER BY bm25(f.prec_cases_fts) LIMIT ?"
    )
    rows = conn.execute(sql, (*args, limit)).fetchall()
    return [r["rowid"] for r in rows]


# ---------- morpheme index: the two-character problem ----------
# A trigram index cannot form a trigram from a two-character token, so
# 살인 (murder) and 사기 (fraud) — ordinary charge names — return nothing.
# A second FTS built over Kiwi morphemes indexes whole words, and the two
# are fused by RRF. The Kiwi instance is shared with the statute tool
# through `_morph.kiwi()`: a per-module cache would load a second analyser
# into the process, measured at +246MB resident.
_MORPH_KEEP_TAGS = ("NNG", "NNP", "NNB", "NR", "NP", "SL", "SN", "SH", "XR", "VV", "VA")


def _morph_match(query: str) -> str:
    """Build an OR match expression from morphemes; no length floor here.

    Each token is quoted so a morpheme that happens to spell an FTS5
    operator is read as a search term rather than as syntax.
    """
    try:
        toks = [t.form.strip() for t in _kiwi().tokenize(query or "")
                if t.tag in _MORPH_KEEP_TAGS and t.form.strip()]
    except Exception:
        return ""
    toks = list(dict.fromkeys(toks))[:FTS_OR_MAX_TOKENS]
    return " OR ".join(f'"{t}"' for t in toks)


def _morph_rank(
    conn: sqlite3.Connection,
    match: str,
    limit: int,
    *,
    court_level: str | None = None,
    court_name: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[int]:
    """BM25 ranking over the morpheme index.

    Returns nothing when the corpus has no morpheme table, so a corpus built
    without one still searches on trigrams alone.
    """
    if not match:
        return []
    cond = ["f.prec_cases_morph_fts MATCH ?"]
    args: list[Any] = [match]
    if court_level:
        cond.append("c.court_level = ?"); args.append(court_level)
    if court_name:
        cond.append("c.court_name LIKE ?"); args.append(f"%{court_name}%")
    if year_from is not None:
        cond.append("COALESCE(c.decision_year, c.case_year) >= ?"); args.append(year_from)
    if year_to is not None:
        cond.append("COALESCE(c.decision_year, c.case_year) <= ?"); args.append(year_to)
    # Same reason as the trigram ranker: a common two-character token matches
    # tens of thousands of rows, and joining them all costs 6s against 0.02s.
    try:
        if len(cond) == 1:
            rows = conn.execute(
                "SELECT rowid FROM prec_cases_morph_fts WHERE prec_cases_morph_fts MATCH ? "
                "ORDER BY bm25(prec_cases_morph_fts) LIMIT ?",
                (match, limit),
            ).fetchall()
            return [r["rowid"] for r in rows]
        sql = (
            f"SELECT f.rowid FROM prec_cases_morph_fts f JOIN {_filter_table(conn)} c ON c.id = f.rowid "
            f"WHERE {' AND '.join(cond)} ORDER BY bm25(f.prec_cases_morph_fts) LIMIT ?"
        )
        rows = conn.execute(sql, (*args, limit)).fetchall()
    except sqlite3.OperationalError:
        return []  # no morpheme index in this corpus
    return [r["rowid"] for r in rows]


def _fts_snippets(
    conn: sqlite3.Connection, ids: Sequence[int], match: str
) -> dict[int, tuple[str, str]]:
    """Per judgment: (excerpt, where it came from).

    FTS5 will pick a column on its own, and the result gives no indication
    which one it chose — so an excerpt from a generated summary looks
    identical to one from the judgment itself, and would be quoted as if it
    were. Checking each column for the match marker recovers the provenance,
    which is what tells the caller whether the text may be quoted.
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
    # Prefer the judgment text itself; fall back to the official summary,
    # then a generated one, then the cited-statute metadata.
    columns = (
        ("original_s", "original_text_excerpt"),
        ("official_s", "official_summary_excerpt"),
        ("generated_s", "generated_summary_excerpt"),
        ("statute_s", "reference_statute_metadata_excerpt"),
    )
    for row in rows:
        for key, source in columns:
            raw = row[key] or ""
            if "⟦" not in raw:  # no match marker: this column did not hit
                continue
            clean = re.sub(r"[\u27e6\u27e7]", "", raw)
            clean = " ".join(re.sub(r"<br\s*/?>", " ", clean).split())
            if clean:
                out[row["rowid"]] = (clean, source)
                break
    return out


# ---------- excerpt around the match ----------
# FTS5's own snippet only centres on tokens the trigram index can represent.
# A judgment matched on a two-character morpheme therefore gets an empty
# snippet and falls back to the head of a summary, which does not show why
# it matched. Extracting the window around the term directly means the
# preview shows the passage the caller was searching for.
_BR_RE = re.compile(r"<br\s*/?>", re.I)


def _excerpt_around(text: str, terms: Sequence[str], half: int = 60) -> str | None:
    """Plain-text window centred on the earliest occurrence of any term."""
    if not text or not terms:
        return None
    clean = _BR_RE.sub(" ", text)
    pos, hit = -1, None
    for t in terms:  # longest first, so a tie prefers the more specific term
        i = clean.find(t)
        if i != -1 and (pos == -1 or i < pos):
            pos, hit = i, t
    if pos == -1:
        return None
    start = max(0, pos - half)
    end = min(len(clean), pos + len(hit) + half)
    seg = " ".join(clean[start:end].split())
    if start > 0:
        seg = "…" + seg
    if end < len(clean):
        seg = seg + "…"
    return seg


def _preview_terms(query: str) -> list[str]:
    """Terms to anchor an excerpt on: trigram words plus morphemes.

    Longest first — a longer term is more distinctive, so it makes the
    better centre for a window.
    """
    toks = [w for w in _safe_fts_query(query).split() if len(w) >= 3]
    try:
        toks += [t.form for t in _kiwi().tokenize(query or "")
                 if t.tag in _MORPH_KEEP_TAGS and t.form.strip()]
    except Exception:
        pass
    return sorted({t for t in toks if t}, key=len, reverse=True)


def _body_excerpts(
    conn: sqlite3.Connection,
    ids: Sequence[int],
    terms: Sequence[str],
    *,
    with_provenance: bool = False,
) -> dict[int, str] | dict[int, tuple[str, str]]:
    """Excerpt a window for judgments the trigram snippet missed.

    AI generated_summary는 제외한다. ``with_provenance``이면 ``(발췌, 출처종류)``를 반환하고,
    기본은 웹 검색 공유 계약을 위해 기존 ``{id: 평문}`` 형태를 유지한다. content_md는 50K 컷.
    """
    if not ids or not terms:
        return {}
    ph = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""SELECT id, summary, reference_statute,
                   substr(COALESCE(content_md,''), 1, 50000) AS content_head
            FROM prec_cases WHERE id IN ({ph})""",
        list(ids),
    ).fetchall()
    out: dict[int, Any] = {}
    for r in rows:
        if not with_provenance:
            # Combined path keeps one ordering: summary, cited statutes, body.
            body = " ".join((
                r["summary"] or "",
                r["reference_statute"] or "",
                r["content_head"] or "",
            ))
            seg = _excerpt_around(body, terms)
            if seg:
                out[r["id"]] = seg
            continue
        # Split by source field, same priority order, so the caller can see
        # which field the excerpt came from.
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


# ---------- holdings: split into items, pick the one the query points at ----
# A holding is the court's own summary of what the case decided. It is shorter
# than the judgment and closer to the words a query uses, and 81,488 cases
# carry one (91.7% of the Supreme Court's). It is in no index — neither the
# trigram FTS nor the morpheme table — so search cannot *find* by it, and the
# same sentences are not in the body either (a holding's first 40 characters
# appear in content_md 3% of the time). So the job here is not finding but
# *choosing*: showing the holdings of results already retrieved lets the model
# pick which case to dive into.
#
# Why not just truncate: holdings divide into `[1] … [2] …`, one legal
# proposition per item. Cutting by character count halves a proposition, and
# taking only the first item answers a different question than the one asked
# 59% of the time (over 3,015 production cases the item matching the query
# best was the first 40.7%, the second 39.3%, the third or later 17.1%).
_HOLD_BR = re.compile(r"<br\s*/?>", re.I)
# Three families of item label. Writing `[가-하]` would swallow most Hangul
# syllables, so the letters are listed.
_GANADA = "가나다라마바사아자차카타파하"
_HOLD_HEAD = re.compile(r"^(?:\[\s*\d+\s*\]|[" + _GANADA + r"]\.|\d+\.)\s*\S")
_HOLD_INLINE_BRACKET = re.compile(r"(?<!\S)\[\s*(\d+)\s*\]")


def _inline_bracket_marks(text: str) -> list[int]:
    """Offsets of `[1] … [2] …` item heads within one line. Empty unless there
    are at least two and they run in order.

    They must start at 1 and ascend. Holdings often refer to their own other
    items ('[2] 위 [1]항의 소멸시효 …', 1,970 cases in the corpus), so cutting
    wherever a number appears would split one item at the cross-reference.
    `_inline_ganada` follows the same rule.
    """
    marks = [(m.start(), int(m.group(1))) for m in _HOLD_INLINE_BRACKET.finditer(text)]
    if len(marks) < 2:
        return []
    if [num for _, num in marks] != list(range(1, len(marks) + 1)):
        return []
    return [pos for pos, _ in marks]


def _inline_ganada(text: str) -> list[str]:
    """Split '가. … 나. … 다. …' written on one line, the variant with no `<br/>`.

    Only in the expected order, or the '다.' ending a sentence ('…하였다.')
    reads as an item head. A head also has to follow a space, `)` or `]`:
    some cases run them together, as in '(소극)다. …'.
    """
    starts: list[int] = []
    at = 0
    for ch in _GANADA:
        pos = text.find(ch + ".", at)
        while pos > 0 and text[pos - 1] not in " \t)]":   # not mid-word
            pos = text.find(ch + ".", pos + 1)
        if pos < 0:
            break
        starts.append(pos)
        at = pos + 2
    if len(starts) < 2:
        return []
    return _cut_at(text, starts)


def _cut_at(text: str, starts: list[int]) -> list[str]:
    """Cut at the given item heads, keeping whatever precedes the first one.

    Constitutional Court holdings sometimes open with a preamble shared by
    every item ('1. … 가. … 나. …'). Dropping it leaves the items with nothing
    to say what they are about; an earlier cut that discarded it showed up as
    three cases losing text in the full-corpus check.
    """
    head = text[:starts[0]].strip()
    items = [text[a:b].strip() for a, b in zip(starts, starts[1:] + [len(text)])]
    if head:
        items[0] = head + " " + items[0]
    return items


def _holdings_items(text: str) -> list[str]:
    """Holding text -> its items, in order. One item if it will not divide.

    `<br/>` owns the boundary: 93% of the corpus has it, and it agrees with the
    `[N]` count 99.3% of the time. The labels are used to *check*, not to cut —
    a fragment that starts without one is a long item the source broke across
    lines, so it rejoins the previous item rather than halving a proposition.

    The labels cannot serve as keys either: there are three numbering families
    (brackets, 가나다, digit-dot), and only the order is needed here.

    Full-corpus check over 81,488 holdings: no text lost, no empty item. Of the
    2.44% where the label count disagreed with the item count, 98.9% were
    in-text self-references ('위 [1]항의 …') where the parser is right, leaving
    22 real failures (0.03%).
    """
    flat = _HOLD_BR.sub("\n", text or "").replace("［", "[").replace("］", "]")
    segs = [" ".join(seg.split()) for seg in flat.split("\n")]
    segs = [seg for seg in segs if seg]
    if not segs:
        return []
    if len(segs) == 1:
        # A single fragment may still carry its labels inline. This branch has
        # to come *before* the `_HOLD_HEAD` test: '가. … 나. …' starts on a
        # label, so testing first would file it as "a document with labels"
        # and fold the whole thing into one item.
        marks = _inline_bracket_marks(segs[0])
        if marks:
            return _cut_at(segs[0], marks)
        return _inline_ganada(segs[0]) or segs
    if any(_HOLD_HEAD.match(seg) for seg in segs):
        items: list[str] = []
        for seg in segs:
            if items and not _HOLD_HEAD.match(seg):
                items[-1] += " " + seg      # same item, broken across lines
            else:
                items.append(seg)
        return items
    return segs


def _pick_holding(text: str, terms: Sequence[str]) -> str:
    """The item the query hits most, capped. First item on a tie or no hit.

    Same rule the preview follows in `_excerpt_around`: this points at where
    the query matched rather than judging what the case says. A trailing `…`
    reports the cut.
    """
    items = _holdings_items(text)
    if not items:
        return ""
    best = items[0]
    if terms and len(items) > 1:
        best = max(items, key=lambda it: sum(1 for t in terms if t in it))
    if len(best) > HOLDING_MAX_CHARS:
        best = best[:HOLDING_MAX_CHARS].rstrip() + "…"
    return best


def _dense_rank(
    conn: sqlite3.Connection,
    embed_client: Any,
    query: str,
    limit: int,
) -> list[int]:
    """Embed the query through an OpenAI-compatible client, then sqlite-vec KNN.

    Only reached when USE_DENSE=1, which is also the only path that needs the
    sqlite-vec package — hence the local import.

    ``input_type=query`` marks the asymmetric-retrieval side (the index was
    built with "passage") and ``dimensions`` truncates a Matryoshka embedding
    to the stored width. Both are outside the OpenAI schema, so they travel in
    ``extra_body``.
    """
    import sqlite_vec

    resp = embed_client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
        dimensions=EMBED_DIM,
        extra_body={"input_type": "query"},
    )
    qv = sqlite_vec.serialize_float32(resp.data[0].embedding)
    rows = conn.execute(
        """
        SELECT rowid FROM prec_vec
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance
        """,
        (qv, limit),
    ).fetchall()
    return [r["rowid"] for r in rows]


# ---------- RRF fusion ----------

def _rrf_fuse(
    rankings: Sequence[Sequence[int]], *, k: int = 60, limit: int
) -> list[int]:
    """Reciprocal Rank Fusion. score(d) = Σ 1/(k + rank_r(d))."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ordered[:limit]]


# ---------- hydrate metadata, apply filters ----------

def _hydrate(
    conn: sqlite3.Connection,
    case_ids: Sequence[int],
    court_level: str | None,
    year_from: int | None,
    year_to: int | None,
    cap: int | None = LIMIT,
    court_name: str | None = None,
) -> list[dict[str, Any]]:
    """Load metadata for case ids, preserving their ranked order.

    ``cap=None`` returns everything, which the rerank and recency paths need.

    court_name: 법원 *이름/지역* 부분매칭(LIKE) — '부산'→부산지방법원·부산고등법원,
      '특허법원'·'서울행정법원' 등. court_level(심급)과 직교. 웹 search_cases 와 동일 방식.
    """
    if not case_ids:
        return []
    placeholders = ",".join("?" for _ in case_ids)
    where = [f"id IN ({placeholders})"]
    params: list[Any] = list(case_ids)
    if court_level:
        where.append("court_level = ?")
        params.append(court_level)
    if court_name:
        where.append("court_name LIKE ?")
        params.append(f"%{court_name}%")
    if year_from is not None:
        where.append("COALESCE(decision_year, case_year) >= ?")
        params.append(year_from)
    if year_to is not None:
        where.append("COALESCE(decision_year, case_year) <= ?")
        params.append(year_to)

    rows = conn.execute(
        f"""
        SELECT id, case_number, case_name, court_name, court_level,
               COALESCE(decision_year, case_year) AS year, reference_statute,
               COALESCE(holdings, '') AS holdings,
               COALESCE(
                   NULLIF(TRIM(generated_summary), ''),
                   NULLIF(TRIM(summary), ''),
                   NULLIF(TRIM(content_md), '')
               ) AS display_summary,
               CASE
                   WHEN NULLIF(TRIM(generated_summary), '') IS NOT NULL THEN 'generated_summary_excerpt'
                   WHEN NULLIF(TRIM(summary), '') IS NOT NULL THEN 'official_summary_excerpt'
                   ELSE 'original_text_excerpt'
               END AS display_summary_source
        FROM prec_cases
        WHERE {' AND '.join(where)}
        """,
        params,
    ).fetchall()

    by_id = {r["id"]: r for r in rows}
    out: list[dict[str, Any]] = []
    for cid in case_ids:
        if cid not in by_id:
            continue
        r = by_id[cid]
        full_summary = (r["display_summary"] or "").strip()
        out.append(
            {
                "id": r["id"],
                "case_number": r["case_number"],
                "case_name": r["case_name"],
                "court_name": r["court_name"],
                "court_level": r["court_level"],
                "year": r["year"],
                "reference_statute": r["reference_statute"],
                "_full_summary": full_summary,   # internal: rerank input + sentence source
                "_full_summary_source": r["display_summary_source"],
                "_holdings": r["holdings"],      # internal: `_pick_holding` selects an item
                "preview": "",                   # filled by sentence-level rerank
            }
        )
        if cap is not None and len(out) >= cap:
            break
    return out


def _rerank_scores(
    client: Any, query: str, texts: list[str]
) -> list[float]:
    """Call the reranker. Scores come back in input order, not ranked order.

    NIM schema: `{model, query:{text}, passages:[{text},...]}` → `{rankings:[{index, logit}]}`.
    logit 은 unbounded raw score (높을수록 관련). 정렬·비교 의미는 동일.
    """
    if not texts:
        return []
    path = f"/v1/retrieval/{RERANK_MODEL}/reranking"
    resp = client.post(
        path,
        json={
            "model": RERANK_MODEL,
            "query": {"text": query},
            "passages": [{"text": t} for t in texts],
        },
    )
    resp.raise_for_status()
    data = resp.json()
    # The service answers ranked by logit; put it back in input order.
    by_idx = {r["index"]: r["logit"] for r in data["rankings"]}
    # Missing entries sink to the bottom. Zero would not: these are logits,
    # so zero sits in the middle of the distribution, not below it.
    return [by_idx.get(i, float("-inf")) for i in range(len(texts))]


def _truncate_rerank(text: str) -> str:
    """Truncate to fit the reranker's window. p99 input is ~2.2K characters;
    the longest run to 17K."""
    if len(text) <= RERANK_INPUT_MAX_CHARS:
        return text
    return text[:RERANK_INPUT_MAX_CHARS]


# Sentence boundaries. Two constructs must not be treated as one, because
# both end in a period followed by a space:
#   - Numeric periods: dates ("2019. 5. 1.") and numbered items.
#   - Outline markers: Korean judgments enumerate with 가. 나. 다. the way
#     English uses a. b. c.
_SENT_BOUND = re.compile(
    r"(?<=[.!?。])"
    r"(?<!\d\.)"
    r"(?<![\s(\[][가나다라마바사아자차카타파하]\.)"
    r"\s+"
)


def _split_sentences(text: str) -> list[str]:
    """Split into sentences, absorbing fragments under eight characters.

    A fragment on its own is usually a piece of a word left by a split that
    should not have happened.

    양방향 흡수가 필요한 이유: 요약본이 "가. 채무자가..." 처럼 *목차 단편으로
    시작*할 때 lookbehind가 빈 앞부분을 보지 못해 단편이 첫 fragment로 떨어짐 →
    직후 정상 문장이 흡수해야 단독 노출 방지.
    """
    if not text:
        return []
    parts = _SENT_BOUND.split(text.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip().replace("\n", " ")
        if not p:
            continue
        # Merge into the previous sentence if either side is a fragment.
        if out and (len(p) < 8 or len(out[-1]) < 8):
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


def _sentence_previews_batch(
    client: Any, query: str, summaries: list[str], top_k: int
) -> list[str]:
    """Rerank sentences across all cases in one batch, keeping the best per case.

    - 같은 query에 대한 (query, sentence) 쌍이라 case 경계와 무관하게 한 호출에 처리 가능.
    - 선택된 top_k 문장은 *원문 등장 순서*로 join (문맥 흐름 보존).
    - 문장 수 ≤ top_k 인 case는 전체 문장 그대로 반환.
    - 요약본 빈 case는 ""; 분할 0개면 요약본 머리 200자 fallback.
    """
    case_sents: list[list[str]] = [_split_sentences(s) for s in summaries]
    all_sentences: list[str] = []
    boundaries: list[int] = [0]
    for sents in case_sents:
        all_sentences.extend(sents)
        boundaries.append(len(all_sentences))

    if not all_sentences:
        return [(s[:PREVIEW_FALLBACK_CHARS].replace("\n", " ") if s else "") for s in summaries]

    scores = _rerank_scores(client, query, all_sentences)

    previews: list[str] = []
    for ci, sents in enumerate(case_sents):
        if not sents:
            s = summaries[ci]
            previews.append(s[:PREVIEW_FALLBACK_CHARS].replace("\n", " ") if s else "")
            continue
        start, end = boundaries[ci], boundaries[ci + 1]
        case_scores = scores[start:end]
        # Best top_k sentences, or all of them if the case has fewer.
        n_pick = min(top_k, len(sents))
        ranked = sorted(range(len(sents)), key=lambda i: -case_scores[i])[:n_pick]
        # Cap the preview length, but never return an empty preview: one
        # sentence is kept even if it exceeds the cap on its own.
        accepted: list[int] = []
        total = 0
        for idx in ranked:
            seg_len = len(sents[idx]) + (1 if accepted else 0)  # joining space
            if accepted and total + seg_len > PREVIEW_MAX_CHARS:
                continue
            accepted.append(idx)
            total += seg_len
        accepted.sort()  # restore reading order
        previews.append(" ".join(sents[i] for i in accepted))
    return previews


# ---------- markdown serialisation ----------

# Provenance as one line the model reads, where the value itself says
# whether the text may be quoted. A separate boolean field alongside it
# carried the same one bit of information — the two agreed in all 120 cases
# checked — and cost about 90 characters per result. Worse, callers came to
# depend on the field name, so renaming it broke them. A marker inside the
# value has no name to depend on.
#
# Only a verbatim excerpt is quotable. A displayed body string is a head
# extract and does not match the source exactly, so it is not.
_MARKUP_RE = re.compile(r"</?\s*br\s*/?\s*>", re.I)


def _strip_markup(text: str) -> str:
    """Strip HTML fragments that come through in corpus text."""
    return _MARKUP_RE.sub(" ", str(text or "")).strip()


_NO_QUOTE = "(직접인용 불가)"
_PREVIEW_KINDS = {
    "original_text_excerpt": "원문 발췌",
    "original_text": f"원문 머리 발췌{_NO_QUOTE}",
    "official_summary": f"공식 요약{_NO_QUOTE}",
    "official_summary_excerpt": f"공식 요약 발췌{_NO_QUOTE}",
    "generated_summary": f"AI 요약{_NO_QUOTE}",
    "generated_summary_excerpt": f"AI 요약 발췌{_NO_QUOTE}",
    "reference_statute_metadata_excerpt": f"참조조문 메타·판시 아님{_NO_QUOTE}",
}


def _set_preview(match: dict[str, Any], text: str, source: str) -> None:
    """Set the preview and its provenance together, so neither can be lost."""
    match["preview"] = text
    match["preview_provenance"] = source


def _set_holding(match: dict[str, Any], terms: Sequence[str]) -> None:
    """Attach the holding item the query points at. A case without one — most
    trial-court judgments — gets no field at all."""
    holding = _pick_holding(match.get("_holdings") or "", terms)
    if holding:
        match["holding"] = holding


def _drop_preview_internals(match: dict[str, Any]) -> None:
    match.pop("_full_summary", None)
    match.pop("_full_summary_source", None)
    match.pop("_holdings", None)

def _format_response_md(resp: dict[str, Any]) -> str:
    """Response dict -> markdown-KV string."""
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
                f"  url: {case_url_base()}/cases/{m['id']}",
                f"  case_no: {m.get('case_number','')}",
                f"  case_name: {m.get('case_name','')}",
                f"  court: {m.get('court_name','')} {m.get('court_level','')} {m.get('year','')}".rstrip(),
            ]
            if m.get("reference_statute"):
                # Corpus text carries stray `<br/>`; strip it rather than
                # pass HTML fragments to the caller.
                block.append(f"  statute: {_strip_markup(m['reference_statute'])}")
            if m.get("holding"):
                # The court wrote this, so it carries no label of its own:
                # whether it can be quoted is the tool docstring's business,
                # and a trailing `…` reports a cut. A label line would cost
                # its length once per result.
                block.append(f"  holding: {m['holding']}")
            if m.get("preview"):
                block.append(f"  preview: {m['preview']}")
                source = m.get("preview_provenance") or "unknown"
                block.append(f"  preview_kind: "
                             f"{_PREVIEW_KINDS.get(source, f'{source}{_NO_QUOTE}')}")
            lines.append("\n".join(block))
    elif status == "ok":
        lines.append("## matches: (없음)")
    return "\n".join(lines)


# ---------- public tool ----------

@dedup_guard("precedent_search")
def precedent_search(
    ctx: RunContext[HarnessDeps],
    query: str | None = None,
    case_number: str | None = None,
    court_level: str | None = None,
    court_name: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> str:
    """판례 검색 — query 또는 case_number 중 하나는 필수이며, 사건번호 직접 조회 또는
    사실관계 키워드 검색을 수행합니다.
    법원이 쓴 판시사항(holding)과 짧은 preview를 반환합니다. 판결문 본문의 판단·법리가
    필요하면 가장 관련된 결과의 id로 precedent_dive를 이어 호출해 확인합니다.

    언제:
    - 유사 사실관계 판례(민사·형사·행정·가사)를 찾거나 주장·전망의 근거가 필요할 때 → query.
      "보통 이렇게 됩니다" 류의 전망을 기억으로 말하기 전에 먼저 부르세요.
    - 특정 사건번호가 주어졌을 때 → case_number.
    - 판례가 적용한 조문의 본문·현행 여부는 statute_lookup 으로 이어 확인하세요 — 판례는
      선고 당시 조문을 적용하므로 지금 본문과 다를 수 있습니다.

    응답: markdown-KV.
    - `holding`은 법원이 쓴 판시사항(그 사건이 무엇을 정했는지)이라 직접인용할 수 있습니다.
      쟁점이 여럿인 판례는 질의에 가장 가까운 항 하나만 나가며, 꼬리 `…`는 그 항이 길어
      잘렸다는 뜻입니다(잘린 문장은 인용하지 말고 필요하면 precedent_dive로 확인하세요).
      하급심 등 판시사항이 없는 판례는 이 칸이 아예 나오지 않습니다.
    - `preview_kind: 원문 발췌`만 따옴표로 직접인용할 수 있고, `(직접인용 불가)`가 붙은
      요약은 바꿔 쓰세요.
    답에 쓴 판례는 직접 인용이든 요약이든 반환 url을 링크로 함께 제시하세요.

    Args:
      query: 사건번호를 뺀 사실관계·죄명·법조의 변별력 있는 명사 어간 여러 개. 2자 죄명도 지원.
      case_number: 특정 사건번호(예 "2010다89012"). 긴 인용문이면 사건번호 부분만.
      court_level: 심급 '1심'|'2심'|'대법원'|'헌재'.
      court_name: 법원 이름·지역 부분매칭(예 "부산", "특허법원").
      year_from: 사건년도 범위 시작(선고년도 기준, 없으면 접수년도).
      year_to: 사건년도 범위 끝. year_from 과 함께 또는 단독 사용.
    """
    query = coerce_str(query)
    case_number = coerce_str(case_number)
    court_level = coerce_str(court_level)
    court_level, court_level_note = _normalize_court_level(court_level)
    court_name = coerce_str(court_name)
    year_from = coerce_int(year_from)
    year_to = coerce_int(year_to)

    # Validate before opening the database: argument errors should not cost
    # a connection to a large file. The case number is normalised here too,
    # so a valid one wins even when a short query accompanies it.
    cno = _extract_case_number(case_number) if case_number else None
    if not cno and not query:
        return _format_response_md({
            "status": "error",
            "message": (
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
        # Direct lookup by case number. The number is taken from its own
        # argument only, never mined out of the free-text query — keeping the
        # two arguments' jobs distinct is what makes each of them
        # predictable. A citation passed in that argument is normalised to
        # the number it contains.
        if cno:
            id_rows = conn.execute(
                "SELECT id FROM prec_cases WHERE REPLACE(case_number,' ','')=? ORDER BY id LIMIT ?",
                (cno, LIMIT),
            ).fetchall()
            if not id_rows:  # 병합사건('2015두38917·38924') 등은 부분일치로
                id_rows = conn.execute(
                    "SELECT id FROM prec_cases WHERE REPLACE(case_number,' ','') LIKE ? "
                    "ORDER BY length(case_number), id LIMIT ?",
                    (f"%{cno}%", LIMIT),
                ).fetchall()
            if id_rows:
                matches = _hydrate(
                    conn, [r["id"] for r in id_rows], court_level, year_from, year_to,
                    cap=LIMIT, court_name=court_name,
                )
                # A case-number lookup has no query terms to choose by, so
                # the first item stands as the representative issue.
                cno_terms = _preview_terms(query) if query else []
                for m in matches:
                    _set_preview(
                        m,
                        m["_full_summary"][:PREVIEW_FALLBACK_CHARS].replace("\n", " "),
                        m["_full_summary_source"],
                    )
                    _set_holding(m, cno_terms)
                    _drop_preview_internals(m)
                if matches:
                    return _format_response_md({"status": "ok", "matches": matches})
            if case_number and not query:
                # Case number given, nothing found, and no query to fall
                # back on: say so instead of returning empty.
                return _format_response_md({
                    "status": "ok",
                    "message": (
                        f"사건번호 '{case_number}' 에 해당하는 판례를 찾지 못했습니다. "
                        "사건번호를 확인하거나 사실관계 키워드(query)로 재검색하세요."
                    ),
                    "matches": [],
                })
            # Otherwise fall through to keyword search — the number may be
            # mistyped or formatted differently.

        # ----- keyword search -----

        pool_size = max(LIMIT * OVERSAMPLE, 30)

        # ===== lexical path (the default): trigram + morpheme, fused =====
        if not USE_DENSE:
            tri_match = _or_match(query)       # trigram: 3자+ 부분일치·오타 내성
            morph_match = _morph_match(query)  # 형태소: 단어 단위(2자 '살인'·'사기' 포함)
            # Ranked by relevance only. A recency sort was removed: over an
            # OR-token union, sorting by date alone promotes recent judgments
            # that matched one common token and nothing else, so the top of
            # the list stops being about the query.
            # Filters go into both queries, not over their results: a
            # selective filter applied afterwards would leave almost nothing.
            tri_ids = _fts_or_rank(
                conn, tri_match, pool_size,
                court_level=court_level, court_name=court_name,
                year_from=year_from, year_to=year_to,
            )
            morph_ids = _morph_rank(
                conn, morph_match, pool_size,
                court_level=court_level, court_name=court_name,
                year_from=year_from, year_to=year_to,
            )
            # Fuse the two rankings.
            rankings = [r for r in (tri_ids, morph_ids) if r]
            ids = _rrf_fuse(rankings, k=RRF_K, limit=pool_size) if rankings else []
            matches = _hydrate(
                conn, ids, court_level, year_from, year_to,
                cap=LIMIT, court_name=court_name,
            ) if ids else []
            # Prefer the FTS snippet, which sits on the matched phrase.
            snips = _fts_snippets(conn, [m["id"] for m in matches], tri_match) if (matches and tri_match) else {}
            # Judgments matched only on a short morpheme get no snippet, so
            # the window is excerpted from the body instead — otherwise the
            # preview shows the head of a summary and never the match.
            missed = [m["id"] for m in matches if m["id"] not in snips]
            terms = _preview_terms(query)
            excerpts = (
                _body_excerpts(conn, missed, terms, with_provenance=True)
                if missed else {}
            )
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
            resp = {
                "status": "ok",
                "matches": matches,
                "_debug": {"n_tri": len(tri_ids), "n_morph": len(morph_ids), "mode": "tri+morph_rrf"},
            }
            # Say what was ignored and why, rather than returning a bare
            # empty result the caller cannot diagnose.
            notes = [court_level_note]
            if not matches:
                notes.append(_case_no_in_query_hint(query))
            notes = [n for n in notes if n]
            if notes:
                resp["note"] = " / ".join(notes)
            return _format_response_md(resp)

        # ===== dense path (USE_DENSE=1): embeddings, fusion, rerank =====
        embed = ctx.deps.embed
        fts_ids = _fts_rank(conn, query, pool_size)
        dense_ids = _dense_rank(conn, embed, query, pool_size)

        rankings = [r for r in (fts_ids, dense_ids) if r]
        if not rankings:
            return _format_response_md({"status": "ok", "matches": []})

        fused = _rrf_fuse(rankings, k=RRF_K, limit=pool_size)

        if USE_RERANK:
            # Stage 1: rerank the whole candidate pool on full summaries.
            candidates = _hydrate(
                conn, fused, court_level, year_from, year_to,
                cap=None, court_name=court_name,
            )
            if candidates:
                texts = [_truncate_rerank(c["_full_summary"]) for c in candidates]
                scores = _rerank_scores(ctx.deps.rerank, query, texts)
                ordered = sorted(zip(candidates, scores), key=lambda p: -p[1])
                matches = [c for c, _ in ordered[:LIMIT]]
                # Stage 2: rerank sentences within the surviving cases to
                # pick what the preview shows.
                summaries = [m["_full_summary"] for m in matches]
                previews = _sentence_previews_batch(
                    ctx.deps.rerank, query, summaries, PREVIEW_TOP_K
                )
                for m, p in zip(matches, previews):
                    _set_preview(m, p, m["_full_summary_source"])
            else:
                matches = []
        else:
            # Reranking off: fall back to the head of the summary.
            matches = _hydrate(
                conn, fused, court_level, year_from, year_to,
                cap=LIMIT, court_name=court_name,
            )
            for m in matches:
                _set_preview(
                    m,
                    m["_full_summary"][:PREVIEW_FALLBACK_CHARS].replace("\n", " "),
                    m["_full_summary_source"],
                )

        dense_terms = _preview_terms(query)
        for m in matches:
            _set_holding(m, dense_terms)
            _drop_preview_internals(m)

        return _format_response_md({
            "status": "ok",
            "matches": matches,
            "_debug": {
                "n_fts": len(fts_ids),
                "n_dense": len(dense_ids),
                "n_fused": len(fused),
                "rerank": USE_RERANK,
            },
        })
    finally:
        conn.close()
