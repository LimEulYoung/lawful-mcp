"""statute_search and statute_read — statutes (`st_statutes`), administrative rules (`st_notices`) and tax treaties
(`treaties`). Search finds, read reads: the two take no common argument but offense_date, so a call cannot mix them.

statute_search(query):
  A) a query                     → list (one line per statute + the article the query's words matched)
  D) a query naming one statute and article numbers ('민법 제750조 제751조') → that statute's article text
statute_read(statute_id):
  B) `statute_id` only           → article outline (article_no + title; over OUTLINE_TOC_CHARS, the 편·장·절 contents)
  C) `statute_id` + `articles`   → article text (also `articles=['부칙 N']` → addenda articles, `st_addenda`)

Search matches statute names (morpheme tokens, and names the query writes whole) and article text (FTS5 trigram).
"""
from __future__ import annotations

import bisect
import datetime
import functools
import itertools
import re
import sqlite3
from typing import Any

from pydantic_ai import RunContext

from ..deps import HarnessDeps, open_db
from ._coerce import BAD_DATE_MESSAGE, coerce_list, coerce_str, to_iso_date, unreadable_date
from ._morph import kiwi as _kiwi
from . import treaties
from ._fts import query_tokens as _query_tokens, quoted_and as _quoted_and
from .precedent_search import bm25_expr  # one source of BM25 column weighting (shared by judgments and statutes)

# BM25 column weights for the article FTS — in column order (title, article_text).
# An article whose title ('제26조(해고의 예고)' — Art. 26 (Notice of dismissal)) matches the query ranks above one the
# query only brushes in the body.
_ART_BM25_WEIGHTS = (3.0, 1.0)

# Articles one call returns at most; the rest are named in the response.
ARTICLES_MAX = 8

# Character cap for one response. Article text is the ground of the answer, so the cap is not a budget but the
# largest response seen (about 40,000 characters, eight long articles) plus a third: only an outlier is cut, and
# then the response says what was left out (`_bounded_response_md`).
RESPONSE_MAX_CHARS = 53_000

# Statutes one search lists. Fixed, not the model's choice: in the recorded runs, the id a model went on to read was
# in the first 10 of its list in 99.7% of reads (6th to 10th: 5%), and a longer list only lengthened the answer.
SEARCH_LIMIT = 10

# Lawful statute page urls, so the model can link the original: statute /statutes/{law_id}, article
# /statutes/{law_id}/{jo}, administrative rule /statutes/admrul-{st_notices.id}.
LAWFUL_WEB_BASE = "https://lawful.crow-tit.com"
NOTICE_URL_PREFIX = "admrul-"


def _parse_statute_ref(raw: Any) -> tuple[bool, int] | tuple[str, str] | tuple[str, int] | None:
    """Model-supplied `statute_id` → `(False, id)` statute (`st_statutes`), `(True, id)` administrative rule
    (`st_notices`, written 'admrul-18060'), `("trty", id)` tax treaty ('trty-1307'), or `("law", law_id)` for a web
    path (`/statutes/{law_id}`) or a digit string starting with 0: the statute slot of a web url is a law_id
    ('004704'), not the id, and `_statute_lookup_impl` resolves it to the current version. None if unreadable.

    The identifier itself names the corpus: statute and administrative-rule ids overlap (almost every
    administrative-rule id is also a statute id), so a number read in the wrong table returns an unrelated document.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return (False, raw)
    if isinstance(raw, float):
        return (False, int(raw))
    if isinstance(raw, (list, tuple)):
        return _parse_statute_ref(raw[0]) if raw else None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        # The prefix is accepted regardless of case/whitespace (the model also writes 'ADMRUL-18060').
        low = s.lower().replace(" ", "")
        if low.startswith(NOTICE_URL_PREFIX):
            rest = low[len(NOTICE_URL_PREFIX):]
            return (True, int(rest)) if rest.isdigit() else None
        if low.startswith(treaties.TREATY_URL_PREFIX):
            tid = treaties.parse_ref(low)
            return ("trty", tid) if tid is not None else None
        # Also accept the model pasting a url or web path as is.
        if "/statutes/" in low:
            tail = (low.rsplit("/statutes/", 1)[1]
                    .split("/")[0].split("?")[0].split("#")[0])
            if tail.startswith(NOTICE_URL_PREFIX):
                rest = tail[len(NOTICE_URL_PREFIX):]
                return (True, int(rest)) if rest.isdigit() else None
            if tail.startswith(treaties.TREATY_URL_PREFIX):
                tid = treaties.parse_ref(tail)
                return ("trty", tid) if tid is not None else None
            # A web path holds a law_id. It is not zero-padded: a made-up /statutes/584 fails to resolve rather
            # than open law_id 000584.
            return ("law", tail) if tail.isdigit() else None
        if s.isdigit() and s.startswith("0") and len(s) >= 2:
            return ("law", s)             # 0-padding appears only in law_id notation
        return (False, int(s)) if s.isdigit() else None
    return None


def _statute_ref(d: dict[str, Any] | None) -> str:
    """Statute/administrative-rule/treaty dict → **the identifier the model sees**: an integer string, `'admrul-{id}'`
    or `'trty-{id}'`. The corpus test is the one `_statute_web_url` uses (presence of law_id), so the id and the
    url the model receives point at the same document."""
    if not d:
        return ""
    if d.get("treaty_id"):
        return f"{treaties.TREATY_URL_PREFIX}{d['treaty_id']}"
    nid = d.get("id")
    if not isinstance(nid, int):
        nid = d.get("statute_id")
    if d.get("law_id"):
        return str(nid) if isinstance(nid, int) else ""
    if d.get("category") or d.get("notice_id"):
        return f"{NOTICE_URL_PREFIX}{nid}" if isinstance(nid, int) else ""
    return str(nid) if isinstance(nid, int) else ""

# Search lists only the current edition of an administrative rule: an amendment adds a row under a new id, and
# `history_status` ('현행' current / '시행예정' not yet in force / '구판' old edition) marks the current one. A lookup
# by id still opens an old edition.
_NOTICE_CURRENT = "n.history_status='현행'"

# Search takes every administrative rule with text to read (`has_text_content=1`, which includes text recovered from
# attachments), whatever its layout: short designation notices can set rights and duties too (the simple-labour
# occupations designated under Art. 5 of the Minimum Wage Act).


# ---------- statute-name normalization (abbreviations + morpheme tokens) ----------
# Korean statute names are often compounds without spaces ("광업제조업자"), which trigram FTS / substring
# matching cannot catch. Morphological analysis splits them into noun tokens used for partial statute-name
# matching (LIKE), and common names are mapped to the official name via the alias map.

# Common name / abbreviation → official statute name: the `statute_aliases` table (common and practice
# abbreviations outside `st_statutes.short_name` such as 산재·특가법·아청법, old names, hanja spellings; data, not code).
# Maps common names that morpheme tokenization cannot resolve ('산재' has no characters in common with the name
# '산업재해보상보험법'). Rows marked '분리 승계' name where a split statute's penalties went — for charge names only
# (`_charge_index.LawNames`): a statute search keeps a name's own statute.
def statute_aliases(conn: sqlite3.Connection) -> dict[str, str]:
    """Alias (space- and interpunct-stripped) → statute name as `st_statutes.name` writes it."""
    rows = conn.execute("SELECT alias, statute_name, how FROM statute_aliases").fetchall()
    return {_strip_dots(a.replace(" ", "")): v for a, v, how in rows if how != "분리 승계"}


# Interpunct variants — stored names (statutes, notices) hold U+318D 'ㆍ' (arae-a) ('초ㆍ중등교육법') or U+00B7 '·',
# while the model types the name without the dot ('초중등교육법') or with any visually identical code point.
# Matching *removes* them on both sides (query and stored name) so they are treated alike.
_DOT_CHARS = ("ㆍ", "·", "‧", "∙", "・", "․")  # U+318D U+00B7 U+2027 U+2219 U+30FB U+2024


def _strip_dots(s: str) -> str:
    for ch in _DOT_CHARS:
        s = s.replace(ch, "")
    return s


def _name_norm_sql(col: str) -> str:
    """SQL expression of a stored name for matching — without the spaces and the two interpuncts stored names hold
    (the same result as `_strip_dots` and removing spaces)."""
    return f"REPLACE(REPLACE(REPLACE({col},' ',''),'ㆍ',''),'·','')"


_SPAN_MAX_TOKENS = 24     # cap on the number of morphemes used to build candidate spans (prevents O(n²) span blowup)
_SPAN_MAX_CHARS = 64      # even the longest statute name fits within this


def _query_name_spans(conn: sqlite3.Connection, query: str) -> list[str]:
    """Candidates from the query for **strings that could be a statute name** — contiguous spans that
    start and end on morpheme boundaries.

    They must not be character-level substrings: '손해배상 법률 상담' ('damages legal consultation'), read character
    by character, contains '상법' (Commercial Act) across '손해배상' + '법률', which would put the Commercial Act first.
    Spans aligned with morpheme boundaries keep a name that overlaps the start of another token, like '형법'
    (Criminal Act) in '군형법 제92조' (Military Criminal Act Art. 92), and the length comparison lifts '군형법'.

    Returned in space/interpunct-stripped form (same shape as `_name_norm_sql`) so they match the
    SQL-normalized names character for character. If Kiwi is missing or fails, only the normalized query
    remains (exact match)."""
    raw = _strip_dots((query or "").strip())
    # Common-name normalization (alias map) result = the exact-match slot. ⚠ The map's *values* still contain
    # interpuncts ('총포ㆍ도검ㆍ…'), so they must be stripped once more here to match the SQL-side normalized name.
    spans = {_strip_dots(statute_aliases(conn).get(raw.replace(" ", ""), raw).replace(" ", ""))}
    toks = _kiwi().tokenize(raw)[:_SPAN_MAX_TOKENS]
    # raw is already the original with interpuncts removed, so cutting spans and removing spaces gives the same
    # form as the SQL side.
    for i, ti in enumerate(toks):
        start = ti.start
        for tj in toks[i:]:
            end = tj.start + tj.len
            seg = raw[start:end].replace(" ", "")
            if 2 <= len(seg) <= _SPAN_MAX_CHARS:
                spans.add(seg)
            elif len(seg) > _SPAN_MAX_CHARS:
                break
    return [s for s in spans if s]


def _name_in_query_sql(col: str, n_spans: int) -> str:
    """**Length of the name if the query contains the whole name, else 0** (parameters = the span list).

    Exact match (`name = query`) alone misses most real use. Queries have the form 「statute name + topic word」,
    so a single word after the name turns exact match off, and token coverage alone lets long administrative rules
    whose names also hold the topic word push out the statute named ("개인정보 보호법 … 주민등록번호" would put a
    resident-registration-number committee guideline first).

    Why score by length: short names sit inside long ones. '군형법 제92조' contains both '형법' and
    '군형법', but the answer is the longer one, and the length comparison makes that call.
    Exact match is this expression's maximum (= the full query length), so it needs no separate case.

    Candidates are the **set of morpheme-boundary spans** built by `_query_name_spans`, and here only set
    membership is checked — opening it up to character substrings (`instr`) lets in accidents across words."""
    n = _name_norm_sql(col)
    if n_spans <= 0:
        return "0"
    holes = ",".join("?" for _ in range(n_spans))
    return f"(CASE WHEN {n} IN ({holes}) THEN length({n}) ELSE 0 END)"


def _statute_name_tokens(conn: sqlite3.Connection, query: str) -> tuple[str, list[str]]:
    """Returns (normalized query, noun tokens) for statute-name matching. Alias normalization, then morpheme split.

    e.g. '광업제조업자' → ('광업제조업자', ['광업','제조','업자'])
         '헌법'         → ('대한민국헌법', ['대한민국','헌법'])   # alias normalization
    Tokens split compounds and are used only for statute-name LIKE coverage (not for body FTS — common fragments
    cause noise). If kiwipiepy is missing or fails, tokens are empty → the caller falls back to the normalized query.
    """
    raw = (query or "").strip()
    # Strip interpunct variants before alias lookup/tokenization — unify '초·중등교육법'/'초중등교육법' into the
    # dotless form to match the DB ('초ㆍ중등교육법' → the SQL side strips the dots too, via _name_norm_sql).
    raw = _strip_dots(raw)
    normalized = statute_aliases(conn).get(raw.replace(" ", ""), raw)
    toks = [_strip_dots(t.form) for t in _kiwi().tokenize(normalized)
            if t.tag in ("NNG", "NNP") and len(_strip_dots(t.form)) >= 2]
    return _strip_dots(normalized.replace(" ", "")), list(dict.fromkeys(toks))


def _short_words_message(words: list[str], matches: list[dict[str, Any]]) -> str | None:
    """st_articles_fts is a trigram index: a word under 3 characters matches no article body, and SQLite leaves it
    out of the expression. When every topic word of the query (`_topic_words`) is that short, the list comes from
    statute names alone — say so, or a 0 or a list of name look-alikes reads as "no statute says this" ('절도' finds
    nothing; '형법 간통' lists the Criminal Act with no word of where). Plain statute names ('민법') are found by name
    and get no message."""
    if not words or any(len(w) >= 3 for w in words):
        return None
    names = "".join(re.sub(r"\s+", "", m.get("name", "")) for m in matches)
    unmatched = [w for w in words if w not in names]
    if not unmatched:
        return None
    found = "아래는 법령 이름에 낱말이 든 것뿐입니다" if matches else "이름에 이 낱말이 든 법령도 없습니다"
    return (
        f"{'·'.join(repr(w) for w in unmatched)}: 두 글자 이하 낱말은 조문 본문에서 찾지 못합니다"
        f"(본문 검색은 세 글자부터) — {found}. 본문은 조문에 나올 세 글자 이상 문구로 찾으세요."
    )


# ---------- query words: statute names, topic words, article numbers ----------

_BODY_WORDS_MAX = 6           # topic words searched in article text; combinations are tried when no article has all
_NO_BODY_MATCH = "x" * 1000   # an expression no article matches (an empty one makes FTS5 raise an error)
_QUERY_ARTICLE_RE = re.compile(r"(부\s*칙\s*)?제?\s*(\d+)\s*조(?![원억])(?:\s*의\s*(\d+))?")


def _named_spans(conn: sqlite3.Connection, query: str | None) -> set[str]:
    """Statute names or official abbreviations the query writes in full, space- and interpunct-stripped."""
    spans = _query_name_spans(conn, query) if query else []
    if not spans:
        return set()
    holes = ",".join("?" * len(spans))
    name, short = _name_norm_sql("name"), _name_norm_sql("short_name")
    rows = conn.execute(
        f"SELECT {name} FROM (SELECT DISTINCT name FROM st_statutes) WHERE {name} IN ({holes}) "
        f"UNION SELECT {short} FROM (SELECT DISTINCT short_name FROM st_statutes WHERE short_name IS NOT NULL) "
        f"WHERE {short} IN ({holes})", [*spans, *spans]).fetchall()
    return {r[0] for r in rows}


def _topic_words(query: str | None, named: set[str]) -> list[str]:
    """Query words other than the statute names it writes (found by name) and article numbers."""
    out = []
    for w in _query_tokens(query or ""):
        bare = _strip_dots(w)
        if _QUERY_ARTICLE_RE.fullmatch(bare) or bare == "부칙" or any(bare in s or s in bare for s in named):
            continue
        out.append(w)
    return out


def _articles_in_query(query: str | None) -> list[str]:
    """Article numbers written in the query ('제48조', '160조의5', '부칙 제2조') as `articles` tokens, all of them: the
    search reads the first ARTICLES_MAX and names the rest."""
    return [("부칙 " if m[1] else "") + m[2] + (f"의{m[3]}" if m[3] else "")
            for m in _QUERY_ARTICLE_RE.finditer(query or "")]


def _over_cap_note(source: str, dropped: list, statute_id: int | str | None = None) -> str:
    """The note for articles over ARTICLES_MAX, naming the ones left out so the model can ask for them."""
    preview = ", ".join(str(a) for a in dropped[:12])
    if len(dropped) > 12:
        preview += f" 외 {len(dropped) - 12}개"
    then = (f"statute_read(statute_id={statute_id}, articles=[…]) 로 부르세요" if statute_id is not None
            else "같은 statute_id 로 나눠 재호출하세요")
    return (f"{source} {ARTICLES_MAX + len(dropped)}개 중 앞 {ARTICLES_MAX}개만 조회했습니다(호출당 최대 {ARTICLES_MAX}개). "
            f"나머지({preview})는 {then}.")


# Articles joined in one string that `coerce_list` keeps as one token: without brackets ('185,188' — 38 of 5,418
# `articles` in 7 days of production chats to 11/9), with quotes ("'185','188'"), or a JSON list behind a stray '='
# ('=["462", "463", …]' — 3 of 20,533 in 30 days). `_parse_articles` reads the first number only, so the rest went
# missing unsaid. Split when every part, without the quotes, brackets and '=' around it, is an article, addendum or
# annexed-table token; anything else ('750조 제1항, 제2항') stays as it was.
_ARTICLE_TOKEN_RE = re.compile(
    r"(제\s*)?\d+\s*(조)?(\s*(의|-)\s*\d+)?|부\s*칙(\s*제?\s*\d+\s*조?(\s*의\s*\d+)?)?|별\s*표\s*\d+(\s*의\s*\d+)?")
_JOINED_WRAPPING = " \t'\"‘’“”[]="


def _split_joined_articles(articles: list | None) -> list | None:
    if not articles:
        return articles
    out: list = []
    for a in articles:
        parts = [p.strip(_JOINED_WRAPPING) for p in re.split(r"[,，、]", a)] if isinstance(a, str) else []
        kept = [p for p in parts if p]
        if len(parts) > 1 and kept and all(_ARTICLE_TOKEN_RE.fullmatch(p) for p in kept):
            out.extend(kept)
        else:
            out.append(a)
    return out


def _body_match(conn: sqlite3.Connection, fts_table: str, words: list[str]) -> tuple[str, int, list[str]]:
    """FTS expression for the articles holding the most of `words` together → (expression, words held, words
    searched). Only words of 3+ characters can match the trigram index. All of them if some article holds them all;
    otherwise every combination of the largest number (at least two) that some article holds."""
    words = [w for w in words if len(w) >= 3][:_BODY_WORDS_MAX]
    for k in range(len(words), 1 if len(words) > 1 else 0, -1):
        expr = " OR ".join(f"({_quoted_and(c)})" for c in itertools.combinations(words, k))
        try:
            if conn.execute(f"SELECT 1 FROM {fts_table} WHERE {fts_table} MATCH ? LIMIT 1", (expr,)).fetchone():
                return expr, k, words
        except sqlite3.OperationalError:
            break
    return _NO_BODY_MATCH, 0, words


def _body_message(held: int, words: list[str]) -> str | None:
    """Say when the article text matched only some of the topic words, or none (the list is then by name only)."""
    if held == len(words):
        return None
    if not held:
        return "조문 본문에서 맞는 조문을 찾지 못해, 법령 이름으로만 찾았습니다."
    return (f"조문 본문에 {'·'.join(repr(w) for w in words)}가 모두 든 조문이 없어, 그중 {held}개가 함께 든 "
            "조문으로 찾았습니다.")


def _named_law(matches: list[dict[str, Any]], named: set[str]) -> dict[str, Any] | None:
    """The one statute the query names: the longest name it writes, when every other name it writes is part of
    that one (a statute and its decree) and a single law in the list holds it."""
    if not named:
        return None
    longest = max(named, key=len)
    if any(s not in longest for s in named):
        return None
    laws = {m["law_id"]: m for m in matches if m["name_in_query"] == len(longest)}
    return next(iter(laws.values())) if len(laws) == 1 else None


# ---------- addenda (부칙, `st_addenda`) ----------

_ADDENDA_RE = re.compile(r"^\s*부\s*칙\s*(?:제?\s*(\d+)\s*조?(?:\s*의\s*(\d+))?)?")
ADDENDA_MAX = 8           # addenda articles per call, newest promulgation first
ADDENDA_LIST_MAX = 10     # promulgations listed for '부칙' alone


def _addenda_specs(articles: list[str | int] | None) -> list[tuple[int | None, int | None]]:
    """'부칙' → (None, None) (the list); '부칙 3' · '부칙 제3조의2' → (3, None) · (3, 2)."""
    out = []
    for token in articles or []:
        m = _ADDENDA_RE.match(str(token).replace("-", "의"))
        if m:
            out.append((int(m[1]) if m[1] else None, int(m[2]) if m[2] else None))
    return out


def _addendum_label(no: Any, branch: Any, title: Any) -> str:
    return f"제{no}조" + (f"의{branch}" if branch else "") + (f"({title})" if title else "")


def _addenda(
    conn: sqlite3.Connection, law_id: str, specs: list[tuple[int | None, int | None]], date_iso: str,
) -> dict[str, Any]:
    """Addenda of the law's amendments promulgated by `date_iso`, newest first: the requested article of each
    amendment, or with '부칙' alone the list of amendments and their article titles."""
    rows = conn.execute(
        "SELECT promul_date, promul_no, article_no, article_branch, title, article_text FROM st_addenda "
        "WHERE law_id=? AND promul_date<=? ORDER BY promul_date DESC, promul_no DESC, article_no, article_branch",
        (law_id, date_iso)).fetchall()
    if not rows:
        return {"addenda_note": "이 법령의 부칙은 코퍼스에 없습니다(부칙은 일부 법령만 수록)."}
    out: dict[str, Any] = {}
    if any(no is None for no, _ in specs):
        groups: dict[tuple, list] = {}
        for r in rows:
            groups.setdefault((r["promul_date"], r["promul_no"]), []).append(
                _addendum_label(r["article_no"], r["article_branch"], r["title"]))
        out["addenda_list"] = [{"date": d, "no": n, "articles": a}
                               for (d, n), a in list(groups.items())[:ADDENDA_LIST_MAX]]
        out["addenda_total"] = len(groups)
    wanted = [(no, br) for no, br in specs if no is not None]
    if wanted:
        hits = [r for r in rows if any(r["article_no"] == no and (br is None or r["article_branch"] == br)
                                       for no, br in wanted)]
        out["addenda"] = [{"date": r["promul_date"], "no": r["promul_no"],
                           "label": _addendum_label(r["article_no"], r["article_branch"], r["title"]),
                           "text": r["article_text"]} for r in hits[:ADDENDA_MAX]]
        out["addenda_more"] = max(0, len(hits) - ADDENDA_MAX)
        if not hits:
            out["addenda_note"] = "요청한 번호의 부칙 조문이 없습니다 — '부칙' 만 주면 부칙 목록이 나옵니다."
    return out


# Non-article units (annexed tables, forms, addenda …). A token starting with one of these words is not scanned for
# digits: '별표5' (annexed table 5) would otherwise read as Article 5 and return an unrelated article as official
# text. Annexed tables and forms are not in the corpus (`_bad_articles_response` says so); addenda are in
# `st_addenda` (`_addenda`).
_NON_ARTICLE_UNIT_RE = re.compile(r"별\s*표|별\s*지|별\s*첨|별\s*도|서\s*식|양\s*식|붙\s*임|부\s*칙|부\s*록")


def _parse_articles(
    spec: list[str | int] | None,
) -> list[tuple[int, int | None]] | None:
    """`articles` parameter → list of `(article_no_num, branch | None)` tuples.

    Broadly absorbs the Korean article notations an LLM naturally uses — every notation of the same
    article normalizes to the same spec:
      - `"347"` / `"제347조"` / `"347조"`              → `(347, None)` — main article + all branches (`의2` ...)
      - `"347-2"` / `"347의2"` / `"제347조의2"`         → `(347, 2)`    — pinpoints Article 347-2 only
      - expand contiguous ranges into a list: `["3","4","5","6","7"]`

    Tokens with no digits and non-article structural units (`별표5`·`부칙` — `_NON_ARTICLE_UNIT_RE`) are
    ignored; if every token is, the caller (`_statute_lookup_impl`) answers with format guidance.
    """
    if spec is None:
        return None
    if not isinstance(spec, list):
        spec = [spec]

    out: list[tuple[int, int | None]] = []
    for token in spec:
        # '제76조의2'·'76조의2'·'76의2'·'76-2'·'제76조'·'76' all → (76, 2)/(76, None).
        # Unify '-' (branch separator) into '의', then extract only the main and branch numbers
        # from the '제…조…의N' pattern.
        s = str(token).replace("-", "의")
        # A structural word (`별표`·`부칙`·`서식`…) at the start of the token marks a structural unit. Only the start:
        # a real article with that word in its title ('제106조(별도합산과세대상)') starts with '제'.
        if _NON_ARTICLE_UNIT_RE.match(s.lstrip()):
            continue
        m = re.search(r"(\d+)\s*조?\s*(?:의\s*(\d+))?", s)
        if not m:
            continue
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else None
        out.append((a, b))
    return out


def _statute_meta(conn: sqlite3.Connection, sid: int) -> dict[str, Any] | None:
    r = conn.execute(
        """
        SELECT id, law_id, name, short_name, kind, issuing_agency, effective_date
        FROM st_statutes WHERE id=?
        """,
        (sid,),
    ).fetchone()
    return dict(r) if r else None


def _notice_meta(conn: sqlite3.Connection, nid: int) -> dict[str, Any] | None:
    r = conn.execute(
        """
        SELECT id, notice_id, name, kind, issuing_agency, category
        FROM st_notices WHERE id=?
        """,
        (nid,),
    ).fetchone()
    return dict(r) if r else None


# ---------- point-in-time lookup helpers ----------

def _today_iso() -> str:
    """Today as 'YYYYMMDD'. With no offense_date, 'current' is the version in force today — decided by date, not by
    the '현행'/'시행예정' labels, which can lag. The evaluation fixes it to the run date."""
    return datetime.date.today().strftime('%Y%m%d')


# `st_articles` rows without an article number (`article_no_num IS NULL`) are not articles — part/chapter/section
# headings ('제22장 성풍속에 관한 죄', '[제1절에서 제2절로 변경 …]', 민법 '제2항 재판상 파양'), a repeal version's
# '…은 폐지한다.' line: a lookup by number never meets them. A row with no branch has branch 0, and one version holds
# each (number, branch) once.
#
# An article is in force from the later of its version's date and its own date (`article_eff_date`: an amendment can
# defer one article, 공공감사에 관한 법률 §22 — version 2010-07-01, article 2011-07-02).
_IN_FORCE_FROM = "MAX(s.effective_date, a.article_eff_date)"

# The per-article queries below start from the law's versions (`st_statutes s CROSS JOIN st_articles a`). Joined the
# other way, the planner (no ANALYZE statistics) starts from the article-number index `idx_st_articles_no` and walks
# that number in every law and version — article 1 is 28,307 rows of 1.67M after the history backfill. Looking up
# every article of a law that way read 8.6 GB per call (산지관리법), 1–4 minutes on the production box, against 24 MB
# along the law.


def _get_historic_article(
    conn: sqlite3.Connection,
    law_id: str,
    art_no: int,
    art_br: int,
    offense_iso: str,
) -> dict[str, Any] | None:
    """Same law_id + article (number, branch) + offense_iso → the article's row in *the last version that has it in
    force on the date* — a changed row or a whole text alike. A deleted article yields its '제N조 삭제 <…>' marker."""
    r = conn.execute(
        f"""SELECT s.effective_date, a.article_text, a.title, a.article_no, a.article_no_num, a.article_branch
           FROM st_statutes s CROSS JOIN st_articles a ON s.id=a.statute_id
           WHERE s.law_id=? AND a.article_no_num=? AND a.article_branch=? AND {_IN_FORCE_FROM} <= ?
           ORDER BY s.effective_date DESC LIMIT 1""",
        (law_id, art_no, art_br, offense_iso),
    ).fetchone()
    return dict(r) if r else None


def _pending_article(
    conn: sqlite3.Connection, law_id: str, as_of_iso: str, art_no: int, art_br: int | None,
) -> dict[str, Any] | None:
    """If the article number comes into force only after both as_of and today, the {effective_date, title, branch}
    of the earliest such row, else None. If `art_br` is None any branch matches (main article first). Decided by
    date, not by the '시행예정' label, which can lag behind the effective date."""
    q = (f"SELECT {_IN_FORCE_FROM} AS eff, a.title, a.article_branch FROM st_statutes s "
         "CROSS JOIN st_articles a ON s.id=a.statute_id "
         f"WHERE s.law_id=? AND {_IN_FORCE_FROM}>? AND a.article_no_num=?")
    params: list[Any] = [law_id, max(as_of_iso, _today_iso()), art_no]
    if art_br is not None:
        q += " AND a.article_branch=?"
        params.append(art_br)
    r = conn.execute(q + " ORDER BY eff, a.article_branch LIMIT 1", params).fetchone()
    return {"effective_date": r['eff'], "title": r['title'], "branch": r['article_branch']} if r else None


def _added_article(
    conn: sqlite3.Connection, law_id: str, as_of_iso: str, art_no: int, art_br: int | None,
) -> str | None:
    """The date the article number came into force after as_of, if it is in force by today — a number added after
    the asked date (a new article, or a full amendment that renumbered), else None. If `art_br` is None any branch
    matches."""
    q = (f"SELECT {_IN_FORCE_FROM} AS eff FROM st_statutes s CROSS JOIN st_articles a ON s.id=a.statute_id "
         f"WHERE s.law_id=? AND {_IN_FORCE_FROM}>? AND {_IN_FORCE_FROM}<=? AND a.article_no_num=?")
    params: list[Any] = [law_id, as_of_iso, _today_iso(), art_no]
    if art_br is not None:
        q += " AND a.article_branch=?"
        params.append(art_br)
    r = conn.execute(q + " ORDER BY eff LIMIT 1", params).fetchone()
    return r['eff'] if r else None


def _same_title(a: Any, b: Any) -> bool:
    """Are the article titles the same — ignoring whitespace/interpunct variants (same normalization as
    `_norm_law_name`)."""
    return bool(a) and _norm_law_name(a) == _norm_law_name(b)


def _is_repeal_kind(change_kind: Any) -> bool:
    """Is the amendment kind a version saying 'the law died' — '폐지' (repeal) / '타법폐지' (repeal by another
    act). '폐지제정' is a version that repealed an old law and enacted this one (the Income Tax Act of 1962), so it
    is not a repeal."""
    ck = str(change_kind or "")
    return "폐지" in ck and "제정" not in ck


def _is_full_text_kind(change_kind: Any) -> bool:
    """Does a row of this amendment kind hold the whole text — an enactment or a full amendment."""
    ck = str(change_kind or "")
    return "제정" in ck or "전부개정" in ck


def _is_whole(r: Any) -> bool:
    """Does the version row hold the law's whole text — loaded whole (it carries the version's serial number, which
    rows holding only the articles their amendment changed leave empty), or an enactment or full amendment."""
    return bool(r['serial_id']) or _is_full_text_kind(r['change_kind'])


def _pick_current_version(group: list, as_of_iso: str) -> dict | None:
    """From a group of versions of the same law_id, the one *current as of as_of*: the latest whole text
    (`_is_whole`) in force by then — a law not yet in force, its first version. A repeal version is passed over for
    the version in force just before it; a '폐지제정' row (an old law repealed and this one enacted under the same ID,
    e.g. 외국환거래법 1999) is a new whole text and stays. With no whole text among them (a law found only by
    an old name whose matched versions hold changed articles), the latest of them."""
    if not group:
        return None
    as_of_iso = max(as_of_iso, min(r['effective_date'] for r in group))
    in_force = [r for r in group if r['effective_date'] <= as_of_iso]
    non_repealed = [r for r in in_force if not _is_repeal_kind(r['change_kind'])] or in_force
    pool = [r for r in non_repealed if _is_whole(r)] or non_repealed
    return max(pool, key=lambda r: (r['effective_date'], r['id']))


def _resolve_current_statute_id(
    conn: sqlite3.Connection, law_id: str, as_of_iso: str,
) -> int | None:
    """law_id → statute_id current as of as_of (`_pick_current_version` rules)."""
    group = conn.execute(
        "SELECT id, effective_date, change_kind, serial_id FROM st_statutes WHERE law_id=?", (law_id,),
    ).fetchall()
    chosen = _pick_current_version(group, as_of_iso)
    return chosen['id'] if chosen else None


def _articles_in_force(
    conn: sqlite3.Connection, law_id: str, as_of_iso: str,
) -> set[tuple[int, int]]:
    """Article keys `(article_no_num, branch)` **alive** as of as_of: the articles of the latest whole text by then
    (`_is_whole`, picked as in `_pick_current_version`) and of the amendments after it.

    The per-article walk (`_get_historic_article`) returns the last version that had a number, so a number a full
    amendment retired would come back with its old text as if current (근로기준법 §32의2 of 2007, now §27). The whole
    text alone is not enough either: articles a later amendment changed would revert to old text.
    """
    group = conn.execute(
        "SELECT id, effective_date, change_kind, serial_id FROM st_statutes WHERE law_id=? AND effective_date <= ?",
        (law_id, as_of_iso),
    ).fetchall()
    base = _pick_current_version([r for r in group if _is_whole(r)], as_of_iso)
    if base is None:   # before the law's first version, which is a whole text
        return set()
    keys = {tuple(r) for r in conn.execute(
        "SELECT article_no_num, article_branch FROM st_articles WHERE statute_id=? AND article_no_num IS NOT NULL",
        (base['id'],))}
    keys.update(tuple(r) for r in conn.execute(
        """SELECT a.article_no_num, a.article_branch
           FROM st_articles a JOIN st_statutes s ON s.id=a.statute_id
           WHERE s.law_id=? AND s.effective_date > ? AND s.effective_date <= ?
             AND a.article_no_num IS NOT NULL""",
        (law_id, base['effective_date'], as_of_iso)))
    return keys


def _fmt_eff_iso(eff: Any) -> str:
    """'20260102' → '2026-01-02' for display."""
    e = str(eff or "")
    return f"{e[:4]}-{e[4:6]}-{e[6:8]}" if len(e) == 8 and e.isdigit() else e


def _norm_law_name(s: Any) -> str:
    """Normalization for statute-name comparison — removes whitespace + interpunct variants (same shape as
    `_name_norm_sql`)."""
    return _strip_dots((s or "").replace(" ", ""))


def _current_version_ref(conn: sqlite3.Connection, law_id: str, sid: int, name: Any) -> dict[str, Any] | None:
    """Marker of the version current today (`_pick_current_version`, not the '현행' label, which an old version
    loaded whole can lack) when `sid`, named `name`, is not it; None if it is. `renamed`: the current version has
    another name (a full amendment renamed the law) — a rename, not a repeal, which callers test by
    `_is_repealed_at`."""
    cur_id = _resolve_current_statute_id(conn, law_id, _today_iso())
    if cur_id == sid:
        return None
    cur = _statute_meta(conn, cur_id)
    return {
        "id": cur["id"],
        "name": cur["name"],
        "effective_date": cur["effective_date"],
        "renamed": _norm_law_name(cur["name"]) != _norm_law_name(name),
    }


def _to_current(conn: sqlite3.Connection, meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The version current today in place of `meta`, and the name asked for when the current version has another
    (`asked_name`: a lookup by an old version's id must say why another name came back)."""
    cur_id = _resolve_current_statute_id(conn, meta["law_id"], _today_iso())
    if cur_id == meta["id"]:
        return meta, None
    cur = _statute_meta(conn, cur_id)
    if _norm_law_name(cur["name"]) == _norm_law_name(meta["name"]):
        return cur, None
    return cur, {"name": meta["name"], "effective_date": meta["effective_date"]}


def _is_repealed_as_of(conn: sqlite3.Connection, law_id: str, as_of_iso: str) -> bool:
    """Whether law_id is no longer in force as of as_of (`_is_repealed_at`, the one test). Queries the full history —
    correct even if a partial search group misses the repeal row."""
    return _is_repealed_at(conn, law_id, as_of_iso) is not None


def _fold_versions(conn: sqlite3.Connection, rows: list, offense_iso: str | None) -> list:
    """Among search result rows, select only one per law_id.

    - offense_iso given: the maximum effective_date <= offense_iso (valid at that time)
    - not given (current): `_pick_current_version` as of today. Laws repealed by the search date are dropped
      afterwards (`_drop_repealed`).

    The chosen version carries the best evidence of any matched version of its law (`_with_law_evidence`).
    """
    by_law: dict[str, list] = {}
    for r in rows:
        by_law.setdefault(r['law_id'], []).append(r)

    out = {}
    for group in by_law.values():
        # point-in-time filter
        if offense_iso:
            candidates = [r for r in group if r['effective_date'] <= offense_iso]
            if not candidates:
                # no version before the act date — skip (a statute that didn't exist at that time at all)
                continue
            chosen = max(candidates, key=lambda r: r['effective_date'])
        else:
            chosen = _pick_current_version(group, _today_iso())
        out[chosen['id']] = _with_law_evidence(chosen, group, dated=bool(offense_iso))
    # keep the original sort order (rank-based) — order by first appearance of the chosen rows
    return [out[r['id']] for r in rows if r['id'] in out]


def _with_law_evidence(chosen: Any, group: list, *, dated: bool) -> Any:
    """The chosen version with the best search evidence of any matched version of its law — the name the query
    named (`name_in_query`, and that row's names as `matched_name`/`matched_short_name` for `_rank_name_rows`) and
    name coverage, and for a point-in-time search the body hits too. The search window ranks laws by their best
    version row and holds all their matched versions; ranked by the chosen row's own scores, a law asked for by an
    old name would lose the evidence that the query named it ('주택건설촉진법', now 주택법), and a past version, which
    is not in the body index, its law's body hits. A current search keeps the current version's own body hits."""
    named = max(group, key=lambda r: r['name_in_query'])
    best = {"name_in_query": named['name_in_query'],
            "fts_hits": max(r['fts_hits'] for r in group) if dated else chosen['fts_hits'],
            "name_cover": max(r['name_cover'] for r in group)}
    if all(chosen[k] >= v for k, v in best.items()):
        return chosen
    out = dict(chosen) | best
    if best["name_in_query"] > chosen['name_in_query']:
        out |= {"matched_name": named['name'], "matched_short_name": named['short_name']}
    return out


def _drop_repealed(conn: sqlite3.Connection, rows: list, offense_iso: str | None, named: int = 0) -> list:
    """Drop laws repealed by the search date — today for a current search (the version just before a repeal would
    show up as a 'zombie'; the current law usually exists under another law_id), the offense date for a
    point-in-time search (a law repealed before then was not in force; one repealed later still was).

    Both keep a repealed law that holds the longest name the query names (`named`, the largest `name_in_query` in
    the search window) when no law in force then holds it, e.g. '증권거래법': the model asked for that law, and its
    row says it is repealed. Laws the query only brushes still go (교육법시행령 for '소득세법 시행령', '시장법' for
    '자본시장법'), and so does a repealed namesake of a law in force (the 1953 근로기준법 beside the 1997 one)."""
    when = offense_iso or _today_iso()
    dead = {r['id'] for r in rows if _is_repealed_as_of(conn, r['law_id'], when)}
    keep = named
    if keep and any(r['name_in_query'] == keep for r in rows if r['id'] not in dead):
        keep = 0
    return [r for r in rows if r['id'] not in dead or (keep and r['name_in_query'] == keep)]


def _is_repealed_at(conn: sqlite3.Connection, law_id: str, offense_iso: str | None) -> dict | None:
    """{effective_date, change_kind} if law_id is no longer in force on offense_iso (or now), else None.

    Repealed = the latest repeal or enactment version by then is a repeal: a repeal ends the law, and only a new
    enactment under the same law_id brings it back (국회의원선거법: 타법폐지 1958, 제정 1960). An amendment listed after the
    repeal does not — law.go.kr lists some (항공법 시행령: repealed 2017-03-30, '타법개정' 2017-05-01; the 1961
    헌법재판소법, repealed 1963, '타법개정' 1998), and taking the latest version of any kind made those laws current.
    Laws that lost effect with no repeal version at law.go.kr are in `st_lapsed`: temporary acts from the day after
    their last day, and laws whose day the corpus does not know (empty `lapsed_on`: old colonial and
    military-government laws and the like) for a lookup of today, with an empty date. Search, the list's repeal line,
    the article lookups' repeal line and the charge index all use this one test.
    """
    target = offense_iso or '99999999'
    r = conn.execute(
        """SELECT effective_date, change_kind FROM st_statutes
           WHERE law_id=? AND effective_date<=? AND (change_kind LIKE '%폐지%' OR change_kind LIKE '%제정%')
           ORDER BY effective_date DESC LIMIT 1""",
        (law_id, target),
    ).fetchone()
    if r and _is_repeal_kind(r[1]):
        return {"effective_date": r[0], "change_kind": r[1]}
    lapsed = conn.execute("SELECT lapsed_on FROM st_lapsed WHERE law_id=?", (law_id,)).fetchone()
    if lapsed and target >= (lapsed[0] or _today_iso()):
        return {"effective_date": lapsed[0], "change_kind": "효력 상실"}
    return None


# ---------- matched article in list search ----------
#
# A list line shows the article the query's words matched best, cut at its head: `article_text` starts like
# '제10조의4(권리금 회수기회 보호 등) ① …', so the head keeps which article matched.
MATCHED_ARTICLE_MAX_CHARS = 200


def _clip_article(text: str | None) -> str:
    """Article text → one line; a trailing `…` says it was cut."""
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if len(t) > MATCHED_ARTICLE_MAX_CHARS:
        return t[:MATCHED_ARTICLE_MAX_CHARS].rstrip() + "…"
    return t


def _matched_articles(
    conn: sqlite3.Connection,
    ids: list[int],
    fts_query: str,
    *,
    fts_table: str,
    art_table: str,
    parent_col: str,
    articles_only: str = "",
) -> dict[int, str]:
    """{parent id: head of the article the query matched best}. Statutes whose body the query didn't match have no key.
    `articles_only` is the condition on the article row (`a.`) that leaves out headings, where the table has them.
    One query for all results: the window passes only article ids, and the text is read from the winning row only."""
    if not ids:
        return {}
    ph = ",".join("?" * len(ids))
    try:
        rows = conn.execute(
            f"""
            WITH best AS (
              SELECT a.{parent_col} AS pid, a.id AS aid,
                     ROW_NUMBER() OVER (PARTITION BY a.{parent_col}
                                        ORDER BY {bm25_expr(fts_table, _ART_BM25_WEIGHTS)}) AS rn
              FROM {fts_table} f JOIN {art_table} a ON a.id = f.rowid
              WHERE {fts_table} MATCH ? AND a.{parent_col} IN ({ph}) {articles_only}
            )
            SELECT b.pid, substr(x.article_text, 1, ?) AS t
            FROM best b JOIN {art_table} x ON x.id = b.aid
            WHERE b.rn = 1
            """,
            (fts_query, *ids, MATCHED_ARTICLE_MAX_CHARS * 2),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}          # FTS syntax error or no index: the list goes out without matched articles
    return {r["pid"]: _clip_article(r["t"]) for r in rows if (r["t"] or "").strip()}


# ---------- mode A: list search ----------

def _search_statutes(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    *,
    body: str,
    offense_date: str | None = None,
) -> list[dict[str, Any]]:
    """Statute search by name and article text. A law has many version rows; one per law is shown
    (`_fold_versions`). `body` is the article-text FTS expression (`_body_match`).

    - default (no offense_date): among the same law_id, the version current today (`_pick_current_version`)
    - offense_date given: among the same law_id, *the last version before that time* — past versions included;
      laws repealed by then only when the query names them (`_drop_repealed`)
    """
    offense_iso = to_iso_date(offense_date)
    # folding buffer — so one can be selected even when many point-in-time versions of the same law_id come in
    sql_limit = max(limit * 8, 80)

    # Alias normalization + morpheme tokens — split compounds without spaces ("광업제조업자") into noun tokens
    # ('광업'/'제조'/'업자') used for statute-name LIKE coverage. Without tokens (kiwipiepy not installed etc.),
    # fall back to the whole normalized query. The article text is matched by `body`, not by these tokens:
    # common fragments would add noise.
    normalized, tokens = _statute_name_tokens(conn, query)
    like_terms = tokens or [normalized]
    # Name scores are computed once per distinct (name, abbreviation) and joined to the version rows — a law
    # keeps its name over many versions, and the normalization below is the costly part of this query.
    name_norm = _name_norm_sql("n.name")
    cover_sql = " + ".join(f"({name_norm} LIKE ?)" for _ in like_terms)
    name_or = " OR ".join(f"{name_norm} LIKE ?" for _ in like_terms)
    like_params = [f"%{t}%" for t in like_terms]
    # Length of whichever of the name / official abbreviation is wholly contained in the query
    # (`_name_in_query_sql`).
    spans = _query_name_spans(conn, query)
    name_in_q = _name_in_query_sql("n.name", len(spans))
    short_in_q = _name_in_query_sql("COALESCE(n.short_name,'')", len(spans))
    # A candidate if the statute name contains at least one token, the official abbreviation is in the query,
    # or there is a body FTS hit. The abbreviation condition is needed separately because the abbreviation's
    # characters may be absent from the official name ('상증세법' ↔ '상속세 및 증여세법'), so token LIKE alone
    # wouldn't even make it a candidate.
    # Order: does the query wholly contain the name (longer name first) → body hits → statute-name token
    # coverage → name length. Body hits rank above name length, or a law whose name a fragment brushes
    # ('도시철도운전규칙' for `음주운전`, drunk driving) outranks the one whose text matches (the Road Traffic Act).
    # The order here decides the **candidate window** for the Python re-ranking below (`_rank_name_rows`);
    # the final order is owned by the Python side, which knows morpheme-boundary coverage.
    # The window holds laws, each ranked by its best version row, with all their matched versions for folding —
    # otherwise the many versions of one law fill it and push other laws out. Ties go by row id, so the result
    # does not depend on the query plan.
    sql = f"""
    WITH fts_hits AS (
      SELECT a.statute_id, COUNT(*) AS hits
      FROM st_articles_fts f
      JOIN st_articles a ON a.id = f.rowid
      WHERE st_articles_fts MATCH ?
      GROUP BY a.statute_id
    ), names AS MATERIALIZED (
      SELECT n.name, n.short_name, ({cover_sql}) AS name_cover,
             MAX({name_in_q}, {short_in_q}) AS name_in_query,
             (({name_or}) OR {short_in_q} > 0) AS name_hit
      FROM (SELECT DISTINCT name, short_name FROM st_statutes) n
    ), scored AS (
      SELECT s.id, s.law_id, s.name, s.short_name, s.kind, s.effective_date, s.change_kind, s.serial_id,
             COALESCE(h.hits, 0) AS fts_hits, m.name_cover, m.name_in_query
      FROM st_statutes s
      JOIN names m ON m.name = s.name AND m.short_name IS s.short_name
      LEFT JOIN fts_hits h ON h.statute_id = s.id
      WHERE m.name_hit OR h.hits > 0
    ), top AS (
      SELECT law_id, MAX(name_in_query) AS q, MAX(fts_hits) AS f,
             MAX(name_cover) AS c, MIN(length(name)) AS n, MIN(id) AS first
      FROM scored GROUP BY law_id ORDER BY q DESC, f DESC, c DESC, n ASC, first ASC LIMIT ?
    )
    SELECT scored.* FROM scored JOIN top USING (law_id)
    ORDER BY name_in_query DESC, fts_hits DESC, name_cover DESC, length(name) ASC, scored.id ASC
    """
    # params in order of appearance: MATCH, cover, name_in_q, short_in_q, name_or, short_in_q, LIMIT
    rows = conn.execute(
        sql, [body, *like_params, *spans, *spans, *like_params, *spans, sql_limit]).fetchall()

    # folding — select one per law_id
    named = max((r['name_in_query'] for r in rows), default=0)
    rows = _fold_versions(conn, rows, offense_iso)
    rows = _drop_repealed(conn, rows, offense_iso, named)
    # Final order and drops are owned by Python, which knows morpheme boundaries (SQL only builds the candidate window).
    rows = _rank_name_rows(conn, rows, query, limit)

    matched = _matched_articles(
        conn, [r["id"] for r in rows], body,
        fts_table="st_articles_fts", art_table="st_articles", parent_col="statute_id",
        articles_only="AND a.article_no_num IS NOT NULL",   # a chapter heading is no matched article ('제4관 등록세')
    )
    today = _today_iso()
    out = []
    for r in rows:
        repealed = _is_repealed_as_of(conn, r["law_id"], today)
        out.append(
            {
                "statute_id": r["id"],
                "law_id": r["law_id"],
                "name": r["name"],
                # Official abbreviation — not printed; the merge (`_merge_law_notice_matches`) ranks by it too.
                "short_name": r["short_name"],
                "kind": r["kind"],
                "effective_date": r["effective_date"],
                "matched_article": matched.get(r["id"]),
                # A line found by an old name stays, marked with the current version (a repealed law has none).
                "is_repealed": repealed,
                "current": None if repealed else _current_version_ref(conn, r["law_id"], r["id"], r["name"]),
                # The evidence the merge ranks by, as within the corpus.
                "fts_hits": r["fts_hits"],
                "name_in_query": r["name_in_query"],
            }
        )
    return out


def _search_notices(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    *,
    body: str,
) -> list[dict[str, Any]]:
    """Administrative-rule search by name and article text (`body` as in `_search_statutes`)."""
    # The same morpheme-token coverage as statute search: a name with particles ('의') and spaces, like
    # '개인정보의 안전성 확보조치 기준', ranks by how many query tokens ('개인'/'정보'/'안전') it contains (name_cover).
    normalized, tokens = _statute_name_tokens(conn, query)
    spans = _query_name_spans(conn, query)
    like_terms = tokens or [normalized]
    name_norm = _name_norm_sql("n.name")
    cover_sql = " + ".join(f"({name_norm} LIKE ?)" for _ in like_terms)
    name_or = " OR ".join(f"{name_norm} LIKE ?" for _ in like_terms)
    like_params = [f"%{t}%" for t in like_terms]
    sql = f"""
    WITH fts_hits AS (
      SELECT a.notice_id, COUNT(*) AS hits
      FROM st_notice_articles_fts f
      JOIN st_notice_articles a ON a.id = f.rowid
      WHERE st_notice_articles_fts MATCH ?
      GROUP BY a.notice_id
    )
    SELECT n.id, n.notice_id, n.name, n.kind, n.effective_date, n.category,
           COALESCE(h.hits, 0) AS fts_hits,
           ({cover_sql}) AS name_cover,
           {_name_in_query_sql("n.name", len(spans))} AS name_in_query
    FROM st_notices n
    LEFT JOIN fts_hits h ON h.notice_id = n.id
    WHERE n.has_text_content = 1
      AND {_NOTICE_CURRENT}
      AND (({name_or}) OR h.hits > 0)
    ORDER BY name_in_query DESC, name_cover DESC, length(n.name) ASC, fts_hits DESC
    LIMIT ?
    """
    # Uses the same keys as statute search — if only one side knew containment matching, the merge would apply
    # different yardsticks.
    rows = conn.execute(
        sql,
        [body, *like_params, *spans, *like_params, limit],
    ).fetchall()

    # Same rule as statutes — name-boundary re-ranking and drops are owned by `_rank_name_rows` (notices have
    # no official abbreviation, hence `alt_key=""`).
    rows = _rank_name_rows(conn, rows, query, limit, alt_key="")

    matched = _matched_articles(
        conn, [r["id"] for r in rows], body,
        fts_table="st_notice_articles_fts", art_table="st_notice_articles", parent_col="notice_id",
    )
    return [
        {
            "statute_id": r["id"],
            "notice_id": r["notice_id"],
            "name": r["name"],
            "kind": r["kind"],
            "category": r["category"],
            "effective_date": r["effective_date"],
            "matched_article": matched.get(r["id"]),
            "fts_hits": r["fts_hits"],
            "name_in_query": r["name_in_query"],
        }
        for r in rows
    ]


@functools.lru_cache(maxsize=100_000)
def _name_morphemes(name: str) -> frozenset[str]:
    """Statute/notice **name** → set of content morphemes — the single source for name-side boundary tests.

    Why names must be seen as morphemes: matching query tokens against names with `LIKE '%token%'` lets in
    accidents cutting across the middle of a word — `사기` (fraud) is inside '공공감사기준'·'자원봉사기본법'·'군사기밀
    보호법', and `운전` inside '도시철도운전규칙'. Split with the same analyzer, '감사기준' is `[감사, 기준]`, with no
    '사기', while '보험사기방지 특별법' is `[보험, 사기, 방지, 특별, 법]`.

    A name without nouns gives an empty set, and the caller falls back to substring matching."""
    return frozenset(_strip_dots(t.form) for t in _kiwi().tokenize(name or "")
                     if t.tag in ("NNG", "NNP") and len(_strip_dots(t.form)) >= 2)


def _name_token_hits(name: str, tokens: list[str], n_norm: str) -> int:
    """Number of query tokens the name contains — **by morpheme boundary** (accidental mid-word matches don't count).

    `n_norm` is the name with spaces/interpuncts removed (for the fallback comparison)."""
    morph = _name_morphemes(name)
    if morph:
        return sum(1 for t in tokens if t in morph)
    return sum(1 for t in tokens if t in n_norm)


def _name_cover_key(name: str, q_norm: str, tokens: list[str], spans=()):
    """Name-relevance sort key shared by statutes and notices (smaller ranks higher). **The same metric** as
    name_in_query/name_cover in `_search_statutes`: does the query wholly contain the name (longer name first)
    > higher token coverage. Used to interleave statutes + notices *on merit*, without quotas.

    Coverage is counted on name-side morpheme boundaries (`_name_token_hits`), so '사기' is not in '공공감사기준'.

    ⚠ If this key disagrees with the SQL ordering, the merge undoes the search — administrative rules with merely
    long names cover up again the statute the search raised. The two places move together. `spans` is the same
    as in SQL (`_query_name_spans`), and the name is normalized as `_name_norm_sql` does (spaces and interpuncts
    removed, so '초ㆍ중등교육법' can match exactly)."""
    n = _strip_dots((name or "").replace(" ", ""))
    if n and n in spans:
        # Exact match is this branch's maximum length (= the full query), so it needs no separate case.
        return (0, -len(n))
    if tokens:
        cover = _name_token_hits(name, tokens, n)
        return (1, len(tokens) - cover)        # higher coverage (= smaller deficiency) ranks higher
    return (1, 0) if (q_norm and q_norm in n) else (2, 0)


def _name_rank_key(cover_key, name_in_query: int, fts_hits: int, n_tokens: int, name_len: int):
    """**4-tier sort key** on name/body evidence (smaller ranks higher) — used both within a corpus and by the merge.

    ① the query wholly contains the name (`name_in_query`) — longer name (more specific reference) first
    ② the query tokens are **all** name morphemes — more body hits first → shorter name
    ③ the article body matched (`fts_hits`) — smaller name deficiency first → more body hits first
    ④ partial name coverage — smaller deficiency first → shorter name

    ⚠ This function is the only definition of order for `_rank_name_rows` (within each statute/notice corpus)
    and `_merge_law_notice_matches` (interleaving across corpora). If the two diverged, the merge would undo the
    search (a merge by name alone puts a notice back above the Road Traffic Act for `음주운전`)."""
    if name_in_query:
        return (0, cover_key[1], 0, 0)
    deficiency = cover_key[1] if cover_key[0] == 1 else n_tokens
    if cover_key[0] == 1 and deficiency == 0:
        return (1, -int(fts_hits or 0), 0, name_len)
    if fts_hits:
        return (2, deficiency, -int(fts_hits or 0), name_len)
    return (3, deficiency, 0, name_len)


def _rank_name_rows(conn: sqlite3.Connection, rows, query: str, limit: int, *, alt_key: str = "short_name"):
    """**Re-rank** by name relevance (morpheme boundaries) → body hits → name length, **and drop rows without
    evidence**.

    Why not finish with the SQL ordering: computing coverage correctly needs **name-side morpheme boundaries**
    ('공공감사기준' has no '사기' — it is `[감사, 기준]`), which needs Kiwi and can't be done inside SQL. So SQL
    only builds the **candidate window** (ordered `name_in_query` → `fts_hits` so rows with evidence stay inside
    the window), and the final order and drops are decided here.

    Drop rule: eligible if the name is wholly contained (`name_in_query`) · the name's morphemes contain a query
    token · the article body matched (`fts_hits`). If none of the three, drop it (`사기` brushes '공공감사기준').

    ⚠ Rows pushed out of the candidate window (`sql_limit` in `_search_statutes`) are not revived —
    hence the first SQL sort key is `name_in_query` and the second `fts_hits`."""
    _, tokens = _statute_name_tokens(conn, query)
    spans = frozenset(_query_name_spans(conn, query))
    q_norm = _strip_dots((query or "").replace(" ", ""))

    def name_key(r):
        # also the names of the version that matched, when a law's chosen version carries them (`_with_law_evidence`)
        names = [r[k] for k in (alt_key, "matched_name", "matched_short_name") if k in r.keys() and r[k]]
        return min(_name_cover_key(n, q_norm, tokens, spans) for n in [r["name"], *names])

    def rank_key(r):
        return _name_rank_key(name_key(r), r["name_in_query"] or 0, r["fts_hits"] or 0,
                              len(tokens), len(r["name"] or ""))

    ranked = sorted(rows, key=rank_key)
    if not tokens:
        return ranked[:limit]
    out = []
    for r in ranked:
        k = name_key(r)
        if k[0] == 0 or r["fts_hits"] or k[1] < len(tokens):
            out.append(r)
    return out[:limit]


def _merge_law_notice_matches(conn: sqlite3.Connection, matches, query, limit):
    """Interleave statute + notice search results by name/body evidence (_name_rank_key) — **no quotas**. The
    sort is stable, so within equal relevance the input order (statutes first, each corpus's internal ranking)
    is kept.

    A statute is keyed by the better of its name and official abbreviation, so one whose official name is not in
    the query but whose abbreviation is ('신용정보법' ↔ '신용정보의 이용 및 보호에 관한 법률', the Credit Information
    Act) isn't pushed back down in the merge. Notices and treaties have no abbreviation.

    `fts_hits`·`name_in_query` are carried by both corpora's search-result dicts; a missing one reads as 0."""
    normalized, tokens = _statute_name_tokens(conn, query)
    spans = frozenset(_query_name_spans(conn, query))

    def key(m):
        best = _name_cover_key(m.get("name", ""), normalized, tokens, spans)
        if m.get("short_name"):
            best = min(best, _name_cover_key(m["short_name"], normalized, tokens, spans))
        return _name_rank_key(best, m.get("name_in_query") or 0, m.get("fts_hits") or 0,
                              len(tokens), len(m.get("name", "") or ""))

    return sorted(matches, key=key)[:limit]


# ---------- mode B: article outline ----------

OUTLINE_TOC_CHARS = 15_000    # an outline listing more than this comes as its 편·장·절 table of contents (민법, 상법)
_HEADING_RE = re.compile(r"제\s*\d+\s*(편|장|절|관)(?:의\s*\d+)?(?=\s|$)")
_HEADING_LEVEL = {"편": 1, "장": 2, "절": 3, "관": 4}


def _toc(conn: sqlite3.Connection, law_id: str, on_iso: str, articles: list[dict[str, Any]]) -> list[dict] | None:
    """The 편·장·절·관 headings of the last version by the date that has them (heading rows have no article number;
    only whole-text versions carry them), each with the range of the listed articles under it. None when no version
    has headings. The source drops the first sub-heading under a heading (민법 제1편 has no '제1장 통칙'), so a
    heading's range can begin before its first sub-heading's."""
    row = conn.execute(
        """SELECT s.id FROM st_statutes s WHERE s.law_id=? AND s.effective_date<=?
           AND EXISTS (SELECT 1 FROM st_articles a WHERE a.statute_id=s.id AND a.article_no IS NULL)
           ORDER BY s.effective_date DESC LIMIT 1""",
        (law_id, on_iso),
    ).fetchone()
    if row is None:
        return None
    heads: list[dict[str, Any]] = []
    for no, num, branch, text in conn.execute(
        "SELECT article_no, article_no_num, article_branch, article_text FROM st_articles WHERE statute_id=? "
        "ORDER BY id", (row[0],),
    ):
        if no is None:
            line = re.sub(r"\s*<[^>]*>", "", (text or "").strip().split("\n")[0])
            m = _HEADING_RE.match(line)
            if m:
                heads.append({"level": _HEADING_LEVEL[m[1]], "title": " ".join(line.split()), "start": None})
        elif num is not None:
            for h in reversed(heads):   # the headings just above this article start at it
                if h["start"] is not None:
                    break
                h["start"] = (num, branch or 0)
    heads = [h for h in heads if h["start"] is not None]
    if not heads:
        return None
    def key(a):
        return a["no_num"], a["branch"] or 0

    # Each listed article goes under the last heading starting at or before it.
    owned: list[list[dict[str, Any]]] = [[] for _ in heads]
    before = []
    starts = [h["start"] for h in heads]
    for a in articles:
        i = bisect.bisect_right(starts, key(a)) - 1
        (owned[i] if i >= 0 else before).append(a)
    out = []
    if before:
        out.append({"level": heads[0]["level"], "title": "", "first": key(before[0]), "last": key(before[-1])})
    for i, h in enumerate(heads):
        j = i + 1
        while j < len(heads) and heads[j]["level"] > h["level"]:
            j += 1
        under = [a for o in owned[i:j] for a in o]
        if under:
            out.append({"level": h["level"], "title": h["title"], "first": key(under[0]), "last": key(under[-1])})
    top = min(t["level"] for t in out)   # a law with no 편 starts at 장
    for t in out:
        t["level"] -= top - 1
    return out


def _as_toc(conn: sqlite3.Connection, out: dict[str, Any], law_id: str, on_iso: str) -> None:
    """An outline listing more than OUTLINE_TOC_CHARS gives its table of contents in place of the article titles."""
    articles = out["articles"]
    if sum(len(f"- {a['no']}: {a['title']}") + 1 for a in articles) <= OUTLINE_TOC_CHARS:
        return
    toc = _toc(conn, law_id, on_iso, articles)
    if toc:
        out["toc"], out["article_count"] = toc, len(articles)
        del out["articles"]


def _articles_on(conn: sqlite3.Connection, law_id: str, on_iso: str) -> list[dict[str, Any]]:
    """The law's article list on the date: each number alive then (`_articles_in_force`) with the row an article
    lookup reads (`_get_historic_article`) — title, text and the date of its version. Articles without a title
    (the Constitution) and a single-body regulation's body at 0 are listed too."""
    rows = conn.execute(
        f"""SELECT a.article_no, a.article_no_num, a.article_branch, a.title, a.article_text, s.effective_date
           FROM st_articles a JOIN st_statutes s ON s.id=a.statute_id
           WHERE s.law_id=? AND a.article_no_num IS NOT NULL AND {_IN_FORCE_FROM} <= ?
           ORDER BY a.article_no_num, a.article_branch, s.effective_date DESC""",
        (law_id, on_iso),
    ).fetchall()
    in_force = _articles_in_force(conn, law_id, on_iso)
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        key = (r['article_no_num'], r['article_branch'])
        if key in seen or key not in in_force:
            continue
        seen.add(key)
        out.append({'no': r['article_no'], 'no_num': r['article_no_num'], 'branch': r['article_branch'],
                    'title': r['title'] or '', 'text': r['article_text'], 'eff_date': r['effective_date']})
    return out


def _outline_statute(
    conn: sqlite3.Connection, sid: int, offense_date: str | None = None,
) -> dict[str, Any] | None:
    """Article numbers and titles of the statute. On offense_date: the articles in force then, under the version's
    own name, with a marker of the current version. Otherwise (or with no version in force at that date): the
    current version's articles in force today."""
    meta = _statute_meta(conn, sid)
    if meta is None:
        return None

    offense_iso = to_iso_date(offense_date)
    law_id = meta['law_id']
    first = conn.execute("SELECT MIN(effective_date) FROM st_statutes WHERE law_id=?", (law_id,)).fetchone()[0]
    earliest = None
    if offense_iso and first > offense_iso:
        # No version in force at that date: the current outline, marked as such (as in `_detail_statute`).
        earliest, offense_iso = first, None
    if offense_iso:
        gone = _is_repealed_at(conn, law_id, _today_iso())
        out = {
            'mode': 'outline',
            'statute': _outline_head(meta),
            'offense_date': offense_iso,
            'articles': _titles(_articles_on(conn, law_id, offense_iso)),
        }
        _as_toc(conn, out, law_id, offense_iso)
        cur_ref = None if gone else _current_version_ref(conn, law_id, meta['id'], meta.get('name'))
        if cur_ref:
            out['current_version'] = cur_ref
        rep = _is_repealed_at(conn, law_id, offense_iso) or gone
        if rep:
            out['repealed'] = rep
        return out

    # The articles in force today — a law not yet in force, at its first version; a repealed law, its last list.
    meta, asked = _to_current(conn, meta)
    as_of = _today_iso()
    on = max(as_of, first)
    articles = _articles_on(conn, law_id, on)

    # A single-body regulation (no 제N조 structure, one untitled body at number 0 — e.g. the statutory interest rate
    # under 소송촉진 등에 관한 특례법) offers nothing to pick from an outline, so its short body comes as a detail.
    if articles and not any(a["no_num"] > 0 for a in articles):
        out = {
            "mode": "detail",
            "statute": _outline_head(meta),
            "articles": articles,
            "missing": [],
            "note": "제N조 구조가 없는 단일 본문 규정 — 아래가 본문 전문입니다.",
        }
        if asked:
            out["asked_name"] = asked
        return out

    out = {
        "mode": "outline",
        "text_date": _latest_version_date(conn, law_id, on),   # the list holds every amendment by then
        "statute": _outline_head(meta),
        "articles": _titles(articles),
    }
    _as_toc(conn, out, law_id, on)
    if asked:
        out["asked_name"] = asked
    if earliest:
        out["no_version_at"] = {"date": to_iso_date(offense_date), "earliest": earliest}
    # A law repealed by as_of: the list is its last text, and the header says it is no longer in force.
    rep = _is_repealed_at(conn, law_id, as_of)
    if rep:
        del out["text_date"]
        out["repealed"] = rep
    return out


def _outline_head(meta: dict[str, Any]) -> dict[str, Any]:
    return {"id": meta["id"], "law_id": meta["law_id"], "name": meta["name"], "kind": meta["kind"],
            "agency": meta["issuing_agency"]}


def _titles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Outline entries: the article number and title."""
    return [{k: a[k] for k in ("no", "no_num", "branch", "title")} for a in articles]


_NOTICE_TITLE_RE = re.compile(r'^\s*제\s*\d+(?:-\d+)?\s*조(?:의\s*\d+)?\s*\(([^)]*)\)')


def _notice_article_title(text: str) -> str:
    """Notice article title — notices have no title column; the title sits in the '제N조(제목) …' head of
    article_text. Extracts only the title inside the parentheses ('' if none)."""
    m = _NOTICE_TITLE_RE.match(text or "")
    return m.group(1).strip() if m else ""


def _outline_notice(conn: sqlite3.Connection, nid: int) -> dict[str, Any] | None:
    meta = _notice_meta(conn, nid)
    if meta is None:
        return None

    if meta["category"] == "article_form":
        rows = conn.execute(
            """
            SELECT article_no_str, article_text
            FROM st_notice_articles
            WHERE notice_id=? AND article_no IS NOT NULL ORDER BY article_seq
            """,
            (nid,),
        ).fetchall()
        # The number shown is article_no_str ('1', '6의2'): article_no is num*1000+branch. Rows without a number are
        # part/chapter/section headings.
        articles = [{"no": r["article_no_str"], "title": _notice_article_title(r["article_text"])} for r in rows]
    else:
        articles = []

    return {
        "mode": "outline",
        "statute": {
            "id": meta["id"],
            "notice_id": meta["notice_id"],
            "name": meta["name"],
            "kind": meta["kind"],
            "category": meta["category"],
            "agency": meta["issuing_agency"],
        },
        "articles": articles,
        "note": (
            None
            if meta["category"] == "article_form"
            else f"category={meta['category']} — 조문 분리 안 됨, articles=[...] 명시해 detail 모드로 호출"
        ),
    }


# ---------- mode C: article text ----------

def _detail_statute(
    conn: sqlite3.Connection,
    sid: int,
    specs: list[tuple[int, int | None]],
    *,
    offense_date: str | None = None,
) -> dict[str, Any] | None:
    """Article text for specs [(num, None), (347, 2), ...] — (num, None) is the main article and all its branches,
    (num, branch) that branch only. The text is the one in force on offense_date, or today in the current version
    (`_detail_statute_at_date`). No specs: the statute header alone (an addenda-only lookup)."""
    meta = _statute_meta(conn, sid)
    if meta is None:
        return None

    if not specs:
        return {
            "mode": "detail",
            "statute": {"id": meta["id"], "law_id": meta["law_id"], "name": meta["name"], "kind": meta["kind"]},
            "articles": [],
            "missing": [],
        }

    offense_iso = to_iso_date(offense_date)

    # Current and point-in-time text come from the same per-article walk (`_detail_statute_at_date`). Without
    # offense_date it walks to the latest version in force today (pending amendments are not current).
    law_id = meta['law_id']
    first = conn.execute("SELECT MIN(effective_date) FROM st_statutes WHERE law_id=?", (law_id,)).fetchone()[0]
    as_of, asked, earliest, pending = offense_iso, None, None, None
    if as_of and first > as_of:
        # The corpus has no version in force at that date: serve the current text and say so in the header,
        # instead of reporting its articles as "not yet in force" at that date.
        earliest, as_of = first, None
    if as_of is None:
        # The header is the version the text is walked from; a different name asked for goes out as `asked_name`.
        meta, asked = _to_current(conn, meta)
        as_of = _today_iso()
        if first > as_of:   # a law not yet in force: its first version
            as_of = pending = first
    res = _detail_statute_at_date(conn, meta, specs, as_of, asked_iso=None if earliest else offense_iso)
    if asked:
        res["asked_name"] = asked
    if pending:
        res["text_date"] = pending
    elif earliest:
        res["no_version_at"] = {"date": offense_iso, "earliest": earliest}
    elif offense_iso is None and not res.get("repealed"):
        res["text_date"] = _latest_version_date(conn, law_id, as_of)   # current text: every amendment in force today
    return res


def _latest_version_date(conn: sqlite3.Connection, law_id: str, as_of_iso: str) -> str | None:
    """Effective date of the law's latest version in force on the date — the text walked to it holds every amendment
    by then."""
    return conn.execute("SELECT MAX(effective_date) FROM st_statutes WHERE law_id=? AND effective_date <= ?",
                        (law_id, as_of_iso)).fetchone()[0]


def _detail_statute_at_date(
    conn: sqlite3.Connection,
    meta: dict[str, Any],
    specs: list[tuple[int, int | None]],
    offense_iso: str,
    *,
    asked_iso: str | None = None,
) -> dict[str, Any]:
    """Detail response with each spec's text in force on `offense_iso` (`_get_historic_article`, among the articles
    alive then), and why the missing ones are missing.

    ⚠ `offense_iso` is filled in for current lookups too (today, or a pending law's first date), so it does not
    tell whether a date was asked. `asked_iso` (the date the model gave) does, and only it becomes the response's
    `offense_date`."""
    law_id = meta['law_id']
    # Repealed by the date, or in force then and repealed today (the renderer words the two).
    gone = _is_repealed_at(conn, law_id, _today_iso())
    repealed = _is_repealed_at(conn, law_id, offense_iso) or gone
    # Return only articles alive at that time (`_articles_in_force`).
    in_force = _articles_in_force(conn, law_id, offense_iso)

    articles: list[dict] = []
    branch_lists: list[dict] = []   # branch articles listed rather than given in full (over BRANCHES_IN_FULL)
    missing: list[str] = []
    missing_specs: list[tuple[int, int | None]] = []   # for the nearby-article hint (`missing_neighbors`)
    # Numbers that existed only in old versions → effective date of the last version that had them. Unlike "exists
    # nowhere", there is a distinct next action (find today's number via the outline · the wording then via
    # offense_date), so it's reported separately.
    vanished: dict[str, str] = {}
    for num, branch in specs:
        # branch=None (main article + all branches) → point-in-time lookup of *every branch* of
        # article_no_num=num under law_id (last change per branch)
        if branch is None:
            # every possible branch of the same num — looked up at once
            branches_rows = conn.execute(
                """SELECT DISTINCT a.article_branch FROM st_articles a
                   JOIN st_statutes s ON s.id=a.statute_id
                   WHERE s.law_id=? AND a.article_no_num=?
                     AND s.effective_date <= ?""",
                (law_id, num, offense_iso),
            ).fetchall()
            branches = sorted({r['article_branch'] for r in branches_rows})
            alive = [br for br in branches if (num, br) in in_force]
            # A number alive then may still have no article in force (its own date is later: `_IN_FORCE_FROM`).
            found = [r for r in (_get_historic_article(conn, law_id, num, br, offense_iso) for br in alive) if r]
            if not found:
                missing.append(f"{num}")
                missing_specs.append((num, None))
                old_effs = [r['effective_date'] for r in (
                    _get_historic_article(conn, law_id, num, br, offense_iso)
                    for br in branches) if r]
                if old_effs:
                    vanished[f"{num}"] = max(old_effs)
                continue
            twigs = [r for r in found if r['article_branch']]
            if len(twigs) > BRANCHES_IN_FULL:
                # Many branch articles: the main article's text, and the branches by number, title and opening words.
                found = [r for r in found if not r['article_branch']]
                branch_lists.append({"of": num, "articles": [
                    {"no": r['article_no'], "title": r['title'], "head": _text_head(r['article_text'], r['title'])}
                    for r in twigs]})
            articles += [_format_historic_article_row(r) for r in found]
        else:
            r = _get_historic_article(conn, law_id, num, branch, offense_iso)
            if r and (num, branch) not in in_force:
                vanished[f"{num}-{branch}"] = r['effective_date']
                r = None
            if r:
                articles.append(_format_historic_article_row(r))
            else:
                missing.append(f"{num}-{branch}")
                missing_specs.append((num, branch))

    # Missing numbers that a version after the asked date and in force today has were added later (2012 VAT Act
    # Art. 39: the 2013 full amendment renumbered) — not "nowhere", which sends the model away from today's article.
    added: dict[str, str] = {}
    if asked_iso:
        for num, branch in missing_specs:
            key = f"{num}" if branch is None else f"{num}-{branch}"
            if key not in vanished:
                eff = _added_article(conn, law_id, offense_iso, num, branch)
                if eff:
                    added[key] = eff

    # Missing numbers present in a not-yet-in-force version = articles not yet in force → their effective date.
    # Calling them "old numbers" would make the model answer "it was deleted" (Income Tax Act Art. 164-4, in force
    # from 2027-01-01). A number already added by today is not pending even if a pending version amends it
    # (행정규제기본법 §8-2).
    pending: dict[str, str] = {}
    for num, branch in missing_specs:
        key = f"{num}" if branch is None else f"{num}-{branch}"
        if key in added:
            continue
        hit = _pending_article(conn, law_id, offense_iso, num, branch)
        if hit is None:
            continue
        pending[key] = hit["effective_date"]
        # If the same number in an old version is the same article as the pending one (same title), it is not an old
        # number — an amendment with a deferred effective date was printed first in the promulgated version (Income Tax
        # Act Art. 164-4: printed in the 2025-01-01 row but in force 2027-01-01). If the titles differ, it is a new
        # article that only shares the number, so the old-number hint is kept as well.
        last = _get_historic_article(conn, law_id, num, hit["branch"], offense_iso)
        if key in vanished and last and _same_title(last["title"], hit["title"]):
            del vanished[key]

    out = {
        "mode": "detail",
        "statute": {
            "id": meta["id"],
            "law_id": meta["law_id"],
            "name": meta["name"],
            "kind": meta["kind"],
        },
        "articles": articles,
        "missing": missing,
    }
    if branch_lists:
        out["branch_lists"] = branch_lists
    if asked_iso:
        out["offense_date"] = asked_iso
    if vanished:
        out["vanished"] = vanished
    if pending:
        out["pending"] = pending
    if added:
        out["added"] = added
    if missing:
        nbrs = _neighbor_articles(conn, law_id, offense_iso, missing_specs, in_force)
        if nbrs:
            out["missing_neighbors"] = nbrs
    if repealed:
        out["repealed"] = repealed
    # Not the current version: a marker of the current one. A dated lookup of the current version still carries past
    # wording, so its marker is the row itself — the renderer adds it to the point-in-time line as "현행은 … 시행본".
    cur_ref = None if gone else _current_version_ref(conn, law_id, meta['id'], meta.get('name'))
    served = cur_ref is None
    if served and asked_iso and articles and not gone:
        cur_ref = {"id": meta["id"], "name": meta["name"],
                   "effective_date": str(meta.get("effective_date") or ""), "renamed": False}
    if cur_ref:
        out["current_version"] = cur_ref
    # The header is the current version (the latest whole text), yet the walk can bring wording from a later
    # amendment that holds only the articles it changed (개인정보 보호법: whole text 2025-10-02, Art. 58 from
    # 2026-09-11). That is the wording's date, not "not current".
    if served:
        cur_eff = str(meta.get("effective_date") or "")
        newer = [a for a in articles
                 if cur_eff and str(a.get("eff_date") or "") > cur_eff]
        if newer:
            out["text_basis"] = {
                "effective_date": max(str(a.get("eff_date") or "") for a in newer),
                "articles": [_article_label(a["no_num"], a["branch"]) for a in newer[:3]],
                "count": len(newer),
            }
    return out


BRANCHES_IN_FULL = 3          # branch articles given in full with their main article ('76' → 76의2 …); more are listed
BRANCH_HEAD_CHARS = 40        # opening characters of each listed branch article
_ARTICLE_LABEL_RE = re.compile(r"^\s*제\s*\d+\s*조(?:\s*의\s*\d+)?")


def _text_head(text: str | None, title: str | None) -> str:
    """The article's opening words after its own '제N조의M(제목)' heading, on one line, cut at BRANCH_HEAD_CHARS."""
    body = _ARTICLE_LABEL_RE.sub("", text or "", count=1).lstrip()
    if title and body.startswith(f"({title})"):
        body = body[len(title) + 2:]
    body = " ".join(body.split())
    return body if len(body) <= BRANCH_HEAD_CHARS else body[:BRANCH_HEAD_CHARS].rstrip() + "…"


def _format_historic_article_row(r: dict) -> dict:
    return {
        "no": r["article_no"],
        "no_num": r["article_no_num"],
        "branch": r["article_branch"],
        "title": r["title"],
        "text": r["article_text"],
        "eff_date": r["effective_date"],
    }


def _detail_notice(
    conn: sqlite3.Connection,
    nid: int,
    specs: list[tuple[int, int | None]] | None,
) -> dict[str, Any] | None:
    """Notice article_no = num*1000 + branch ('제6조의2' = 6002). Converts the spec (num, branch) to that encoding for
    matching — passing back the natural article numbers the outline shows ('1','6의2') as is finds them."""
    meta = _notice_meta(conn, nid)
    if meta is None:
        return None
    cat = meta["category"]

    if cat == "article_form":
        # With a branch given, exact (num*1000+br); without (main article), the whole thousand block (main + all
        # branches).
        conds, params = [], []
        for num, br in (specs or []):
            if br is None:
                conds.append("(article_no >= ? AND article_no < ?)")
                params += [num * 1000, (num + 1) * 1000]
            else:
                conds.append("article_no = ?")
                params.append(num * 1000 + br)
        where = " OR ".join(conds) if conds else "0"
        rows = conn.execute(
            f"""
            SELECT article_no, article_no_str, article_text
            FROM st_notice_articles
            WHERE notice_id=? AND ({where})
            ORDER BY article_seq
            """,
            [nid, *params],
        ).fetchall()
        articles = [{"no": r["article_no_str"], "title": _notice_article_title(r["article_text"]),
                     "text": r["article_text"]} for r in rows]
        found = {r["article_no"] for r in rows}
        missing = []
        for num, br in (specs or []):
            if br is None:
                if not any(num * 1000 <= a < (num + 1) * 1000 for a in found):
                    missing.append(num)
            elif (num * 1000 + br) not in found:
                missing.append(f"{num}-{br}")
        return {
            "mode": "detail",
            "statute": {"id": nid, "name": meta["name"], "kind": meta["kind"], "category": cat},
            "articles": articles,
            "missing": missing,
        }

    if cat == "other":
        rows = conn.execute(
            "SELECT article_text FROM st_notice_articles WHERE notice_id=? ORDER BY article_seq",
            (nid,),
        ).fetchall()
        return {   # whole; an overlong response is cut at the end and says so (`_bounded_response_md`)
            "mode": "detail",
            "statute": {"id": nid, "name": meta["name"], "kind": meta["kind"], "category": cat},
            "body": "\n\n".join(r["article_text"] for r in rows),
        }

    return {
        "mode": "detail",
        "statute": {"id": nid, "name": meta["name"], "kind": meta["kind"], "category": cat},
        "content_format": cat,
        "articles": [],
        "note": "본문이 이미지로만 제공되거나 비어있음" if cat == "image_only" else "응답 본문 없음",
    }


# ---------- markdown serialization ----------

def _article_label(num: int, branch: int) -> str:
    """Article notation as human-readable Korean — '제58조', '제58조의2'."""
    return f"제{num}조" + (f"의{branch}" if branch else "")


# Nearby articles named with a missing one, so the model does not probe the numbers around it one by one.
_NEIGHBOR_LIMIT = 3


def _neighbor_articles(
    conn: sqlite3.Connection, law_id: str, as_of_iso: str, specs: list[tuple[int, int | None]],
    in_force: set[tuple[int, int]],
) -> list[str]:
    """Missing article specs → labels of the articles alive then (`in_force`, so not a number only old versions had)
    nearest to where they would be, at most `_NEIGHBOR_LIMIT`."""
    rows = conn.execute(
        f"""SELECT DISTINCT a.article_no_num, a.article_branch
           FROM st_articles a JOIN st_statutes s ON s.id=a.statute_id
           WHERE s.law_id=? AND {_IN_FORCE_FROM} <= ? AND a.article_no_num IS NOT NULL""",
        (law_id, as_of_iso),
    ).fetchall()
    keys = sorted({tuple(r) for r in rows} & in_force)
    picked: set[tuple[int, int]] = set()
    for num, branch in specs:
        key = (num, branch or 0)
        # Collect nearest first, left and right of the insertion point (where this number should have been).
        i = 0
        while i < len(keys) and keys[i] < key:
            i += 1
        for j in (i - 1, i, i - 2, i + 1, i - 3, i + 2):
            if 0 <= j < len(keys) and len(picked) < _NEIGHBOR_LIMIT:
                picked.add(keys[j])
        if len(picked) >= _NEIGHBOR_LIMIT:
            break
    return [_article_label(n, b) for n, b in sorted(picked)]


def _statute_web_url(d: dict[str, Any]) -> str | None:
    """Statute/notice dict (the statute of a detail or a search match) → Lawful statute page url.
    Statute = /statutes/{law_id}, notice = /statutes/admrul-{st_notices.id}. None if it can't be resolved.
    Same shape as the judgment tools' `url` field — the model presents it as a markdown link when citing."""
    if not d:
        return None
    if d.get("treaty_id"):
        return f"{LAWFUL_WEB_BASE}/statutes/{treaties.TREATY_URL_PREFIX}{d['treaty_id']}"
    law_id = d.get("law_id")
    if law_id:
        return f"{LAWFUL_WEB_BASE}/statutes/{law_id}"
    # Notice: no law_id + has category/notice_id. id (detail) or statute_id (match) = st_notices.id
    if d.get("category") or d.get("notice_id"):
        nid = d.get("id")
        if not isinstance(nid, int):
            nid = d.get("statute_id")
        if isinstance(nid, int):
            return f"{LAWFUL_WEB_BASE}/statutes/{NOTICE_URL_PREFIX}{nid}"
    return None


def _statute_article_url(stt: dict[str, Any] | None, art: dict[str, Any]) -> str | None:
    """Article-level statute page url /statutes/{law_id}/{jo} (jo='750'·'839의2'). None for notices and treaties (no
    law_id) or a single body at 0 — then only the statute-level url is shown."""
    law_id = (stt or {}).get("law_id")
    if not law_id or not art["no_num"]:
        return None
    return f"{LAWFUL_WEB_BASE}/statutes/{law_id}/{_jo(art['no_num'], art['branch'])}"


def _version_lines(resp: dict[str, Any], mode: str | None) -> list[str]:
    """The lines on which wording a detail or outline response carries: a former name asked, the offense date,
    no version at that date, not the current law, the wording's basis, repeal, not yet in force, or the current
    version — shared by `statute_read` and the articles `sentencing_analysis` embeds."""
    lines: list[str] = []
    # "폐지" (repealed) is said only of a repeal: a renamed law still exists.
    rep = resp.get("repealed")
    what = "조문 목록" if mode == "outline" else "문언"
    pending = (resp.get("text_date") or "") > _today_iso()   # the text of a law not yet in force
    asked = resp.get("asked_name")
    if asked:
        eff = _fmt_eff_iso(asked.get("effective_date"))
        lines.append(
            f"- 조회한 「{asked.get('name','')}」은 이 법의 옛 이름입니다"
            + (f"({eff} 시행본)" if eff else "")
            + (" — 위 이름이 마지막 이름입니다." if rep else " — 위 이름이 현행명이고 아래 본문도 현행입니다.")
        )
    cv = resp.get("current_version")
    tail = ""
    if cv:
        ceff = _fmt_eff_iso(cv.get("effective_date"))
        tail = (f"「{cv.get('name','')}」({ceff} 시행)" if cv.get("renamed")
                else f"{ceff} 시행본")
    od = resp.get("offense_date")
    if od:
        lines.append(f"- 시점 조회: {_fmt_eff_iso(od)} 기준 문언입니다"
                     + (f" — 현행은 {tail}." if tail else "."))
    elif resp.get("no_version_at"):
        nv = resp["no_version_at"]
        lines.append(f"- 시점 조회 불가: 이 코퍼스에는 {_fmt_eff_iso(nv['date'])} 시점에 시행된 판본이 없습니다"
                     f"(가장 이른 판본 {_fmt_eff_iso(nv['earliest'])} 시행)."
                     + ("" if rep or pending else " 아래는 현행 문언이며 그 시점의 문언과 다를 수 있습니다."))
    elif cv:
        lines.append(f"- 현행 아님 — 이 법의 현행은 {tail}, "
                     f"statute_id={cv.get('id')} 입니다.")
    tb = resp.get("text_basis")
    if tb:
        eff = _fmt_eff_iso(tb.get("effective_date"))
        labels = tb.get("articles") or []
        extra = ""
        if labels:
            more = int(tb.get("count") or len(labels)) - len(labels)
            extra = " (" + " · ".join(labels) + (f" 외 {more}건" if more > 0 else "") + ")"
        lines.append(f"- 문언 기준: {eff} 시행본{extra}")
    if rep:
        # A repealed law has no current text: say whether it was in force at the asked date.
        if not rep["effective_date"]:   # lost effect on a day the corpus does not know (`_is_repealed_at`)
            lines.append(f"- 폐지: 이 법령은 효력을 잃었고 그 날짜는 코퍼스에 없습니다 — 지금은 효력이 없고, 아래는 마지막 "
                         f"{what}입니다." + (f" 행위시점({_fmt_eff_iso(od)})에 시행 중이었는지는 확인되지 않습니다." if od else ""))
        else:
            if not od:
                when = f"지금은 효력이 없고, 아래는 그 전 마지막 {what}입니다."
            elif rep["effective_date"] <= od:
                when = f"행위시점({_fmt_eff_iso(od)})에는 이미 효력이 없었고, 아래는 그 전 마지막 {what}입니다."
            else:
                when = f"행위시점({_fmt_eff_iso(od)})에는 시행 중이었고, 지금은 효력이 없습니다."
            lines.append(f"- 폐지: 이 법령은 {_fmt_eff_iso(rep['effective_date'])} {rep['change_kind']} — {when}")
    if pending:
        lines.append(f"- 판본: 아직 시행 전({_fmt_eff_iso(resp['text_date'])} 시행 예정) — "
                     f"아래는 공포된 {what}입니다.")
    elif resp.get("text_date"):
        lines.append(f"- 판본: 현행 {what}({_fmt_eff_iso(resp['text_date'])} 시행본까지 반영) — "
                     "행위 시점의 문언은 offense_date 로 조회하세요.")
    return lines


def _missing_lines(resp: dict[str, Any]) -> list[str]:
    """The articles asked for that the response lacks, and why — shared like `_version_lines`."""
    lines: list[str] = []
    missing = resp.get("missing")
    if missing:
        # Why each number is missing and which are there, so the model does not retry other notations.
        lines.append(f"## 없는 조문: {', '.join(str(x) for x in missing)}")
        vanished = resp.get("vanished") or {}
        pending = resp.get("pending") or {}
        od = resp.get("offense_date")
        added = resp.get("added") or {}
        never = [str(x) for x in missing if str(x) not in vanished and str(x) not in pending
                 and str(x) not in added]
        if never:
            lines.append("- 현행·연혁본 어디에도 이 번호의 조문이 없습니다." if len(never) == len(missing)
                         else f"- {', '.join(never)}: 현행·연혁본 어디에도 없는 번호입니다.")
        if vanished:
            old = ", ".join(f"{k}({_fmt_eff_iso(v)}판까지)" for k, v in vanished.items())
            if od:
                # A point-in-time lookup already gave offense_date — send it to that time's outline rather than
                # today's, and for that number's wording, to the date of the version in which the number existed.
                lines.append(
                    f"- {old}: {_fmt_eff_iso(od)} 시점에는 없는 옛 번호입니다 — 개정으로 조문이 옮겨졌거나 "
                    "사라졌습니다. 그 시점의 번호는 같은 offense_date 로 조문 개요(번호·제목)를 받아 확인하고, "
                    "그 번호가 있던 때의 문언이 필요하면 괄호 속 날짜를 offense_date 로 주세요.")
            else:
                lines.append(
                    f"- {old}: 현행에는 없는 옛 번호입니다 — 개정으로 조문이 옮겨졌거나 사라졌습니다. "
                    "찾는 내용의 지금 번호는 statute_read 에 statute_id 만 넣어 조문 개요(번호·제목)를 받아 확인하고, "
                    "옛 판례가 인용한 그때 문언이 필요하면 offense_date 를 주세요.")
        if added:
            new = ", ".join(f"{k}({_fmt_eff_iso(v)}판부터)" for k, v in added.items())
            lines.append(
                f"- {new}: {_fmt_eff_iso(od)} 시점에는 아직 없던 번호입니다 — 그 뒤 개정으로 생겼거나 번호가 "
                "바뀌었습니다. 그 시점의 번호는 같은 offense_date 로 조문 개요(번호·제목)를 받아 확인하고, "
                "지금 문언이 필요하면 offense_date 없이 조회하세요.")
        if pending:
            when = f"{_fmt_eff_iso(od)} 시점에는" if od else "아직"
            # A 9999 effective date is the source's placeholder for "not set", not a date to look up.
            dated = {k: v for k, v in pending.items() if not str(v).startswith("9999")}
            if dated:
                soon = ", ".join(f"{k}({_fmt_eff_iso(v)} 시행 예정)" for k, v in dated.items())
                lines.append(
                    f"- {soon}: {when} 시행 전인 조문입니다 — 시행될 문언은 괄호 속 시행일을 "
                    "offense_date 로 주면 볼 수 있습니다.")
            undated = [k for k in pending if k not in dated]
            if undated:
                lines.append(
                    f"- {', '.join(undated)}: 시행일이 정해지지 않은 개정본에만 있는 번호입니다 — "
                    f"{when} 시행 중인 조문이 아닙니다.")
        nbrs = resp.get("missing_neighbors")
        if nbrs:
            lines.append(f"- 근처 조문: {' · '.join(nbrs)}")
    return lines


def _jo(num: int, branch: int) -> str:
    return f"{num}의{branch}" if branch else f"{num}"


def article_sections(conn: sqlite3.Connection, statute_id: int, wanted: list[tuple[int, int]],
                     offense_date: str | None) -> list[tuple[str, str]]:
    """Articles of one statute to embed in another tool's response, as (heading, body): the statute's name and the
    wording version once, each asked article with its link and text, and the asked articles the version lacks.
    Branch articles the lookup adds (asked 347, it also gives 347의2) are left out."""
    resp = _statute_lookup_impl(conn, None, _parse_statute_ref(statute_id), [_jo(n, b) for n, b in wanted],
                                SEARCH_LIMIT, offense_date=offense_date)
    if resp.get("status") != "ok" or resp.get("mode") != "detail":
        return []
    stt = resp.get("statute") or {}
    head = _version_lines(resp, "detail") + [f"- text_kind: {resp.get('text_kind') or '공식 조문 원문'}"]
    out = [(f"## 근거 조문 — {stt.get('name', '')}", "\n".join(head))]
    for a in resp.get("articles", []):
        if (a["no_num"], a["branch"]) not in wanted:
            continue
        when = f" ({_fmt_eff_iso(a['eff_date'])} 시행판)" if resp.get("offense_date") and a.get("eff_date") else ""
        url = _statute_article_url(stt, a)
        out.append((f"### §{_jo(a['no_num'], a['branch'])}{when}" + (f" — {url}" if url else ""), a["text"]))
    missing = _missing_lines(resp)
    if missing:
        out.append((missing[0], "\n".join(missing[1:])))
    return out


def _format_response_md(resp: dict[str, Any]) -> str:
    """statute_search / statute_read response dict → markdown-KV string."""
    status = resp.get("status", "ok")
    mode = resp.get("mode")
    lines: list[str] = [f"## status: {status}"]
    if mode:
        lines.append(f"## mode: {mode}")

    if "message" in resp:
        lines.append(f"- message: {resp['message']}")

    inp = resp.get("input")
    if inp:
        for k, v in inp.items():
            if v is not None:
                lines.append(f"- {k}: {v}")

    if mode == "list":
        matches = resp.get("matches", [])
        lines.append(f"## matches ({len(matches)})")
        for m in matches:
            # Administrative rules are emitted with the 'admrul-' prefixed identifier — the model must pass it back
            # to statute_id verbatim to get the same document (a bare integer is in the statute id space, and
            # values overlap).
            mid = _statute_ref(m) or m.get("statute_id") or m.get("id")
            kind = m.get("kind") or ""
            head = f"- {mid} {m.get('name','')}" + (f" ({kind})" if kind else "")
            eff = _fmt_eff_iso(m.get("effective_date"))
            if eff:
                head += f" · {eff} 시행"
            lines.append(head)
            # Whether the line is the law's current version, so an old name is not read as current.
            cur = m.get("current")
            if m.get("is_repealed"):   # a repealed law has no current version to point to
                lines.append("  폐지된 법령 — 현행 후속본이 없습니다")
            elif cur:
                ceff = _fmt_eff_iso(cur.get("effective_date"))
                tail = (f"「{cur.get('name','')}」({ceff} 시행)" if cur.get("renamed")
                        else f"{ceff} 시행본")
                lines.append(f"  현행 아님 — 이 법의 현행은 {tail}, "
                             f"statute_id={cur.get('id')}")
            if m.get("matched_article"):
                lines.append(f"  맞은 조문: {m['matched_article']}")
            url = _statute_web_url(m)
            if url:
                lines.append(f"  url: {url}")
        if resp.get("note"):
            lines.append(f"## note: {resp['note']}")
        return "\n".join(lines)

    stt = resp.get("statute")
    if stt:
        sid = _statute_ref(stt) or stt.get("id")
        head = f"## statute: {stt.get('name','')}"
        if sid:
            head += f" (id={sid})"
        meta = []
        if stt.get("kind"):
            meta.append(stt["kind"])
        if stt.get("agency"):
            meta.append(stt["agency"])
        if meta:
            head += f" — {', '.join(meta)}"
        lines.append(head)
        url = _statute_web_url(stt)
        if url:
            lines.append(f"- url: {url}")
        lines += _version_lines(resp, mode)

    if resp.get("content_format"):
        lines.append(f"## content_format: {resp['content_format']}")
    if resp.get("note"):
        lines.append(f"## note: {resp['note']}")

    if mode == "outline" and resp.get("toc"):
        name = (resp.get("statute") or {}).get("name") or ""
        lines.append(f"## 목차 — 조문 {resp['article_count']}개는 목록이 길어 편·장·절과 조문 범위만 보입니다. 본문은 "
                     f"articles로(최대 8개), 조문 찾기는 statute_search(query='{name} 낱말')로 부르세요.")
        for t in resp["toc"]:
            span = _article_label(*t["first"]) + ("" if t["first"] == t["last"] else f"~{_article_label(*t['last'])}")
            lines.append(f"{'  ' * (t['level'] - 1)}- " + (f"{t['title']} ({span})" if t["title"] else span))
    elif mode == "outline":
        arts = resp.get("articles", [])
        lines.append(f"## articles ({len(arts)})")
        for a in arts:
            lines.append(f"- {a['no']}: {a.get('title','')}")
    elif mode == "detail":
        arts = resp.get("articles", [])
        if arts:
            lines.append("## articles")
            for a in arts:
                # title may be None (statutes without titles — the Constitution, single-provision regulations).
                # a.get(...,'') prints 'None' as is when the key exists with a None value, so normalize with `or ''`.
                lines.append(f"### {a['no']} {a.get('title') or ''}")
                art_url = _statute_article_url(stt, a)
                if art_url:
                    lines.append(f"- url: {art_url}")
                if resp.get("offense_date") and a.get("eff_date"):
                    # Each article's version date, in point-in-time lookups only (the articles can differ).
                    lines.append(f"- 시행: {_fmt_eff_iso(a['eff_date'])}판")
                if a.get("text"):
                    lines.append(f"- text_kind: {resp.get('text_kind') or '공식 조문 원문'}")
                    lines.append(a["text"])
        for bl in resp.get("branch_lists") or []:
            twigs = bl["articles"]
            lines.append(f"## 제{bl['of']}조의 가지 조문 {len(twigs)}개 — 많아서 제목과 본문 앞부분만 보입니다. "
                         f"본문은 articles로 부르세요(예: [\"{twigs[0]['no']}\"])")
            lines += [f"- {a['no']}: " + " — ".join(filter(None, [a.get("title"), a["head"]])) for a in twigs]
        if resp.get("body"):  # notice of category other
            lines.append("## body")
            lines.append(f"- text_kind: {resp.get('body_kind') or '공식 고시 원문'}")
            lines.append(resp["body"])
        lines += _missing_lines(resp)

    if resp.get("addenda_list"):
        shown, total = resp["addenda_list"], resp.get("addenda_total") or len(resp["addenda_list"])
        lines.append(f"## 부칙 목록 ({total}건" + (f" 중 최근 {len(shown)}건" if total > len(shown) else "")
                     + ") — 조문은 articles=['부칙 3'] 처럼 번호로 받습니다")
        for g in shown:
            heads = " · ".join(g["articles"])
            lines.append(f"- {_fmt_eff_iso(g['date'])} 공포 제{g['no']}호: "
                         + (heads[:300].rstrip() + "…" if len(heads) > 300 else heads))
    if resp.get("addenda"):
        more = resp.get("addenda_more") or 0
        until = resp.get("addenda_until")
        lines.append("## 부칙 (" + (f"{_fmt_eff_iso(until)}까지 공포된 것, " if until else "") + "공포일이 늦은 것부터"
                     + (f", 이 밖에 {more}건" if more else "") + ")")
        for a in resp["addenda"]:
            lines.append(f"### 부칙 <{_fmt_eff_iso(a['date'])} 공포 제{a['no']}호> {a['label']}")
            lines.append("- text_kind: 공식 조문 원문")
            lines.append(a["text"])
    if resp.get("addenda_note"):
        lines.append(f"## 부칙: {resp['addenda_note']}")

    if resp.get("attachment"):
        lines.append(f"## 첨부 원문: {resp['attachment']} — 이 문서의 본문은 첨부파일에만 있을 수 있습니다(코퍼스에 없음).")
    # Tax treaties: every document with the same partner, by date — the source has no consolidated text.
    if resp.get("related"):
        lines.append(f"## {resp['related_title']}")
        for d in resp["related"]:
            lines.append(f"- {d['ref']} · {d['date']} 발효 · {d['role']} · {d['name']}"
                         + (" ← 이 문서" if d.get("self") else ""))

    # Annexed-table index — the body isn't in the corpus; the file address stands in for it. Attached in the same
    # place with the same shape regardless of mode (outline = all · detail = what that article points to).
    byeolpyo = resp.get("byeolpyo")
    if byeolpyo:
        total = resp.get("byeolpyo_total") or len(byeolpyo)
        head = f"## 별표 ({total})"
        if total > len(byeolpyo):
            head += f" — 앞 {len(byeolpyo)}건만"
        lines.append(head + " · 본문은 코퍼스에 없습니다.")
        for b in byeolpyo:
            lines.append(f"- {b['label']} {b.get('title') or ''}".rstrip())
            lines.append(f"  file: {b['file']}")

    return "\n".join(lines)


def _bounded_response_md(resp: dict[str, Any]) -> str:
    """Response dict → markdown-KV. Over `RESPONSE_MAX_CHARS` the tail is cut at a line end, and a closing `message`
    says what was left out and how to get it."""
    md = _format_response_md(resp)
    if len(md) <= RESPONSE_MAX_CHARS:
        return md
    note = (f"\n\n- message: 응답이 {RESPONSE_MAX_CHARS:,}자를 넘어 **꼬리를 생략**했습니다"
            f"(전체 {len(md):,}자). 빠진 조문·outline 은 articles=[...] 로 나눠 재호출하세요.")
    keep = RESPONSE_MAX_CHARS - len(note)
    cut = md.rfind("\n", 0, max(keep, 0))       # don't cut in the middle of a line
    return md[:cut if cut > 0 else max(keep, 0)].rstrip() + note


# ---------- public tool ----------


def statute_search(
    ctx: RunContext[HarnessDeps],
    query: str | None = None,
    offense_date: str | None = None,
) -> str:
    """법령 찾기 — 헌법·법률·대통령령·부령·행정규칙(고시·훈령·예규)·조세조약을 찾습니다. 본문은 statute_read로 봅니다.

    - 법령명이나 키워드를 넣으면 맞는 법령(최대 10개)과 낱말이 든 조문이 옵니다. 법령명과 함께 쓰면 그 법령의
      조문을 찾기 쉽습니다(예: '형사소송법 공소취소'; 낱말은 조문에 나올 세 글자 이상 문구로).
    - 법령명과 조문 번호를 함께 쓰면 그 조문 본문이 바로 옵니다. 여러 조문은 한 번에 씁니다(예: '민법 제750조 제751조').
    - 조세조약은 '상대국 이름 + 조세조약'으로 찾습니다.

    Args:
      query: 법령명·키워드·조문 번호.
      offense_date: 조문 본문의 판본 기준 날짜(예: '2013.7.30'). 없으면 오늘 시행 중인 판본.
    """
    query = coerce_str(query)
    if unreadable_date(offense_date):
        return _bounded_response_md(_bad_date_response(offense_date))
    if not query:
        return _bounded_response_md({
            "status": "missing_input",
            "message": "query(법령명·키워드)가 필요합니다. 받은 statute_id 의 본문은 statute_read 로 보세요.",
        })
    return _lookup(query, None, None, offense_date)


def statute_read(
    ctx: RunContext[HarnessDeps],
    statute_id: int | str | None = None,
    articles: list[str | int] | str | int | None = None,
    offense_date: str | None = None,
) -> str:
    """법령 본문 — statute_search가 준 statute_id로 조문 본문을 봅니다.

    - articles를 넣으면 그 조문 본문이 옵니다. 없으면 조문 제목 목록이 옵니다(민법·상법처럼 아주 큰 법령은
      편·장·절 목차와 조문 범위).

    Args:
      statute_id: statute_search 결과의 id를 그대로(예: 584, 'admrul-18060', 'trty-1307'). 행정규칙·조약은
        접두사까지 넣습니다.
      articles: 조문 번호 목록, 최대 8개(예: ["347"], "제347조의2", "347-2"). "347"은 가지 조문(347조의2 등)까지
        함께 옵니다. 부칙은 '부칙' 또는 '부칙 3'. 별표는 '별표 5'처럼 따로 넣으면 본문 대신 제목과 파일 주소가
        옵니다. 서식·별지는 받지 않습니다.
      offense_date: 판본 기준 날짜(예: '2013.7.30'). 주면 그날 시행 중이던 판본, 없으면 오늘 시행 중인 판본.
    """
    articles = _split_joined_articles(coerce_list(articles))
    ref = _parse_statute_ref(statute_id)
    if unreadable_date(offense_date):
        return _bounded_response_md(_bad_date_response(offense_date))
    if ref is None:
        # An unreadable statute_id is told apart from none, or the model resends the same misspelling.
        if statute_id is not None:
            return _bounded_response_md({
                "status": "bad_statute_id",
                "input": {"statute_id": statute_id},
                "message": (
                    "statute_id 를 읽지 못했습니다 — 법령은 정수(예: 584), "
                    "행정규칙은 'admrul-18060' 형태입니다. statute_search 결과가 준 식별자를 "
                    "글자 그대로 넘기거나, statute_search 로 다시 찾으세요."
                ),
            })
        return _bounded_response_md({
            "status": "missing_input",
            "message": "statute_id(statute_search 가 준 식별자)가 필요합니다. 법령을 모르면 statute_search 로 찾으세요.",
        })
    return _lookup(None, ref, articles, offense_date)


def _bad_date_response(offense_date: Any) -> dict[str, Any]:
    return {"status": "bad_offense_date", "message": BAD_DATE_MESSAGE, "input": {"offense_date": offense_date}}


def _lookup(query: str | None, ref: tuple | None, articles: list[str | int] | None, offense_date: str | None) -> str:
    """A search (query) or a read (ref, articles) through `_statute_lookup_impl`, with the notes on what was left
    out: articles over ARTICLES_MAX and annexed tables asked among the articles."""
    # Over the cap, the first ARTICLES_MAX are looked up and the rest named, rather than the call refused.
    dropped_articles: list = []
    if articles is not None and len(articles) > ARTICLES_MAX:
        dropped_articles = list(articles[ARTICLES_MAX:])
        articles = list(articles[:ARTICLES_MAX])

    conn = open_db()
    try:
        resp = _statute_lookup_impl(conn, query, ref, articles, SEARCH_LIMIT, offense_date=offense_date)
        if dropped_articles:
            note = _over_cap_note("articles", dropped_articles)
            prev = resp.get("message")
            resp["message"] = note if not prev else f"{note} · {prev}"
        # Articles read and an annexed table dropped (`["13","별표5"]`): name what was dropped, or the model may take the
        # table as answered. (When nothing could be read, `_bad_articles_response` says it.)
        unsupported = _unsupported_units(articles) if resp.get("status") == "ok" else []
        if unsupported:
            note = f"{', '.join(unsupported)} 는 제외했습니다. {_UNSUPPORTED_UNIT_HINT}"
            if _requested_byeolpyo_numbers(unsupported):
                note += f" {_BYEOLPYO_ALONE_HINT}"
            prev = resp.get("message")
            resp["message"] = note if not prev else f"{prev} · {note}"
        return _bounded_response_md(resp)
    finally:
        conn.close()


def _unsupported_units(articles: list[str | int] | None) -> list[str]:
    """Non-article structural-unit tokens among articles (annexed tables, forms …) — echoed back as is in the hint.
    Addenda ('부칙 …') are looked up (`_addenda`)."""
    return [str(a) for a in (articles or [])
            if _NON_ARTICLE_UNIT_RE.match(str(a).lstrip()) and not _ADDENDA_RE.match(str(a))]


# ── Annexed-table index (`st_byeolpyo`) ────────────────────────────────────────────────
#
# The index holds where each annexed table is, not its text (the source provides annexed tables only as hwp/pdf
# files): which tables exist and their file addresses, for citation.
BYEOLPYO_MAX = 25


def _byeolpyo_owner(conn: sqlite3.Connection, is_notice: bool, sid: int) -> tuple[str, str] | None:
    """(scope, owner_id) — owner key of the annexed-table index. law_id for statutes, notice_id for administrative
    rules."""
    if is_notice:
        r = conn.execute("SELECT notice_id FROM st_notices WHERE id=?", (sid,)).fetchone()
        return ("admrul", r["notice_id"]) if r else None
    r = conn.execute("SELECT law_id FROM st_statutes WHERE id=?", (sid,)).fetchone()
    return ("law", r["law_id"]) if r else None


def _byeolpyo_fmt(no: Any, branch: Any) -> str:
    return f"별표 {no}" + (f"의{branch}" if branch else "")


def _byeolpyo_rows(
    conn: sqlite3.Connection,
    owner: tuple[str, str] | None,
    *,
    specs: list[tuple[int, int | None]] | None = None,
    numbers: list[tuple[int, int | None]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Annexed-table index rows. `specs` = only those the given articles point to · `numbers` = only those
    annexed-table numbers."""
    if not owner:
        return [], 0
    scope, owner_id = owner
    where = ["scope=?", "owner_id=?", "kind='별표'"]
    params: list[Any] = [scope, owner_id]
    if specs:
        ors = []
        for no, branch in specs:
            if branch is None:      # '13' = main article + all branches (same rule as article lookup)
                ors.append("article_no_num=?")
                params.append(no)
            else:
                ors.append("(article_no_num=? AND article_branch=?)")
                params += [no, branch]
        where.append("(" + " OR ".join(ors) + ")")
    if numbers:
        ors = []
        for no, branch in numbers:
            ors.append("(byl_no_num=? AND byl_branch=?)")
            params += [no, branch or 0]
        where.append("(" + " OR ".join(ors) + ")")
    sql = ("SELECT byl_no_num, byl_branch, title, file_url FROM st_byeolpyo "
           f"WHERE {' AND '.join(where)} ORDER BY byl_no_num, byl_branch LIMIT ?")
    rows = conn.execute(sql, [*params, BYEOLPYO_MAX]).fetchall()
    total = conn.execute(f"SELECT COUNT(*) FROM st_byeolpyo WHERE {' AND '.join(where)}", params).fetchone()[0]
    out = [{
        "label": _byeolpyo_fmt(r["byl_no_num"], r["byl_branch"]),
        "title": r["title"] or "",
        "file": r["file_url"],   # every annexed table has its hwp address
    } for r in rows]
    return out, total


# '별표5' · '별표 4의2' · '별표 5-2' · '별표 28' → (5,None) · (4,2) · (5,2) · (28,None)
# Unify '-' (branch separator) into '의' — the same rule as the article parser (_parse_articles); the model mixes
# the two.
_BYEOLPYO_TOKEN_RE = re.compile(r"별\s*표\s*(\d+)(?:\s*의\s*(\d+))?")


def _requested_byeolpyo_numbers(articles: list[str | int] | None) -> list[tuple[int, int | None]]:
    out: list[tuple[int, int | None]] = []
    for token in articles or []:
        m = _BYEOLPYO_TOKEN_RE.search(str(token).replace("-", "의"))
        if m:
            out.append((int(m.group(1)), int(m.group(2)) if m.group(2) else None))
    return out


# Annexed tables and forms are not in the corpus; without this hint the model repeats the call in other formats. An
# annexed table asked alone comes as its title and file address (`_byeolpyo_rows`): the API reads that address with
# web_fetch, so no hint may say there is no annexed table.
_UNSUPPORTED_UNIT_HINT = "별표·별지·서식은 조문으로 오지 않습니다(본문이 코퍼스에 없습니다)."
_BYEOLPYO_ALONE_HINT = "별표는 그것만 따로 부르면(예: articles=['별표 5']) 제목과 파일 주소가 옵니다."


def _bad_articles_response(statute_id: int | str | None, articles: list[str | int]) -> dict[str, Any]:
    """Format guidance when not a single articles token could be read."""
    message = (
        "articles 파싱 실패 — 조문 번호를 숫자로 주세요. "
        "'76'(76조 본조+가지), '76-2'(제76조의2만). "
        "예: articles=['76-2','76-3']"
    )
    if _requested_byeolpyo_numbers(articles):
        # Reached only when the index has none of the asked annexed tables (those it has are answered).
        message = ("요청한 번호의 별표가 이 법령의 별표 목록에 없습니다 — statute_id 만 넣으면 별표 목록(제목과 파일 "
                   "주소)이 함께 옵니다.")
    elif _unsupported_units(articles):
        message = f"{_UNSUPPORTED_UNIT_HINT} {message}"
    return {
        "status": "bad_articles",
        "input": {"statute_id": statute_id, "articles": articles},
        "message": message,
    }


def _statute_lookup_impl(
    conn,
    query: str | None,
    ref: tuple[bool, int] | tuple[str, str] | None,
    articles: list[str | int] | None,
    limit: int,
    *,
    offense_date: str | None = None,
) -> dict[str, Any]:
    """ref = result of `_parse_statute_ref`: `(is administrative rule, id)` or `("law", law_id)`.
    None means search mode: a query that names one statute and article numbers ('민법 제750조') reaches those articles
    in the same call. A search passes no articles and a read no query (`statute_search`, `statute_read`); other
    callers (`article_sections`) read with a ref and articles."""
    named = _named_spans(conn, query) if query else set()
    if not articles and query:
        articles = _articles_in_query(query) or None

    if ref is None:
        # Statutes, administrative rules and tax treaties, interleaved by name relevance (_merge_law_notice_matches)
        # without quotas, so the model need not choose a kind in advance.
        words = _topic_words(query, named)
        topical = any(len(w) >= 3 for w in words)
        every = _query_tokens(query)

        def body_for(table):
            # The articles holding the most topic words. A query of statute names only (or of words too short for
            # the text index) searches the text with all its words together, as before; that article is not shown.
            if topical:
                return _body_match(conn, table, words)
            return (_quoted_and(every) if any(len(w) >= 3 for w in every) else _NO_BODY_MATCH), 0, []

        body, held, searched = body_for("st_articles_fts")
        stat_matches = _search_statutes(conn, query, max(limit, 10), offense_date=offense_date, body=body)
        law = _named_law(stat_matches, named) if articles else None
        if law:
            res = _statute_lookup_impl(conn, None, (False, law["statute_id"]), articles[:ARTICLES_MAX], limit,
                                       offense_date=offense_date)
            said = f"query 가 쓴 「{law['name']}」(statute_id={law['statute_id']})의 조문입니다."
            over = (_over_cap_note("query 의 조문", articles[ARTICLES_MAX:], law["statute_id"])
                    if len(articles) > ARTICLES_MAX else None)
            res["message"] = " · ".join(filter(None, [said, over, res.get("message")]))
            return res
        notice_body, notice_held, _ = body_for("st_notice_articles_fts")
        notice_matches = _search_notices(conn, query, limit, body=notice_body)
        # Tax treaties join only queries about a tax treaty (`treaties.about_tax_treaty`).
        treaty_matches = treaties.search(conn, query, limit)
        matches = _merge_law_notice_matches(conn, stat_matches[:limit] + notice_matches + treaty_matches, query, limit)
        if not topical:
            for m in matches:
                m.pop("matched_article", None)
        out = {"status": "ok", "mode": "list", "matches": matches}
        held = max(held, notice_held)
        if not held and any(m["fts_hits"] for m in treaty_matches):
            held = len(searched)  # a tax treaty found by its text: the list is not by name only
        messages = [_short_words_message(words, matches), _body_message(held, searched)]
        if any(messages):
            out["message"] = " · ".join(filter(None, messages))
        return out

    if ref[0] == "law":
        # law_id from a web path → the current version's statute_id (a dated lookup walks to its date from there).
        resolved = _resolve_current_statute_id(conn, ref[1], _today_iso())
        if resolved is None:
            return {
                "status": "not_found",
                "input": {"law_id": ref[1]},
                "message": (
                    "웹 경로의 법령 id(law_id)를 코퍼스에서 찾지 못했습니다 — "
                    "statute_search 로 법령명을 찾아 statute_id 를 받아 다시 부르세요."
                ),
            }
        ref = (False, resolved)

    if ref[0] == "trty":
        tid = ref[1]
        specs = _parse_articles(articles)
        res = treaties.lookup(conn, tid, specs, articles)
        if res is None:
            return {"status": "missing", "input": {"statute_id": f"{treaties.TREATY_URL_PREFIX}{tid}"}}
        return {"status": "ok", **res}

    is_notice, sid = ref
    shown_id = f"{NOTICE_URL_PREFIX}{sid}" if is_notice else sid
    specs = _parse_articles(articles)
    addenda = [] if is_notice else _addenda_specs(articles)

    # Articles given but not one token readable: say how to write them, rather than return an empty detail.
    if articles and not specs and not addenda:
        # An annexed table asked for: where it is, if the index has it; otherwise the format guidance.
        wanted = _requested_byeolpyo_numbers(articles)
        found, _ = _byeolpyo_rows(conn, _byeolpyo_owner(conn, is_notice, sid),
                                  numbers=wanted) if wanted else ([], 0)
        if found:
            return {
                "status": "byeolpyo",
                "input": {"statute_id": shown_id, "articles": articles},
                "message": (
                    "별표는 조문 조회 대상이 아니고 본문이 코퍼스에 없습니다 "
                    "— 아래는 그 별표의 제목과 파일 주소입니다."
                ),
                "byeolpyo": found,
                "byeolpyo_total": len(found),
            }
        return _bad_articles_response(shown_id, articles)

    if is_notice:
        res = (
            _detail_notice(conn, sid, specs)
            if specs is not None
            else _outline_notice(conn, sid)
        )
    else:
        res = (
            _detail_statute(conn, sid, specs, offense_date=offense_date)
            if specs is not None
            else _outline_statute(conn, sid, offense_date=offense_date)
        )

    if res is None:
        return {
            "status": "missing",
            "input": {"statute_id": shown_id},
        }
    if addenda:
        res.update(_addenda(conn, res["statute"]["law_id"], addenda,
                            to_iso_date(offense_date) or _today_iso()))
        res["addenda_until"] = to_iso_date(offense_date)
    # Annexed-table index — all of it for outlines, only what the articles point to for article text (an article
    # ending '별표 5와 같다', 'as in annexed table 5', comes with that table's address).
    if specs or not addenda:
        byeolpyo, total = _byeolpyo_rows(
            conn, _byeolpyo_owner(conn, is_notice, sid), specs=specs or None)
        if byeolpyo:
            res["byeolpyo"] = byeolpyo
            res["byeolpyo_total"] = total
    return {"status": "ok", **res}
