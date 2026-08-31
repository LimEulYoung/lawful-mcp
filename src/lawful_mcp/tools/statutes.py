"""Look up statutes and administrative rules, current or as of a date.

Three modes, chosen by which arguments arrive:
  - a query                -> a list of matching laws, one line each
  - a statute id           -> that law's article outline
  - a statute id + article numbers -> the article text

Statutes and administrative rules (고시·훈령·예규) are searched together and
ranked by name relevance; which corpus a lookup means is carried by the
identifier, not by a separate argument (see `_parse_statute_ref`).

Retrieval is lexical: trigram full text over article bodies, plus substring
matching on law names. Dense retrieval was tried and only added noise here —
statute text is short, formulaic and shares vocabulary across acts.

**Statutes have versions, and that is the hard part of this module.** A law
exists as a series of amendments, each stored as its own row, and asking
"what does article 347 say" has no answer without a date. The default is
today; passing an offence date returns the text as it stood then, which is
what a criminal matter actually turns on.

Two consequences run through the code below. A version row usually holds
only the articles that amendment changed, so the full text of a law lives in
its consolidated snapshot rather than in the latest row. And a search that
matched several versions of one law must fold them to one result, or the
same law fills the page.
"""
from __future__ import annotations

import datetime
import re
import sqlite3
from typing import Any

from pydantic_ai import RunContext

from ..config import case_url_base
from ..deps import HarnessDeps, open_db
from ._coerce import coerce_int, coerce_list, coerce_str, to_iso_date
from ._dedup import dedup_guard
from ._morph import kiwi as _kiwi

# There is deliberately no kind filter, and a `NOTICE_KINDS` set used to sit
# here to support one. Narrowing by kind is not worth reviving: the column
# holds 62 distinct values — 42 ministry-specific 부령 among them, plus
# historical leftovers — so a caller had to reproduce an exact string, and a
# near miss returned a silent zero rather than an error. Name relevance does
# that work instead.

# Articles per call. Without a cap a caller asks for a range like 1-79 and
# the response alone exceeds the context it was meant to inform.
ARTICLES_MAX = 8

# Search results. The caller's limit must not reach SQL unbounded: the
# search oversamples it eightfold, and version folding then issues a further
# query per law. A limit of a million would become a scan of eight million
# rows plus eight million follow-up queries. Results are relevance-ordered,
# so real use stays under ten and 50 is already generous.
LIMIT_MAX = 50

# Statute page URLs, same shape as the `url` field of the precedent tools:
# a law is /statutes/{law_id}, one article /statutes/{law_id}/{jo}, an
# administrative rule /statutes/admrul-{st_notices.id}. Attached to responses
# so a model citing the text can link to the source.
NOTICE_URL_PREFIX = "admrul-"   # law_id namespace for administrative rules


def _parse_statute_ref(raw: Any) -> tuple[bool, int] | tuple[str, str] | None:
    """Read a caller's ``statute_id`` into ``(is_rule, id)``, or None if unreadable.

    A bare integer means a statute (``st_statutes``); ``'admrul-18060'`` means an
    administrative rule (``st_notices``). One shape is neither: a web path
    ``/statutes/{law_id}`` — or a zero-padded digit string — carries a *law_id*
    ('004704') rather than a primary key, and comes back as ``("law", law_id)``
    for `_statute_lookup_impl` to resolve against the database.

    **An identifier has to name its own corpus.** The two tables number their
    rows independently and the ranges collide: 21,738 of 21,749 administrative
    rules (99.9%) share an integer with some statute. A separate `kind` argument
    used to carry that distinction, and when a caller left it off the answer was
    not "no such document" but an unrelated one, returned as `status: ok`. Over
    seven weeks of traffic, 29 of 48 lookups that fed a rule id back took that
    path. Hence the prefix.
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
        # Case and stray spaces do not distinguish anything, so accept both.
        low = s.lower().replace(" ", "")
        if low.startswith(NOTICE_URL_PREFIX):
            rest = low[len(NOTICE_URL_PREFIX):]
            return (True, int(rest)) if rest.isdigit() else None
        # A whole page url, pasted through from a previous answer.
        if "/statutes/" in low:
            tail = (low.rsplit("/statutes/", 1)[1]
                    .split("/")[0].split("?")[0].split("#")[0])
            if tail.startswith(NOTICE_URL_PREFIX):
                rest = tail[len(NOTICE_URL_PREFIX):]
                return (True, int(rest)) if rest.isdigit() else None
            # The statute slot of a web path is a law_id by definition. No
            # zero-padding is applied: a made-up /statutes/584 silently landing
            # on law_id 000584 is worse than failing loudly into a search.
            return ("law", tail) if tail.isdigit() else None
        if s.isdigit() and s.startswith("0") and len(s) >= 2:
            return ("law", s)             # zero padding only occurs in law_ids
        return (False, int(s)) if s.isdigit() else None
    return None


def _statute_ref(d: dict[str, Any] | None) -> str:
    """A result row -> the identifier the caller sees: a bare integer for a
    statute, ``'admrul-{id}'`` for an administrative rule. The corpus test is
    the same one `_statute_web_url` uses (does the row carry a law_id) — were
    the two to disagree, the id and the url in one response would point at
    different documents.
    """
    if not d:
        return ""
    nid = d.get("id")
    if not isinstance(nid, int):
        nid = d.get("statute_id")
    if d.get("law_id"):
        return str(nid) if isinstance(nid, int) else ""
    if d.get("category") or d.get("notice_id"):
        return f"{NOTICE_URL_PREFIX}{nid}" if isinstance(nid, int) else ""
    return str(nid) if isinstance(nid, int) else ""


# Which edition of an administrative rule search may surface. Amending one
# does not rewrite its row: the source publishes a *new row* under a new
# serial, so superseded editions ('구판') and not-yet-effective ones
# ('시행예정') sit in the table beside the current text. Unlabelled rows are
# treated as current — if labelling ever lags, showing a stale rule beats
# emptying the search entirely. Detail lookups deliberately skip this filter,
# so an indexed url for an old edition still opens instead of 404-ing.
_NOTICE_CURRENT = "COALESCE(n.history_status,'현행')='현행'"

# The exposure test is **whether there is text to read**, not how the document
# is typeset. An earlier `category='article_form'` condition admitted only
# rules laid out as numbered articles, which hid designation and approval
# notices wholesale — and some of those set rights and duties directly (the
# 최저임금법 §5 notice designating simple-labour occupations is one; it was
# excluded for being short). `has_text_content=1` covers the same article-form
# rows and additionally the bodies recovered from attachments, while the empty
# and image-only shells carry 0 and drop out on their own.


# ---------- helpers ----------

from ._fts import safe_fts_query


# ---------- law names: abbreviations and morpheme tokens ----------
# Korean law names run words together without spaces, so neither substring
# nor trigram matching finds the part a caller typed. Splitting the name
# into noun morphemes gives usable pieces to match on. Common abbreviations
# do not decompose that way at all and are mapped explicitly.

# Everyday and abbreviated names -> the official name, keyed without
# spaces. Values must match the stored name exactly. These are the cases
# morpheme splitting cannot reach: 산재, the ordinary word for a workplace
# injury claim, shares no characters with the act that governs it.
_STATUTE_ALIASES = {
    "헌법": "대한민국헌법",
    "산재": "산업재해보상보험법",
    "산재보험법": "산업재해보상보험법",
    "근기법": "근로기준법",
    "도교법": "도로교통법",
    "정통망법": "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
    "개인정보법": "개인정보 보호법",
    "개보법": "개인정보 보호법",
    "최임법": "최저임금법",
    "남녀고용평등법": "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률",
    # Renamed acts: the same law under its former title, which is how older
    # judgments cite it.
    "조세감면규제법": "조세특례제한법",
    "총포도검화약류등단속법": "총포ㆍ도검ㆍ화약류 등의 안전관리에 관한 법률",
    "학원의설립운영에관한법률": "학원의 설립ㆍ운영 및 과외교습에 관한 법률",
    # Hanja spellings, as they appear in older judgments.
    "憲法": "대한민국헌법",
    "憲法裁判所法": "헌법재판소법",
    # Abbreviations judgments use without ever defining them.
    "상증세법": "상속세 및 증여세법",
    "공익사업법": "공익사업을 위한 토지 등의 취득 및 보상에 관한 법률",
    "공선법": "공직선거법",
    "산재법": "산업재해보상보험법",
    "보험료징수법": "고용보험 및 산업재해보상보험의 보험료징수 등에 관한 법률",
    "주촉법": "주택건설촉진법",
    "공특법": "공공용지의취득및손실보상에관한특례법",
    "중개업법": "부동산중개업법",
    "특가법": "특정범죄 가중처벌 등에 관한 법률",
    "조특법": "조세특례제한법",
    "택상법": "택지소유상한에관한법률",
    "성폭법": "성폭력범죄의 처벌 등에 관한 특례법",
}


# Interpunct variants. Stored law names use U+318D; a caller types the name
# with no dot at all, or with one of several visually identical code points.
# Removing the dot from both sides makes all of them match.
_DOT_CHARS = ("ㆍ", "·", "‧", "∙", "・", "․")  # U+318D U+00B7 U+2027 U+2219 U+30FB U+2024


def _strip_dots(s: str) -> str:
    for ch in _DOT_CHARS:
        s = s.replace(ch, "")
    return s


def _name_norm_sql(col: str) -> str:
    """SQL expression normalising a stored name the same way `_strip_dots` does."""
    expr = col
    for ch in (" ", *_DOT_CHARS):
        expr = f"REPLACE({expr},'{ch}','')"
    return expr


_SPAN_MAX_TOKENS = 24     # morphemes considered, bounding the O(n^2) span set
_SPAN_MAX_CHARS = 64      # the longest law name fits well inside this


def _query_name_spans(query: str) -> list[str]:
    """Substrings of the query that could be a law name: the runs that both
    start and end on a morpheme boundary.

    Character-level substrings will not do. `손해배상 법률 상담` contains 「상법」
    by letters alone (손해배`상` + `법`률), and reading that coincidence as "the
    query named this law" puts the commercial code on top of a damages
    question — which is exactly what a first cut did. Runs that merely overlap
    another token's start are still candidates: 「군형법 제92조」 offers both
    '군형법' and '형법', and length comparison picks the longer.

    Spaces and interpuncts are removed, the same shape `_name_norm_sql`
    produces, so a span compares character for character against a normalised
    name in SQL. With no analyser the normalised query is the only span, which
    degrades to the exact matching this replaced.
    """
    raw = _strip_dots((query or "").strip())
    # An everyday name resolved through the alias table stands in for the
    # exact match. Values in that table keep their interpuncts
    # (「총포ㆍ도검ㆍ…」), so strip them once more or the span will never equal
    # the normalised name on the SQL side.
    spans = {_strip_dots(_STATUTE_ALIASES.get(raw.replace(" ", ""), raw).replace(" ", ""))}
    try:
        toks = _kiwi().tokenize(raw)[:_SPAN_MAX_TOKENS]
    except Exception:
        toks = []
    # raw already has its interpuncts gone, so dropping the spaces from a
    # slice of it leaves the shape the SQL side produces.
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
    """The name's length when the query contains it whole, otherwise 0. The
    parameters are the span list.

    Exact matching alone (`name = query`) misses most real queries. They read
    「law name + topic」, so a single trailing word switches it off, and the
    only measure left is token coverage — under which an administrative rule
    long enough to carry the topic words in its own name outranks the law the
    query pointed at. Over fourteen days of traffic, 322 of the 1,768 searches
    that named a law put an unrelated document first; the model then fetched
    articles by that id, so the wrong law's text reached the answer.

    Length is the score because a short name sits inside a longer one.
    '군형법 제92조' contains '형법' and '군형법' both, the answer is the longer,
    and comparing lengths makes that call. An exact match is this expression's
    maximum — the whole query — so it needs no branch of its own.

    Candidates come from `_query_name_spans` as morpheme-aligned runs and this
    only tests membership. Opening it to character substrings (`instr`) lets
    the coincidences back in.
    """
    n = _name_norm_sql(col)
    if n_spans <= 0:
        return "0"
    holes = ",".join("?" for _ in range(n_spans))
    return f"(CASE WHEN {n} IN ({holes}) THEN length({n}) ELSE 0 END)"


def _statute_name_tokens(query: str) -> tuple[str, list[str]]:
    """Return (normalised query, noun tokens) for matching a law name.

    광업제조업자 -> ['광업', '제조', '업자'], and 헌법 resolves through the
    alias table to the constitution's full name before being split.

    Tokens widen name matching only. With no analyser available the token
    list is empty and the caller falls back to the whole query.
    """
    raw = (query or "").strip()
    # Strip dots before anything else, so every spelling reaches the alias
    # table and the database in the same shape.
    raw = _strip_dots(raw)
    normalized = _STATUTE_ALIASES.get(raw.replace(" ", ""), raw)
    try:
        toks = [_strip_dots(t.form) for t in _kiwi().tokenize(normalized)
                if t.tag in ("NNG", "NNP") and len(_strip_dots(t.form)) >= 2]
    except Exception:
        toks = []
    return _strip_dots(normalized.replace(" ", "")), list(dict.fromkeys(toks))



def _fts_query_ok(q: str) -> bool:
    return any(len(w) >= 3 for w in q.split())


# Structural units that are not articles, filtered out before any number is
# read off the token.
#
# '별표5' used to match the article regex below and come back as 제5조, with
# `text_kind: 공식 조문 원문` on the response. Three consecutive requests for a
# rule's 별표 5 were each answered with an unrelated article, and the model
# kept retrying in different spellings. That is worse than not finding it —
# it says something exists when it does not. Tables and forms are not
# collected into the corpus at all, and addenda do not live in the article
# tables, so a token carrying one of these words is dropped without reading
# its number and ends at the fail-loud hint in `_bad_articles_response`.
_NON_ARTICLE_UNIT_RE = re.compile(
    r"별\s*표|별\s*지|별\s*첨|별\s*도|서\s*식|양\s*식|붙\s*임|부\s*칙|부\s*록")


def _parse_articles(
    spec: list[str | int] | None,
) -> list[tuple[int, int | None]] | None:
    """Parse article references into (number, branch) pairs.

    An article number alone selects the article and every branch under it;
    a number with a branch selects just that branch. A range is passed as a
    list of numbers rather than as a range expression.

    Unparseable tokens are skipped, as are structural units that are not
    articles (별표5, 부칙 — see `_NON_ARTICLE_UNIT_RE`). If that leaves nothing
    at all, the caller returns a format hint rather than silently searching
    for nothing.
    """
    if spec is None:
        return None
    if not isinstance(spec, list):
        spec = [spec]

    out: list[tuple[int, int | None]] = []
    for token in spec:
        # Accept every way an article gets written: 제76조의2, 76조의2,
        # 76의2, 76-2, 제76조, 76. Normalise the hyphen form first, then
        # pull the article number and its branch out of the rest.
        s = str(token).replace("-", "의")
        # A structural word at the *head* of the token makes it a structural
        # unit, so no number is read off it. A real article whose title
        # carries one of those words ('제106조(별도합산과세대상)') starts with
        # 제 and survives; '부칙 제2조' leads with 부칙 and is dropped. Matching
        # anywhere with `.search` swallowed legitimate articles.
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
        SELECT id, law_id, name, short_name, kind, issuing_agency,
               promulgation_date, effective_date
        FROM st_statutes WHERE id=?
        """,
        (sid,),
    ).fetchone()
    return dict(r) if r else None


def _notice_meta(conn: sqlite3.Connection, nid: int) -> dict[str, Any] | None:
    r = conn.execute(
        """
        SELECT id, serial_id, notice_id, name, kind, issuing_agency,
               notice_no, issued_date, effective_date,
               category, has_articles, has_text_content, history_status, body_source
        FROM st_notices WHERE id=?
        """,
        (nid,),
    ).fetchone()
    return dict(r) if r else None



# ---------- resolving a law as of a date ----------

def _today_iso() -> str:
    """Today, as YYYYMMDD.

    "Current" means in force today, decided by date rather than by the
    status label on a row: that excludes amendments not yet in force, and
    corrects rows whose label was never updated after their date passed.
    """
    return datetime.date.today().strftime('%Y%m%d')



# Structural headings — "Part 1, General Provisions", "Chapter 22, Offences
# Against Public Morals" — are stored as rows too, and they carry the
# article number of whatever follows them. Asking for that article can
# therefore return the heading instead. Article text always begins with
# 제<number>조; a heading names a part, chapter, section or subsection.
_HEADING_RE = re.compile(r'^\s*제\s*\d+\s*[편장절관]')


def _is_structural_heading(text: str | None) -> bool:
    """True for a structural heading rather than an article."""
    return bool(text and _HEADING_RE.match(text))


def _get_historic_article(
    conn: sqlite3.Connection,
    law_id: str,
    art_no: int,
    art_br: int | None,
    offense_iso: str,
) -> dict[str, Any] | None:
    """The article as it read on a given date: the last version in force by then.

    Takes the last version in force by that date, whether it comes from an
    amendment or from the baseline snapshot. Looking only at amended rows
    would miss every article that has never been amended — which includes
    much of the general part of the Criminal Code. Where both exist on the
    same date, the amendment wins.

    Structural headings are skipped: they share an article number with what
    follows, so taking the first row would return a chapter title as the
    text of an article. Repealed articles still return their repeal marker.
    """
    art_br = art_br or 0
    rows = conn.execute(
        """SELECT s.effective_date, s.mst, s.history_status, s.change_kind,
                  a.article_text, a.title, a.article_no, a.article_no_num, a.article_branch
           FROM st_articles a JOIN st_statutes s ON s.id=a.statute_id
           WHERE s.law_id=? AND a.article_no_num=? AND COALESCE(a.article_branch,0)=?
             AND s.effective_date <= ?
           ORDER BY s.effective_date DESC, (a.article_changed='Y') DESC""",
        (law_id, art_no, art_br, offense_iso),
    ).fetchall()
    for r in rows:
        if _is_structural_heading(r['article_text']):
            continue
        return dict(r)
    return None


def _pick_current_version(conn: sqlite3.Connection, group: list, as_of_iso: str) -> dict | None:
    """Pick the one version of a law that was current on a given date.

    In order: keep versions in force by that date (or all of them, for a
    law whose only rows are future amendments); prefer one that has not been
    repealed; then prefer the consolidated snapshot.

    That last step is the one that matters. An amendment row holds only the
    articles it changed — sometimes a single article — and the status label
    does not distinguish it from a consolidated text. Choosing by label
    alone therefore yields a "law" with one article in it, which was
    measured on 101 laws. Choosing the row with the most articles finds the
    consolidated text instead; full-snapshot origin and then recency break
    the remaining ties.
    """
    if not group:
        return None
    in_force = [r for r in group if (r['effective_date'] or '') <= as_of_iso] or list(group)
    non_repealed = [
        r for r in in_force
        if not (r['change_kind'] and '폐지' in str(r['change_kind']))
    ] or in_force
    snapshots = [r for r in non_repealed if r['history_status'] in (None, '현행')]
    pool = snapshots or non_repealed
    if len(pool) == 1:
        return pool[0]
    ids = [r['id'] for r in pool]
    counts = {sid: 0 for sid in ids}
    q = ("SELECT statute_id, COUNT(*) FROM st_articles WHERE statute_id IN (%s) "
         "GROUP BY statute_id" % ",".join("?" * len(ids)))
    for sid, c in conn.execute(q, ids):
        counts[sid] = c
    return max(pool, key=lambda r: (counts[r['id']], r['mst'] is None, r['effective_date'] or ''))


def _resolve_current_statute_id(
    conn: sqlite3.Connection, law_id: str, as_of_iso: str,
) -> int | None:
    """Law -> the id of the version current on that date."""
    group = conn.execute(
        "SELECT id, effective_date, history_status, change_kind, mst "
        "FROM st_statutes WHERE law_id=?",
        (law_id,),
    ).fetchall()
    chosen = _pick_current_version(conn, group, as_of_iso)
    return chosen['id'] if chosen else None


def _fmt_eff_iso(eff: Any) -> str:
    """'20260102' -> '2026-01-02'. For response fields a model reads as dates;
    the web renders the same value the Korean way, to be read by a person."""
    e = str(eff or "")
    return f"{e[:4]}-{e[4:6]}-{e[6:8]}" if len(e) == 8 and e.isdigit() else e


def _norm_law_name(s: Any) -> str:
    """Normalise a law name for comparison, as `_name_norm_sql` does in SQL."""
    return _strip_dots((s or "").replace(" ", ""))


def _current_version_ref(
    conn: sqlite3.Connection, law_id: str | None, sid: int, this_name: Any = None,
) -> dict[str, Any] | None:
    """A marker for the law's current version, when `sid` is not it; else None.

    The test is `_pick_current_version` and nothing else. Splitting on
    `history_status` gets it wrong in both directions: an older single-row
    load is the current text while carrying a NULL status, and the newest
    delta sometimes carries a '현행' label of its own.

    `renamed` means the name changed, not that the law was repealed, and the
    two must never share a badge. 「신기술사업금융지원에관한법률」 was never
    repealed; it became 「기술보증기금법」 under the same law_id. An actual
    repeal is what `change_kind` records.
    """
    if not law_id:
        return None
    cur_id = _resolve_current_statute_id(conn, law_id, _today_iso())
    if cur_id is None or cur_id == sid:
        return None
    cur = _statute_meta(conn, cur_id)
    if cur is None:
        return None
    return {
        "id": cur["id"],
        "name": cur["name"],
        "effective_date": cur["effective_date"],
        "renamed": _norm_law_name(cur["name"]) != _norm_law_name(this_name),
    }


def _backfilled_sids(conn: sqlite3.Connection, sids) -> set[int]:
    """The statute ids in `st_backfill_log` — 738 superseded editions loaded
    after the fact because judgments cite them. Empty when the table is absent,
    as it is in the bundled sample.

    This is a load record, not a repeal register, and using it as one is wrong.
    The loader's premise was "repealed laws with no successor in force", on the
    reasoning that a rename folds to its current edition and so never gets
    backfilled — but `_fold_versions` folds only among rows the query matched,
    and that candidate window is cut at `sql_limit`. A law matched under its
    old name alone has no current row in the window to fold into, so it goes
    out unfolded. Of the 738, **278 are renames whose successor is alive**:
    원자력법 -> 원자력 진흥법, 상속세법 -> 상속세 및 증여세법, 주택건설촉진법 ->
    주택법, 토지수용법 -> 공익사업을 위한 토지 등의 취득 및 보상에 관한 법률.
    The repeal verdict belongs to `_repealed_sids`.
    """
    sids = [s for s in sids if s is not None]
    if not sids:
        return set()
    try:
        q = ("SELECT statute_id FROM st_backfill_log WHERE status='ok' "
             "AND statute_id IN (%s)" % ",".join("?" * len(sids)))
        return {r[0] for r in conn.execute(q, sids)}
    except sqlite3.OperationalError:
        return set()


def _repealed_sids(conn: sqlite3.Connection, sids) -> set[int]:
    """Repealed = in the backfill record *and* with no later edition in force.

    Which is what the loader meant in the first place. A successor means the
    law was not killed but renamed, and then "현행 아님, and here is the current
    one" is the accurate thing to say rather than a repeal badge. Amendments
    not yet in force do not count as successors — a pending amendment cannot
    undo a repeal that already happened.
    """
    back = _backfilled_sids(conn, sids)
    if not back:
        return set()
    ids = sorted(back)
    q = ("SELECT s.id FROM st_statutes s WHERE s.id IN (%s) AND NOT EXISTS("
         "SELECT 1 FROM st_statutes s2 WHERE s2.law_id = s.law_id "
         "AND s2.effective_date > s.effective_date "
         "AND COALESCE(s2.history_status,'') != '시행예정')" % ",".join("?" * len(ids)))
    return {r[0] for r in conn.execute(q, ids)}


def _current_refs(conn: sqlite3.Connection, rows) -> dict[int, dict[str, Any]]:
    """Rows -> {statute_id: marker for the current edition}. A row that is
    already current has no key.

    Decided from *every* edition of the law_id, read in one query, not from
    whichever editions the search happened to match. Compare within the
    matched set — the grouping `_fold_versions` works on — and a law matched
    only under its old name is alone in its group and so becomes "the current
    one". That is the seat a superseded edition sat in at the top of a result
    list. The whole-law read costs 0.1-0.4ms against a 72-120ms search.

    Results are not swapped out here. Someone who searched by an old name and
    is shown only the new one is no less confused; the choice stands and only
    the marker is added.
    """
    lids = sorted({r["law_id"] for r in rows if r["law_id"]})
    if not lids:
        return {}
    q = ("SELECT id, law_id, name, effective_date, history_status, change_kind, mst "
         "FROM st_statutes WHERE law_id IN (%s)" % ",".join("?" * len(lids)))
    by_law: dict[str, list] = {}
    for r in conn.execute(q, lids):
        by_law.setdefault(r["law_id"], []).append(r)
    today = _today_iso()
    picked: dict[str, Any] = {}
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        lid = r["law_id"]
        if not lid:
            continue
        if lid not in picked:
            picked[lid] = _pick_current_version(conn, by_law.get(lid, []), today)
        cur = picked[lid]
        if cur is None or cur["id"] == r["id"]:
            continue
        out[r["id"]] = {
            "id": cur["id"],
            "name": cur["name"],
            "effective_date": cur["effective_date"],
            "renamed": _norm_law_name(cur["name"]) != _norm_law_name(r["name"]),
        }
    return out


def resolve_web_law_id(
    conn: sqlite3.Connection, law_id: str, as_of_iso: str,
) -> tuple[int, str] | None:
    """Web-namespace law_id ('004704') -> (current statute id, law name), or None.

    The ``/statutes/{law_id}`` path and the tool's ``statute_id`` are different
    id spaces, and this is the only place they are joined; the version pick is
    `_pick_current_version`'s, so a resolved id means the same edition a plain
    lookup would return.
    """
    sid = _resolve_current_statute_id(conn, law_id, as_of_iso)
    if sid is None:
        return None
    row = conn.execute(
        "SELECT name FROM st_statutes WHERE id=?", (sid,)).fetchone()
    return (sid, row["name"] if row and row["name"] else "")


def _is_repealed_as_of(
    conn: sqlite3.Connection, law_id: str | None, as_of_iso: str,
) -> bool:
    """Was the law repealed as of that date?
    Decided from the latest version in force by then. A law repealed and
    later re-enacted under the same id reads as not repealed, which is
    correct. The whole history is consulted, so a partial result set cannot
    hide the repeal.
    """
    if not law_id:
        return False
    r = conn.execute(
        """SELECT change_kind FROM st_statutes
           WHERE law_id=? AND effective_date<=?
           ORDER BY effective_date DESC LIMIT 1""",
        (law_id, as_of_iso),
    ).fetchone()
    return bool(r and r['change_kind'] and '폐지' in str(r['change_kind']))


def _fold_versions(conn: sqlite3.Connection, rows: list, offense_iso: str | None) -> list:
    """Fold search hits so each law appears once.

    With a date, the latest version in force by then. Without one, the
    version current today. Excluding laws that are themselves repealed is
    the search path's job, not this one's.
    """
    by_law: dict[str, list] = {}
    for r in rows:
        lid = r['law_id']
        if not lid:
            # A row with no law id cannot be grouped; keep it alone.
            by_law.setdefault(f'_solo_{r["id"]}', []).append(r)
        else:
            by_law.setdefault(lid, []).append(r)

    out = []
    for lid, group in by_law.items():
        if len(group) == 1 and lid.startswith('_solo_'):
            out.append(group[0])
            continue
        # Filter by date.
        if offense_iso:
            candidates = [
                r for r in group if r['effective_date'] and r['effective_date'] <= offense_iso
            ]
            if not candidates:
                # Nothing in force by then: the law did not yet exist.
                continue
            chosen = max(candidates, key=lambda r: r['effective_date'])
        else:
            # No date given means current: the consolidated snapshot in
            # force today, with future amendments excluded.
            chosen = _pick_current_version(conn, group, _today_iso())
        out.append(chosen)
    # Keep the original ranking order.
    seen = set()
    ordered = []
    for r in rows:
        if r['id'] in seen:
            continue
        if any(c['id'] == r['id'] for c in out):
            ordered.append(r)
            seen.add(r['id'])
    return ordered


def _is_repealed_at(conn: sqlite3.Connection, law_id: str, offense_iso: str | None) -> dict | None:
    """Was the law repealed or expired as of that date, or today?

    Returns the repeal's date and kind, or None.
    """
    target_date = offense_iso or '99999999'
    r = conn.execute(
        """SELECT effective_date, change_kind FROM st_statutes
           WHERE law_id=? AND effective_date<=? AND change_kind LIKE '%폐지%'
           ORDER BY effective_date DESC LIMIT 1""",
        (law_id, target_date),
    ).fetchone()
    return dict(r) if r else None


# ---------- article previews for the list mode ----------
#
# The line under a search hit is what tells a reader which law this is. It used
# to be article 1 unconditionally, but a purpose clause reads "…을 목적으로 한다"
# in every act: it says nothing about the query and nothing that separates
# 「신용정보법」 from 「신용카드업법」. When the query matched the body, show *the
# article it matched* — the rule the precedent search already follows for
# holdings, which is choosing among rows already retrieved rather than finding
# new ones.
#
# The cut is at the head of the article, not at the match. By corpus convention
# `article_text` opens with '제10조의4(권리금 회수기회 보호 등) ① …', so keeping
# the head keeps the one thing worth showing: which article this was. An
# excerpt centred on the match loses that.
SEARCH_PREVIEW_MAX_CHARS = 200


def _clip_preview(text: str | None) -> str:
    """An article body -> one line, with a trailing … when it was cut."""
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if len(t) > SEARCH_PREVIEW_MAX_CHARS:
        return t[:SEARCH_PREVIEW_MAX_CHARS].rstrip() + "…"
    return t


def _matched_article_previews(
    conn: sqlite3.Connection,
    ids: list[int],
    fts_query: str | None,
    *,
    fts_table: str,
    art_table: str,
    parent_col: str,
) -> dict[int, str]:
    """{parent id: head of the article the query matched best}. A law the query
    only named, without matching its text, has no key here.

    One query however many results there are. The shape this replaced ran one
    or two per result, so twenty hits meant up to forty round trips. The window
    passes article *ids* only and the body is read from the winning row with
    ``substr``, so a common token matching hundreds of articles within one law
    does not drag their text along with it.

    A miss is not reopened with an OR over the query's tokens. That was tried
    and rejected: 「상가건물 임대차보호법 권리금」 does improve to article 10-4,
    but 「개인정보 보호법 주민등록번호」 then answers with article 25, on video
    surveillance, because bm25 rewards common tokens like '정보' and '처리'.
    The index is trigram besides, so two-syllable tokens ('청약', '해고') never
    match at all and the very shapes the OR was meant to rescue stayed misses.
    An article shown here reads as "this is the provision you asked about", so
    a confident wrong one is worse than a purpose clause that merely fails to
    inform. Reviving it needs an index that can see two-syllable tokens first.
    """
    if not ids or not fts_query:
        return {}
    ph = ",".join("?" * len(ids))
    try:
        rows = conn.execute(
            f"""
            WITH best AS (
              SELECT a.{parent_col} AS pid, a.id AS aid,
                     ROW_NUMBER() OVER (PARTITION BY a.{parent_col}
                                        ORDER BY bm25({fts_table})) AS rn
              FROM {fts_table} f JOIN {art_table} a ON a.id = f.rowid
              WHERE {fts_table} MATCH ? AND a.{parent_col} IN ({ph})
            )
            SELECT b.pid, substr(x.article_text, 1, ?) AS t
            FROM best b JOIN {art_table} x ON x.id = b.aid
            WHERE b.rn = 1
            """,
            (fts_query, *ids, SEARCH_PREVIEW_MAX_CHARS * 2),
        ).fetchall()
    except sqlite3.OperationalError:
        # Malformed FTS query, or no index built. The preview is optional, so
        # fail open and let the caller fall back to the first article.
        return {}
    return {r["pid"]: _clip_preview(r["t"]) for r in rows if (r["t"] or "").strip()}


# ---------- list mode: search for laws ----------

def _search_statutes(
    conn: sqlite3.Connection,
    query: str | None,
    limit: int,
    *,
    offense_date: str | None = None,
) -> list[dict[str, Any]]:
    """Search by law name.

    Several versions of one law can match,
    so results are folded to one per law: the current version by default,
    or the one in force on a given date, including repealed and superseded
    versions when the date calls for them.
    """
    offense_iso = to_iso_date(offense_date)
    # No kind filter — a `kind` argument used to add a clause here.
    where_parts = []
    params: list[Any] = []
    fts_query_str: str | None = None   # the preview below reads it again

    # Oversample so folding versions still leaves enough distinct laws.
    sql_limit = max(limit * 8, 80)

    if query:
        # Morpheme tokens are used for the name match only. Scattering
        # common fragments across article bodies would match almost
        # everything, so the full-text query keeps the caller's wording.
        # With no analyser available, the whole normalised query is used.
        normalized, tokens = _statute_name_tokens(query)
        like_terms = tokens or [normalized]
        name_norm = _name_norm_sql("s.name")
        cover_sql = " + ".join(f"({name_norm} LIKE ?)" for _ in like_terms)
        name_or = " OR ".join(f"{name_norm} LIKE ?" for _ in like_terms)
        like_params = [f"%{t}%" for t in like_terms]
        # Length of whichever of the name or the official abbreviation the
        # query contains whole (`_name_in_query_sql`).
        spans = _query_name_spans(query)
        name_in_q = _name_in_query_sql("s.name", len(spans))
        short_in_q = _name_in_query_sql("COALESCE(s.short_name,'')", len(spans))
        sql = f"""
        WITH fts_hits AS (
          SELECT a.statute_id, COUNT(*) AS hits
          FROM st_articles_fts f
          JOIN st_articles a ON a.id = f.rowid
          WHERE st_articles_fts MATCH ?
          GROUP BY a.statute_id
        )
        SELECT s.id, s.law_id, s.name, s.short_name, s.kind, s.issuing_agency,
               s.effective_date, s.history_status, s.change_kind, s.mst,
               COALESCE(h.hits, 0) AS fts_hits,
               ({cover_sql}) AS name_cover,
               MAX({name_in_q}, {short_in_q}) AS name_in_query
        FROM st_statutes s
        LEFT JOIN fts_hits h ON h.statute_id = s.id
        """
        safe_q = safe_fts_query(query)
        fts_query_str = safe_q if _fts_query_ok(safe_q) else "x" * 1000
        # A candidate matches on the name, on an abbreviation the query
        # contains, or in the text — not necessarily all three. The
        # abbreviation needs its own clause because its characters need not
        # appear in the full name at all ('상증세법' against 「상속세 및
        # 증여세법」), so a token LIKE never reaches it.
        where_parts.append(f"(({name_or}) OR {short_in_q} > 0 OR h.hits > 0)")
        sql += "WHERE " + " AND ".join(where_parts) + "\n"
        # Rank by whether the query contains the name whole, longest first,
        # then how many query tokens the name covers, then name length, then
        # text hits. An exact match is the first key's maximum, so it has no
        # column of its own (`_name_in_query_sql`).
        sql += """
        ORDER BY name_in_query DESC, name_cover DESC, length(s.name) ASC, fts_hits DESC
        LIMIT ?
        """
        # Parameter order follows the order they appear in the SQL:
        #   fts MATCH, cover (SELECT), name_in_q (spans), short_in_q (spans),
        #   name_or (WHERE), short_in_q (WHERE, spans), LIMIT
        params_q = [fts_query_str, *like_params, *spans, *spans]
        rows = conn.execute(
            sql, [*params_q, *params, *like_params, *spans, sql_limit]).fetchall()
    else:
        sql = ("SELECT s.id, s.law_id, s.name, s.kind, s.issuing_agency, "
               "s.effective_date, s.history_status, s.change_kind, s.mst, "
               "0 AS fts_hits, 0 AS name_hit FROM st_statutes s")
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
        sql += " ORDER BY s.name LIMIT ?"
        rows = conn.execute(sql, [*params, sql_limit]).fetchall()

    # Fold to one row per law.
    rows = _fold_versions(conn, rows, offense_iso)
    if offense_iso is None:
        # Current search drops laws that are repealed today, otherwise the
        # last version before repeal surfaces as though it were live — the
        # replacement usually exists under a different id. A dated search
        # keeps them, because on that date they were the law.
        today = _today_iso()
        rows = [r for r in rows if not _is_repealed_as_of(conn, r['law_id'], today)]
    rows = rows[:limit]

    # A law whose text the query matched previews *that* article; one the query
    # only named previews article 1. The matched side arrives in a single query
    # and only the leftovers fall back to a per-row lookup, which is an index
    # hit limited to one row and covers few results in practice.
    matched = _matched_article_previews(
        conn, [r["id"] for r in rows], fts_query_str,
        fts_table="st_articles_fts", art_table="st_articles", parent_col="statute_id",
    )
    # Time-axis markers, computed here so the tool and the web read the same
    # value. This used to live in the web wrapper alone, which is why a person
    # saw the badge and a model did not.
    cur_refs = _current_refs(conn, rows)
    repealed = _repealed_sids(conn, [r["id"] for r in rows])

    out = []
    for r in rows:
        preview = matched.get(r["id"], "")
        if not preview:
            prev = conn.execute(
                """
                SELECT article_text FROM st_articles
                WHERE statute_id=? AND article_no_num=1
                  AND title IS NOT NULL
                ORDER BY article_branch NULLS FIRST LIMIT 1
                """,
                (r["id"],),
            ).fetchone()
            if prev is None:
                prev = conn.execute(
                    """
                    SELECT article_text FROM st_articles
                    WHERE statute_id=? AND length(article_text) > 30
                    ORDER BY article_no_num, article_branch LIMIT 1
                    """,
                    (r["id"],),
                ).fetchone()
            preview = _clip_preview(prev["article_text"] if prev else "")
        out.append(
            {
                "statute_id": r["id"],
                "law_id": r["law_id"],
                "name": r["name"],
                # The official abbreviation. Nothing in the response text uses
                # it; the merge sort in `_merge_law_notice_matches` reads it,
                # so a law the query named by abbreviation is not pushed back
                # down after the two corpora are interleaved.
                "short_name": (r["short_name"] if "short_name" in r.keys() else None),
                "kind": r["kind"],
                "agency": r["issuing_agency"],
                "effective_date": r["effective_date"],
                "history_status": r["history_status"] if "history_status" in r.keys() else None,
                "change_kind": r["change_kind"] if "change_kind" in r.keys() else None,
                "preview": preview,
                # Is this row the law's current edition, and if not, what is.
                "is_repealed": r["id"] in repealed,
                "current": cur_refs.get(r["id"]),
            }
        )
    return out


def _search_notices(
    conn: sqlite3.Connection,
    query: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    fts_query_str: str | None = None   # the preview below reads it again
    if query:
        # Same token-coverage ranking as the statute search. Names carry
        # particles and spacing that a substring match cannot survive — one
        # intervening 의 was enough to break it — so rank by how many of the
        # query's tokens the name contains.
        normalized, tokens = _statute_name_tokens(query)
        spans = _query_name_spans(query)
        like_terms = tokens or [normalized]
        name_norm = _name_norm_sql("n.name")
        cover_sql = " + ".join(f"({name_norm} LIKE ?)" for _ in like_terms)
        name_or = " OR ".join(f"{name_norm} LIKE ?" for _ in like_terms)
        like_params = [f"%{t}%" for t in like_terms]
        safe_q = safe_fts_query(query)
        fts_query_str = safe_q if _fts_query_ok(safe_q) else "x" * 1000
        sql = f"""
        WITH fts_hits AS (
          SELECT a.notice_id, COUNT(*) AS hits
          FROM st_notice_articles_fts f
          JOIN st_notice_articles a ON a.id = f.rowid
          WHERE st_notice_articles_fts MATCH ?
          GROUP BY a.notice_id
        )
        SELECT n.id, n.notice_id, n.name, n.kind, n.issuing_agency,
               n.effective_date, n.category,
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
        # The same key as the statute search. If only one side knew about
        # containment the merge would be comparing two different measures.
        rows = conn.execute(
            sql,
            [fts_query_str, *like_params, *spans, *like_params, limit],
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, notice_id, name, kind, issuing_agency, effective_date, category,
                   0 AS fts_hits, 0 AS name_hit
            FROM st_notices
            WHERE has_text_content=1
              AND COALESCE(history_status,'현행')='현행'
            ORDER BY name LIMIT ?
""",
            (limit,),
        ).fetchall()

    # Same rule as the laws: the matched article first, else the opening one.
    matched = _matched_article_previews(
        conn, [r["id"] for r in rows], fts_query_str,
        fts_table="st_notice_articles_fts", art_table="st_notice_articles",
        parent_col="notice_id",
    )

    out = []
    for r in rows:
        preview = matched.get(r["id"], "")
        if not preview:
            prev = conn.execute(
                "SELECT article_text FROM st_notice_articles WHERE notice_id=?"
                " ORDER BY article_seq LIMIT 1",
                (r["id"],),
            ).fetchone()
            preview = _clip_preview(prev["article_text"] if prev else "")
        out.append(
            {
                "statute_id": r["id"],
                "notice_id": r["notice_id"],
                "name": r["name"],
                "kind": r["kind"],
                "category": r["category"],
                "agency": r["issuing_agency"],
                "effective_date": r["effective_date"],
                "preview": preview,
            }
        )
    return out


def _name_cover_key(name: str, q_norm: str, tokens: list[str], spans=()):
    """Name-relevance sort key, lower first, shared by laws and rules.

    Having one measure lets the two be interleaved on merit rather than by
    giving each a fixed share of the results. It has to be the *same* measure
    `_search_statutes` sorts by — whether the query contains the name whole,
    longest first, then token coverage. When the two drift apart the merge
    undoes the search, and a rule with a long name buries the law the search
    correctly raised, so the two move together in one change.

    `spans` is what `_query_name_spans` produced for the SQL side. Imitating
    it with character substrings here would readmit the coincidences that
    cross word boundaries (「상법」 inside '손해배상 법률'). Interpuncts come out
    here too, matching `_name_norm_sql`: stripping only spaces meant names
    like 「초ㆍ중등교육법」 never took the containment branch at all.
    """
    n = _strip_dots((name or "").replace(" ", ""))
    if n and n in spans:
        # An exact match is this branch's longest possible span — the whole
        # query — so it needs no case of its own.
        return (0, -len(n))
    if tokens:
        cover = sum(1 for t in tokens if t in n)
        return (1, len(tokens) - cover)        # fewer missing tokens sorts higher
    return (1, 0) if (q_norm and q_norm in n) else (2, 0)


def _merge_law_notice_matches(matches, query, limit, name_of=None, alt_name_of=None):
    """Interleave law and rule results by name relevance, with no fixed share.

    The sort is stable, so within one relevance tier the input order
    survives.

    ``alt_name_of`` reads the official abbreviation. A law the query named
    only by that (「신용정보법」 for 「신용정보의 이용 및 보호에 관한 법률」) is
    scored on whichever of the two names does better, so the merge does not
    push it back down. Rules have no abbreviation and yield an empty string,
    leaving only the full-name key.
    """
    if name_of is None:
        name_of = lambda m: m.get("name", "")
    if alt_name_of is None:
        alt_name_of = lambda m: m.get("short_name") or ""
    normalized, tokens = _statute_name_tokens(query) if query else ("", [])
    spans = frozenset(_query_name_spans(query)) if query else frozenset()

    def key(m):
        best = _name_cover_key(name_of(m), normalized, tokens, spans)
        alt = alt_name_of(m)
        return min(best, _name_cover_key(alt, normalized, tokens, spans)) if alt else best

    return sorted(matches, key=key)[:limit]


# ---------- outline mode: a law's article titles ----------

def _outline_statute(
    conn: sqlite3.Connection, sid: int, offense_date: str | None = None,
) -> dict[str, Any] | None:
    """Outline: the title of every article in a law.

    With a date, the outline is assembled from each article's last version
    in force by then — including articles since repealed, if they were in
    force on that date.
    """
    meta = _statute_meta(conn, sid)
    if meta is None:
        return None

    offense_iso = to_iso_date(offense_date)
    if offense_iso and meta.get('law_id'):
        # Assemble from the last version of each article by that date.
        law_id = meta['law_id']
        rows = conn.execute(
            """SELECT a.article_no, a.article_no_num, a.article_branch, a.title,
                      a.article_eff_date
               FROM st_articles a JOIN st_statutes s ON s.id=a.statute_id
               WHERE s.law_id=? AND a.article_changed='Y' AND a.title IS NOT NULL
                 AND s.effective_date <= ?
               ORDER BY a.article_no_num, a.article_branch, s.effective_date DESC""",
            (law_id, offense_iso),
        ).fetchall()
        # Keep the newest row per article; already ordered by date.
        seen: set[tuple] = set()
        articles: list[dict] = []
        for r in rows:
            key = (r['article_no_num'], r['article_branch'] or 0)
            if key in seen:
                continue
            seen.add(key)
            articles.append({
                'no': r['article_no'], 'no_num': r['article_no_num'],
                'branch': r['article_branch'], 'title': r['title'],
                'eff_date': r['article_eff_date'],
            })
        articles.sort(key=lambda a: (a['no_num'], a['branch'] or 0))
        out = {
            'mode': 'outline',
            'statute': {
                'id': meta['id'], 'law_id': meta['law_id'],
                'name': meta['name'], 'kind': meta['kind'],
                'agency': meta['issuing_agency'],
            },
            'offense_date': offense_iso,
            'articles': articles,
        }
        # A dated lookup keeps the name the law had then — that one is
        # correct — and only notes what the current version is.
        cur_ref = _current_version_ref(conn, law_id, meta['id'], meta.get('name'))
        if cur_ref:
            out['current_version'] = cur_ref
        return out

    # Redirect to today's consolidated snapshot even when the caller passed
    # the id of an amendment row, so an outline is the whole law rather than
    # the handful of articles that amendment touched.
    as_of = _today_iso()
    asked = None
    if meta.get('law_id'):
        cur_id = _resolve_current_statute_id(conn, meta['law_id'], as_of)
        if cur_id is not None and cur_id != sid:
            cur_meta = _statute_meta(conn, cur_id)
            if cur_meta:
                # A redirect that changes the name loses the caller its own
                # question. Carry the asked-for name so the response can say
                # why a different one came back; `_detail_statute` is the
                # other half of this.
                if _norm_law_name(cur_meta["name"]) != _norm_law_name(meta["name"]):
                    asked = {"name": meta["name"],
                             "effective_date": meta["effective_date"]}
                meta = cur_meta
            sid = cur_id

    # Filtering on a non-null title would empty out laws that have none:
    # the constitution numbers its articles without parenthesised titles, and
    # single-provision regulations have neither a title nor an article number.
    # Both would return an empty outline while plainly having text. Exclude
    # structural headings and nothing else.
    rows = conn.execute(
        """
        SELECT article_no, article_no_num, article_branch, title, article_text
        FROM st_articles
        WHERE statute_id = ?
        ORDER BY article_no_num, article_branch, id
        """,
        (sid,),
    ).fetchall()
    articles, seen = [], set()
    for r in rows:
        if _is_structural_heading(r["article_text"]):
            continue
        if r["article_no_num"] is None:
            continue
        if r["article_no_num"] > 0:
            key = (r["article_no_num"], r["article_branch"] or 0)
            if key in seen:
                continue
            seen.add(key)
        articles.append({
            "no": r["article_no"],
            "no_num": r["article_no_num"],
            "branch": r["article_branch"],
            "title": r["title"] or "",
            "text": r["article_text"] or "",
        })

    statute_kv = {
        "id": meta["id"],
        "law_id": meta["law_id"],
        "name": meta["name"],
        "kind": meta["kind"],
        "agency": meta["issuing_agency"],
    }

    # Regulations with no article structure at all — a single untitled body,
    # such as the statutory interest rate. An outline of these is a blank
    # entry the caller cannot follow up on: it guesses "article 1", gets a
    # miss, and gives up. They are short (under about 650 characters), so
    # return the text itself instead of an outline.
    if articles and not any((a["no_num"] or 0) > 0 for a in articles):
        out = {
            "mode": "detail",
            "statute": statute_kv,
            "articles": articles,
            "missing": [],
            "note": "제N조 구조가 없는 단일 본문 규정 — 아래가 본문 전문입니다.",
        }
        if asked:
            out["asked_name"] = asked
        return out

    # Ordinary laws: outline is titles only, no article text.
    for a in articles:
        a.pop("text", None)
    out = {
        "mode": "outline",
        "as_of": as_of,
        "statute": statute_kv,
        "articles": articles,
    }
    if asked:
        out["asked_name"] = asked
    return out


_NOTICE_TITLE_RE = re.compile(r'^\s*제\s*\d+(?:-\d+)?\s*조(?:의\s*\d+)?\s*\(([^)]*)\)')


def _notice_article_title(text: str) -> str:
    """Title of an administrative-rule article.

    These have no title column; the title sits in parentheses at the head of
    the article text itself.
    """
    m = _NOTICE_TITLE_RE.match(text or "")
    return m.group(1).strip() if m else ""


def _outline_notice(conn: sqlite3.Connection, nid: int) -> dict[str, Any] | None:
    meta = _notice_meta(conn, nid)
    if meta is None:
        return None

    if meta["category"] == "article_form":
        rows = conn.execute(
            """
            SELECT article_seq, article_no, article_no_str, article_text,
                   length(article_text) AS textlen
            FROM st_notice_articles
            WHERE notice_id=? ORDER BY article_seq
            """,
            (nid,),
        ).fetchall()
        # Show the natural article number. The stored value packs the branch
        # into the integer, so article 1 would otherwise display as 1000.
        articles = [
            {
                "seq": r["article_seq"],
                # Both can be NULL: a part/chapter/section heading carries no
                # article number, and `str(None)` would hand the caller the
                # string "None" as one. No article-form rule had such a row
                # before 훈령·예규 joined the corpus, so it never showed.
                "no": r["article_no_str"] or (str(r["article_no"]) if r["article_no"] else ""),
                "no_str": r["article_no_str"],
                "title": _notice_article_title(r["article_text"]),
                "textlen": r["textlen"],
            }
            for r in rows
        ]
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


# ---------- detail mode: article text ----------

def _detail_statute(
    conn: sqlite3.Connection,
    sid: int,
    specs: list[tuple[int, int | None]],
    *,
    offense_date: str | None = None,
) -> dict[str, Any] | None:
    """Article text for the requested specs.

    A spec is ``(number, branch)``: a branch of None means the article and
    every branch under it, a branch given means that one alone. With a date,
    each is answered from its last change in force by then — the version rows
    hold amendments, so a date is resolved by walking back, not by reading
    one row.
    """
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

    # Current text and dated text resolve the same way. Where amendments are
    # stored as deltas, "the current article" just means "the article as of
    # the latest date in force", so both go through the same walk back to
    # each article's last change. Querying the current version row directly
    # instead returned nothing for every article that amendment did not
    # touch — 86 of 89 laws in a check. Amendments not yet in force are
    # excluded: they are not current.
    law_id = meta.get('law_id')
    if law_id:
        as_of = offense_iso
        asked = None
        if as_of is None:
            # Serving the current body means serving the current name. This
            # used to keep the requested row's metadata, so current text went
            # out under a superseded name — while an outline of the same id
            # redirected and answered with the current one. One statute_id
            # appeared to hold two different laws, and a model reading that
            # concluded the corpus was contaminated and abandoned it, in a
            # production turn that then spent 636 seconds and 36 tool calls
            # fetching from the government site. The article it needed was in
            # the corpus the whole time. The asked-for name is not discarded:
            # `asked_name` carries it so the response answers the question the
            # discrepancy raises.
            cur_id = _resolve_current_statute_id(conn, law_id, _today_iso())
            if cur_id is not None and cur_id != sid:
                cur_meta = _statute_meta(conn, cur_id)
                if cur_meta:
                    if _norm_law_name(cur_meta["name"]) != _norm_law_name(meta["name"]):
                        asked = {"name": meta["name"],
                                 "effective_date": meta["effective_date"]}
                    meta = cur_meta
            cur = conn.execute(
                "SELECT MAX(effective_date) AS d FROM st_statutes "
                "WHERE law_id=? AND COALESCE(history_status,'') != '시행예정'",
                (law_id,),
            ).fetchone()
            as_of = cur["d"] if cur and cur["d"] else None
        if as_of:
            res = _detail_statute_at_date(conn, meta, specs, as_of,
                                          asked_iso=offense_iso)
            if asked:
                res["asked_name"] = asked
            return res

    # Fallback for a row with no law id, or when no date can be resolved.
    conditions: list[str] = []
    params: list[Any] = [sid]
    for num, branch in specs:
        if branch is None:
            conditions.append("article_no_num = ?")
            params.append(num)
        else:
            conditions.append("(article_no_num = ? AND article_branch = ?)")
            params.extend([num, branch])
    where = " OR ".join(conditions)

    rows = conn.execute(
        f"""
        SELECT article_no, article_no_num, article_branch, title, article_text
        FROM st_articles
        WHERE statute_id = ? AND ({where})
          AND title IS NOT NULL
        ORDER BY article_no_num, article_branch
        """,
        params,
    ).fetchall()

    # Report which requested articles produced nothing.
    def _matched(spec: tuple[int, int | None]) -> bool:
        num, branch = spec
        if branch is None:
            return any(r["article_no_num"] == num for r in rows)
        return any(r["article_no_num"] == num and r["article_branch"] == branch for r in rows)

    def _spec_str(spec: tuple[int, int | None]) -> str:
        num, branch = spec
        return str(num) if branch is None else f"{num}-{branch}"

    missing = [_spec_str(s) for s in specs if not _matched(s)]
    result = {
        "mode": "detail",
        "statute": {
            "id": meta["id"],
            "law_id": meta["law_id"],
            "name": meta["name"],
            "kind": meta["kind"],
        },
        "articles": [
            {
                "no": r["article_no"],
                "no_num": r["article_no_num"],
                "branch": r["article_branch"],
                "title": r["title"],
                "text": r["article_text"],
            }
            for r in rows
        ],
        "missing": missing,
    }
    if missing:
        result["hint"] = (
            f"missing {missing} — '{meta['name']}'에 해당 article 없음. "
            "정확한 조문 번호는 statute_lookup(statute_id, articles=null)로 outline 호출해 확인하세요."
        )
    return result


def _detail_statute_at_date(
    conn: sqlite3.Connection,
    meta: dict[str, Any],
    specs: list[tuple[int, int | None]],
    offense_iso: str,
    *,
    asked_iso: str | None = None,
) -> dict[str, Any]:
    """Article text as of a date, walking back to each article's last change.

    Handles an article and its branches alike.

    `offense_iso` is filled in for a current lookup too — "current" means the
    latest date in force, resolved by the same walk — so it cannot be used to
    decide whether the caller asked about a date. That is what `asked_iso` is
    for, and the response's `offense_date` carries only that. Conflate them
    and every ordinary lookup reports itself as a dated one.
    """
    law_id = meta['law_id']
    # Check whether the law was repealed by then.
    repealed = _is_repealed_at(conn, law_id, offense_iso)

    articles: list[dict] = []
    missing: list[str] = []
    for num, branch in specs:
        # No branch requested: resolve the article and every branch under
        # it, each at its own last change.
        if branch is None:
            # Fetch all branches of this article in one query.
            branches_rows = conn.execute(
                """SELECT DISTINCT a.article_branch FROM st_articles a
                   JOIN st_statutes s ON s.id=a.statute_id
                   WHERE s.law_id=? AND a.article_no_num=?
                     AND s.effective_date <= ?""",  # baseline rows included
                (law_id, num, offense_iso),
            ).fetchall()
            branches = sorted({r['article_branch'] or 0 for r in branches_rows})
            if not branches:
                missing.append(f"{num}")
                continue
            for br in branches:
                r = _get_historic_article(conn, law_id, num, br, offense_iso)
                if r:
                    articles.append(_format_historic_article_row(r))
        else:
            r = _get_historic_article(conn, law_id, num, branch, offense_iso)
            if r:
                articles.append(_format_historic_article_row(r))
            else:
                missing.append(f"{num}-{branch}")

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
    if asked_iso:
        out["offense_date"] = asked_iso
    if repealed:
        out["repealed"] = {
            "effective_date": repealed['effective_date'],
            "change_kind": repealed['change_kind'],
            "note": f"이 법령은 {repealed['effective_date']} {repealed['change_kind']}. "
                    f"행위시점 {offense_iso} 에는 유효였을 수 있음 (확인 필요)."
        }
    # Where `latest_version` used to be. That one took the newest row even
    # when it was a future amendment, and no renderer ever printed it, so it
    # was computed and dropped.
    cur_ref = _current_version_ref(conn, law_id, meta['id'], meta.get('name'))
    if cur_ref is None and offense_iso and articles:
        # A dated lookup walks each article separately, so the row can be the
        # current one while the text served is not: 도로교통법 id=557 is the
        # current row, and a 2019 lookup against it returns 2019 wording.
        # Comparing rows alone misses that, and the response loses the line
        # saying which date the text is from.
        served = max((a.get("eff_date") or "") for a in articles)
        cur_eff = str(meta.get("effective_date") or "")
        if served and cur_eff and served != cur_eff:
            cur_ref = {"id": meta["id"], "name": meta["name"],
                       "effective_date": cur_eff, "renamed": False}
    if cur_ref:
        out["current_version"] = cur_ref
    return out


def _format_historic_article_row(r: dict) -> dict:
    return {
        "no": r["article_no"],
        "no_num": r["article_no_num"],
        "branch": r["article_branch"],
        "title": r["title"],
        "text": r["article_text"],
        "eff_date": r["effective_date"],
        "version_status": r.get("history_status"),
        "change_kind": r.get("change_kind"),
    }


def _detail_notice(
    conn: sqlite3.Connection,
    nid: int,
    specs: list[tuple[int, int | None]] | None,
    text_max: int = 10_000,
) -> dict[str, Any] | None:
    """Look up rule articles, translating to their stored encoding.

    Administrative rules pack an article and its branch into one integer
    (article 6-2 becomes 6002). Callers pass the natural numbers the outline
    showed them; the translation happens here.
    """
    meta = _notice_meta(conn, nid)
    if meta is None:
        return None
    cat = meta["category"]

    if cat == "article_form":
        # A requested branch matches exactly; without one, take the whole
        # thousand-block, meaning the article and all its branches.
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
            SELECT article_seq, article_no, article_no_str, article_text
            FROM st_notice_articles
            WHERE notice_id=? AND ({where})
            ORDER BY article_seq
            """,
            [nid, *params],
        ).fetchall()
        articles = [
            {
                "seq": r["article_seq"],
                # Both can be NULL: a part/chapter/section heading carries no
                # article number, and `str(None)` would hand the caller the
                # string "None" as one. No article-form rule had such a row
                # before 훈령·예규 joined the corpus, so it never showed.
                "no": r["article_no_str"] or (str(r["article_no"]) if r["article_no"] else ""),
                "no_str": r["article_no_str"],
                "title": _notice_article_title(r["article_text"]),
                "text": r["article_text"],
            }
            for r in rows
        ]
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
        body = "\n\n".join(r["article_text"] for r in rows)
        truncated = len(body) > text_max
        return {
            "mode": "detail",
            "statute": {"id": nid, "name": meta["name"], "kind": meta["kind"], "category": cat},
            "body": body[:text_max],
            "truncated": truncated,
        }

    return {
        "mode": "detail",
        "statute": {"id": nid, "name": meta["name"], "kind": meta["kind"], "category": cat},
        "content_format": cat,
        "articles": [],
        "note": "본문이 이미지로만 제공되거나 비어있음" if cat == "image_only" else "응답 본문 없음",
    }


# ---------- markdown serialisation ----------

def _fmt_article_no(no: Any, branch: Any) -> str:
    """Display form of an article number, with its branch if it has one.

    `st_articles.article_no` already carries the branch in some load
    generations. 도로교통법 제148조의2 is stored as no='148'/branch=2 on the
    current row but as no='148의2'/branch=2 on the 2019 one. Appending
    regardless produces '148의2-2', a number that does not exist: a model
    prints it, or asks for it and gets a miss. 형사소송법 shows the same in a
    plain outline ('16의2-2', '59의3-3'), so the outline-to-detail round trip
    breaks there. When the branch is already in the number, leave it alone.
    """
    s = str(no)
    return f"{s}-{branch}" if branch and "의" not in s else s


def _statute_web_url(d: dict[str, Any]) -> str | None:
    """Page url for a law or an administrative rule, or None if undetermined."""
    if not d:
        return None
    law_id = d.get("law_id")
    if law_id:
        return f"{case_url_base()}/statutes/{law_id}"
    # Administrative rules have no law id; they carry a category and rule id.
    if d.get("category") or d.get("notice_id"):
        nid = d.get("id")
        if not isinstance(nid, int):
            nid = d.get("statute_id")
        if isinstance(nid, int):
            return f"{case_url_base()}/statutes/{NOTICE_URL_PREFIX}{nid}"
    return None


def _statute_article_url(stt: dict[str, Any] | None, art: dict[str, Any]) -> str | None:
    """Page url for a single article, or None when there is no article to link.

    Falls back to the law-level url in that case.
    """
    law_id = (stt or {}).get("law_id")
    if not law_id:
        return None
    no_num = art.get("no_num")
    if no_num is None:
        return None
    try:
        n = int(no_num)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    b = int(art.get("branch") or 0)
    jo = f"{n}의{b}" if b else str(n)
    return f"{case_url_base()}/statutes/{law_id}/{jo}"


def _format_response_md(resp: dict[str, Any]) -> str:
    """Response dict -> markdown-KV string."""
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
            # Administrative rules go out under the 'admrul-' prefix: handing
            # that identifier straight back is what reaches the same document,
            # since a bare integer is read in the statute id space.
            mid = _statute_ref(m) or m.get("statute_id") or m.get("id")
            kind = m.get("kind") or ""
            head = f"- {mid} {m.get('name','')}" + (f" ({kind})" if kind else "")
            eff = _fmt_eff_iso(m.get("effective_date"))
            if eff:
                head += f" · {eff} 시행"
            lines.append(head)
            # Whether this is a name still in use belongs in the list too.
            # Without these two lines a model reads a name last current in
            # 2000 as the law in force and builds an answer on top of it.
            cur = m.get("current")
            if cur:
                ceff = _fmt_eff_iso(cur.get("effective_date"))
                tail = (f"「{cur.get('name','')}」({ceff} 시행)" if cur.get("renamed")
                        else f"{ceff} 시행본")
                lines.append(f"  현행 아님 — 이 법의 현행은 {tail}, "
                             f"statute_id={cur.get('id')}")
            elif m.get("is_repealed"):
                lines.append("  폐지된 법령 — 현행 후속본이 없습니다")
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
        # Three time-axis lines. The word "폐지" belongs to `repealed` alone:
        # folding a rename into it makes a model read a live law as gone and
        # reason from that instead.
        asked = resp.get("asked_name")
        if asked:
            eff = _fmt_eff_iso(asked.get("effective_date"))
            lines.append(
                f"- 조회한 「{asked.get('name','')}」은 이 법의 옛 이름입니다"
                + (f"({eff} 시행본)" if eff else "")
                + " — 위 이름이 현행명이고 아래 본문도 현행입니다."
            )
        cv = resp.get("current_version")
        tail = ""
        if cv:
            ceff = _fmt_eff_iso(cv.get("effective_date"))
            tail = (f"「{cv.get('name','')}」({ceff} 시행)" if cv.get("renamed")
                    else f"{ceff} 시행본")
        od = resp.get("offense_date")
        if od:
            # A dated lookup has to say so in the header, or the caller gets
            # superseded wording with nothing marking it as such — which is
            # what happened while `offense_date` sat in the dict with no
            # branch here to print it.
            lines.append(f"- 시점 조회: {_fmt_eff_iso(od)} 기준 문언입니다"
                         + (f" — 현행은 {tail}." if tail else "."))
        elif cv:
            lines.append(f"- 현행 아님 — 이 법의 현행은 {tail}, "
                         f"statute_id={cv.get('id')} 입니다.")
        rep = resp.get("repealed")
        if rep and rep.get("note"):
            lines.append(f"- 폐지: {rep['note']}")

    if resp.get("content_format"):
        lines.append(f"## content_format: {resp['content_format']}")
    if resp.get("note"):
        lines.append(f"## note: {resp['note']}")

    if mode == "outline":
        if resp.get("as_of"):
            lines.append(f"## as_of: {resp['as_of']} (현행 — 오늘 시점 시행본)")
        arts = resp.get("articles", [])
        lines.append(f"## articles ({len(arts)})")
        for a in arts:
            lines.append(f"- {_fmt_article_no(a.get('no'), a.get('branch'))}: {a.get('title','')}")
    elif mode == "detail":
        arts = resp.get("articles", [])
        if arts:
            lines.append("## articles")
            for a in arts:
                num = _fmt_article_no(a.get("no"), a.get("branch"))
                # The title can be None for laws that have none. A default
                # in .get() would not help: the key exists holding None, so
                # the string "None" would be printed.
                lines.append(f"### {num} {a.get('title') or ''}")
                art_url = _statute_article_url(stt, a)
                if art_url:
                    lines.append(f"- url: {art_url}")
                if resp.get("offense_date") and a.get("eff_date"):
                    # Articles of one law can take effect on different dates,
                    # so a dated lookup marks each. Only there: on a current
                    # lookup this line multiplies by the article count.
                    lines.append(f"- 시행: {_fmt_eff_iso(a['eff_date'])}판")
                if a.get("text"):
                    # One marker rather than two lines per article: a single
                    # lookup can carry dozens of articles, so per-line cost
                    # multiplies.
                    lines.append("- text_kind: 공식 조문 원문")
                    lines.append(a["text"])
        if resp.get("body"):  # 고시 other category
            lines.append("## body")
            lines.append("- text_kind: 공식 고시 원문")
            lines.append(resp["body"])
        missing = resp.get("missing")
        if missing:
            lines.append(f"## missing: {', '.join(str(x) for x in missing)}")
        if resp.get("hint"):
            lines.append(f"## hint: {resp['hint']}")

    return "\n".join(lines)


# ---------- public tool ----------

@dedup_guard("statute_lookup")
def statute_lookup(
    ctx: RunContext[HarnessDeps],
    query: str | None = None,
    statute_id: int | str | None = None,
    articles: list[str | int] | str | int | None = None,
    limit: int = 10,
    offense_date: str | None = None,
) -> str:
    """법령·행정규칙 조회 — 법령명/키워드 검색과 조문 본문 확인(현행 또는 행위시점 기준).

    법률·대통령령·부령·규칙과 **행정규칙(고시·훈령·예규)**을 한 번에 검색합니다. 종류를
    가리는 인자는 없습니다 — 관련도 순으로 함께 나오고 각 결과에 종류가 붙습니다.

    언제:
    - 법령의 요건·효과·기간·절차가 답의 뼈대가 되는 모든 국면 — 조문을 인용할 때만이 아니라
      기억으로 서술하려 할 때도. 조문은 개정되므로 현행 본문은 이 도구만 압니다.
    - 죄명·법조 식별 후 정확한 조문 확인. 행위 시점이 문제되면 offense_date 로 시점본 확인.
    - 조문을 확인했으면 그 요건이 실제 사건에서 어떻게 판단됐는지는 precedent_search 로 이어
      확인하세요 — 조문과 판례는 택일이 아닙니다. 양형기준은 compute_sentencing_range.

    규칙:
    - query 또는 statute_id 중 하나는 필수입니다. 둘 다 없으면 임의 법령 목록을 반환하지 않고
      missing_input으로 종료합니다.
    - **자주 쓰는 법령은 quick-access id 로 바로**(one-step): statute_id + articles 직접 호출.
      대한민국헌법 468, 민법 584, 상법 583, 민사소송법 581, 형법 578,
      형사소송법 574, 행정기본법 4953, 행정절차법 437, 행정소송법 386,
      헌법재판소법 3629.
    - **그 외는 two-step**: ① query=법령명 으로 검색해 후보 목록(id 포함)을 받고,
      ② 맞는 법령을 골라 그 statute_id + articles 로 본문 호출.
      조문 번호는 법령-종속이라(같은 '제3조'도 법마다 다름) 법령을 먼저 확정한 뒤
      본문을 받는 게 안전합니다.
    - **검색이 준 id 를 글자 그대로 되넘기세요.** 행정규칙 id 는 `admrul-18060` 처럼
      접두사가 붙어 나옵니다 — 접두사를 떼면 **같은 번호의 다른 법령**이 조회됩니다.
    - **위 quick-access 목록 밖의 statute_id 는 추측 금지** (DB primary key — 예: 574 는
      근로기준법이 아니라 형사소송법). 모르면 query 로 검색해 받은 id 를 쓰세요.
    - articles 미지정 시 outline (모든 조문 title) — 형법처럼 큰 법령은 응답이 거대해지니
      가능하면 articles 명시.

    응답: markdown-KV. 상세 조문·행정규칙 본문은 `text_kind: 공식 … 원문`으로 표시되며 그대로
    직접인용할 수 있습니다. 답에 쓴 조문은 직접 인용이든 요약이든 반환 url(법령/조문 페이지)을
    링크로 함께 제시하세요.

    Args:
      query: 법령명·행정규칙명 또는 본문 키워드. id 모를 때 검색용. 받은 목록에서 id 를 골라 재호출.
      statute_id: 검색 결과가 준 식별자를 그대로. 법령은 정수(예: 584), 행정규칙은
        `'admrul-18060'`. 한글 이름 받지 않음. 모르면 query 로 검색(추측 금지).
      articles: 조문 번호 리스트(문자열/정수 모두 허용) **최대 8개** — 넘치면 앞 8개만
        조회하고 나머지는 응답 message 로 알린다(나눠 재호출). 아래 표기 모두 허용(자동 정규화):
        - "347" / "제347조": 347조 + 가지(의2, 의3 ...) 함께
        - "347-2" / "347의2" / "제347조의2": 제347조의2만 콕
        - 연속 범위: ["3","4","5","6","7"]
        **별표·별지·서식·부칙은 받지 않습니다**(코퍼스에 본문이 없습니다) — 별표가 필요하면
        국가법령정보센터에서 해당 파일을 직접 확인하세요.
        행정규칙은 가지 없음. 조문이 분리되지 않은 행정규칙은 articles 를 아무 값으로나 주면
        본문 전문이 나옵니다(개요 응답의 note 가 그렇게 안내합니다).
      limit: 검색 모드 최대 결과 수(기본 10, 최대 50).
      offense_date: 행위 일자 (예: '2013.7.30', '20130730'). 지정 시 *행위시점 기준*
        본문/시점본 응답 — 형법 §1 ① "범죄의 성립과 처벌은 행위시의 법률에 의한다" 원칙.
        **미지정 시 오늘 날짜(시스템 시계) 기준 현행본** — 미래 시행예정본은 자동 제외.
    """
    query = coerce_str(query)
    limit = min(max(coerce_int(limit) or 10, 1), LIMIT_MAX)
    articles = coerce_list(articles)
    ref = _parse_statute_ref(statute_id)

    if ref is None and not query:
        # An unreadable statute_id and no statute_id at all are different
        # mistakes. Folding the first into missing_input leaves the caller no
        # reason not to send the same malformed id again.
        if statute_id is not None:
            return _format_response_md({
                "status": "bad_statute_id",
                "input": {"statute_id": statute_id},
                "message": (
                    "statute_id 를 읽지 못했습니다 — 법령은 정수(예: 584), "
                    "행정규칙은 'admrul-18060' 형태입니다. 검색 결과가 준 식별자를 "
                    "글자 그대로 넘기거나, query 로 다시 검색하세요."
                ),
            })
        return _format_response_md({
            "status": "missing_input",
            "message": (
                "query(법령명·키워드) 또는 statute_id(검색이 준 식별자) 중 하나는 필요합니다. "
                "법령을 모르면 query로 검색한 뒤 받은 statute_id로 재호출하세요."
            ),
        })

    # Over the cap, answer the first ARTICLES_MAX and say what was deferred
    # rather than refusing outright. A refusal makes the caller rewrite the
    # whole call: measured over two days of traffic, nineteen requests of nine
    # to fifteen articles each came back with nothing at all.
    dropped_articles: list = []
    if articles is not None and len(articles) > ARTICLES_MAX:
        dropped_articles = list(articles[ARTICLES_MAX:])
        articles = list(articles[:ARTICLES_MAX])

    conn = open_db()
    try:
        resp = _statute_lookup_impl(
            conn, query, ref, articles, limit, offense_date=offense_date,
        )
        if dropped_articles:
            preview = ", ".join(str(a) for a in dropped_articles[:12])
            if len(dropped_articles) > 12:
                preview += f" 외 {len(dropped_articles) - 12}개"
            note = (
                f"articles {ARTICLES_MAX + len(dropped_articles)}개 중 앞 "
                f"{ARTICLES_MAX}개만 조회했습니다(호출당 최대 {ARTICLES_MAX}개). "
                f"나머지({preview})는 같은 statute_id 로 나눠 재호출하세요."
            )
            prev = resp.get("message")
            resp["message"] = note if not prev else f"{note} · {prev}"
        # A mixed call (`["13","별표5"]`) where the articles read and only the
        # table was dropped. Articles coming back look like a complete answer,
        # and the caller then invents the rest, so the dropped tokens are
        # echoed by name. When nothing parsed at all, `_bad_articles_response`
        # has already said the same thing.
        unsupported = _unsupported_units(articles) if resp.get("status") == "ok" else []
        if unsupported:
            note = f"{', '.join(unsupported)} 는 제외했습니다. {_UNSUPPORTED_UNIT_HINT}"
            prev = resp.get("message")
            resp["message"] = note if not prev else f"{prev} · {note}"
        return _format_response_md(resp)
    finally:
        conn.close()


def _unsupported_units(articles: list[str | int] | None) -> list[str]:
    """Tokens in `articles` naming a structural unit rather than an article,
    echoed back verbatim so the caller sees which of its own words was cut."""
    return [str(a) for a in (articles or []) if _NON_ARTICLE_UNIT_RE.search(str(a))]


# Tables and forms are not in the corpus — collection skips them, because the
# ministry serves them as separate files rather than as article text. A caller
# that does not know this retries the same lookup in new spellings, so the
# response says where the text does live instead of only refusing.
_UNSUPPORTED_UNIT_HINT = (
    "별표·별지·서식·부칙은 조문 조회 대상이 아니고 본문이 코퍼스에 없습니다 — "
    "국가법령정보센터에서 해당 파일을 직접 확인하세요."
)


def _bad_articles_response(statute_id: int | str | None, articles: list[str | int]) -> dict[str, Any]:
    """Format hint for when nothing in `articles` could be parsed."""
    message = (
        "articles 파싱 실패 — 조문 번호를 숫자로 주세요. "
        "'76'(76조 본조+가지), '76-2'(제76조의2만). "
        "예: articles=['76-2','76-3']"
    )
    if _unsupported_units(articles):
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
    """``ref`` is `_parse_statute_ref`'s result — ``(is_rule, id)`` or
    ``("law", law_id)``. None selects search mode."""
    if ref is None:
        # No kind filter, and no fixed share between laws and rules: take up
        # to the limit from each and interleave by name relevance. A rule
        # whose name matches rises above the laws; one that merely mentions
        # the words sorts below them, so the caller need not pick a kind up
        # front.
        stat_matches = _search_statutes(
            conn, query, limit, offense_date=offense_date,
        )
        notice_matches = _search_notices(conn, query, limit) if query else []
        matches = _merge_law_notice_matches(stat_matches + notice_matches, query, limit)
        out = {
            "status": "ok", "mode": "list",
            "matches": matches,
            "offense_date": to_iso_date(offense_date),
        }
        # Article numbers mean nothing without a law — the same number is a
        # different provision in every act. Rather than ignore them, say so
        # and describe the two-step call.
        if articles:
            out["note"] = (
                "본문 조회는 statute_id 가 필요합니다 — 아래 matches 에서 맞는 법령의 "
                "id 를 골라 statute_id+articles 로 재호출하세요."
            )
        return out

    if ref[0] == "law":
        # A law_id off a web path. Resolve it to the current version's id and
        # carry on; the detail flow below re-picks a dated version when
        # offense_date calls for one.
        resolved = resolve_web_law_id(conn, ref[1], _today_iso())
        if resolved is None:
            return {
                "status": "not_found",
                "input": {"law_id": ref[1]},
                "message": (
                    "웹 경로의 법령 id(law_id)를 코퍼스에서 찾지 못했습니다 — "
                    "query 로 법령명을 검색해 statute_id 를 받아 재호출하세요."
                ),
            }
        ref = (False, resolved[0])

    is_notice, sid = ref
    shown_id = f"{NOTICE_URL_PREFIX}{sid}" if is_notice else sid
    specs = _parse_articles(articles)

    # If nothing in `articles` parsed, say what the format is. An empty
    # result gives the caller no clue why, so it retries with variations of
    # the same unparseable input.
    if articles and not specs:
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
    return {"status": "ok", **res}
