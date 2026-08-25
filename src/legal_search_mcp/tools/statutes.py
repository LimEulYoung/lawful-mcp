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


def _parse_articles(
    spec: list[str | int] | None,
) -> list[tuple[int, int | None]] | None:
    """Parse article references into (number, branch) pairs.

    An article number alone selects the article and every branch under it;
    a number with a branch selects just that branch. A range is passed as a
    list of numbers rather than as a range expression.

    Unparseable tokens are skipped. If that leaves nothing at all, the
    caller returns a format hint rather than silently searching for nothing.
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
        sql = f"""
        WITH fts_hits AS (
          SELECT a.statute_id, COUNT(*) AS hits
          FROM st_articles_fts f
          JOIN st_articles a ON a.id = f.rowid
          WHERE st_articles_fts MATCH ?
          GROUP BY a.statute_id
        )
        SELECT s.id, s.law_id, s.name, s.kind, s.issuing_agency,
               s.effective_date, s.history_status, s.change_kind, s.mst,
               COALESCE(h.hits, 0) AS fts_hits,
               ({cover_sql}) AS name_cover,
               (CASE WHEN {name_norm} = ? THEN 1 ELSE 0 END) AS exact_name
        FROM st_statutes s
        LEFT JOIN fts_hits h ON h.statute_id = s.id
        """
        safe_q = safe_fts_query(query)
        fts_query_str = safe_q if _fts_query_ok(safe_q) else "x" * 1000
        # A candidate matches on the name or in the text, not necessarily both.
        where_parts.append(f"(({name_or}) OR h.hits > 0)")
        sql += "WHERE " + " AND ".join(where_parts) + "\n"
        # Rank by exact name, then how many query tokens the name covers,
        # then name length, then text hits. Compound names rise through
        # coverage; everyday names through the alias table and exact match.
        sql += """
        ORDER BY exact_name DESC, name_cover DESC, length(s.name) ASC, fts_hits DESC
        LIMIT ?
        """
        # Parameter order follows the order they appear in the SQL:
        #               fts MATCH, cover/exact (SELECT), name_or (WHERE), LIMIT
        params_q = [fts_query_str, *like_params, normalized]
        rows = conn.execute(sql, [*params_q, *params, *like_params, sql_limit]).fetchall()
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

    out = []
    for r in rows:
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
        preview = (prev["article_text"][:200] if prev else "").replace("\n", " ")
        out.append(
            {
                "statute_id": r["id"],
                "law_id": r["law_id"],
                "name": r["name"],
                "kind": r["kind"],
                "agency": r["issuing_agency"],
                "effective_date": r["effective_date"],
                "history_status": r["history_status"] if "history_status" in r.keys() else None,
                "change_kind": r["change_kind"] if "change_kind" in r.keys() else None,
                "preview": preview,
            }
        )
    return out


def _search_notices(
    conn: sqlite3.Connection,
    query: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    if query:
        # Same token-coverage ranking as the statute search. Names carry
        # particles and spacing that a substring match cannot survive — one
        # intervening 의 was enough to break it — so rank by how many of the
        # query's tokens the name contains.
        normalized, tokens = _statute_name_tokens(query)
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
               (CASE WHEN {name_norm} = ? THEN 1 ELSE 0 END) AS exact_name
        FROM st_notices n
        LEFT JOIN fts_hits h ON h.notice_id = n.id
        WHERE n.has_text_content = 1
          AND {_NOTICE_CURRENT}
          AND (({name_or}) OR h.hits > 0)
        ORDER BY exact_name DESC, name_cover DESC, length(n.name) ASC, fts_hits DESC
        LIMIT ?
        """
        rows = conn.execute(
            sql,
            [fts_query_str, *like_params, normalized, *like_params, limit],
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

    out = []
    for r in rows:
        prev = conn.execute(
            "SELECT article_text FROM st_notice_articles WHERE notice_id=? ORDER BY article_seq LIMIT 1",
            (r["id"],),
        ).fetchone()
        preview = (prev["article_text"][:200] if prev else "").replace("\n", " ")
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


def _name_cover_key(name: str, q_norm: str, tokens: list[str]):
    """Name-relevance sort key, lower first, shared by laws and rules.

    Having one measure lets the two be interleaved on merit rather than by
    giving each a fixed share of the results.
    """
    n = (name or "").replace(" ", "")
    if q_norm and n == q_norm:
        return (0, 0)                          # exact match sorts first
    if tokens:
        cover = sum(1 for t in tokens if t in n)
        return (1, len(tokens) - cover)        # fewer missing tokens sorts higher
    return (1, 0) if (q_norm and q_norm in n) else (2, 0)


def _merge_law_notice_matches(matches, query, limit, name_of=None):
    """Interleave law and rule results by name relevance, with no fixed share.

    The sort is stable, so within one relevance tier the input order
    survives.
    """
    if name_of is None:
        name_of = lambda m: m.get("name", "")
    normalized, tokens = _statute_name_tokens(query) if query else ("", [])
    return sorted(matches, key=lambda m: _name_cover_key(name_of(m), normalized, tokens))[:limit]


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
        return {
            'mode': 'outline',
            'statute': {
                'id': meta['id'], 'law_id': meta['law_id'],
                'name': meta['name'], 'kind': meta['kind'],
                'agency': meta['issuing_agency'],
            },
            'offense_date': offense_iso,
            'articles': articles,
        }

    # Redirect to today's consolidated snapshot even when the caller passed
    # the id of an amendment row, so an outline is the whole law rather than
    # the handful of articles that amendment touched.
    as_of = _today_iso()
    if meta.get('law_id'):
        cur_id = _resolve_current_statute_id(conn, meta['law_id'], as_of)
        if cur_id is not None and cur_id != sid:
            sid = cur_id
            meta = _statute_meta(conn, sid) or meta

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
        return {
            "mode": "detail",
            "statute": statute_kv,
            "articles": articles,
            "missing": [],
            "note": "제N조 구조가 없는 단일 본문 규정 — 아래가 본문 전문입니다.",
        }

    # Ordinary laws: outline is titles only, no article text.
    for a in articles:
        a.pop("text", None)
    return {
        "mode": "outline",
        "as_of": as_of,
        "statute": statute_kv,
        "articles": articles,
    }


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
    """specs: [(num, None), (347, 2), ...]
      - (num, None): article_no_num = num — 본조 + 모든 가지
      - (num, branch): article_no_num = num AND article_branch = branch — 가지 콕 짚음

    offense_date 지정 시 (M28 연혁 적재 활용):
      - statute_id 의 law_id 추출
      - 각 spec 의 *행위시점 이전 최후 변경* 조문 본문 응답
      - 도구가 *시점 정확 본문* 보장. 변경분 적재 안 된 시점은 *이전 변경분* 사용
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
        if as_of is None:
            cur = conn.execute(
                "SELECT MAX(effective_date) AS d FROM st_statutes "
                "WHERE law_id=? AND COALESCE(history_status,'') != '시행예정'",
                (law_id,),
            ).fetchone()
            as_of = cur["d"] if cur and cur["d"] else None
        if as_of:
            return _detail_statute_at_date(conn, meta, specs, as_of)

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
) -> dict[str, Any]:
    """Article text as of a date, walking back to each article's last change.

    Handles an article and its branches alike.
    """
    law_id = meta['law_id']
    # Check whether the law was repealed by then.
    repealed = _is_repealed_at(conn, law_id, offense_iso)
    # Metadata from the latest version, for the time-axis display.
    latest = conn.execute(
        """SELECT effective_date, history_status FROM st_statutes
           WHERE law_id=? ORDER BY effective_date DESC LIMIT 1""",
        (law_id,),
    ).fetchone()

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
        "offense_date": offense_iso,
        "articles": articles,
        "missing": missing,
    }
    if repealed:
        out["repealed"] = {
            "effective_date": repealed['effective_date'],
            "change_kind": repealed['change_kind'],
            "note": f"이 법령은 {repealed['effective_date']} {repealed['change_kind']}. "
                    f"행위시점 {offense_iso} 에는 유효였을 수 있음 (확인 필요)."
        }
    if latest and latest['effective_date'] != articles[0]['eff_date'] if articles else False:
        out["latest_version"] = {
            "effective_date": latest['effective_date'],
            "history_status": latest['history_status'],
        }
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
    """Display form of an article number, with its branch if it has one."""
    return f"{no}-{branch}" if branch else f"{no}"


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
            lines.append(head)
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
        return _format_response_md(resp)
    finally:
        conn.close()


def _bad_articles_response(statute_id: int | str | None, articles: list[str | int]) -> dict[str, Any]:
    """Format hint for when nothing in `articles` could be parsed."""
    return {
        "status": "bad_articles",
        "input": {"statute_id": statute_id, "articles": articles},
        "message": (
            "articles 파싱 실패 — 조문 번호를 숫자로 주세요. "
            "'76'(76조 본조+가지), '76-2'(제76조의2만). "
            "예: articles=['76-2','76-3']"
        ),
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
