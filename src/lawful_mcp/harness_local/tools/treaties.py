"""Tax treaties for statute_search and statute_read — a third corpus beside statutes and administrative rules.

Only tax treaties are loaded (double-taxation conventions, tax-information-exchange agreements, their amending
instruments and the multilateral conventions). This module reads `st_treaties`, `st_treaty_articles`,
`st_treaty_articles_fts` and `st_treaty_country_names`.

**No in-force status exists in the source.** The national law portal and the foreign ministry list every treaty
ever concluded; amendments are separate instruments (protocols, exchanges of notes) and there is no consolidated
text. So a lookup shows the treaty's own text and, beside it, every document with the same partner by date, and
never says which one is current (grounding first — a fact the source lacks is not made up here).
"""
from __future__ import annotations

import re
import sqlite3
from typing import Any

from ._fts import query_tokens, quoted_and

TREATY_URL_PREFIX = "trty-"   # web law_id namespace: tax treaty = 'trty-{st_treaties.id}' (the portal's serial)
BODY_MAX = 10_000              # a document not split into articles is returned whole up to this length

# A query is about a tax treaty only with a treaty word and a tax word (or '이중과세'). A country name alone
# ('베트남 투자') or a trade agreement must not pull tax treaties into a statute search. The tax words are the income-tax
# ones only: '과세'·'세율' sit inside customs terms ('협정관세율', '과세가격'), which put the Korea-US tax convention
# on top of a Korea-US FTA search.
_TREATY_WORD = re.compile(r"조약|협약|협정")
_TAX_WORD = re.compile(r"이중과세|조세|소득세|법인세|원천징수|BEPS|MLI", re.IGNORECASE)
# An article number ('제4조') names no treaty: searched as text it matches every treaty and ranks an arbitrary one first.
_ARTICLE_TOKEN = re.compile(r"(?:제)?\d+조(?:의\d+)?")
# Separators people put between the two partners: '한-미', '한·베', '대한민국-베트남'.
_SEP = re.compile(r"[\s\-‐–—·ㆍ‧・/]")
_KOREA = ("대한민국", "한국", "한")


def about_tax_treaty(text: str | None) -> bool:
    t = text or ""
    return "이중과세" in t or bool(_TREATY_WORD.search(t) and _TAX_WORD.search(t))


def parse_ref(s: str) -> int | None:
    """'trty-1307' (any case, spaces allowed) → 1307. None if not a treaty identifier."""
    low = (s or "").lower().replace(" ", "")
    if not low.startswith(TREATY_URL_PREFIX):
        return None
    rest = low[len(TREATY_URL_PREFIX):]
    return int(rest) if rest.isdigit() else None


def countries_in(conn: sqlite3.Connection, text: str | None) -> list[str]:
    """Partner countries named in the text, in order of appearance. Longer names win ('인도네시아' over '인도');
    a one-letter name ('미', '중') counts only right after Korea starting a word ('한미', '한-중' — not '위한 독립').
    """
    flat, starts = "", set()   # the text without separators; where its words began
    for i, ch in enumerate(text or ""):
        if _SEP.fullmatch(ch):
            continue
        if i == 0 or not "가" <= text[i - 1] <= "힣":
            starts.add(len(flat))
        flat += ch
    names = conn.execute("SELECT name, country FROM st_treaty_country_names").fetchall()
    taken: list[tuple[int, int, str]] = []
    for name, country in sorted(names, key=lambda r: -len(r[0])):
        pattern = (f"(?:{'|'.join(_KOREA)})({re.escape(name)})" if len(name) == 1 else f"({re.escape(name)})")
        for m in re.finditer(pattern, flat):
            if len(name) == 1 and m.start() not in starts:
                continue
            a, b = m.span(1)
            if all(b <= s or a >= e for s, e, _ in taken):
                taken.append((a, b, country))
    out: list[str] = []
    for _, _, country in sorted(taken):
        if country not in out:
            out.append(country)
    return out


def _clip(text: str | None, n: int = 160) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    return t if len(t) <= n else t[:n].rstrip() + "…"


def _kind(role: str) -> str:
    return "조세조약" if role == "협약" else "조세조약 개정 문서"


def _fmt_date(d: str | None) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if d and len(d) == 8 else (d or "")


def _match(row, *, named: bool, hits: int = 0) -> dict[str, Any]:
    return {
        "treaty_id": row["id"],
        "name": row["name"],
        # The kind line carries the entry-into-force date: the list's '시행' (in force) label is a statute notion.
        "kind": f"{_kind(row['role'])} · {_fmt_date(row['effective_date'])} 발효",
        # A partner named together with a tax-treaty word is as strong as a statute name found in the query.
        "name_in_query": 1 if named else 0,
        "fts_hits": hits,
    }


def search(conn: sqlite3.Connection, query: str | None, limit: int) -> list[dict[str, Any]]:
    """Tax-treaty matches for a statute search, in the merge's match shape. Empty unless the query is about a tax
    treaty. A named partner lists that partner's documents (treaties first, newest first); otherwise the remaining
    words search the treaty texts."""
    if not query or not about_tax_treaty(query):
        return []
    countries = countries_in(conn, query)
    if countries:
        marks = ",".join("?" * len(countries))
        rows = conn.execute(
            f"SELECT * FROM st_treaties WHERE country IN ({marks}) "
            "ORDER BY role != '협약', effective_date DESC LIMIT ?", [*countries, limit]).fetchall()
        return [_match(r, named=True) for r in rows]
    words = [t for t in query_tokens(query)
             if len(t) >= 3 and not about_tax_treaty(t) and not _ARTICLE_TOKEN.fullmatch(t)]
    if not words:
        return []
    hits = conn.execute(
        """SELECT a.treaty_id, COUNT(*) AS n
           FROM st_treaty_articles_fts f JOIN st_treaty_articles a ON a.id = f.rowid
           WHERE st_treaty_articles_fts MATCH ? GROUP BY a.treaty_id ORDER BY n DESC LIMIT ?""",
        (quoted_and(words), limit)).fetchall()
    return [_match(conn.execute("SELECT * FROM st_treaties WHERE id=?", (h["treaty_id"],)).fetchone(),
                   named=False, hits=h["n"]) for h in hits]


def _label(r) -> str:
    """'4', '4의2', or the part's name ('전문', '말미')."""
    if r["article_no"] is None:
        return r["title"]
    return f"{r['article_no']}의{r['article_branch']}" if r["article_branch"] else str(r["article_no"])


def _related(conn: sqlite3.Connection, meta) -> tuple[str, list[dict[str, Any]]]:
    """Every document with the same partner and the multilateral tax conventions, by entry-into-force date."""
    rows = conn.execute(
        "SELECT * FROM st_treaties WHERE country IS ? OR country IS NULL ORDER BY country IS NULL, effective_date",
        (meta["country"],)).fetchall()
    docs = [{"ref": f"{TREATY_URL_PREFIX}{r['id']}", "date": _fmt_date(r["effective_date"]),
             "role": "다자 조세조약" if r["country"] is None else ("협약" if r["role"] == "협약" else "개정 문서"),
             "name": r["name"], "self": r["id"] == meta["id"]} for r in rows]
    title = (f"같은 상대국({meta['country']}) 조세조약 문서와 다자 조세조약 — 발효일순" if meta["country"]
             else "다자 조세조약 — 발효일순")
    return title, docs


def lookup(
    conn: sqlite3.Connection,
    tid: int,
    specs: list[tuple[int, int | None]] | None,
    articles: list[Any] | None = None,
) -> dict[str, Any] | None:
    """Outline (specs None) or the requested articles of one treaty, with its related documents. '전문' (preamble) and
    '말미'/'의정서' (signature block and the protocol signed with it) can be asked by name in `articles`.
    A document not split into articles (letters, short protocols) comes whole."""
    meta = conn.execute("SELECT * FROM st_treaties WHERE id=?", (tid,)).fetchone()
    if meta is None:
        return None
    rows = conn.execute("SELECT article_seq, article_no, article_branch, title, article_text "
                        "FROM st_treaty_articles WHERE treaty_id=? ORDER BY article_seq", (tid,)).fetchall()
    title, related = _related(conn, meta)
    resp: dict[str, Any] = {
        "statute": {"treaty_id": tid, "name": meta["name"], "kind": _kind(meta["role"]),
                    "agency": f"{_fmt_date(meta['effective_date'])} 발효"},
        "related_title": title, "related": related,
        "note": ("법제처 원문은 종료 여부를 표시하지 않고, 개정 문서는 원 조약을 고치되 통합본이 없습니다 — "
                 "사건 시점까지 발효된 개정 문서가 이 조문을 고쳤는지 아래 목록에서 함께 확인하세요."),
        "text_kind": "공식 조약 원문",
    }
    numbered = [r for r in rows if r["article_no"] is not None]
    if not numbered:
        body = "\n\n".join(r["article_text"] for r in rows)
        resp.update(mode="detail", body=body[:BODY_MAX], body_kind="공식 조약 원문")
        if meta["attachment_url"]:
            resp["attachment"] = meta["attachment_url"]
        return resp
    if specs is None and not articles:
        resp.update(mode="outline", articles=[{"no": _label(r), "title": _clip(r["article_text"], 40)} for r in rows])
        return resp
    named = {t for t in ("전문", "말미") if any(t in str(a) for a in (articles or []))}
    if any("의정서" in str(a) for a in (articles or [])):
        named.add("말미")
    picked, missing = [], []
    for num, br in specs or []:
        got = [r for r in numbered if r["article_no"] == num and (br is None or r["article_branch"] == br)]
        picked += got
        if not got:
            missing.append(num if br is None else f"{num}-{br}")
    picked += [r for r in rows if r["title"] in named]
    picked.sort(key=lambda r: r["article_seq"])
    resp.update(mode="detail", missing=missing,
                articles=[{"no": _label(r), "title": "", "text": r["article_text"]} for r in picked])
    return resp
