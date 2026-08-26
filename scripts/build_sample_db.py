#!/usr/bin/env python3
"""Build a bounded sample corpus from a full legal corpus database.

The full corpus behind the hosted service is not distributed with this
repository. This script carves a self-consistent subset that keeps every
table the MCP tools read, so a clone of the repo can run and be tested
against real data. It is also the reference for the corpus schema if you
want to build your own database from public sources.

Three tiers:
  1. Reference tables copied in full (small): sentencing guidelines
     (``sg_*``), charge taxonomy (``charge_*``), statutory-penalty history
     (``clm_versions``).
  2. Statutes: a core-law whitelist or everything (``--statutes``). Version
     rows are cheap metadata and always come along for whitelisted laws;
     article bodies are the weight. ``core-current`` ships only each law's
     consolidated current snapshot (picked the same way the lookup tool
     does), ``core`` ships every version's articles, ``all`` ships the lot.
  3. Precedents: a bounded selection (``--cases N``). Cases that carry the
     sentencing sample (``prec_defendants``) are preferred so
     ``sentence_statistics`` stays meaningful, then cases carrying the
     court's own issue summary (``holdings``), then recent decisions.

Full-text indexes:
  - The trigram FTS tables are external-content FTS5: base rows are copied,
    then each index is rebuilt with the ``'rebuild'`` special command.
  - ``prec_cases_morph_fts`` is contentless: it is re-tokenized with Kiwi
    using the same recipe as the production indexer (content morphemes
    only: N*, SL, SN, SH, XR and VV/VA lemmas, whitespace-joined).

Usage:
    python build_sample_db.py --source /path/to/harness.db --dest fixture.db \
        --cases 500 --statutes core --notices 20

Requires: Python 3.11+, kiwipiepy (for the morpheme index).
"""
from __future__ import annotations

import argparse
import functools
import os
import sqlite3
import time

# --- what the tools read -----------------------------------------------------

# Copied in full: small reference data.
FULL_TABLES = [
    "sg_categories",
    "sg_subtypes",
    "sg_factors",
    "sg_probation_factors",
    "sg_fine_conditions",
    "sg_ranges",
    "charge_taxonomy",
    "charge_family",
    "charge_legal_map",
    "clm_versions",
]

# Statute whitelist for --statutes core. Matched against both ``name`` and
# ``short_name``; every historical version of a matched law is kept.
CORE_STATUTES = [
    "대한민국헌법",
    "형법",
    "형사소송법",
    "민법",
    "민사소송법",
    "민사집행법",
    "상법",
    "행정소송법",
    "행정심판법",
    "근로기준법",
    "노동조합 및 노동관계조정법",
    "최저임금법",
    "임금채권보장법",
    "주택임대차보호법",
    "상가건물 임대차보호법",
    "도로교통법",
    "교통사고처리 특례법",
    "특정범죄 가중처벌 등에 관한 법률",
    "특정경제범죄 가중처벌 등에 관한 법률",
    "폭력행위 등 처벌에 관한 법률",
    "성폭력범죄의 처벌 등에 관한 특례법",
    "아동ㆍ청소년의 성보호에 관한 법률",
    "마약류 관리에 관한 법률",
    "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
    "개인정보 보호법",
    "부정수표 단속법",
]

# Tokenizer recipe for the contentless morpheme index. Mirrors the production
# indexer: content morphemes only, verbs/adjectives as lemma form.
MORPH_KEEP_TAGS = ("NNG", "NNP", "NNB", "NR", "NP", "SL", "SN", "SH", "XR", "VV", "VA")
MORPH_SOURCE_COLUMNS = (
    "case_name",
    "content_md",
    "summary",
    "reference_statute",
    "generated_summary",
)


@functools.lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def morph_tokens(text: str) -> str:
    """Text -> whitespace-joined content morphemes (index-side recipe)."""
    if not text:
        return ""
    out = []
    for t in _kiwi().tokenize(text):
        if t.tag in MORPH_KEEP_TAGS:
            f = t.form.strip()
            if f:
                out.append(f)
    return " ".join(out)


# --- schema helpers (source tables visible as src.<name> after ATTACH) --------


def copy_ddl(dest: sqlite3.Connection, table: str) -> None:
    row = dest.execute(
        "SELECT sql FROM src.sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if row is None or not row[0]:
        raise SystemExit(f"source has no table {table!r}")
    dest.execute(row[0])


def copy_indexes(dest: sqlite3.Connection, table: str) -> None:
    for (sql,) in dest.execute(
        "SELECT sql FROM src.sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL",
        (table,),
    ).fetchall():
        dest.execute(sql)


def count(dest: sqlite3.Connection, table: str) -> int:
    return dest.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]


def pick_current_snapshot(dest: sqlite3.Connection, law_id: str, as_of_iso: str) -> int | None:
    """Pick the consolidated current version of a law — mirror of the tool.

    Version rows carry delta article sets, so status labels alone cannot
    tell a consolidated snapshot from the latest amendment. The tool picks,
    among in-force, non-repealed, current/NULL-status rows, the one with the
    most articles (then full-snapshot origin, then latest effective date);
    this must select the same row.
    """
    group = dest.execute(
        "SELECT id, effective_date, history_status, change_kind, mst "
        "FROM src.st_statutes WHERE law_id=?",
        (law_id,),
    ).fetchall()
    if not group:
        return None
    in_force = [r for r in group if (r[1] or "") <= as_of_iso] or list(group)
    non_repealed = [r for r in in_force if not (r[3] and "폐지" in str(r[3]))] or in_force
    pool = [r for r in non_repealed if r[2] in (None, "현행")] or non_repealed
    if len(pool) == 1:
        return pool[0][0]
    ids = [r[0] for r in pool]
    counts = dict.fromkeys(ids, 0)
    q = (
        "SELECT statute_id, COUNT(*) FROM src.st_articles WHERE statute_id IN (%s) "
        "GROUP BY statute_id" % ",".join("?" * len(ids))
    )
    for sid, c in dest.execute(q, ids):
        counts[sid] = c
    best = max(pool, key=lambda r: (counts[r[0]], r[4] is None, r[1] or ""))
    return best[0]


# --- build --------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=os.environ.get("HARNESS_DB"), help="full corpus db")
    ap.add_argument("--dest", required=True, help="sample db to create")
    ap.add_argument("--cases", type=int, default=500, help="max precedent cases")
    ap.add_argument(
        "--linked-share",
        type=float,
        default=0.6,
        help="share of --cases reserved for sentencing-sample cases (0..1)",
    )
    ap.add_argument(
        "--holdings-share",
        type=float,
        default=0.05,
        help="share of --cases reserved for cases carrying the court's own "
        "issue summary (0..1)",
    )
    ap.add_argument(
        "--statutes",
        choices=["core-current", "core", "all"],
        default="core-current",
        help="core-current: whitelist laws, current-snapshot articles only; "
        "core: whitelist laws with every version's articles; all: everything",
    )
    ap.add_argument(
        "--notices", type=int, default=20,
        help="recent current-edition notices to keep",
    )
    ap.add_argument("--force", action="store_true", help="overwrite dest")
    args = ap.parse_args()

    if not args.source:
        raise SystemExit("--source (or HARNESS_DB) is required")
    if os.path.exists(args.dest):
        if not args.force:
            raise SystemExit(f"{args.dest} exists (use --force)")
        os.remove(args.dest)

    t0 = time.time()
    # uri=True so the read-only file: URI in ATTACH is honored.
    dest = sqlite3.connect(args.dest, uri=True)
    dest.execute("PRAGMA journal_mode=OFF")
    dest.execute("PRAGMA synchronous=OFF")
    dest.execute("PRAGMA foreign_keys=OFF")
    dest.execute(f"ATTACH DATABASE 'file:{args.source}?mode=ro' AS src")

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    # 1. reference tables, complete.
    for t in FULL_TABLES:
        copy_ddl(dest, t)
        dest.execute(f"INSERT INTO main.{t} SELECT * FROM src.{t}")
        copy_indexes(dest, t)
    log(f"reference tables: {', '.join(f'{t}={count(dest, t)}' for t in FULL_TABLES[:3])}, ...")

    # 2. statutes (+ all historical versions of selected laws), notices.
    for t in ("st_statutes", "st_articles", "st_notices", "st_notice_articles"):
        copy_ddl(dest, t)
    # Effective dates are stored compact (YYYYMMDD); compare in kind.
    as_of = time.strftime("%Y%m%d")
    snap_ids: list[int] = []
    if args.statutes == "all":
        dest.execute("INSERT INTO main.st_statutes SELECT * FROM src.st_statutes")
    else:
        marks = ",".join("?" for _ in CORE_STATUTES)
        law_ids = [
            r[0]
            for r in dest.execute(
                f"SELECT DISTINCT law_id FROM src.st_statutes "
                f"WHERE name IN ({marks}) OR short_name IN ({marks})",
                [*CORE_STATUTES, *CORE_STATUTES],
            ).fetchall()
        ]
        # Every version row of a selected law comes along: the rows are cheap
        # metadata and old-law resolution walks the version timeline.
        lmarks = ",".join("?" for _ in law_ids)
        dest.execute(
            f"INSERT INTO main.st_statutes SELECT * FROM src.st_statutes "
            f"WHERE law_id IN ({lmarks})",
            law_ids,
        )
        snap_ids = [
            s for s in (pick_current_snapshot(dest, lid, as_of) for lid in law_ids) if s
        ]
    # Statutes referenced by the charge->penalty map must exist for FK
    # integrity. Penalty bounds live in charge_legal_map itself, so these
    # closure laws keep their row; article bodies only come along outside
    # core-current mode.
    dest.execute(
        "INSERT INTO main.st_statutes SELECT * FROM src.st_statutes "
        "WHERE id IN (SELECT statute_id FROM main.charge_legal_map "
        "             WHERE statute_id IS NOT NULL) "
        "AND id NOT IN (SELECT id FROM main.st_statutes)"
    )
    if args.statutes == "core-current":
        # Version rows carry delta article sets (changed articles only); the
        # full text of a law as it stands lives in its consolidated snapshot.
        # Ship only that snapshot per law — the pick mirrors the tool.
        smarks = ",".join("?" for _ in snap_ids)
        dest.execute(
            f"INSERT INTO main.st_articles SELECT * FROM src.st_articles "
            f"WHERE statute_id IN ({smarks})",
            snap_ids,
        )
    else:
        dest.execute(
            "INSERT INTO main.st_articles SELECT * FROM src.st_articles "
            "WHERE statute_id IN (SELECT id FROM main.st_statutes)"
        )
    # Only rules the tool would actually surface. An administrative rule is
    # re-issued as a *new row* on amendment, so the corpus holds superseded
    # editions ('구판') and future ones ('시행예정') alongside the current text;
    # ``statute_lookup`` searches the current edition with a readable body, and
    # a sample drawn without that filter fills up with rows no search can reach.
    dest.execute(
        "INSERT INTO main.st_notices SELECT * FROM src.st_notices "
        "WHERE has_text_content=1 AND COALESCE(history_status,'현행')='현행' "
        "ORDER BY issued_date DESC, id DESC LIMIT ?",
        (args.notices,),
    )
    dest.execute(
        "INSERT INTO main.st_notice_articles SELECT * FROM src.st_notice_articles "
        "WHERE notice_id IN (SELECT id FROM main.st_notices)"
    )
    for t in ("st_statutes", "st_articles", "st_notices", "st_notice_articles"):
        copy_indexes(dest, t)
    log(
        f"statutes: {count(dest, 'st_statutes')} laws, {count(dest, 'st_articles')} articles, "
        f"{count(dest, 'st_notices')} notices"
    )

    # 3. precedent selection: sentencing-sample cases first, then recent.
    linked_cap = int(args.cases * args.linked_share)
    dest.execute("CREATE TEMP TABLE pick(id INTEGER PRIMARY KEY)")
    dest.execute(
        "INSERT OR IGNORE INTO pick "
        "SELECT DISTINCT case_id FROM src.prec_defendants ORDER BY case_id DESC LIMIT ?",
        (linked_cap,),
    )
    n_linked = dest.execute("SELECT COUNT(*) FROM pick").fetchone()[0]
    # Cases carrying the court's own issue summary (``holdings``), which
    # precedent_search prints for every result it returns. These need a quota
    # of their own because neither other tier reaches them. Only the Supreme
    # Court and the Constitutional Court publish a holding, and both are a
    # thin slice of any recent window -- of the sampled decisions from the
    # last few years that carry none, 126 are Constitutional Court and 69 are
    # trial court against 9 Supreme Court -- so the recency tier below fills
    # up before it reaches them. The sentencing tier is trial-court criminal
    # judgments, which carry none at all. Without this quota the sample holds
    # not one and the holding path never runs against real data.
    #
    # Ordered by the normalised year, not by ``decision_date``: judgments from
    # the 1950s carry a Dangi-calendar date, and '4289-06-19' sorts above
    # '2026-01-01' as a string, so ordering on the raw column fills the quota
    # with the oldest cases in the corpus.
    holdings_cap = int(args.cases * args.holdings_share)
    dest.execute(
        "INSERT OR IGNORE INTO pick "
        "SELECT id FROM src.prec_cases "
        "WHERE holdings IS NOT NULL AND TRIM(holdings) <> '' "
        "  AND id NOT IN (SELECT id FROM pick) "
        "ORDER BY decision_year DESC, decision_date DESC, id DESC LIMIT ?",
        (holdings_cap,),
    )
    n_holdings = dest.execute("SELECT COUNT(*) FROM pick").fetchone()[0] - n_linked
    # Restrict the recency scan to the last few years so the indexed year
    # column narrows it; decision_date itself carries no index.
    recent_floor = time.localtime().tm_year - 4
    dest.execute(
        "INSERT OR IGNORE INTO pick "
        "SELECT id FROM src.prec_cases "
        "WHERE decision_year >= ? AND id NOT IN (SELECT id FROM pick) "
        "ORDER BY decision_date DESC, id DESC LIMIT ?",
        (recent_floor, max(args.cases - n_linked - n_holdings, 0)),
    )
    log(
        f"case selection: {n_linked} sentencing-linked + {n_holdings} with holdings + "
        f"{dest.execute('SELECT COUNT(*) FROM pick').fetchone()[0] - n_linked - n_holdings}"
        " recent"
    )

    copy_ddl(dest, "prec_cases")
    dest.execute(
        "INSERT INTO main.prec_cases SELECT * FROM src.prec_cases "
        "WHERE id IN (SELECT id FROM pick)"
    )
    copy_indexes(dest, "prec_cases")
    for t in ("prec_defendants", "prec_defendant_charges", "prec_sentences"):
        copy_ddl(dest, t)
        dest.execute(
            f"INSERT INTO main.{t} SELECT * FROM src.{t} "
            "WHERE case_id IN (SELECT id FROM pick)"
        )
        copy_indexes(dest, t)
    log(
        f"prec_cases: {count(dest, 'prec_cases')} rows "
        f"(+{count(dest, 'prec_defendants')} defendants, "
        f"{count(dest, 'prec_sentences')} dispositions)"
    )

    # 4. full-text indexes.
    for fts in ("prec_cases_fts", "st_articles_fts", "st_notice_articles_fts"):
        copy_ddl(dest, fts)
        dest.execute(f"INSERT INTO main.{fts}({fts}) VALUES('rebuild')")
        log(f"{fts}: rebuilt from content table")

    copy_ddl(dest, "prec_cases_morph_fts")
    cols = ", ".join(f"COALESCE({c}, '')" for c in MORPH_SOURCE_COLUMNS)
    read = dest.cursor()
    n_morph = 0
    for row in read.execute(f"SELECT id, {cols} FROM main.prec_cases"):
        text = "\n".join(part for part in row[1:] if part)
        dest.execute(
            "INSERT INTO main.prec_cases_morph_fts(rowid, morph) VALUES(?, ?)",
            (row[0], morph_tokens(text)),
        )
        n_morph += 1
    log(f"prec_cases_morph_fts: tokenized {n_morph} cases with Kiwi")

    # 5. finalize: integrity, stats, compact.
    violations = dest.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise SystemExit(f"foreign key violations in sample: {violations[:5]}")
    dest.execute(
        "INSERT INTO main.prec_cases_morph_fts(prec_cases_morph_fts) VALUES('integrity-check')"
    )
    dest.commit()
    dest.execute("ANALYZE main")
    dest.execute("DETACH DATABASE src")
    dest.execute("VACUUM")
    dest.commit()

    log(f"done: {args.dest} = {os.path.getsize(args.dest) / 1048576:.1f} MB")
    names = [
        r[0]
        for r in dest.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%_data' "
            "AND name NOT LIKE '%_idx' AND name NOT LIKE '%_content' "
            "AND name NOT LIKE '%_docsize' AND name NOT LIKE '%_config' ORDER BY name"
        )
    ]
    for name in names:
        try:
            print(f"  {name}: {dest.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]}")
        except sqlite3.Error:
            pass
    dest.close()


if __name__ == "__main__":
    main()
