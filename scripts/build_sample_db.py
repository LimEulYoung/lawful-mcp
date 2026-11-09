#!/usr/bin/env python3
"""Build a bounded sample corpus from a full legal corpus database.

The full corpus behind the hosted service is not distributed with this
repository. This script carves a self-consistent subset that keeps every
table the MCP tools read, so a clone of the repo can run and be tested
against real data. It is also the reference for the corpus schema if you
want to build your own database from public sources: every table is created
from the source's own definition.

The tools read the data as written and do not repair it, so the source has to
follow the data rules the tools rely on (one row per judgment, one row per
article number in a version, an effective date on every article, and so on),
and hold every table they read: there is no fallback for a missing one, and
the build stops on it. The sample inherits the rules by copying rows
unchanged.

Three tiers:
  1. Reference tables copied in full (small): sentencing guidelines
     (``sg_*``), charge taxonomy (``charge_*``, ``official_*``),
     statutory-penalty history (``clm_versions``), statute aliases.
  2. Statutes: a core-law whitelist or everything (``--statutes``). Version
     rows are cheap metadata and come along for whitelisted laws, with the
     addenda (부칙), the annexed-table index and the lapse register of those
     laws; article bodies are the weight. ``core-current`` ships the articles
     of each law's current whole text (picked by the lookup tool's own rule)
     and of every version after it, ``core`` ships every version's articles,
     ``all`` ships the lot.
     Tax treaties: a few partners' documents plus the multilateral ones
     (``--treaty-countries``).
  3. Precedents: a bounded selection (``--cases N``). Cases that carry the
     sentencing sample (``prec_defendants``) are preferred so the actual
     sentences stay meaningful, then cases carrying the court's own issue
     summary (``holdings``), then recent decisions.

Full-text indexes:
  - The trigram FTS tables are external-content FTS5: base rows are copied,
    then each index is rebuilt with the ``'rebuild'`` special command.
  - ``prec_cases_morph_fts`` is contentless: it is re-tokenized with Kiwi
    using the same recipe as the search tool's query side (content morphemes
    only: N*, SL, SN, SH, XR and VV/VA lemmas, whitespace-joined).

Usage:
    python build_sample_db.py --source /path/to/corpus.db --dest fixture.db \
        --cases 700 --statutes core-current --notices 20

Requires: Python 3.11+ and this package installed (the statute version pick
is imported from the lookup tool), kiwipiepy for the morpheme index.
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
    "sg_ranges",
    "charge_legal_map",
    "clm_versions",
    "official_charges",
    "charge_aliases",
    "charge_norm_map",
    "official_guideline_map",
    "statute_aliases",
]

# Columns that record when or how a loader or an extraction ran, not what the
# law or the judgment says. No tool reads them; they are NULL in the sample.
LOADER_COLUMNS = {
    "clm_versions": ("extracted_at",),
    "prec_defendants": ("label_audit_at",),
    "prec_sentences": ("extraction_model", "extracted_at"),
    "st_byeolpyo": ("listed_at",),
    "st_lapsed": ("note", "noted_at"),
    "st_treaties": ("listed_at",),
}

# Statute whitelist for --statutes core. Matched against both ``name`` and
# ``short_name``; every historical version row of a matched law is kept.
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

# Tax treaties for --treaty-countries. The full set (147 documents, 5.3 MiB of
# text before its trigram index) does not fit under the size limit, so the
# sample takes the partners whose documents cover every lookup path: 베트남 (a
# convention and the protocol amending it — the example in the tool
# description), 미국 (a convention carrying an attachment) and 나이지리아 (a
# document not split into articles). The multilateral conventions come along
# too: every treaty lookup lists them beside the partner's documents.
SAMPLE_TREATY_COUNTRIES = ("베트남", "미국", "나이지리아")

# Cases the harness tests (tests/harness) read by id: a judgment with a body
# for the dive failure path (2016구합95) and an appellate case number joined
# with its court name (의정부지방법원 2025나204080). Neither tier reaches them
# on its own — the first has no decision date.
PINNED_CASES = (205021, 234954)

# Tokenizer recipe for the contentless morpheme index. Mirrors the query side
# (``precedent_search._MORPH_KEEP_TAGS``): content morphemes only,
# verbs/adjectives as lemma form.
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


def copy_rows(dest: sqlite3.Connection, table: str, where: str = "1", params=()) -> None:
    """Copy the rows of ``table`` matching ``where``, writing NULL into loader columns.

    NULL is written, not left to the column default: a run-stamp column may
    default to the current time, which would stamp the build instead.
    """
    omit = LOADER_COLUMNS.get(table, ())
    cols = [r[1] for r in dest.execute(f"PRAGMA src.table_info({table})")]
    names = ", ".join(f'"{c}"' for c in cols)
    values = ", ".join("NULL" if c in omit else f'"{c}"' for c in cols)
    dest.execute(
        f"INSERT INTO main.{table} ({names}) SELECT {values} FROM src.{table} WHERE {where}",
        params,
    )


def copy_table(dest: sqlite3.Connection, table: str, where: str = "1", params=()) -> None:
    """Create ``table`` from the source definition and copy the matching rows."""
    copy_ddl(dest, table)
    copy_rows(dest, table, where, params)
    copy_indexes(dest, table)


def count(dest: sqlite3.Connection, table: str) -> int:
    return dest.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]


# --- statutes -----------------------------------------------------------------


def plan_versions(dest: sqlite3.Connection, law_id: str, as_of_iso: str) -> tuple[list[int], list[int]]:
    """(versions of ``law_id`` whose articles ``core-current`` ships, versions it leaves out).

    The lookup tool reads a law on a date as the latest whole text in force by
    then plus the amendments after it (``statutes._articles_in_force``). The
    sample ships that whole text and the amendments after it: those in force
    by ``as_of_iso`` and those not yet in force that hold only the articles
    they change. A version not yet in force that is stored whole is left out,
    row and all: the tool picks the current version by date, not by label, so
    a whole text shipped without its articles would read as an empty law once
    its date passed, and shipped with them it doubles the law (measured over
    the 27 sample laws: 13,636 articles against 6,695). The sample then keeps
    serving the text current at build time. Earlier versions keep their rows,
    so the version timeline stays whole, but not their articles.
    """
    from lawful_mcp.harness_local.tools.statutes import _is_whole, _pick_current_version

    group = [
        dict(zip(("id", "effective_date", "change_kind", "serial_id"), r))
        for r in dest.execute(
            "SELECT id, effective_date, change_kind, serial_id FROM src.st_statutes WHERE law_id=?",
            (law_id,),
        )
    ]
    base = _pick_current_version([r for r in group if _is_whole(r)], as_of_iso)
    if base is None:
        return [r["id"] for r in group], []
    pending_whole = [
        r["id"] for r in group
        if r["effective_date"] > max(as_of_iso, base["effective_date"]) and _is_whole(r)
    ]
    ship = [
        r["id"] for r in group
        if r["effective_date"] >= base["effective_date"] and r["id"] not in pending_whole
    ]
    return ship, pending_whole


def copy_statute_status_tables(dest: sqlite3.Connection) -> None:
    """Copy the lapse register of the sampled laws and the addenda of the laws whose articles ship."""
    copy_table(dest, "st_lapsed", "law_id IN (SELECT law_id FROM main.st_statutes)")
    copy_table(
        dest, "st_addenda",
        "law_id IN (SELECT s.law_id FROM main.st_statutes s "
        "WHERE s.id IN (SELECT statute_id FROM main.st_articles))",
    )


def copy_annex_index(dest: sqlite3.Connection) -> None:
    """Copy the annexed-table index (별표·서식) of the sampled laws and rules."""
    copy_table(
        dest, "st_byeolpyo",
        "(scope='law' AND owner_id IN (SELECT law_id FROM main.st_statutes)) "
        "OR (scope='admrul' AND owner_id IN (SELECT notice_id FROM main.st_notices))",
    )


def copy_treaty_tables(dest: sqlite3.Connection, countries: tuple[str, ...] | None) -> int:
    """Copy the tax treaties of ``countries`` (None = all) and the multilateral ones, with their index.

    The country-name table comes along whole (a few hundred short rows), so a
    query naming any partner is read as that partner rather than searched as
    text.
    """
    if countries is None:
        copy_table(dest, "st_treaties")
    else:
        marks = ",".join("?" * len(countries))
        copy_table(dest, "st_treaties", f"country IS NULL OR country IN ({marks})", countries)
    copy_table(dest, "st_treaty_articles", "treaty_id IN (SELECT id FROM main.st_treaties)")
    copy_table(dest, "st_treaty_country_names")
    copy_ddl(dest, "st_treaty_articles_fts")
    dest.execute("INSERT INTO main.st_treaty_articles_fts(st_treaty_articles_fts) VALUES('rebuild')")
    return count(dest, "st_treaties")


def copy_triggers(dest: sqlite3.Connection) -> None:
    """Copy the source's triggers on the sample's tables, once the rows are in.

    They keep ``prec_meta`` and the trigram index in step with ``prec_cases``
    for anyone who writes to the sample; the tools only read.
    """
    tables = {r[0] for r in dest.execute("SELECT name FROM main.sqlite_master WHERE type='table'")}
    for name, tbl, sql in dest.execute(
        "SELECT name, tbl_name, sql FROM src.sqlite_master WHERE type='trigger' ORDER BY name"
    ).fetchall():
        if tbl in tables:
            dest.execute(sql)


# --- build --------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=os.environ.get("CORPUS_DB"), help="full corpus db")
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
        help="core-current: whitelist laws, articles of the current whole text and later "
        "versions; core: whitelist laws with every version's articles; all: everything",
    )
    ap.add_argument(
        "--notices", type=int, default=20,
        help="recent current-edition administrative rules to keep",
    )
    ap.add_argument(
        "--treaty-countries",
        default=",".join(SAMPLE_TREATY_COUNTRIES),
        help="comma-separated treaty partners to keep (multilateral treaties always), or 'all'",
    )
    ap.add_argument("--force", action="store_true", help="overwrite dest")
    args = ap.parse_args()

    if not args.source:
        raise SystemExit("--source (or CORPUS_DB) is required")
    if os.path.exists(args.dest):
        if not args.force:
            raise SystemExit(f"{args.dest} exists (use --force)")
        os.remove(args.dest)

    from lawful_mcp.harness_local.tools.statutes import _NOTICE_CURRENT, _today_iso

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
        copy_table(dest, t)
    log(f"reference tables: {', '.join(f'{t}={count(dest, t)}' for t in FULL_TABLES[:3])}, ...")

    # 2. statutes (+ all historical version rows of selected laws), notices.
    for t in ("st_statutes", "st_articles"):
        copy_ddl(dest, t)
    # Effective dates are stored compact (YYYYMMDD); compare in kind.
    as_of = _today_iso()
    shipped: list[int] = []
    left_out: list[int] = []
    if args.statutes == "all":
        copy_rows(dest, "st_statutes")
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
        # metadata, and repeal and point-in-time answers walk the timeline.
        plans = [plan_versions(dest, lid, as_of) for lid in law_ids]
        shipped = [sid for ship, _ in plans for sid in ship]
        left_out = ([sid for _, drop in plans for sid in drop]
                    if args.statutes == "core-current" else [])
        lmarks = ",".join("?" for _ in law_ids)
        dmarks = ",".join("?" for _ in left_out)
        copy_rows(dest, "st_statutes", f"law_id IN ({lmarks}) AND id NOT IN ({dmarks})",
                  [*law_ids, *left_out])
    # Statutes referenced by the charge->penalty map must exist for FK
    # integrity. Penalty bounds live in charge_legal_map itself, so these
    # closure laws keep their row; article bodies only come along outside
    # core-current mode.
    copy_rows(
        dest, "st_statutes",
        "id IN (SELECT statute_id FROM main.charge_legal_map WHERE statute_id IS NOT NULL) "
        "AND id NOT IN (SELECT id FROM main.st_statutes)",
    )
    if left_out:
        back = dest.execute(
            f"SELECT COUNT(*) FROM main.st_statutes WHERE id IN ({dmarks})", left_out).fetchone()[0]
        log(f"statutes: left out {len(left_out) - back} versions not yet in force"
            + (f" ({back} kept for the charge map, without articles)" if back else ""))
    if args.statutes == "core-current":
        smarks = ",".join("?" for _ in shipped)
        copy_rows(dest, "st_articles", f"statute_id IN ({smarks})", shipped)
    else:
        copy_rows(dest, "st_articles", "statute_id IN (SELECT id FROM main.st_statutes)")
    # Only rules the tool would actually surface. An administrative rule is
    # re-issued as a *new row* on amendment, so the corpus holds superseded and
    # future editions alongside the current one; ``statute_search`` searches
    # the current edition with a readable body, and a sample drawn without that
    # filter fills up with rows no search can reach.
    copy_table(
        dest, "st_notices",
        f"id IN (SELECT n.id FROM src.st_notices n WHERE n.has_text_content=1 AND {_NOTICE_CURRENT} "
        "ORDER BY n.effective_date DESC, n.id DESC LIMIT ?)",
        (args.notices,),
    )
    copy_table(dest, "st_notice_articles", "notice_id IN (SELECT id FROM main.st_notices)")
    copy_statute_status_tables(dest)
    copy_annex_index(dest)
    for t in ("st_statutes", "st_articles"):
        copy_indexes(dest, t)
    log(
        f"statutes: {count(dest, 'st_statutes')} versions, {count(dest, 'st_articles')} articles, "
        f"{count(dest, 'st_notices')} rules"
    )
    partners = None if args.treaty_countries == "all" else tuple(
        c.strip() for c in args.treaty_countries.split(",") if c.strip())
    log(f"treaties: {copy_treaty_tables(dest, partners)} documents")

    # 3. precedent selection: sentencing-sample cases first, then holdings, then recent.
    dest.execute("CREATE TEMP TABLE pick(id INTEGER PRIMARY KEY)")
    pmarks = ",".join("?" * len(PINNED_CASES))
    dest.execute(f"INSERT INTO pick SELECT id FROM src.prec_cases WHERE id IN ({pmarks})", PINNED_CASES)
    n_pinned = dest.execute("SELECT COUNT(*) FROM pick").fetchone()[0]
    linked_cap = int(args.cases * args.linked_share)
    dest.execute(
        "INSERT OR IGNORE INTO pick "
        "SELECT DISTINCT case_id FROM src.prec_defendants ORDER BY case_id DESC LIMIT ?",
        (linked_cap,),
    )
    n_linked = dest.execute("SELECT COUNT(*) FROM pick").fetchone()[0] - n_pinned
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
    n_holdings = dest.execute("SELECT COUNT(*) FROM pick").fetchone()[0] - n_pinned - n_linked
    # Restrict the recency scan to the last few years so the indexed year
    # column narrows it; decision_date itself carries no index.
    recent_floor = time.localtime().tm_year - 4
    dest.execute(
        "INSERT OR IGNORE INTO pick "
        "SELECT id FROM src.prec_cases "
        "WHERE decision_year >= ? AND id NOT IN (SELECT id FROM pick) "
        "ORDER BY decision_date DESC, id DESC LIMIT ?",
        (recent_floor, max(args.cases - n_pinned - n_linked - n_holdings, 0)),
    )
    n_recent = dest.execute("SELECT COUNT(*) FROM pick").fetchone()[0] - n_pinned - n_linked - n_holdings
    log(f"case selection: {n_pinned} pinned + {n_linked} sentencing-linked + "
        f"{n_holdings} with holdings + {n_recent} recent")

    copy_table(dest, "prec_cases", "id IN (SELECT id FROM pick)")
    copy_table(dest, "prec_meta", "id IN (SELECT id FROM pick)")
    for t in ("prec_case_numbers", "prec_sentences", "prec_defendants", "prec_defendant_charges"):
        copy_table(dest, t, "case_id IN (SELECT id FROM pick)")
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
    copy_triggers(dest)

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

    log(f"done: {args.dest} = {os.path.getsize(args.dest) / 1048576:.2f} MiB")
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
