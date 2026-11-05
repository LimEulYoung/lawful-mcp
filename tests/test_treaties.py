"""Tax treaties — the third corpus of ``statute_lookup`` (``tools/treaties.py``).

The corpus here is empty statute and rule tables plus the treaty schema with
a few rows. A database without the treaty tables (loaded before treaties)
must answer "no treaty" at every entry point (fail open). The last tests run
on the committed sample, which carries a few partners' treaties.
"""
from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from lawful_mcp.config import case_url_base

statutes = importlib.import_module("lawful_mcp.tools.statutes")

BASE_SCHEMA = """
CREATE TABLE st_statutes (
    id INTEGER PRIMARY KEY, law_id TEXT, name TEXT, short_name TEXT, kind TEXT, issuing_agency TEXT,
    promulgation_no TEXT, promulgation_date TEXT, effective_date TEXT, history_status TEXT, change_kind TEXT,
    mst TEXT, serial_id TEXT);
CREATE TABLE st_articles (
    id INTEGER PRIMARY KEY, statute_id INTEGER, article_no TEXT, article_no_num INTEGER, article_branch INTEGER,
    title TEXT, article_text TEXT, article_changed TEXT, article_eff_date TEXT);
CREATE VIRTUAL TABLE st_articles_fts USING fts5(
    title, article_text, content='st_articles', content_rowid='id', tokenize='trigram');
CREATE TABLE st_notices (
    id INTEGER PRIMARY KEY, serial_id TEXT, notice_id TEXT, name TEXT, kind TEXT, issuing_agency TEXT,
    notice_no TEXT, issued_date TEXT, effective_date TEXT, category TEXT, has_articles INTEGER,
    has_text_content INTEGER, history_status TEXT, body_source TEXT);
CREATE TABLE st_notice_articles (
    id INTEGER PRIMARY KEY, notice_id INTEGER, article_seq INTEGER, article_no INTEGER, article_no_str TEXT,
    article_text TEXT);
CREATE VIRTUAL TABLE st_notice_articles_fts USING fts5(
    article_text, content='st_notice_articles', content_rowid='id', tokenize='trigram');
"""
# The loader's treaty schema.
TREATY_SCHEMA = """
CREATE TABLE st_treaties (id INTEGER PRIMARY KEY, treaty_no TEXT, name TEXT NOT NULL, name_en TEXT, country TEXT,
    role TEXT NOT NULL, signed_date TEXT, effective_date TEXT, attachment_url TEXT, listed_at TEXT NOT NULL);
CREATE TABLE st_treaty_articles (id INTEGER PRIMARY KEY, treaty_id INTEGER NOT NULL, article_seq INTEGER NOT NULL,
    article_no INTEGER, article_branch INTEGER NOT NULL DEFAULT 0, title TEXT, article_text TEXT NOT NULL);
CREATE VIRTUAL TABLE st_treaty_articles_fts USING fts5(
    article_text, content='st_treaty_articles', content_rowid='id', tokenize='trigram');
CREATE TABLE st_treaty_country_names (name TEXT PRIMARY KEY, country TEXT NOT NULL);
"""
VN = "대한민국 정부와 베트남사회주의공화국 정부간의 소득에 대한 조세의 이중과세회피와 탈세방지를 위한 협정"


def _build(path: Path, *, treaties: bool) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(BASE_SCHEMA)
    # One statute — a treaty query mixed into a statute search must not break it.
    conn.execute("INSERT INTO st_statutes (id, law_id, name, kind, effective_date, history_status, serial_id) "
                 "VALUES (1, '001234', '국제조세조정에 관한 법률', '법률', '20200101', '현행', '1')")
    conn.execute("INSERT INTO st_articles (statute_id, article_no, article_no_num, article_branch, title, article_text) "
                 "VALUES (1, '1', 1, 0, '목적', '제1조(목적) 이 법은 국제조세 조정을 목적으로 한다.')")
    conn.execute("INSERT INTO st_articles_fts(st_articles_fts) VALUES('rebuild')")
    if treaties:
        conn.executescript(TREATY_SCHEMA)
        conn.executemany(
            "INSERT INTO st_treaties (id, name, country, role, signed_date, effective_date, attachment_url, listed_at) "
            "VALUES (?,?,?,?,?,?,?,'t')", [
                (1307, VN, "베트남", "협약", "19940520", "19940909", None),
                (15611, f"1994년 5월 20일 하노이에서 서명된 {VN}을 개정하는 제2의정서", "베트남", "개정", "20190000",
                 "20210120", None),
                (1304, "대한민국 정부와 중화인민공화국 정부간의 소득에 대한 조세의 이중과세회피와 탈세방지를 위한 협정",
                 "중국", "협약", None, "19940928", None),
                (12279, "대한민국 정부와 중화인민공화국 홍콩특별행정구 정부 간의 소득에 대한 조세의 이중과세 방지 협정",
                 "홍콩", "협약", None, "20160927", None),
                (5735, "대한민국 정부와 바하마연방 정부 간의 조세정보 교환에 관한 협정", "바하마", "협약", None,
                 "20130715", "http://law.go.kr/flDownload.do?flSeq=1"),
            ])
        conn.executemany(
            "INSERT INTO st_treaty_articles (treaty_id, article_seq, article_no, title, article_text) VALUES (?,?,?,?,?)",
            [
                (1307, 0, None, "전문", "양국 정부는 다음과 같이 합의하였다."),
                (1307, 1, 4, None, "제4조 거주자 1. 중대한 이해관계의 중심지가 있는 체약국의 거주자로 본다."),
                (1307, 2, 9, None, "제9조 특수관계기업 1. 이윤은 과세할 수 있다."),
                (15611, 0, 4, None, "제4조 이 협정의 제9조(특수관계기업)를 삭제하고 다음으로 대체한다."),
                (1304, 0, 4, None, "제4조 거주자"),
                (12279, 0, 4, None, "제4조 거주자"),
                (5735, 0, None, None, "각하, 본인은 조세정보 교환을 제의합니다."),
            ])
        conn.execute("INSERT INTO st_treaty_articles_fts(st_treaty_articles_fts) VALUES('rebuild')")
        conn.executemany("INSERT INTO st_treaty_country_names VALUES (?,?)", [
            ("베트남", "베트남"), ("월남", "베트남"), ("베", "베트남"), ("중국", "중국"), ("중화인민공화국", "중국"),
            ("중", "중국"), ("홍콩", "홍콩"), ("바하마", "바하마")])
    conn.commit()
    conn.close()
    return path


@pytest.fixture(scope="module")
def treaty_file(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("treaties") / "with_treaties.db", treaties=True)


@pytest.fixture(scope="module")
def bare_file(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("treaties") / "without_treaties.db", treaties=False)


@pytest.fixture
def treaty_db(treaty_file, monkeypatch):
    monkeypatch.setenv("CORPUS_DB", str(treaty_file))
    return treaty_file


@pytest.fixture
def bare_db(bare_file, monkeypatch):
    monkeypatch.setenv("CORPUS_DB", str(bare_file))
    return bare_file


def call(**kwargs) -> str:
    return statutes.statute_lookup(SimpleNamespace(deps=None), **kwargs)


@pytest.mark.parametrize("query", ["한-베트남 조세조약", "대한민국-베트남 조세협정 제4조", "월남과의 이중과세방지협정",
                                   "한·베 조세협정 제4조"])
def test_partner_and_tax_treaty_word_find_the_treaty_first(treaty_db, query):
    out = call(query=query, limit=5)
    first = next(line for line in out.splitlines() if line.startswith("- "))
    assert first.startswith(f"- trty-1307 {VN}")
    assert "trty-15611" in out and "/statutes/trty-1307" in out


def test_one_letter_partner_and_territory(treaty_db):
    out = call(query="한중 조세조약", limit=5)
    assert "trty-1304" in out and "trty-12279" not in out      # the Hong Kong agreement is not '한중'


@pytest.mark.parametrize("query", [
    "한미 FTA 협정 원산지", "베트남 투자", "고정사업장 거주자",
    # Customs terms contain '세율' and '과세' — the Korea-US tax convention once led a Korea-US FTA search.
    "한미 FTA 협정관세율 원산지증명서", "한-EU FTA 협정세율", "한미 FTA 과세가격 협정",
    # An article number alone points at no treaty — as a text term it put an arbitrary treaty first.
    "조세협정 제4조",
])
def test_queries_not_about_a_tax_treaty_list_no_treaty(treaty_db, query):
    assert "trty-" not in call(query=query, limit=10)


def test_article_lookup_shows_the_text_and_every_document_of_the_partner(treaty_db):
    out = call(statute_id="trty-1307", articles=["제4조"])
    assert out.startswith("## status: ok")
    assert "text_kind: 공식 조약 원문" in out and "중대한 이해관계의 중심지" in out
    assert "제9조 특수관계기업" not in out
    assert "- trty-15611 · 2021-01-20 발효 · 개정 문서" in out and "← 이 문서" in out
    assert "trty-1304" not in out                               # another partner's documents stay out
    assert f"- url: {case_url_base()}/statutes/trty-1307" in out


def test_preamble_by_name_missing_article_and_unsplit_document(treaty_db):
    assert "합의하였다" in call(statute_id="trty-1307", articles=["전문"])
    assert "## 없는 조문: 30" in call(statute_id="trty-1307", articles=["30"])
    whole = call(statute_id="TRTY-5735")
    assert "## body" in whole and "조세정보 교환을 제의" in whole
    # This server registers no web tool, so the attachment note names the file, not a tool to read it with.
    assert "## 첨부 원문: http://law.go.kr/flDownload.do?flSeq=1" in whole and "web_fetch" not in whole
    assert call(statute_id="trty-99999").startswith("## status: missing")
    assert call(statute_id=f"{case_url_base()}/statutes/trty-1307/4").startswith("## status: ok")


def test_without_treaty_tables_statute_search_still_answers(bare_db):
    out = call(query="한-베트남 조세조약")
    assert out.startswith("## status: ok") and "trty-" not in out
    assert call(statute_id="trty-1307").startswith("## status: missing")


def test_the_sample_answers_a_partner_query_with_its_treaties(monkeypatch):
    monkeypatch.delenv("CORPUS_DB", raising=False)
    out = call(query="베트남 조세조약", limit=5)
    first = next(line for line in out.splitlines() if line.startswith("- "))
    assert first.startswith("- trty-1307 ")
    detail = call(statute_id="trty-1307", articles=["4"])
    assert "text_kind: 공식 조약 원문" in detail and "- trty-15611 " in detail
    assert "다자 조세조약" in detail                               # multilateral conventions listed beside
    unsplit = call(statute_id="trty-17985")                        # a document not split into articles
    assert "## body" in unsplit and "## 첨부 원문: " in unsplit
