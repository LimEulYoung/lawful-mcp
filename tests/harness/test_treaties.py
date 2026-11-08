"""Tax treaties in statute_lookup — a copy of the sample database (its treaty tables are empty) with a few treaty
rows added; only the columns read here are filled. No network, no model calls."""

import os
import shutil
import sqlite3

import pytest
from test_harness import call

VN = "대한민국 정부와 베트남사회주의공화국 정부간의 소득에 대한 조세의 이중과세회피와 탈세방지를 위한 협정"


@pytest.fixture(scope="module")
def treaty_copy(tmp_path_factory):
    path = tmp_path_factory.mktemp("treaties") / "sample_with_treaties.db"
    shutil.copy(os.environ["HARNESS_DB"], path)
    conn = sqlite3.connect(path)
    # lawful-mcp: the bundled sample carries real treaties; empty the tables the research sample ships empty.
    conn.executescript("DELETE FROM st_treaty_articles; DELETE FROM st_treaties; DELETE FROM st_treaty_country_names;")
    treaties = [
        (1307, VN, "베트남", "협약", "19940909"),
        (15611, f"1994년 5월 20일 하노이에서 서명된 {VN}을 개정하는 제2의정서", "베트남", "개정", "20210120"),
        (1304, "대한민국 정부와 중화인민공화국 정부간의 소득에 대한 조세의 이중과세회피와 탈세방지를 위한 협정",
         "중국", "협약", "19940928"),
        (12279, "대한민국 정부와 중화인민공화국 홍콩특별행정구 정부 간의 소득에 대한 조세의 이중과세 방지 협정",
         "홍콩", "협약", "20160927"),
        (5735, "대한민국 정부와 바하마연방 정부 간의 조세정보 교환에 관한 협정", "바하마", "협약", "20130715"),
    ]
    conn.executemany("INSERT INTO st_treaties (id, name, country, role, effective_date) VALUES (?,?,?,?,?)", treaties)
    articles = [
        (1307, 0, None, "전문", "양국 정부는 다음과 같이 합의하였다."),
        (1307, 1, 4, None, "제4조 거주자 1. 중대한 이해관계의 중심지가 있는 체약국의 거주자로 본다."),
        (1307, 2, 9, None, "제9조 특수관계기업 1. 이윤은 과세할 수 있다."),
        (15611, 0, 4, None, "제4조 이 협정의 제9조(특수관계기업)를 삭제하고 다음으로 대체한다."),
        (1304, 0, 4, None, "제4조 거주자"),
        (12279, 0, 4, None, "제4조 거주자"),
        (5735, 0, None, None, "각하, 본인은 조세정보 교환을 제의합니다."),
    ]
    conn.executemany("INSERT INTO st_treaty_articles (treaty_id, article_seq, article_no, title, article_text) "
                     "VALUES (?,?,?,?,?)", articles)
    conn.execute("INSERT INTO st_treaty_articles_fts(st_treaty_articles_fts) VALUES('rebuild')")
    conn.executemany("INSERT INTO st_treaty_country_names VALUES (?,?)", [
        ("베트남", "베트남"), ("월남", "베트남"), ("베", "베트남"), ("중국", "중국"), ("중화인민공화국", "중국"), ("중", "중국"),
        ("홍콩", "홍콩"), ("바하마", "바하마")])
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def treaty_db(treaty_copy, monkeypatch):
    monkeypatch.setenv("HARNESS_DB", str(treaty_copy))
    return treaty_copy


@pytest.mark.parametrize("query", ["한-베트남 조세조약", "대한민국-베트남 조세협정 제4조", "월남과의 이중과세방지협정",
                                   "한·베 조세협정 제4조"])
def test_partner_and_tax_treaty_word_find_the_treaty_first(treaty_db, query):
    out = call("statute_lookup", query=query, limit=5)
    first = next(line for line in out.splitlines() if line.startswith("- ") and not line.startswith("- message:"))
    assert first.startswith(f"- trty-1307 {VN}")
    assert "trty-15611" in out and "/statutes/trty-1307" in out


def test_one_letter_partner_and_territory(treaty_db):
    out = call("statute_lookup", query="한중 조세조약", limit=5)
    assert "trty-1304" in out and "trty-12279" not in out      # 홍콩 협정은 '한중'이 아니다


@pytest.mark.parametrize("query", [
    "한미 FTA 협정 원산지", "베트남 투자", "고정사업장 거주자",
    # customs terms carry '세율'·'과세' — a Korea-US FTA search put the Korea-US tax convention on top
    "한미 FTA 협정관세율 원산지증명서", "한-EU FTA 협정세율", "한미 FTA 과세가격 협정",
    # an article number alone names no treaty; searched as text it ranked an arbitrary treaty first
    "조세협정 제4조",
])
def test_queries_not_about_a_tax_treaty_list_no_treaty(treaty_db, query):
    assert "trty-" not in call("statute_lookup", query=query, limit=10)


def test_article_lookup_shows_the_text_and_every_document_of_the_partner(treaty_db):
    out = call("statute_lookup", statute_id="trty-1307", articles=["제4조"])
    assert out.startswith("## status: ok")
    assert "text_kind: 공식 조약 원문" in out and "중대한 이해관계의 중심지" in out
    assert "제9조 특수관계기업" not in out
    assert "- trty-15611 · 2021-01-20 발효 · 개정 문서" in out and "← 이 문서" in out
    assert "trty-1304" not in out                               # 다른 상대국 문서는 섞지 않는다


def test_preamble_by_name_missing_article_and_unsplit_document(treaty_db):
    assert "합의하였다" in call("statute_lookup", statute_id="trty-1307", articles=["전문"])
    assert "## 없는 조문: 30" in call("statute_lookup", statute_id="trty-1307", articles=["30"])
    whole = call("statute_lookup", statute_id="TRTY-5735")
    assert "## body" in whole and "조세정보 교환을 제의" in whole
    assert call("statute_lookup", statute_id="trty-99999").startswith("## status: missing")
    assert call("statute_lookup", statute_id="https://lawful.crow-tit.com/statutes/trty-1307/4").startswith(
        "## status: ok")
