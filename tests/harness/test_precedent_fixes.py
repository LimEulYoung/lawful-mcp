"""Judgment output and search fixes: full decision dates, the Supreme Court level, the court filter and the excerpt
guidance — on small databases built from the sample database's table definitions."""

import asyncio
import importlib
import sqlite3
from pathlib import Path

from lawful_mcp.harness_local.deps import HarnessDeps

ps = importlib.import_module("lawful_mcp.harness_local.tools.precedent_search")
dive = importlib.import_module("lawful_mcp.harness_local.tools.precedent_dive")
SAMPLE = Path(__file__).resolve().parents[2] / "data/fixture.db"
ROWS = [  # id, case_number, court_name, court_level, decision_date, year, holdings
    (1, "92다3083", "대법원", "대법원", "1993-02-26", 1993, "토지대장에 기재된 소유권이전등록의 추정력"),
    (3, "2018누10258", "부산고등법원", "2심", None, 2018, None),
    (4, "2011다52727", "대법원", "대법원", "2012-02-16", 2012, "구상권자인 공동불법행위자 측에 과실이 없는 경우"),
    (7, "2016고단1667", "수원지방법원", "1심", "2016-09-21", 2016, None),
    (8, "2016고단1667", "수원지방법원 성남지원", "1심", "2016-09-21", 2016, None),
]


def small_db(path):
    """A judgment table with the sample database's definition and the rows above."""
    schema = sqlite3.connect(SAMPLE).execute("SELECT sql FROM sqlite_master WHERE name='prec_cases'").fetchone()[0]
    db = sqlite3.connect(path)
    db.execute(schema)
    db.executemany(
        "INSERT INTO prec_cases (id, source, source_id, case_number, case_name, court_name, court_level, decision_date,"
        " decision_year, holdings, content_md) VALUES (?, 's', ?, ?, '구상금', ?, ?, ?, ?, ?, '본문')",
        [(i, str(i), n, c, lv, d, y, h) for i, n, c, lv, d, y, h in ROWS],
    )
    db.commit()
    db.row_factory = sqlite3.Row
    return db


def test_court_label_gives_the_full_date_and_drops_a_level_the_name_states():
    assert ps.court_label("대법원", "대법원", "2012-02-16", 2012) == "대법원 2012-02-16"
    assert ps.court_label("헌법재판소", "헌재", "2018-06-28", 2018) == "헌법재판소 2018-06-28"
    assert ps.court_label("서울고등법원", "2심", "2019-05-02", 2019) == "서울고등법원 2심 2019-05-02"
    assert ps.court_label("부산고등법원", "2심", None, 2018) == "부산고등법원 2심 2018"   # date unknown


def test_hydrate_keeps_the_order_and_filters_by_court_level_and_court(tmp_path):
    db = small_db(tmp_path / "t.db")
    matches = ps._hydrate(db, [4, 1, 3, 8, 7], None, None, None, cap=None)
    assert [m["id"] for m in matches] == [4, 1, 3, 8, 7]
    assert [m["id"] for m in ps._hydrate(db, [1, 3, 4], "대법원", None, None, cap=None)] == [1, 4]
    assert [m["id"] for m in ps._hydrate(db, [1, 3, 4], "2심", None, None, cap=None)] == [3]
    # The court filter takes a part of the name, spaces aside: the district court with its branch, or the branch.
    assert [m["id"] for m in ps._hydrate(db, [7, 8], None, None, None, cap=None, court_name="수원지방법원")] == [7, 8]
    assert [m["id"] for m in ps._hydrate(db, [7, 8], None, None, None, cap=None, court_name="수원지방법원성남지원")] == [8]
    card = ps._format_response_md({"status": "ok", "matches": matches[1:3]}) + "\n"
    assert "  court: 대법원 1993-02-26\n" in card and "  court: 부산고등법원 2심 2018\n" in card


def test_the_judgment_reader_gives_the_court_and_date_to_the_model_and_the_sub_agent(tmp_path, monkeypatch):
    db_path = tmp_path / "t.db"
    small_db(db_path).close()
    monkeypatch.setenv("HARNESS_DB", str(db_path))
    prompts = []

    class Reader:
        async def run(self, prompt, **kwargs):
            prompts.append(prompt)
            return type("R", (), {"output": type("D", (), {"summary": "요약", "not_in_text": False})()})()

    class Ctx:
        deps = HarnessDeps(dive_subagent=Reader())

    out = asyncio.run(dive.precedent_dive(Ctx(), case_id=4, question="법리"))
    assert "- court: 대법원 2012-02-16" in out
    assert "(사건번호: 2011다52727, 대법원 2012-02-16)" in prompts[0]
