"""Sentencing statistics: groups by the charges a judgment holds, convictions only, real severity order, provision
groups, last ten years, old spellings, what each shown judgment noted; and the candidate list."""

import importlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

# Modules, not the functions of the same name that lawful_mcp.harness_local.tools exports.
ss = importlib.import_module("lawful_mcp.harness_local.tools.sentence_statistics")
penalty = importlib.import_module("lawful_mcp.harness_local.tools.statutory_penalty")
statutes = importlib.import_module("lawful_mcp.harness_local.tools.statutes")
sa = importlib.import_module("lawful_mcp.harness_local.tools.sentencing_analysis")
ci = importlib.import_module("lawful_mcp.harness_local.tools._charge_index")

CTX = SimpleNamespace(deps=SimpleNamespace(exclude_case_ids=frozenset()))


def row(case_id, kind, months=None, fine=None, probation=0, year=2020, defendant="A"):
    return {"case_id": case_id, "defendant_id": defendant, "sentence_type": kind, "sentence_months": months,
            "fine_amount": fine, "probation": probation, "probation_months": 24 if probation else None,
            "decision_year": year, "term_kind": None, "sentence_months_min": None,
            "case_number": f"2020고단{case_id}", "sentencing_reason": None}


@pytest.fixture(autouse=True)
def run_date(monkeypatch):
    monkeypatch.setattr(statutes, "_today_iso", lambda: "20261104")


def test_fines_sort_by_amount_not_year():
    rows = [row(1, "fine", fine=700_000, year=2025), row(2, "fine", fine=500_000, year=2017),
            row(3, "fine", fine=200_000, year=2013), row(4, "fine", fine=1_000_000, year=2010),
            row(5, "imprisonment", months=6, year=2012), row(6, "imprisonment", months=6, probation=1, year=2011)]
    ordered = [r["case_id"] for _q, r in ss._stratified_sample(rows)]
    assert ordered == [3, 2, 1, 4, 6, 5]  # suspended before executed of the same length


def test_one_judgment_once_in_comparables():
    rows = [row(i, "fine", fine=100_000 * i) for i in range(1, 30)]
    rows += [row(99, "imprisonment", months=12, defendant=d) for d in "ABCD"]
    sample = ss._stratified_sample(rows)
    assert len({r["case_id"] for _q, r in sample}) == len(sample)


def test_medians_are_observed_values():
    rows = [row(i, "fine", fine=f) for i, f in enumerate([300_000, 500_000, 1_000_000, 2_000_000, 9_000_000])]
    assert ss._medians(rows) == ["벌금 중앙값 1,000,000원 (가운데 50% 500,000~2,000,000원)"]


def test_old_spellings_are_named_beside_the_pool():
    from lawful_mcp.harness_local.deps import open_db

    conn = open_db()
    try:
        index = ss.official_index(conn)
        assert "폭력행위등처벌에관한법률위반(집단·흉기등상해)" in ss._former_names(index, "특수상해")
        assert "교통사고처리특례법위반" in ss._umbrella_names(conn, index, "교통사고처리특례법위반(치상)")
        assert "특정범죄가중처벌등에관한법률위반(도주차량)" in ss._umbrella_names(
            conn, index, "특정범죄가중처벌등에관한법률위반(도주치상)")
        assert ss._umbrella_names(conn, index, "사기") == []
    finally:
        conn.close()


def test_shown_cases_carry_what_their_judgment_noted():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE prec_sentences (case_id INTEGER PRIMARY KEY, raw_v2 TEXT)")
    shown = [{**row(1, "imprisonment", months=12), "case_number": "2020고단1", "sentencing_reason": "요약"},
             {**row(2, "fine", fine=500_000), "case_number": "2020고정2", "sentencing_reason": "요약"}]
    assert ss._noted(conn, shown) == {}  # no v2 data
    v2 = {"defendants": [{"id": "A", "prior_same": "예", "repeat_or_probation": "없음", "settlement": "아니오",
                          "restitution": "없음", "confession": "아니오", "magnitude": "편취액 합계 4,800만 원",
                          "guideline": {"type": "일반사기 > 제1유형(1억 원 미만)", "zone": "기본영역",
                                        "range": "6월 - 1년 6월", "factors": ""}}]}
    conn.execute("INSERT INTO prec_sentences VALUES (1, ?)", (json.dumps(v2, ensure_ascii=False),))
    noted = ss._noted(conn, shown)
    # Unmentioned circumstances are left out rather than written as 없음.
    assert noted == {(1, "A"): ["  사정: 동종 전과 있음 · 합의·처벌불원 없음 · 부인 · 크기 「편취액 합계 4,800만 원」",
                                "  판결의 양형기준: 기본영역 · 6월 - 1년 6월 · 일반사기 > 제1유형(1억 원 미만)"]}
    lines = ss._case_lines(shown[0], noted[(1, "A")], frozenset({"공갈", "협박"}))
    assert lines[2:] == ["  다른 죄: 공갈 · 협박", "  선고: 징역 12월", *noted[(1, "A")]]  # no reason line
    assert ss._case_lines(shown[1])[-1] == "  양형 이유: 요약"


def grouped(monkeypatch, pool, names, **kw):
    """`_actual_sentences` over a made-up pool of (row, charges)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE prec_sentences (case_id INTEGER PRIMARY KEY, raw_v2 TEXT)")
    monkeypatch.setattr(ss, "_fetch_pool", lambda _c, _n, _e: [(r, frozenset(c)) for r, c in pool])
    status, sections = ss._actual_sentences(conn, names, exclude=frozenset(), reference_year=None, **kw)
    return status, [line for heading, lines in sections for line in (heading, *lines)]


def test_statistics_take_the_groups_holding_every_charge_nearest_first(monkeypatch):
    pool = ([(row(i, "fine", fine=3_000_000), {"음주운전", "무면허운전"}) for i in range(10)]
            + [(row(100 + i, "imprisonment", months=12), {"음주운전", "무면허운전", "도주"}) for i in range(25)]
            + [(row(200 + i, "imprisonment", months=30), {"음주운전", "무면허운전", "도주", "상해"}) for i in range(9)]
            + [(row(300 + i, "fine", fine=1_000_000), {"음주운전"}) for i in range(100)]
            + [(row(400, "not_guilty"), {"음주운전", "무면허운전"})])
    status, lines = grouped(monkeypatch, pool, ["음주운전", "무면허운전"])
    text = "\n".join(lines)
    assert status == "ok"
    assert "- 이 죄들만: 유죄 10건 · 벌금 10 (100%) · 벌금 중앙값 3,000,000원 · 무죄 1건" in text
    assert "- 이 죄들 + 다른 죄 1개: 유죄 25건" in text and "- 음주운전만: 유죄 100건" in text
    # 10 + 25 reach 30; the third group and the single-charge judgments stay out of the statistics.
    assert "## 통계 표본 — 이 죄들만 ~ 이 죄들 + 다른 죄 1개" in text and "- 유죄 35건 · 무죄 1건" in text
    assert "- 형종: 징역 25 (71%) · 벌금 10 (29%)" in text
    assert "다른 죄: 도주" in text and "다른 죄: 상해" not in text


def test_fewer_than_thirty_convictions_are_listed_not_counted(monkeypatch):
    pool = ([(row(i, "imprisonment", months=6 + i), {"사기", "폭행"}) for i in range(3)]
            + [(row(10, "imprisonment", months=40), {"사기", "폭행", "공갈"})]
            + [(row(20 + i, "imprisonment", months=12), {"사기"}) for i in range(50)])
    status, lines = grouped(monkeypatch, pool, ["사기", "폭행"])
    text = "\n".join(lines)
    assert status == "low_n_grid" and "## 사례 나열" in text and "## 비교 판례" not in text
    assert "참고: 유죄 표본 4건 < 30" in text and text.count("- 사건:") == 4
    assert text.index("다른 죄: 공갈") > text.rindex("- 사건:") - 200  # the nearer group is listed first
    assert "- 사기만: 유죄 50건" in text and "- 폭행만: 유죄 0건" in text


def test_statistics_answer_for_several_charges_on_the_sample():
    one = sa.sentencing_analysis(CTX, charges=["사기"])
    assert one.startswith("## status: ok") and "- 이 죄만: 유죄" in one and "## 통계 표본 — 이 죄만" in one
    assert "sentence: 무죄" not in one and "mean=" not in one and "- 최근 10년(2017~)" in one
    two = sa.sentencing_analysis(CTX, charges=["사기", "폭행", "없는죄명"])
    assert "- 사기만: 유죄" in two and "- 폭행만: 유죄" in two
    assert "## 묶음별 (1심) — 기준 죄명: 사기, 폭행" in two and "실선고에 넣지 못한 이름(판결 죄명표의 공식 죄명이 아님): 없는죄명" in two
    assert sa.sentencing_analysis(CTX, charges=["347"]).startswith("## status: charge_numeric")


def test_each_provision_is_read_once_per_guideline():
    # 강도상해: one article, mapped under two guideline categories — one provision each, never the same twice.
    from lawful_mcp.harness_local.deps import open_db

    with open_db() as conn:
        reading = penalty.read_charge(conn, "강도상해")
    assert reading.kind == "ok" and len(reading.provisions) == len(set(reading.provisions))
    assert len({p.article for p in reading.provisions}) == 1 and len({p.category_id for p in reading.provisions}) == 2
    assert all(p.penalty is not None for p in reading.provisions)


def test_an_unknown_spelling_gets_names_like_it_and_a_name_to_choose_gets_its_list_only():
    from lawful_mcp.harness_local.deps import open_db

    with open_db() as conn:
        r = penalty.read_charge(conn, "직무집행방해")
    assert r.kind == "candidates" and r.resolution.kind == "similar" and r.resolution.candidates[0] == "공무집행방해"
    out = sa.sentencing_analysis(CTX, charges=["직무집행방해"])
    assert out.startswith("## status: candidates") and "\n- 공무집행방해 — 양형기준 형법 §136" in out
    assert "## 양형 판단 규칙" not in out
    # A bare statute name: its charges to choose from, without the rules or the bare name's mixed pool.
    bare = sa.sentencing_analysis(CTX, charges=["도로교통법위반"])
    assert bare.startswith("## status: candidates") and "## 묶음별" not in bare and "## 양형 판단 규칙" not in bare
    mixed = sa.sentencing_analysis(CTX, charges=["상해", "도로교통법위반"])
    assert mixed.startswith("## status: ok") and "## 양형 판단 규칙" in mixed
    assert "## 묶음별 (1심) — 기준 죄명: 상해\n" in mixed


def test_name_maps_are_read_from_the_database():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE st_statutes (law_id TEXT, name TEXT, short_name TEXT, kind TEXT, history_status TEXT,
            effective_date TEXT, change_kind TEXT);
        CREATE TABLE statute_aliases (alias TEXT PRIMARY KEY, statute_name TEXT NOT NULL, how TEXT NOT NULL);
        INSERT INTO statute_aliases VALUES ('특가법', '특정범죄 가중처벌 등에 관한 법률', '약칭'),
            ('해사안전법', '해상교통안전법', '분리 승계');
    """)
    # A split successor is for charge names only: a statute search keeps the name's own statute.
    assert statutes.statute_aliases(conn) == {"특가법": "특정범죄 가중처벌 등에 관한 법률"}
    laws = ci.LawNames(conn)
    assert laws.alias["특가법"] == "특정범죄가중처벌등에관한법률" and laws.alias["해사안전법"] == "해상교통안전법"
