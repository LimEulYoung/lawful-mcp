"""statute_lookup — 법령(`st_statutes`) + 고시(`st_notices`) 통합 조회.

3 모드 (chapter 7 §3, chapter 8 §4):
  A) `query` 또는 `kind`만        → 목록 (statute별 1줄 + preview)
  B) `statute_id`만              → 조문 개요 (article_no + title)
  C) `statute_id` + `articles`   → 조문 본문

검색은 FTS5 trigram + 법령명 substring (dense는 노이즈만 추가, chapter 8 §3.1).
`statute_id`는 int만 (DB primary key).
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

NOTICE_KINDS = {"고시", "admrul"}

# 한 호출에 articles 인자로 받을 수 있는 조문 수 cap. 컨텍스트 다이어트 — 1~79조
# 통째 dump로 16K 토큰 limit 초과한 케이스 방어.
ARTICLES_MAX = 8

# 검색 결과 수 cap. **호출자가 준 값을 그대로 SQL 에 넣으면 안 된다** — `_search_statutes`
# 가 `max(limit * 8, 80)` 을 `LIMIT ?` 에 쓰고, 그 뒤 `_fold_versions`·`_is_repealed_as_of`
# 가 **law_id 마다 추가 질의**를 한다. 상한이 없으면 `limit=1_000_000` 한 번이
# `LIMIT 8_000_000` + 그만큼의 per-row 질의가 된다.
# ⚠ 이 도구는 MCP 에도 상시 등록되고 그 프로세스에는 rate limit 이 없다
# (`mcp_server` 가 그렇게 적어 뒀다) — 즉 **무인증 외부 호출로 닿는다.**
# 응답은 이미 관련도 순이라 실사용은 10 안쪽이고, 50 은 넉넉한 여유다.
LIMIT_MAX = 50

# Statute page URLs, same shape as the `url` field of the precedent tools:
# a law is /statutes/{law_id}, one article /statutes/{law_id}/{jo}, an
# administrative rule /statutes/admrul-{st_notices.id}. Attached to responses
# so a model citing the text can link to the source.
NOTICE_URL_PREFIX = "admrul-"   # law_id namespace for administrative rules


# ---------- 헬퍼 ----------

from ._fts import safe_fts_query


# ---------- 법령명 정규화 (약칭 + 형태소 토큰) ----------
# 한국어 법령명은 띄어쓰기 없는 합성어가 많아("광업제조업자") trigram FTS·substring 으로
# 못 잡는다. 형태소로 명사 토큰을 쪼개 법령명 부분매칭(LIKE)에 쓰고, 통칭은 약칭맵으로
# 정식명에 매핑한다. 채팅 statute_lookup·웹 search_statutes(harness_repo) 공용 단일 출처.

# 통칭/약칭 → 정식 법령명 (공백 제거 기준 키). value 는 DB 법령명과 글자까지 일치해야
# 한다(검증 완료). 형태소 토큰화로 안 풀리는 통칭(예: '산재'는 '산업재해보상보험법' 이름에
# 글자가 없음)을 매핑. 운영하며 점진 확장.
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
    # 개명 구법 → 현행명 (같은 계보의 명칭 변경 — 판례 reference_statute 미앵커 상위.
    # 키는 공백·가운뎃점 제거 형태)
    "조세감면규제법": "조세특례제한법",
    "총포도검화약류등단속법": "총포ㆍ도검ㆍ화약류 등의 안전관리에 관한 법률",
    "학원의설립운영에관한법률": "학원의 설립ㆍ운영 및 과외교습에 관한 법률",
    # 한자 표기 (옛 판례 참조조문)
    "憲法": "대한민국헌법",
    "憲法裁判所法": "헌법재판소법",
    # 판례 관용 약칭 (본문 추출 미앵커 상위 — 정의구 없이도 쓰이는 통용 약칭)
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


# 가운뎃점 변형 — DB 법령명엔 U+318D 'ㆍ'(아래아)가 들어있고('초ㆍ중등교육법'), 사용자는
# 점 없이('초중등교육법') 치거나 U+00B7 '·' 등 시각적으로 동일한 다른 코드포인트로 입력한다.
# 매칭 시 양쪽(질의·저장명)에서 *제거* 해 셋을 동일 취급한다. _STATUTE_ALIASES value 에도
# 'ㆍ'가 있으나 alias 키는 점 없는 통칭이라 영향 없음.
_DOT_CHARS = ("ㆍ", "·", "‧", "∙", "・", "․")  # U+318D U+00B7 U+2027 U+2219 U+30FB U+2024


def _strip_dots(s: str) -> str:
    for ch in _DOT_CHARS:
        s = s.replace(ch, "")
    return s


def _name_norm_sql(col: str) -> str:
    """법령명 매칭용 정규화 SQL 식 — 공백 + 가운뎃점 변형 제거(_strip_dots 와 동형)."""
    expr = col
    for ch in (" ", *_DOT_CHARS):
        expr = f"REPLACE({expr},'{ch}','')"
    return expr


def _statute_name_tokens(query: str) -> tuple[str, list[str]]:
    """법령명 매칭용 (정규화 질의, 명사 토큰) 반환. 약칭 정규화 후 형태소 분해.

    예) '광업제조업자' → ('광업제조업자', ['광업','제조','업자'])
        '헌법'         → ('대한민국헌법', ['대한민국','헌법'])   # 약칭 정규화
    토큰은 합성어를 쪼개 법령명 LIKE 커버리지에만 쓴다(본문 FTS 엔 미사용 — 흔한 조각이
    노이즈를 유발). kiwipiepy 미설치/실패 시 빈 토큰 → 호출부가 정규화 질의로 폴백.
    """
    raw = (query or "").strip()
    # alias 조회·토큰화 전에 가운뎃점 변형 제거 — '초·중등교육법'/'초중등교육법' 모두 점 없는
    # 형태로 통일해 DB('초ㆍ중등교육법' → SQL 측도 _name_norm_sql 로 점 제거)와 맞춘다.
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
    """`articles` 파라미터 → `(article_no_num, branch | None)` 튜플 리스트.

    LLM 이 자연스럽게 쓰는 한국어 조문 표기를 폭넓게 흡수한다 — 같은 조문을 모두
    동일 spec 으로 정규화:
      - `"347"` / `"제347조"` / `"347조"`              → `(347, None)` — 본조 + 모든 가지(의2 ...)
      - `"347-2"` / `"347의2"` / `"제347조의2"`         → `(347, 2)`    — 제347조의2만 콕
      - 연속 범위는 list로 풀어서: `["3","4","5","6","7"]`

    숫자를 못 찾은 토큰은 무시(graceful). 단 *모든* 토큰이 무시돼 빈 list 가 되면
    호출부(`_statute_lookup_impl`)가 fail-loud 로 형식 안내를 돌려준다(침묵 금지).
    """
    if spec is None:
        return None
    if not isinstance(spec, list):
        spec = [spec]

    out: list[tuple[int, int | None]] = []
    for token in spec:
        # '제76조의2'·'76조의2'·'76의2'·'76-2'·'제76조'·'76' 모두 → (76, 2)/(76, None).
        # '-'(가지 구분)를 '의'로 통일한 뒤 '제…조…의N' 패턴에서 본조번호·가지번호만 추출.
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
               category, has_articles, has_text_content
        FROM st_notices WHERE id=?
        """,
        (nid,),
    ).fetchone()
    return dict(r) if r else None



# ---------- 시점 lookup helper (M28 연혁 적재 활용) ----------

def _today_iso() -> str:
    """오늘 날짜 'YYYYMMDD'. offense_date 미지정 시 '현행' = *오늘 시점 시행본*
    판정 기준. 시행예정(미래 eff>today) 자동 제외 + 시간축 라벨 오류(stale
    '시행예정') 도 날짜로 교정."""
    return datetime.date.today().strftime('%Y%m%d')



# 편/장/절/관 구조 제목 row 판별 — '제22장 성풍속에 관한 죄', '제1편 총칙' 등.
# 이 row 들은 title=NULL 로 적재되며 *바로 뒤 조문의 article_no_num 을 공유*해서
# (예: '제22장'→num 241), 조문 본문 조회 시 엉뚱하게 잡힌다. 조문 본문은 항상
# '제<번호>조' 로 시작하므로 [편장절관] 로 구분.
_HEADING_RE = re.compile(r'^\s*제\s*\d+\s*[편장절관]')


def _is_structural_heading(text: str | None) -> bool:
    """article_text 가 편/장/절/관 구조 제목이면 True (조문 아님 → 조회서 제외)."""
    return bool(text and _HEADING_RE.match(text))


def _get_historic_article(
    conn: sqlite3.Connection,
    law_id: str,
    art_no: int,
    art_br: int | None,
    offense_iso: str,
) -> dict[str, Any] | None:
    """같은 law_id + 조문 spec + offense_iso → *행위시점 이전 최후 버전* 조문 본문.

    M37(f) — 종전엔 `article_changed='Y'`(변경분) row 만 대상이라, *한 번도 개정 안 된*
    조문(baseline 스냅샷 status=None·changed=None 에만 존재 — 형법 §1·§2 등 多)을
    전부 놓쳤다. as_of 이전 *최후 버전*(변경분이든 baseline 이든) 을 채택. 동일
    effective_date 면 변경분('Y') 우선. (offense_date 무영향: 과거 행위는 recent
    baseline 이 effective_date<=offense 에서 자연 제외, 변경분 row 가 그대로 채택됨.)

    편/장/절/관 구조 제목 row(`_is_structural_heading`)는 건너뜀 — 같은 num 을
    공유해 LIMIT 1 로 잡으면 '제22장 …' 같은 장 제목이 §241 본문으로 반환된다.
    제목을 건너뛰면 삭제 조문은 '제N조 삭제 <…>' 마커가 자연 채택된다.
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
    """같은 law_id 시점본 group 중 *as_of 시점 현행* 1개 select.

    우선순위:
      1. eff <= as_of 시행본만 (없으면 group 전체 — 미래본만 적재된 법령 구제)
      2. 폐지(change_kind '폐지') 비선호 — 있으면 직전 시행본
      3. **통합본(full snapshot) 우선** — 연혁/일부개정 row 는 변경분(delta)만 적재돼
         조문이 1~몇 개뿐이다. 최신 amendment 가 lsHistory 에서 '현행' 라벨을 달면
         history_status 만으론 통합본과 구분 안 돼 eff 최신 delta 가 뽑혀 '조문 1개'
         법령으로 깨진다(실측 101개 law_id — 행정소송법 등). → pool 에서 **조문 수가
         가장 많은(=통합본)** 을 최우선, mst NULL(eflaw 풀스냅샷) 차순, 동급이면 최신 eff.
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
    """law_id → as_of 시점 현행 statute_id (`_pick_current_version` 규칙)."""
    group = conn.execute(
        "SELECT id, effective_date, history_status, change_kind, mst "
        "FROM st_statutes WHERE law_id=?",
        (law_id,),
    ).fetchall()
    chosen = _pick_current_version(conn, group, as_of_iso)
    return chosen['id'] if chosen else None


def _is_repealed_as_of(
    conn: sqlite3.Connection, law_id: str | None, as_of_iso: str,
) -> bool:
    """law_id 가 as_of 시점에 폐지 상태인지 — eff<=as_of *최신본* 의 change_kind
    가 '폐지'(타법폐지·폐지)면 True. 폐지 후 같은 law_id 로 재제정되면 최신본이
    非폐지라 False. 전체 이력 조회 — 검색 부분 group 이 폐지 row 를 누락해도 정확.
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
    """search 결과 row 들 중 같은 law_id 면 1개만 select.

    - offense_iso 지정: effective_date <= offense_iso 중 최대 (그 시점 유효)
    - 미지정(현행): `_pick_current_version` — eff<=오늘 중 full snapshot 우선·
      시행예정(미래본) 제외. *오늘 시점 폐지된 법 자체*의 검색 제외는
      `_search_statutes`(offense_iso None 분기)에서 수행.
    """
    by_law: dict[str, list] = {}
    for r in rows:
        lid = r['law_id']
        if not lid:
            # law_id 없는 row (예전 적재) — 단독 group 으로 보존
            by_law.setdefault(f'_solo_{r["id"]}', []).append(r)
        else:
            by_law.setdefault(lid, []).append(r)

    out = []
    for lid, group in by_law.items():
        if len(group) == 1 and lid.startswith('_solo_'):
            out.append(group[0])
            continue
        # 시점 filter
        if offense_iso:
            candidates = [
                r for r in group if r['effective_date'] and r['effective_date'] <= offense_iso
            ]
            if not candidates:
                # 행위시 이전 시점본 없음 — skip (그 시점에 아예 없던 법령)
                continue
            chosen = max(candidates, key=lambda r: r['effective_date'])
        else:
            # offense_date 미지정 = '현행'(오늘 시점). full snapshot 우선 + 시행예정
            # (미래본) 자동 제외 (`_pick_current_version`).
            chosen = _pick_current_version(conn, group, _today_iso())
        out.append(chosen)
    # 원래 정렬 순서 유지 (rank 기반) — chosen 의 첫 등장 순으로 정렬
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
    """law_id 가 offense_iso 시점 (또는 현재) 폐지·시행종료 상태인지.

    return: 폐지 정보 {effective_date, change_kind} 또는 None.
    """
    target_date = offense_iso or '99999999'
    r = conn.execute(
        """SELECT effective_date, change_kind FROM st_statutes
           WHERE law_id=? AND effective_date<=? AND change_kind LIKE '%폐지%'
           ORDER BY effective_date DESC LIMIT 1""",
        (law_id, target_date),
    ).fetchone()
    return dict(r) if r else None


# ---------- 모드 A: 목록 검색 ----------

def _search_statutes(
    conn: sqlite3.Connection,
    query: str | None,
    kind: str | None,
    limit: int,
    *,
    offense_date: str | None = None,
) -> list[dict[str, Any]]:
    """법령명 검색. M28 연혁 적재 후 같은 law_id 의 시점본 다수 row 존재 →
    folding 으로 law_id 별 1개만 노출.

    - default (offense_date 없음): 같은 law_id 중
      *현행* (history_status='현행' 우선, NULL=기존 단일 row fallback) 1개
    - offense_date 지정: 같은 law_id 중 *그 시점 이전 최후 시점본* 1개. 폐지·연혁 포함
    """
    offense_iso = to_iso_date(offense_date)
    where_parts = []
    params: list[Any] = []
    if kind and kind not in NOTICE_KINDS:
        where_parts.append("s.kind = ?")
        params.append(kind)

    # folding buffer — 같은 law_id 의 시점본 다수가 들어와도 1개 select 가능하도록
    sql_limit = max(limit * 8, 80)

    if query:
        # 약칭 정규화 + 형태소 토큰 — 띄어쓰기 없는 합성어("광업제조업자")를 명사 토큰
        # (광업/제조/업자)으로 쪼개 법령명 LIKE 커버리지에 쓴다. 토큰이 없으면(kiwipiepy
        # 미설치 등) 정규화 질의 전체로 폴백. 본문 FTS(아래 MATCH)는 원본 질의 그대로 —
        # 흔한 토큰 조각을 본문에 뿌리면 노이즈가 커지므로 형태소는 법령명에만 적용.
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
        # 법령명에 토큰이 하나라도 들거나 본문 FTS hit 가 있으면 후보.
        where_parts.append(f"(({name_or}) OR h.hits > 0)")
        sql += "WHERE " + " AND ".join(where_parts) + "\n"
        # 완전일치 최상단 → 법령명 토큰 커버리지(몇 개 토큰이 이름에 들었나) → 이름 길이
        # → 본문 hit. 합성어는 커버리지로, 통칭은 약칭맵+완전일치로 정확히 위로 온다.
        sql += """
        ORDER BY exact_name DESC, name_cover DESC, length(s.name) ASC, fts_hits DESC
        LIMIT ?
        """
        # params 순서 = SQL 등장 순: MATCH, cover(SELECT n), exact(=normalized),
        #               [kind(where_parts[0])], name_or(WHERE n), LIMIT
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

    # folding — law_id 별 1개 select
    rows = _fold_versions(conn, rows, offense_iso)
    if offense_iso is None:
        # 현행 검색 — 오늘 시점 *폐지된 법* 제외. 폐지 직전 버전이 '좀비'로
        # 노출되던 것 차단 (현행본은 보통 다른 law_id 에 별도 존재). 과거시점
        # (offense_date) 검색은 그 시점 유효본을 봐야 하므로 미적용.
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
        # 법령명 검색(_search_statutes)과 동일한 형태소 토큰 커버리지 — '개인정보의 안전성
        # 확보조치 기준' 처럼 이름에 조사('의')·띄어쓰기가 섞여도 토큰(개인/정보/안전)이 몇 개
        # 포함됐는지(name_cover)로 랭크. 공백만 지운 연속 substring 매칭은 '의' 하나에 깨졌다.
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
          AND n.category = 'article_form'
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
            WHERE has_text_content=1 AND category='article_form'
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
    """법령·고시 공통 이름-관련도 정렬 키(작을수록 상위). _search_statutes 의 name_cover/
    exact_name 와 동일 지표(같은 형태소 토큰·공백무시 substring): 완전일치 > 토큰 커버리지
    높은 순. 쿼터 없이 법령+고시를 *merit* 으로 인터리브하는 데 쓴다."""
    n = (name or "").replace(" ", "")
    if q_norm and n == q_norm:
        return (0, 0)                          # 완전일치 최상
    if tokens:
        cover = sum(1 for t in tokens if t in n)
        return (1, len(tokens) - cover)        # 커버리지 높을수록(=결손 적을수록) 상위
    return (1, 0) if (q_norm and q_norm in n) else (2, 0)


def _merge_law_notice_matches(matches, query, limit, name_of=None):
    """법령+고시 검색결과를 이름 관련도(_name_cover_key)로 인터리브 — **쿼터 없음**. 안정
    정렬이라 같은 관련도 내에선 입력 순서(법령 먼저·각 코퍼스 내부 랭킹) 유지. 채팅
    statute_lookup 과 웹 search_statutes 가 공용(name_of 로 dict 키만 달리)."""
    if name_of is None:
        name_of = lambda m: m.get("name", "")
    normalized, tokens = _statute_name_tokens(query) if query else ("", [])
    return sorted(matches, key=lambda m: _name_cover_key(name_of(m), normalized, tokens))[:limit]


# ---------- 모드 B: 조문 개요 ----------

def _outline_statute(
    conn: sqlite3.Connection, sid: int, offense_date: str | None = None,
) -> dict[str, Any] | None:
    """statute_id 의 모든 조문 title outline.

    offense_date 지정 시 (M28+M29):
      - 같은 law_id 의 *행위시점 이전 최후 변경* 조문 title 모음
      - 폐지된 조문도 *그 시점 유효* 면 노출
    """
    meta = _statute_meta(conn, sid)
    if meta is None:
        return None

    offense_iso = to_iso_date(offense_date)
    if offense_iso and meta.get('law_id'):
        # 시점본 outline — 같은 law_id 의 모든 (art_no, art_branch) 별 행위시점
        # 최후 변경 row 의 title 모음
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
        # (art_no_num, art_branch) 별 최신 1개만 keep (이미 ORDER BY eff_date DESC)
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

    # offense_date 미지정 = 현행(오늘). 넘어온 sid 가 현행본이 아니어도(연혁/시행예정
    # delta) 오늘 시점 현행 snapshot 으로 redirect — '전체 조문 목록' 보장.
    as_of = _today_iso()
    if meta.get('law_id'):
        cur_id = _resolve_current_statute_id(conn, meta['law_id'], as_of)
        if cur_id is not None and cur_id != sid:
            sid = cur_id
            meta = _statute_meta(conn, sid) or meta

    # title IS NOT NULL 로 거르면 *제목이 아예 없는 법령*이 통째 0건이 된다 —
    # 대한민국헌법('제3조 대한민국의 영토는…'처럼 괄호 제목 없이 조번호로 시작 → 전 조문
    # title=NULL) 과 단일조항 규정('…법정이율에 관한 규정' = 제목·조번호 없는 본문만,
    # article_no_num=0) 이 그 예다(둘 다 웹 페이지엔 본문이 보이는데 outline 만 0건).
    # 웹 SSR(_structured_articles)과 동일하게 **편/장/절/관 구조 헤딩만 제외**하고 실조문은
    # 제목 유무·조번호 0 여부와 무관하게 노출한다. (헤딩 판정에 article_text 필요 → SELECT
    # 포함; num>0 은 (조,가지) 중복행 방어 — 헤딩은 다음 조문과 num 을 공유하나 위에서 제외됨)
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

    # 무구조 단일 본문 규정 — 제N조 구조 없이 article_no_num=0 의 untitled 본문만 있는
    # '…에 관한 규정/규칙' 류(소촉법 법정이율·이자제한법 최고이자율 등 35개). outline 의 빈
    # '- 0:' 만 주면 모델이 본문을 못 가져온다(실측: 모델이 '제1조'로 추측 → detail['1'] →
    # missing → 답변 실패). 본문(이 클래스는 전부 ≤~650자로 짧음)을 곧장 detail 형태로 실어
    # 한 번에 답하게 한다 — 웹 페이지가 본문을 바로 렌더하는 것과 동형.
    if articles and not any((a["no_num"] or 0) > 0 for a in articles):
        return {
            "mode": "detail",
            "statute": statute_kv,
            "articles": articles,
            "missing": [],
            "note": "제N조 구조가 없는 단일 본문 규정 — 아래가 본문 전문입니다.",
        }

    # 일반 법령 — outline 은 제목 목록만(본문 text 제외).
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
    """고시 조문 제목 — 고시는 title 컬럼이 없고 제목이 article_text 의 '제N조(제목) …'
    머리에 들어있다. 괄호 안 제목만 추출(없으면 '')."""
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
        # no = 표시용 조번호(no_str: '1','6의2'). 고시 article_no 는 num*1000+가지 내부값이라
        # 그대로 노출하면 '1000' 처럼 보인다. 제목은 article_text 머리에서 추출.
        articles = [
            {
                "seq": r["article_seq"],
                "no": r["article_no_str"] or str(r["article_no"]),
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


# ---------- 모드 C: 조문 본문 ----------

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

    # M37(f) — 현행·시점 본문 통일. 변경분(delta) 적재 DB 에서 "현행 조문" = "최신 시행
    # 시점의 조문" 이므로, 현행 조회도 offense_date 조회와 *같은* delta-walk
    # (`_detail_statute_at_date` = 조문별 최후 변경분 후방 추적) 로 푼다. 종전엔 현행만
    # folding 된 단일 statute_id (= '현행' delta row, 바뀐 조문만 적재) 를 직접 조회해
    # *최근 개정 안 된 조문* 이 전부 missing 이었다 (86/89 법령). offense_date 없으면
    # as_of = 최신 *시행* 시점 (시행예정 제외 — 미시행 개정분은 현행 아님).
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

    # fallback — law_id 없거나(예전 solo 적재) 시점 산출 불가 시 단일-id 직접 조회
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

    # missing 판정: spec 별로 매칭 row 있는지
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
    """offense_date 시점 본문 응답 — 변경분 적재 row 의 시점 후방 referencing.

    각 spec 의 *행위시점 이전 최후 변경* 본문 모음. 본조 + 가지 모두 처리.
    """
    law_id = meta['law_id']
    # 폐지 여부 점검
    repealed = _is_repealed_at(conn, law_id, offense_iso)
    # 같은 law_id 의 *최후 시점본 메타* (시간 축 비대칭 표시용)
    latest = conn.execute(
        """SELECT effective_date, history_status FROM st_statutes
           WHERE law_id=? ORDER BY effective_date DESC LIMIT 1""",
        (law_id,),
    ).fetchone()

    articles: list[dict] = []
    missing: list[str] = []
    for num, branch in specs:
        # branch=None (본조 + 가지 모두) → law_id 의 article_no_num=num 의 *모든 branch*
        # 의 시점 lookup (각 branch 별 최후 변경)
        if branch is None:
            # 같은 num 의 모든 branch 가능성 — 한번에 lookup
            branches_rows = conn.execute(
                """SELECT DISTINCT a.article_branch FROM st_articles a
                   JOIN st_statutes s ON s.id=a.statute_id
                   WHERE s.law_id=? AND a.article_no_num=?
                     AND s.effective_date <= ?""",  # M37(f) — baseline 가지 포함 (Y 필터 제거)
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
    """고시 article_no = num*1000 + 가지(제6조의2 = 6002). spec (num, branch) 를 그 인코딩으로
    변환해 매칭 — outline 이 보여주는 자연 조번호('1','6의2')를 그대로 다시 넘기면 조회된다."""
    meta = _notice_meta(conn, nid)
    if meta is None:
        return None
    cat = meta["category"]

    if cat == "article_form":
        # branch 지정 시 정확히(num*1000+br), 미지정(본조) 시 그 천단위 블록 전체(본조+모든 가지).
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
                "no": r["article_no_str"] or str(r["article_no"]),
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


# ---------- markdown 직렬화 ----------

def _fmt_article_no(no: Any, branch: Any) -> str:
    """article 번호 표기 — branch 있으면 '347-2' 형태."""
    return f"{no}-{branch}" if branch else f"{no}"


def _statute_web_url(d: dict[str, Any]) -> str | None:
    """법령/고시 dict(detail 의 statute 또는 search match) → 법령 페이지 url.
    법령 = /statutes/{law_id}, 고시 = /statutes/admrul-{st_notices.id}. 해석 불가 시 None.
    판례 도구의 `url` 필드와 동형 — 모델이 인용할 때 마크다운 링크로 제시한다."""
    if not d:
        return None
    law_id = d.get("law_id")
    if law_id:
        return f"{case_url_base()}/statutes/{law_id}"
    # 고시: law_id 없음 + category/notice_id 보유. id(detail) 또는 statute_id(match) = st_notices.id
    if d.get("category") or d.get("notice_id"):
        nid = d.get("id")
        if not isinstance(nid, int):
            nid = d.get("statute_id")
        if isinstance(nid, int):
            return f"{case_url_base()}/statutes/{NOTICE_URL_PREFIX}{nid}"
    return None


def _statute_article_url(stt: dict[str, Any] | None, art: dict[str, Any]) -> str | None:
    """법령 조문 단위 페이지 url /statutes/{law_id}/{jo} (jo='750'·'839의2'). 고시(law_id
    없음)·조번호 결손 시 None — 그땐 법령 단위 url 만 노출. jo 슬러그는 web `_jo_slug` 와 동일
    규칙(가지 있으면 'N의M', 없으면 'N')."""
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
    """statute_lookup 응답 dict → markdown-KV 문자열."""
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
            mid = m.get("statute_id") or m.get("id")
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
        sid = stt.get("id")
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
                # title 이 None 일 수 있음(제목 없는 법령 — 헌법·단일조항 규정). a.get(...,'')
                # 는 키가 None 값으로 존재하면 'None' 을 그대로 찍으므로 `or ''` 로 정규화.
                lines.append(f"### {num} {a.get('title') or ''}")
                art_url = _statute_article_url(stt, a)
                if art_url:
                    lines.append(f"- url: {art_url}")
                if a.get("text"):
                    # 조문마다 두 줄(provenance + quote_eligible)이 나가던 것을 표지 하나로 합쳤다.
                    # 조회 하나가 조문 수십 개를 담으므로 줄당 비용이 그대로 곱해진다.
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
    statute_id: int | None = None,
    articles: list[str | int] | str | int | None = None,
    kind: str | None = None,
    limit: int = 10,
    offense_date: str | None = None,
) -> str:
    """법령·고시 조회 — 법령명/키워드 검색과 조문 본문 확인(현행 또는 행위시점 기준).

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
    - **위 quick-access 목록 밖의 statute_id 는 추측 금지** (DB primary key — 예: 574 는
      근로기준법이 아니라 형사소송법). 모르면 query 로 검색해 받은 id 를 쓰세요.
    - articles 미지정 시 outline (모든 조문 title) — 형법처럼 큰 법령은 응답이 거대해지니
      가능하면 articles 명시.

    응답: markdown-KV. 상세 조문·고시 본문은 `text_kind: 공식 … 원문`으로 표시되며 그대로
    직접인용할 수 있습니다. 답에 쓴 조문은 직접 인용이든 요약이든 반환 url(법령/조문 페이지)을
    링크로 함께 제시하세요.

    Args:
      query: 법령명 또는 본문 키워드. id 모를 때 검색용. 받은 목록에서 id 를 골라 재호출.
      statute_id: int (DB primary key). 한글 이름 받지 않음. 모르면 query 로 검색(추측 금지).
      articles: 조문 번호 리스트(문자열/정수 모두 허용) **최대 8개**. 아래 표기 모두 허용(자동 정규화):
        - "347" / "제347조": 347조 + 가지(의2, 의3 ...) 함께
        - "347-2" / "347의2" / "제347조의2": 제347조의2만 콕
        - 연속 범위: ["3","4","5","6","7"]
        고시는 가지 없음.
      kind: "법률"|"대통령령"|"부령"|"고시" 필터. "고시"면 고시 테이블.
      limit: 검색 모드 최대 결과 수(기본 10, 최대 50).
      offense_date: 행위 일자 (예: '2013.7.30', '20130730'). 지정 시 *행위시점 기준*
        본문/시점본 응답 — 형법 §1 ① "범죄의 성립과 처벌은 행위시의 법률에 의한다" 원칙.
        **미지정 시 오늘 날짜(시스템 시계) 기준 현행본** — 미래 시행예정본은 자동 제외.
    """
    query = coerce_str(query)
    kind = coerce_str(kind)
    limit = min(max(coerce_int(limit) or 10, 1), LIMIT_MAX)
    statute_id = coerce_int(statute_id)
    articles = coerce_list(articles)
    is_notice_target = kind in NOTICE_KINDS

    if statute_id is None and not query:
        return _format_response_md({
            "status": "missing_input",
            "message": (
                "query(법령명·키워드) 또는 statute_id(DB 식별자) 중 하나는 필요합니다. "
                "법령을 모르면 query로 검색한 뒤 받은 statute_id로 재호출하세요."
            ),
        })

    if articles is not None and len(articles) > ARTICLES_MAX:
        return _format_response_md({
            "status": "too_many_articles",
            "input": {"statute_id": statute_id, "n_articles": len(articles)},
            "message": (
                f"한 호출당 articles는 최대 {ARTICLES_MAX}개까지. "
                f"{len(articles)}개 요청됨 — 작은 묶음으로 나눠 호출해 주세요."
            ),
        })

    conn = open_db()
    try:
        return _format_response_md(
            _statute_lookup_impl(
                conn, query, statute_id, articles, kind, limit, is_notice_target,
                offense_date=offense_date,
            )
        )
    finally:
        conn.close()


def _bad_articles_response(statute_id: int | None, articles: list[str | int]) -> dict[str, Any]:
    """articles 토큰을 한 개도 못 읽었을 때의 fail-loud 안내(침묵 금지)."""
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
    statute_id: int | None,
    articles: list[str | int] | None,
    kind: str | None,
    limit: int,
    is_notice_target: bool,
    *,
    offense_date: str | None = None,
) -> dict[str, Any]:
    if statute_id is None:
        if is_notice_target:
            matches = _search_notices(conn, query, limit)
        elif kind:
            matches = _search_statutes(
                conn, query, kind, limit,
                offense_date=offense_date,
            )
        else:
            # 쿼터 없음 — 법령·고시 각각 limit 까지 뽑아 이름 관련도(_merge_law_notice_matches)
            # 로 인터리브. 이름이 맞는 고시는 위로, 본문만 스친 고시는 법령 아래로 자연 정렬.
            stat_matches = _search_statutes(
                conn, query, None, limit,
                offense_date=offense_date,
            )
            notice_matches = _search_notices(conn, query, limit) if query else []
            matches = _merge_law_notice_matches(stat_matches + notice_matches, query, limit)
        out = {
            "status": "ok", "mode": "list",
            "matches": matches,
            "offense_date": to_iso_date(offense_date),
        }
        # query 와 articles 를 함께 줬는데 statute_id 가 없으면 본문은 못 준다(조문은
        # 법령-종속). articles 를 *조용히 무시* 하지 않고 two-step 으로 안내(fail-loud).
        if articles:
            out["note"] = (
                "본문 조회는 statute_id 가 필요합니다 — 아래 matches 에서 맞는 법령의 "
                "id 를 골라 statute_id+articles 로 재호출하세요."
            )
        return out

    sid = statute_id  # 이미 coerce_int로 정수화됨
    specs = _parse_articles(articles)

    # articles 를 줬는데 한 토큰도 못 읽으면 *침묵 금지* — 빈 detail 로 떨어지면 모델이
    # 왜 본문이 안 나오는지 단서가 없어 형식을 바꿔가며 헛돈다(실측). 형식 안내로 fail-loud.
    if articles and not specs:
        return _bad_articles_response(statute_id, articles)

    if is_notice_target:
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
            "input": {"statute_id": statute_id, "kind": kind},
        }
    return {"status": "ok", **res}
