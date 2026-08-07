"""precedent_search — 판례 retrieval (chapter 8 §4.2).

USE_DENSE 로 두 경로 분기:
  - USE_DENSE=0 (default): **FTS-only** = OR-모드 BM25(`_or_match`) + snippet 프리뷰. NIM 미사용.
  - USE_DENSE=1 (연구/eval 명시 opt-in): FTS5 BM25(AND) + dense(prec_vec NIM 임베딩) RRF + rerank.
    공개 NIM 임베딩이 호출당 ~18s 라 dense 를 빼고, known-item A/B(N=140)에서 동급+인 OR-모드
    키워드 검색으로 대체(상수 USE_DENSE 주석 참조). 핵심: OR-모드여야 함(암묵 AND 는 긴 질의 붕괴).

query <2자면 input validation 에러. 3자 이상은 trigram, 2자 죄명은 형태소 FTS가 보강한다.

`ctx.deps.embed`로 OpenAI 호환 client (NIM `/v1/embeddings`) 사용.
`ctx.deps.rerank`로 NIM rerank (`/v1/retrieval/{model}/reranking`) 사용.
"""
from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Sequence

from pydantic_ai import RunContext

from ..config import case_url_base
from ..deps import HarnessDeps, open_db
from ._coerce import coerce_int, coerce_str
from ._dedup import dedup_guard
from ._morph import kiwi as _kiwi

EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/llama-nemotron-embed-1b-v2")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "nvidia/llama-nemotron-rerank-1b-v2")
# NIM Matryoshka truncation — prec_vec dim 1024 스키마 유지.
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))

# Retrieval 튜닝 상수 (LLM 노출 X — 시스템 고정).
LIMIT = 8            # 응답 최대 건수. 5→8(2026-06): A/B 에서 recall@10≫@5(짧은 질의), 리랭크 제거로
                     #   BM25 순위가 거칠어 후보 더 줘서 헷지. 8×(~250자 프리뷰+메타)≈~900토큰=무시가능.
RRF_K = 60           # Reciprocal Rank Fusion 파라미터
OVERSAMPLE = 5       # 후보 풀 = LIMIT * OVERSAMPLE (필터 후 LIMIT 못 채울 위험 ↓)
USE_RERANK = os.environ.get("USE_RERANK", "1") == "1"  # default ON, ablation 시 "0"
# dense(임베딩) 신호 on/off. 외부 NIM 전송이 없는 FTS-only가 안전한 기본이다.
# 연구/eval에서 hybrid가 필요할 때만 Environment=USE_DENSE=1로 명시한다.
#   왜: 공개 NIM 임베딩이 호출당 ~18s 라 챗 판례검색 병목. known-item recall A/B(N=140)에서
#   OR-모드 BM25(fts_or)가 dense 와 동급 이상(짧은 질의 동률, 풍부한 질의 1.00 vs 0.80) →
#   dense 의 18s 값어치 없음. 단 반드시 OR-모드 FTS 와 함께여야 함(암묵 AND 는 0.11 로 붕괴).
USE_DENSE = os.environ.get("USE_DENSE", "0") == "1"
FTS_OR_MAX_TOKENS = int(os.environ.get("FTS_OR_MAX_TOKENS", "40"))   # OR 질의 토큰 상한
SNIPPET_TOKENS = int(os.environ.get("SNIPPET_TOKENS", "256"))        # snippet() 발췌 토큰(≈자). 64→256
                                                                     #   (2026-06): 66자=업계(Tavily 500·Exa)
                                                                     #   대비 과소 → ~250자(판시 [1][2] 포함)로.
PREVIEW_TOP_K = int(os.environ.get("PREVIEW_TOP_K", "3"))           # preview 추출 문장 최대 수
PREVIEW_MAX_CHARS = int(os.environ.get("PREVIEW_MAX_CHARS", "400")) # preview 누적 길이 cap (strict, 단 최소 1문장 보장)
RERANK_INPUT_MAX_CHARS = int(os.environ.get("RERANK_INPUT_MAX_CHARS", "4000"))  # rerank input head cap (p99 outlier 방어)
PREVIEW_FALLBACK_CHARS = 200                                         # USE_RERANK=0 fallback 컷

# FTS5 query sanitize: 공용 모듈로 분리 (`harness/tools/_fts.py`). statute_lookup도
# 동일 sanitize 사용해 v6의 syntax error 2건 해소.
from ._fts import safe_fts_query as _safe_fts_query


# ---------- 사건번호 직접 라우팅 (Layer 0) ----------
# prec_cases_fts 는 case_number 를 인덱싱하지 않아 "2010다89012" 같은 사건번호는
# FTS·임베딩 어느 쪽으로도 못 잡는다. 형식이 규칙적(연도+사건부호+번호)이라 정규식으로
# 감지해 case_number 직접 조회로 라우팅한다. (웹 레이어 harness_repo 도 이걸 import 해 공유.)
_CASE_NO_RE = re.compile(r"\d{2,4}[가-힣]{1,5}\d+")
# 표준 인용 형식의 사건번호: '선고' 직후 또는 '판결/결정' 직전. 한국 법조 인용은
# '법원명 + 선고일 + 선고 + 사건번호 + 판결/결정' 꼴(선고일은 생략되기도 함)이라
# 사건번호 앞뒤로 날짜·법원명·꼬리말이 붙어 fullmatch 가 깨진다. '선고/판결/결정' 은
# 사건부호 집합(167종, '다/두/노/헌마' …)에 없어 부호로 오인될 일 없는 강한 컨텍스트
# → 그에 인접한 토큰만 안전하게 떼어낸다(일반 질의는 이 인접 조건에 안 걸려 통과).
_CASE_NO_CITED_RES = (
    re.compile(r"선고(\d{2,4}[가-힣]{1,5}\d+)"),            # '…선고 2010다89012 …'
    re.compile(r"(\d{2,4}[가-힣]{1,5}\d+)(?=판결|결정)"),   # '… 2018노1234 판결'
)


def _extract_case_number(query: str) -> str | None:
    """사건번호 직접조회 라우팅 키(정규화 사건번호) 또는 None. 두 형태를 잡는다:

    ① 표준 인용 형식 — '대법원 2010. 5. 26. 선고 2010다89012 판결'처럼 법원명·선고일·
       꼬리말이 섞여도 '선고' 직후 또는 '판결/결정' 직전 사건번호를 추출. 이 강한
       컨텍스트가 인접할 때만 임베디드 추출하므로 일반 질의를 오인하지 않는다.
    ② 단독 사건번호 — 질의 전체가 사건번호(+'판결/결정' 등 접미). '민법 제750조' 같은
       일반 질의 오인 방지를 위해 *전체 일치*만 허용.
    """
    raw = (query or "").strip().replace(" ", "")
    for rx in _CASE_NO_CITED_RES:
        m = rx.search(raw)
        if m:
            return m.group(1)
    q = re.sub(r"(전원합의체|판결|결정|선고|사건|판례)$", "", raw)
    return q if _CASE_NO_RE.fullmatch(q) else None


# ---------- court_level 정규화 + 슬롯 오용 안내 (fail-loud) ----------
# prec_cases.court_level DB enum 은 이 4값뿐. LLM 이 '고등법원'·'지방법원' 등 법원 *이름*
# 으로 주면 SQL 필터가 통째로 0건이 돼 *조용히* 빈 결과가 나간다(실측). 별칭 매핑 +
# 미지값 무시+안내로 침묵 0건 방지.
_VALID_COURT_LEVELS = ("1심", "2심", "대법원", "헌재")
_COURT_LEVEL_ALIASES = {
    "고등법원": "2심", "고법": "2심", "항소심": "2심", "2심법원": "2심",
    "지방법원": "1심", "지법": "1심", "1심법원": "1심", "단독": "1심", "1심판결": "1심",
    "대법": "대법원", "대법원판결": "대법원",
    "헌법재판소": "헌재",
}


def _normalize_court_level(cl: str | None) -> tuple[str | None, str | None]:
    """court_level → (DB enum|None, note|None). 유효값/별칭은 enum 으로, 미지값은
    필터 무시(None) + note 로 안내(전체 0건으로 조용히 떨어지지 않게)."""
    if not cl:
        return None, None
    if cl in _VALID_COURT_LEVELS:
        return cl, None
    if cl in _COURT_LEVEL_ALIASES:
        return _COURT_LEVEL_ALIASES[cl], None
    return None, f"court_level '{cl}' 은 인식 못 해 무시함 — 유효값: 1심/2심/대법원/헌재"


def _case_no_in_query_hint(query: str | None) -> str | None:
    """query 슬롯에 사건번호를 넣어 0건이 난 경우의 안내 — case_number 슬롯으로 유도."""
    if _CASE_NO_RE.fullmatch((query or "").replace(" ", "")):
        return ("query 가 사건번호 형식입니다 — 사건번호로 특정 판례를 찾으려면 "
                "case_number 인자에 넣으세요(query 는 사실관계 키워드 전용).")
    return None


# ---------- 3 ranker ----------

def _fts_rank(conn: sqlite3.Connection, query: str, limit: int) -> list[int]:
    safe = _safe_fts_query(query)
    words = [w for w in safe.split() if len(w) >= 3]
    if not words:
        return []
    rows = conn.execute(
        """
        SELECT rowid FROM prec_cases_fts
        WHERE prec_cases_fts MATCH ?
        ORDER BY bm25(prec_cases_fts) LIMIT ?
        """,
        (safe, limit),
    ).fetchall()
    return [r["rowid"] for r in rows]


def _or_match(query: str) -> str:
    """변별 토큰 OR 매치식. 긴/구절형 query 의 암묵 AND 폭발(전토큰 동시매치=0건)을 피해
    bag-of-words BM25 로 검색하기 위함. 토큰=≥3자(trigram floor), 중복 제거, 상한 컷.
    known-item A/B(N=140): AND-mode fts_facts 0.11 → OR-mode fts_or_facts 1.00."""
    toks = [w for w in _safe_fts_query(query).split() if len(w) >= 3]
    seen: set[str] = set()
    toks = [t for t in toks if not (t in seen or seen.add(t))][:FTS_OR_MAX_TOKENS]
    return " OR ".join(toks)


_FILTER_TABLE: str | None = None


def _filter_table(conn: sqlite3.Connection) -> str:
    """필터(법원·연도) JOIN 대상 테이블명. content_md(본문 14GB) 없는 슬림 사본 prec_meta 가
    있으면 그걸 쓴다 — 필터 검색 시 매치 수만 건을 prec_cases 에서 룩업하면 본문 페이지까지
    throttled 디스크에서 읽혀 폭증(실측 27s)하는데, prec_meta(수 MB, 캐시 상주)로 JOIN 하면 1s.
    prec_meta 부재 시 prec_cases 폴백(안전). 프로세스당 1회 조회 후 캐시(재시작 시 갱신)."""
    global _FILTER_TABLE
    if _FILTER_TABLE is None:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prec_meta'"
        ).fetchone()
        _FILTER_TABLE = "prec_meta" if row else "prec_cases"
    return _FILTER_TABLE


def _fts_or_rank(
    conn: sqlite3.Connection,
    match: str,
    limit: int,
    *,
    court_level: str | None = None,
    court_name: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[int]:
    """OR-모드 BM25 랭킹 (dense 없이 단일 신호로 견고). match = _or_match() 결과.

    필터(심급·법원/지역·연도)는 prec_cases JOIN 으로 **SQL 레벨에서** 적용한다 — 상위 pool 을
    먼저 뽑고 나중에 거르면 희소 필터(예: 흔한 키워드+'부산')가 전역 상위권 밖이라 통째 사라진다
    (웹 `_web_fts_rank` 와 동일 원칙)."""
    if not match:
        return []
    cond = ["f.prec_cases_fts MATCH ?"]
    args: list[Any] = [match]
    if court_level:
        cond.append("c.court_level = ?"); args.append(court_level)
    if court_name:
        cond.append("c.court_name LIKE ?"); args.append(f"%{court_name}%")
    if year_from is not None:
        cond.append("COALESCE(c.decision_year, c.case_year) >= ?"); args.append(year_from)
    if year_to is not None:
        cond.append("COALESCE(c.decision_year, c.case_year) <= ?"); args.append(year_to)
    # 필터가 없으면 prec_cases JOIN 을 생략한다. JOIN 을 걸면 흔한 토큰 OR 매치(수만 건)를
    # 전부 prec_cases 에서 룩업해 throttled 디스크에서 폭증한다(실측 5.97s). FTS 단독 bm25 는 0.09s
    # (60×). JOIN 은 오직 필터(법원·연도)에만 필요하므로 그때만 건다.
    if len(cond) == 1:
        rows = conn.execute(
            "SELECT rowid FROM prec_cases_fts WHERE prec_cases_fts MATCH ? "
            "ORDER BY bm25(prec_cases_fts) LIMIT ?",
            (match, limit),
        ).fetchall()
        return [r["rowid"] for r in rows]
    # 필터 JOIN 은 본문 없는 prec_meta 로(폭증 회피).
    sql = (
        f"SELECT f.rowid FROM prec_cases_fts f JOIN {_filter_table(conn)} c ON c.id = f.rowid "
        f"WHERE {' AND '.join(cond)} ORDER BY bm25(f.prec_cases_fts) LIMIT ?"
    )
    rows = conn.execute(sql, (*args, limit)).fetchall()
    return [r["rowid"] for r in rows]


# ---------- 형태소 FTS (prec_cases_morph_fts) — 2글자 단어 검색 보강 ----------
# trigram 은 3글자 미만('살인'·'사기')을 못 잡는다. Kiwi 형태소로 색인한 별도 FTS 를
# RRF 로 융합해 단어 단위(2글자 포함) 매칭을 더한다. 색인은 build_morph_index.py.
# Kiwi 인스턴스는 `_morph.kiwi()` 하나를 statutes 도구와 공유한다 — 모듈마다 `lru_cache` 를
# 두면 프로세스에 둘이 뜬다(2026-07-30 실측: 두 번째가 RSS +246MB).
_MORPH_KEEP_TAGS = ("NNG", "NNP", "NNB", "NR", "NP", "SL", "SN", "SH", "XR", "VV", "VA")


def _morph_match(query: str) -> str:
    """질의 → 형태소 OR 매치식(2글자 floor 없음 — trigram 이 못 잡는 단어 보강).
    각 토큰을 따옴표로 감싸 FTS5 연산자(OR/AND/영문 morpheme) 충돌 방지."""
    try:
        toks = [t.form.strip() for t in _kiwi().tokenize(query or "")
                if t.tag in _MORPH_KEEP_TAGS and t.form.strip()]
    except Exception:
        return ""
    toks = list(dict.fromkeys(toks))[:FTS_OR_MAX_TOKENS]
    return " OR ".join(f'"{t}"' for t in toks)


def _morph_rank(
    conn: sqlite3.Connection,
    match: str,
    limit: int,
    *,
    court_level: str | None = None,
    court_name: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[int]:
    """형태소 FTS BM25 랭킹. _fts_or_rank 와 동일한 SQL 레벨 필터·정렬. 형태소 FTS 가
    아직 안 빌드됐으면(테이블 부재) 빈 list 반환 — 점진 배포 시 trigram 만으로 동작."""
    if not match:
        return []
    cond = ["f.prec_cases_morph_fts MATCH ?"]
    args: list[Any] = [match]
    if court_level:
        cond.append("c.court_level = ?"); args.append(court_level)
    if court_name:
        cond.append("c.court_name LIKE ?"); args.append(f"%{court_name}%")
    if year_from is not None:
        cond.append("COALESCE(c.decision_year, c.case_year) >= ?"); args.append(year_from)
    if year_to is not None:
        cond.append("COALESCE(c.decision_year, c.case_year) <= ?"); args.append(year_to)
    # 필터 없으면 JOIN 생략(=대폭 가속, _fts_or_rank 주석 참조). 흔한 2글자 토큰
    # OR 매치는 수만 건이라 JOIN 룩업이 throttled 디스크에서 폭증(실측 6s→0.02s).
    try:
        if len(cond) == 1:
            rows = conn.execute(
                "SELECT rowid FROM prec_cases_morph_fts WHERE prec_cases_morph_fts MATCH ? "
                "ORDER BY bm25(prec_cases_morph_fts) LIMIT ?",
                (match, limit),
            ).fetchall()
            return [r["rowid"] for r in rows]
        sql = (
            f"SELECT f.rowid FROM prec_cases_morph_fts f JOIN {_filter_table(conn)} c ON c.id = f.rowid "
            f"WHERE {' AND '.join(cond)} ORDER BY bm25(f.prec_cases_morph_fts) LIMIT ?"
        )
        rows = conn.execute(sql, (*args, limit)).fetchall()
    except sqlite3.OperationalError:
        return []  # 형태소 FTS 미빌드 — trigram 만으로 폴백
    return [r["rowid"] for r in rows]


def _fts_snippets(
    conn: sqlite3.Connection, ids: Sequence[int], match: str
) -> dict[int, tuple[str, str]]:
    """매칭 판례별 ``(발췌, 출처종류)``.

    FTS의 자동 column 선택(-1)은 generated_summary와 원문을 구별할 수 없어 생성 요약을
    판결문 직접인용으로 오인하게 만든다. 열별 marker를 검사해 provenance를 함께 반환한다.
    """
    if not ids or not match:
        return {}
    ph = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""
        SELECT rowid,
               snippet(prec_cases_fts, 1, '⟦', '⟧', '…', {SNIPPET_TOKENS}) AS original_s,
               snippet(prec_cases_fts, 2, '⟦', '⟧', '…', {SNIPPET_TOKENS}) AS official_s,
               snippet(prec_cases_fts, 4, '⟦', '⟧', '…', {SNIPPET_TOKENS}) AS generated_s,
               snippet(prec_cases_fts, 3, '⟦', '⟧', '…', {SNIPPET_TOKENS}) AS statute_s
        FROM prec_cases_fts
        WHERE prec_cases_fts MATCH ? AND rowid IN ({ph})
        """,
        [match, *ids],
    ).fetchall()
    out: dict[int, tuple[str, str]] = {}
    # 판결·결정 원문을 우선하되, 없으면 공식요약→생성요약→참조조문 메타 순서.
    columns = (
        ("original_s", "original_text_excerpt"),
        ("official_s", "official_summary_excerpt"),
        ("generated_s", "generated_summary_excerpt"),
        ("statute_s", "reference_statute_metadata_excerpt"),
    )
    for row in rows:
        for key, source in columns:
            raw = row[key] or ""
            if "⟦" not in raw:  # 이 열에는 실제 query hit 없음
                continue
            clean = re.sub(r"[\u27e6\u27e7]", "", raw)
            clean = " ".join(re.sub(r"<br\s*/?>", " ", clean).split())
            if clean:
                out[row["rowid"]] = (clean, source)
                break
    return out


# ---------- 매칭 지점 발췌 (trigram snippet 미스 = 2글자 형태소-only 보완) ----------
# trigram snippet.은 3자+만 매칭 중심 윈도우를 뜬다. 2글자어('살인'·'부당')는 snippet 이
# 비어 요약 머리로 떨어져 *왜 매칭됐는지* 안 보인다. 본문에서 매칭 지점 윈도우를 직접 발췌해
# chat preview·웹 하이라이트 양쪽이 '나오는 부분'을 보여주도록 — 웹 harness_repo 가 공유 import.
_BR_RE = re.compile(r"<br\s*/?>", re.I)


def _excerpt_around(text: str, terms: Sequence[str], half: int = 60) -> str | None:
    """text 에서 terms 중 *가장 먼저* 등장하는 위치 중심 ±half 윈도우 평문 발췌(없으면 None)."""
    if not text or not terms:
        return None
    clean = _BR_RE.sub(" ", text)
    pos, hit = -1, None
    for t in terms:  # terms 는 긴 토큰 우선 — 동위치면 긴 단어 채택
        i = clean.find(t)
        if i != -1 and (pos == -1 or i < pos):
            pos, hit = i, t
    if pos == -1:
        return None
    start = max(0, pos - half)
    end = min(len(clean), pos + len(hit) + half)
    seg = " ".join(clean[start:end].split())
    if start > 0:
        seg = "…" + seg
    if end < len(clean):
        seg = seg + "…"
    return seg


def _preview_terms(query: str) -> list[str]:
    """프리뷰 발췌·매칭용 토큰 — trigram 단어(≥3자) + 형태소(2자 포함), 긴 것 우선
    (긴 단어가 더 변별적이라 먼저 매칭). 발췌 윈도우의 앵커로 쓴다."""
    toks = [w for w in _safe_fts_query(query).split() if len(w) >= 3]
    try:
        toks += [t.form for t in _kiwi().tokenize(query or "")
                 if t.tag in _MORPH_KEEP_TAGS and t.form.strip()]
    except Exception:
        pass
    return sorted({t for t in toks if t}, key=len, reverse=True)


def _body_excerpts(
    conn: sqlite3.Connection,
    ids: Sequence[int],
    terms: Sequence[str],
    *,
    with_provenance: bool = False,
) -> dict[int, str] | dict[int, tuple[str, str]]:
    """trigram snippet 미스 id의 공식요약→참조조문→본문head에서 매칭 윈도우 발췌.

    AI generated_summary는 제외한다. ``with_provenance``이면 ``(발췌, 출처종류)``를 반환하고,
    기본은 웹 검색 공유 계약을 위해 기존 ``{id: 평문}`` 형태를 유지한다. content_md는 50K 컷.
    """
    if not ids or not terms:
        return {}
    ph = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""SELECT id, summary, reference_statute,
                   substr(COALESCE(content_md,''), 1, 50000) AS content_head
            FROM prec_cases WHERE id IN ({ph})""",
        list(ids),
    ).fetchall()
    out: dict[int, Any] = {}
    for r in rows:
        if not with_provenance:
            # 웹 검색 공유 경로는 기존 공식요약→참조조문→본문 순서를 보존한다.
            body = " ".join((
                r["summary"] or "",
                r["reference_statute"] or "",
                r["content_head"] or "",
            ))
            seg = _excerpt_around(body, terms)
            if seg:
                out[r["id"]] = seg
            continue
        # 챗 경로는 어느 필드에서 나온 발췌인지 분리한다. 기존 결합 순서와 같은 우선순위.
        candidates = (
            (r["summary"] or "", "official_summary_excerpt"),
            (r["reference_statute"] or "", "reference_statute_metadata_excerpt"),
            (r["content_head"] or "", "original_text_excerpt"),
        )
        for text, source in candidates:
            seg = _excerpt_around(text, terms)
            if seg:
                out[r["id"]] = (seg, source)
                break
    return out


def _dense_rank(
    conn: sqlite3.Connection,
    embed_client: Any,
    query: str,
    limit: int,
) -> list[int]:
    """Embed the query through an OpenAI-compatible client, then sqlite-vec KNN.

    Only reached when USE_DENSE=1, which is also the only path that needs the
    sqlite-vec package — hence the local import.

    ``input_type=query`` marks the asymmetric-retrieval side (the index was
    built with "passage") and ``dimensions`` truncates a Matryoshka embedding
    to the stored width. Both are outside the OpenAI schema, so they travel in
    ``extra_body``.
    """
    import sqlite_vec

    resp = embed_client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
        dimensions=EMBED_DIM,
        extra_body={"input_type": "query"},
    )
    qv = sqlite_vec.serialize_float32(resp.data[0].embedding)
    rows = conn.execute(
        """
        SELECT rowid FROM prec_vec
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance
        """,
        (qv, limit),
    ).fetchall()
    return [r["rowid"] for r in rows]


# ---------- RRF fusion ----------

def _rrf_fuse(
    rankings: Sequence[Sequence[int]], *, k: int = 60, limit: int
) -> list[int]:
    """Reciprocal Rank Fusion. score(d) = Σ 1/(k + rank_r(d))."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ordered[:limit]]


# ---------- 메타 hydration + 필터 ----------

def _hydrate(
    conn: sqlite3.Connection,
    case_ids: Sequence[int],
    court_level: str | None,
    year_from: int | None,
    year_to: int | None,
    cap: int | None = LIMIT,
    court_name: str | None = None,
) -> list[dict[str, Any]]:
    """case_ids 순서대로 메타 hydrate. cap=None이면 전체 반환 (rerank/최신정렬용).

    court_name: 법원 *이름/지역* 부분매칭(LIKE) — '부산'→부산지방법원·부산고등법원,
      '특허법원'·'서울행정법원' 등. court_level(심급)과 직교. 웹 search_cases 와 동일 방식.
    """
    if not case_ids:
        return []
    placeholders = ",".join("?" for _ in case_ids)
    where = [f"id IN ({placeholders})"]
    params: list[Any] = list(case_ids)
    if court_level:
        where.append("court_level = ?")
        params.append(court_level)
    if court_name:
        where.append("court_name LIKE ?")
        params.append(f"%{court_name}%")
    if year_from is not None:
        where.append("COALESCE(decision_year, case_year) >= ?")
        params.append(year_from)
    if year_to is not None:
        where.append("COALESCE(decision_year, case_year) <= ?")
        params.append(year_to)

    rows = conn.execute(
        f"""
        SELECT id, case_number, case_name, court_name, court_level,
               COALESCE(decision_year, case_year) AS year, reference_statute,
               COALESCE(
                   NULLIF(TRIM(generated_summary), ''),
                   NULLIF(TRIM(summary), ''),
                   NULLIF(TRIM(content_md), '')
               ) AS display_summary,
               CASE
                   WHEN NULLIF(TRIM(generated_summary), '') IS NOT NULL THEN 'generated_summary_excerpt'
                   WHEN NULLIF(TRIM(summary), '') IS NOT NULL THEN 'official_summary_excerpt'
                   ELSE 'original_text_excerpt'
               END AS display_summary_source
        FROM prec_cases
        WHERE {' AND '.join(where)}
        """,
        params,
    ).fetchall()

    by_id = {r["id"]: r for r in rows}
    out: list[dict[str, Any]] = []
    for cid in case_ids:
        if cid not in by_id:
            continue
        r = by_id[cid]
        full_summary = (r["display_summary"] or "").strip()
        out.append(
            {
                "id": r["id"],
                "case_number": r["case_number"],
                "case_name": r["case_name"],
                "court_name": r["court_name"],
                "court_level": r["court_level"],
                "year": r["year"],
                "reference_statute": r["reference_statute"],
                "_full_summary": full_summary,   # internal: rerank input + sentence source
                "_full_summary_source": r["display_summary_source"],
                "preview": "",                   # filled by sentence-level rerank
            }
        )
        if cap is not None and len(out) >= cap:
            break
    return out


def _rerank_scores(
    client: Any, query: str, texts: list[str]
) -> list[float]:
    """NVIDIA NIM rerank 호출. 반환 score는 *texts 입력 순서*로 정렬됨.

    NIM schema: `{model, query:{text}, passages:[{text},...]}` → `{rankings:[{index, logit}]}`.
    logit 은 unbounded raw score (높을수록 관련). 정렬·비교 의미는 동일.
    """
    if not texts:
        return []
    path = f"/v1/retrieval/{RERANK_MODEL}/reranking"
    resp = client.post(
        path,
        json={
            "model": RERANK_MODEL,
            "query": {"text": query},
            "passages": [{"text": t} for t in texts],
        },
    )
    resp.raise_for_status()
    data = resp.json()
    # NIM 은 logit 내림차순으로 반환 — input 순서로 재정렬
    by_idx = {r["index"]: r["logit"] for r in data["rankings"]}
    # 누락된 index 는 음의 큰 값 (사실상 최하위) 부여 — 0.0 이면 logit 분포상 상위로 오인됨
    return [by_idx.get(i, float("-inf")) for i in range(len(texts))]


def _truncate_rerank(text: str) -> str:
    """rerank 컨텍스트 윈도우 초과 대비 head truncate (p99 ≈ 2.2K자, max 17K outlier)."""
    if len(text) <= RERANK_INPUT_MAX_CHARS:
        return text
    return text[:RERANK_INPUT_MAX_CHARS]


# 한국어 문장 종결: 마침표/물음표/느낌표/。 뒤 공백·줄바꿈.
# 분할 차단 케이스 (lookbehind 2종):
#   - `\d\.` — 날짜·번호 마침표 (예: "2019. 5. 1.", "제3항 1.")
#   - 공백/줄머리 + 가/나/다/라/.../하 + `.` — 한국 판결문 목차 표기 ("가. 사실관계 나. 판단")
_SENT_BOUND = re.compile(
    r"(?<=[.!?。])"
    r"(?<!\d\.)"
    r"(?<![\s(\[][가나다라마바사아자차카타파하]\.)"
    r"\s+"
)


def _split_sentences(text: str) -> list[str]:
    """요약본 → 문장 list. 8자 미만 단편은 직전·직후 문장에 흡수 (단어 조각 방지).

    양방향 흡수가 필요한 이유: 요약본이 "가. 채무자가..." 처럼 *목차 단편으로
    시작*할 때 lookbehind가 빈 앞부분을 보지 못해 단편이 첫 fragment로 떨어짐 →
    직후 정상 문장이 흡수해야 단독 노출 방지.
    """
    if not text:
        return []
    parts = _SENT_BOUND.split(text.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip().replace("\n", " ")
        if not p:
            continue
        # 단편이거나, 직전 fragment가 단편이면 직전과 병합
        if out and (len(p) < 8 or len(out[-1]) < 8):
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


def _sentence_previews_batch(
    client: Any, query: str, summaries: list[str], top_k: int
) -> list[str]:
    """case별 요약본을 한 batch에 묶어 sentence rerank → case별 top_k 문장 추출.

    - 같은 query에 대한 (query, sentence) 쌍이라 case 경계와 무관하게 한 호출에 처리 가능.
    - 선택된 top_k 문장은 *원문 등장 순서*로 join (문맥 흐름 보존).
    - 문장 수 ≤ top_k 인 case는 전체 문장 그대로 반환.
    - 요약본 빈 case는 ""; 분할 0개면 요약본 머리 200자 fallback.
    """
    case_sents: list[list[str]] = [_split_sentences(s) for s in summaries]
    all_sentences: list[str] = []
    boundaries: list[int] = [0]
    for sents in case_sents:
        all_sentences.extend(sents)
        boundaries.append(len(all_sentences))

    if not all_sentences:
        return [(s[:PREVIEW_FALLBACK_CHARS].replace("\n", " ") if s else "") for s in summaries]

    scores = _rerank_scores(client, query, all_sentences)

    previews: list[str] = []
    for ci, sents in enumerate(case_sents):
        if not sents:
            s = summaries[ci]
            previews.append(s[:PREVIEW_FALLBACK_CHARS].replace("\n", " ") if s else "")
            continue
        start, end = boundaries[ci], boundaries[ci + 1]
        case_scores = scores[start:end]
        # 후보: top_k 문장 (점수 내림차순). 단 case가 더 짧으면 전체.
        n_pick = min(top_k, len(sents))
        ranked = sorted(range(len(sents)), key=lambda i: -case_scores[i])[:n_pick]
        # 누적 길이 cap (strict, 단 최소 1문장 보장 — 1문장이 cap 초과해도 포함)
        accepted: list[int] = []
        total = 0
        for idx in ranked:
            seg_len = len(sents[idx]) + (1 if accepted else 0)  # join 공백 1자
            if accepted and total + seg_len > PREVIEW_MAX_CHARS:
                continue
            accepted.append(idx)
            total += seg_len
        accepted.sort()  # 원문 등장 순서
        previews.append(" ".join(sents[i] for i in accepted))
    return previews


# ---------- markdown 직렬화 ----------

# provenance → 모델이 읽을 한 줄. **인용 가부를 값 자체가 말한다** — 종전에는
# `preview_provenance`(enum + 괄호 설명)와 `quote_eligible`(불리언)이 따로 나갔는데, 120건
# 전수에서 둘의 상관이 1:1 이었다(같은 1비트를 두 번 적었고 결과당 91자였다). 그리고
# 시스템 프롬프트가 `quote_eligible=true` 라는 **필드 이름을 직접 부르고** 있어 필드를
# 건드릴 때마다 프롬프트가 같이 깨질 자리였다. 이제 프롬프트는 `직접인용 불가` 라는
# **값의 표지**만 보고 필드 이름에 묶이지 않는다(2026-07-30).
#
# ⚠ 인용 가능한 것은 `original_text_excerpt` 하나뿐이다. `original_text` 는 표시 문자열이
# 머리 발췌라 원문과 정확히 같지 않아 제외한다 — 종전 불리언 계약과 같다.
_MARKUP_RE = re.compile(r"</?\s*br\s*/?\s*>", re.I)


def _strip_markup(text: str) -> str:
    """코퍼스 원문에 섞여 오는 HTML 조각을 걷는다 — 모델은 markdown-KV 만 받아야 한다."""
    return _MARKUP_RE.sub(" ", str(text or "")).strip()


_NO_QUOTE = "(직접인용 불가)"
_PREVIEW_KINDS = {
    "original_text_excerpt": "원문 발췌",
    "original_text": f"원문 머리 발췌{_NO_QUOTE}",
    "official_summary": f"공식 요약{_NO_QUOTE}",
    "official_summary_excerpt": f"공식 요약 발췌{_NO_QUOTE}",
    "generated_summary": f"AI 요약{_NO_QUOTE}",
    "generated_summary_excerpt": f"AI 요약 발췌{_NO_QUOTE}",
    "reference_statute_metadata_excerpt": f"참조조문 메타·판시 아님{_NO_QUOTE}",
}


def _set_preview(match: dict[str, Any], text: str, source: str) -> None:
    """프리뷰 문자열과 provenance를 항상 함께 설정해 출처 유실을 막는다."""
    match["preview"] = text
    match["preview_provenance"] = source


def _drop_preview_internals(match: dict[str, Any]) -> None:
    match.pop("_full_summary", None)
    match.pop("_full_summary_source", None)

def _format_response_md(resp: dict[str, Any]) -> str:
    """precedent_search 응답 dict → markdown-KV 문자열 (토큰 효율 ↑)."""
    status = resp.get("status", "ok")
    lines: list[str] = [f"## status: {status}"]
    if "message" in resp:
        lines.append(f"- message: {resp['message']}")
    if resp.get("note"):
        lines.append(f"- note: {resp['note']}")
    matches = resp.get("matches") or []
    if matches:
        lines.append("## matches")
        for m in matches:
            block = [
                f"- id: {m['id']}",
                f"  url: {case_url_base()}/cases/{m['id']}",
                f"  case_no: {m.get('case_number','')}",
                f"  case_name: {m.get('case_name','')}",
                f"  court: {m.get('court_name','')} {m.get('court_level','')} {m.get('year','')}".rstrip(),
            ]
            if m.get("reference_statute"):
                # 코퍼스 원본에 `<br/>` 이 섞여 있다 — 모델 컨텍스트에 HTML 조각을 흘리지 않는다.
                block.append(f"  statute: {_strip_markup(m['reference_statute'])}")
            if m.get("preview"):
                block.append(f"  preview: {m['preview']}")
                source = m.get("preview_provenance") or "unknown"
                block.append(f"  preview_kind: "
                             f"{_PREVIEW_KINDS.get(source, f'{source}{_NO_QUOTE}')}")
            lines.append("\n".join(block))
    elif status == "ok":
        lines.append("## matches: (없음)")
    return "\n".join(lines)


# ---------- public tool ----------

@dedup_guard("precedent_search")
def precedent_search(
    ctx: RunContext[HarnessDeps],
    query: str | None = None,
    case_number: str | None = None,
    court_level: str | None = None,
    court_name: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> str:
    """판례 검색 — query 또는 case_number 중 하나는 필수이며, 사건번호 직접 조회 또는
    사실관계 키워드 검색을 수행합니다.
    짧은 preview와 출처 종류를 반환합니다. 판결문 본문의 판단·법리가 필요하면 가장 관련된
    결과의 id로 precedent_dive를 이어 호출해 확인합니다.

    언제:
    - 유사 사실관계 판례(민사·형사·행정·가사)를 찾거나 주장·전망의 근거가 필요할 때 → query.
      "보통 이렇게 됩니다" 류의 전망을 기억으로 말하기 전에 먼저 부르세요.
    - 특정 사건번호가 주어졌을 때 → case_number.
    - 판례가 적용한 조문의 본문·현행 여부는 statute_lookup 으로 이어 확인하세요 — 판례는
      선고 당시 조문을 적용하므로 지금 본문과 다를 수 있습니다.

    응답: markdown-KV. `preview_kind: 원문 발췌`만 따옴표로 직접인용할 수 있고,
    `(직접인용 불가)`가 붙은 요약은 바꿔 쓰세요. 답에 쓴 판례는 직접 인용이든 요약이든
    반환 url을 링크로 함께 제시하세요.

    Args:
      query: 사건번호를 뺀 사실관계·죄명·법조의 변별력 있는 명사 어간 여러 개. 2자 죄명도 지원.
      case_number: 특정 사건번호(예 "2010다89012"). 긴 인용문이면 사건번호 부분만.
      court_level: 심급 '1심'|'2심'|'대법원'|'헌재'.
      court_name: 법원 이름·지역 부분매칭(예 "부산", "특허법원").
      year_from: 사건년도 범위 시작(선고년도 기준, 없으면 접수년도).
      year_to: 사건년도 범위 끝. year_from 과 함께 또는 단독 사용.
    """
    query = coerce_str(query)
    case_number = coerce_str(case_number)
    court_level = coerce_str(court_level)
    court_level, court_level_note = _normalize_court_level(court_level)
    court_name = coerce_str(court_name)
    year_from = coerce_int(year_from)
    year_to = coerce_int(year_to)

    # 교차 필수/최소길이는 14GB DB를 열기 전에 fail-loud. 사건번호 문자열도 여기서 정규화해
    # 유효한 case_number가 있으면 짧은 query가 함께 와도 정확조회 경로를 우선한다.
    cno = _extract_case_number(case_number) if case_number else None
    if not cno and not query:
        return _format_response_md({
            "status": "error",
            "message": (
                "query(사실관계 키워드) 또는 case_number(사건번호) 중 하나는 필요합니다. "
                "사건번호가 있으면 case_number 인자에 넣으세요."
            ),
            "matches": [],
        })
    if not cno and query and len(query) < 2:
        return _format_response_md({
            "status": "error",
            "message": (
                f"query 길이 {len(query)}자 — 2자 이상 필요. "
                "사실관계 키워드 또는 죄명+맥락(예: '사기 피해 1억')으로 재호출하세요."
            ),
            "matches": [],
        })

    conn = open_db()
    try:
        # Layer 0: 사건번호 직접조회 (FTS/dense 로는 못 잡음 — case_number 미인덱싱).
        # 역할 분리 — 사건번호는 명시 슬롯 case_number 로만 받는다. LLM 이 사건번호를 인식해
        # 이 인자에 넣도록 docstring 으로 유도하고, query 는 순수 사실관계 키워드 검색 전용으로
        # 둔다(query 본문에서 사건번호를 추출하지 않음 — 두 경로의 역할이 갈리지 않게). 인용
        # 형식('대법원 … 선고 2010다89012 판결')으로 들어와도 case_number 만 정규화해 조회한다.
        if cno:
            id_rows = conn.execute(
                "SELECT id FROM prec_cases WHERE REPLACE(case_number,' ','')=? ORDER BY id LIMIT ?",
                (cno, LIMIT),
            ).fetchall()
            if not id_rows:  # 병합사건('2015두38917·38924') 등은 부분일치로
                id_rows = conn.execute(
                    "SELECT id FROM prec_cases WHERE REPLACE(case_number,' ','') LIKE ? "
                    "ORDER BY length(case_number), id LIMIT ?",
                    (f"%{cno}%", LIMIT),
                ).fetchall()
            if id_rows:
                matches = _hydrate(
                    conn, [r["id"] for r in id_rows], court_level, year_from, year_to,
                    cap=LIMIT, court_name=court_name,
                )
                for m in matches:
                    _set_preview(
                        m,
                        m["_full_summary"][:PREVIEW_FALLBACK_CHARS].replace("\n", " "),
                        m["_full_summary_source"],
                    )
                    _drop_preview_internals(m)
                if matches:
                    return _format_response_md({"status": "ok", "matches": matches})
            if case_number and not query:
                # 사건번호를 명시했는데 못 찾음 + 폴백할 키워드(query)도 없음 → 안내.
                return _format_response_md({
                    "status": "ok",
                    "message": (
                        f"사건번호 '{case_number}' 에 해당하는 판례를 찾지 못했습니다. "
                        "사건번호를 확인하거나 사실관계 키워드(query)로 재검색하세요."
                    ),
                    "matches": [],
                })
            # 그 외(query 있음)는 아래 키워드 하이브리드 검색으로 폴백 (형식 차이·오타 대비)

        # ----- 키워드 하이브리드 검색 (query 필요) -----
        # cno가 없을 때 query 필수·최소길이는 DB open 전에 이미 검증했다.

        pool_size = max(LIMIT * OVERSAMPLE, 30)

        # ===== FTS-only 경로 (USE_DENSE=0, lawful 챗): trigram + 형태소 RRF 융합, NIM 미사용 =====
        if not USE_DENSE:
            tri_match = _or_match(query)       # trigram: 3자+ 부분일치·오타 내성
            morph_match = _morph_match(query)  # 형태소: 단어 단위(2자 '살인'·'사기' 포함)
            # 정렬은 관련도(BM25) 단일. 최신순(sort=recent) 은 제거했다 — OR-토큰 union 을 날짜로만
            # 정렬하면 흔한 토큰 하나(예: '송금'·'이득')만 걸린 무관한 최신 판례가 상위를 점령해
            # 관련도가 소실됐다(실측 '착오송금 상계 부당이득' → 증거인멸·조합원지분 등 무관 2026 판례).
            # 필터(법원·지역·연도)는 양쪽 FTS 모두 SQL 레벨 적용(희소 필터 보존, _fts_or_rank docstring).
            tri_ids = _fts_or_rank(
                conn, tri_match, pool_size,
                court_level=court_level, court_name=court_name,
                year_from=year_from, year_to=year_to,
            )
            morph_ids = _morph_rank(
                conn, morph_match, pool_size,
                court_level=court_level, court_name=court_name,
                year_from=year_from, year_to=year_to,
            )
            # trigram·형태소 두 랭킹을 RRF 융합(둘의 장점 합산).
            rankings = [r for r in (tri_ids, morph_ids) if r]
            ids = _rrf_fuse(rankings, k=RRF_K, limit=pool_size) if rankings else []
            matches = _hydrate(
                conn, ids, court_level, year_from, year_to,
                cap=LIMIT, court_name=court_name,
            ) if ids else []
            # 발췌는 trigram snippet 우선(매칭 구절), 형태소-only 매치는 요약 머리로 폴백.
            snips = _fts_snippets(conn, [m["id"] for m in matches], tri_match) if (matches and tri_match) else {}
            # trigram snippet 미스(2글자 형태소-only)는 본문에서 매칭 윈도우를 직접 발췌해
            # *왜 매칭됐는지*를 LLM 에 보여준다(요약 머리 대신). 미스 id 만 배치 조회.
            missed = [m["id"] for m in matches if m["id"] not in snips]
            excerpts = (
                _body_excerpts(conn, missed, _preview_terms(query), with_provenance=True)
                if missed else {}
            )
            for m in matches:
                preview = snips.get(m["id"]) or excerpts.get(m["id"])
                if preview:
                    text, source = preview
                else:
                    text = m["_full_summary"][:PREVIEW_FALLBACK_CHARS].replace("\n", " ")
                    source = m["_full_summary_source"]
                _set_preview(m, text, source)
                _drop_preview_internals(m)
            resp = {
                "status": "ok",
                "matches": matches,
                "_debug": {"n_tri": len(tri_ids), "n_morph": len(morph_ids), "mode": "tri+morph_rrf"},
            }
            # fail-loud 안내: 잘못된 court_level(필터 무시됨) + 0건일 때 사건번호 슬롯 오용.
            notes = [court_level_note]
            if not matches:
                notes.append(_case_no_in_query_hint(query))
            notes = [n for n in notes if n]
            if notes:
                resp["note"] = " / ".join(notes)
            return _format_response_md(resp)

        # ===== hybrid 경로 (USE_DENSE=1, 연구/eval 명시 opt-in): RRF + rerank =====
        embed = ctx.deps.embed
        fts_ids = _fts_rank(conn, query, pool_size)
        dense_ids = _dense_rank(conn, embed, query, pool_size)

        rankings = [r for r in (fts_ids, dense_ids) if r]
        if not rankings:
            return _format_response_md({"status": "ok", "matches": []})

        fused = _rrf_fuse(rankings, k=RRF_K, limit=pool_size)

        if USE_RERANK:
            # Stage 1: 후보 전체(pool_size)를 요약본 *전체*로 case-level rerank → top LIMIT
            candidates = _hydrate(
                conn, fused, court_level, year_from, year_to,
                cap=None, court_name=court_name,
            )
            if candidates:
                texts = [_truncate_rerank(c["_full_summary"]) for c in candidates]
                scores = _rerank_scores(ctx.deps.rerank, query, texts)
                ordered = sorted(zip(candidates, scores), key=lambda p: -p[1])
                matches = [c for c, _ in ordered[:LIMIT]]
                # Stage 2: top LIMIT의 요약본 문장 sentence-level rerank → preview top_k 문장
                summaries = [m["_full_summary"] for m in matches]
                previews = _sentence_previews_batch(
                    ctx.deps.rerank, query, summaries, PREVIEW_TOP_K
                )
                for m, p in zip(matches, previews):
                    _set_preview(m, p, m["_full_summary_source"])
            else:
                matches = []
        else:
            # rerank OFF (ablation) — 요약본 머리 PREVIEW_FALLBACK_CHARS자 fallback
            matches = _hydrate(
                conn, fused, court_level, year_from, year_to,
                cap=LIMIT, court_name=court_name,
            )
            for m in matches:
                _set_preview(
                    m,
                    m["_full_summary"][:PREVIEW_FALLBACK_CHARS].replace("\n", " "),
                    m["_full_summary_source"],
                )

        for m in matches:
            _drop_preview_internals(m)

        return _format_response_md({
            "status": "ok",
            "matches": matches,
            "_debug": {
                "n_fts": len(fts_ids),
                "n_dense": len(dense_ids),
                "n_fused": len(fused),
                "rerank": USE_RERANK,
            },
        })
    finally:
        conn.close()
