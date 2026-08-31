"""MCP application — five read-only corpus tools over stdio or HTTP.

No authentication and no metering: this server is meant to run next to the
client that uses it, against a corpus you hold. (The hosted deployment puts
those concerns in a separate layer, which is not part of this package.)

Tool descriptions are the contract an external model reads. They are written
in Korean and carry the operating rules, because a client has no system
prompt of ours to fall back on — each description has to stand alone.
"""
from __future__ import annotations

import argparse
import logging
import os
import threading
from types import SimpleNamespace

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from . import tools as _t
from .config import corpus_db_path, dive_config
from .deps import build_deps, build_dive_subagent

logger = logging.getLogger("lawful-mcp")

_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
_PORT = os.environ.get("MCP_PORT", "8100")
# Extra Host headers to accept, comma separated. The SDK's DNS-rebinding
# guard checks the Host header, so a reverse proxy's public name must be
# listed here or requests through it are rejected.
_EXTRA_HOSTS = [
    h.strip() for h in os.environ.get("MCP_EXTRA_HOSTS", "").split(",") if h.strip()
]

_INSTRUCTIONS = (
    "한국 법률 검색 도구 모음 — 판례·법령·행정규칙(고시·훈령·예규)·양형기준 corpus 기반. "
    "판례 검색(precedent_search) 후 필요할 때 본문 생성요약(precedent_dive), "
    "법령·행정규칙 조문(statute_lookup), 양형 분포(sentence_statistics), "
    "양형기준 계산(compute_sentencing_range). 판례 출처 표기는 도구가 반환한 url만 사용."
)

mcp = FastMCP(
    "lawful-mcp",
    instructions=_INSTRUCTIONS,
    # Stateless request/response JSON: no session state, no SSE, so a plain
    # reverse proxy in front needs no special handling.
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(
        allowed_hosts=[
            "127.0.0.1",
            f"127.0.0.1:{_PORT}",
            "localhost",
            f"localhost:{_PORT}",
            *_EXTRA_HOSTS,
        ],
        allowed_origins=[f"https://{h}" for h in _EXTRA_HOSTS],
    ),
)

# One sub-agent for the process: building a model client per call would pay
# connection setup on every dive. None when unconfigured, which is why the
# dive tool is registered conditionally.
_dive_subagent = build_dive_subagent()


def _ctx() -> SimpleNamespace:
    """Shuttle context for a tool call.

    The tools take a pydantic-ai ``RunContext`` but only ever read ``.deps``
    and ``.usage``, so a namespace with those two fields is enough. Fresh per
    call, so the dedup guard does not carry state between calls.
    """
    return SimpleNamespace(deps=build_deps(_dive_subagent), usage=None)


def _tool(desc: str, *, title: str, **hints):
    """Register a tool with its description and behaviour annotations.

    Every tool here is read-only, so ``destructiveHint`` is set explicitly:
    the MCP default is true, and a client that sees no annotation has to
    assume the worst.
    """
    return mcp.tool(description=desc, annotations=ToolAnnotations(title=title, **hints))


_DESC_PRECEDENT_SEARCH = (
    "판례 검색 — query 또는 case_number 중 하나는 필수. 사건번호로 정확 조회하거나 사실관계 키워드로 유사 판례 검색. "
    "매치는 짧은 preview와 그 출처를 알리는 preview_kind를 반환합니다. "
    "본문 확인이 필요하면 가장 관련된 id로 precedent_dive를 이어 호출하세요. "
    "민·형사·행정·가사 분쟁 질의의 기본 도구이며, 주장·전망의 근거를 유사 판례의 실제 결과로 뒷받침할 때 씁니다.\n"
    "Args: query=사실관계·죄명·법조 키워드(명사 어간 여러 개; 2자 죄명도 지원; 사건번호 제외). "
    "case_number=특정 사건번호로 바로 찾을 때(예 '2010다89012'; '대법원 … 선고 2010다89012 판결'이면 사건번호 부분만). "
    "court_level='1심'|'2심'|'대법원'|'헌재'. court_name=법원명·지역 부분매칭(예 '부산','특허법원'). year_from/year_to=사건년도 범위.\n"
    "url만 인용 링크로 씁니다. preview_kind 이름에 '원문'이 있는 것만 직접인용하고 요약 계열은 바꿔 쓰세요."
)
_DESC_STATUTE_LOOKUP = (
    "법령·행정규칙 조회 — 법령의 요건·효과·기간·절차가 답의 뼈대가 되는 국면의 기본 도구이며, "
    "죄명·법조 식별 후 조문 본문 확인에도 씁니다. 조문은 개정되므로 현행 본문은 이 도구만 압니다. "
    "법률·대통령령·부령·규칙과 행정규칙(고시·훈령·예규)을 한 번에 검색합니다 — 종류를 가리는 인자는 없고 "
    "관련도 순으로 함께 나오며 각 결과에 종류가 붙습니다. "
    "자주 쓰는 법령은 quick-access id로 바로 호출(statute_id+articles): 헌법 468·민법 584·상법 583·민사소송법 581·"
    "형법 578·형사소송법 574·행정기본법 4953·행정절차법 437·행정소송법 386·헌법재판소법 3629. "
    "query 또는 statute_id가 필수. 그 외는 two-step — ① query=법령명으로 후보 id를 받고 ② statute_id+articles로 본문 호출"
    "(조문 번호는 법령마다 달라 법령을 먼저 확정).\n"
    "Args: query=법령명·행정규칙명 또는 본문 키워드(id 모를 때). "
    "statute_id=검색이 준 식별자를 글자 그대로 — 법령은 정수(예 584; 위 목록 밖은 추측 금지 — 574는 형사소송법), "
    "행정규칙은 'admrul-18060'. **접두사를 떼면 같은 번호의 다른 법령이 조회됩니다.** "
    "articles=조문 번호 list[str] 최대 8개('347'/'제347조'=본조+가지, '347의2'=가지만, 범위 ['3','4','5']; 미지정 시 outline; "
    "8개 초과는 앞 8개만 조회하고 나머지를 message로 알림 — 나눠 재호출). "
    "limit=검색 모드 최대 결과 수(기본 10·최대 50). "
    "offense_date=행위 일자(예 '2013.7.30') 지정 시 행위시점 조문, 미지정 시 현행.\n"
    "url만 인용 링크로 쓰고, `text_kind: 공식 … 원문`인 조문·행정규칙 본문만 직접인용하세요."
)
_DESC_SENTENCE_STATISTICS = (
    "양형 선고 통계 — charges 또는 charge_id 중 하나는 필수. ① charges 하나로 정제된 죄명 후보(charge_id+표본수)를 반환(status=candidates), "
    "② 후보에서 고른 charge_id 하나를 주면 그 죄명 하나만 유죄인 피고인의 1심 선고 분포(표본 30↑=형종별 평균·표준편차·집유율 + 11분위 비교판례 / 미만=개별 사례 그리드). "
    "죄명이 이미 특정되면 곧바로 통계가 옵니다. 형량 전망·구형/양형 근거·자기 사건 위치를 가늠할 때 결론 전에 호출"
    "(compute_sentencing_range 공식 '범위'를 실데이터로 보완). status=candidates는 통계가 아니라 선택지 — 사안에 맞는 charge_id 하나로 재호출. "
    "죄명·charge_id는 한 번에 하나씩. 경합 사안의 죄명별 단독 분포를 합산·평균·1.5배해 경합범 분포로 만들지 마세요. 형법 38조는 가장 중한 죄 장기(벌금은 다액)의 1/2까지 가중하되 각 죄 장기·다액 합계를 넘지 못하게 하는 처단형 상한이지 통계 결합식이 아닙니다. status=low_n_grid는 단독 개별 사례라 일반화 금지.\n"
    "Args: charges=죄명 텍스트 하나(후보 검색용; 구어 '보이스피싱·몰카·마약'·카테고리 '성범죄'·법률명 '도로교통법위반'도 인식). charge_id=후보에서 고른 pool id 하나(int; 통계 조회용, charges와 택일). "
    "year_from/year_to=판결 연도 범위. reference_year=비교 판례·그리드 기준 연도(가까운 사건 우선; None=최근).\n"
    "비교 판례·그리드의 url만 인용 링크로, 집계 분포 수치는 링크 없이 제시."
)
_DESC_COMPUTE_SENTENCING_RANGE = (
    "통합 양형 도구 — 죄명에서 법정형→처단형→권고형→선고 검증까지, 인자를 채울수록 깊은 단계로 자동 진행: "
    "charge만=lookup(법정형·leaf 후보·인자 enum) / +statutory_modifications=처단형(형법§56 순서 적용) / "
    "+guideline_leaf_id·guideline_factors=권고형 / +sentence_months·fine_amount(+probation_factors)=final(선고형·집행유예 검증). "
    "결과는 양형기준이 정한 '범위'(예측 아님). 호출 간 상태가 없으므로 후속 호출마다 charge와 확정한 선택·플래그·offense_date를 반복하고 새 인자를 추가.\n"
    "Args: charge=판결문형 죄명 문자열(예 '살인','도로교통법위반(음주운전)') — 호출당 하나"
    "(list는 죄명별 매칭 유도[multiple_charges]), 숫자·ID 불가"
    "(조문 번호·sentence_statistics의 charge_id·leaf id 아님; 숫자면 죄명 문자열로 유도[charge_numeric]). "
    "offense_date=행위 일자(지정 시 행위시 조문). "
    "sg_category_id·statute_choice·branch_key·reference_choice=ambiguous_* 응답이 후보를 줄 때. "
    "is_attempted·is_accessory·is_solicitor=미수·방조·교사. statutory_modifications=가중감경 list(lookup enum에서). "
    "guideline_leaf_id·guideline_factors=양형기준 leaf·특별인자. "
    "sentence_months=검증 선고형(자유형·월)·fine_amount=벌금(원)·probation_factors=집행유예 인자(dict). act_count=동종 다행위 수(≥2면 경합범 가중 자동).\n"
    "후속 단계 값은 이전 응답 enum에 있는 key만 쓰고 추측 금지. 응답의 '출처'(양형기준 해설서 PDF)가 있으면 인용 링크로 제시."
)
_DESC_PRECEDENT_DIVE = (
    "단건 판결·결정 본문 추출(외부 sub-agent 위임) — precedent_search preview가 부족할 때 case id로 호출하면 question에 답하는 300자 내외 생성 요약을 반환. "
    "summary는 직접인용이 아니며 text_truncated=true이면 not_in_text도 전체 원문 부재를 확정하지 못합니다.\n"
    "Args: case_id=precedent_search 결과 id. question=공개 판례에서 추출할 쟁점·항목만(사용자 이름·주소·연락처·계정·사적 첨부사실 등 개인정보 금지).\n"
    "url만 인용 링크로 쓰고 생성 요약인 summary 는 원문 직접인용으로 쓰지 마세요."
)


@_tool(
    _DESC_PRECEDENT_SEARCH,
    title="판례 검색",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
def precedent_search(
    query: str | None = None,
    case_number: str | None = None,
    court_level: str | None = None,
    court_name: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> str:
    return _t.precedent_search(
        _ctx(),
        query=query,
        case_number=case_number,
        court_level=court_level,
        court_name=court_name,
        year_from=year_from,
        year_to=year_to,
    )


@_tool(
    _DESC_STATUTE_LOOKUP,
    title="법령·행정규칙 조회",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
def statute_lookup(
    query: str | None = None,
    statute_id: int | str | None = None,
    articles: list[str] | None = None,
    limit: int = 10,
    offense_date: str | None = None,
) -> str:
    return _t.statute_lookup(
        _ctx(),
        query=query,
        statute_id=statute_id,
        articles=articles,
        limit=limit,
        offense_date=offense_date,
    )


@_tool(
    _DESC_SENTENCE_STATISTICS,
    title="양형 선고 통계",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
def sentence_statistics(
    charges: str | None = None,
    charge_id: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    reference_year: int | None = None,
) -> str:
    return _t.sentence_statistics(
        _ctx(),
        charges=charges,
        charge_id=charge_id,
        year_from=year_from,
        year_to=year_to,
        reference_year=reference_year,
    )


@_tool(
    _DESC_COMPUTE_SENTENCING_RANGE,
    title="양형기준 범위 계산",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
def compute_sentencing_range(
    # Only this argument also accepts `list`. The MCP layer runs json.loads over
    # any JSON-looking string argument *before* the tool does (SDK
    # `func_metadata.pre_parse_json`, applied wherever the annotation is not
    # exactly `str`), so `charge='[299,298,297]'` arrives as a list and a narrow
    # `str | None` rejects the value that layer just produced. That is precisely
    # the shape the charge_numeric hint exists for — article numbers carried over
    # from a previous tool result — so the hint never reached the caller here,
    # while `'298'` and `'297의2'` (scalar, non-JSON) always worked. Taking it
    # wide lets the tool's own coerce_str fold it back to "299, 298, 297" and the
    # hint fires as intended.
    # Sibling string arguments stay narrow on purpose: widening only pays off
    # where a prepared answer exists for the wide value. A list reaching `query`
    # or `charges` would instead be joined into one comma-separated search term
    # and searched silently, which is worse than a validation error.
    charge: str | list | None = None,
    sg_category_id: int | None = None,
    statute_choice: str | None = None,
    branch_key: str | None = None,
    reference_choice: str | None = None,
    is_attempted: bool = False,
    is_accessory: bool = False,
    is_solicitor: bool = False,
    statutory_modifications: list[dict] | None = None,
    guideline_leaf_id: int | None = None,
    guideline_factors: dict | None = None,
    sentence_months: int | None = None,
    fine_amount: int | None = None,
    probation_factors: dict | None = None,
    act_count: int = 1,
    offense_date: str | None = None,
) -> str:
    return _t.compute_sentencing_range(
        _ctx(),
        charge=charge,
        sg_category_id=sg_category_id,
        statute_choice=statute_choice,
        branch_key=branch_key,
        reference_choice=reference_choice,
        is_attempted=is_attempted,
        is_accessory=is_accessory,
        is_solicitor=is_solicitor,
        statutory_modifications=statutory_modifications,
        guideline_leaf_id=guideline_leaf_id,
        guideline_factors=guideline_factors,
        sentence_months=sentence_months,
        fine_amount=fine_amount,
        probation_factors=probation_factors,
        act_count=act_count,
        offense_date=offense_date,
    )


if _dive_subagent is not None:

    @_tool(
        _DESC_PRECEDENT_DIVE,
        title="판례 본문 분석",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
    async def precedent_dive(case_id: int, question: str) -> str:
        return await _t.precedent_dive(_ctx(), case_id=case_id, question=question)


def build_app():
    """Starlette app serving the MCP endpoint at ``/mcp``."""
    import contextlib

    from starlette.applications import Starlette

    http_app = mcp.streamable_http_app()

    @contextlib.asynccontextmanager
    async def _lifespan(_app):
        async with mcp.session_manager.run():
            yield

    return Starlette(routes=[*http_app.routes], lifespan=_lifespan)


def _warmup() -> None:
    """Load the morphological analyser and warm the FTS page cache.

    First call after start would otherwise pay both. Failure is not fatal:
    the work happens at call time anyway.
    """
    try:
        _t.precedent_search(_ctx(), query="손해배상 계약 해지")
        logger.info("warmup done")
    except Exception as e:  # noqa: BLE001
        logger.warning("warmup skipped: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lawful MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="stdio for a local client (default), http for a network endpoint",
    )
    parser.add_argument("--host", default=_HOST)
    parser.add_argument("--port", type=int, default=int(_PORT))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("corpus: %s", corpus_db_path())
    if dive_config() is None:
        logger.info("precedent_dive disabled (set DIVE_API_KEY, DIVE_BASE_URL, DIVE_MODEL)")

    if args.transport == "stdio":
        mcp.run()
        return

    threading.Thread(target=_warmup, daemon=True).start()
    import uvicorn

    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
