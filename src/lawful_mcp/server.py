"""MCP application — four read-only corpus tools over stdio or HTTP.

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
import sys
import threading
from types import SimpleNamespace

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from . import tools as _t
from .config import corpus_db_path, dive_config
from .deps import build_deps, build_dive_subagent
from .tools.compute_sentencing_range import ChargeArg

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
    "법령·행정규칙 조문(statute_lookup), 양형(sentencing_analysis — 기준·인자·실선고 분포를 "
    "한 응답에, 인자를 더하면 처단형·권고형·선고형 계산). 판례 출처 표기는 도구가 반환한 url만 사용."
)

mcp = MCPServer("lawful-mcp", instructions=_INSTRUCTIONS)


def _reclaim_root_logging() -> None:
    """Reclaim root logger hijacked by MCP SDK's MCPServer.

    MCPServer(...) calls basicConfig(level=INFO, handlers=[RichHandler])
    internally. RichHandler wraps at 80 columns in non-TTY pipes and lets
    libraries like httpx log URL queries at INFO. We replace it with a
    standard stream handler.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.__class__.__module__.startswith("rich"):
            root.removeHandler(handler)
    if not root.handlers:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(stream)
    root.setLevel(logging.INFO)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


_reclaim_root_logging()


def _streamable_app():
    """Build the streamable HTTP ASGI app for the MCP endpoint.

    mcp 2.x passes transport security and JSON response flags to
    streamable_http_app() rather than the MCPServer constructor.
    """
    return mcp.streamable_http_app(
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
    call.
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
_DESC_SENTENCING_ANALYSIS = (
    "양형 — 죄명 하나로 판단에 필요한 것을 한 응답에: 법정형, 형법 §56 처단형 계산방법, 대법원 양형기준의 "
    "범죄유형별 권고 형량범위, 양형인자, 집행유예 참작사유, 근거 조문 원문, 1심 실선고 분포와 비교판례."
    " 형량·구형·양형 의견을 볼 때 결론 전에 죄명만으로 부르고, "
    "유형·양형인자가 정해지면 계산 인자를 더해 다시 부르세요 — 두 번째 응답은 그 사건의 처단형·권고형·"
    "선고 가능 범위·집행유예만 싣고 기준 본문을 반복하지 않습니다. 조항·분기가 여럿이면 되묻지 않고 후보 정량을 전부 싣습니다. 권고범위는 공식 기준의 '범위'이지 예측이 아니고 실선고 분포는 표본의 관측입니다. "
    "죄명은 판결문 표기로 한 번에 하나씩(형법 '특수상해', 특별법 '도로교통법위반(음주운전)'). 법률명만 주면"
    "('스토킹처벌법위반') 그 법률의 죄명 후보를 돌려줍니다. 경합 사안은 죄명별로 각각 호출하고 죄명별 분포를 "
    "합산·평균해 경합범 분포로 만들지 마세요.\n"
    "Args: charge=죄명 하나(숫자·ID 불가). offense_date=행위 일자(예 '2013.7.30') 지정 시 행위시 조문·정량. "
    "statute_choice·branch_key·reference_choice·sg_category_id=후보가 여럿일 때 좁히는 값. "
    "is_attempted·is_accessory·is_solicitor=미수·방조·교사. "
    "statutory_modifications=형법 §56 가중·감경 list(예 [{\"kind\": \"누범_가중\"}]). "
    "guideline_type=응답 목록의 따옴표 안 유형 명칭 그대로. "
    # The eight key names are spelled out **here**. A pydantic-ai agent reads them
    # from the tool docstring, but for an MCP client this description is all there
    # is, and a briefing response is `protocol=False` so it carries no schema line
    # either. Leave them out and a client invents names, paying one more
    # `bad_factor_key` round trip — breaking this tool's promise not to ask back,
    # on the MCP surface alone.
    "guideline_factors=특별양형인자 dict — key 는 special_act_aggravators·"
    "special_act_mitigators·special_actor_aggravators·special_actor_mitigators 넷뿐. "
    "probation_factors=집행유예 참작사유 dict — key 는 major_positive·major_negative·"
    "general_positive·general_negative 넷뿐. 값은 응답의 해당 그룹 아래 문장 그대로 넣고, "
    "이 여덟 밖의 key 는 계산 없이 되돌아옵니다. "
    "sentence_months·fine_amount=검증할 선고형. act_count=같은 죄명의 별개 행위 수"
    "(2 이상이면 경합범 가중). reference_year=비교 판례 기준 연도.\n"
    "판례·조문 url만 인용 링크로 쓰세요."
)
_DESC_PRECEDENT_DIVE = (
    "단건 판결·결정 본문 추출(외부 sub-agent 위임) — precedent_search preview가 부족할 때 case id로 호출하면 question에 답하는 500자 내외 생성 요약을 반환. "
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
    _DESC_SENTENCING_ANALYSIS,
    title="양형 분석",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
def sentencing_analysis(
    # charge shares the same ChargeArg as the core tool (wire schema `string` and required;
    # lists are folded by BeforeValidator before schema validation). Sibling string arguments
    # remain narrow: widening is only beneficial when an answer is prepared for the wide value.
    charge: ChargeArg,
    statute_choice: str | None = None,
    branch_key: str | None = None,
    reference_choice: str | None = None,
    sg_category_id: int | None = None,
    offense_date: str | None = None,
    is_attempted: bool = False,
    is_accessory: bool = False,
    is_solicitor: bool = False,
    statutory_modifications: list[dict] | None = None,
    guideline_type: str | None = None,
    guideline_factors: dict | None = None,
    probation_factors: dict | None = None,
    sentence_months: int | None = None,
    fine_amount: int | None = None,
    act_count: int = 1,
    reference_year: int | None = None,
) -> str:
    return _t.sentencing_analysis(
        _ctx(),
        charge=charge,
        statute_choice=statute_choice,
        branch_key=branch_key,
        reference_choice=reference_choice,
        sg_category_id=sg_category_id,
        offense_date=offense_date,
        is_attempted=is_attempted,
        is_accessory=is_accessory,
        is_solicitor=is_solicitor,
        statutory_modifications=statutory_modifications,
        guideline_type=guideline_type,
        guideline_factors=guideline_factors,
        probation_factors=probation_factors,
        sentence_months=sentence_months,
        fine_amount=fine_amount,
        act_count=act_count,
        reference_year=reference_year,
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

    http_app = _streamable_app()

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
