"""dedup_guard — 같은 (tool, args) 호출이 직전 N턴에 MAX_REPEATS 초과 시 차단.

LLM이 temperature=0 + vLLM prefix cache 환경에서 동일 응답을 받고 같은 도구를
같은 인자로 무한 호출하는 패턴 차단 (실측: 시체유기 case에서 statute_lookup 35회).

차단되면 status="duplicate_call"로 응답해 LLM에게 다른 접근을 유도한다.
"""
from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from functools import wraps
from typing import Any

from pydantic_ai import RunContext

from ..deps import HarnessDeps

DEDUP_WINDOW = 5   # 직전 N개 호출만 검사
MAX_REPEATS = 2    # 같은 (tool, args)가 window 안에 2회 초과면 차단


def _check_and_record(deps: HarnessDeps, tool_name: str, kwargs: dict[str, Any],
                      key_salt: Callable[[Any], str] | None = None) -> str | None:
    # 장부(recent_calls)가 없는 deps — REST 셔틀 ctx·단위 테스트의 SimpleNamespace 등 —
    # 에서는 가드를 끈다(fail-open). 차단은 선택 기능이고 도구 실행이 본체다.
    recent_calls = getattr(deps, "recent_calls", None)
    if recent_calls is None:
        return None
    salt = key_salt(deps) if key_salt is not None else ""
    key = (tool_name, salt + json.dumps(kwargs, sort_keys=True, ensure_ascii=False, default=str))
    recent = list(recent_calls)[-DEDUP_WINDOW:]
    if recent.count(key) >= MAX_REPEATS:
        args_str = json.dumps(kwargs, ensure_ascii=False)
        return (
            "## status: duplicate_call\n"
            f"- tool: {tool_name}\n"
            f"- args: {args_str}\n"
            f"- message: 같은 인자로 {tool_name}를 직전 {DEDUP_WINDOW}턴 안에 "
            f"{MAX_REPEATS + 1}회 이상 호출했습니다. 그 결과는 이미 이 대화에 있습니다 — "
            "받은 결과를 사용해 다음 단계로 진행하거나, 다른 인자·다른 도구를 쓰세요."
        )
    recent_calls.append(key)
    return None


def dedup_guard(tool_name: str, *, key_salt: Callable[[Any], str] | None = None) -> Callable:
    """sync/async 함수 모두 지원하는 데코레이터.

    함수 시그니처: ``fn(ctx: RunContext[HarnessDeps], **kwargs)``.
    차단 시 markdown-KV 문자열 반환 (도구 응답 포맷과 일치).

    ``key_salt(deps) -> str`` 는 중복 키에 섞는 상태 소금이다 — 같은 인자라도 그사이
    세계가 바뀌어 재호출이 정당한 도구(예: 문서 read 는 create/edit 뒤에 결과가 달라진다)에
    쓴다. 소금이 바뀌면 새 호출로 취급하므로 **무변화 반복만** 루프로 잡는다.
    """
    def deco(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):
            @wraps(fn)
            async def awrapper(ctx: RunContext[HarnessDeps], *args, **kwargs):
                # 위치 인자 = 직접/테스트 호출 경로다 — 모델(pydantic-ai)은 항상 키워드로
                # 부른다. 이 경로는 가드 없이 그대로 통과시킨다(키에 담기지도 않으므로
                # 섞이면 같은 호출이 다른 키가 된다).
                blocked = (None if args else
                           _check_and_record(ctx.deps, tool_name, kwargs, key_salt))
                if blocked is not None:
                    return blocked
                return await fn(ctx, *args, **kwargs)
            return awrapper

        @wraps(fn)
        def wrapper(ctx: RunContext[HarnessDeps], *args, **kwargs):
            blocked = (None if args else
                       _check_and_record(ctx.deps, tool_name, kwargs, key_salt))
            if blocked is not None:
                return blocked
            return fn(ctx, *args, **kwargs)
        return wrapper
    return deco
