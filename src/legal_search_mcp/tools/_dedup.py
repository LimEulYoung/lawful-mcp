"""Guard against a model calling the same tool with the same arguments in a loop.

A model that gets an unhelpful result sometimes retries it verbatim rather
than changing approach, and with deterministic decoding the retry returns the
identical result — so it retries again. Observed running to 35 consecutive
identical calls.

Rather than failing, a blocked call answers with ``status: duplicate_call``
and says the result is already in the conversation, which gives the model
something to act on.
"""
from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from functools import wraps
from typing import Any

from pydantic_ai import RunContext

from ..deps import HarnessDeps

DEDUP_WINDOW = 5   # how many recent calls to look back over
MAX_REPEATS = 2    # identical calls tolerated within that window


def _check_and_record(deps: HarnessDeps, tool_name: str, kwargs: dict[str, Any],
                      key_salt: Callable[[Any], str] | None = None) -> str | None:
    # Deps without a call log — a bare namespace from a test or an embedding
    # host — disable the guard rather than fail. Blocking is the optional
    # part here; running the tool is the point.
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
    """Decorate a tool, sync or async.

    Expects ``fn(ctx: RunContext[HarnessDeps], **kwargs)``. A blocked call
    returns the same markdown-KV shape the tools answer in, so the model
    reads it as an ordinary result.

    ``key_salt(deps) -> str`` mixes state into the duplicate key. Use it for
    a tool whose answer can legitimately change between identical calls: when
    the salt changes the call counts as new, so only genuinely unchanged
    repeats are treated as a loop.
    """
    def deco(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):
            @wraps(fn)
            async def awrapper(ctx: RunContext[HarnessDeps], *args, **kwargs):
                # Positional arguments mean a direct or test call; a model
                # always calls by keyword. Let those through unguarded —
                # they are not part of the key, so mixing them in would make
                # the same call hash differently.
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
