"""Optional per-answer records for tool-internal requests. Never records credentials."""

from contextvars import ContextVar

records = ContextVar("tool_request_records", default=None)


def emit(record):
    target = records.get()
    if target is not None:
        target.append(record)
    return record
