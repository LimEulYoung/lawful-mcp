"""HarnessDeps — dependencies injected into the pydantic-ai Agent.

  - dive_subagent: pydantic-ai Agent — precedent_dive delegates extraction from the selected public judgment body to it.

The sqlite connection is not taken through deps — each tool opens its own connection with `open_db()` at call
time (avoids SQLite cursor conflicts when one conn is shared across async code and worker threads).
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai import Agent


@dataclass
class HarnessDeps:
    dive_subagent: "Agent | None"


def open_db() -> sqlite3.Connection:
    """Open the database named by HARNESS_DB read-only with the Row factory.

    `check_same_thread=False` lets async eval (asyncio.gather) share the same connection.
    """
    if not os.environ.get("HARNESS_DB"):
        raise RuntimeError("HARNESS_DB is not set: point it to the harness's SQLite database")
    p = Path(os.environ["HARNESS_DB"])
    if not p.exists():
        raise FileNotFoundError(f"harness.db not found: {p}")
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
