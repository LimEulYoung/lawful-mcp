"""Every test runs on the bundled sample (data/fixture.db) with the network blocked and no keys.

``HARNESS_TEST_DB`` runs the same tests on another copy (a sample before it is committed).
"""

import os
import socket
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixture.db"

os.environ.pop("CORPUS_DB", None)
os.environ["HARNESS_DB"] = os.environ.get("HARNESS_TEST_DB") or str(FIXTURE)
# Keys in the developer's shell must not change what the tests see.
for _name in ("OPENROUTER_API_KEY", "JEV_API_KEY", "DIVE_API_KEY", "DIVE_BASE_URL", "DIVE_MODEL", "DIVE_TIMEOUT"):
    os.environ.pop(_name, None)


def _blocked(*args, **kwargs):
    raise OSError("network access in tests")


socket.socket.connect = _blocked
socket.create_connection = _blocked
