# Harness tests

The research harness's own tests, copied with `src/lawful_mcp/harness_local` at the commit in
`src/lawful_mcp/HARNESS_UPSTREAM`. They run on the bundled sample (`../conftest.py`).

Rewrites applied when copying:

- `harness_local` becomes `lawful_mcp.harness_local` in every file.
- The research sample's path (`data/sample.db`) becomes `data/fixture.db`.
- Assertions that depend on what the research sample lacks are adapted and marked `lawful-mcp:`.
  This sample ships addenda (`test_statute_fixes.py`). It also carries real tax treaties, which
  `test_treaties.py` deletes from its working copy before adding its own rows.

The research `conftest.py` is not copied; `tests/conftest.py` does its job.
