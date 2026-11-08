"""Mini legal harness — the tools the lawful condition gives the model.

Layout:
  deps.py        — HarnessDeps (dive_subagent) + open_db factory
  agents.py      — the tool list + builder for the public-judgment dive sub-agent
  tools/         — the 4 tools and their helpers (RunContext signature; each opens its own db via open_db)
  prompts.py     — DIVE_PROMPT (for precedent_dive only)
  agent_api.py   — the date line the Agent API appends to the system prompt
"""
