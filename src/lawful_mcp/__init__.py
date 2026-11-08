"""Lawful MCP — Korean case law, statutes and sentencing data as MCP tools.

Four read-only tools over a SQLite corpus of Korean court judgments,
statutes, administrative rules, tax treaties and sentencing-guideline data:

  precedent_search     find judgments by facts, charge or case number
  precedent_dive       read one judgment body and answer a question
  statute_lookup       statute, administrative-rule and tax-treaty articles,
                       current or as of a date
  sentencing_analysis  statutory penalties, guidelines, factors and
                       first-instance sentences for a case's charges

``harness_local`` holds the tools: the research harness of the Lawful Agent
API, copied unmodified at the commit in ``HARNESS_UPSTREAM`` — the same code
the hosted service runs. Comments in it that name ``eval/`` or
``harness/README.md`` refer to that research repository. ``server``
assembles the MCP application around it, and ``config`` maps this package's
settings onto the variables the copy reads.

Importing the package points the copy at the corpus, so
``harness_local.agents.TOOLS`` runs against the bundled sample with no setup.
"""
from . import config as _config

_config.export_corpus_path()

__version__ = "0.2.0"
