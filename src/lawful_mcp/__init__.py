"""Lawful MCP — Korean case law, statutes and sentencing data as MCP tools.

Four read-only tools over a SQLite corpus of Korean court judgments,
statutes, administrative rules and sentencing-guideline data:

  precedent_search     find judgments by facts, charge or case number
  precedent_dive       read one judgment body and answer a question
  statute_lookup       statute and administrative-rule articles, as of a date
  sentencing_analysis  the guideline, the factors and the observed sentences
                       for a charge, and the arithmetic for a case

``server`` assembles the MCP application; ``tools`` holds the tool bodies.
The corpus is not built by this package — see the README for the sample
fixture and for building your own.
"""

__version__ = "0.1.0"
