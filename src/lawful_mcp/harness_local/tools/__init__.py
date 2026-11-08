"""Tools — RunContext[HarnessDeps] signature.

The 4 registered tools (`agents.TOOLS`):
  sentencing_analysis     (the one sentencing surface — rules, guideline, factors, articles, actual sentences; a list of charges)
  statute_lookup          (deterministic SQL; statute and administrative-rule text)
  precedent_search        (trigram + morpheme FTS; optional Jev issue reordering)
  precedent_dive          (the sub-agent extracts and summarizes the body of a selected public judgment)

What `sentencing_analysis` reads, one module per source — not registered, data in and sections out:
  statutory_penalty    (charge_legal_map, clm_versions: each provision of a charge with its statutory penalty)
  sentencing_guideline (sg_*: a guideline category's types, recommended ranges and factors)
  sentence_statistics  (prec_*: first-instance sentences of the judgments holding the given charges)
"""
