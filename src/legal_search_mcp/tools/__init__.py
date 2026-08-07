"""The five corpus tools, all with a ``RunContext[HarnessDeps]`` signature.

  compute_sentencing_range  deterministic sentencing arithmetic: statutory
                            range, processing under the Criminal Act, the
                            guideline range, and verification of a sentence
  statute_lookup            statute and administrative-rule article text
  precedent_search          lexical retrieval (trigram + morpheme FTS fused
                            by RRF), optionally embeddings + rerank
  precedent_dive            delegate one judgment body to a sub-agent
  sentence_statistics       sentencing distribution for a single charge

Each is a plain function: import them directly, or take ``agents.TOOLS`` to
register the set with a pydantic-ai agent. Their docstrings are the tool
descriptions a model reads, which is why they are written in Korean.
"""
from .compute_sentencing_range import compute_sentencing_range
from .precedent_dive import precedent_dive
from .precedent_search import precedent_search
from .sentence_statistics import sentence_statistics
from .statutes import statute_lookup

__all__ = [
    "compute_sentencing_range",
    "statute_lookup",
    "precedent_search",
    "precedent_dive",
    "sentence_statistics",
]
