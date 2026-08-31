"""Sentencing-guideline range calculation.

Despite the package name, ``recommended_range`` is not evaluation-only code.
``tools/compute_sentencing_range`` imports ``determine_range``,
``RecommendedRange``, ``AppliedFactor``, ``in_range`` and
``within_range_position`` from it to produce the recommended range — it is a
live dependency of a shipped tool, not dead code.
"""
