"""Evaluation across four axes: quality, latency, memory and size.

A model is not "better" because its loss is lower. It is better because it is
good enough, fast enough and small enough for where it has to run. This package
measures all four so that :mod:`forgeml.optimization` can trade between them.

``quality`` is pure Python and unit-tested in CI. ``latency``, ``memory`` and
``perplexity`` need torch and a real model.
"""

from __future__ import annotations

from forgeml.evaluation.quality import (
    QualityScores,
    exact_match,
    rouge_l,
    score_predictions,
    token_f1,
)

__all__ = [
    "QualityScores",
    "exact_match",
    "rouge_l",
    "score_predictions",
    "token_f1",
]
