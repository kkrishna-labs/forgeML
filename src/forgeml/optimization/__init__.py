"""Turning a table of experiments into one deployable decision."""

from __future__ import annotations

from forgeml.optimization.selector import (
    ModelCandidate,
    SelectionResult,
    normalize_metric,
    pareto_frontier,
    select_champion,
)

__all__ = [
    "ModelCandidate",
    "SelectionResult",
    "normalize_metric",
    "pareto_frontier",
    "select_champion",
]
