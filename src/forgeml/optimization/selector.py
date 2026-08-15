"""Automated champion selection.

This is the module that makes ForgeML a platform rather than a training script.
Picking ``max(quality)`` is not a decision, it is the absence of one — it ignores
that the winning model might be four times slower and need double the VRAM for
half a point of ROUGE.

The policy implemented here has three stages, and the ordering matters:

1. **Constraints** — hard gates (minimum quality versus baseline, latency and
   memory ceilings). A candidate that fails any gate is out, whatever its score.
   Gates come first because they express requirements; scores express preferences,
   and no amount of preference should override a requirement.
2. **Normalization** — every metric is mapped to ``[0, 1]`` across the surviving
   candidates, with lower-is-better metrics inverted. Without this you would be
   adding milliseconds to megabytes.
3. **Weighted utility** — a single scalar, so the ranking is total and explainable.

The Pareto frontier is computed alongside but deliberately not used to pick the
winner: it tells you which candidates are *defensible*, not which one to ship. The
frontier plus the weights together are the honest answer — "here are the six
non-dominated options, and here is the one our stated priorities select".
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forgeml.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from forgeml.config import SelectionConfig

log = get_logger(__name__)

# metric name -> True when a LARGER value is better
_HIGHER_IS_BETTER = {
    "quality": True,
    "latency": False,
    "memory": False,
    "model_size": False,
    "cost": False,
}


@dataclass
class ModelCandidate:
    """One evaluated model, flattened to the axes the policy trades between.

    Deliberately decoupled from :class:`~forgeml.evaluation.evaluator.EvaluationReport`
    so the selector can be fed straight from an MLflow search — which is how the
    Databricks job uses it, and how you can re-run selection with different weights
    months later without re-running a single GPU job.
    """

    run_id: str
    run_name: str
    method: str
    quality: float
    latency_ms: float
    memory_mb: float
    model_size_mb: float
    cost_per_1k_usd: float = 0.0
    is_baseline: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # Populated by select_champion
    normalized: dict[str, float] = field(default_factory=dict)
    utility: float = 0.0
    rejected_reasons: list[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return not self.rejected_reasons

    def raw_metrics(self) -> dict[str, float]:
        return {
            "quality": self.quality,
            "latency": self.latency_ms,
            "memory": self.memory_mb,
            "model_size": self.model_size_mb,
            "cost": self.cost_per_1k_usd,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "method": self.method,
            "quality": self.quality,
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "model_size_mb": self.model_size_mb,
            "cost_per_1k_usd": self.cost_per_1k_usd,
            "is_baseline": self.is_baseline,
            "utility": round(self.utility, 6),
            "normalized": {k: round(v, 4) for k, v in self.normalized.items()},
            "eligible": self.eligible,
            "rejected_reasons": self.rejected_reasons,
            "metadata": self.metadata,
        }


@dataclass
class SelectionResult:
    """The decision, the runner-up, and every reason behind both."""

    champion: ModelCandidate | None
    challenger: ModelCandidate | None
    ranked: list[ModelCandidate]
    rejected: list[ModelCandidate]
    baseline: ModelCandidate | None
    pareto: list[ModelCandidate]
    weights: dict[str, float]

    @property
    def has_champion(self) -> bool:
        return self.champion is not None

    def quality_gain_vs_baseline(self) -> float | None:
        """Absolute quality improvement of the champion over the baseline."""
        if self.champion is None or self.baseline is None:
            return None
        return self.champion.quality - self.baseline.quality

    def speedup_vs_baseline(self) -> float | None:
        """Baseline latency / champion latency. >1 means the champion is faster."""
        if self.champion is None or self.baseline is None:
            return None
        if self.champion.latency_ms <= 0:
            return None
        return self.baseline.latency_ms / self.champion.latency_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion": self.champion.to_dict() if self.champion else None,
            "challenger": self.challenger.to_dict() if self.challenger else None,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "weights": self.weights,
            "pareto_frontier": [c.run_name for c in self.pareto],
            "ranked": [c.to_dict() for c in self.ranked],
            "rejected": [c.to_dict() for c in self.rejected],
            "quality_gain_vs_baseline": self.quality_gain_vs_baseline(),
            "speedup_vs_baseline": self.speedup_vs_baseline(),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    def to_markdown(self) -> str:
        """A results table you can paste straight into the README."""
        header = (
            "| Model | Method | Quality | Latency p95 | Memory | Size | Utility | Status |\n"
            "|---|---|---:|---:|---:|---:|---:|---|"
        )
        rows = []
        for candidate in [*self.ranked, *self.rejected]:
            if candidate is self.champion:
                status = "**champion**"
            elif candidate is self.challenger:
                status = "challenger"
            elif candidate.is_baseline:
                status = "baseline"
            elif candidate.rejected_reasons:
                status = f"rejected ({candidate.rejected_reasons[0]})"
            else:
                status = "-"
            marker = " *" if candidate in self.pareto else ""
            rows.append(
                f"| {candidate.run_name}{marker} | {candidate.method} | "
                f"{candidate.quality:.4f} | {candidate.latency_ms:.0f} ms | "
                f"{candidate.memory_mb:.0f} MB | {candidate.model_size_mb:.0f} MB | "
                f"{candidate.utility:.4f} | {status} |"
            )
        legend = "\n\n`*` = on the Pareto frontier (not dominated on every axis)."
        return "\n".join([header, *rows]) + legend


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_metric(
    values: Sequence[float],
    higher_is_better: bool,
) -> list[float]:
    """Min-max scale to ``[0, 1]``, inverting when smaller is better.

    Two edge cases handled explicitly, because both occur constantly in practice
    and both produce garbage if ignored:

    * **all values identical** — the metric carries no information, so every
      candidate scores 1.0. Scoring 0.0 instead would let a weight silently
      penalise everybody, and dividing by the zero range would raise.
    * **a single candidate** — same reasoning; it is trivially the best and the
      worst, so it gets 1.0.
    """
    if not values:
        return []

    lowest, highest = min(values), max(values)
    if highest - lowest < 1e-12:
        return [1.0] * len(values)

    span = highest - lowest
    if higher_is_better:
        return [(v - lowest) / span for v in values]
    return [(highest - v) / span for v in values]


def _normalize_all(candidates: Sequence[ModelCandidate]) -> None:
    """Populate ``candidate.normalized`` in place across the candidate set."""
    if not candidates:
        return
    for metric, higher_better in _HIGHER_IS_BETTER.items():
        raw = [c.raw_metrics()[metric] for c in candidates]
        scaled_values = normalize_metric(raw, higher_better)
        for candidate, scaled in zip(candidates, scaled_values, strict=True):
            candidate.normalized[metric] = scaled


def compute_utility(candidate: ModelCandidate, weights: dict[str, float]) -> float:
    """Weighted sum over normalized metrics.

    Weights are normalized to sum to 1 first, so ``{quality: 10, latency: 5}`` and
    ``{quality: 0.67, latency: 0.33}`` behave identically. That removes a whole
    class of "why did my score jump" confusion when someone edits the config.
    """
    total_weight = sum(abs(w) for w in weights.values())
    if total_weight <= 0:
        return 0.0
    return sum(
        (weight / total_weight) * candidate.normalized.get(metric, 0.0)
        for metric, weight in weights.items()
    )


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def _apply_constraints(
    candidates: Sequence[ModelCandidate],
    config: SelectionConfig,
    baseline: ModelCandidate | None,
) -> None:
    """Record every failed gate on each candidate. In place."""
    for candidate in candidates:
        if candidate.is_baseline:
            # The baseline is a reference point, never a deployment candidate.
            candidate.rejected_reasons.append("is_baseline")
            continue

        if baseline is not None and baseline.quality > 0:
            ratio = candidate.quality / baseline.quality
            if ratio < config.min_quality_ratio_vs_baseline:
                candidate.rejected_reasons.append(
                    f"quality {ratio:.1%} of baseline "
                    f"(< {config.min_quality_ratio_vs_baseline:.0%})"
                )
            elif config.require_beats_baseline and candidate.quality <= baseline.quality:
                candidate.rejected_reasons.append("does not beat baseline quality")

        if config.max_latency_ms is not None and candidate.latency_ms > config.max_latency_ms:
            candidate.rejected_reasons.append(
                f"latency {candidate.latency_ms:.0f}ms > {config.max_latency_ms:.0f}ms"
            )

        if config.max_memory_mb is not None and candidate.memory_mb > config.max_memory_mb:
            candidate.rejected_reasons.append(
                f"memory {candidate.memory_mb:.0f}MB > {config.max_memory_mb:.0f}MB"
            )


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------


def _dominates(a: ModelCandidate, b: ModelCandidate) -> bool:
    """True when ``a`` is at least as good as ``b`` everywhere and strictly better once."""
    at_least_as_good = True
    strictly_better = False

    for metric, higher_better in _HIGHER_IS_BETTER.items():
        a_value = a.raw_metrics()[metric]
        b_value = b.raw_metrics()[metric]
        if higher_better:
            if a_value < b_value:
                at_least_as_good = False
                break
            if a_value > b_value:
                strictly_better = True
        else:
            if a_value > b_value:
                at_least_as_good = False
                break
            if a_value < b_value:
                strictly_better = True

    return at_least_as_good and strictly_better


def pareto_frontier(candidates: Iterable[ModelCandidate]) -> list[ModelCandidate]:
    """The non-dominated set.

    A candidate is on the frontier when no other candidate beats it on every axis
    at once. Anything *off* the frontier is strictly worse than something else and
    should never be deployed regardless of how the weights are set — which makes
    this the one part of the ranking that survives disagreement about priorities.
    """
    items = list(candidates)
    return [a for a in items if not any(_dominates(b, a) for b in items if b is not a)]


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def select_champion(
    candidates: Sequence[ModelCandidate],
    config: SelectionConfig,
) -> SelectionResult:
    """Apply the full policy and return the decision with its full justification.

    Ties on utility are broken by ``config.tie_breaker`` (lower wins). Without an
    explicit tie-break the winner would depend on dict ordering, which is exactly
    the kind of hidden nondeterminism that makes a pipeline untrustworthy.
    """
    candidates = list(candidates)
    if not candidates:
        return SelectionResult(None, None, [], [], None, [], dict(config.weights))

    baseline = next((c for c in candidates if c.is_baseline), None)
    if baseline is None:
        log.warning(
            "no baseline candidate found — quality gates relative to baseline are skipped"
        )

    _normalize_all(candidates)
    for candidate in candidates:
        candidate.utility = compute_utility(candidate, config.weights)

    _apply_constraints(candidates, config, baseline)

    eligible = [c for c in candidates if c.eligible]
    rejected = [c for c in candidates if not c.eligible]

    tie_key = {
        "latency_ms": lambda c: c.latency_ms,
        "memory_mb": lambda c: c.memory_mb,
        "model_size_mb": lambda c: c.model_size_mb,
    }[config.tie_breaker]

    # Sort by utility descending, then by the tie-breaker ascending.
    ranked = sorted(eligible, key=lambda c: (-round(c.utility, 6), tie_key(c)))

    champion = ranked[0] if ranked else None
    challenger = ranked[1] if len(ranked) > 1 else None
    frontier = pareto_frontier([c for c in candidates if not c.is_baseline])

    result = SelectionResult(
        champion=champion,
        challenger=challenger,
        ranked=ranked,
        rejected=rejected,
        baseline=baseline,
        pareto=frontier,
        weights=dict(config.weights),
    )

    _log_decision(result)
    return result


def _log_decision(result: SelectionResult) -> None:
    if result.champion is None:
        log.warning(
            "no champion selected — all %d candidates failed the constraints",
            len(result.rejected),
        )
        for candidate in result.rejected:
            log.warning("  %s: %s", candidate.run_name, "; ".join(candidate.rejected_reasons))
        return

    champion = result.champion
    log.info("champion: %s (utility %.4f)", champion.run_name, champion.utility)
    log.info(
        "  quality %.4f | latency %.0fms | memory %.0fMB | size %.0fMB",
        champion.quality, champion.latency_ms, champion.memory_mb, champion.model_size_mb,
    )

    gain = result.quality_gain_vs_baseline()
    speedup = result.speedup_vs_baseline()
    if gain is not None:
        log.info("  vs baseline: %+.4f quality", gain)
    if speedup is not None:
        log.info("  vs baseline: %.2fx latency", speedup)
    if result.challenger:
        log.info(
            "challenger: %s (utility %.4f)",
            result.challenger.run_name, result.challenger.utility,
        )


def candidate_from_report(
    report: Any,
    run_id: str = "",
    is_baseline: bool = False,
    primary_metric: str | None = None,
) -> ModelCandidate:
    """Adapt an :class:`EvaluationReport` into a :class:`ModelCandidate`."""
    metric_name = primary_metric or "rouge_l"
    return ModelCandidate(
        run_id=run_id,
        run_name=report.run_name,
        method=report.method,
        quality=report.quality.scores.get(metric_name, 0.0),
        latency_ms=report.latency.p95_ms if report.latency else 0.0,
        memory_mb=report.memory.peak_allocated_mb if report.memory else 0.0,
        model_size_mb=report.model_size_mb,
        cost_per_1k_usd=report.cost.cost_per_1k_requests_usd if report.cost else 0.0,
        is_baseline=is_baseline,
        metadata={"num_eval_examples": report.num_eval_examples},
    )
