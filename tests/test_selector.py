"""Champion selection.

This is the module that decides what gets deployed, so it gets the most tests.
The scenarios below are the ones that actually occur: a model that wins on quality
and loses everywhere else, a candidate that fails a hard gate, an experiment where
nothing clears the bar, and ties.
"""

from __future__ import annotations

import pytest

from forgeml.config import SelectionConfig
from forgeml.optimization.selector import (
    ModelCandidate,
    compute_utility,
    normalize_metric,
    pareto_frontier,
    select_champion,
)


def make(
    name: str,
    quality: float,
    latency: float = 200.0,
    memory: float = 2000.0,
    size: float = 1000.0,
    cost: float = 0.04,
    baseline: bool = False,
) -> ModelCandidate:
    return ModelCandidate(
        run_id=f"run-{name}",
        run_name=name,
        method="lora",
        quality=quality,
        latency_ms=latency,
        memory_mb=memory,
        model_size_mb=size,
        cost_per_1k_usd=cost,
        is_baseline=baseline,
    )


@pytest.fixture
def config() -> SelectionConfig:
    return SelectionConfig()


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------


def test_normalize_higher_is_better() -> None:
    assert normalize_metric([0.0, 5.0, 10.0], higher_is_better=True) == [0.0, 0.5, 1.0]


def test_normalize_inverts_lower_is_better() -> None:
    """Latency of 100 must score 1.0, not 0.0."""
    assert normalize_metric([100.0, 200.0, 300.0], higher_is_better=False) == [1.0, 0.5, 0.0]


def test_normalize_identical_values_all_score_one() -> None:
    """A metric that carries no information must not silently penalise anyone."""
    assert normalize_metric([5.0, 5.0, 5.0], higher_is_better=True) == [1.0, 1.0, 1.0]


def test_normalize_single_candidate() -> None:
    assert normalize_metric([42.0], higher_is_better=False) == [1.0]


def test_normalize_empty() -> None:
    assert normalize_metric([], higher_is_better=True) == []


# ---------------------------------------------------------------------------
# utility
# ---------------------------------------------------------------------------


def test_utility_normalizes_the_weights() -> None:
    """Scaling every weight must not change the score."""
    candidate = make("a", 0.8)
    candidate.normalized = {
        "quality": 1.0,
        "latency": 0.5,
        "memory": 0.0,
        "model_size": 0.0,
        "cost": 0.0,
    }
    small = compute_utility(candidate, {"quality": 0.5, "latency": 0.5})
    large = compute_utility(candidate, {"quality": 50, "latency": 50})
    assert small == pytest.approx(large)


def test_utility_is_zero_when_all_weights_are_zero() -> None:
    candidate = make("a", 0.8)
    candidate.normalized = {"quality": 1.0}
    assert compute_utility(candidate, {"quality": 0.0}) == 0.0


# ---------------------------------------------------------------------------
# the core trade-off
# ---------------------------------------------------------------------------


def test_highest_quality_does_not_automatically_win(config: SelectionConfig) -> None:
    """The whole reason this module exists.

    `slow_best` wins on quality by half a point and is three times slower and
    twice as heavy. Under the default weights it must lose.
    """
    candidates = [
        make("baseline", 0.70, baseline=True),
        make("slow_best", 0.90, latency=600, memory=6000, size=2400),
        make("balanced", 0.85, latency=200, memory=2000, size=800),
    ]
    result = select_champion(candidates, config)

    assert result.champion is not None
    assert result.champion.run_name == "balanced"
    assert result.challenger is not None
    assert result.challenger.run_name == "slow_best"


def test_quality_wins_when_the_weights_say_so(config: SelectionConfig) -> None:
    """Same candidates, different priorities, different answer — as it should be."""
    config.weights = {"quality": 1.0, "latency": 0.0, "memory": 0.0, "model_size": 0.0, "cost": 0.0}
    candidates = [
        make("baseline", 0.70, baseline=True),
        make("slow_best", 0.90, latency=600, memory=6000, size=2400),
        make("balanced", 0.85, latency=200, memory=2000, size=800),
    ]
    result = select_champion(candidates, config)
    assert result.champion is not None
    assert result.champion.run_name == "slow_best"


# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------


def test_baseline_is_never_the_champion(config: SelectionConfig) -> None:
    """Even when the baseline scores highest, it is a reference point, not a candidate."""
    config.require_beats_baseline = False
    config.min_quality_ratio_vs_baseline = 0.0

    result = select_champion([make("baseline", 0.99, baseline=True), make("tuned", 0.80)], config)
    assert result.champion is not None
    assert result.champion.run_name == "tuned"
    assert result.baseline is not None
    assert "is_baseline" in result.baseline.rejected_reasons


def test_candidate_below_the_quality_floor_is_rejected(config: SelectionConfig) -> None:
    config.min_quality_ratio_vs_baseline = 0.90
    result = select_champion(
        [make("baseline", 0.70, baseline=True), make("weak", 0.50), make("ok", 0.95)],
        config,
    )
    assert result.champion is not None
    assert result.champion.run_name == "ok"

    weak = next(c for c in result.rejected if c.run_name == "weak")
    assert "71.4% of baseline" in weak.rejected_reasons[0]


def test_candidate_that_does_not_beat_baseline_is_rejected(config: SelectionConfig) -> None:
    config.require_beats_baseline = True
    result = select_champion(
        [make("baseline", 0.80, baseline=True), make("no_better", 0.80)], config
    )
    assert result.champion is None
    assert "does not beat baseline quality" in result.rejected[-1].rejected_reasons


def test_latency_ceiling_is_enforced(config: SelectionConfig) -> None:
    config.max_latency_ms = 300
    result = select_champion(
        [
            make("baseline", 0.70, baseline=True),
            make("fast", 0.80, latency=250),
            make("slow", 0.95, latency=500),
        ],
        config,
    )
    assert result.champion is not None
    assert result.champion.run_name == "fast"


def test_memory_ceiling_is_enforced(config: SelectionConfig) -> None:
    config.max_memory_mb = 2500
    result = select_champion(
        [
            make("baseline", 0.70, baseline=True),
            make("light", 0.80, memory=2000),
            make("heavy", 0.95, memory=8000),
        ],
        config,
    )
    assert result.champion is not None
    assert result.champion.run_name == "light"


def test_no_champion_when_everything_fails(config: SelectionConfig) -> None:
    """A pipeline that always ships something will eventually ship a regression."""
    config.max_latency_ms = 10
    result = select_champion(
        [make("baseline", 0.70, baseline=True), make("a", 0.90), make("b", 0.95)], config
    )
    assert result.champion is None
    assert not result.has_champion
    assert len(result.rejected) == 3


def test_constraints_are_evaluated_even_without_a_baseline(config: SelectionConfig) -> None:
    config.max_latency_ms = 300
    result = select_champion([make("a", 0.9, latency=500), make("b", 0.8, latency=100)], config)
    assert result.baseline is None
    assert result.champion is not None
    assert result.champion.run_name == "b"


# ---------------------------------------------------------------------------
# pareto
# ---------------------------------------------------------------------------


def test_dominated_candidate_is_off_the_frontier() -> None:
    """`worse` loses on every single axis, so nothing can justify deploying it."""
    better = make("better", 0.90, latency=100, memory=1000, size=500, cost=0.01)
    worse = make("worse", 0.80, latency=200, memory=2000, size=1000, cost=0.02)
    frontier = pareto_frontier([better, worse])
    assert [c.run_name for c in frontier] == ["better"]


def test_trade_offs_both_stay_on_the_frontier() -> None:
    fast = make("fast", 0.80, latency=100)
    accurate = make("accurate", 0.95, latency=400)
    assert len(pareto_frontier([fast, accurate])) == 2


def test_identical_candidates_both_survive() -> None:
    """Domination requires being strictly better somewhere, not merely equal."""
    a, b = make("a", 0.8), make("b", 0.8)
    assert len(pareto_frontier([a, b])) == 2


def test_frontier_excludes_the_baseline(config: SelectionConfig) -> None:
    result = select_champion(
        [
            make("baseline", 0.99, latency=10, memory=10, size=10, cost=0.001, baseline=True),
            make("tuned", 0.80),
        ],
        config,
    )
    assert "baseline" not in [c.run_name for c in result.pareto]


# ---------------------------------------------------------------------------
# ordering + reporting
# ---------------------------------------------------------------------------


def test_ties_are_broken_deterministically(config: SelectionConfig) -> None:
    """Without an explicit tie-break the winner depends on dict ordering."""
    config.tie_breaker = "latency_ms"
    a = make("a", 0.85, latency=300)
    b = make("b", 0.85, latency=100)
    # identical on every axis except latency, so utility ties on the rest
    first = select_champion([make("base", 0.7, baseline=True), a, b], config)
    second = select_champion([make("base", 0.7, baseline=True), b, a], config)
    assert first.champion is not None and second.champion is not None
    assert first.champion.run_name == second.champion.run_name == "b"


def test_ranking_is_sorted_by_utility(config: SelectionConfig) -> None:
    result = select_champion(
        [make("base", 0.70, baseline=True), make("a", 0.75), make("b", 0.90), make("c", 0.82)],
        config,
    )
    utilities = [c.utility for c in result.ranked]
    assert utilities == sorted(utilities, reverse=True)


def test_comparisons_against_baseline(config: SelectionConfig) -> None:
    result = select_champion(
        [make("baseline", 0.70, latency=400, baseline=True), make("tuned", 0.85, latency=200)],
        config,
    )
    assert result.quality_gain_vs_baseline() == pytest.approx(0.15)
    assert result.speedup_vs_baseline() == pytest.approx(2.0)


def test_result_serialises_and_renders(config: SelectionConfig, tmp_path) -> None:
    result = select_champion(
        [make("baseline", 0.70, baseline=True), make("a", 0.85), make("b", 0.80)], config
    )
    payload = result.to_dict()
    assert payload["champion"]["run_name"] == "a"
    assert "quality_gain_vs_baseline" in payload

    path = result.save(tmp_path / "selection.json")
    assert path.exists()

    markdown = result.to_markdown()
    assert "champion" in markdown
    assert "| a " in markdown


def test_empty_candidate_list_is_handled(config: SelectionConfig) -> None:
    result = select_champion([], config)
    assert result.champion is None
    assert result.ranked == []
