"""Quality metrics, cost model and percentile maths.

Metrics implemented by hand need tests more than metrics imported from a library,
so these pin down the exact behaviour — including the cases where the metric is
deliberately harsh.
"""

from __future__ import annotations

import pytest

from forgeml.evaluation.cost import CostEstimate, estimate_training_cost
from forgeml.evaluation.latency import percentile
from forgeml.evaluation.quality import (
    _lcs_length,
    exact_match,
    length_ratio,
    normalize_answer,
    rouge_l,
    score_predictions,
    token_f1,
)

# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------


def test_normalization_strips_case_punctuation_and_articles() -> None:
    assert normalize_answer("The Cat, sat!") == "cat sat"
    assert normalize_answer("  A   dog  ") == "dog"
    assert normalize_answer("") == ""


def test_normalization_makes_trivial_differences_score_perfectly() -> None:
    """Without it, punctuation alone would dominate the metric."""
    assert exact_match("The answer is 42.", "answer is 42") == 1.0


# ---------------------------------------------------------------------------
# exact match / token f1
# ---------------------------------------------------------------------------


def test_exact_match_is_binary() -> None:
    assert exact_match("hello world", "hello world") == 1.0
    assert exact_match("hello world", "hello there") == 0.0


def test_token_f1_is_order_insensitive() -> None:
    assert token_f1("cat sat mat", "mat sat cat") == pytest.approx(1.0)


def test_token_f1_partial_overlap() -> None:
    # prediction 3 tokens, reference 4, overlap 3 -> P=1.0 R=0.75 F1=6/7
    assert token_f1("cat sat mat", "cat sat mat quietly") == pytest.approx(6 / 7)


def test_token_f1_handles_repeats_as_a_multiset() -> None:
    """'cat cat' vs 'cat' overlaps once, not twice."""
    assert token_f1("cat cat", "cat") == pytest.approx(2 / 3)


def test_token_f1_empty_cases() -> None:
    assert token_f1("", "") == 1.0
    assert token_f1("something", "") == 0.0
    assert token_f1("", "something") == 0.0


# ---------------------------------------------------------------------------
# rouge-l
# ---------------------------------------------------------------------------


def test_lcs_length() -> None:
    assert _lcs_length(list("abcde"), list("ace")) == 3
    assert _lcs_length(list("abc"), list("xyz")) == 0
    assert _lcs_length([], list("abc")) == 0


def test_rouge_l_is_one_for_identical_text() -> None:
    assert rouge_l("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_rouge_l_respects_word_order() -> None:
    """This is the property that distinguishes ROUGE-L from token F1."""
    ordered = rouge_l("a b c d", "a b c d")
    shuffled = rouge_l("d c b a", "a b c d")
    assert ordered > shuffled
    # token F1 cannot tell them apart at all
    assert token_f1("d c b a", "a b c d") == pytest.approx(1.0)


def test_rouge_l_zero_when_disjoint() -> None:
    assert rouge_l("alpha beta", "gamma delta") == 0.0


def test_rouge_l_weights_recall_above_precision() -> None:
    """beta=1.2 means omitting content costs more than padding it."""
    reference = "one two three four"
    too_short = rouge_l("one two", reference)  # high precision, low recall
    too_long = rouge_l("one two three four five six", reference)  # the reverse
    assert too_long > too_short


# ---------------------------------------------------------------------------
# diagnostics + aggregation
# ---------------------------------------------------------------------------


def test_length_ratio_detects_rambling_and_truncation() -> None:
    assert length_ratio("one two three four", "one two") == pytest.approx(2.0)
    assert length_ratio("one", "one two three four") == pytest.approx(0.25)


def test_length_ratio_measures_post_normalization_tokens() -> None:
    """Articles are stripped before counting, so 'a b' is one token, not two.

    Worth pinning down: it is the difference between a ratio of 2.0 and 3.0 on
    text that happens to start with an article.
    """
    assert length_ratio("a b c d", "a b") == pytest.approx(3.0)


def test_score_predictions_macro_averages() -> None:
    scores = score_predictions(
        ["perfect match", "completely wrong"],
        ["perfect match", "something else"],
        metrics=("exact_match",),
    )
    assert scores.scores["exact_match"] == pytest.approx(0.5)
    assert scores.num_examples == 2
    assert len(scores.per_example) == 2


def test_score_predictions_always_computes_length_ratio() -> None:
    """A diagnostic you have to remember to ask for is a diagnostic nobody uses."""
    scores = score_predictions(["a"], ["a b"], metrics=("exact_match",))
    assert "length_ratio" in scores.scores


def test_score_predictions_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        score_predictions(["a", "b"], ["a"])


def test_score_predictions_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        score_predictions(["a"], ["a"], metrics=("bleu",))


def test_worst_examples_are_the_lowest_scoring() -> None:
    scores = score_predictions(
        ["exact", "wrong", "exact"],
        ["exact", "right", "exact"],
        metrics=("exact_match",),
    )
    assert scores.worst_examples("exact_match", n=1) == [1]


def test_primary_raises_for_a_metric_that_was_not_computed() -> None:
    scores = score_predictions(["a"], ["a"], metrics=("exact_match",))
    with pytest.raises(KeyError, match="rouge_l"):
        scores.primary("rouge_l")


# ---------------------------------------------------------------------------
# percentiles
# ---------------------------------------------------------------------------


def test_percentile_uses_nearest_rank() -> None:
    values = list(range(1, 101))  # 1..100
    assert percentile(values, 0.0) == 1
    assert percentile(values, 1.0) == 100
    assert percentile(values, 0.5) == pytest.approx(50, abs=1)


def test_percentile_is_order_independent_and_safe_when_empty() -> None:
    assert percentile([3, 1, 2], 1.0) == 3
    assert percentile([], 0.95) == 0.0


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------


def test_cost_scales_inversely_with_latency() -> None:
    fast = CostEstimate(gpu_hourly_usd=1.0, assumed_gpu="T4", latency_ms=100)
    slow = CostEstimate(gpu_hourly_usd=1.0, assumed_gpu="T4", latency_ms=200)
    assert slow.cost_per_1k_requests_usd == pytest.approx(2 * fast.cost_per_1k_requests_usd)


def test_cost_arithmetic_is_what_it_claims() -> None:
    # 100ms per request -> 10 req/s -> 36,000 req/h at $1/h
    estimate = CostEstimate(gpu_hourly_usd=1.0, assumed_gpu="T4", latency_ms=100)
    assert estimate.requests_per_hour == pytest.approx(36_000)
    assert estimate.cost_per_1k_requests_usd == pytest.approx(1000 / 36_000)


def test_cost_is_infinite_when_latency_is_unmeasured() -> None:
    """Zero latency means "not measured", and pretending it is free is worse."""
    assert CostEstimate(1.0, "T4", latency_ms=0).cost_per_request_usd == float("inf")


def test_batching_reduces_cost_per_request() -> None:
    single = CostEstimate(1.0, "T4", latency_ms=100, batch_size=1)
    batched = CostEstimate(1.0, "T4", latency_ms=150, batch_size=8)
    assert batched.cost_per_request_usd < single.cost_per_request_usd


def test_training_cost_is_linear_in_runtime() -> None:
    assert estimate_training_cost(3600, 0.60) == pytest.approx(0.60)
    assert estimate_training_cost(1800, 0.60) == pytest.approx(0.30)
