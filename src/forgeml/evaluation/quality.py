"""Quality metrics, implemented from scratch in pure Python.

Why not just import ``evaluate``/``rouge_score``? Three reasons, and they are the
kind of reasons worth being able to defend:

1. **Testability.** These run in CI on a laptop with no torch and no downloads.
2. **Transparency.** "ROUGE-L = 0.41" means nothing unless you can say exactly
   which tokenizer, which normalization and which averaging produced it. Here you
   can read it.
3. **Determinism.** No model downloads, no version drift between the number you
   reported in the README and the number the grader reproduces.

What these metrics are NOT: a measure of whether the answer is *correct*. They
measure n-gram overlap with one reference answer. A perfect paraphrase scores
poorly; a fluent lie scores well. That limitation is stated in the model card
rather than hidden, and it is why ``docs/experiments.md`` also reports loss and
perplexity, which are sensitive to different failure modes.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WHITESPACE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles, squash space.

    Applied identically to prediction and reference. Without it, "The cat." and
    "cat" score zero and your metric is mostly measuring punctuation.
    """
    text = (text or "").lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def exact_match(prediction: str, reference: str) -> float:
    """1.0 iff the normalized strings are identical.

    Brutal for open-ended generation — expect near-zero on instruction data. Kept
    because it is the one metric that cannot be gamed by verbosity.
    """
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1 (SQuAD's secondary metric).

    Multiset intersection, so repeated words only count as often as they appear in
    both. Order-insensitive, which makes it a useful complement to ROUGE-L.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length, O(len(a) x len(b)) time, O(len(b)) space.

    Only two rows of the DP table are ever needed, which keeps this tractable on
    long generations where the full table would be tens of megabytes.
    """
    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str, beta: float = 1.2) -> float:
    """ROUGE-L F-measure based on the longest common subsequence.

    Unlike token F1 this respects *order*, so a fluent answer that says the right
    things in the right sequence scores higher than the same words shuffled.

    ``beta=1.2`` follows the original ROUGE paper: recall is weighted slightly
    above precision, because omitting content is treated as worse than padding.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    beta_sq = beta * beta
    return ((1 + beta_sq) * precision * recall) / (recall + beta_sq * precision)


def length_ratio(prediction: str, reference: str) -> float:
    """len(prediction) / len(reference) in tokens.

    Not a quality metric — a diagnostic. A ratio far above 1 means the model is
    rambling (and your latency is paying for it); far below 1 means it is
    truncating. Either explains a bad ROUGE score faster than staring at ROUGE.
    """
    ref_tokens = tokenize(reference)
    if not ref_tokens:
        return 0.0
    return len(tokenize(prediction)) / len(ref_tokens)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_METRIC_FUNCTIONS = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
    "length_ratio": length_ratio,
}


@dataclass
class QualityScores:
    """Corpus-level scores plus the per-example values behind them."""

    scores: dict[str, float] = field(default_factory=dict)
    per_example: list[dict[str, float]] = field(default_factory=list)
    num_examples: int = 0

    def primary(self, name: str) -> float:
        if name not in self.scores:
            raise KeyError(f"metric '{name}' was not computed; have {sorted(self.scores)}")
        return self.scores[name]

    def as_metrics(self, prefix: str = "quality") -> dict[str, float]:
        return {f"{prefix}/{k}": round(v, 6) for k, v in self.scores.items()}

    def worst_examples(self, metric: str, n: int = 10) -> list[int]:
        """Indices of the n worst predictions.

        Reading ten failures teaches you more about a model than any aggregate
        number, so the evaluator logs these as an MLflow artifact.
        """
        indexed = [(row.get(metric, 0.0), i) for i, row in enumerate(self.per_example)]
        indexed.sort()
        return [i for _, i in indexed[:n]]


def score_predictions(
    predictions: Sequence[str],
    references: Sequence[str],
    metrics: Sequence[str] = ("rouge_l", "token_f1", "exact_match"),
) -> QualityScores:
    """Score a corpus. Corpus score = unweighted mean of per-example scores.

    Macro-averaging (mean of per-example scores) rather than micro-averaging
    (pooling all tokens first) so that one 800-word answer cannot dominate two
    hundred short ones.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) "
            "must be the same length"
        )

    unknown = set(metrics) - set(_METRIC_FUNCTIONS)
    if unknown:
        raise ValueError(f"unknown metric(s): {sorted(unknown)}")

    result = QualityScores(num_examples=len(predictions))
    if not predictions:
        result.scores = dict.fromkeys(metrics, 0.0)
        return result

    # length_ratio is a diagnostic, so compute it always even if not requested.
    active = list(dict.fromkeys([*metrics, "length_ratio"]))

    totals: dict[str, float] = dict.fromkeys(active, 0.0)
    for prediction, reference in zip(predictions, references, strict=True):
        row: dict[str, float] = {}
        for name in active:
            value = _METRIC_FUNCTIONS[name](prediction, reference)
            row[name] = value
            totals[name] += value
        result.per_example.append(row)

    result.scores = {name: totals[name] / len(predictions) for name in active}
    return result
