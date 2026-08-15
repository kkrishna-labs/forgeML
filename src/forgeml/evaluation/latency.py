"""Latency and throughput measurement.

Benchmarking a GPU is easy to get wrong in ways that quietly produce numbers off
by 10x. Four rules, all enforced below:

1. **Warm up.** The first forward pass pays for CUDA context creation, kernel
   autotuning and lazy weight materialization. Timing it measures the driver.
2. **Synchronize.** CUDA kernels are asynchronous. Without ``torch.cuda.synchronize()``
   you time how long it took to *enqueue* the work, which is close to zero.
3. **Report percentiles.** A mean hides the tail. Production capacity planning is
   done on p95, so p95 is what gets logged.
4. **Fix the token count.** Latency scales with tokens generated. Comparing a model
   that emitted 20 tokens against one that emitted 200 measures verbosity, not
   speed — so generation length is pinned for the benchmark.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from forgeml.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from forgeml.config import EvaluationConfig

log = get_logger(__name__)


@dataclass
class LatencyResult:
    """Wall-clock latency for a fixed generation workload."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    runs: int
    batch_size: int
    new_tokens: int
    prompt_tokens: int
    tokens_per_second: float
    raw_ms: list[float] = field(default_factory=list)

    def as_metrics(self, prefix: str = "latency") -> dict[str, float]:
        return {
            f"{prefix}/mean_ms": round(self.mean_ms, 3),
            f"{prefix}/median_ms": round(self.median_ms, 3),
            f"{prefix}/p95_ms": round(self.p95_ms, 3),
            f"{prefix}/p99_ms": round(self.p99_ms, 3),
            f"{prefix}/stdev_ms": round(self.stdev_ms, 3),
            f"{prefix}/tokens_per_second": round(self.tokens_per_second, 3),
        }

    def summary(self) -> str:
        return (
            f"batch={self.batch_size} tokens={self.new_tokens} | "
            f"mean {self.mean_ms:.1f}ms  p95 {self.p95_ms:.1f}ms  "
            f"{self.tokens_per_second:.1f} tok/s"
        )


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. No interpolation, no numpy dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round(q * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]


def _synchronize() -> None:
    """Block until every queued CUDA kernel has actually finished."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def measure_generation_latency(
    model: Any,
    tokenizer: Any,
    config: EvaluationConfig,
    batch_size: int = 1,
    prompt: str | None = None,
) -> LatencyResult:
    """Time ``generate()`` over a fixed prompt and a fixed number of new tokens.

    Uses a synthetic prompt of exactly ``latency_prompt_tokens`` tokens so the
    measurement is comparable across models with different tokenizers, and forces
    ``min_new_tokens == max_new_tokens`` so early EOS cannot make a model look fast
    by refusing to answer.
    """
    import torch

    device = next(model.parameters()).device
    text = prompt or _synthetic_prompt(tokenizer, config.latency_prompt_tokens)

    encoded = tokenizer([text] * batch_size, return_tensors="pt", padding=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[1])

    was_training = model.training
    model.eval()
    previous_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True  # KV cache ON: this is the serving configuration

    generate_kwargs = {
        **encoded,
        "max_new_tokens": config.latency_new_tokens,
        "min_new_tokens": config.latency_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.no_grad():
        for _ in range(config.latency_warmup_runs):
            model.generate(**generate_kwargs)
        _synchronize()

        timings_ms: list[float] = []
        for _ in range(config.latency_measured_runs):
            start = time.perf_counter()
            model.generate(**generate_kwargs)
            _synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    model.config.use_cache = previous_cache
    if was_training:
        model.train()

    mean_ms = statistics.fmean(timings_ms)
    total_new_tokens = config.latency_new_tokens * batch_size
    result = LatencyResult(
        mean_ms=mean_ms,
        median_ms=statistics.median(timings_ms),
        p95_ms=percentile(timings_ms, 0.95),
        p99_ms=percentile(timings_ms, 0.99),
        min_ms=min(timings_ms),
        max_ms=max(timings_ms),
        stdev_ms=statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        runs=len(timings_ms),
        batch_size=batch_size,
        new_tokens=config.latency_new_tokens,
        prompt_tokens=prompt_tokens,
        tokens_per_second=total_new_tokens / (mean_ms / 1000.0) if mean_ms > 0 else 0.0,
        raw_ms=timings_ms,
    )
    log.info("latency: %s", result.summary())
    return result


def measure_throughput_curve(
    model: Any,
    tokenizer: Any,
    config: EvaluationConfig,
) -> dict[int, LatencyResult]:
    """Sweep batch size to find where throughput stops improving.

    The shape is the point: per-request latency rises roughly linearly with batch
    size while total throughput rises faster, until memory bandwidth saturates and
    both get worse. Where that knee sits is the serving configuration you actually
    want, and it is invisible if you only ever benchmark batch size 1.
    """
    results: dict[int, LatencyResult] = {}
    for batch_size in config.latency_batch_sizes:
        try:
            results[batch_size] = measure_generation_latency(
                model, tokenizer, config, batch_size=batch_size
            )
        except RuntimeError as exc:
            # OOM at large batch is a legitimate data point, not a crash.
            if "out of memory" in str(exc).lower():
                log.warning("batch_size=%d does not fit in memory — stopping sweep", batch_size)
                _empty_cache()
                break
            raise
    return results


def _synthetic_prompt(tokenizer: Any, target_tokens: int) -> str:
    """Build a prompt of approximately ``target_tokens`` tokens.

    Repeats a neutral sentence and trims by decoding a truncated token slice, so
    the length is right regardless of the tokenizer's vocabulary.
    """
    seed = (
        "Explain the following concept clearly and give one concrete example. "
        "Keep the explanation grounded and precise. "
    )
    text = seed * max(1, target_tokens // 12)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:target_tokens]
    return tokenizer.decode(ids)


def _empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
