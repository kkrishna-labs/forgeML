"""The evaluation orchestrator.

One call in — a model and a test split — and one :class:`EvaluationReport` out,
carrying every axis the selector needs. Crucially, every arm goes through this
same function, so "QLoRA is 30ms faster" is a statement about the models rather
than about two people's benchmarking scripts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forgeml.data.schema import InstructionRecord
from forgeml.evaluation.cost import CostEstimate
from forgeml.evaluation.latency import LatencyResult, measure_generation_latency
from forgeml.evaluation.memory import MemoryResult, measure_inference_memory, weights_size_mb
from forgeml.evaluation.perplexity import PerplexityResult, compute_perplexity
from forgeml.evaluation.quality import QualityScores, score_predictions
from forgeml.logging_utils import get_logger, log_kv
from forgeml.prompts import format_prompt

if TYPE_CHECKING:  # pragma: no cover
    from forgeml.config import ForgeConfig

log = get_logger(__name__)


@dataclass
class EvaluationReport:
    """Every measured axis for one model, plus the samples behind the numbers."""

    run_name: str
    method: str
    quality: QualityScores
    latency: LatencyResult | None = None
    memory: MemoryResult | None = None
    perplexity: PerplexityResult | None = None
    cost: CostEstimate | None = None
    model_size_mb: float = 0.0
    num_eval_examples: int = 0
    samples: list[dict[str, str]] = field(default_factory=list)

    def as_metrics(self) -> dict[str, float]:
        """Flatten everything into one MLflow-loggable dict."""
        metrics: dict[str, float] = {"model/size_mb": round(self.model_size_mb, 2)}
        metrics.update(self.quality.as_metrics())
        if self.perplexity:
            metrics.update(self.perplexity.as_metrics())
        if self.latency:
            metrics.update(self.latency.as_metrics())
        if self.memory:
            metrics.update(self.memory.as_metrics())
        if self.cost:
            metrics.update(self.cost.as_metrics())
        return metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "method": self.method,
            "num_eval_examples": self.num_eval_examples,
            "model_size_mb": self.model_size_mb,
            "quality": self.quality.scores,
            "perplexity": asdict(self.perplexity) if self.perplexity else None,
            "latency": {k: v for k, v in asdict(self.latency).items() if k != "raw_ms"}
            if self.latency
            else None,
            "memory": asdict(self.memory) if self.memory else None,
            "cost": self.cost.as_metrics() if self.cost else None,
            "samples": self.samples,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    def summary(self) -> str:
        lines = [f"{self.run_name}  [{self.method}]"]
        for name, value in self.quality.scores.items():
            lines.append(f"  {name:<16} {value:.4f}")
        if self.perplexity:
            lines.append(f"  {'perplexity':<16} {self.perplexity.perplexity:.3f}")
        if self.latency:
            lines.append(f"  {'latency p95':<16} {self.latency.p95_ms:.1f} ms")
        if self.memory:
            lines.append(f"  {'peak memory':<16} {self.memory.peak_allocated_mb:.0f} MB")
        lines.append(f"  {'model size':<16} {self.model_size_mb:.0f} MB")
        return "\n".join(lines)


def generate_predictions(
    model: Any,
    tokenizer: Any,
    records: Sequence[InstructionRecord],
    config: ForgeConfig,
    batch_size: int = 8,
) -> list[str]:
    """Batch-generate one answer per record.

    Padding is switched to the **left** for the duration. With right padding the
    pad tokens sit between the prompt and the first generated token, so the model
    continues from padding and the outputs degrade in a way that looks like a bad
    fine-tune rather than a bug.
    """
    import torch

    device = next(model.parameters()).device
    generation = config.evaluation.generation

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    was_training = model.training
    model.eval()
    previous_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True

    predictions: list[str] = []
    prompts = [format_prompt(r.instruction, r.context, tokenizer=tokenizer) for r in records]

    try:
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                chunk = prompts[start : start + batch_size]
                encoded = tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.model.max_seq_length - generation.max_new_tokens,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}

                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=generation.max_new_tokens,
                    do_sample=generation.do_sample,
                    temperature=generation.temperature if generation.do_sample else None,
                    top_p=generation.top_p if generation.do_sample else None,
                    top_k=generation.top_k if generation.do_sample else None,
                    repetition_penalty=generation.repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                )

                # Slice off the prompt so we score only what the model produced.
                prompt_length = encoded["input_ids"].shape[1]
                for sequence in output_ids:
                    text = tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
                    predictions.append(text.strip())

                if start % (batch_size * 10) == 0:
                    log.info("generated %d/%d", len(predictions), len(prompts))
    finally:
        tokenizer.padding_side = original_padding_side
        model.config.use_cache = previous_cache
        if was_training:
            model.train()

    return predictions


def evaluate_model(
    model: Any,
    tokenizer: Any,
    test_records: Sequence[InstructionRecord],
    config: ForgeConfig,
    run_name: str | None = None,
    include_latency: bool = True,
    include_memory: bool = True,
) -> EvaluationReport:
    """Run the full evaluation battery and return one report.

    ``include_latency`` / ``include_memory`` exist so the mid-training evaluation
    can skip the expensive hardware benchmarks and still get a quality number.
    """
    records = list(test_records)[: config.evaluation.max_eval_samples]
    name = run_name or config.run_slug()

    log.info("evaluating %s on %d examples", name, len(records))

    predictions = generate_predictions(model, tokenizer, records, config)
    references = [r.response for r in records]

    quality = score_predictions(predictions, references, config.evaluation.metrics)

    perplexity = None
    if config.evaluation.compute_perplexity:
        perplexity = compute_perplexity(model, tokenizer, records, config)

    latency = None
    if include_latency:
        latency = measure_generation_latency(model, tokenizer, config.evaluation)

    memory = None
    if include_memory:
        memory = measure_inference_memory(model, tokenizer)

    cost = None
    if latency is not None:
        cost = CostEstimate(
            gpu_hourly_usd=config.evaluation.gpu_hourly_usd,
            assumed_gpu=config.evaluation.assumed_gpu,
            latency_ms=latency.mean_ms,
            batch_size=latency.batch_size,
            tokens_per_second=latency.tokens_per_second,
        )

    report = EvaluationReport(
        run_name=name,
        method=config.training.method,
        quality=quality,
        latency=latency,
        memory=memory,
        perplexity=perplexity,
        cost=cost,
        model_size_mb=weights_size_mb(model),
        num_eval_examples=len(records),
        samples=_collect_samples(records, predictions, quality, config.evaluation.primary_metric),
    )

    log_kv(log, "evaluation complete", {"run": name, "examples": len(records)})
    log.info("\n%s", report.summary())
    return report


def _collect_samples(
    records: Sequence[InstructionRecord],
    predictions: Sequence[str],
    quality: QualityScores,
    primary_metric: str,
    n_best: int = 5,
    n_worst: int = 10,
) -> list[dict[str, str]]:
    """Keep the best and worst generations as an artifact.

    The worst ones are the useful half. An aggregate score tells you *that* a model
    is weak; ten concrete failures tell you *how*, and that is what decides the
    next experiment.
    """
    if not quality.per_example:
        return []

    scored = [(row.get(primary_metric, 0.0), i) for i, row in enumerate(quality.per_example)]
    scored.sort()
    chosen = [(i, "worst") for _, i in scored[:n_worst]]
    chosen += [(i, "best") for _, i in scored[-n_best:]]

    samples: list[dict[str, str]] = []
    for index, bucket in chosen:
        samples.append(
            {
                "bucket": bucket,
                "score": f"{quality.per_example[index].get(primary_metric, 0.0):.4f}",
                "instruction": records[index].instruction[:400],
                "reference": records[index].response[:400],
                "prediction": predictions[index][:400],
            }
        )
    return samples
