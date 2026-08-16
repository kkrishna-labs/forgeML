"""The single training entry point shared by all three arms.

``run_training`` does not know whether it is running full fine-tuning, LoRA or
QLoRA. It asks :mod:`model_factory` for a model and trains it. Every difference
between the arms is expressed in configuration, which is exactly why the resulting
comparison means something.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forgeml.data.schema import InstructionRecord
from forgeml.logging_utils import get_logger, log_kv
from forgeml.reproducibility import seed_everything
from forgeml.training.callbacks import build_mlflow_callback
from forgeml.training.dataset import SupervisedCollator, build_dataset
from forgeml.training.model_factory import (
    count_parameters,
    load_model,
    load_tokenizer,
    model_size_mb,
)

if TYPE_CHECKING:  # pragma: no cover
    from forgeml.config import ForgeConfig

log = get_logger(__name__)


@dataclass
class TrainingResult:
    """Everything a downstream stage needs, and nothing it does not."""

    output_dir: str
    method: str
    train_loss: float | None = None
    eval_loss: float | None = None
    train_runtime_s: float = 0.0
    train_samples_per_second: float = 0.0
    total_steps: int = 0
    peak_gpu_memory_mb: float = 0.0
    params_total: int = 0
    params_trainable: int = 0
    model_size_mb: float = 0.0
    train_examples: int = 0
    train_tokens: int = 0
    supervised_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def params_trainable_pct(self) -> float:
        return 100.0 * self.params_trainable / self.params_total if self.params_total else 0.0

    def as_metrics(self) -> dict[str, float]:
        """Flatten to MLflow metrics. Only numeric, only meaningful."""
        metrics: dict[str, float] = {
            "train/runtime_s": round(self.train_runtime_s, 2),
            "train/samples_per_second": round(self.train_samples_per_second, 3),
            "train/total_steps": float(self.total_steps),
            "train/peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 2),
            "train/examples": float(self.train_examples),
            "train/tokens": float(self.train_tokens),
            "train/supervised_tokens": float(self.supervised_tokens),
            "model/params_total": float(self.params_total),
            "model/params_trainable": float(self.params_trainable),
            "model/params_trainable_pct": round(self.params_trainable_pct, 6),
            "model/size_mb": self.model_size_mb,
        }
        if self.train_loss is not None:
            metrics["train/final_loss"] = round(self.train_loss, 6)
        if self.eval_loss is not None:
            metrics["eval/loss"] = round(self.eval_loss, 6)
        return metrics


def run_training(
    config: ForgeConfig,
    train_records: Sequence[InstructionRecord],
    eval_records: Sequence[InstructionRecord] | None = None,
    tokenizer: Any | None = None,
    log_to_mlflow: bool = True,
) -> tuple[Any, Any, TrainingResult]:
    """Fine-tune and return ``(model, tokenizer, result)``.

    The model is returned in memory rather than only written to disk so that the
    caller can evaluate it immediately without a reload — reloading is the step
    where "the adapter silently did not apply" bugs come from.
    """
    from transformers import Trainer

    seed_everything(config.seed)

    tokenizer = tokenizer or load_tokenizer(config)
    model = load_model(config, for_training=True)

    train_dataset = build_dataset(train_records, tokenizer, config)
    eval_dataset = (
        build_dataset(eval_records, tokenizer, config)
        if eval_records and config.training.eval_steps
        else None
    )

    output_dir = Path(config.training.output_dir) / config.run_slug()
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = count_parameters(model)
    log_kv(
        log,
        f"training {config.run_slug()}",
        {
            "method": config.training.method,
            "train examples": len(train_dataset),
            "eval examples": len(eval_dataset) if eval_dataset else 0,
            "effective batch": config.training.effective_batch_size,
            "learning rate": config.training.learning_rate,
            "epochs": config.training.epochs,
            "trainable params": f"{counts.trainable:,} ({counts.trainable_pct:.4f}%)",
            "output dir": str(output_dir),
        },
    )

    args = _build_training_arguments(config, output_dir, eval_dataset is not None)
    callback = build_mlflow_callback() if log_to_mlflow else None

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedCollator(tokenizer),
        callbacks=[callback] if callback else None,
    )

    _reset_peak_memory()
    started = time.perf_counter()
    train_output = trainer.train()
    elapsed = time.perf_counter() - started

    eval_loss: float | None = None
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss")

    peak_memory = max(
        _peak_gpu_memory_mb(),
        getattr(callback, "peak_memory_mb", 0.0) if callback else 0.0,
    )

    result = TrainingResult(
        output_dir=str(output_dir),
        method=config.training.method,
        train_loss=train_output.metrics.get("train_loss"),
        eval_loss=eval_loss,
        train_runtime_s=elapsed,
        train_samples_per_second=(len(train_dataset) * config.training.epochs) / elapsed
        if elapsed > 0
        else 0.0,
        total_steps=int(train_output.global_step),
        peak_gpu_memory_mb=peak_memory,
        params_total=counts.total,
        params_trainable=counts.trainable,
        model_size_mb=model_size_mb(model),
        train_examples=len(train_dataset),
        train_tokens=train_dataset.total_tokens,
        supervised_tokens=train_dataset.supervised_tokens,
    )

    _save_artifacts(trainer, tokenizer, output_dir, config)

    log.info(
        "training finished in %.1fs | %d steps | final loss %.4f",
        elapsed,
        result.total_steps,
        result.train_loss or float("nan"),
    )
    return model, tokenizer, result


def _build_training_arguments(config: ForgeConfig, output_dir: Path, has_eval: bool) -> Any:
    from transformers import TrainingArguments

    training = config.training

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": training.epochs,
        "max_steps": training.max_steps,
        "learning_rate": training.learning_rate,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "per_device_eval_batch_size": training.per_device_eval_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "warmup_ratio": training.warmup_ratio,
        "weight_decay": training.weight_decay,
        "lr_scheduler_type": training.lr_scheduler_type,
        "max_grad_norm": training.max_grad_norm,
        "optim": training.optim,
        "gradient_checkpointing": training.gradient_checkpointing,
        "logging_steps": training.logging_steps,
        "save_total_limit": training.save_total_limit,
        "seed": training.seed,
        "data_seed": training.seed,
        "bf16": training.bf16,
        "fp16": training.fp16,
        "report_to": [],  # we own MLflow logging; disable the built-in integration
        "remove_unused_columns": False,  # our dataset yields plain dicts
        "save_strategy": "steps" if training.save_steps else "no",
        "logging_first_step": True,
        "disable_tqdm": False,
    }

    if training.save_steps:
        kwargs["save_steps"] = training.save_steps

    if has_eval and training.eval_steps:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = training.eval_steps
    else:
        kwargs["eval_strategy"] = "no"

    # Required when gradient checkpointing meets PEFT, otherwise you get
    # "element 0 of tensors does not require grad".
    if training.gradient_checkpointing:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return TrainingArguments(**kwargs)


def _save_artifacts(trainer: Any, tokenizer: Any, output_dir: Path, config: ForgeConfig) -> None:
    """Persist the model (or adapter) plus tokenizer.

    For LoRA/QLoRA this writes only the adapter — a few MB instead of gigabytes.
    That size difference is itself a result worth reporting.
    """
    from forgeml.config import dump_config

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    dump_config(config, output_dir / "resolved_config.yaml")

    size_mb = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / 1024**2
    log.info("saved artifacts to %s (%.1f MB on disk)", output_dir, size_mb)

    # Intermediate checkpoints are dead weight once the final model is saved and
    # will happily fill a Databricks volume.
    for checkpoint in output_dir.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint, ignore_errors=True)


def _reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _peak_gpu_memory_mb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:  # noqa: BLE001
        return 0.0
