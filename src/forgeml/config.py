"""Typed, composable YAML configuration.

Every experiment in ForgeML is fully described by one config object. That object is
what gets logged to MLflow as parameters, so "reproducible run" reduces to
"same config + same dataset version + same seed".

Composition
-----------
A config file may declare ``extends: base.yaml``. The parent is loaded first and the
child is deep-merged on top. This is why ``lora.yaml`` is fifteen lines instead of
a hundred, and why a change to the base model propagates everywhere at once.

Overrides
---------
``load_config("configs/lora.yaml", overrides={"training.learning_rate": 1e-4})``
lets a notebook or the CLI sweep a single knob without editing files on disk.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

TrainingMethod = Literal["full", "lora", "qlora"]
ComputeDType = Literal["float32", "float16", "bfloat16"]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Which base checkpoint we start from."""

    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "main"
    trust_remote_code: bool = False
    max_seq_length: int = Field(default=1024, ge=64, le=8192)
    dtype: ComputeDType = "bfloat16"
    attn_implementation: str | None = None  # e.g. "sdpa", "flash_attention_2"

    @property
    def short_name(self) -> str:
        """``Qwen/Qwen2.5-0.5B-Instruct`` -> ``Qwen2.5-0.5B-Instruct``."""
        return self.name.split("/")[-1]


class DeltaConfig(BaseModel):
    """Where the medallion tables live in Unity Catalog.

    Free Edition workspaces expose a ``workspace`` catalog; if yours differs, this is
    the single place to change it.
    """

    enabled: bool = False
    catalog: str = "workspace"
    schema_bronze: str = "forgeml_bronze"
    schema_silver: str = "forgeml_silver"
    schema_gold: str = "forgeml_gold"
    table_raw: str = "raw_instructions"
    table_clean: str = "cleaned_instructions"
    table_train: str = "train"
    table_validation: str = "validation"
    table_test: str = "test"

    def fqn(self, layer: Literal["bronze", "silver", "gold"], table: str) -> str:
        """Fully qualified `catalog.schema.table` name."""
        schema = {
            "bronze": self.schema_bronze,
            "silver": self.schema_silver,
            "gold": self.schema_gold,
        }[layer]
        return f"{self.catalog}.{schema}.{table}"


class DataConfig(BaseModel):
    """Dataset identity, cleaning thresholds and split ratios."""

    source: str = "databricks/databricks-dolly-15k"
    source_type: Literal["hf", "jsonl", "delta"] = "hf"
    split: str = "train"
    version: str = "v1.0"
    max_examples: int | None = 5000
    seed: int = 42

    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Cleaning thresholds — every one of these is enforced by data.validator
    min_instruction_chars: int = 8
    max_instruction_chars: int = 4000
    min_response_chars: int = 1
    max_response_chars: int = 4000
    max_context_chars: int = 8000
    dedupe: bool = True
    drop_empty_response: bool = True

    local_dir: str = "data/processed"
    delta: DeltaConfig = Field(default_factory=DeltaConfig)

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> DataConfig:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train/val/test ratios must sum to 1.0, got {total:.6f} "
                f"({self.train_ratio} + {self.val_ratio} + {self.test_ratio})"
            )
        return self


class LoRAConfig(BaseModel):
    """Low-Rank Adaptation hyper-parameters.

    ``target_modules=None`` lets PEFT auto-detect the attention projections for the
    architecture. Set it explicitly when you want the config to be self-documenting.
    """

    r: int = Field(default=8, ge=1, le=256)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] | None = None
    modules_to_save: list[str] | None = None
    task_type: str = "CAUSAL_LM"

    @property
    def scaling(self) -> float:
        """The factor LoRA multiplies its delta by: ``alpha / r``.

        Two configs with the same ratio behave similarly at low ranks; this is worth
        logging as its own metric so the MLflow table stays interpretable.
        """
        return self.alpha / self.r


class QuantizationConfig(BaseModel):
    """bitsandbytes 4-bit / 8-bit loading options (the Q in QLoRA)."""

    enabled: bool = False
    bits: Literal[4, 8] = 4
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: ComputeDType = "bfloat16"


class TrainingConfig(BaseModel):
    """Optimizer / schedule / batching. Shared by all three methods."""

    method: TrainingMethod = "full"
    # 0 is legal and means "evaluate only" — that is how the baseline arm is expressed.
    epochs: float = Field(default=1.0, ge=0)
    learning_rate: float = Field(default=2e-5, gt=0)
    per_device_train_batch_size: int = Field(default=4, ge=1)
    per_device_eval_batch_size: int = Field(default=8, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_steps: int | None = 50
    save_steps: int | None = None
    save_total_limit: int = 1
    max_steps: int = -1  # -1 = derive from epochs; >=0 = hard cap (smoke tests, baseline)
    seed: int = 42
    output_dir: str = "outputs"
    bf16: bool = True
    fp16: bool = False

    @property
    def effective_batch_size(self) -> int:
        """What the optimizer actually sees per update — the number that matters."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @model_validator(mode="after")
    def _one_precision_flag(self) -> TrainingConfig:
        if self.bf16 and self.fp16:
            raise ValueError("set exactly one of bf16 / fp16, not both")
        return self


class GenerationConfig(BaseModel):
    """Decoding settings used during evaluation and serving.

    Evaluation defaults to greedy (``do_sample=False``) because a metric you cannot
    reproduce is not a metric.
    """

    max_new_tokens: int = Field(default=256, ge=1)
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0


class EvaluationConfig(BaseModel):
    """What we measure, and how many samples we measure it on."""

    max_eval_samples: int = Field(default=200, ge=1)
    metrics: list[str] = Field(default_factory=lambda: ["rouge_l", "token_f1", "exact_match"])
    primary_metric: str = "rouge_l"
    compute_perplexity: bool = True

    latency_warmup_runs: int = Field(default=3, ge=0)
    latency_measured_runs: int = Field(default=20, ge=1)
    latency_prompt_tokens: int = 128
    latency_new_tokens: int = 64
    latency_batch_sizes: list[int] = Field(default_factory=lambda: [1])

    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    # Cost model — a transparent, documented approximation, not a vendor quote.
    gpu_hourly_usd: float = 0.60
    assumed_gpu: str = "T4-16GB"


class SelectionConfig(BaseModel):
    """The policy that turns a table of runs into one champion.

    ``weights`` are applied to *normalized* metrics (0 = worst observed, 1 = best
    observed) so that milliseconds and gigabytes can be summed without nonsense.
    """

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "quality": 0.55,
            "latency": 0.20,
            "memory": 0.15,
            "model_size": 0.05,
            "cost": 0.05,
        }
    )
    # Hard gates. A run failing any of these can never be champion, whatever its score.
    min_quality_ratio_vs_baseline: float = 0.90
    max_latency_ms: float | None = None
    max_memory_mb: float | None = None
    require_beats_baseline: bool = True
    tie_breaker: Literal["latency_ms", "memory_mb", "model_size_mb"] = "latency_ms"


class TrackingConfig(BaseModel):
    """MLflow wiring."""

    experiment_name: str = "/Shared/forgeml"
    run_name: str | None = None
    registry_uri: str | None = None  # "databricks-uc" for Unity Catalog
    tracking_uri: str | None = None  # None -> local ./mlruns, or Databricks default
    registered_model_name: str = "forgeml_champion"
    log_model: bool = True
    log_system_metrics: bool = True
    tags: dict[str, str] = Field(default_factory=dict)


class ForgeConfig(BaseModel):
    """The whole experiment in one object."""

    project: str = "forgeml"
    seed: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoRAConfig | None = None
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @field_validator("seed")
    @classmethod
    def _non_negative_seed(cls, v: int) -> int:
        if v < 0:
            raise ValueError("seed must be >= 0")
        return v

    @model_validator(mode="after")
    def _method_consistency(self) -> ForgeConfig:
        """Catch the three mistakes people actually make."""
        method = self.training.method
        if method in ("lora", "qlora") and self.lora is None:
            raise ValueError(f"training.method='{method}' requires a `lora:` section")
        if method == "qlora" and not self.quantization.enabled:
            raise ValueError("training.method='qlora' requires quantization.enabled=true")
        if method == "full" and self.lora is not None:
            raise ValueError("training.method='full' must not define a `lora:` section")
        return self

    # -- MLflow-facing helpers ---------------------------------------------

    def run_slug(self) -> str:
        """Short, stable, human-readable run name. Shows up in the MLflow UI."""
        parts = [self.model.short_name, self.training.method]
        if self.lora is not None:
            parts.append(f"r{self.lora.r}a{self.lora.alpha}")
        if self.quantization.enabled:
            parts.append(f"{self.quantization.bits}bit")
        parts.append(self.data.version)
        return "-".join(parts)

    def flat_params(self) -> dict[str, Any]:
        """Flatten to ``section.key`` pairs for ``mlflow.log_params``.

        MLflow stores params as strings and caps them at 500 chars, so lists and
        dicts are JSON-encoded and long values are truncated rather than dropped.
        """
        flat: dict[str, Any] = {}

        def walk(prefix: str, value: Any) -> None:
            if isinstance(value, BaseModel):
                value = value.model_dump()
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(value, (list, tuple)):
                flat[prefix] = json.dumps(list(value))
            elif value is None:
                flat[prefix] = "null"
            else:
                flat[prefix] = value

        walk("", self.model_dump())

        # Derived values worth having as first-class columns in the MLflow table.
        flat["training.effective_batch_size"] = self.training.effective_batch_size
        if self.lora is not None:
            flat["lora.scaling"] = round(self.lora.scaling, 4)

        return {k: _truncate(v) for k, v in flat.items()}


def _truncate(value: Any, limit: int = 490) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - 3] + "..."
    return value


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge. ``override`` wins; ``None`` is a real value (it clears)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return loaded


def _resolve_chain(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Follow ``extends:`` upward, then merge back down."""
    seen = _seen or set()
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"circular `extends` detected at {resolved}")
    seen.add(resolved)

    raw = _read_yaml(resolved)
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw

    parent_path = (resolved.parent / str(parent_ref)).resolve()
    return _deep_merge(_resolve_chain(parent_path, seen), raw)


def _apply_dotted(tree: dict[str, Any], dotted_key: str, value: Any) -> None:
    """``training.learning_rate`` -> ``tree["training"]["learning_rate"]``."""
    parts = dotted_key.split(".")
    node = tree
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> ForgeConfig:
    """Load a YAML config, resolve ``extends``, apply dotted overrides, validate.

    Environment overrides are applied last: any ``FORGEML__training__epochs=2`` style
    variable maps to ``training.epochs``. Handy on Databricks job clusters where you
    can set env vars but not edit files.
    """
    tree = _resolve_chain(Path(path))

    for key, value in (overrides or {}).items():
        _apply_dotted(tree, key, value)

    for env_key, env_value in os.environ.items():
        if not env_key.upper().startswith("FORGEML__"):
            continue
        # Windows normalises environment variable names to upper case, so
        # FORGEML__training__epochs arrives as FORGEML__TRAINING__EPOCHS. Every
        # config key is lower-case snake_case, so folding the case here makes the
        # override behave identically on Windows and on a Databricks job cluster.
        dotted = env_key[len("FORGEML__") :].replace("__", ".").lower()
        _apply_dotted(tree, dotted, yaml.safe_load(env_value))

    return ForgeConfig.model_validate(tree)


def dump_config(config: ForgeConfig, path: str | Path) -> Path:
    """Write the *fully resolved* config next to the run artifacts.

    This is the file that makes a run reproducible six months later — not the
    fifteen-line override you happened to launch it with.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.model_dump(), fh, sort_keys=False, default_flow_style=False)
    return target
