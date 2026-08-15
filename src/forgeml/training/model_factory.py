"""Model and tokenizer construction — the single place the three arms diverge.

::

    full   -> AutoModelForCausalLM, every parameter trainable
    lora   -> same model, frozen, + PEFT adapters on the target projections
    qlora  -> model loaded in 4-bit NF4, frozen, + PEFT adapters in bf16

Everything downstream (dataset, collator, trainer, evaluator) is identical. Keeping
the divergence to one function is what makes the results table trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from forgeml.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from forgeml.config import ForgeConfig

log = get_logger(__name__)


@dataclass
class ParameterCount:
    """Trainable vs total parameters — the headline number for a PEFT run."""

    total: int
    trainable: int

    @property
    def trainable_pct(self) -> float:
        return 100.0 * self.trainable / self.total if self.total else 0.0

    @property
    def frozen(self) -> int:
        return self.total - self.trainable

    def as_metrics(self) -> dict[str, float]:
        return {
            "params_total": float(self.total),
            "params_trainable": float(self.trainable),
            "params_trainable_pct": round(self.trainable_pct, 6),
        }

    def __str__(self) -> str:
        return (
            f"{self.trainable:,} trainable / {self.total:,} total "
            f"({self.trainable_pct:.4f}%)"
        )


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_tokenizer(config: ForgeConfig) -> Any:
    """Load the tokenizer and make it safe for batched causal-LM training.

    Two adjustments that are easy to miss and expensive to debug:

    * **pad token** — most causal LMs ship without one. Reusing EOS as PAD is the
      standard fix; the padding is masked out of the loss anyway.
    * **padding side** — left for generation (so the newest token is last and the
      KV cache lines up), right for training. We set right here and flip it
      explicitly in the evaluator.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("tokenizer had no pad token; reusing eos (%r)", tokenizer.eos_token)

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = config.model.max_seq_length
    return tokenizer


def _build_bnb_config(config: ForgeConfig) -> Any | None:
    """Translate our quantization section into a ``BitsAndBytesConfig``."""
    if not config.quantization.enabled:
        return None

    from transformers import BitsAndBytesConfig

    quant = config.quantization
    if quant.bits == 8:
        log.info("loading base model in 8-bit")
        return BitsAndBytesConfig(load_in_8bit=True)

    log.info(
        "loading base model in 4-bit (%s, double_quant=%s, compute=%s)",
        quant.quant_type, quant.double_quant, quant.compute_dtype,
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant.quant_type,
        bnb_4bit_use_double_quant=quant.double_quant,
        bnb_4bit_compute_dtype=_torch_dtype(quant.compute_dtype),
    )


def _build_peft_config(config: ForgeConfig) -> Any:
    from peft import LoraConfig

    lora = config.lora
    assert lora is not None  # guaranteed by ForgeConfig validation

    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        task_type=lora.task_type,
        target_modules=lora.target_modules,
        modules_to_save=lora.modules_to_save,
    )


def load_model(config: ForgeConfig, for_training: bool = True) -> Any:
    """Build the model for the configured method.

    ``for_training=False`` skips PEFT wrapping and gradient plumbing — that is the
    path the baseline evaluation and the latency harness use.
    """
    import torch
    from transformers import AutoModelForCausalLM

    quantization_config = _build_bnb_config(config)

    kwargs: dict[str, Any] = {
        "revision": config.model.revision,
        "trust_remote_code": config.model.trust_remote_code,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        # device_map is required for bitsandbytes; "auto" is right for single-GPU too.
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = _torch_dtype(config.model.dtype)
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"

    if config.model.attn_implementation:
        kwargs["attn_implementation"] = config.model.attn_implementation

    log.info("loading %s (method=%s)", config.model.name, config.training.method)
    model = AutoModelForCausalLM.from_pretrained(config.model.name, **kwargs)

    # Training-time cache is wasted memory, and it is incompatible with
    # gradient checkpointing. Generation re-enables it in the evaluator.
    model.config.use_cache = not for_training

    if not for_training:
        model.eval()
        return model

    if config.training.method == "full":
        _log_parameters(model, "full fine-tuning")
        return model

    # --- PEFT path (lora / qlora) -----------------------------------------
    from peft import get_peft_model, prepare_model_for_kbit_training

    if quantization_config is not None:
        # Casts layer norms to fp32, makes the embedding output require grad, and
        # generally stops 4-bit training from silently producing NaNs.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.training.gradient_checkpointing
        )

    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(model, _build_peft_config(config))
    _log_parameters(model, config.training.method)
    return model


def _log_parameters(model: Any, label: str) -> None:
    counts = count_parameters(model)
    log.info("%s: %s", label, counts)


def count_parameters(model: Any) -> ParameterCount:
    """Count total and trainable parameters, 4-bit-aware.

    bitsandbytes packs two 4-bit values into one uint8 element, so ``numel()``
    reports half the true parameter count for quantized weights. Without the
    ``* 2`` correction a QLoRA run appears to have half as many parameters as the
    identical LoRA run, and the comparison table becomes nonsense.
    """
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        if getattr(param, "element_size", None) and param.__class__.__name__ == "Params4bit":
            count *= 2
        total += count
        if param.requires_grad:
            trainable += count
    return ParameterCount(total=total, trainable=trainable)


def model_size_mb(model: Any) -> float:
    """On-disk-equivalent size of the weights, in MB.

    Measured from live tensors rather than the checkpoint file so it works for a
    merged adapter that has not been saved yet.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return round(total_bytes / 1024**2, 2)


def merge_adapter(model: Any) -> Any:
    """Fold LoRA adapters back into the base weights.

    Worth doing before serving: a merged model has zero adapter overhead at
    inference and is a single standard checkpoint. It cannot be done on a 4-bit
    base — merging into quantized weights is lossy — so QLoRA runs must either
    serve the adapter on top of the quantized base, or dequantize first.
    """
    if not hasattr(model, "merge_and_unload"):
        return model
    log.info("merging LoRA adapters into base weights")
    return model.merge_and_unload()
