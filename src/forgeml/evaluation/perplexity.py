"""Perplexity — the metric that does not depend on a reference answer's wording.

ROUGE punishes a correct paraphrase. Perplexity does not care about wording at
all: it asks "how surprised was the model by the true continuation?". Reporting
both is what stops a single metric from steering the whole project.

::

    perplexity = exp(mean token-level cross-entropy loss)

Two properties worth internalising:

* it is **vocabulary dependent** — you may compare perplexities between two
  fine-tunes of the same base model, never between different tokenizers;
* it is computed here on the **answer tokens only** (prompt masked), matching the
  training objective. Include the prompt and every model looks better than it is,
  because prompts are easy to predict.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from forgeml.data.schema import InstructionRecord
from forgeml.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from forgeml.config import ForgeConfig

log = get_logger(__name__)


@dataclass
class PerplexityResult:
    perplexity: float
    mean_loss: float
    total_tokens: int
    num_examples: int

    def as_metrics(self, prefix: str = "quality") -> dict[str, float]:
        return {
            f"{prefix}/perplexity": round(self.perplexity, 4),
            f"{prefix}/token_loss": round(self.mean_loss, 6),
            f"{prefix}/eval_tokens": float(self.total_tokens),
        }


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    records: Sequence[InstructionRecord],
    config: ForgeConfig,
    batch_size: int = 4,
) -> PerplexityResult:
    """Token-weighted perplexity over the answer spans of ``records``.

    Token-weighted, not example-averaged: perplexity is defined per token, so a
    long answer must count for more than a short one. Averaging per-example
    perplexities is a common and quietly wrong shortcut.
    """
    import torch

    from forgeml.training.dataset import IGNORE_INDEX, SupervisedCollator, build_dataset

    dataset = build_dataset(records, tokenizer, config, mask_prompt=True)
    if len(dataset) == 0:
        return PerplexityResult(float("inf"), float("inf"), 0, 0)

    collator = SupervisedCollator(tokenizer)
    device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            features = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            batch = {k: v.to(device) for k, v in collator(features).items()}

            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits

            # Causal shift: position i predicts token i+1.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()

            # reduction="sum" so we can weight by real token count ourselves;
            # "mean" would average per batch and silently over-weight short rows.
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )

            num_tokens = int((shift_labels != IGNORE_INDEX).sum().item())
            total_loss += float(loss.item())
            total_tokens += num_tokens

    if was_training:
        model.train()

    if total_tokens == 0:
        log.warning("no supervised tokens found — perplexity is undefined")
        return PerplexityResult(float("inf"), float("inf"), 0, len(dataset))

    mean_loss = total_loss / total_tokens
    # exp() overflows to inf for losses above ~709; clamp so the number stays
    # loggable and obviously-bad rather than crashing the run.
    perplexity = float(torch.exp(torch.tensor(min(mean_loss, 20.0))).item())

    log.info(
        "perplexity %.3f (mean token loss %.4f over %d tokens)",
        perplexity,
        mean_loss,
        total_tokens,
    )
    return PerplexityResult(
        perplexity=perplexity,
        mean_loss=mean_loss,
        total_tokens=total_tokens,
        num_examples=len(dataset),
    )
