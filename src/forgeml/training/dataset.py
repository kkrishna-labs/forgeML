"""Tokenization, loss masking and batching.

The important idea here is **completion-only loss masking**.

A naive supervised fine-tune computes loss over the entire sequence, so the model
spends most of its gradient budget learning to predict the prompt — text that is
always given to it at inference and never needs predicting. Masking the prompt
tokens to ``-100`` (which PyTorch's cross-entropy ignores) concentrates the whole
gradient signal on the answer.

On small models and small datasets this is not a rounding error; it is often the
difference between a fine-tune that helps and one that does nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from forgeml.data.schema import InstructionRecord
from forgeml.logging_utils import get_logger
from forgeml.prompts import format_prompt

if TYPE_CHECKING:  # pragma: no cover
    from forgeml.config import ForgeConfig

log = get_logger(__name__)

IGNORE_INDEX = -100


@dataclass
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]

    @property
    def length(self) -> int:
        return len(self.input_ids)

    @property
    def num_supervised_tokens(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_INDEX)


def tokenize_example(
    record: InstructionRecord,
    tokenizer: Any,
    max_seq_length: int,
    mask_prompt: bool = True,
) -> TokenizedExample:
    """Tokenize one record into ``input_ids`` plus prompt-masked ``labels``.

    The prompt and the answer are tokenized separately so the boundary is exact.
    Tokenizing the joined string and then trying to locate the split by decoding
    is fragile — tokenizers merge across the boundary and the mask ends up
    one or two tokens off, which silently corrupts the loss.
    """
    prompt_text = format_prompt(record.instruction, record.context, tokenizer=tokenizer)
    answer_text = record.response.strip()

    prompt_ids: list[int] = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids: list[int] = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    # Teach the model to stop. Without EOS every generation runs to
    # max_new_tokens and your measured latency is meaningless.
    if tokenizer.eos_token_id is not None:
        answer_ids = [*answer_ids, tokenizer.eos_token_id]

    input_ids = prompt_ids + answer_ids

    # -100 is what torch's cross-entropy ignores, so masked positions contribute
    # no gradient and the answer tokens carry the entire learning signal.
    labels = list(input_ids)
    if mask_prompt:
        labels[: len(prompt_ids)] = [IGNORE_INDEX] * len(prompt_ids)

    # Truncate from the LEFT of the prompt, never the right of the answer: losing
    # the end of the target is far worse than losing the start of the context.
    if len(input_ids) > max_seq_length:
        overflow = len(input_ids) - max_seq_length
        keep_prompt = max(len(prompt_ids) - overflow, 0)
        input_ids = prompt_ids[len(prompt_ids) - keep_prompt :] + answer_ids
        labels = labels[len(labels) - len(input_ids) :]
        input_ids = input_ids[-max_seq_length:]
        labels = labels[-max_seq_length:]

    return TokenizedExample(input_ids=input_ids, labels=labels)


class SupervisedDataset:
    """A plain map-style dataset of tokenized examples.

    Deliberately not ``datasets.Dataset``: the corpus here is a few thousand short
    rows, it fits in RAM comfortably, and an in-memory list removes an Arrow
    dependency plus a whole class of caching surprises during debugging.
    """

    def __init__(
        self,
        records: Sequence[InstructionRecord],
        tokenizer: Any,
        max_seq_length: int,
        mask_prompt: bool = True,
    ) -> None:
        self.examples: list[TokenizedExample] = []
        dropped = 0

        for record in records:
            example = tokenize_example(record, tokenizer, max_seq_length, mask_prompt)
            # An example whose answer was entirely truncated away contributes no
            # gradient and would divide-by-zero the token accounting.
            if example.num_supervised_tokens == 0:
                dropped += 1
                continue
            self.examples.append(example)

        if dropped:
            log.warning(
                "%d/%d examples had no supervised tokens after truncation and were "
                "dropped — max_seq_length=%d may be too small",
                dropped,
                len(records),
                max_seq_length,
            )

        self._log_stats()

    def _log_stats(self) -> None:
        if not self.examples:
            log.warning("tokenized dataset is empty")
            return
        lengths = sorted(e.length for e in self.examples)
        supervised = sum(e.num_supervised_tokens for e in self.examples)
        total = sum(lengths)
        log.info(
            "tokenized %d examples | len mean=%.0f p95=%d max=%d | supervised tokens "
            "%.1f%% of total",
            len(lengths),
            total / len(lengths),
            lengths[int(0.95 * (len(lengths) - 1))],
            lengths[-1],
            100.0 * supervised / total if total else 0.0,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        return {"input_ids": example.input_ids, "labels": example.labels}

    @property
    def total_tokens(self) -> int:
        return sum(e.length for e in self.examples)

    @property
    def supervised_tokens(self) -> int:
        return sum(e.num_supervised_tokens for e in self.examples)


@dataclass
class SupervisedCollator:
    """Pad a batch to its own longest sequence.

    Dynamic padding rather than padding everything to ``max_seq_length``: with a
    corpus whose p95 length is a quarter of the cap, this alone can cut step time
    by more than half. ``pad_to_multiple_of=8`` keeps tensor cores happy.
    """

    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        pad_id = self.tokenizer.pad_token_id
        longest = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of > 1:
            remainder = longest % self.pad_to_multiple_of
            if remainder:
                longest += self.pad_to_multiple_of - remainder

        input_ids, labels, attention_mask = [], [], []
        for feature in features:
            ids = feature["input_ids"]
            lab = feature["labels"]
            pad_len = longest - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            # Padding must be IGNORE_INDEX, not pad_id: otherwise the model is
            # trained to emit padding tokens.
            labels.append(lab + [IGNORE_INDEX] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def build_dataset(
    records: Sequence[InstructionRecord],
    tokenizer: Any,
    config: ForgeConfig,
    mask_prompt: bool = True,
) -> SupervisedDataset:
    return SupervisedDataset(
        records=records,
        tokenizer=tokenizer,
        max_seq_length=config.model.max_seq_length,
        mask_prompt=mask_prompt,
    )
