"""Prompt formatting.

One rule holds the whole project together: **training and evaluation must format
prompts identically**. If they drift, your evaluation measures the formatting gap
rather than the fine-tuning, and every number downstream is a lie. So all prompt
construction goes through this module and nowhere else.
"""

from __future__ import annotations

from typing import Any, Protocol

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's instruction accurately and "
    "concisely. If context is provided, base your answer on it."
)

# Fallback used when the tokenizer has no chat template of its own.
_PLAIN_TEMPLATE_WITH_CONTEXT = (
    "### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
)
_PLAIN_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


class _Tokenizer(Protocol):
    """The slice of a HF tokenizer we depend on (kept structural to avoid the import)."""

    chat_template: Any

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str: ...


def build_user_message(instruction: str, context: str | None = None) -> str:
    """Merge the instruction and its optional context into one user turn."""
    instruction = (instruction or "").strip()
    context = (context or "").strip()
    if context:
        return f"{instruction}\n\nContext:\n{context}"
    return instruction


def format_prompt(
    instruction: str,
    context: str | None = None,
    tokenizer: _Tokenizer | None = None,
    system_prompt: str | None = SYSTEM_PROMPT,
) -> str:
    """Render the model-ready prompt string, *without* the answer.

    Uses the tokenizer's own chat template when it has one — that is the format the
    instruct checkpoint was trained on, and matching it is worth several points of
    quality for free. Falls back to a plain Alpaca-style block otherwise.
    """
    user = build_user_message(instruction, context)

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    if (context or "").strip():
        return _PLAIN_TEMPLATE_WITH_CONTEXT.format(
            instruction=(instruction or "").strip(), context=context.strip()
        )
    return _PLAIN_TEMPLATE.format(instruction=(instruction or "").strip())


def format_training_example(
    instruction: str,
    response: str,
    context: str | None = None,
    tokenizer: _Tokenizer | None = None,
    system_prompt: str | None = SYSTEM_PROMPT,
    eos_token: str = "",
) -> str:
    """Prompt + target, i.e. one full row of the training corpus.

    The EOS token matters more than it looks: without it the model never learns to
    stop, and every generation runs to ``max_new_tokens``. That single omission will
    quietly ruin your latency numbers.
    """
    prompt = format_prompt(instruction, context, tokenizer, system_prompt)
    return f"{prompt}{(response or '').strip()}{eos_token}"


def prompt_and_completion(
    record: dict[str, Any],
    tokenizer: _Tokenizer | None = None,
) -> tuple[str, str]:
    """Split a canonical record into ``(prompt, reference_answer)`` for evaluation."""
    prompt = format_prompt(
        record.get("instruction", ""), record.get("context"), tokenizer=tokenizer
    )
    return prompt, (record.get("response") or "").strip()
