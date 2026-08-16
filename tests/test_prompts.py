"""Prompt formatting.

Small module, high blast radius: if training and evaluation render prompts even
slightly differently, every metric in the project silently measures the gap
between the two formats instead of the quality of the model.
"""

from __future__ import annotations

from typing import Any

from forgeml.prompts import (
    SYSTEM_PROMPT,
    build_user_message,
    format_prompt,
    format_training_example,
    prompt_and_completion,
)


class FakeChatTokenizer:
    """Minimal stand-in for a tokenizer that has a chat template."""

    chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str:
        rendered = "".join(f"<|{m['role']}|>{m['content']}" for m in conversation)
        if kwargs.get("add_generation_prompt"):
            rendered += "<|assistant|>"
        return rendered


def test_context_is_merged_into_the_user_turn() -> None:
    message = build_user_message("Summarise this.", "Some passage.")
    assert "Summarise this." in message
    assert "Some passage." in message


def test_no_context_leaves_the_instruction_alone() -> None:
    assert build_user_message("Do a thing", None) == "Do a thing"
    assert build_user_message("Do a thing", "   ") == "Do a thing"


def test_plain_fallback_used_without_a_chat_template() -> None:
    prompt = format_prompt("Explain X")
    assert "### Instruction:" in prompt
    assert "### Response:" in prompt
    assert "### Context:" not in prompt


def test_plain_fallback_includes_context_when_present() -> None:
    assert "### Context:" in format_prompt("Explain X", "background")


def test_chat_template_is_preferred_when_available() -> None:
    """Matching the format the instruct checkpoint was trained on is free quality."""
    prompt = format_prompt("Explain X", tokenizer=FakeChatTokenizer())
    assert prompt.startswith(f"<|system|>{SYSTEM_PROMPT}")
    assert prompt.endswith("<|assistant|>")
    assert "### Instruction:" not in prompt


def test_training_example_appends_the_answer_and_eos() -> None:
    """Without EOS the model never learns to stop and latency becomes meaningless."""
    text = format_training_example("Q?", "A.", eos_token="</s>")
    assert text.endswith("A.</s>")


def test_prompt_is_a_strict_prefix_of_the_training_example() -> None:
    """The invariant that makes completion-only loss masking correct."""
    record = {"instruction": "Explain X", "context": "ctx", "response": "Because Y."}
    prompt = format_prompt(record["instruction"], record["context"])
    full = format_training_example(record["instruction"], record["response"], record["context"])
    assert full.startswith(prompt)


def test_prompt_and_completion_splits_a_record() -> None:
    prompt, answer = prompt_and_completion(
        {"instruction": "Explain X", "context": "", "response": "  Because Y.  "}
    )
    assert "Explain X" in prompt
    assert answer == "Because Y."


def test_formatting_is_deterministic() -> None:
    assert format_prompt("Explain X", "ctx") == format_prompt("Explain X", "ctx")


def test_whitespace_is_stripped_consistently() -> None:
    assert format_prompt("  Explain X  ") == format_prompt("Explain X")
