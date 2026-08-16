"""Shared fixtures.

Everything here is torch-free and offline. A test suite that needs a GPU or a
network call is a test suite nobody runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forgeml.config import ForgeConfig, load_config
from forgeml.data.schema import InstructionRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


@pytest.fixture
def base_config() -> ForgeConfig:
    return load_config(CONFIG_DIR / "base.yaml")


@pytest.fixture
def lora_config() -> ForgeConfig:
    return load_config(CONFIG_DIR / "lora.yaml")


@pytest.fixture
def records() -> list[InstructionRecord]:
    """A small, deliberately messy corpus.

    Contains one duplicate, one empty response, one echo and one too-short
    instruction, so the validation tests have something real to catch.
    """
    return [
        InstructionRecord(
            id="1",
            instruction="Explain gradient descent in simple terms.",
            response="It walks downhill on the loss surface, one small step at a time.",
            category="open_qa",
        ),
        InstructionRecord(
            id="2",
            instruction="Summarise the passage.",
            context="Quantization lowers the numerical precision of weights.",
            response="Quantization reduces weight precision to save memory.",
            category="summarization",
        ),
        InstructionRecord(
            id="3",
            instruction="What is overfitting?",
            response="Memorising the training set instead of learning the pattern.",
            category="open_qa",
        ),
        # duplicate content of record 1, different id
        InstructionRecord(
            id="4",
            instruction="Explain gradient descent in simple terms.",
            response="It walks downhill on the loss surface, one small step at a time.",
            category="open_qa",
        ),
        # empty response
        InstructionRecord(id="5", instruction="Name three optimizers.", response=""),
        # response echoes the instruction
        InstructionRecord(id="6", instruction="Define perplexity.", response="Define perplexity."),
        # instruction below min_instruction_chars
        InstructionRecord(id="7", instruction="Hi", response="Hello there."),
    ]


@pytest.fixture
def many_records() -> list[InstructionRecord]:
    """200 unique records — enough for the split ratios to be meaningful."""
    return [
        InstructionRecord(
            id=f"rec-{i:04d}",
            instruction=f"Explain concept number {i} clearly and precisely.",
            response=f"Concept {i} is best understood as a worked example of idea {i % 7}.",
            category=["open_qa", "summarization", "closed_qa"][i % 3],
        )
        for i in range(200)
    ]
