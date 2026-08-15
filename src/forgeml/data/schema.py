"""The canonical record shape.

Every dataset ForgeML touches — Dolly, Alpaca, a JSONL file you exported by hand —
is normalized into this one shape at the door. Downstream code then never has to
ask "which dataset is this?", and swapping datasets becomes a config change
instead of a refactor.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from pydantic import BaseModel, Field

CANONICAL_FIELDS = ("id", "instruction", "context", "response", "category")

# Field aliases seen across the common public instruction datasets.
_INSTRUCTION_KEYS = ("instruction", "prompt", "question", "input_text", "query")
_CONTEXT_KEYS = ("context", "input", "passage", "document", "source")
_RESPONSE_KEYS = ("response", "output", "answer", "completion", "target", "text_output")
_CATEGORY_KEYS = ("category", "task", "type", "domain")

_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class InstructionRecord(BaseModel):
    """One supervised instruction-tuning example."""

    id: str
    instruction: str
    context: str = ""
    response: str
    category: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)

    def content_hash(self) -> str:
        """Stable hash of the semantic content — the basis for dedupe and versioning.

        ``id`` is deliberately excluded: two records with different ids but identical
        text are duplicates, and that is precisely what we want to catch.
        """
        payload = "\x1f".join(
            [
                _canonical_text(self.instruction),
                _canonical_text(self.context),
                _canonical_text(self.response),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def total_chars(self) -> int:
        return len(self.instruction) + len(self.context) + len(self.response)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def _canonical_text(value: str) -> str:
    """Lowercase + collapse whitespace. Used for hashing only, never for training."""
    return _WHITESPACE.sub(" ", (value or "").strip().lower())


def clean_text(value: Any) -> str:
    """Light normalization applied to text we DO train on.

    Deliberately conservative — collapse runs of spaces/tabs and 3+ newlines, strip
    the ends, and nothing more. Aggressive cleaning (unicode folding, punctuation
    stripping) destroys signal the model needs.
    """
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def normalize_record(row: dict[str, Any], index: int, source: str = "") -> InstructionRecord:
    """Map an arbitrary source row onto :class:`InstructionRecord`.

    ``index`` is only a fallback for the id — a deterministic content-derived id is
    preferred so that re-running the loader does not renumber everything.
    """
    instruction = clean_text(_first_present(row, _INSTRUCTION_KEYS))
    context = clean_text(_first_present(row, _CONTEXT_KEYS))
    response = clean_text(_first_present(row, _RESPONSE_KEYS))
    category = clean_text(_first_present(row, _CATEGORY_KEYS))

    raw_id = row.get("id")
    if raw_id in (None, ""):
        seed = f"{source}|{_canonical_text(instruction)}|{_canonical_text(response)}"
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        raw_id = f"{digest}-{index:06d}"

    known = set(_INSTRUCTION_KEYS + _CONTEXT_KEYS + _RESPONSE_KEYS + _CATEGORY_KEYS) | {"id"}
    meta = {k: v for k, v in row.items() if k not in known}

    return InstructionRecord(
        id=str(raw_id),
        instruction=instruction,
        context=context,
        response=response,
        category=category,
        meta=meta,
    )
