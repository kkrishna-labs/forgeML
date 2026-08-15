"""Data validation as a first-class, reportable step.

The report is an MLflow artifact, not a print statement. When a run's quality
drops three weeks from now, "what changed in the data?" is answerable by diffing
two of these JSON files.

Each rule is a named predicate so the report can say *which* rule dropped how many
rows, instead of the useless "cleaned: 4,812 -> 4,510".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable

from forgeml.data.schema import InstructionRecord
from forgeml.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from forgeml.config import DataConfig

log = get_logger(__name__)

Rule = Callable[[InstructionRecord], bool]


@dataclass
class ValidationReport:
    """What came in, what went out, and exactly why."""

    total_input: int = 0
    total_kept: int = 0
    dropped_by_rule: Counter = field(default_factory=Counter)
    duplicates_removed: int = 0
    category_counts: Counter = field(default_factory=Counter)
    examples_dropped: list[dict[str, str]] = field(default_factory=list)

    @property
    def total_dropped(self) -> int:
        return self.total_input - self.total_kept

    @property
    def keep_rate(self) -> float:
        return self.total_kept / self.total_input if self.total_input else 0.0

    @property
    def is_healthy(self) -> bool:
        """A keep rate under 70% almost always means a broken rule, not dirty data."""
        return self.keep_rate >= 0.70 and self.total_kept > 0

    def to_dict(self) -> dict:
        return {
            "total_input": self.total_input,
            "total_kept": self.total_kept,
            "total_dropped": self.total_dropped,
            "keep_rate": round(self.keep_rate, 4),
            "duplicates_removed": self.duplicates_removed,
            "dropped_by_rule": dict(self.dropped_by_rule),
            "category_counts": dict(self.category_counts),
            "is_healthy": self.is_healthy,
            "examples_dropped": self.examples_dropped[:20],
        }

    def summary(self) -> str:
        lines = [
            f"input           : {self.total_input}",
            f"kept            : {self.total_kept}  ({self.keep_rate:.1%})",
            f"dropped         : {self.total_dropped}",
            f"duplicates      : {self.duplicates_removed}",
        ]
        for rule, count in self.dropped_by_rule.most_common():
            lines.append(f"  - {rule:<28} {count}")
        return "\n".join(lines)


def _build_rules(config: DataConfig) -> dict[str, Rule]:
    """Name -> predicate. A record is kept only if every predicate returns True."""
    rules: dict[str, Rule] = {
        "instruction_missing": lambda r: bool(r.instruction.strip()),
        "instruction_too_short": lambda r: len(r.instruction) >= config.min_instruction_chars,
        "instruction_too_long": lambda r: len(r.instruction) <= config.max_instruction_chars,
        "response_too_long": lambda r: len(r.response) <= config.max_response_chars,
        "context_too_long": lambda r: len(r.context) <= config.max_context_chars,
        # A response that merely echoes the instruction teaches the model to parrot.
        "response_echoes_instruction": lambda r: (
            r.response.strip().lower() != r.instruction.strip().lower()
        ),
        # Control characters break tokenizers in ways that are painful to debug.
        "contains_control_chars": lambda r: not _has_control_chars(r),
    }

    if config.drop_empty_response:
        rules["response_missing"] = lambda r: bool(r.response.strip())
        rules["response_too_short"] = lambda r: len(r.response) >= config.min_response_chars

    return rules


def _has_control_chars(record: InstructionRecord) -> bool:
    blob = f"{record.instruction}{record.context}{record.response}"
    return any(ord(ch) < 32 and ch not in "\n\t" for ch in blob)


def validate_records(
    records: Iterable[InstructionRecord],
    config: DataConfig,
) -> tuple[list[InstructionRecord], ValidationReport]:
    """Apply every rule, then dedupe. Returns the survivors plus the report."""
    rules = _build_rules(config)
    report = ValidationReport()
    kept: list[InstructionRecord] = []

    for record in records:
        report.total_input += 1
        failed = [name for name, rule in rules.items() if not rule(record)]
        if failed:
            # Attribute the drop to the FIRST failing rule so the counts sum to
            # total_dropped instead of double-counting messy records.
            report.dropped_by_rule[failed[0]] += 1
            if len(report.examples_dropped) < 20:
                report.examples_dropped.append(
                    {
                        "id": record.id,
                        "rule": failed[0],
                        "instruction": record.instruction[:160],
                    }
                )
            continue
        kept.append(record)

    if config.dedupe:
        kept, removed = _dedupe(kept)
        report.duplicates_removed = removed
        if removed:
            report.dropped_by_rule["duplicate"] += removed

    # A duplicated id with distinct content is a loader bug, not dirty data —
    # fail loudly rather than silently training on a broken index.
    _assert_unique_ids(kept)

    report.total_kept = len(kept)
    report.category_counts = Counter(r.category or "uncategorized" for r in kept)

    log.info("validation: kept %d / %d (%.1f%%)", report.total_kept,
             report.total_input, report.keep_rate * 100)
    if not report.is_healthy:
        log.warning(
            "keep rate %.1f%% is suspiciously low — check the rules before training",
            report.keep_rate * 100,
        )
    return kept, report


def _dedupe(records: list[InstructionRecord]) -> tuple[list[InstructionRecord], int]:
    """Exact-content dedupe, keeping the first occurrence.

    Near-duplicate detection (MinHash / embeddings) is deliberately out of scope:
    it is slow, it needs tuning, and exact dedupe already removes the overwhelming
    majority of leakage in public instruction sets.
    """
    seen: set[str] = set()
    unique: list[InstructionRecord] = []
    for record in records:
        digest = record.content_hash()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(record)
    return unique, len(records) - len(unique)


def _assert_unique_ids(records: list[InstructionRecord]) -> None:
    counts = Counter(r.id for r in records)
    collisions = [rid for rid, n in counts.items() if n > 1]
    if collisions:
        raise ValueError(
            f"{len(collisions)} duplicate record id(s) after cleaning, e.g. "
            f"{collisions[:5]} — the loader is generating unstable ids"
        )


def clean_records(
    records: Iterable[InstructionRecord],
    config: DataConfig,
) -> tuple[list[InstructionRecord], ValidationReport]:
    """Alias kept for readability at call sites in notebooks."""
    return validate_records(records, config)
