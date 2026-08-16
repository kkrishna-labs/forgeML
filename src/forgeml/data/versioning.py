"""Dataset fingerprinting — the thing that turns "I trained on Dolly" into a fact.

An MLflow run that records ``data.version = v1.0`` proves nothing; a version string
is just a label a human typed. A *content hash* over every example is verifiable:
re-run the loader, recompute the hash, compare. If it matches, you are looking at
the same data. If it does not, the run is not reproducible and you now know it.

Cheap to compute, and it closes the biggest hole in most portfolio ML projects.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from forgeml.data.schema import InstructionRecord

if TYPE_CHECKING:  # pragma: no cover
    from forgeml.config import DataConfig
    from forgeml.data.splitter import SplitResult


@dataclass
class DatasetFingerprint:
    """An immutable identity card for one materialized dataset version."""

    version: str
    source: str
    num_examples: int
    content_hash: str
    split_hashes: dict[str, str] = field(default_factory=dict)
    split_sizes: dict[str, int] = field(default_factory=dict)
    created_at: str = ""
    stats: dict[str, float] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "num_examples": self.num_examples,
            "content_hash": self.content_hash,
            "split_hashes": self.split_hashes,
            "split_sizes": self.split_sizes,
            "created_at": self.created_at,
            "stats": self.stats,
            "environment": self.environment,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    def mlflow_params(self) -> dict[str, str]:
        """The subset worth pinning as MLflow *params* (searchable, immutable)."""
        return {
            "dataset.version": self.version,
            "dataset.source": self.source,
            "dataset.num_examples": str(self.num_examples),
            "dataset.content_hash": self.content_hash[:16],
        }

    def matches(self, other: DatasetFingerprint) -> bool:
        return self.content_hash == other.content_hash


def hash_records(records: Sequence[InstructionRecord]) -> str:
    """Order-independent hash over a collection of records.

    Sorting the per-record hashes before combining them means a shuffled dataset
    fingerprints identically — correct, because shuffling changes nothing about
    *what* the model sees over an epoch.
    """
    digests = sorted(record.content_hash() for record in records)
    combined = hashlib.sha256()
    for digest in digests:
        combined.update(digest.encode("ascii"))
    return combined.hexdigest()


def _length_stats(records: Sequence[InstructionRecord]) -> dict[str, float]:
    """Character-length statistics. A proxy for token length that needs no tokenizer.

    Worth logging: if p95 instruction length exceeds your ``max_seq_length``, you are
    silently truncating a chunk of the corpus and wondering why quality plateaus.
    """
    if not records:
        return {}

    instruction_lengths = sorted(len(r.instruction) for r in records)
    response_lengths = sorted(len(r.response) for r in records)
    total_lengths = sorted(r.total_chars for r in records)

    def pct(values: list[int], q: float) -> float:
        if not values:
            return 0.0
        idx = min(int(q * (len(values) - 1)), len(values) - 1)
        return float(values[idx])

    return {
        "instruction_chars_mean": round(sum(instruction_lengths) / len(instruction_lengths), 2),
        "instruction_chars_p95": pct(instruction_lengths, 0.95),
        "response_chars_mean": round(sum(response_lengths) / len(response_lengths), 2),
        "response_chars_p95": pct(response_lengths, 0.95),
        "total_chars_p95": pct(total_lengths, 0.95),
        "total_chars_max": float(total_lengths[-1]),
        "pct_with_context": round(100.0 * sum(1 for r in records if r.context) / len(records), 2),
    }


def fingerprint_records(
    records: Sequence[InstructionRecord],
    config: DataConfig,
    splits: SplitResult | None = None,
) -> DatasetFingerprint:
    """Build the fingerprint for a materialized dataset."""
    fingerprint = DatasetFingerprint(
        version=config.version,
        source=config.source,
        num_examples=len(records),
        content_hash=hash_records(records),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        stats=_length_stats(records),
        environment={
            "python": platform.python_version(),
            "platform": platform.system(),
        },
    )

    if splits is not None:
        fingerprint.split_hashes = {
            name: hash_records(split) for name, split in splits.as_dict().items()
        }
        fingerprint.split_sizes = splits.sizes

    return fingerprint


def load_fingerprint(path: str | Path) -> DatasetFingerprint:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetFingerprint(**payload)
