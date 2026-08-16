"""Train / validation / test splitting.

Uses **hash bucketing**, not ``random.shuffle``. The difference matters:

* random shuffle — grow the dataset from 5k to 15k rows and every example lands in
  a different split. Your test set is now polluted with rows the previous model
  trained on, and the two runs are no longer comparable.
* hash bucketing — an example's split is a pure function of its content hash. Add
  10k rows and the original 5k stay exactly where they were.

That property is what makes "dataset v1.0 vs v2.0" an honest comparison.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from forgeml.data.schema import InstructionRecord
from forgeml.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from forgeml.config import DataConfig

log = get_logger(__name__)

_BUCKETS = 10_000


@dataclass
class SplitResult:
    """The three splits plus enough metadata to prove they are disjoint."""

    train: list[InstructionRecord]
    validation: list[InstructionRecord]
    test: list[InstructionRecord]

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    @property
    def total(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)

    def as_dict(self) -> dict[str, list[InstructionRecord]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}

    def assert_disjoint(self) -> None:
        """Leakage check. Cheap to run, catastrophic to skip."""
        train_hashes = {r.content_hash() for r in self.train}
        for name, split in (("validation", self.validation), ("test", self.test)):
            overlap = train_hashes & {r.content_hash() for r in split}
            if overlap:
                raise ValueError(
                    f"{len(overlap)} example(s) leak between train and {name} — "
                    "evaluation results would be meaningless"
                )

    def category_distribution(self) -> dict[str, dict[str, int]]:
        return {
            name: dict(Counter(r.category or "uncategorized" for r in split))
            for name, split in self.as_dict().items()
        }


def _bucket_of(record: InstructionRecord, salt: str) -> int:
    """Deterministic bucket in ``[0, _BUCKETS)``.

    The salt is the dataset version: bumping the version reshuffles on purpose,
    which is what you want when the *task* changes rather than the row count.
    """
    payload = f"{salt}|{record.content_hash()}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16) % _BUCKETS


def split_records(
    records: Sequence[InstructionRecord],
    config: DataConfig,
) -> SplitResult:
    """Split deterministically by content hash, then verify disjointness."""
    train_cut = int(config.train_ratio * _BUCKETS)
    val_cut = train_cut + int(config.val_ratio * _BUCKETS)
    salt = f"{config.version}:{config.seed}"

    train: list[InstructionRecord] = []
    validation: list[InstructionRecord] = []
    test: list[InstructionRecord] = []

    for record in records:
        bucket = _bucket_of(record, salt)
        if bucket < train_cut:
            train.append(record)
        elif bucket < val_cut:
            validation.append(record)
        else:
            test.append(record)

    result = SplitResult(train=train, validation=validation, test=test)
    result.assert_disjoint()

    log.info(
        "split (%s): train=%d validation=%d test=%d",
        salt,
        len(train),
        len(validation),
        len(test),
    )
    if not validation or not test:
        log.warning(
            "an eval split is empty — with %d records the ratios are too fine-grained",
            len(records),
        )
    return result
