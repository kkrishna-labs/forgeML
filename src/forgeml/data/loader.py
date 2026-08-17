"""Read raw instruction data from Hugging Face, JSONL or a Delta table.

Whatever the source, the output is always ``list[InstructionRecord]``.
``datasets`` and ``pyspark`` are imported lazily so that the module remains
importable — and testable — in an environment that has neither.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forgeml.data.schema import InstructionRecord, normalize_record
from forgeml.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from forgeml.config import DataConfig

log = get_logger(__name__)


def load_raw_records(config: DataConfig, spark: Any | None = None) -> list[InstructionRecord]:
    """Load and normalize the configured dataset.

    ``max_examples`` is applied *before* cleaning, deliberately: it makes a small
    run a genuine subsample of the pipeline rather than "the first N survivors",
    so smoke-test behaviour predicts full-run behaviour.
    """
    source_type = config.source_type

    if source_type == "hf":
        rows = _load_from_huggingface(config)
    elif source_type == "jsonl":
        rows = _load_from_jsonl(config.source)
    elif source_type == "delta":
        rows = _load_from_delta(config, spark)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown source_type: {source_type}")

    if config.max_examples is not None:
        rows = rows[: config.max_examples]

    records = [normalize_record(row, i, source=config.source) for i, row in enumerate(rows)]
    log.info("loaded %d raw records from %s (%s)", len(records), config.source, source_type)
    return records


def _load_from_huggingface(config: DataConfig) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        # Deliberately the editable form: forgeml is not published to PyPI, so
        # `pip install 'forgeml[train]'` fails with an unhelpful resolver error.
        raise ImportError(
            "source_type='hf' needs the datasets library. From the repo root:\n"
            '    uv pip install -e ".[train]"'
        ) from exc

    dataset = load_dataset(config.source, split=config.split)

    # Shuffle before truncating, otherwise `max_examples` slices off whichever
    # category happens to sit at the top of the file and the subsample is biased.
    dataset = dataset.shuffle(seed=config.seed)
    if config.max_examples is not None:
        dataset = dataset.select(range(min(config.max_examples, len(dataset))))

    return [dict(row) for row in dataset]


def _load_from_jsonl(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"jsonl source not found: {file_path}")

    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_no} is not valid JSON") from exc
    return rows


def _load_from_delta(config: DataConfig, spark: Any | None) -> list[dict[str, Any]]:
    if spark is None:
        raise ValueError(
            "source_type='delta' needs an active SparkSession — pass spark=spark "
            "from inside a Databricks notebook"
        )
    table = config.delta.fqn("bronze", config.delta.table_raw)
    log.info("reading delta table %s", table)
    return [row.asDict(recursive=True) for row in spark.read.table(table).collect()]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def records_to_frame(records: Iterable[InstructionRecord]) -> pd.DataFrame:
    """Records -> pandas. Used for Delta writes and for the evaluation tables."""
    import pandas as pd

    rows = []
    for record in records:
        row = record.model_dump()
        # Spark cannot infer a schema for a free-form dict column, so meta is
        # JSON-encoded on the way out.
        row["meta"] = json.dumps(row.get("meta") or {}, ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)


def write_jsonl(records: Iterable[InstructionRecord], path: str | Path) -> Path:
    """Persist records as JSONL — the local counterpart of a Delta table."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
            count += 1
    log.info("wrote %d records -> %s", count, target)
    return target


def read_jsonl(path: str | Path) -> list[InstructionRecord]:
    """Read back what :func:`write_jsonl` produced."""
    rows = _load_from_jsonl(str(path))
    return [InstructionRecord.model_validate(row) for row in rows]
