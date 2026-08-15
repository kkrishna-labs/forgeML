"""Dataset loading, validation, splitting and versioning.

Nothing in this subpackage imports torch. That is on purpose: data bugs are the
most common cause of a wasted GPU hour, so this is the layer that must be fully
unit-tested on a laptop before any training starts.
"""

from __future__ import annotations

from forgeml.data.loader import load_raw_records, records_to_frame, write_jsonl
from forgeml.data.schema import CANONICAL_FIELDS, InstructionRecord, normalize_record
from forgeml.data.splitter import SplitResult, split_records
from forgeml.data.validator import ValidationReport, clean_records, validate_records
from forgeml.data.versioning import DatasetFingerprint, fingerprint_records

__all__ = [
    "CANONICAL_FIELDS",
    "DatasetFingerprint",
    "InstructionRecord",
    "SplitResult",
    "ValidationReport",
    "clean_records",
    "fingerprint_records",
    "load_raw_records",
    "normalize_record",
    "records_to_frame",
    "split_records",
    "validate_records",
    "write_jsonl",
]
