"""Data layer: normalization, validation, splitting and fingerprinting.

These are the cheapest tests in the project and they protect the most expensive
mistake — discovering after an hour of GPU time that the test set leaked.
"""

from __future__ import annotations

import pytest

from forgeml.config import ForgeConfig
from forgeml.data.loader import read_jsonl, write_jsonl
from forgeml.data.schema import InstructionRecord, clean_text, normalize_record
from forgeml.data.splitter import split_records
from forgeml.data.validator import validate_records
from forgeml.data.versioning import fingerprint_records, hash_records

# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_normalize_maps_alpaca_field_names() -> None:
    record = normalize_record(
        {"instruction": "Do the thing", "input": "some context", "output": "done"}, 0
    )
    assert record.instruction == "Do the thing"
    assert record.context == "some context"
    assert record.response == "done"


def test_normalize_maps_qa_field_names() -> None:
    record = normalize_record({"question": "Why?", "answer": "Because."}, 0)
    assert record.instruction == "Why?"
    assert record.response == "Because."


def test_normalize_generates_stable_ids() -> None:
    """Same content, same id — otherwise re-running the loader renumbers everything."""
    row = {"instruction": "Explain X", "output": "X is a thing"}
    assert normalize_record(row, 0, "src").id == normalize_record(row, 0, "src").id


def test_unknown_columns_are_kept_as_meta() -> None:
    record = normalize_record(
        {"instruction": "a", "output": "b", "source_url": "http://x", "score": 5}, 0
    )
    assert record.meta == {"source_url": "http://x", "score": 5}


def test_clean_text_collapses_whitespace_but_keeps_paragraphs() -> None:
    assert clean_text("a    b\t\tc") == "a b c"
    assert clean_text("para1\n\n\n\n\npara2") == "para1\n\npara2"
    assert clean_text(None) == ""


def test_content_hash_ignores_id_and_case() -> None:
    a = InstructionRecord(id="1", instruction="Hello  World", response="Hi")
    b = InstructionRecord(id="2", instruction="hello world", response="hi")
    assert a.content_hash() == b.content_hash()


def test_content_hash_separates_fields() -> None:
    """Field boundaries must be real, or 'ab'+'c' and 'a'+'bc' would collide."""
    a = InstructionRecord(id="1", instruction="ab", response="c")
    b = InstructionRecord(id="2", instruction="a", response="bc")
    assert a.content_hash() != b.content_hash()


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_validation_drops_each_bad_record_for_the_right_reason(
    records: list[InstructionRecord], base_config: ForgeConfig
) -> None:
    kept, report = validate_records(records, base_config.data)

    assert report.total_input == 7
    assert len(kept) == 3
    assert report.duplicates_removed == 1
    assert report.dropped_by_rule["response_missing"] == 1
    assert report.dropped_by_rule["response_echoes_instruction"] == 1
    assert report.dropped_by_rule["instruction_too_short"] == 1


def test_dropped_counts_sum_to_total_dropped(
    records: list[InstructionRecord], base_config: ForgeConfig
) -> None:
    """Each record is attributed to exactly one rule, so the books must balance."""
    _, report = validate_records(records, base_config.data)
    assert sum(report.dropped_by_rule.values()) == report.total_dropped


def test_dedupe_can_be_disabled(
    records: list[InstructionRecord], base_config: ForgeConfig
) -> None:
    base_config.data.dedupe = False
    kept, report = validate_records(records, base_config.data)
    assert report.duplicates_removed == 0
    assert len(kept) == 4


def test_report_flags_an_unhealthy_keep_rate(base_config: ForgeConfig) -> None:
    bad = [InstructionRecord(id=str(i), instruction="x", response="y") for i in range(10)]
    _, report = validate_records(bad, base_config.data)
    assert not report.is_healthy


def test_control_characters_are_rejected(base_config: ForgeConfig) -> None:
    record = InstructionRecord(
        id="1", instruction="Explain this properly", response="ok\x07bell"
    )
    kept, report = validate_records([record], base_config.data)
    assert kept == []
    assert report.dropped_by_rule["contains_control_chars"] == 1


def test_report_serialises(records: list[InstructionRecord], base_config: ForgeConfig) -> None:
    _, report = validate_records(records, base_config.data)
    payload = report.to_dict()
    assert payload["total_input"] == 7
    assert 0.0 <= payload["keep_rate"] <= 1.0
    assert isinstance(payload["dropped_by_rule"], dict)


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------


def test_splits_are_disjoint_and_complete(
    many_records: list[InstructionRecord], base_config: ForgeConfig
) -> None:
    result = split_records(many_records, base_config.data)
    assert result.total == len(many_records)
    result.assert_disjoint()  # raises if any content leaks


def test_split_is_deterministic(
    many_records: list[InstructionRecord], base_config: ForgeConfig
) -> None:
    first = split_records(many_records, base_config.data)
    second = split_records(many_records, base_config.data)
    assert [r.id for r in first.train] == [r.id for r in second.train]


def test_split_is_order_independent(
    many_records: list[InstructionRecord], base_config: ForgeConfig
) -> None:
    """Shuffling the input must not move a single example between splits."""
    shuffled = list(reversed(many_records))
    a = split_records(many_records, base_config.data)
    b = split_records(shuffled, base_config.data)
    assert {r.id for r in a.test} == {r.id for r in b.test}


def test_growing_the_dataset_does_not_reshuffle_existing_examples(
    many_records: list[InstructionRecord], base_config: ForgeConfig
) -> None:
    """The property that makes dataset v1 vs v2 an honest comparison.

    With random shuffling, adding rows moves old examples across the train/test
    boundary and silently contaminates the comparison. Hash bucketing does not.
    """
    original = split_records(many_records, base_config.data)
    original_test_ids = {r.id for r in original.test}

    grown = many_records + [
        InstructionRecord(
            id=f"new-{i}", instruction=f"A brand new question {i}?", response=f"Answer {i}."
        )
        for i in range(200)
    ]
    after = split_records(grown, base_config.data)
    after_test_ids = {r.id for r in after.test}

    # every originally-held-out example is still held out
    assert original_test_ids <= after_test_ids


def test_split_ratios_are_approximately_honoured(
    many_records: list[InstructionRecord], base_config: ForgeConfig
) -> None:
    result = split_records(many_records, base_config.data)
    assert result.sizes["train"] / result.total == pytest.approx(0.80, abs=0.08)


def test_changing_the_dataset_version_reshuffles(
    many_records: list[InstructionRecord], base_config: ForgeConfig
) -> None:
    """A version bump is an explicit invitation to re-split."""
    v1 = split_records(many_records, base_config.data)
    base_config.data.version = "v2.0"
    v2 = split_records(many_records, base_config.data)
    assert {r.id for r in v1.test} != {r.id for r in v2.test}


# ---------------------------------------------------------------------------
# versioning
# ---------------------------------------------------------------------------


def test_fingerprint_is_order_independent(many_records: list[InstructionRecord]) -> None:
    assert hash_records(many_records) == hash_records(list(reversed(many_records)))


def test_fingerprint_changes_when_content_changes(
    many_records: list[InstructionRecord],
) -> None:
    modified = [*many_records[:-1], InstructionRecord(id="x", instruction="new", response="new")]
    assert hash_records(many_records) != hash_records(modified)


def test_fingerprint_records_captures_splits(
    many_records: list[InstructionRecord], base_config: ForgeConfig
) -> None:
    splits = split_records(many_records, base_config.data)
    fingerprint = fingerprint_records(many_records, base_config.data, splits)

    assert fingerprint.num_examples == 200
    assert set(fingerprint.split_hashes) == {"train", "validation", "test"}
    assert fingerprint.stats["instruction_chars_mean"] > 0
    assert fingerprint.mlflow_params()["dataset.version"] == "v1.0"


def test_fingerprints_compare(many_records: list[InstructionRecord], base_config: ForgeConfig) -> None:
    a = fingerprint_records(many_records, base_config.data)
    b = fingerprint_records(list(reversed(many_records)), base_config.data)
    assert a.matches(b)


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


def test_jsonl_round_trip(many_records: list[InstructionRecord], tmp_path) -> None:
    path = write_jsonl(many_records, tmp_path / "out.jsonl")
    reloaded = read_jsonl(path)
    assert len(reloaded) == len(many_records)
    assert reloaded[0].content_hash() == many_records[0].content_hash()
