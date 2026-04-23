from datetime import UTC, datetime

import pytest

from dltoolbox.dataset.metadata.dataset_metadata import DatasetMetadata


def _build(**overrides) -> DatasetMetadata:
    defaults = dict(name="Test Dataset", split="train", num_samples=4, origin_path="/path/to/dataset/origin/")
    defaults.update(overrides)
    return DatasetMetadata(**defaults)


def test_round_trip_minimal() -> None:
    meta = _build()
    restored = DatasetMetadata.from_json_bytes(meta.to_json_bytes())
    assert meta == restored


def test_round_trip_full_fields() -> None:
    meta = _build(
        version=2,
        description="description text",
        extra_data={"foo": 1, "bar": [1, 2, 3], "nested": {"a": True}},
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        git_hash="deadbeef",
        comment="a comment",
    )
    restored = DatasetMetadata.from_json_bytes(meta.to_json_bytes())
    assert meta == restored


def test_equality_is_by_value() -> None:
    # pin created_at so two independent constructions share every field
    # (the default factory would otherwise return distinct datetime.now values)
    fixed = datetime(2026, 1, 1, tzinfo=UTC)
    assert _build(created_at=fixed) == _build(created_at=fixed)
    assert _build(created_at=fixed) != _build(created_at=fixed, name="other")


@pytest.mark.parametrize(
    ("name", "split", "expected"),
    [
        ("simple", "train", "simple__train"),
        ("with spaces", "test", "with_spaces__test"),
        ("with/slashes\\here", "valid", "withslasheshere__valid"),
        ("café résumé", "train", "cafe_resume__train"),
    ],
    ids=["plain", "spaces", "slashes", "unicode"],
)
def test_filename(name: str, split: str, expected: str) -> None:
    meta = _build(name=name, split=split)
    assert meta.filename == expected
