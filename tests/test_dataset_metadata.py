import json
from dataclasses import dataclass

import pytest
from attrs import evolve, frozen
from cattrs.preconf.json import make_converter

from dltoolbox.dataset.metadata import DatasetMetadata


@frozen(kw_only=True)
class SampleMetadata:
    name: str
    size: int
    confidence: float


@dataclass(kw_only=True)
class SampleMetadataDataclass:
    name: str
    size: int
    confidence: float


@pytest.mark.parametrize(
    "sample_meta",
    [
        "str",
        1,
        2.3,
        ["str", 1],
        {"field": "value"},
        SampleMetadata(name="test", size=42, confidence=0.96),
        SampleMetadataDataclass(name="test", size=42, confidence=0.96),
    ],
    ids=["str", "int", "float", "list", "dict", "attrs", "dataclass"],
)
def test_metadata_serialization(sample_meta) -> None:
    meta = DatasetMetadata(
        name="Test Dataset",
        split="train",
        origin_path="/path/to/dataset/origin/",
        sample_ids=["01", "02", "03", "04"],
        sample_meta={"01": sample_meta},
    )

    meta_serialized = meta.to_json_bytes()
    meta_structured = DatasetMetadata.from_json_bytes(meta_serialized, type(sample_meta))
    assert meta == meta_structured


@frozen(kw_only=True)
class ResolvableSampleMetadata:
    id: str | None = None
    name: str
    size: int
    confidence: float
    split: str | None = None

    def resolve(
        self, sample_id: str, dataset_metadata: "DatasetMetadata[ResolvableSampleMetadata]"
    ) -> "ResolvableSampleMetadata":
        return evolve(self, id=sample_id, split=dataset_metadata.split)


def test_resolvable_sample_meta() -> None:
    meta: DatasetMetadata[ResolvableSampleMetadata] = DatasetMetadata(
        name="Test Dataset",
        split="train",
        origin_path="/path/to/dataset/origin/",
        sample_ids=["01"],
        sample_meta={"01": ResolvableSampleMetadata(name="test", size=42, confidence=0.96)},
    )

    meta_serialized = meta.to_json_bytes()
    meta_structured = DatasetMetadata.from_json_bytes(meta_serialized, ResolvableSampleMetadata)
    assert meta == meta_structured

    sample_meta = meta.get_sample_meta_by_id("01")
    assert sample_meta.id == "01"
    assert sample_meta.split == "train"


def test_from_json_bytes_structures_unregistered_sample_meta_type() -> None:
    """Deserialization must work for a sample_meta type the converter has never seen.

    Guards the reliance on cattrs' native generic dispatch: no hook is pre-registered for
    ``UnseenSampleMetadata`` (it's defined locally here), yet ``from_json_bytes`` is
    expected to structure the nested dict into instances of it.
    """

    @frozen(kw_only=True)
    class UnseenSampleMetadata:
        name: str

    meta_serialized = json.dumps(
        {
            "name": "Test Dataset",
            "split": "train",
            "origin_path": "/path/to/dataset/origin/",
            "sample_ids": ["01"],
            "sample_meta": {"01": {"name": "Item 01"}},
        }
    ).encode("utf-8")
    meta_structured = DatasetMetadata.from_json_bytes(meta_serialized, UnseenSampleMetadata)
    assert isinstance(meta_structured, DatasetMetadata)
    assert isinstance(meta_structured.sample_meta["01"], UnseenSampleMetadata)


def test_custom_converter_is_used() -> None:
    converter = make_converter()
    converter.register_unstructure_hook(str, lambda s: f"<<{s}>>")

    meta = DatasetMetadata[str](
        name="custom",
        split="train",
        origin_path="/",
        sample_ids=["01"],
        sample_meta={"01": "meta"},
    )
    serialized = meta.to_json_bytes(converter=converter)
    assert b"<<custom>>" in serialized
    assert b"<<meta>>" in serialized


def test_sample_meta_keys_must_be_subset_of_sample_ids() -> None:
    with pytest.raises(ValueError, match="sample_meta contains"):
        DatasetMetadata(
            name="Test Dataset",
            split="train",
            origin_path="/path/",
            sample_ids=["01", "02"],
            sample_meta={"03": "orphan"},
        )


def test_get_sample_meta_by_id_returns_none_for_missing() -> None:
    meta = DatasetMetadata[str](
        name="Test Dataset",
        split="train",
        origin_path="/path/",
        sample_ids=["01", "02"],
        sample_meta={"01": "meta-a"},
    )
    assert meta.get_sample_meta_by_id("unknown") is None
    assert meta.get_sample_meta_by_id("02") is None
    assert meta.get_sample_meta_by_id("01") == "meta-a"


def test_get_sample_meta_by_index() -> None:
    meta = DatasetMetadata[str](
        name="Test Dataset",
        split="train",
        origin_path="/path/",
        sample_ids=["01", "02"],
        sample_meta={"01": "meta-a"},
    )
    assert meta.get_sample_meta_by_index(0) == "meta-a"
    assert meta.get_sample_meta_by_index(1) is None
    with pytest.raises(IndexError):
        meta.get_sample_meta_by_index(5)


def test_num_samples() -> None:
    meta = DatasetMetadata[str](
        name="Test Dataset",
        split="train",
        origin_path="/path/",
        sample_ids=["01", "02", "03"],
    )
    assert meta.num_samples == 3


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
    meta = DatasetMetadata[str](
        name=name,
        split=split,  # type: ignore[arg-type]
        origin_path="/",
        sample_ids=[],
    )
    assert meta.filename == expected
