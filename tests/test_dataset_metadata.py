from dataclasses import dataclass

import pytest
from attrs import evolve, frozen

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

    def resolve(self, sample_id: str, dataset_metadata: DatasetMetadata) -> "ResolvableSampleMetadata":
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
