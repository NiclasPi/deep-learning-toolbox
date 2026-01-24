import pytest
from attrs import frozen

from dltoolbox.dataset.metadata import DatasetMetadata


@frozen(kw_only=True)
class SampleMetadata:
    name: str
    size: int
    confidence: float


@pytest.mark.parametrize(
    "sample_meta",
    ["str", 1, 2.3, ["str", 1], {"field": "value"}, SampleMetadata(name="test", size=42, confidence=0.96)],
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
