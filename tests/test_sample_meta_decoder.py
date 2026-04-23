from dataclasses import dataclass

import pytest
from attrs import evolve, frozen

from dltoolbox.dataset.metadata.dataset_metadata import DatasetMetadata
from dltoolbox.dataset.metadata.sample_meta_decoder import SampleMetaDecoder


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


def _header(num_samples: int = 1) -> DatasetMetadata:
    return DatasetMetadata(name="Test", split="train", num_samples=num_samples, origin_path="/path/")


@pytest.mark.parametrize(
    "value",
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
def test_decode_round_trips_various_types(value) -> None:
    from cattrs.preconf.json import make_converter

    converter = make_converter()
    raw = converter.dumps(value).encode("utf-8")

    decoder = SampleMetaDecoder(sample_meta_type=type(value), header=_header())
    assert decoder.decode(raw, sample_id="01") == value


def test_decode_applies_resolve_protocol() -> None:
    @frozen(kw_only=True)
    class ResolvableSampleMetadata:
        id: str | None = None
        name: str
        split: str | None = None

        def resolve(self, sample_id: str, dataset_metadata: DatasetMetadata) -> "ResolvableSampleMetadata":
            return evolve(self, id=sample_id, split=dataset_metadata.split)

    from cattrs.preconf.json import make_converter

    converter = make_converter()
    original = ResolvableSampleMetadata(name="test")
    raw = converter.dumps(original).encode("utf-8")

    header = _header()
    decoder = SampleMetaDecoder(sample_meta_type=ResolvableSampleMetadata, header=header)
    resolved = decoder.decode(raw, sample_id="01")
    assert resolved.id == "01"
    assert resolved.split == header.split


def test_decode_structures_unregistered_sample_meta_type() -> None:
    """The decoder relies on cattrs' native generic dispatch, no hook pre-registration."""

    @frozen(kw_only=True)
    class UnseenSampleMetadata:
        name: str

    raw = b'{"name": "Item 01"}'
    decoder = SampleMetaDecoder(sample_meta_type=UnseenSampleMetadata, header=_header())
    result = decoder.decode(raw, sample_id="01")
    assert isinstance(result, UnseenSampleMetadata)
    assert result.name == "Item 01"
