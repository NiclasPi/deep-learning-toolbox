from __future__ import annotations

from cattrs import Converter

from dltoolbox.dataset.metadata.dataset_metadata import DatasetMetadata, _default_converter
from dltoolbox.dataset.metadata.resolvable_sample_meta import is_resolvable


class SampleMetaDecoder[T]:
    """Decodes a single sample's JSON bytes into T, applying the resolve protocol if present."""

    def __init__(
        self, *, sample_meta_type: type[T], header: DatasetMetadata, converter: Converter | None = None
    ) -> None:
        self._sample_meta_type = sample_meta_type
        self._header = header
        self._converter = converter if converter is not None else _default_converter

    def decode(self, raw: bytes | str, sample_id: str) -> T:
        obj = self._converter.loads(raw, self._sample_meta_type)
        if is_resolvable(obj):
            return obj.resolve(sample_id, self._header)
        return obj
