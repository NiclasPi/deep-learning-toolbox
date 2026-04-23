from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from typing import Literal, Protocol, TypeGuard, TypeVar, runtime_checkable

from attrs import field, frozen
from cattrs import Converter
from cattrs.preconf.json import make_converter

T = TypeVar("T")

_default_converter = make_converter()


@runtime_checkable
class ResolvableSampleMeta(Protocol[T]):
    def resolve(self, sample_id: str, dataset_metadata: DatasetMetadata[T]) -> T: ...


@frozen(kw_only=True)
class DatasetMetadata[T]:
    version: int = 1

    name: str
    split: Literal["train", "valid", "test"]
    description: str = ""

    origin_path: str
    sample_ids: list[str]
    sample_meta: dict[str, T] = field(factory=dict)

    created_at: datetime = field(factory=lambda: datetime.now(UTC))
    git_hash: str | None = None

    comment: str | None = None

    def __attrs_post_init__(self) -> None:
        unknown = set(self.sample_meta).difference(self.sample_ids)
        if unknown:
            raise ValueError(
                f"sample_meta contains {len(unknown)} id(s) not present in sample_ids: "
                f"{sorted(unknown)[:5]}"
            )

    @classmethod
    def from_json_bytes(
            cls, json_bytes: bytes, sample_meta_type: type[T], converter: Converter | None = None
    ) -> DatasetMetadata[T]:
        """Parse DatasetMetadata from JSON bytes."""

        if converter is None:
            converter = _default_converter
        return converter.loads(json_bytes, DatasetMetadata[sample_meta_type])

    def to_json_bytes(self, converter: Converter | None = None) -> bytes:
        """Return JSON representation as bytes."""

        if converter is None:
            converter = _default_converter
        return converter.dumps(converter.unstructure(self)).encode("utf-8")

    @property
    def filename(self) -> str:
        """Derive a safe filename from the dataset's name.

        The returned value is not the original source filename but a filesystem-safe identifier
        derived from `self.name`. Can be used to create new files that are derived from this dataset.
        """
        # turn accents into ASCII equivalents and drop non-ASCII
        s = unicodedata.normalize("NFKD", self.name).encode("ascii", "ignore").decode("ascii")
        # replace spaces with underscore
        s = s.replace(" ", "_")
        # remove any slashes (forward or back)
        s = re.sub(r"[\\/]", "", s)
        # add the dataset split indicator
        s += "__" + self.split
        return s

    @property
    def num_samples(self) -> int:
        return len(self.sample_ids)

    @staticmethod
    def _is_resolvable(obj: T) -> TypeGuard[ResolvableSampleMeta[T]]:
        return isinstance(obj, ResolvableSampleMeta)

    def get_sample_meta_by_id(self, sample_id: str) -> T | None:
        if sample_id in self.sample_meta:
            sample_meta = self.sample_meta[sample_id]
            if sample_meta is not None:
                if self._is_resolvable(sample_meta):
                    return sample_meta.resolve(sample_id, self)
                return sample_meta
        return None

    def get_sample_meta_by_index(self, index: int) -> T | None:
        sample_id = self.sample_ids[index]
        return self.get_sample_meta_by_id(sample_id)
