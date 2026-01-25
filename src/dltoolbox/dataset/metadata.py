from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Protocol, Type, TypeGuard, TypeVar, runtime_checkable

from attrs import field, frozen
from cattrs.preconf.json import make_converter

T = TypeVar("T")
_CONVERTER = make_converter()


@runtime_checkable
class ResolvableSampleMeta(Protocol[T]):
    def resolve(self, sample_id: str, dataset_metadata: DatasetMetadata[T]) -> T: ...


def _is_resolvable(obj) -> TypeGuard[ResolvableSampleMeta[T]]:
    try:
        if isinstance(obj, ResolvableSampleMeta):
            return True
    except Exception:
        pass

    return hasattr(obj, "resolve") and callable(obj.resolve)


@frozen(kw_only=True)
class DatasetMetadata[T]:
    _version: int = 1

    name: str
    split: Literal["train", "valid", "test"]
    description: str = ""

    origin_path: str
    sample_ids: list[str]
    sample_meta: dict[str, T] = field(factory=dict)

    created_at: datetime = field(factory=lambda: datetime.now(UTC))
    git_hash: str | None = None

    comment: str | None = None

    @staticmethod
    def from_json_bytes(json_bytes: bytes, sample_meta_type: Type[T]) -> DatasetMetadata[T]:
        """Parse DatasetMetadata from JSON bytes."""
        return _CONVERTER.loads(json_bytes, DatasetMetadata[sample_meta_type])

    def to_json_bytes(self) -> bytes:
        """Return JSON representation as bytes."""
        return _CONVERTER.dumps(_CONVERTER.unstructure(self)).encode("utf-8")

    @property
    def num_samples(self) -> int:
        return len(self.sample_ids)

    def get_sample_meta_by_id(self, sample_id: str) -> T | None:
        if sample_id in self.sample_meta:
            sample_meta = self.sample_meta[sample_id]
            if sample_meta is not None:
                if _is_resolvable(sample_meta):
                    return sample_meta.resolve(sample_id, self)
                return sample_meta
        return None

    def get_sample_meta_by_index(self, index: int) -> T | None:
        sample_id = self.sample_ids[index]
        return self.get_sample_meta_by_id(sample_id)
