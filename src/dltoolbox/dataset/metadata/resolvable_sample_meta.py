from typing import Protocol, TypeGuard, TypeVar, runtime_checkable

from dltoolbox.dataset.metadata.dataset_metadata import DatasetMetadata

T = TypeVar("T")


@runtime_checkable
class ResolvableSampleMeta(Protocol[T]):
    def resolve(self, sample_id: str, dataset_metadata: DatasetMetadata) -> T: ...


def is_resolvable(obj: T) -> TypeGuard[ResolvableSampleMeta[T]]:
    return isinstance(obj, ResolvableSampleMeta)
