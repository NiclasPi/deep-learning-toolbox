from typing import Protocol, TypeGuard, TypeVar, runtime_checkable

from dltoolbox.dataset.metadata.dataset_metadata import DatasetMetadata


@runtime_checkable
class SampleMetaDecoder[T](Protocol):
    """Decodes one sample's stored bytes back into T.

    `sample_id` is passed through so decoders can stamp identity onto the
    returned object, or close over dataset-level context (e.g. `DatasetMetadata`)
    to enrich it — without the library imposing a specific structuring library.
    """

    def __call__(self, raw: bytes, sample_id: str) -> T: ...


@runtime_checkable
class SampleMetaEncoder[T](Protocol):
    """Encodes one sample's in-memory T into UTF-8 JSON bytes for storage.

    The library ships a dict-based default in `create_hdf5_file`; callers with
    typed payloads (e.g. dataclasses) supply their own.
    """

    def __call__(self, obj: T) -> bytes: ...


T = TypeVar("T")


@runtime_checkable
class ResolvableSampleMeta(Protocol[T]):
    """Opt-in protocol: a sample meta type that can enrich itself from the dataset header.

    When the decoder produces an object implementing `resolve`, `with_resolve` dispatches
    to it automatically so callers don't have to thread the header through their own
    decoder. Implementations are expected to return a new instance rather than mutate.
    """

    def resolve(self, sample_id: str, dataset_metadata: DatasetMetadata) -> T: ...


def is_resolvable(obj: T) -> TypeGuard[ResolvableSampleMeta[T]]:
    return isinstance(obj, ResolvableSampleMeta)


def with_resolve[T](inner: SampleMetaDecoder[T], header: DatasetMetadata) -> SampleMetaDecoder[T]:
    """Wrap a decoder so objects implementing `ResolvableSampleMeta` are resolved automatically.

    The header is bound at wrap time, so per-call overhead is one `isinstance` check plus
    the resolve call when applicable — no extra header lookups in `__getitem__`.
    """

    def decode(raw: bytes, sample_id: str) -> T:
        obj = inner(raw, sample_id)
        if is_resolvable(obj):
            return obj.resolve(sample_id, header)
        return obj

    return decode
