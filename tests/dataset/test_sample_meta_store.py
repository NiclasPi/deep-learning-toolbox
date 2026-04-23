import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace

import h5py
import pytest

from dltoolbox.dataset.errors import SampleMetaLengthMismatchError
from dltoolbox.dataset.metadata._utils import read_ids
from dltoolbox.dataset.metadata.dataset_metadata import DatasetMetadata
from dltoolbox.dataset.metadata.eager_sample_meta_store import EagerSampleMetaStore
from dltoolbox.dataset.metadata.isample_meta_store import ISampleMetaStore
from dltoolbox.dataset.metadata.lazy_sample_meta_store import LazySampleMetaStore
from dltoolbox.dataset.metadata.sample_meta_protocols import SampleMetaDecoder, with_resolve


@dataclass(frozen=True, kw_only=True)
class Meta:
    name: str
    size: int


@dataclass(frozen=True, kw_only=True)
class ResolvableMeta:
    name: str
    id: str | None = None
    split: str | None = None

    def resolve(self, sample_id: str, dataset_metadata: DatasetMetadata) -> "ResolvableMeta":
        return replace(self, id=sample_id, split=dataset_metadata.split)


def _header(num_samples: int) -> DatasetMetadata:
    return DatasetMetadata(name="Test", split="train", num_samples=num_samples, origin_path="/path/")


def _decode_meta(raw: bytes, sample_id: str) -> Meta:
    return Meta(**json.loads(raw))


def _decode_resolvable(raw: bytes, sample_id: str) -> ResolvableMeta:
    return ResolvableMeta(**json.loads(raw))


@pytest.fixture
def h5() -> Iterator[h5py.File]:
    with h5py.File("mem.h5", "w", driver="core", backing_store=False) as f:
        yield f


def _write_meta_datasets(h5: h5py.File, sample_ids: list[str], meta_objs: list) -> None:
    h5.create_dataset("/metadata/sample_ids", data=sample_ids, dtype=h5py.string_dtype())
    h5.create_dataset(
        "/metadata/sample_meta",
        data=[json.dumps(asdict(m)).encode("utf-8") for m in meta_objs],
        dtype=h5py.string_dtype(),
    )


@pytest.fixture(params=[EagerSampleMetaStore, LazySampleMetaStore], ids=["eager", "lazy"])
def store_cls(request) -> type[ISampleMetaStore]:
    return request.param


def test_round_trip_and_lookup(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    ids = ["01", "02", "03"]
    metas = [Meta(name=f"n{i}", size=i) for i in range(3)]
    _write_meta_datasets(h5, ids, metas)

    store = store_cls(
        decoder=_decode_meta, sample_ids_ds=h5["/metadata/sample_ids"], sample_meta_ds=h5["/metadata/sample_meta"]
    )

    assert len(store) == 3
    for i, (sid, expected) in enumerate(zip(ids, metas)):
        assert store.get_by_index(i) == expected
        assert store.get_by_id(sid) == expected


def test_resolve_dispatch(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    ids = ["01", "02"]
    metas = [ResolvableMeta(name=f"n{i}") for i in range(2)]
    _write_meta_datasets(h5, ids, metas)

    header = _header(2)
    decoder: SampleMetaDecoder[ResolvableMeta] = with_resolve(_decode_resolvable, header)
    store = store_cls(
        decoder=decoder, sample_ids_ds=h5["/metadata/sample_ids"], sample_meta_ds=h5["/metadata/sample_meta"]
    )

    item = store.get_by_index(0)
    assert item.id == "01"
    assert item.split == header.split

    item = store.get_by_id("02")
    assert item.id == "02"
    assert item.split == header.split


def test_raises_on_missing_index(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    _write_meta_datasets(h5, ["01"], [Meta(name="a", size=1)])
    store = store_cls(
        decoder=_decode_meta, sample_ids_ds=h5["/metadata/sample_ids"], sample_meta_ds=h5["/metadata/sample_meta"]
    )
    with pytest.raises(IndexError):
        store.get_by_index(5)


def test_raises_on_missing_id(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    _write_meta_datasets(h5, ["01"], [Meta(name="a", size=1)])
    store = store_cls(
        decoder=_decode_meta, sample_ids_ds=h5["/metadata/sample_ids"], sample_meta_ds=h5["/metadata/sample_meta"]
    )
    with pytest.raises(KeyError):
        store.get_by_id("unknown")


def test_select_indices_exposes_view(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    ids = ["01", "02", "03", "04", "05"]
    metas = [Meta(name=f"n{i}", size=i) for i in range(5)]
    _write_meta_datasets(h5, ids, metas)

    selection = [4, 0, 2]
    store = store_cls(
        decoder=_decode_meta,
        sample_ids_ds=h5["/metadata/sample_ids"],
        sample_meta_ds=h5["/metadata/sample_meta"],
        select_indices=selection,
    )

    assert len(store) == len(selection)
    for ext, orig in enumerate(selection):
        assert store.get_by_index(ext) == metas[orig]
        assert store.get_by_id(ids[orig]) == metas[orig]

    # ids outside the selection are not addressable
    with pytest.raises(KeyError):
        store.get_by_id("02")  # index 1 was not selected
    with pytest.raises(IndexError):
        store.get_by_index(len(selection))


def test_length_mismatch_raises(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    h5.create_dataset("/metadata/sample_ids", data=["01", "02"], dtype=h5py.string_dtype())
    h5.create_dataset("/metadata/sample_meta", data=['{"name":"a","size":1}'], dtype=h5py.string_dtype())

    with pytest.raises(SampleMetaLengthMismatchError):
        store_cls(
            decoder=_decode_meta, sample_ids_ds=h5["/metadata/sample_ids"], sample_meta_ds=h5["/metadata/sample_meta"]
        )


def test_decoder_receives_sample_id(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    """The decoder protocol surfaces sample_id so callers can stamp identity without closure tricks."""

    ids = ["alpha", "beta"]
    metas = [Meta(name="a", size=1), Meta(name="b", size=2)]
    _write_meta_datasets(h5, ids, metas)

    seen: list[str] = []

    def capture(raw: bytes, sample_id: str) -> Meta:
        seen.append(sample_id)
        return Meta(**json.loads(raw))

    store = store_cls(
        decoder=capture, sample_ids_ds=h5["/metadata/sample_ids"], sample_meta_ds=h5["/metadata/sample_meta"]
    )
    # force decode for lazy
    store.get_by_index(0)
    store.get_by_index(1)
    assert set(seen) == set(ids)


def test_read_ids_decodes_bytes_entries(h5: h5py.File) -> None:
    h5.create_dataset("/ids_bytes", data=[b"01", b"02", b"03"], dtype=h5py.string_dtype())
    assert read_ids(h5["/ids_bytes"]) == ["01", "02", "03"]


def test_read_ids_passes_str_entries_through() -> None:
    # h5py's returned dtype depends on how the dataset was written and on the h5py
    # version — str entries are a real possibility, not a hypothetical. Covering this
    # branch guards against a regression where a future refactor (e.g. unconditional
    # .decode()) would crash on str input. A plain list is sufficient: read_ids only
    # requires ds[:] to yield an iterable, and the branch under test is the
    # isinstance(s, bytes) check — no h5py machinery needed.
    assert read_ids(["a", "b"]) == ["a", "b"]
