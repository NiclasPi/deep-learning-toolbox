import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace

import h5py
import pytest

from dltoolbox.dataset.errors import SampleMetaLengthMismatchError
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


def test_get_all_ids_returns_view_order(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    ids = ["01", "02", "03"]
    metas = [Meta(name=f"n{i}", size=i) for i in range(3)]
    _write_meta_datasets(h5, ids, metas)

    store = store_cls(
        decoder=_decode_meta, sample_ids_ds=h5["/metadata/sample_ids"], sample_meta_ds=h5["/metadata/sample_meta"]
    )

    assert list(store.get_all_ids()) == ids


def test_get_all_ids_respects_select_indices(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
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

    assert list(store.get_all_ids()) == [ids[i] for i in selection]


def test_get_all_ids_returns_independent_list(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    """Mutating the returned list must not affect subsequent reads from the store."""
    ids = ["01", "02"]
    metas = [Meta(name="a", size=1), Meta(name="b", size=2)]
    _write_meta_datasets(h5, ids, metas)

    store = store_cls(
        decoder=_decode_meta, sample_ids_ds=h5["/metadata/sample_ids"], sample_meta_ds=h5["/metadata/sample_meta"]
    )

    first = store.get_all_ids()
    first.clear()
    assert store.get_all_ids() == ids


def test_get_all_returns_independent_list(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    """Mutating the returned list must not affect subsequent reads from the store."""
    ids = ["01", "02"]
    metas = [Meta(name="a", size=1), Meta(name="b", size=2)]
    _write_meta_datasets(h5, ids, metas)

    store = store_cls(
        decoder=_decode_meta, sample_ids_ds=h5["/metadata/sample_ids"], sample_meta_ds=h5["/metadata/sample_meta"]
    )

    first = store.get_all()
    first.clear()
    assert store.get_all() == list(zip(ids, metas, strict=True))


def test_get_all_returns_id_item_pairs(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    ids = ["01", "02", "03"]
    metas = [Meta(name=f"n{i}", size=i) for i in range(3)]
    _write_meta_datasets(h5, ids, metas)

    store = store_cls(
        decoder=_decode_meta, sample_ids_ds=h5["/metadata/sample_ids"], sample_meta_ds=h5["/metadata/sample_meta"]
    )

    assert list(store.get_all()) == list(zip(ids, metas, strict=True))


def test_get_all_respects_select_indices(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
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

    assert list(store.get_all()) == [(ids[i], metas[i]) for i in selection]


def test_get_all_applies_resolve(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    ids = ["01", "02"]
    metas = [ResolvableMeta(name=f"n{i}") for i in range(2)]
    _write_meta_datasets(h5, ids, metas)

    header = _header(2)
    decoder: SampleMetaDecoder[ResolvableMeta] = with_resolve(_decode_resolvable, header)
    store = store_cls(
        decoder=decoder, sample_ids_ds=h5["/metadata/sample_ids"], sample_meta_ds=h5["/metadata/sample_meta"]
    )

    pairs = list(store.get_all())
    assert [sid for sid, _ in pairs] == ids
    for sid, item in pairs:
        assert item.id == sid
        assert item.split == header.split


def _write_second_dataset(h5: h5py.File, values: list[str]) -> None:
    h5.create_dataset(
        "/metadata/processing_record", data=[v.encode("utf-8") for v in values], dtype=h5py.string_dtype()
    )


def _decode_pair(raw: bytes | tuple[bytes, ...], sample_id: str) -> tuple[Meta, dict]:
    assert isinstance(raw, tuple)
    meta_raw, record_raw = raw
    return Meta(**json.loads(meta_raw)), json.loads(record_raw)


def test_multiple_keys_decoder_receives_tuple_in_key_order(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    _write_meta_datasets(h5, ["01", "02"], [Meta(name="a", size=1), Meta(name="b", size=2)])
    _write_second_dataset(h5, ['{"step": 1}', '{"step": 2}'])

    store = store_cls(
        decoder=_decode_pair,
        sample_ids_ds=h5["/metadata/sample_ids"],
        sample_meta_ds=[h5["/metadata/sample_meta"], h5["/metadata/processing_record"]],
    )

    assert len(store) == 2
    assert store.get_by_index(1) == (Meta(name="b", size=2), {"step": 2})
    assert store.get_by_id("01") == (Meta(name="a", size=1), {"step": 1})

    # reversed key order is reflected in the tuple order
    seen: list[tuple[bytes, ...]] = []

    def capture(raw: bytes | tuple[bytes, ...], sample_id: str) -> int:
        assert isinstance(raw, tuple)
        seen.append(raw)
        return 0

    store = store_cls(
        decoder=capture,
        sample_ids_ds=h5["/metadata/sample_ids"],
        sample_meta_ds=[h5["/metadata/processing_record"], h5["/metadata/sample_meta"]],
    )
    store.get_by_index(0)
    # eager decodes every row at construction, so the first capture is sample 0 in both modes
    assert seen[0][0] == b'{"step": 1}'
    assert json.loads(seen[0][1]) == {"name": "a", "size": 1}


def test_single_dataset_gets_bytes_and_one_element_sequence_gets_tuple(
    h5: h5py.File, store_cls: type[ISampleMetaStore]
) -> None:
    _write_meta_datasets(h5, ["01"], [Meta(name="a", size=1)])
    received: list[object] = []

    def capture(raw: bytes | tuple[bytes, ...], sample_id: str) -> int:
        received.append(raw)
        return 0

    store_cls(
        decoder=capture, sample_ids_ds=h5["/metadata/sample_ids"], sample_meta_ds=h5["/metadata/sample_meta"]
    ).get_by_index(0)
    store_cls(
        decoder=capture, sample_ids_ds=h5["/metadata/sample_ids"], sample_meta_ds=[h5["/metadata/sample_meta"]]
    ).get_by_index(0)

    assert isinstance(received[0], bytes)
    assert isinstance(received[1], tuple)
    assert len(received[1]) == 1


def test_multiple_keys_select_indices(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    _write_meta_datasets(h5, ["01", "02", "03"], [Meta(name=n, size=i) for i, n in enumerate("abc")])
    _write_second_dataset(h5, [f'{{"step": {i}}}' for i in range(3)])

    store = store_cls(
        decoder=_decode_pair,
        sample_ids_ds=h5["/metadata/sample_ids"],
        sample_meta_ds=[h5["/metadata/sample_meta"], h5["/metadata/processing_record"]],
        select_indices=[2, 0],
    )

    assert store.get_all_ids() == ["03", "01"]
    assert [item[1]["step"] for _, item in store.get_all()] == [2, 0]


def test_multiple_keys_length_mismatch_raises(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    _write_meta_datasets(h5, ["01", "02"], [Meta(name="a", size=1), Meta(name="b", size=2)])
    _write_second_dataset(h5, ['{"step": 1}'])

    with pytest.raises(SampleMetaLengthMismatchError):
        store_cls(
            decoder=_decode_pair,
            sample_ids_ds=h5["/metadata/sample_ids"],
            sample_meta_ds=[h5["/metadata/sample_meta"], h5["/metadata/processing_record"]],
        )


def test_multiple_keys_with_resolve(h5: h5py.File, store_cls: type[ISampleMetaStore]) -> None:
    _write_meta_datasets(h5, ["01"], [ResolvableMeta(name="a")])
    _write_second_dataset(h5, ['{"extra": 1}'])

    def decode(raw: bytes | tuple[bytes, ...], sample_id: str) -> ResolvableMeta:
        assert isinstance(raw, tuple)
        return ResolvableMeta(**json.loads(raw[0]))

    store = store_cls(
        decoder=with_resolve(decode, _header(1)),
        sample_ids_ds=h5["/metadata/sample_ids"],
        sample_meta_ds=[h5["/metadata/sample_meta"], h5["/metadata/processing_record"]],
    )

    assert store.get_by_index(0) == ResolvableMeta(name="a", id="01", split="train")
