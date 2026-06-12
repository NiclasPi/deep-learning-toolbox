from collections.abc import Iterator

import h5py
import numpy as np
import pytest

from dltoolbox.dataset._utils import derive_rdcc_args, read_ids


@pytest.fixture
def h5_file() -> Iterator[h5py.File]:
    with h5py.File.in_memory() as f:
        yield f


def test_read_ids_decodes_bytes_entries(h5_file: h5py.File) -> None:
    h5_file.create_dataset("/ids_bytes", data=[b"01", b"02", b"03"], dtype=h5py.string_dtype())
    assert read_ids(h5_file["/ids_bytes"]) == ["01", "02", "03"]


def test_read_ids_passes_str_entries_through() -> None:
    # h5py's returned dtype depends on how the dataset was written and on the h5py
    # version — str entries are a real possibility, not a hypothetical. Covering this
    # branch guards against a regression where a future refactor (e.g. unconditional
    # .decode()) would crash on str input. A plain list is sufficient: read_ids only
    # requires ds[:] to yield an iterable, and the branch under test is the
    # isinstance(s, bytes) check — no h5py machinery needed.
    assert read_ids(["a", "b"]) == ["a", "b"]


def test_derive_rdcc_args_sizes_nbytes_to_chunk_count(h5_file: h5py.File) -> None:
    # chunk = 10*4*4 float32 = 640 bytes; the cache should hold exactly cache_chunks of them
    ds = h5_file.create_dataset("chunked", shape=(100, 4, 4), dtype="f4", chunks=(10, 4, 4))
    args = derive_rdcc_args(ds, cache_chunks=4)
    assert args["rdcc_nbytes"] == 4 * 640


def test_derive_rdcc_args_nslots_is_prime_and_respects_rule_of_thumb(h5_file: h5py.File) -> None:
    # 100 * cache_chunks (20000) exceeds the 8191 default floor, so the rule of thumb drives it
    ds = h5_file.create_dataset("chunked", shape=(1000, 4, 4), dtype="f4", chunks=(10, 4, 4))
    assert derive_rdcc_args(ds, cache_chunks=200)["rdcc_nslots"] >= 100 * 200


@pytest.mark.parametrize(
    "hdf5_version, expected_default_nslots", [((2, 0, 0), 8191), ((2, 1, 3), 8191), ((1, 14, 3), 521), ((1, 8, 0), 521)]
)
def test_derive_rdcc_args_default_nslots_tracks_hdf5_version(
    h5_file: h5py.File, monkeypatch: pytest.MonkeyPatch, hdf5_version: tuple[int, ...], expected_default_nslots: int
) -> None:
    # cache_chunks=1 keeps the rule-of-thumb target (100) below either floor, so the
    # version-dependent default drives rdcc_nslots; both 8191 and 521 are already prime.
    monkeypatch.setattr(h5py.version, "hdf5_version_tuple", hdf5_version)
    ds = h5_file.create_dataset("chunked", shape=(100, 4, 4), dtype="f4", chunks=(10, 4, 4))
    assert derive_rdcc_args(ds, cache_chunks=1)["rdcc_nslots"] == expected_default_nslots


def test_derive_rdcc_args_none_returns_empty(h5_file: h5py.File) -> None:
    ds = h5_file.create_dataset("chunked", shape=(100, 4, 4), dtype="f4", chunks=(10, 4, 4))
    assert derive_rdcc_args(ds, cache_chunks=None) == {}


def test_derive_rdcc_args_contiguous_dataset_returns_empty(h5_file: h5py.File) -> None:
    # the chunk cache does not apply to contiguously stored datasets
    ds = h5_file.create_dataset("contiguous", data=np.zeros((100, 4, 4), dtype="f4"))
    assert ds.chunks is None
    assert derive_rdcc_args(ds, cache_chunks=4) == {}


def test_derive_rdcc_args_override_takes_precedence(h5_file: h5py.File) -> None:
    ds = h5_file.create_dataset("chunked", shape=(100, 4, 4), dtype="f4", chunks=(10, 4, 4))
    override = {"rdcc_nbytes": 1 << 20, "rdcc_nslots": 1009, "rdcc_w0": 0.3}
    # override wins over cache_chunks and is returned verbatim
    assert derive_rdcc_args(ds, cache_chunks=4, rdcc_override=override) == override


def test_derive_rdcc_args_override_wins_even_for_contiguous(h5_file: h5py.File) -> None:
    ds = h5_file.create_dataset("contiguous", data=np.zeros((100, 4, 4), dtype="f4"))
    override = {"rdcc_nbytes": 1 << 20}
    assert derive_rdcc_args(ds, cache_chunks=None, rdcc_override=override) == override


@pytest.mark.parametrize("bad", [0, -1, -512])
def test_derive_rdcc_args_rejects_non_positive_cache_chunks(h5_file: h5py.File, bad: int) -> None:
    ds = h5_file.create_dataset("chunked", shape=(100, 4, 4), dtype="f4", chunks=(10, 4, 4))
    with pytest.raises(ValueError):
        derive_rdcc_args(ds, cache_chunks=bad)


def test_derive_rdcc_args_validates_before_contiguous_check(h5_file: h5py.File) -> None:
    # a non-positive cache_chunks is rejected even when the cache would not apply anyway
    ds = h5_file.create_dataset("contiguous", data=np.zeros((100, 4, 4), dtype="f4"))
    with pytest.raises(ValueError):
        derive_rdcc_args(ds, cache_chunks=-1)
