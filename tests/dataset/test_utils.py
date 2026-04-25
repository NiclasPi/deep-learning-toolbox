from collections.abc import Iterator

import h5py
import pytest

from dltoolbox.dataset._utils import read_ids


@pytest.fixture
def h5() -> Iterator[h5py.File]:
    with h5py.File("mem.h5", "w", driver="core", backing_store=False) as f:
        yield f


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
