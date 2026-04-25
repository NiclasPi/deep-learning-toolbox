from __future__ import annotations

import h5py

from dltoolbox.dataset.errors import H5DatasetMissingKeyError


def read_ids(ds: h5py.Dataset) -> list[str]:
    return [s.decode("utf-8") if isinstance(s, bytes) else s for s in ds[:]]


def require_key(h5_file: h5py.File, key: str) -> None:
    if key not in h5_file:
        raise H5DatasetMissingKeyError(key)
