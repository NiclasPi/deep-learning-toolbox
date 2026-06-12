from __future__ import annotations

import h5py
import numpy as np

from dltoolbox.dataset.errors import H5DatasetMissingKeyError


def read_ids(ds: h5py.Dataset) -> list[str]:
    return [s.decode("utf-8") if isinstance(s, bytes) else s for s in ds[:]]


def require_key(h5_file: h5py.File, key: str) -> None:
    if key not in h5_file:
        raise H5DatasetMissingKeyError(key)


def derive_rdcc_args(
    h5_dataset: h5py.Dataset, cache_chunks: int | None, rdcc_override: dict[str, int | float] | None = None
) -> dict[str, int | float]:
    """Resolve HDF5 raw-data chunk-cache (rdcc_*) args for opening a file.

    rdcc_* must be set at file-open time but depend on the dataset's on-disk chunk
    geometry, so this takes an already-open dataset. Precedence is centralized here so
    every call site behaves identically: ``rdcc_override`` wins outright; otherwise, when
    ``cache_chunks`` is given, rdcc_nbytes is sized to exactly that many chunks and
    rdcc_nslots follows h5py's rule of thumb (~100x the cached chunk count, rounded up to
    a prime, never below the linked HDF5's default of 8191 since 2.0 / 521 before).

    Returns an empty mapping (h5py defaults) when nothing is requested or the dataset is
    contiguous, where the chunk cache does not apply.
    """
    if rdcc_override is not None:
        return rdcc_override
    if cache_chunks is None:
        return {}
    if cache_chunks <= 0:
        raise ValueError(f"cache_chunks must be a positive integer, got {cache_chunks}")
    if h5_dataset.chunks is None:
        return {}

    def _next_prime(n: int) -> int:
        """Smallest prime >= n; h5py recommends a prime number of chunk-cache slots."""

        def is_prime(x: int) -> bool:
            if x < 2:
                return False
            if x % 2 == 0:
                return x == 2
            return all(x % i for i in range(3, int(x**0.5) + 1, 2))

        while not is_prime(n):
            n += 1
        return n

    chunk_nbytes = int(np.prod(h5_dataset.chunks)) * h5_dataset.dtype.itemsize
    # h5py's default rdcc_nslots is 8191 since HDF5 2.0 and 521 before
    default_nslots = 8191 if h5py.version.hdf5_version_tuple >= (2, 0, 0) else 521
    return {
        "rdcc_nbytes": cache_chunks * chunk_nbytes,
        "rdcc_nslots": _next_prime(max(default_nslots, 100 * cache_chunks)),
    }
