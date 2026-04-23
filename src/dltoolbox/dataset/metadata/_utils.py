from __future__ import annotations

import h5py


def read_ids(ds: h5py.Dataset) -> list[str]:
    return [s.decode("utf-8") if isinstance(s, bytes) else s for s in ds[:]]
