from __future__ import annotations

from collections.abc import Sequence

import h5py

from dltoolbox.dataset._utils import read_ids, validate_meta_datasets
from dltoolbox.dataset.metadata.isample_meta_store import ISampleMetaStore
from dltoolbox.dataset.metadata.sample_meta_protocols import SampleMetaDecoder


class EagerSampleMetaStore[T](ISampleMetaStore[T]):
    """Decodes all sample metadata rows at construction and serves from memory.

    If `select_indices` is given, only the selected rows are decoded and retained; the
    store exposes just that view (length, index lookup, and id lookup all operate on
    the selection).
    """

    def __init__(
        self,
        *,
        sample_ids_ds: h5py.Dataset,
        sample_meta_ds: h5py.Dataset | Sequence[h5py.Dataset],
        decoder: SampleMetaDecoder[T],
        select_indices: Sequence[int] | None = None,
    ) -> None:
        meta_dss, multi = validate_meta_datasets(sample_ids_ds, sample_meta_ds)

        all_ids = read_ids(sample_ids_ds)
        all_raw = [ds[:] for ds in meta_dss]

        if select_indices is None:
            view_ids = all_ids
            view_raw = all_raw
        else:
            view_ids = [all_ids[i] for i in select_indices]
            view_raw = [[raw[i] for i in select_indices] for raw in all_raw]

        def pack(row: tuple[object, ...]) -> bytes | tuple[bytes, ...]:
            rows = tuple(bytes(r) for r in row)  # type: ignore[call-overload]
            return rows if multi else rows[0]

        self._ids: list[str] = list(view_ids)
        self._items: list[T] = [
            decoder(pack(row), sid) for sid, row in zip(view_ids, zip(*view_raw, strict=True), strict=True)
        ]
        self._id_to_index: dict[str, int] = {sid: i for i, sid in enumerate(view_ids)}

    def __len__(self) -> int:
        return len(self._items)

    def get_by_index(self, index: int) -> T:
        return self._items[index]

    def get_by_id(self, identifier: str) -> T:
        return self._items[self._id_to_index[identifier]]

    def get_all_ids(self) -> list[str]:
        return list(self._ids)

    def get_all(self) -> list[tuple[str, T]]:
        return list(zip(self._ids, self._items, strict=True))
