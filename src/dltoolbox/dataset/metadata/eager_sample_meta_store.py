from __future__ import annotations

from collections.abc import Sequence

import h5py

from dltoolbox.dataset.errors import SampleMetaLengthMismatchError
from dltoolbox.dataset.metadata._utils import read_ids
from dltoolbox.dataset.metadata.isample_meta_store import ISampleMetaStore
from dltoolbox.dataset.metadata.sample_meta_decoder import SampleMetaDecoder


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
        sample_meta_ds: h5py.Dataset,
        decoder: SampleMetaDecoder[T],
        select_indices: Sequence[int] | None = None,
    ) -> None:
        if len(sample_ids_ds) != len(sample_meta_ds):
            raise SampleMetaLengthMismatchError(len(sample_ids_ds), len(sample_meta_ds))

        all_ids = read_ids(sample_ids_ds)
        all_raw = sample_meta_ds[:]

        if select_indices is None:
            view_ids = all_ids
            view_raw = all_raw
        else:
            view_ids = [all_ids[i] for i in select_indices]
            view_raw = [all_raw[i] for i in select_indices]

        self._items: list[T] = [decoder.decode(raw, sid) for sid, raw in zip(view_ids, view_raw, strict=True)]
        self._id_to_index: dict[str, int] = {sid: i for i, sid in enumerate(view_ids)}

    def __len__(self) -> int:
        return len(self._items)

    def get_by_index(self, index: int) -> T:
        return self._items[index]

    def get_by_id(self, identifier: str) -> T:
        return self._items[self._id_to_index[identifier]]
