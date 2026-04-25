from __future__ import annotations

from collections.abc import Sequence

import h5py

from dltoolbox.dataset._utils import read_ids
from dltoolbox.dataset.errors import SampleMetaLengthMismatchError
from dltoolbox.dataset.metadata.isample_meta_store import ISampleMetaStore
from dltoolbox.dataset.metadata.sample_meta_protocols import SampleMetaDecoder


class LazySampleMetaStore[T](ISampleMetaStore[T]):
    """Holds references to the sample_ids and sample_meta datasets; reads rows on demand.

    Builds an id-to-index map at construction so `get_by_id` stays O(1). Sample ids
    themselves are not kept as a list — `get_by_index` reads the id from the HDF5
    dataset per call alongside the meta read.

    If `select_indices` is given, the store exposes just that view — `__len__`,
    `get_by_index`, and `get_by_id` all operate on the selection.
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

        self._sample_ids_ds = sample_ids_ds
        self._sample_meta_ds = sample_meta_ds
        self._original_indices: list[int] | None = list(select_indices) if select_indices is not None else None
        self._decoder = decoder

        all_ids = read_ids(sample_ids_ds)
        if self._original_indices is None:
            self._ids: list[str] = list(all_ids)
            self._id_to_index: dict[str, int] = {sid: i for i, sid in enumerate(all_ids)}
        else:
            self._ids = [all_ids[orig] for orig in self._original_indices]
            self._id_to_index = {sid: ext for ext, sid in enumerate(self._ids)}

    def __len__(self) -> int:
        if self._original_indices is not None:
            return len(self._original_indices)
        return len(self._sample_meta_ds)

    def get_by_index(self, index: int) -> T:
        original_index = self._original_indices[index] if self._original_indices is not None else index
        sample_id = self._sample_ids_ds[original_index]
        if isinstance(sample_id, bytes):
            sample_id = sample_id.decode("utf-8")
        raw = self._sample_meta_ds[original_index]
        return self._decoder(bytes(raw), sample_id)

    def get_by_id(self, identifier: str) -> T:
        return self.get_by_index(self._id_to_index[identifier])

    def get_all_ids(self) -> Sequence[str]:
        return list(self._ids)

    def get_all(self) -> Sequence[tuple[str, T]]:
        return [(sid, self.get_by_index(i)) for i, sid in enumerate(self._ids)]
