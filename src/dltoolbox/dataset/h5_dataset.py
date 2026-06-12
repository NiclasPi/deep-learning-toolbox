from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Union

import h5py
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, default_collate

from dltoolbox.dataset._utils import read_ids, require_key
from dltoolbox.dataset.errors import (
    ConflictingSelectorsError,
    DatasetNumSamplesMismatchError,
    DuplicateSampleIdsError,
    DuplicateSelectIndicesError,
    SampleMetaStoreUnavailableError,
    UnknownSampleIdsError,
)
from dltoolbox.dataset.h5_dataset_disk import H5DatasetDisk
from dltoolbox.dataset.h5_dataset_memory import H5DatasetMemory
from dltoolbox.dataset.metadata.dataset_metadata import DatasetMetadata
from dltoolbox.dataset.metadata.eager_sample_meta_store import EagerSampleMetaStore
from dltoolbox.dataset.metadata.isample_meta_store import ISampleMetaStore
from dltoolbox.dataset.metadata.lazy_sample_meta_store import LazySampleMetaStore
from dltoolbox.dataset.metadata.sample_meta_protocols import SampleMetaDecoder, with_resolve
from dltoolbox.transforms import Transformer


class H5Dataset[T](Dataset):
    _instance: H5DatasetDisk | H5DatasetMemory
    _header: DatasetMetadata | None
    _store: ISampleMetaStore[T] | None

    def __init__(
        self,
        mode: Literal["disk", "memory"],
        h5_path: str,
        h5_data_key: str = "data",
        h5_label_key: str = "labels",
        h5_sample_ids_key: str = "metadata/sample_ids",
        h5_sample_meta_key: str = "metadata/sample_meta",
        select_indices: Sequence[int] | None = None,
        select_sample_ids: Sequence[str] | None = None,
        static_label: Union[np.ndarray, Tensor, None] = None,
        data_row_dim: int = 0,
        label_row_dim: int = 0,
        data_transform: Transformer | None = None,
        label_transform: Transformer | None = None,
        cache_chunks: int | None = None,
        rdcc_override: dict[str, int | float] | None = None,
        ignore_user_block: bool = True,
        sample_meta_decoder: SampleMetaDecoder[T] | None = None,
        sample_meta_store_mode: Literal["lazy", "eager", "auto"] = "auto",
    ) -> None:
        """Open an HDF5-backed dataset, dispatching to a disk- or memory-resident backend.

        Args:
            mode: ``"disk"`` keeps the file open and reads each sample on indexing; ``"memory"`` reads the whole
                (or selected) dataset into RAM up front and closes the file.
            h5_path: path to the HDF5 file.
            h5_data_key: key of the data dataset.
            h5_label_key: key of the label dataset; ignored when ``static_label`` is set.
            h5_sample_ids_key: key of the per-sample id dataset, used to resolve ``select_sample_ids`` and to back
                the sample-meta store.
            h5_sample_meta_key: key of the per-sample metadata dataset backing the store.
            select_indices: positional row indices to expose, in caller order; mutually exclusive with
                ``select_sample_ids``. ``None`` exposes every row.
            select_sample_ids: sample ids to expose, in caller order; resolved to positional indices via
                ``h5_sample_ids_key``. Mutually exclusive with ``select_indices``.
            static_label: a single label returned for every sample; when given, no label dataset is read.
            data_row_dim: axis of the data dataset indexed as the sample (row) dimension.
            label_row_dim: axis of the label dataset indexed as the sample (row) dimension.
            data_transform: transform applied to each data sample before it is returned.
            label_transform: transform applied to each label before it is returned.
            cache_chunks: number of chunks the HDF5 raw-data chunk cache should hold for the data dataset. Sizes
                ``rdcc_nbytes`` to exactly that many chunks and derives a prime ``rdcc_nslots`` (~100x the chunk
                count, never below the linked HDF5's default). ``None`` leaves the h5py defaults in place; must be
                positive otherwise. Ignored for contiguous (unchunked) datasets.
            rdcc_override: raw ``rdcc_*`` keyword arguments (``rdcc_nbytes``, ``rdcc_nslots``, ``rdcc_w0``) passed
                straight to ``h5py.File``. Takes precedence over ``cache_chunks`` — an escape hatch for tuning the
                cache by hand.
            ignore_user_block: when ``False``, parse the file's user block as dataset metadata and validate its
                sample count against the data dataset.
            sample_meta_decoder: decoder that turns raw per-sample metadata into ``T``; when given, a sample-meta
                store is built and exposed via ``__getitem__``.
            sample_meta_store_mode: ``"lazy"`` reads metadata on access, ``"eager"`` decodes it all into RAM,
                ``"auto"`` picks lazy for disk mode and eager for memory mode.

        Note:
            Chunk-cache settings (``cache_chunks`` / ``rdcc_override``) take effect only on the first open of a
            given file within the process. HDF5 shares one handle per file path, so a second ``H5Dataset`` over the
            same ``h5_path`` while the first is still open (e.g. train/val splits via ``select_indices`` in
            ``"disk"`` mode) silently reuses the first open's cache settings and ignores its own.
        """
        if select_indices is not None and select_sample_ids is not None:
            raise ConflictingSelectorsError()

        # reject duplicate positional indices
        if select_indices is not None:
            seen: set[int] = set()
            duplicates: list[int] = []
            for idx in select_indices:
                if idx in seen:
                    duplicates.append(idx)
                else:
                    seen.add(idx)
            if duplicates:
                raise DuplicateSelectIndicesError(duplicates)

        # translate id-based selection to positional indices by reading the sample_ids
        # dataset once; preserves caller-supplied order and validates that every
        # requested id exists and is unique before any instance is built
        if select_sample_ids is not None:
            select_indices = self._resolve_sample_ids_to_indices(
                h5_path=h5_path, h5_sample_ids_key=h5_sample_ids_key, sample_ids=select_sample_ids
            )

        if mode == "disk":
            self._instance = H5DatasetDisk(
                h5_path=h5_path,
                h5_data_key=h5_data_key,
                h5_label_key=h5_label_key,
                select_indices=select_indices,
                static_label=static_label,
                data_row_dim=data_row_dim,
                label_row_dim=label_row_dim,
                data_transform=data_transform,
                label_transform=label_transform,
                cache_chunks=cache_chunks,
                rdcc_override=rdcc_override,
            )
        elif mode == "memory":
            self._instance = H5DatasetMemory(
                h5_path=h5_path,
                h5_data_key=h5_data_key,
                h5_label_key=h5_label_key,
                select_indices=select_indices,
                static_label=static_label,
                data_row_dim=data_row_dim,
                label_row_dim=label_row_dim,
                data_transform=data_transform,
                label_transform=label_transform,
                cache_chunks=cache_chunks,
                rdcc_override=rdcc_override,
            )
        else:
            raise ValueError(f'unknown mode "{mode}"')

        # pre-register on self._instance (via __setattr__ delegation) so reads succeed
        # on the no-metadata path; without this, later access would AttributeError
        # because the instance classes don't define these attributes.
        self._header = None
        self._store = None

        # header (user-block) and sample-meta store are independent concerns:
        # * the header describes the file
        # * the store exposes per-sample payloads
        # build each if the corresponding inputs are present

        if not ignore_user_block and self._instance.ub_bytes is not None:
            self._header = DatasetMetadata.from_json_bytes(self._instance.ub_bytes.rstrip(b"\x00"))

            # validate header.num_samples against the original (pre-selection) data length
            with h5py.File(h5_path, "r") as f:
                actual_num_samples = f[h5_data_key].shape[data_row_dim]
            if self._header.num_samples != actual_num_samples:
                raise DatasetNumSamplesMismatchError(self._header.num_samples, actual_num_samples)

        if sample_meta_decoder is not None:
            if sample_meta_store_mode == "auto":
                store_mode = "lazy" if mode == "disk" else "eager"
            else:
                store_mode = sample_meta_store_mode

            self._store = self._build_store(
                h5_path=h5_path,
                store_mode=store_mode,  # type: ignore
                dataset_metadata=self._header,
                h5_sample_ids_key=h5_sample_ids_key,
                h5_sample_meta_key=h5_sample_meta_key,
                select_indices=select_indices,
                sample_meta_decoder=sample_meta_decoder,
            )

    @staticmethod
    def _resolve_sample_ids_to_indices(*, h5_path: str, h5_sample_ids_key: str, sample_ids: Sequence[str]) -> list[int]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for sid in sample_ids:
            if sid in seen:
                duplicates.append(sid)
            else:
                seen.add(sid)
        if duplicates:
            raise DuplicateSampleIdsError(duplicates)

        with h5py.File(h5_path, "r") as f:
            require_key(f, h5_sample_ids_key)
            all_ids = read_ids(f[h5_sample_ids_key])

        id_to_index = {sid: i for i, sid in enumerate(all_ids)}
        resolved: list[int] = []
        missing: list[str] = []
        for sid in sample_ids:
            idx = id_to_index.get(sid)
            if idx is None:
                missing.append(sid)
            else:
                resolved.append(idx)
        if missing:
            raise UnknownSampleIdsError(missing)
        return resolved

    def _build_store(
        self,
        *,
        h5_path: str,
        store_mode: Literal["lazy", "eager"],
        dataset_metadata: DatasetMetadata | None,
        h5_sample_ids_key: str,
        h5_sample_meta_key: str,
        select_indices: Sequence[int] | None,
        sample_meta_decoder: SampleMetaDecoder[T],
    ) -> ISampleMetaStore[T]:
        # skip resolution wrapping when no dataset metadata is available
        decoder = (
            with_resolve(sample_meta_decoder, dataset_metadata) if dataset_metadata is not None else sample_meta_decoder
        )
        if store_mode == "lazy":
            # share the already-open file handle from the disk instance
            f = self._instance.h5_file
            require_key(f, h5_sample_ids_key)
            require_key(f, h5_sample_meta_key)
            return LazySampleMetaStore(
                sample_ids_ds=f[h5_sample_ids_key],
                sample_meta_ds=f[h5_sample_meta_key],
                decoder=decoder,
                select_indices=select_indices,
            )
        elif store_mode != "eager":
            raise ValueError(f'unknown mode "{store_mode}" for sample meta store')
        # memory mode: open briefly to eagerly decode into RAM, then drop the handle
        with h5py.File(h5_path, "r") as f:
            require_key(f, h5_sample_ids_key)
            require_key(f, h5_sample_meta_key)
            return EagerSampleMetaStore(
                sample_ids_ds=f[h5_sample_ids_key],
                sample_meta_ds=f[h5_sample_meta_key],
                decoder=decoder,
                select_indices=select_indices,
            )

    @property
    def metadata(self) -> DatasetMetadata | None:
        return self._header

    @property
    def ub_size(self) -> int:
        return self._instance.ub_size

    @property
    def ub_bytes(self) -> bytes | None:
        return self._instance.ub_bytes

    def __len__(self) -> int:
        return self._instance.__len__()

    def __getitem__(self, index: int) -> tuple[Union[np.ndarray, Tensor], Union[np.ndarray, Tensor], T | None]:
        data, label = self._instance.__getitem__(index)
        meta = self._store.get_by_index(index) if self._store is not None else None
        return data, label, meta

    def __getattr__(self, name: str) -> Any:
        # delegate attribute access to the instance
        return getattr(self._instance, name)

    def __setattr__(self, key: str, value: Any) -> None:
        if key != "_instance":
            # delegate to the instance
            setattr(self._instance, key, value)
        else:
            self.__dict__[key] = value

    def get_all_sample_ids(self) -> list[str]:
        if self._store is None:
            raise SampleMetaStoreUnavailableError
        return self._store.get_all_ids()

    def get_all_sample_ids_with_meta(self) -> list[tuple[str, T]]:
        if self._store is None:
            raise SampleMetaStoreUnavailableError
        return self._store.get_all()

    @staticmethod
    def collate_fn(batch) -> tuple[Any, Any, list[Any]]:
        """Handle missing metadata or any metadata type by turning them into a list."""

        data, labels, meta = zip(*batch)
        batched_data = default_collate(data)
        batched_labels = default_collate(labels)
        batched_meta = list(meta)
        return batched_data, batched_labels, batched_meta
