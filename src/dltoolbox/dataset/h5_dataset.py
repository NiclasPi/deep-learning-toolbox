from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

import h5py
import numpy as np
from cattrs import Converter
from torch import Tensor
from torch.utils.data import Dataset, default_collate

from dltoolbox.dataset.errors import DatasetNumSamplesMismatchError
from dltoolbox.dataset.h5_dataset_disk import H5DatasetDisk
from dltoolbox.dataset.h5_dataset_memory import H5DatasetMemory
from dltoolbox.dataset.metadata.dataset_metadata import DatasetMetadata, _default_converter
from dltoolbox.dataset.metadata.eager_sample_meta_store import EagerSampleMetaStore
from dltoolbox.dataset.metadata.isample_meta_store import ISampleMetaStore
from dltoolbox.dataset.metadata.lazy_sample_meta_store import LazySampleMetaStore
from dltoolbox.dataset.metadata.sample_meta_decoder import SampleMetaDecoder
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
        select_indices: Optional[Sequence[int]] = None,
        static_label: Optional[Union[np.ndarray, Tensor]] = None,
        data_row_dim: int = 0,
        label_row_dim: int = 0,
        data_transform: Optional[Transformer] = None,
        label_transform: Optional[Transformer] = None,
        ignore_user_block: bool = True,
        sample_meta_type: type[T] | None = None,
    ) -> None:
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
            )
        else:
            raise ValueError(f'unknown mode "{mode}"')

        # pre-register on self._instance (via __setattr__ delegation) so reads succeed
        # on the no-metadata path; without this, later access would AttributeError
        # because the instance classes don't define these attributes.
        self._header = None
        self._store = None

        if not ignore_user_block and self._instance.ub_bytes is not None:
            self._header = DatasetMetadata.from_json_bytes(self._instance.ub_bytes.rstrip(b"\x00"))

            # validate header.num_samples against the original (pre-selection) data length
            with h5py.File(h5_path, "r") as f:
                actual_num_samples = f[h5_data_key].shape[data_row_dim]
            if self._header.num_samples != actual_num_samples:
                raise DatasetNumSamplesMismatchError(self._header.num_samples, actual_num_samples)

            if sample_meta_type is not None:
                self._store = self._build_store(
                    h5_path=h5_path,
                    mode=mode,
                    header=self._header,
                    h5_sample_ids_key=h5_sample_ids_key,
                    h5_sample_meta_key=h5_sample_meta_key,
                    select_indices=select_indices,
                    sample_meta_type=sample_meta_type,
                )

    def _build_store(
        self,
        *,
        h5_path: str,
        mode: Literal["disk", "memory"],
        header: DatasetMetadata,
        h5_sample_ids_key: str,
        h5_sample_meta_key: str,
        select_indices: Sequence[int] | None,
        sample_meta_type: type[T],  # TODO: remove soon
    ) -> ISampleMetaStore[T]:
        decoder = SampleMetaDecoder(sample_meta_type=sample_meta_type, header=header)
        if mode == "disk":
            # share the already-open file handle from the disk instance
            f = self._instance.h5_file
            return LazySampleMetaStore(
                sample_ids_ds=f[h5_sample_ids_key],
                sample_meta_ds=f[h5_sample_meta_key],
                decoder=decoder,
                select_indices=select_indices,
            )
        # memory mode: open briefly to eagerly decode into RAM, then drop the handle
        with h5py.File(h5_path, "r") as f:
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
    def ub_bytes(self) -> Optional[bytes]:
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

    @staticmethod
    def collate_fn(batch) -> tuple[Any, Any, list[Any]]:
        """Handle missing metadata or any metadata type by turning them into a list."""

        data, labels, meta = zip(*batch)
        batched_data = default_collate(data)
        batched_labels = default_collate(labels)
        batched_meta = list(meta)
        return batched_data, batched_labels, batched_meta


def create_hdf5_file(
    output_path: str,
    data_arr: np.ndarray,
    labels_arr: Optional[np.ndarray],
    *,
    data_key: str = "data",
    labels_key: str = "labels",
    sample_ids_key: str = "metadata/sample_ids",
    sample_meta_key: str = "metadata/sample_meta",
    user_block: DatasetMetadata | bytes | None = None,
    sample_ids: Sequence[str] | None = None,
    sample_meta: Sequence[Any] | None = None,
    sample_meta_converter: Converter | None = None,
    metadata_compression: str | None = "gzip",
) -> None:
    if (sample_ids is None) != (sample_meta is None):
        raise ValueError("sample_ids and sample_meta must be provided together")
    if sample_ids is not None and len(sample_ids) != len(sample_meta):
        raise ValueError(f"sample_ids and sample_meta length mismatch: {len(sample_ids)} vs {len(sample_meta)}")

    ub_size: int = 0
    ub_bytes: bytes | None = None
    if user_block is not None:
        if isinstance(user_block, DatasetMetadata):
            ub_bytes = user_block.to_json_bytes()
        else:
            ub_bytes = user_block
        ub_size = max(512, int(2 ** np.ceil(np.log2(len(ub_bytes)))))

    with h5py.File(output_path, "w", userblock_size=ub_size, libver="latest") as h5_file:
        h5_file.create_dataset(data_key, data=data_arr)
        if labels_arr is not None:
            h5_file.create_dataset(labels_key, data=labels_arr)

        if sample_ids is not None:
            converter = sample_meta_converter if sample_meta_converter is not None else _default_converter
            h5_file.create_dataset(
                sample_ids_key, data=list(sample_ids), dtype=h5py.string_dtype(), compression=metadata_compression
            )
            h5_file.create_dataset(
                sample_meta_key,
                data=[converter.dumps(m) for m in sample_meta],
                dtype=h5py.string_dtype(),
                compression=metadata_compression,
            )

    if ub_size > 0 and ub_bytes:
        with open(output_path, "br+") as h5_file:
            h5_file.write(ub_bytes)
