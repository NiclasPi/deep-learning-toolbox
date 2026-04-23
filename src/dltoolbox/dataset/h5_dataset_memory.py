from typing import Optional, Sequence, Union

import h5py
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from dltoolbox.dataset.errors import H5DatasetMissingKeyError, H5DatasetShapeMismatchError
from dltoolbox.transforms import Transformer


class H5DatasetMemory(Dataset):
    """HDF5 dataset class that reads the entire dataset into main memory"""

    data: np.ndarray
    labels: Optional[np.ndarray]

    def __init__(
        self,
        h5_path: str,
        h5_data_key: str = "data",
        h5_label_key: str = "labels",
        select_indices: Optional[Sequence[int]] = None,
        static_label: Optional[Union[np.ndarray, Tensor]] = None,
        data_row_dim: int = 0,
        label_row_dim: int = 0,
        data_transform: Optional[Transformer] = None,
        label_transform: Optional[Transformer] = None,
    ) -> None:
        self.static_label = static_label
        self.data_row_dim: int = data_row_dim
        self.label_row_dim: int = label_row_dim
        self.data_transform = data_transform
        self.label_transform = label_transform

        self._ub_size: int = 0
        self._ub_bytes: Optional[bytes] = None
        # check for user block in HDF5 file
        with h5py.File(h5_path, "r") as h5_file:
            self._ub_size = h5_file.userblock_size

        # read user block if available
        if self._ub_size > 0:
            with open(h5_path, "br") as h5_file:
                self._ub_bytes = h5_file.read(self._ub_size)

        with h5py.File(h5_path, "r") as h5_file:
            if h5_data_key not in h5_file:
                raise H5DatasetMissingKeyError(h5_data_key)
            if self.static_label is None:
                if h5_label_key not in h5_file:
                    raise H5DatasetMissingKeyError(h5_label_key)
                if h5_file[h5_data_key].shape[data_row_dim] != h5_file[h5_label_key].shape[label_row_dim]:
                    raise H5DatasetShapeMismatchError(
                        h5_file[h5_data_key].shape[data_row_dim], h5_file[h5_label_key].shape[label_row_dim]
                    )

            h5_data: h5py.Dataset = h5_file[h5_data_key]
            if select_indices is not None:
                data_shape = tuple(
                    [s if d != data_row_dim else len(select_indices) for d, s in enumerate(h5_data.shape)]
                )
                data_source_sel = tuple(
                    [slice(None) if d != data_row_dim else select_indices for d, s in enumerate(h5_data.shape)]
                )
                self.data = np.empty(data_shape, dtype=h5_data.dtype)
                h5_data.read_direct(self.data, source_sel=data_source_sel)
            else:
                self.data = np.empty(h5_data.shape, dtype=h5_data.dtype)
                h5_data.read_direct(self.data)

            if self.static_label is None:
                h5_labels: h5py.Dataset = h5_file[h5_label_key]
                if select_indices is not None:
                    labels_shape = tuple(
                        [s if d != label_row_dim else len(select_indices) for d, s in enumerate(h5_labels.shape)]
                    )
                    labels_source_sel = tuple(
                        [slice(None) if d != label_row_dim else select_indices for d, s in enumerate(h5_labels.shape)]
                    )
                    self.labels = np.empty(labels_shape, dtype=h5_labels.dtype)
                    h5_labels.read_direct(self.labels, source_sel=labels_source_sel)
                else:
                    self.labels = np.empty(h5_labels.shape, dtype=h5_labels.dtype)
                    h5_labels.read_direct(self.labels)

                assert self.data.shape[data_row_dim] == self.labels.shape[label_row_dim]
            else:
                self.labels = None

    @property
    def ub_size(self) -> int:
        return self._ub_size

    @property
    def ub_bytes(self) -> Optional[bytes]:
        return self._ub_bytes

    def __len__(self) -> int:
        return self.data.shape[self.data_row_dim]

    def __getitem__(self, index: int) -> tuple[Union[np.ndarray, Tensor], Union[np.ndarray, Tensor]]:
        data = np.take(self.data, index, axis=self.data_row_dim)
        label = (
            self.static_label if self.static_label is not None else np.take(self.labels, index, axis=self.label_row_dim)
        )

        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.label_transform is not None:
            label = self.label_transform(label)

        return data, label
