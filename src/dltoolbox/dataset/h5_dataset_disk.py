from typing import Optional, Sequence, Union

import h5py
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from dltoolbox.dataset._utils import derive_rdcc_args, require_key
from dltoolbox.dataset.errors import H5DatasetShapeMismatchError
from dltoolbox.transforms import Transformer


class H5DatasetDisk(Dataset):
    """HDF5 dataset class that reads from disk when indexed"""

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
        cache_chunks: Optional[int] = None,
        rdcc_override: Optional[dict[str, int | float]] = None,
    ) -> None:
        self.h5_data_key = h5_data_key
        self.h5_label_key = h5_label_key
        self.selected_indices = select_indices
        self.static_label = static_label
        self.data_row_dim = data_row_dim
        self.label_row_dim = label_row_dim
        self.data_transform = data_transform
        self.label_transform = label_transform

        # check for user block in HDF5 file and derive the raw data chunk cache args
        self._ub_size: int = 0
        self._ub_bytes: Optional[bytes] = None
        rdcc_args: dict[str, int | float] = {}
        with h5py.File(h5_path, "r") as h5_file:
            self._ub_size = h5_file.userblock_size
            require_key(h5_file, h5_data_key)
            rdcc_args = derive_rdcc_args(h5_file[h5_data_key], cache_chunks, rdcc_override)

        # read user block if available
        if self._ub_size > 0:
            with open(h5_path, "br") as h5_file:
                self._ub_bytes = h5_file.read(self._ub_size)

        self.h5_file = h5py.File(h5_path, "r", **rdcc_args)
        self._data_ds = self.h5_file[h5_data_key]
        if self.static_label is None:
            require_key(self.h5_file, h5_label_key)
            self._label_ds = self.h5_file[h5_label_key]
            if self._data_ds.shape[data_row_dim] != self._label_ds.shape[label_row_dim]:
                raise H5DatasetShapeMismatchError(
                    self._data_ds.shape[data_row_dim], self._label_ds.shape[label_row_dim]
                )

        self.h5_data_len: int
        if select_indices is not None:
            self.h5_data_len = len(select_indices)
        else:
            self.h5_data_len = self._data_ds.shape[data_row_dim]

    @staticmethod
    def _h5_take(dataset: h5py.Dataset, index: int, axis: int) -> np.ndarray:
        slices = [index if d == axis else slice(None) for d in range(dataset.ndim)]
        return dataset[tuple(slices)]

    @property
    def ub_size(self) -> int:
        return self._ub_size

    @property
    def ub_bytes(self) -> Optional[bytes]:
        return self._ub_bytes

    def __len__(self) -> int:
        return self.h5_data_len

    def __getitem__(self, index: int) -> tuple[Union[np.ndarray, Tensor], Union[np.ndarray, Tensor]]:
        if self.selected_indices is not None:
            index = self.selected_indices[index]

        data = self._h5_take(self._data_ds, index, axis=self.data_row_dim)
        label = (
            self.static_label
            if self.static_label is not None
            else self._h5_take(self._label_ds, index, axis=self.label_row_dim)
        )

        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.label_transform is not None:
            label = self.label_transform(label)

        return data, label
