import h5py
import numpy as np
from torch.utils.data import Dataset
from typing import Literal, Optional, Sequence, Tuple


class H5DatasetDisk(Dataset):
    """HDF5 dataset class that reads from disk when indexed"""

    def __init__(self,
                 h5_path: str,
                 h5_data_key: str = "data",
                 h5_label_key: str = "labels",
                 select_indices: Optional[Sequence[int]] = None,
                 data_row_dim: int = 0,
                 labels_row_dim: int = 0,
                 ) -> None:
        self.h5_data_key = h5_data_key
        self.h5_label_key = h5_label_key
        self.selected_indices = select_indices
        self.data_row_dim = data_row_dim
        self.labels_row_dim = labels_row_dim

        # check for user block in HDF5 file
        self._ub_size: int = 0
        self._ub_bytes: Optional[bytes] = None
        with h5py.File(h5_path, "r") as h5_file:
            self._ub_size = h5_file.userblock_size

        # read user block if available
        if self._ub_size > 0:
            with open(h5_path, "br") as h5_file:
                self._ub_bytes = h5_file.read(self._ub_size)

        self.h5_file = h5py.File(h5_path, "r")
        assert h5_data_key in self.h5_file and type(self.h5_file[h5_data_key]) == h5py.Dataset
        assert h5_label_key in self.h5_file and type(self.h5_file[h5_label_key]) == h5py.Dataset
        assert self.h5_file[h5_data_key].shape[data_row_dim] == self.h5_file[h5_label_key].shape[labels_row_dim]

        self.h5_data_len: int
        if select_indices is not None:
            self.h5_data_len = len(select_indices)
        else:
            self.h5_data_len = self.h5_file[h5_data_key].shape[data_row_dim]

    def _data(self) -> h5py.Dataset:
        return self.h5_file[self.h5_data_key]

    def _labels(self) -> h5py.Dataset:
        return self.h5_file[self.h5_label_key]

    @staticmethod
    def _h5_take(dataset: h5py.Dataset, index: int, axis: int) -> np.array:
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

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.selected_indices is not None:
            index = self.selected_indices[index]

        return (self._h5_take(self._data(), index, axis=self.data_row_dim),
                self._h5_take(self._labels(), index, axis=self.labels_row_dim))


class H5DatasetMemory(Dataset):
    """HDF5 dataset class that reads the entire dataset into main memory"""

    def __init__(self,
                 h5_path: str,
                 h5_data_key: str = "data",
                 h5_label_key: str = "labels",
                 select_indices: Optional[Sequence[int]] = None,
                 data_row_dim: int = 0,
                 labels_row_dim: int = 0,
                 ) -> None:
        self.data: np.array
        self.labels: np.array
        self.data_row_dim: int = data_row_dim
        self.labels_row_dim: int = labels_row_dim

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
            assert h5_data_key in h5_file and type(h5_file[h5_data_key]) == h5py.Dataset
            assert h5_label_key in h5_file and type(h5_file[h5_label_key]) == h5py.Dataset
            assert h5_file[h5_data_key].shape[data_row_dim] == h5_file[h5_label_key].shape[labels_row_dim]

            h5_data: h5py.Dataset = h5_file[h5_data_key]
            h5_labels: h5py.Dataset = h5_file[h5_label_key]

            if select_indices is not None:
                data_shape = tuple(
                    [s if d != data_row_dim else len(select_indices) for d, s in enumerate(h5_data.shape)]
                )
                data_source_sel = tuple(
                    [slice(None) if d != data_row_dim else select_indices for d, s in enumerate(h5_data.shape)]
                )

                self.data = np.empty(data_shape, dtype=h5_data.dtype)
                h5_data.read_direct(self.data, source_sel=data_source_sel)

                labels_shape = tuple(
                    [s if d != labels_row_dim else len(select_indices) for d, s in enumerate(h5_labels.shape)]
                )
                labels_source_sel = tuple(
                    [slice(None) if d != labels_row_dim else select_indices for d, s in enumerate(h5_labels.shape)]
                )
                self.labels = np.empty(labels_shape, dtype=h5_labels.dtype)
                h5_labels.read_direct(self.labels, source_sel=labels_source_sel)
            else:
                self.data = np.empty(h5_data.shape, dtype=h5_data.dtype)
                h5_data.read_direct(self.data)
                self.labels = np.empty(h5_labels.shape, dtype=h5_labels.dtype)
                h5_labels.read_direct(self.labels)

            assert self.data.shape[data_row_dim] == self.labels.shape[labels_row_dim]

    @property
    def ub_size(self) -> int:
        return self._ub_size

    @property
    def ub_bytes(self) -> Optional[bytes]:
        return self._ub_bytes

    def __len__(self) -> int:
        return self.data.shape[self.data_row_dim]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return (np.take(self.data, index, axis=self.data_row_dim),
                np.take(self.labels, index, axis=self.labels_row_dim))


class H5Dataset(Dataset):
    def __init__(self,
                 mode: Literal["disk", "memory"],
                 h5_path: str,
                 h5_data_key: str = "data",
                 h5_label_key: str = "labels",
                 select_indices: Optional[Sequence[int]] = None,
                 data_row_dim: int = 0,
                 labels_row_dim: int = 0,
                 ) -> None:
        if mode == "disk":
            self.dataset = H5DatasetDisk(h5_path=h5_path,
                                         h5_data_key=h5_data_key,
                                         h5_label_key=h5_label_key,
                                         select_indices=select_indices,
                                         data_row_dim=data_row_dim,
                                         labels_row_dim=labels_row_dim)
        elif mode == "memory":
            self.dataset = H5DatasetMemory(h5_path=h5_path,
                                           h5_data_key=h5_data_key,
                                           h5_label_key=h5_label_key,
                                           select_indices=select_indices,
                                           data_row_dim=data_row_dim,
                                           labels_row_dim=labels_row_dim)
        else:
            raise ValueError(f'unknown mode "{mode}"')

    @property
    def ub_size(self) -> int:
        return self.dataset.ub_size

    @property
    def ub_bytes(self) -> Optional[bytes]:
        return self.dataset.ub_bytes

    def __len__(self) -> int:
        return self.dataset.__len__()

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.dataset.__getitem__(index)


def create_hdf5_file(output_path: str,
                     data_arr: np.ndarray,
                     labels_arr: np.ndarray,
                     data_key: str = "data",
                     labels_key: str = "labels",
                     ub_bytes: Optional[bytes] = None,
                     ) -> None:
    ub_size: int = 0
    if ub_bytes is not None and len(ub_bytes) > 0:
        ub_size = max(512, int(2 ** np.ceil(np.log2(len(ub_bytes)))))

    with h5py.File(output_path, "w", userblock_size=ub_size, libver="latest") as h5_file:
        h5_file.create_dataset(data_key, data=data_arr)
        h5_file.create_dataset(labels_key, data=labels_arr)

    if ub_size > 0:
        with open(output_path, "br+") as h5_file:
            h5_file.write(ub_bytes)
