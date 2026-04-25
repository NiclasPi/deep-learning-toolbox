from typing import Optional, Sequence, Union

import h5py
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from dltoolbox.dataset._utils import require_key
from dltoolbox.dataset.errors import H5DatasetShapeMismatchError
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
            require_key(h5_file, h5_data_key)
            if self.static_label is None:
                require_key(h5_file, h5_label_key)
                if h5_file[h5_data_key].shape[data_row_dim] != h5_file[h5_label_key].shape[label_row_dim]:
                    raise H5DatasetShapeMismatchError(
                        h5_file[h5_data_key].shape[data_row_dim], h5_file[h5_label_key].shape[label_row_dim]
                    )

            h5_data: h5py.Dataset = h5_file[h5_data_key]
            self.data = self._read_into_memory(h5_data, select_indices=select_indices, row_dim=data_row_dim)

            if self.static_label is None:
                h5_labels: h5py.Dataset = h5_file[h5_label_key]
                self.labels = self._read_into_memory(h5_labels, select_indices=select_indices, row_dim=label_row_dim)
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

    @staticmethod
    def _read_into_memory(
        h5_dataset: h5py.Dataset, *, select_indices: Optional[Sequence[int]], row_dim: int
    ) -> np.ndarray:
        """Materialize an h5 dataset (or a row selection of it) into a numpy array.

        With ``select_indices=None`` the full dataset is read in one ``read_direct`` call.
        With a selection, rows are read in sorted-unique order (h5py's ``source_sel``
        rejects non-monotonic indices) and then projected back into the caller's order,
        which may repeat or reorder rows freely. The extra in-memory shuffle keeps the
        selection contract identical to disk mode.

        Args:
            h5_dataset: source dataset on disk.
            select_indices: row positions along ``row_dim`` to materialize, in caller
                order; ``None`` reads the full dataset.
            row_dim: axis to which ``select_indices`` applies.

        Returns:
            A new numpy array; mutating it does not touch ``h5_dataset``.
        """
        if select_indices is None:
            out = np.empty(h5_dataset.shape, dtype=h5_dataset.dtype)
            h5_dataset.read_direct(out)
            return out

        sorted_unique = sorted(set(select_indices))
        compact_shape = tuple(s if d != row_dim else len(sorted_unique) for d, s in enumerate(h5_dataset.shape))
        compact_source_sel = tuple(slice(None) if d != row_dim else sorted_unique for d in range(h5_dataset.ndim))
        compact = np.empty(compact_shape, dtype=h5_dataset.dtype)
        h5_dataset.read_direct(compact, source_sel=compact_source_sel)

        position_in_compact = {orig: pos for pos, orig in enumerate(sorted_unique)}
        reorder = [position_in_compact[i] for i in select_indices]
        return np.take(compact, reorder, axis=row_dim)
