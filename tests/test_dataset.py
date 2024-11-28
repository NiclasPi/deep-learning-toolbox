import h5py
import numpy as np
import pytest
import tempfile
from typing import Literal, Tuple

from dltoolbox.dataset import H5DatasetDisk, H5DatasetMemory, H5Dataset


@pytest.fixture(scope="class")
def data_shape() -> Tuple[int, ...]:
    return 100, 16, 16, 3


@pytest.fixture(scope="class")
def labels_shape() -> Tuple[int, ...]:
    return 100, 16, 1


@pytest.fixture(scope="class")
def create_hdf5_file(data_shape, labels_shape) -> Tuple[tempfile.NamedTemporaryFile, np.ndarray, np.ndarray]:
    tmp_file = tempfile.NamedTemporaryFile("w+b")
    h5_file = h5py.File(tmp_file, "w")

    data = np.random.randn(*data_shape).astype(np.float16)
    labels = np.random.randn(*labels_shape).astype(np.float16)

    h5_file.create_dataset("data", data=data)
    h5_file.create_dataset("labels", data=labels)
    h5_file.close()

    return tmp_file, data, labels


class TestDataset:
    @pytest.mark.parametrize("axis", [0, 1])
    def test_h5ds_disk(self, data_shape, labels_shape, create_hdf5_file, axis: int) -> None:
        h5_file, data, labels = create_hdf5_file
        dataset = H5DatasetDisk(h5_file.name,
                                data_row_dim=axis,
                                labels_row_dim=axis)

        assert len(dataset) == data_shape[axis] > 0
        sample, label = dataset[0]
        assert sample.shape == tuple([s for d, s in enumerate(data_shape) if d != axis])
        assert label.shape == tuple([s for d, s in enumerate(labels_shape) if d != axis])
        assert np.allclose(np.take(data, 0, axis=axis), sample)
        assert np.allclose(np.take(labels, 0, axis=axis), label)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_h5ds_memory(self, data_shape, labels_shape, create_hdf5_file, axis: int) -> None:
        h5_file, data, labels = create_hdf5_file
        dataset = H5DatasetMemory(h5_file.name,
                                  data_row_dim=axis,
                                  labels_row_dim=axis)
        assert len(dataset) == data_shape[axis] > 0
        sample, label = dataset[0]
        assert sample.shape == tuple([s for d, s in enumerate(data_shape) if d != axis])
        assert label.shape == tuple([s for d, s in enumerate(labels_shape) if d != axis])
        assert np.allclose(np.take(data, 0, axis=axis), sample)
        assert np.allclose(np.take(labels, 0, axis=axis), label)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_h5ds_disk_indices(self, data_shape, labels_shape, create_hdf5_file, axis: int) -> None:
        h5_file, data, labels = create_hdf5_file
        indices = [0, 1, 2, 15]
        dataset = H5DatasetDisk(h5_file.name,
                                data_row_dim=axis,
                                labels_row_dim=axis,
                                select_indices=indices)

        assert len(dataset) == 4
        for i in range(len(indices)):
            sample, label = dataset[i]
            assert sample.shape == tuple([s for d, s in enumerate(data_shape) if d != axis])
            assert label.shape == tuple([s for d, s in enumerate(labels_shape) if d != axis])
            assert np.allclose(np.take(data, indices[i], axis=axis), sample)
            assert np.allclose(np.take(labels, indices[i], axis=axis), label)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_h5ds_memory_indices(self, data_shape, labels_shape, create_hdf5_file, axis: int) -> None:
        h5_file, data, labels = create_hdf5_file
        indices = [0, 1, 2, 15]
        dataset = H5DatasetMemory(h5_file.name,
                                  data_row_dim=axis,
                                  labels_row_dim=axis,
                                  select_indices=indices)

        assert len(dataset) == 4
        for i in range(len(indices)):
            sample, label = dataset[i]
            assert sample.shape == tuple([s for d, s in enumerate(data_shape) if d != axis])
            assert label.shape == tuple([s for d, s in enumerate(labels_shape) if d != axis])
            assert np.allclose(np.take(data, indices[i], axis=axis), sample)
            assert np.allclose(np.take(labels, indices[i], axis=axis), label)

    @pytest.mark.parametrize("mode", ["disk", "memory"])
    def test_h5ds_mode(self, data_shape, labels_shape, create_hdf5_file, mode: Literal["disk", "memory"]) -> None:
        h5_file, data, labels = create_hdf5_file
        dataset = H5Dataset(mode, h5_file.name)
        assert len(dataset) == data_shape[0]
        sample, label = dataset[0]
        assert sample.shape == tuple([s for d, s in enumerate(data_shape) if d != 0])
        assert label.shape == tuple([s for d, s in enumerate(labels_shape) if d != 0])
        assert np.allclose(np.take(data, 0, axis=0), sample)
        assert np.allclose(np.take(labels, 0, axis=0), label)
