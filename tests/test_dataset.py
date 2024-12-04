import numpy as np
import pytest
import tempfile
import torch
import os
from typing import Literal, Tuple

from dltoolbox.dataset import H5DatasetDisk, H5DatasetMemory, H5Dataset, create_hdf5_file
from dltoolbox.transforms import ToTensor


@pytest.fixture(scope="class")
def data_shape() -> Tuple[int, ...]:
    return 100, 16, 16, 3


@pytest.fixture(scope="class")
def labels_shape() -> Tuple[int, ...]:
    return 100, 16, 1


@pytest.fixture(scope="class")
def create_temporary_hdf5(data_shape, labels_shape) -> Tuple[tempfile.NamedTemporaryFile, np.ndarray, np.ndarray]:
    tmp_file = tempfile.NamedTemporaryFile("w+b", delete=False)
    tmp_file.close()

    try:
        data = np.random.randn(*data_shape).astype(np.float16)
        labels = np.random.randn(*labels_shape).astype(np.float16)

        create_hdf5_file(tmp_file.name, data, labels,
                         ub_bytes=bytes("Hello from HDF5 user block!", encoding="utf-8"))

        yield tmp_file, data, labels
    finally:
        tmp_file.close()
        os.unlink(tmp_file.name)


@pytest.fixture(scope="class")
def create_temporary_hdf5_data_only(data_shape) -> Tuple[tempfile.NamedTemporaryFile, np.ndarray]:
    tmp_file = tempfile.NamedTemporaryFile("w+b", delete=False)
    tmp_file.close()

    try:
        data = np.random.randn(*data_shape).astype(np.float16)

        create_hdf5_file(tmp_file.name, data, labels_arr=None,
                         ub_bytes=bytes("Hello from HDF5 user block!", encoding="utf-8"))

        yield tmp_file, data
    finally:
        tmp_file.close()
        os.unlink(tmp_file.name)


class TestDataset:
    @pytest.mark.parametrize("axis", [0, 1])
    def test_h5_disk(self, labels_shape, create_temporary_hdf5, axis: int) -> None:
        h5_file, data, labels = create_temporary_hdf5
        dataset = H5DatasetDisk(h5_file.name,
                                data_row_dim=axis,
                                label_row_dim=axis)

        assert len(dataset) == data.shape[axis] > 0
        sample, label = dataset[0]
        assert sample.shape == tuple([s for d, s in enumerate(data.shape) if d != axis])
        assert label.shape == tuple([s for d, s in enumerate(labels_shape) if d != axis])
        assert np.allclose(np.take(data, 0, axis=axis), sample)
        assert np.allclose(np.take(labels, 0, axis=axis), label)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_h5_memory(self, create_temporary_hdf5, axis: int) -> None:
        h5_file, data, labels = create_temporary_hdf5
        dataset = H5DatasetMemory(h5_file.name,
                                  data_row_dim=axis,
                                  label_row_dim=axis)
        assert len(dataset) == data.shape[axis] > 0
        sample, label = dataset[0]
        assert sample.shape == tuple([s for d, s in enumerate(data.shape) if d != axis])
        assert label.shape == tuple([s for d, s in enumerate(labels.shape) if d != axis])
        assert np.allclose(np.take(data, 0, axis=axis), sample)
        assert np.allclose(np.take(labels, 0, axis=axis), label)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_h5_disk_indices(self, create_temporary_hdf5, axis: int) -> None:
        h5_file, data, labels = create_temporary_hdf5
        indices = [0, 1, 2, 15]
        dataset = H5DatasetDisk(h5_file.name,
                                data_row_dim=axis,
                                label_row_dim=axis,
                                select_indices=indices)

        assert len(dataset) == 4
        for i in range(len(indices)):
            sample, label = dataset[i]
            assert sample.shape == tuple([s for d, s in enumerate(data.shape) if d != axis])
            assert label.shape == tuple([s for d, s in enumerate(labels.shape) if d != axis])
            assert np.allclose(np.take(data, indices[i], axis=axis), sample)
            assert np.allclose(np.take(labels, indices[i], axis=axis), label)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_h5_memory_indices(self, create_temporary_hdf5, axis: int) -> None:
        h5_file, data, labels = create_temporary_hdf5
        indices = [0, 1, 2, 15]
        dataset = H5DatasetMemory(h5_file.name,
                                  data_row_dim=axis,
                                  label_row_dim=axis,
                                  select_indices=indices)

        assert len(dataset) == 4
        for i in range(len(indices)):
            sample, label = dataset[i]
            assert sample.shape == tuple([s for d, s in enumerate(data.shape) if d != axis])
            assert label.shape == tuple([s for d, s in enumerate(labels.shape) if d != axis])
            assert np.allclose(np.take(data, indices[i], axis=axis), sample)
            assert np.allclose(np.take(labels, indices[i], axis=axis), label)

    @pytest.mark.parametrize("mode", ["disk", "memory"])
    def test_h5_mode(self, create_temporary_hdf5, mode: Literal["disk", "memory"]) -> None:
        h5_file, data, labels = create_temporary_hdf5
        dataset = H5Dataset(mode, h5_file.name)
        assert len(dataset) == data.shape[0]
        sample, label = dataset[0]
        assert sample.shape == tuple([s for d, s in enumerate(data.shape) if d != 0])
        assert label.shape == tuple([s for d, s in enumerate(labels.shape) if d != 0])
        assert np.allclose(np.take(data, 0, axis=0), sample)
        assert np.allclose(np.take(labels, 0, axis=0), label)

    @pytest.mark.parametrize("mode", ["disk", "memory"])
    def test_h5_static_label(self, create_temporary_hdf5_data_only, mode: Literal["disk", "memory"]) -> None:
        h5_file, data = create_temporary_hdf5_data_only
        dataset = H5Dataset(mode, h5_file.name, static_label=np.array([0], dtype=np.float32))
        assert len(dataset) == data.shape[0]
        sample, label = dataset[0]
        assert sample.shape == tuple([s for d, s in enumerate(data.shape) if d != 0])
        assert label.shape == (1,)
        assert np.allclose(np.take(data, 0, axis=0), sample)
        assert np.allclose(0, label)

        with pytest.raises(AssertionError):
            # raises because hdf5 file does not contain labels, and we are not providing a static label
            H5Dataset(mode, h5_file.name)

    @pytest.mark.parametrize("mode", ["disk", "memory"])
    def test_h5_user_block(self, create_temporary_hdf5, mode: Literal["disk", "memory"]) -> None:
        h5_file, _, _ = create_temporary_hdf5
        dataset = H5Dataset(mode, h5_file.name)

        assert dataset.ub_size == 512
        assert dataset.ub_bytes.decode(encoding="utf-8").rstrip("\x00") == "Hello from HDF5 user block!"

    @pytest.mark.parametrize("mode", ["disk", "memory"])
    def test_h5_transform(self, create_temporary_hdf5, mode: Literal["disk", "memory"]) -> None:
        h5_file, data, labels = create_temporary_hdf5
        dataset = H5Dataset(mode, h5_file.name, data_transform=ToTensor(), label_transform=ToTensor())
        assert len(dataset) == data.shape[0]
        sample, label = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert isinstance(label, torch.Tensor)
