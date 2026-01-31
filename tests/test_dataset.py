import os
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from typing import IO, Literal, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dltoolbox.dataset.errors import H5DatasetMissingKeyError
from dltoolbox.dataset.h5dataset import H5Dataset, H5DatasetDisk, H5DatasetMemory, create_hdf5_file
from dltoolbox.dataset.metadata import DatasetMetadata
from dltoolbox.transforms import ToTensor


@pytest.fixture(scope="class")
def data_shape() -> Tuple[int, ...]:
    return 100, 16, 16, 3


@pytest.fixture(scope="class")
def labels_shape() -> Tuple[int, ...]:
    return 100, 16, 1


@pytest.fixture(scope="class")
def create_temporary_hdf5(data_shape, labels_shape) -> Generator[tuple[IO, np.ndarray, np.ndarray]]:
    tmp_file = tempfile.NamedTemporaryFile("w+b", delete=False)
    tmp_file.close()

    try:
        data = np.random.randn(*data_shape).astype(np.float16)
        labels = np.random.randn(*labels_shape).astype(np.float16)

        create_hdf5_file(tmp_file.name, data, labels, user_block=bytes("Hello from HDF5 user block!", encoding="utf-8"))

        yield tmp_file, data, labels
    finally:
        tmp_file.close()
        os.unlink(tmp_file.name)


@pytest.fixture(scope="class")
def create_temporary_hdf5_data_only(data_shape) -> Generator[tuple[IO, np.ndarray]]:
    tmp_file = tempfile.NamedTemporaryFile("w+b", delete=False)
    tmp_file.close()

    try:
        data = np.random.randn(*data_shape).astype(np.float16)

        create_hdf5_file(
            tmp_file.name, data, labels_arr=None, user_block=bytes("Hello from HDF5 user block!", encoding="utf-8")
        )

        yield tmp_file, data
    finally:
        tmp_file.close()
        os.unlink(tmp_file.name)


@pytest.mark.parametrize("mode", ["disk", "memory"])
@pytest.mark.parametrize("axis", [0, 1])
def test_h5(create_temporary_hdf5, mode: Literal["disk", "memory"], axis: int) -> None:
    h5_file, data, labels = create_temporary_hdf5
    dataset = H5Dataset(mode, h5_file.name, data_row_dim=axis, label_row_dim=axis)
    if mode == "disk":
        assert isinstance(dataset._instance, H5DatasetDisk)
    elif mode == "memory":
        assert isinstance(dataset._instance, H5DatasetMemory)

    assert len(dataset) == data.shape[axis]
    sample, label, _ = dataset[0]
    assert sample.shape == tuple([s for d, s in enumerate(data.shape) if d != axis])
    assert label.shape == tuple([s for d, s in enumerate(labels.shape) if d != axis])
    assert np.allclose(np.take(data, 0, axis=axis), sample)
    assert np.allclose(np.take(labels, 0, axis=axis), label)


@pytest.mark.parametrize("mode", ["disk", "memory"])
@pytest.mark.parametrize("axis", [0, 1])
def test_h5_indices(create_temporary_hdf5, mode: Literal["disk", "memory"], axis: int) -> None:
    h5_file, data, labels = create_temporary_hdf5
    indices = [0, 1, 2, 15]
    dataset = H5Dataset(mode, h5_file.name, data_row_dim=axis, label_row_dim=axis, select_indices=indices)

    assert len(dataset) == 4
    for i in range(len(indices)):
        sample, label, _ = dataset[i]
        assert sample.shape == tuple([s for d, s in enumerate(data.shape) if d != axis])
        assert label.shape == tuple([s for d, s in enumerate(labels.shape) if d != axis])
        assert np.allclose(np.take(data, indices[i], axis=axis), sample)
        assert np.allclose(np.take(labels, indices[i], axis=axis), label)


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_h5_static_label(create_temporary_hdf5_data_only, mode: Literal["disk", "memory"]) -> None:
    h5_file, data = create_temporary_hdf5_data_only
    dataset = H5Dataset(mode, h5_file.name, static_label=np.array([0], dtype=np.float32))
    assert len(dataset) == data.shape[0]
    sample, label, _ = dataset[0]
    assert sample.shape == tuple([s for d, s in enumerate(data.shape) if d != 0])
    assert label.shape == (1,)
    assert np.allclose(np.take(data, 0, axis=0), sample)
    assert np.allclose(0, label)

    with pytest.raises(H5DatasetMissingKeyError):
        # raises because hdf5 file does not contain labels, and we are not providing a static label
        H5Dataset(mode, h5_file.name)


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_h5_user_block(create_temporary_hdf5, mode: Literal["disk", "memory"]) -> None:
    h5_file, _, _ = create_temporary_hdf5
    dataset = H5Dataset(mode, h5_file.name)

    assert dataset.ub_size == 512
    assert dataset.ub_bytes.decode(encoding="utf-8").rstrip("\x00") == "Hello from HDF5 user block!"


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_h5_transform(create_temporary_hdf5, mode: Literal["disk", "memory"]) -> None:
    h5_file, data, labels = create_temporary_hdf5
    dataset = H5Dataset(mode, h5_file.name, data_transform=ToTensor(), label_transform=ToTensor())
    assert len(dataset) == data.shape[0]
    sample, label, _ = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert isinstance(label, torch.Tensor)


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_h5_collate_no_meta(create_temporary_hdf5, mode: Literal["disk", "memory"]) -> None:
    h5_file, data, labels = create_temporary_hdf5
    dataset = H5Dataset(mode, h5_file.name)
    loader = DataLoader(dataset, batch_size=4, collate_fn=H5Dataset.collate_fn)
    data, labels, meta = next(iter(loader))
    assert len(data) == 4
    assert len(labels) == 4
    assert len(meta) == 4
    assert all(meta[i] is None for i in range(4))


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_h5_collate_with_meta(create_temporary_hdf5, mode: Literal["disk", "memory"]) -> None:
    @dataclass
    class SampleMeta:
        id: int

    h5_file, data, labels = create_temporary_hdf5
    dataset = H5Dataset(mode, h5_file.name)

    dataset._metadata = DatasetMetadata(
        name="test",
        split="train",
        origin_path="/path/to/origin",
        sample_ids=list(str(i) for i in range(100)),
        sample_meta={str(i): SampleMeta(id=i) for i in range(100)},
    )

    loader = DataLoader(dataset, batch_size=4, collate_fn=H5Dataset.collate_fn)
    data, labels, meta = next(iter(loader))
    assert len(data) == 4
    assert len(labels) == 4
    assert len(meta) == 4
    assert all(isinstance(meta[i], SampleMeta) for i in range(4))
