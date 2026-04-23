import json
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dltoolbox.dataset.create import create_dataset_from_arrays
from dltoolbox.dataset.errors import DatasetNumSamplesMismatchError, H5DatasetMissingKeyError
from dltoolbox.dataset.h5_dataset import H5Dataset
from dltoolbox.dataset.h5_dataset_disk import H5DatasetDisk
from dltoolbox.dataset.h5_dataset_memory import H5DatasetMemory
from dltoolbox.dataset.metadata.dataset_metadata import DatasetMetadata
from dltoolbox.transforms import ToTensor

USER_BLOCK_PAYLOAD = b"Hello from HDF5 user block!"


@dataclass
class SampleMeta:
    id: int


def _encode_sample_meta(obj: SampleMeta) -> bytes:
    return json.dumps(asdict(obj)).encode("utf-8")


def _decode_sample_meta(raw: bytes, sample_id: str) -> SampleMeta:
    return SampleMeta(**json.loads(raw))


@pytest.fixture(scope="module")
def data_shape() -> tuple[int, ...]:
    return 100, 16, 16, 3


@pytest.fixture(scope="module")
def labels_shape() -> tuple[int, ...]:
    return 100, 16, 1


@pytest.fixture(scope="module")
def create_temporary_hdf5(tmp_path_factory, data_shape, labels_shape) -> tuple[str, np.ndarray, np.ndarray]:
    path = tmp_path_factory.mktemp("h5") / "data.h5"
    data = np.random.randn(*data_shape).astype(np.float16)
    labels = np.random.randn(*labels_shape).astype(np.float16)
    create_dataset_from_arrays(str(path), data=data, labels=labels, user_block=USER_BLOCK_PAYLOAD)
    return str(path), data, labels


@pytest.fixture(scope="module")
def create_temporary_hdf5_data_only(tmp_path_factory, data_shape) -> tuple[str, np.ndarray]:
    path = tmp_path_factory.mktemp("h5_data_only") / "data.h5"
    data = np.random.randn(*data_shape).astype(np.float16)
    create_dataset_from_arrays(str(path), data=data, user_block=USER_BLOCK_PAYLOAD)
    return str(path), data


@pytest.fixture(scope="module")
def create_temporary_hdf5_with_meta(tmp_path_factory, data_shape, labels_shape) -> tuple[str, np.ndarray, np.ndarray]:
    path = tmp_path_factory.mktemp("h5_with_meta") / "data.h5"
    data = np.random.randn(*data_shape).astype(np.float16)
    labels = np.random.randn(*labels_shape).astype(np.float16)

    header = DatasetMetadata(name="test", split="train", num_samples=data_shape[0], origin_path="/path/to/origin")
    sample_ids = [str(i) for i in range(data_shape[0])]
    sample_meta = [SampleMeta(id=i) for i in range(data_shape[0])]

    create_dataset_from_arrays(
        str(path),
        data=data,
        labels=labels,
        user_block=header,
        sample_ids=sample_ids,
        sample_meta=sample_meta,
        sample_meta_encoder=_encode_sample_meta,
    )
    return str(path), data, labels


@pytest.mark.parametrize("mode", ["disk", "memory"])
@pytest.mark.parametrize("axis", [0, 1])
def test_reads_sample_at_row_dim(create_temporary_hdf5, mode: Literal["disk", "memory"], axis: int) -> None:
    h5_path, data, labels = create_temporary_hdf5
    dataset = H5Dataset(mode, h5_path, data_row_dim=axis, label_row_dim=axis)
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
def test_select_indices_exposes_subset(create_temporary_hdf5, mode: Literal["disk", "memory"], axis: int) -> None:
    h5_path, data, labels = create_temporary_hdf5
    indices = [0, 1, 2, 15]
    dataset = H5Dataset(mode, h5_path, data_row_dim=axis, label_row_dim=axis, select_indices=indices)

    assert len(dataset) == 4
    for i in range(len(indices)):
        sample, label, _ = dataset[i]
        assert sample.shape == tuple([s for d, s in enumerate(data.shape) if d != axis])
        assert label.shape == tuple([s for d, s in enumerate(labels.shape) if d != axis])
        assert np.allclose(np.take(data, indices[i], axis=axis), sample)
        assert np.allclose(np.take(labels, indices[i], axis=axis), label)


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_static_label_fills_in_missing_labels(create_temporary_hdf5_data_only, mode: Literal["disk", "memory"]) -> None:
    h5_path, data = create_temporary_hdf5_data_only
    dataset = H5Dataset(mode, h5_path, static_label=np.array([0], dtype=np.float32))
    assert len(dataset) == data.shape[0]
    sample, label, _ = dataset[0]
    assert sample.shape == tuple([s for d, s in enumerate(data.shape) if d != 0])
    assert label.shape == (1,)
    assert np.allclose(np.take(data, 0, axis=0), sample)
    assert np.allclose(0, label)


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_missing_labels_without_static_label_raises(
    create_temporary_hdf5_data_only, mode: Literal["disk", "memory"]
) -> None:
    h5_path, _ = create_temporary_hdf5_data_only
    with pytest.raises(H5DatasetMissingKeyError):
        H5Dataset(mode, h5_path)


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_user_block_bytes_roundtrip(create_temporary_hdf5, mode: Literal["disk", "memory"]) -> None:
    h5_path, _, _ = create_temporary_hdf5
    dataset = H5Dataset(mode, h5_path)

    assert dataset.ub_size >= len(USER_BLOCK_PAYLOAD)
    assert dataset.ub_bytes.rstrip(b"\x00") == USER_BLOCK_PAYLOAD


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_transforms_convert_samples_to_tensor(create_temporary_hdf5, mode: Literal["disk", "memory"]) -> None:
    h5_path, _, _ = create_temporary_hdf5
    dataset = H5Dataset(mode, h5_path, data_transform=ToTensor(), label_transform=ToTensor())
    sample, label, _ = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert isinstance(label, torch.Tensor)


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_collate_yields_none_when_meta_absent(create_temporary_hdf5, mode: Literal["disk", "memory"]) -> None:
    h5_path, _, _ = create_temporary_hdf5
    dataset = H5Dataset(mode, h5_path)
    loader = DataLoader(dataset, batch_size=4, collate_fn=H5Dataset.collate_fn)
    data, labels, meta = next(iter(loader))
    assert len(data) == 4
    assert len(labels) == 4
    assert len(meta) == 4
    assert all(meta[i] is None for i in range(4))


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_collate_preserves_sample_meta_order(create_temporary_hdf5_with_meta, mode: Literal["disk", "memory"]) -> None:
    h5_path, _, _ = create_temporary_hdf5_with_meta
    dataset = H5Dataset[SampleMeta](mode, h5_path, ignore_user_block=False, sample_meta_decoder=_decode_sample_meta)

    loader = DataLoader(dataset, batch_size=4, collate_fn=H5Dataset.collate_fn)
    _, _, meta = next(iter(loader))
    assert all(isinstance(m, SampleMeta) for m in meta)
    assert [m.id for m in meta] == [0, 1, 2, 3]


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_select_indices_exposes_meta_subset(create_temporary_hdf5_with_meta, mode: Literal["disk", "memory"]) -> None:
    h5_path, _, _ = create_temporary_hdf5_with_meta
    indices = [3, 10, 42]
    dataset = H5Dataset[SampleMeta](
        mode, h5_path, select_indices=indices, ignore_user_block=False, sample_meta_decoder=_decode_sample_meta
    )

    assert len(dataset) == len(indices)
    for i, original_idx in enumerate(indices):
        _, _, meta = dataset[i]
        assert isinstance(meta, SampleMeta)
        assert meta.id == original_idx


def test_h5_header_num_samples_mismatch_raises(data_shape, labels_shape, tmp_path) -> None:
    # Single mode only: the check runs in H5Dataset.__init__ before the disk/memory
    # branch in _build_store, so both modes execute the same code for this rule.
    path = tmp_path / "mismatch.h5"
    data = np.random.randn(*data_shape).astype(np.float16)
    labels = np.random.randn(*labels_shape).astype(np.float16)
    header = DatasetMetadata(
        name="test",
        split="train",
        num_samples=data_shape[0] + 1,  # lie about the count
        origin_path="/origin",
    )
    create_dataset_from_arrays(str(path), data=data, labels=labels, user_block=header)

    with pytest.raises(DatasetNumSamplesMismatchError):
        H5Dataset("memory", str(path), ignore_user_block=False)
