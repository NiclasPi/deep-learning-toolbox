import json
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dltoolbox.dataset.create import create_dataset_from_arrays
from dltoolbox.dataset.errors import (
    ConflictingSelectorsError,
    DatasetNumSamplesMismatchError,
    DuplicateSampleIdsError,
    H5DatasetMissingKeyError,
    UnknownSampleIdsError,
)
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

    metadata = DatasetMetadata(name="test", split="train", num_samples=data_shape[0], origin_path="/path/to/origin")
    sample_ids = [f"s{i}" for i in range(data_shape[0])]
    sample_meta = [SampleMeta(id=i) for i in range(data_shape[0])]

    create_dataset_from_arrays(
        str(path),
        data=data,
        labels=labels,
        user_block=metadata,
        sample_ids=sample_ids,
        sample_meta=sample_meta,
        sample_meta_encoder=_encode_sample_meta,
    )
    return str(path), data, labels


@pytest.fixture(scope="module")
def create_temporary_hdf5_meta_no_user_block(
    tmp_path_factory, data_shape, labels_shape
) -> tuple[str, np.ndarray, np.ndarray]:
    """Per-sample ids+meta present, but no metadata header"""
    path = tmp_path_factory.mktemp("h5_meta_no_ub") / "data.h5"
    data = np.random.randn(*data_shape).astype(np.float16)
    labels = np.random.randn(*labels_shape).astype(np.float16)

    sample_ids = [f"s{i}" for i in range(data_shape[0])]
    sample_meta = [SampleMeta(id=i) for i in range(data_shape[0])]

    create_dataset_from_arrays(
        str(path),
        data=data,
        labels=labels,
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
def test_select_indices_unordered_preserves_caller_order(
    create_temporary_hdf5, mode: Literal["disk", "memory"]
) -> None:
    """Memory mode reads via h5py source_sel which requires monotonic indices internally;
    the caller-facing contract still has to be: rows come back in the order requested."""
    h5_path, data, _ = create_temporary_hdf5
    indices = [12, 3, 7, 0, 9]
    dataset = H5Dataset(mode, h5_path, select_indices=indices)

    assert len(dataset) == len(indices)
    for i, original in enumerate(indices):
        sample, _, _ = dataset[i]
        assert np.allclose(np.take(data, original, axis=0), sample)


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_select_indices_with_duplicates_replicates_rows(create_temporary_hdf5, mode: Literal["disk", "memory"]) -> None:
    """Index-based selection is positional, not a set — repeated indices repeat the row."""
    h5_path, data, _ = create_temporary_hdf5
    indices = [4, 1, 4, 4, 2]
    dataset = H5Dataset(mode, h5_path, select_indices=indices)

    assert len(dataset) == len(indices)
    for i, original in enumerate(indices):
        sample, _, _ = dataset[i]
        assert np.allclose(np.take(data, original, axis=0), sample)


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_select_indices_empty_yields_empty_dataset(create_temporary_hdf5, mode: Literal["disk", "memory"]) -> None:
    h5_path, _, _ = create_temporary_hdf5
    dataset = H5Dataset(mode, h5_path, select_indices=[])
    assert len(dataset) == 0


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


def test_h5_metadata_num_samples_mismatch_raises(data_shape, labels_shape, tmp_path) -> None:
    # Single mode only: the check runs in H5Dataset.__init__ before the disk/memory
    # branch in _build_store, so both modes execute the same code for this rule.
    path = tmp_path / "mismatch.h5"
    data = np.random.randn(*data_shape).astype(np.float16)
    labels = np.random.randn(*labels_shape).astype(np.float16)
    metadata = DatasetMetadata(
        name="test",
        split="train",
        num_samples=data_shape[0] + 1,  # lie about the count
        origin_path="/origin",
    )
    create_dataset_from_arrays(str(path), data=data, labels=labels, user_block=metadata)

    with pytest.raises(DatasetNumSamplesMismatchError):
        H5Dataset("memory", str(path), ignore_user_block=False)


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_sample_meta_store_built_without_user_block(
    create_temporary_hdf5_meta_no_user_block, mode: Literal["disk", "memory"]
) -> None:
    """A sample_meta_decoder alone is enough: no user block needed to enable the store."""
    h5_path, _, _ = create_temporary_hdf5_meta_no_user_block
    dataset = H5Dataset[SampleMeta](mode, h5_path, sample_meta_decoder=_decode_sample_meta)

    assert dataset.metadata is None  # no header was written
    _, _, meta = dataset[5]
    assert isinstance(meta, SampleMeta)
    assert meta.id == 5
    assert list(dataset.get_all_sample_ids())[:3] == ["s0", "s1", "s2"]


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_sample_meta_store_built_when_user_block_is_ignored(
    create_temporary_hdf5_with_meta, mode: Literal["disk", "memory"]
) -> None:
    """Header reading is orthogonal to store building: `ignore_user_block=True` must not gate the store."""
    h5_path, _, _ = create_temporary_hdf5_with_meta
    dataset = H5Dataset[SampleMeta](mode, h5_path, ignore_user_block=True, sample_meta_decoder=_decode_sample_meta)

    assert dataset.metadata is None
    _, _, meta = dataset[7]
    assert isinstance(meta, SampleMeta)
    assert meta.id == 7


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_select_sample_ids_exposes_subset(
    create_temporary_hdf5_meta_no_user_block, mode: Literal["disk", "memory"]
) -> None:
    h5_path, data, labels = create_temporary_hdf5_meta_no_user_block
    sample_ids = ["s3", "s10", "s42"]
    dataset = H5Dataset[SampleMeta](
        mode, h5_path, select_sample_ids=sample_ids, sample_meta_decoder=_decode_sample_meta
    )

    assert len(dataset) == len(sample_ids)
    for i, sid in enumerate(sample_ids):
        original_idx = int(sid[1:])
        sample, label, meta = dataset[i]
        assert np.allclose(np.take(data, original_idx, axis=0), sample)
        assert np.allclose(np.take(labels, original_idx, axis=0), label)
        assert isinstance(meta, SampleMeta)
        assert meta.id == original_idx

    # id view must track the caller-supplied order
    assert list(dataset.get_all_sample_ids()) == sample_ids


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_select_sample_ids_preserves_caller_order(
    create_temporary_hdf5_meta_no_user_block, mode: Literal["disk", "memory"]
) -> None:
    """Order is caller-defined and need not match the on-disk sample_ids order."""
    h5_path, _, _ = create_temporary_hdf5_meta_no_user_block
    sample_ids = ["s5", "s1", "s12", "s0"]
    dataset = H5Dataset[SampleMeta](
        mode, h5_path, select_sample_ids=sample_ids, sample_meta_decoder=_decode_sample_meta
    )

    assert len(dataset) == len(sample_ids)
    assert [dataset[i][2].id for i in range(len(sample_ids))] == [5, 1, 12, 0]
    assert list(dataset.get_all_sample_ids()) == sample_ids


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_select_sample_ids_raises_on_duplicates(
    create_temporary_hdf5_meta_no_user_block, mode: Literal["disk", "memory"]
) -> None:
    """Duplicates in select_sample_ids are rejected — sample ids must uniquely identify samples."""
    h5_path, _, _ = create_temporary_hdf5_meta_no_user_block
    with pytest.raises(DuplicateSampleIdsError) as error:
        H5Dataset(mode, h5_path, select_sample_ids=["s0", "s1", "s0", "s2", "s1"])
    assert error.value.duplicate_ids == ["s0", "s1"]


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_select_sample_ids_works_without_decoder(
    create_temporary_hdf5_meta_no_user_block, mode: Literal["disk", "memory"]
) -> None:
    """Id-based selection is orthogonal to the store: no decoder, no store, but selection still works."""
    h5_path, data, _ = create_temporary_hdf5_meta_no_user_block
    dataset = H5Dataset(mode, h5_path, select_sample_ids=["s2", "s8"])

    assert len(dataset) == 2
    assert np.allclose(np.take(data, 2, axis=0), dataset[0][0])
    assert np.allclose(np.take(data, 8, axis=0), dataset[1][0])
    assert dataset[0][2] is None  # no store → no meta


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_select_sample_ids_raises_on_unknown_id(
    create_temporary_hdf5_meta_no_user_block, mode: Literal["disk", "memory"]
) -> None:
    h5_path, _, _ = create_temporary_hdf5_meta_no_user_block
    with pytest.raises(UnknownSampleIdsError) as error:
        H5Dataset(mode, h5_path, select_sample_ids=["s0", "does-not-exist", "also-missing"])
    assert set(error.value.missing_ids) == {"does-not-exist", "also-missing"}


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_select_sample_ids_conflicts_with_select_indices(
    create_temporary_hdf5_meta_no_user_block, mode: Literal["disk", "memory"]
) -> None:
    h5_path, _, _ = create_temporary_hdf5_meta_no_user_block
    with pytest.raises(ConflictingSelectorsError):
        H5Dataset(mode, h5_path, select_indices=[0], select_sample_ids=["s0"])


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_select_sample_ids_requires_ids_dataset(create_temporary_hdf5, mode: Literal["disk", "memory"]) -> None:
    """Id-based selection needs the sample_ids dataset; error surfaces clearly when it's missing."""
    h5_path, _, _ = create_temporary_hdf5
    with pytest.raises(H5DatasetMissingKeyError):
        H5Dataset(mode, h5_path, select_sample_ids=["anything"])


@pytest.mark.parametrize("mode", ["disk", "memory"])
def test_decoder_without_ids_dataset_raises(create_temporary_hdf5, mode: Literal["disk", "memory"]) -> None:
    """Providing a decoder is a request to build the store; missing ids/meta datasets is a clear error."""
    h5_path, _, _ = create_temporary_hdf5
    with pytest.raises(H5DatasetMissingKeyError):
        H5Dataset[SampleMeta](mode, h5_path, sample_meta_decoder=_decode_sample_meta)
