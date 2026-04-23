import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest

from dltoolbox.dataset.create import create_dataset_from_arrays, create_dataset_from_paths
from dltoolbox.dataset.metadata.dataset_metadata import DatasetMetadata
from dltoolbox.dataset.errors import SampleMetaPairingError, SampleSequenceLengthMismatchError


@dataclass
class SampleMeta:
    id: int


def _encode_sample_meta(obj: SampleMeta) -> bytes:
    return json.dumps(asdict(obj)).encode("utf-8")


def _unused_loader_func(_path: Path) -> np.ndarray:
    raise NotImplementedError("validation should fail before any sample is loaded")


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    return np.random.randn(10, 4, 4).astype(np.float32)


@pytest.fixture(scope="module")
def labels() -> np.ndarray:
    return np.arange(10, dtype=np.int32)


@pytest.fixture(scope="module")
def sample_paths() -> list[Path]:
    return [Path(f"/nonexistent/{i}.bin") for i in range(10)]


class TestCreateDatasetFromArrays:
    def test_writes_data_array(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        create_dataset_from_arrays(str(out), data=data)
        with h5py.File(str(out), "r") as f:
            assert f["data"].shape == data.shape
            assert f["data"].dtype == data.dtype
            assert np.array_equal(f["data"][:], data)

    def test_writes_labels_when_provided(self, tmp_path, data, labels) -> None:
        out = tmp_path / "out.h5"
        create_dataset_from_arrays(str(out), data=data, labels=labels)
        with h5py.File(str(out), "r") as f:
            assert np.array_equal(f["labels"][:], labels)

    def test_omits_labels_when_none(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        create_dataset_from_arrays(str(out), data=data)
        with h5py.File(str(out), "r") as f:
            assert "labels" not in f

    def test_writes_sample_ids_and_meta(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        ids = [f"s{i}" for i in range(data.shape[0])]
        meta = [SampleMeta(id=i) for i in range(data.shape[0])]

        create_dataset_from_arrays(
            str(out), data=data, sample_ids=ids, sample_meta=meta, sample_meta_encoder=_encode_sample_meta
        )

        with h5py.File(str(out), "r") as f:
            stored_ids = [s.decode() for s in f["metadata/sample_ids"][:]]
            stored_meta = [json.loads(raw) for raw in f["metadata/sample_meta"][:]]
        assert stored_ids == ids
        assert stored_meta == [asdict(m) for m in meta]

    def test_writes_raw_user_block_bytes(self, tmp_path, data) -> None:
        payload = b"hello userblock"
        out = tmp_path / "out.h5"
        create_dataset_from_arrays(str(out), data=data, user_block=payload)
        with open(str(out), "rb") as f:
            assert f.read(len(payload)) == payload

    def test_writes_dataset_metadata_as_user_block(self, tmp_path, data) -> None:
        header = DatasetMetadata(name="t", split="train", num_samples=data.shape[0], origin_path="/x")
        out = tmp_path / "out.h5"
        create_dataset_from_arrays(str(out), data=data, user_block=header)
        header_bytes = header.to_json_bytes()
        with open(str(out), "rb") as f:
            assert f.read(len(header_bytes)) == header_bytes

    def test_applies_chunking_and_compression(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        create_dataset_from_arrays(str(out), data=data, h5_chunk_length=2, h5_compression="gzip", h5_compression_opts=4)
        with h5py.File(str(out), "r") as f:
            ds = f["data"]
            assert ds.chunks == (2, *data.shape[1:])
            assert ds.compression == "gzip"
            assert ds.compression_opts == 4

    def test_raises_when_output_path_exists(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        create_dataset_from_arrays(str(out), data=data)
        with pytest.raises(FileExistsError):
            create_dataset_from_arrays(str(out), data=data)

    def test_raises_when_sample_ids_provided_without_sample_meta(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        with pytest.raises(SampleMetaPairingError):
            create_dataset_from_arrays(
                str(out), data=data, sample_ids=[f"s{i}" for i in range(data.shape[0])], sample_meta=None
            )

    def test_raises_when_sample_meta_provided_without_sample_ids(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        with pytest.raises(SampleMetaPairingError):
            create_dataset_from_arrays(
                str(out),
                data=data,
                sample_ids=None,
                sample_meta=[SampleMeta(id=i) for i in range(data.shape[0])],
                sample_meta_encoder=_encode_sample_meta,
            )

    def test_raises_when_sample_ids_and_meta_have_different_lengths(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        with pytest.raises(SampleSequenceLengthMismatchError):
            create_dataset_from_arrays(
                str(out),
                data=data,
                sample_ids=["a", "b", "c"],
                sample_meta=[SampleMeta(id=0), SampleMeta(id=1)],
                sample_meta_encoder=_encode_sample_meta,
            )


class TestCreateDatasetFromPaths:
    @pytest.fixture(autouse=True)
    def _forbid_process_pool_executor(self, monkeypatch) -> None:
        def _raise(*_args, **_kwargs):
            raise AssertionError("ProcessPoolExecutor must not be instantiated in validation-error tests")

        monkeypatch.setattr("dltoolbox.dataset.create.ProcessPoolExecutor", _raise)

    def test_raises_when_sample_ids_provided_without_sample_meta(self, tmp_path, sample_paths) -> None:
        out = tmp_path / "out.h5"
        with pytest.raises(SampleMetaPairingError):
            create_dataset_from_paths(
                str(out),
                sample_paths=sample_paths,
                loader_func=_unused_loader_func,
                sample_shape=(4, 4),
                sample_dtype=np.float32,
                sample_ids=[f"s{i}" for i in range(len(sample_paths))],
                sample_meta=None,
            )

    def test_raises_when_sample_meta_provided_without_sample_ids(self, tmp_path, sample_paths) -> None:
        out = tmp_path / "out.h5"
        with pytest.raises(SampleMetaPairingError):
            create_dataset_from_paths(
                str(out),
                sample_paths=sample_paths,
                loader_func=_unused_loader_func,
                sample_shape=(4, 4),
                sample_dtype=np.float32,
                sample_ids=None,
                sample_meta=[SampleMeta(id=i) for i in range(len(sample_paths))],
                sample_meta_encoder=_encode_sample_meta,
            )

    def test_raises_when_labels_length_differs_from_sample_paths(self, tmp_path, sample_paths) -> None:
        out = tmp_path / "out.h5"
        with pytest.raises(SampleSequenceLengthMismatchError):
            create_dataset_from_paths(
                str(out),
                sample_paths=sample_paths,
                loader_func=_unused_loader_func,
                sample_shape=(4, 4),
                sample_dtype=np.float32,
                labels=np.zeros(len(sample_paths) - 1, dtype=np.int32),
            )

    def test_raises_when_sample_ids_length_differs_from_sample_paths(self, tmp_path, sample_paths) -> None:
        out = tmp_path / "out.h5"
        short_length = len(sample_paths) - 1
        with pytest.raises(SampleSequenceLengthMismatchError):
            create_dataset_from_paths(
                str(out),
                sample_paths=sample_paths,
                loader_func=_unused_loader_func,
                sample_shape=(4, 4),
                sample_dtype=np.float32,
                sample_ids=[f"s{i}" for i in range(short_length)],
                sample_meta=[SampleMeta(id=i) for i in range(short_length)],
                sample_meta_encoder=_encode_sample_meta,
            )

    def test_raises_when_sample_meta_length_differs_from_sample_paths(self, tmp_path, sample_paths) -> None:
        out = tmp_path / "out.h5"
        with pytest.raises(SampleSequenceLengthMismatchError):
            create_dataset_from_paths(
                str(out),
                sample_paths=sample_paths,
                loader_func=_unused_loader_func,
                sample_shape=(4, 4),
                sample_dtype=np.float32,
                sample_ids=[f"s{i}" for i in range(len(sample_paths))],
                sample_meta=[SampleMeta(id=i) for i in range(len(sample_paths) - 1)],
                sample_meta_encoder=_encode_sample_meta,
            )
