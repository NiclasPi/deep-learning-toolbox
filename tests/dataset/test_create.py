import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest

from dltoolbox.dataset.create import create_dataset_from_arrays, create_dataset_from_paths
from dltoolbox.dataset.errors import SampleMetaPairingError, SampleSequenceLengthMismatchError
from dltoolbox.dataset.metadata.dataset_metadata import DatasetMetadata


@dataclass
class SampleMeta:
    id: int


def _encode_sample_meta(obj: SampleMeta) -> bytes:
    return json.dumps(asdict(obj)).encode("utf-8")


def _unused_loader_func(_path: Path) -> np.ndarray:
    raise NotImplementedError("validation should fail before any sample is loaded")


# loaders below run in worker processes and must be picklable, hence module-level

_SAMPLE_SHAPE = (4, 4)


def _index_of(path: Path) -> int:
    return int(path.stem)


def _load_index_array(path: Path) -> np.ndarray:
    return np.full(_SAMPLE_SHAPE, _index_of(path), dtype=np.float32)


def _load_index_array_with_extra(path: Path) -> tuple[np.ndarray, dict]:
    return _load_index_array(path), {"index": _index_of(path)}


def _load_failing_every_third(path: Path) -> tuple[np.ndarray, dict]:
    if _index_of(path) % 3 == 0:
        raise ValueError("unloadable")
    return _load_index_array_with_extra(path)


def _load_with_unencodable_extra_every_third(path: Path) -> tuple[np.ndarray, object]:
    array, extra = _load_index_array_with_extra(path)
    return array, (object() if _index_of(path) % 3 == 0 else extra)


def _encode_index_extra(extra: dict) -> bytes:
    return str(extra["index"]).encode("ascii")


@dataclass
class _FailOnNthCall:
    # pickled afresh for every batch, so the call count restarts per batch
    fail_on_call: int
    calls: int = 0

    def __call__(self, _path: Path) -> np.ndarray:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise ValueError("unloadable")
        return np.full(_SAMPLE_SHAPE, self.calls, dtype=np.float32)


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
    def test_data_array_is_persisted_with_original_shape_and_dtype(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        create_dataset_from_arrays(str(out), data=data)
        with h5py.File(str(out), "r") as f:
            assert f["data"].shape == data.shape
            assert f["data"].dtype == data.dtype
            assert np.array_equal(f["data"][:], data)

    def test_labels_dataset_is_created_when_labels_are_provided(self, tmp_path, data, labels) -> None:
        out = tmp_path / "out.h5"
        create_dataset_from_arrays(str(out), data=data, labels=labels)
        with h5py.File(str(out), "r") as f:
            assert np.array_equal(f["labels"][:], labels)

    def test_labels_dataset_is_absent_when_no_labels_are_provided(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        create_dataset_from_arrays(str(out), data=data)
        with h5py.File(str(out), "r") as f:
            assert "labels" not in f

    def test_sample_ids_and_meta_roundtrip_through_metadata_group(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        ids = [f"s{i}" for i in range(data.shape[0])]
        meta = [SampleMeta(id=i) for i in range(data.shape[0])]

        create_dataset_from_arrays(
            str(out), data=data, sample_ids=ids, sample_meta=meta, sample_meta_encoder=_encode_sample_meta
        )

        with h5py.File(str(out), "r") as f:
            stored_ids = [s.decode() for s in f["metadata/sample_ids"][:]]
            stored_meta = [json.loads(bytes(raw)) for raw in f["metadata/sample_meta"][:]]
        assert stored_ids == ids
        assert stored_meta == [asdict(m) for m in meta]

    def test_raw_bytes_user_block_is_written_verbatim(self, tmp_path, data) -> None:
        payload = b"hello userblock"
        out = tmp_path / "out.h5"
        create_dataset_from_arrays(str(out), data=data, user_block=payload)
        with open(str(out), "rb") as f:
            assert f.read(len(payload)) == payload

    def test_dataset_metadata_is_serialized_into_user_block(self, tmp_path, data) -> None:
        header = DatasetMetadata(name="t", split="train", num_samples=data.shape[0], origin_path="/x")
        out = tmp_path / "out.h5"
        create_dataset_from_arrays(str(out), data=data, user_block=header)
        header_bytes = header.to_json_bytes()
        with open(str(out), "rb") as f:
            assert f.read(len(header_bytes)) == header_bytes

    def test_chunking_and_compression_options_are_honored(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        create_dataset_from_arrays(str(out), data=data, h5_chunk_length=2, h5_compression="gzip", h5_compression_opts=4)
        with h5py.File(str(out), "r") as f:
            ds = f["data"]
            assert ds.chunks == (2, *data.shape[1:])
            assert ds.compression == "gzip"
            assert ds.compression_opts == 4

    def test_refuses_to_overwrite_an_existing_output_file(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        create_dataset_from_arrays(str(out), data=data)
        with pytest.raises(FileExistsError):
            create_dataset_from_arrays(str(out), data=data)

    def test_sample_ids_without_sample_meta_is_rejected(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        with pytest.raises(SampleMetaPairingError):
            create_dataset_from_arrays(
                str(out), data=data, sample_ids=[f"s{i}" for i in range(data.shape[0])], sample_meta=None
            )

    def test_sample_meta_without_sample_ids_is_rejected(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        with pytest.raises(SampleMetaPairingError):
            create_dataset_from_arrays(
                str(out),
                data=data,
                sample_ids=None,
                sample_meta=[SampleMeta(id=i) for i in range(data.shape[0])],
                sample_meta_encoder=_encode_sample_meta,
            )

    def test_sample_ids_and_sample_meta_must_have_equal_length(self, tmp_path, data) -> None:
        out = tmp_path / "out.h5"
        with pytest.raises(SampleSequenceLengthMismatchError):
            create_dataset_from_arrays(
                str(out),
                data=data,
                sample_ids=["a", "b", "c"],
                sample_meta=[SampleMeta(id=0), SampleMeta(id=1)],
                sample_meta_encoder=_encode_sample_meta,
            )

    @pytest.mark.parametrize("ensure_ascii", [True, False])
    def test_sample_meta_preserves_non_ascii_utf8_under_either_encoder_mode(self, tmp_path, data, ensure_ascii) -> None:
        out = tmp_path / "out.h5"
        non_ascii_texts = [
            "日本語",
            "café",
            "naïve",
            "emoji 🚀",
            "Ω≈ç√∫",
            "Привет",
            "한국어",
            "Ελληνικά",
            "ü ö ä ß",
            "𝄞 music",
        ]
        assert len(non_ascii_texts) == data.shape[0]
        ids = [f"s{i}" for i in range(data.shape[0])]
        meta = [{"text": text} for text in non_ascii_texts]

        def encoder(obj):
            return json.dumps(obj, ensure_ascii=ensure_ascii).encode("utf-8")

        create_dataset_from_arrays(str(out), data=data, sample_ids=ids, sample_meta=meta, sample_meta_encoder=encoder)

        with h5py.File(str(out), "r") as f:
            raw_meta = f["metadata/sample_meta"][:]
        decoded = [json.loads(bytes(raw)) for raw in raw_meta]
        assert decoded == meta


class TestCreateDatasetFromPaths:
    @pytest.fixture(autouse=True)
    def _forbid_process_pool_executor(self, monkeypatch) -> None:
        def _raise(*_args, **_kwargs):
            raise AssertionError("ProcessPoolExecutor must not be instantiated in validation-error tests")

        monkeypatch.setattr("dltoolbox.dataset.create.ProcessPoolExecutor", _raise)

    def test_sample_ids_without_sample_meta_is_rejected(self, tmp_path, sample_paths) -> None:
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

    def test_sample_meta_without_sample_ids_is_rejected(self, tmp_path, sample_paths) -> None:
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

    def test_labels_length_must_match_sample_paths_length(self, tmp_path, sample_paths) -> None:
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

    def test_sample_ids_length_must_match_sample_paths_length(self, tmp_path, sample_paths) -> None:
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

    def test_sample_meta_length_must_match_sample_paths_length(self, tmp_path, sample_paths) -> None:
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


class TestCreateDatasetFromPathsLoading:
    num_samples = 20
    # 64-byte samples and 2 workers: 384 bytes leave room for 3 samples per batch, i.e. 7 batches
    multi_batch_kwargs = {"max_workers": 2, "max_memory": 384}

    @pytest.fixture
    def index_paths(self) -> list[Path]:
        # loaders derive everything from the file stem, so the files need not exist
        return [Path(f"/nonexistent/{i}.bin") for i in range(self.num_samples)]

    def _create(self, out: Path, paths: list[Path], loader_func, **kwargs):
        return create_dataset_from_paths(
            str(out),
            sample_paths=paths,
            loader_func=loader_func,
            sample_shape=_SAMPLE_SHAPE,
            sample_dtype=np.float32,
            sample_ids=[f"s{_index_of(p)}" for p in paths],
            sample_meta=[{"index": _index_of(p)} for p in paths],
            **kwargs,
        )

    @staticmethod
    def _read(out: Path, extra_key: str | None = None) -> tuple[list[int], list[str], list[dict], list[bytes] | None]:
        with h5py.File(str(out), "r") as f:
            data_indices = [int(sample[0, 0]) for sample in f["data"][:]]
            ids = [s.decode() for s in f["metadata/sample_ids"][:]]
            meta = [json.loads(bytes(raw)) for raw in f["metadata/sample_meta"][:]]
            extras = [bytes(raw) for raw in f[extra_key][:]] if extra_key is not None else None
        return data_indices, ids, meta, extras

    def test_without_extra_key_no_extra_dataset_is_written(self, tmp_path, index_paths) -> None:
        out = tmp_path / "out.h5"
        errors = self._create(out, index_paths, _load_index_array, **self.multi_batch_kwargs)

        assert errors == []
        data_indices, ids, meta, _ = self._read(out)
        assert sorted(data_indices) == list(range(self.num_samples))
        assert ids == [f"s{i}" for i in data_indices]
        assert meta == [{"index": i} for i in data_indices]
        with h5py.File(str(out), "r") as f:
            assert set(f["metadata"].keys()) == {"sample_ids", "sample_meta"}

    def test_extra_payloads_align_with_data_across_batches(self, tmp_path, index_paths) -> None:
        out = tmp_path / "out.h5"
        errors = self._create(
            out, index_paths, _load_index_array_with_extra, extra_key="metadata/extra", **self.multi_batch_kwargs
        )

        assert errors == []
        data_indices, ids, _, extras = self._read(out, "metadata/extra")
        assert sorted(data_indices) == list(range(self.num_samples))
        assert ids == [f"s{i}" for i in data_indices]
        assert [json.loads(e) for e in extras] == [{"index": i} for i in data_indices]

    def test_custom_extra_encoder_is_applied(self, tmp_path, index_paths) -> None:
        out = tmp_path / "out.h5"
        self._create(
            out,
            index_paths,
            _load_index_array_with_extra,
            extra_key="metadata/extra",
            extra_encoder=_encode_index_extra,
            **self.multi_batch_kwargs,
        )

        data_indices, _, _, extras = self._read(out, "metadata/extra")
        assert extras == [str(i).encode("ascii") for i in data_indices]

    def test_loader_failures_shrink_all_per_sample_datasets_consistently(self, tmp_path, index_paths) -> None:
        out = tmp_path / "out.h5"
        errors = self._create(
            out, index_paths, _load_failing_every_third, extra_key="metadata/extra", **self.multi_batch_kwargs
        )

        expected = [i for i in range(self.num_samples) if i % 3 != 0]
        assert sorted(_index_of(e.file_path) for e in errors) == [i for i in range(self.num_samples) if i % 3 == 0]
        data_indices, ids, meta, extras = self._read(out, "metadata/extra")
        assert sorted(data_indices) == expected
        assert ids == [f"s{i}" for i in data_indices]
        assert meta == [{"index": i} for i in data_indices]
        assert [json.loads(e) for e in extras] == [{"index": i} for i in data_indices]

    def test_extra_encoding_failure_skips_only_that_sample(self, tmp_path, index_paths) -> None:
        out = tmp_path / "out.h5"
        errors = self._create(
            out,
            index_paths,
            _load_with_unencodable_extra_every_third,
            extra_key="metadata/extra",
            **self.multi_batch_kwargs,
        )

        assert all(isinstance(e.original_exception, TypeError) for e in errors)
        data_indices, ids, _, extras = self._read(out, "metadata/extra")
        assert sorted(data_indices) == [i for i in range(self.num_samples) if i % 3 != 0]
        assert ids == [f"s{i}" for i in data_indices]
        assert [json.loads(e) for e in extras] == [{"index": i} for i in data_indices]

    def test_failure_of_one_sample_keeps_other_samples_sharing_its_path(self, tmp_path) -> None:
        shared = tmp_path / "shared.bin"
        shared.touch()
        out = tmp_path / "out.h5"
        # a single worker loads all four samples in one batch, and the second call fails
        errors = create_dataset_from_paths(
            str(out),
            sample_paths=[shared] * 4,
            loader_func=_FailOnNthCall(fail_on_call=2),
            sample_shape=_SAMPLE_SHAPE,
            sample_dtype=np.float32,
            labels=np.arange(4, dtype=np.int32),
            max_workers=1,
        )

        assert len(errors) == 1
        with h5py.File(str(out), "r") as f:
            assert [int(sample[0, 0]) for sample in f["data"][:]] == [1, 3, 4]
            assert f["labels"][:].tolist() == [0, 2, 3]
