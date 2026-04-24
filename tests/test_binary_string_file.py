from pathlib import Path

import pytest

from dltoolbox.io.binary_string_file import BinaryStringFile, CorruptedFileError


def test_invalid_mode_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid mode"):
        BinaryStringFile(tmp_path / "f.bin", mode="x")  # type: ignore[arg-type]


def test_round_trip_single_record(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    with BinaryStringFile(path, mode="w") as f:
        f.write_one("hello")
    with BinaryStringFile(path, mode="r") as f:
        assert f.read_one() == "hello"


def test_write_many_then_iterate_preserves_order(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    names = ["alpha", "beta", "gamma", "δelta"]
    with BinaryStringFile(path, mode="w") as f:
        f.write_many(names)
    with BinaryStringFile(path, mode="r") as f:
        assert list(f) == names


def test_read_one_returns_none_at_eof(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    with BinaryStringFile(path, mode="w") as f:
        f.write_one("only")
    with BinaryStringFile(path, mode="r") as f:
        assert f.read_one() == "only"
        assert f.read_one() is None


def test_iteration_stops_at_eof(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    with BinaryStringFile(path, mode="w") as f:
        f.write_many(["a", "b"])
    with BinaryStringFile(path, mode="r") as f:
        collected = []
        for rec in f:
            collected.append(rec)
        assert collected == ["a", "b"]


def test_write_in_read_mode_raises(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.touch()
    with BinaryStringFile(path, mode="r") as f:
        with pytest.raises(IOError):
            f.write_one("nope")


def test_read_in_write_mode_raises(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    with BinaryStringFile(path, mode="w") as f:
        with pytest.raises(IOError):
            f.read_one()


def test_truncated_length_prefix_raises_corrupted(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"\x01\x00")  # fewer than 4 bytes of prefix
    with BinaryStringFile(path, mode="r") as f:
        with pytest.raises(CorruptedFileError, match="length prefix"):
            f.read_one()


def test_truncated_payload_raises_corrupted(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    # declare 10-byte payload but only write 3 bytes
    path.write_bytes((10).to_bytes(4, "little") + b"abc")
    with BinaryStringFile(path, mode="r") as f:
        with pytest.raises(CorruptedFileError, match="payload"):
            f.read_one()


def test_invalid_utf8_payload_raises_corrupted(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    bad = b"\xff\xfe\xfd"
    path.write_bytes(len(bad).to_bytes(4, "little") + bad)
    with BinaryStringFile(path, mode="r") as f:
        with pytest.raises(CorruptedFileError, match="utf-8"):
            f.read_one()


def test_append_mode_preserves_prior_records(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    with BinaryStringFile(path, mode="w") as f:
        f.write_many(["first", "second"])
    with BinaryStringFile(path, mode="a") as f:
        f.write_one("third")
    with BinaryStringFile(path, mode="r") as f:
        assert list(f) == ["first", "second", "third"]


def test_context_manager_closes_file(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.touch()
    with BinaryStringFile(path, mode="r") as f:
        inner = f
    assert inner._file.closed


def test_close_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.touch()
    f = BinaryStringFile(path, mode="r")
    f.close()
    f.close()  # must not raise
    assert f._file.closed
