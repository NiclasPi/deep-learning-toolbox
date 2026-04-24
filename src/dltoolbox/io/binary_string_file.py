from pathlib import Path
from types import TracebackType
from typing import BinaryIO, ClassVar, Iterable, Iterator, Literal, Self


class CorruptedFileError(Exception):
    """Raised when the file is truncated or a length prefix points past EOF."""


Mode = Literal["r", "w", "a"]


class BinaryStringFile:
    _PREFIX_SIZE: ClassVar[int] = 4
    _BYTE_ORDER: ClassVar[Literal["little", "big"]] = "little"
    ENCODING: ClassVar[str] = "utf-8"

    _READ_MODES: ClassVar[frozenset[str]] = frozenset({"r"})
    _WRITE_MODES: ClassVar[frozenset[str]] = frozenset({"w", "a"})

    def __init__(self, path: Path, mode: Mode = "r") -> None:
        self._path: Path = Path(path)
        self._mode: Mode = mode
        self._file: BinaryIO
        if mode == "r":
            self._file = open(self._path, "rb")
        elif mode == "w":
            self._file = open(self._path, "wb")
        elif mode == "a":
            self._file = open(self._path, "ab")
        else:
            raise ValueError(f"invalid mode {mode!r}; expected one of 'r', 'w', 'a'")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def flush(self) -> None:
        if self._mode in self._WRITE_MODES:
            self._file.flush()

    def __iter__(self) -> Iterator[str]:
        self._require_mode(*self._READ_MODES)
        while (record := self._read_record()) is not None:
            yield record

    def read_one(self) -> str | None:
        self._require_mode(*self._READ_MODES)
        return self._read_record()

    def read_all(self) -> list[str]:
        return list(self)

    def write_one(self, name: str) -> None:
        self._require_mode(*self._WRITE_MODES)
        self._write_record(name)

    def write_many(self, names: Iterable[str]) -> None:
        self._require_mode(*self._WRITE_MODES)
        for name in names:
            self._write_record(name)

    def _read_record(self) -> str | None:
        length_bytes = self._file.read(self._PREFIX_SIZE)
        if len(length_bytes) == 0:
            return None
        if len(length_bytes) < self._PREFIX_SIZE:
            raise CorruptedFileError(
                f"truncated length prefix: expected {self._PREFIX_SIZE} bytes, "
                f"got {len(length_bytes)}"
            )
        length = int.from_bytes(length_bytes, self._BYTE_ORDER)
        payload = self._file.read(length)
        if len(payload) < length:
            raise CorruptedFileError(
                f"truncated payload: expected {length} bytes, got {len(payload)}"
            )
        try:
            return payload.decode(self.ENCODING)
        except UnicodeDecodeError as exc:
            raise CorruptedFileError(f"invalid {self.ENCODING} payload") from exc

    def _write_record(self, name: str) -> None:
        payload = name.encode(self.ENCODING)
        self._file.write(len(payload).to_bytes(self._PREFIX_SIZE, self._BYTE_ORDER))
        self._file.write(payload)

    def _require_mode(self, *allowed: str) -> None:
        if self._mode not in allowed:
            raise IOError(
                f"operation not permitted in mode {self._mode!r}; "
                f"requires one of {sorted(allowed)}"
            )
