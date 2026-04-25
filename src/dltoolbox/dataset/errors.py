from collections.abc import Sequence
from pathlib import Path


class H5DatasetMissingKeyError(Exception):
    def __init__(self, missing_key: str) -> None:
        super().__init__(f"Dataset is missing the following key: '{missing_key}'")


class H5DatasetShapeMismatchError(Exception):
    def __init__(self, shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
        super().__init__(f"Datasets have different shapes: {shape_a!s} and {shape_b!s}")


class SampleMetaLengthMismatchError(Exception):
    def __init__(self, sample_ids_len: int, sample_meta_len: int) -> None:
        self.sample_ids_len = sample_ids_len
        self.sample_meta_len = sample_meta_len
        super().__init__(f"sample_ids_ds and sample_meta_ds length mismatch: {sample_ids_len} vs {sample_meta_len}")


class DatasetNumSamplesMismatchError(Exception):
    def __init__(self, header_num_samples: int, actual_num_samples: int) -> None:
        self.header_num_samples = header_num_samples
        self.actual_num_samples = actual_num_samples
        super().__init__(
            f"header.num_samples does not match the HDF5 data length: {header_num_samples} vs {actual_num_samples}"
        )


class BatchProcessItemError(Exception):
    """Exception raised when processing an individual item in a batch fails."""

    def __init__(self, file_path: Path, original_exception: Exception):
        self.file_path = file_path
        self.original_exception = original_exception
        super().__init__(f"error processing file '{file_path!s}': {original_exception!s}")

    def __reduce__(self):
        # tells pickle how to reconstruct the object
        return self.__class__, (self.file_path, self.original_exception)


class SampleMetaPairingError(ValueError):
    """Raised when `sample_ids` and `sample_meta` are not both provided or both omitted."""

    def __init__(self) -> None:
        super().__init__("sample_ids and sample_meta must be provided together")


class SampleMetaStoreUnavailableError(RuntimeError):
    """Raised when sample metadata is requested but no store was built.

    This happens when the dataset was constructed without a ``sample_meta_decoder``,
    so sample ids/meta are not accessible.
    """

    def __init__(self) -> None:
        super().__init__("sample metadata store is not initialized (constructed without sample_meta_decoder)")


class ConflictingSelectorsError(ValueError):
    """Raised when both ``select_indices`` and ``select_sample_ids`` are given to H5Dataset."""

    def __init__(self) -> None:
        super().__init__("select_indices and select_sample_ids are mutually exclusive; provide at most one")


class UnknownSampleIdsError(ValueError):
    """Raised when ``select_sample_ids`` contains ids not present in the dataset's sample_ids."""

    def __init__(self, missing_ids: Sequence[str]) -> None:
        self.missing_ids: list[str] = list(missing_ids)
        preview = ", ".join(repr(s) for s in self.missing_ids[:5])
        suffix = f" (and {len(self.missing_ids) - 5} more)" if len(self.missing_ids) > 5 else ""
        super().__init__(f"sample ids not found in dataset: [{preview}]{suffix}")


class DuplicateSampleIdsError(ValueError):
    """Raised when ``select_sample_ids`` contains the same id more than once."""

    def __init__(self, duplicate_ids: Sequence[str]) -> None:
        self.duplicate_ids: list[str] = list(duplicate_ids)
        preview = ", ".join(repr(s) for s in self.duplicate_ids[:5])
        suffix = f" (and {len(self.duplicate_ids) - 5} more)" if len(self.duplicate_ids) > 5 else ""
        super().__init__(f"select_sample_ids contains duplicates: [{preview}]{suffix}")


class SampleSequenceLengthMismatchError(ValueError):
    """Raised when two per-sample sequences that must be index-matched have different lengths."""

    def __init__(self, left_name: str, left_length: int, right_name: str, right_length: int) -> None:
        super().__init__(f"{left_name} and {right_name} length mismatch: {left_length} vs {right_length}")
        self.left_name = left_name
        self.left_length = left_length
        self.right_name = right_name
        self.right_length = right_length
