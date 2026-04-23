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
