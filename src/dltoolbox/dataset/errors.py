class H5DatasetMissingKeyError(Exception):
    def __init__(self, missing_key: str) -> None:
        super().__init__(f"Dataset is missing the following key: '{missing_key}'")


class H5DatasetShapeMismatchError(Exception):
    def __init__(self, shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
        super().__init__(f"Datasets have different shapes: {shape_a!s} and {shape_b!s}")
