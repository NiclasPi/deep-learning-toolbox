from typing import Tuple


def make_slices(size: Tuple[int, ...], dims: Tuple[int, ...], slices: Tuple[slice, ...]) -> Tuple[slice, ...]:
    result = [slice(None)] * len(size)

    for d, s in zip(dims, slices):
        result[d] = s

    return tuple(result)
