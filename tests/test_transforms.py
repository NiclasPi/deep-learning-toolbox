import numpy as np
import pytest
import torch
from typing import Literal, Tuple, Union

from dltoolbox.transforms import *


def create_input_sample(backend: Literal["numpy", "torch"],
                        shape: Tuple[int, ...],
                        fill: Union[Literal["random", "zeros", "ones"], int, float] = "random",
                        ) -> Union[np.ndarray, torch.Tensor]:
    if backend == "numpy":
        if fill == "random":
            return np.random.rand(*shape).astype(np.float32)
        elif fill == "zeros":
            return np.zeros(shape, dtype=np.float32)
        elif fill == "ones":
            return np.ones(shape, dtype=np.float32)
        elif isinstance(fill, int) or isinstance(fill, float):
            return np.full(shape, fill, dtype=np.float32)
        else:
            raise ValueError(f"Unrecognized fill")
    elif backend == "torch":
        if fill == "random":
            return torch.rand(shape, dtype=torch.float32)
        elif fill == "zeros":
            return torch.zeros(shape, dtype=torch.float32)
        elif fill == "ones":
            return torch.ones(shape, dtype=torch.float32)
        elif isinstance(fill, int) or isinstance(fill, float):
            return torch.full(shape, fill, dtype=torch.float32)
        else:
            raise ValueError(f"Unrecognized fill")
    else:
        raise ValueError(f"Backend {backend} not supported")


class TestTransforms:
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_normalize(self, backend: Literal["numpy", "torch"]) -> None:
        x = create_input_sample(backend, shape=(10, 10), fill=2)

        tf = Normalize(mean=1.0, std=0.5)
        y = tf(x)

        if backend == "numpy":
            assert np.allclose(y, 2)
        else:
            assert torch.isclose(y.mean(), torch.tensor([2], dtype=torch.float32))
