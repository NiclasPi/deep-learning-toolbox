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
    def test_dict_create(self) -> None:
        x = create_input_sample(backend="numpy", shape=(10, 10))

        tf = DictTransformCreate({"one": NoTransform(), "two": ToTensor()})
        y = tf(x)

        assert "one" in y and isinstance(y["one"], np.ndarray)
        assert "two" in y and isinstance(y["two"], torch.Tensor)

    def test_dict_clone(self) -> None:
        x = {"one": create_input_sample(backend="numpy", shape=(10, 10), fill="zeros")}

        tf = DictTransformClone("one", "two")
        y = tf(x)

        x["one"][0, :] = 1 # default clone is deep copy
        assert "two" in y and not np.any(y["two"] == 1)

        tf = DictTransformClone("one", "two", shallow=True)
        z = tf(x)

        x["one"][1, :] = 2
        assert "two" in y and np.any(y["two"] == 1)


    def test_dict_apply(self) -> None:
        x = {"one": create_input_sample(backend="numpy", shape=(10, 10))}

        tf = DictTransformApply("one", ToTensor())
        y = tf(x)

        assert "one" in y and isinstance(y["one"], torch.Tensor)

        tf = DictTransformApply("two", ToTensor())
        with pytest.raises(KeyError):
            tf(x)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("dims", [(1, 0, 2, 3, 4), (1, 0, 3, 2, 4)])
    def test_permute(self, backend: Literal["numpy", "torch"], dims: Tuple[int, ...]) -> None:
        x = create_input_sample(backend, shape=(2, 4, 6, 8, 10))

        tf = Permute(dims)
        y = tf(x)

        for k in range(len(y.shape)):
            assert y.shape[k] == x.shape[dims[k]]

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_normalize(self, backend: Literal["numpy", "torch"]) -> None:
        x = create_input_sample(backend, shape=(10, 10), fill=2)

        tf = Normalize(mean=1.0, std=0.5)
        y = tf(x)

        if backend == "numpy":
            assert np.allclose(y, 2)
        else:
            assert torch.isclose(y.mean(), torch.tensor([2], dtype=torch.float32))
