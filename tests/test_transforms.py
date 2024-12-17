import numpy as np
import pytest
import torch
from typing import Literal, Tuple, Union

import dltoolbox.transforms as tfs


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

        tf = tfs.DictTransformCreate({"one": tfs.NoTransform(), "two": tfs.ToTensor()})
        y = tf(x)

        assert "one" in y and isinstance(y["one"], np.ndarray)
        assert "two" in y and isinstance(y["two"], torch.Tensor)

    def test_dict_clone(self) -> None:
        x = {"one": create_input_sample(backend="numpy", shape=(10, 10), fill="zeros")}

        tf = tfs.DictTransformClone("one", "two")
        y = tf(x)

        x["one"][0, :] = 1  # default clone is deep copy
        assert "two" in y and not np.any(y["two"] == 1)

        tf = tfs.DictTransformClone("one", "two", shallow=True)
        z = tf(x)

        x["one"][1, :] = 2
        assert "two" in y and np.any(y["two"] == 1)

    def test_dict_apply(self) -> None:
        x = {"one": create_input_sample(backend="numpy", shape=(10, 10))}

        tf = tfs.DictTransformApply("one", tfs.ToTensor())
        y = tf(x)

        assert "one" in y and isinstance(y["one"], torch.Tensor)

        tf = tfs.DictTransformApply("two", tfs.ToTensor())
        with pytest.raises(KeyError):
            tf(x)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("dims", [(1, 0, 2, 3, 4), (1, 0, 3, 2, 4)])
    def test_permute(self, backend: Literal["numpy", "torch"], dims: Tuple[int, ...]) -> None:
        x = create_input_sample(backend, shape=(2, 4, 6, 8, 10))

        tf = tfs.Permute(dims)
        y = tf(x)

        for k in range(len(y.shape)):
            assert y.shape[k] == x.shape[dims[k]]

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_normalize(self, backend: Literal["numpy", "torch"]) -> None:
        x = create_input_sample(backend, shape=(10, 10))

        tf = tfs.Normalize(mean=1.0, std=0.5)
        y = tf(x)

        if backend == "numpy":
            assert np.allclose(y, (x - 1.0) / 0.5)
        else:
            assert torch.allclose(y, (x - 1.0) / 0.5)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_normalize_from_normalization_dataclass(self, backend: Literal["numpy", "torch"]) -> None:
        from dltoolbox.normalization import Normalization
        normalization: Normalization

        mean = np.full((3, 16, 16), fill_value=1.1, dtype=np.float32)
        std = np.full((3, 16, 16), fill_value=0.1, dtype=np.float32)
        if backend == "numpy":
            normalization = Normalization(mean, std)
        else:
            normalization = Normalization(torch.from_numpy(mean), torch.from_numpy(std))

        x = create_input_sample(backend, shape=(3, 16, 16))
        tf = tfs.Normalize.from_normalization(normalization)
        y = tf(x)

        if backend == "numpy":
            assert np.allclose(y, (x - mean) / std)
        else:
            assert torch.allclose(y, (x - torch.from_numpy(mean)) / torch.from_numpy(std))

    @pytest.mark.parametrize("dim", [None, (0,), (0, 3)])
    def test_normalize_from_welford_class(self, dim: Tuple[int, ...]) -> None:
        dataset = create_input_sample(backend="torch", shape=(25, 128, 128, 3))

        welford = tfs.WelfordEstimator(dim=dim)
        welford.update(dataset)

        x = dataset[0]
        tf = tfs.Normalize.from_welford(welford)
        y = tf(x)

        mean, std = torch.mean(dataset, dim=dim), torch.std(dataset, dim=dim)

        if tf._permute is not None:
            x = torch.permute(x, tf._permute)
            y = torch.permute(y, tf._permute)

        assert torch.allclose(y, (x - mean) / std, rtol=1e-2, atol=1e-1)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("dim", [(0,), (-1,), (0, 1)])
    @pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
    def test_random_flip(self, backend: Literal["numpy", "torch"], dim: Tuple[int, ...], prob: float) -> None:
        x = create_input_sample(backend, shape=(10, 10))

        tf = tfs.RandomFlip(dim=dim, prob=prob)
        y = tf(x)
        assert x.shape == y.shape

        if prob == 0.0:
            # no dimension should be flipped
            assert np.array_equal(y, x) if backend == "numpy" else torch.equal(y, x)
        elif prob == 1.0:
            # all dimensions should be flipped
            assert np.array_equal(y, np.flip(x, dim)) if backend == "numpy" else torch.equal(y, torch.flip(x, dim))

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("dim", [(0,), (-1,), (0, 1)])
    def test_random_noise(self, backend: Literal["numpy", "torch"], dim: Tuple[int, ...]) -> None:
        x = create_input_sample(backend, shape=(10, 10))

        tf = tfs.RandomNoise(dim=dim)
        y = tf(x)
        assert x.shape == y.shape
        assert x.dtype == y.dtype

        if backend == "numpy":
            x_uint8 = (x * 255).astype(np.uint8)
            x_uint8[0, 0] = 0
            x_uint8[-1, -1] = 255
            y_uint8 = tf(x_uint8)
            # check if output type is input type
            assert y_uint8.dtype == np.uint8
            # check if no overflows occurred (for scale=1.0)
            assert not np.any(np.abs(x_uint8.astype(np.int16) - y_uint8.astype(np.int16)) > 1)
        elif backend == "torch":
            x_uint8 = (x * 255).to(torch.uint8)
            x_uint8[0, 0] = 0
            x_uint8[-1, -1] = 255
            y_uint8 = tf(x_uint8)
            # check if output type is input type
            assert y_uint8.dtype == torch.uint8
            # check if no overflows occurred (for scale=1.0)
            assert not torch.any(torch.abs(x_uint8.to(torch.int16) - y_uint8.to(torch.int16)) > 1)

    @pytest.mark.parametrize("dim", [(0,), (1,), (0, 2)])
    def test_fft(self, dim: Tuple[int, ...]) -> None:
        x = create_input_sample("numpy", shape=(3, 128, 128))

        tf = tfs.FFT(dim=dim)
        y = tf(x)
        z = tf(torch.from_numpy(x))

        assert torch.allclose(torch.from_numpy(y), z, rtol=1e-3, atol=1e-5)
