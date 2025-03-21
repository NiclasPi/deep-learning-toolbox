import numpy as np
import pytest
import torch
from typing import Literal, Tuple

import dltoolbox.transforms as tfs
from dltoolbox.normalization import WelfordEstimator

from tests.utils import transform_create_input


class TestTransformsUniversal:
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("dim", [0, 1, -1, (0,), (1,), (-1,), (0, 1), (0, -1)])
    def test_flip(self, backend: Literal["numpy", "torch"], dim: int | Tuple[int, ...]) -> None:
        x = transform_create_input(backend, shape=(2, 4))

        tf = tfs.Flip(dim=dim)
        y = tf(x)

        assert x.dtype == y.dtype
        assert x.shape == y.shape

        # get the first element for all dimensions
        x_slices = [slice(0, 1)] * x.ndim
        y_slices = [slice(0, 1)] * y.ndim
        for d in (dim if isinstance(dim, tuple) else (dim,)):
            # for the flipped dimensions, the first element is now expected to be the last element
            y_slices[d] = slice(-1, None)

        assert x[tuple(x_slices)] == y[tuple(y_slices)]

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_normalize(self, backend: Literal["numpy", "torch"]) -> None:
        x = transform_create_input(backend, shape=(10, 10))

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

        x = transform_create_input(backend, shape=(3, 16, 16))
        tf = tfs.Normalize.from_normalization(normalization)
        y = tf(x)

        if backend == "numpy":
            assert np.allclose(y, (x - mean) / std)
        else:
            assert torch.allclose(y, (x - torch.from_numpy(mean)) / torch.from_numpy(std))

    @pytest.mark.parametrize("dim", [None, (0,), (0, 3)])
    def test_normalize_from_welford_class(self, dim: Tuple[int, ...]) -> None:
        dataset = transform_create_input(backend="torch", shape=(25, 128, 128, 3))

        welford = WelfordEstimator(dim=dim)
        welford.update(dataset)

        x = dataset[0]
        tf = tfs.Normalize.from_welford(welford)
        y = tf(x)

        mean, std = torch.mean(dataset, dim=dim), torch.std(dataset, dim=dim)

        if tf._permute is not None:
            x = torch.permute(x, tf._permute)
            y = torch.permute(y, tf._permute)

        assert torch.allclose(y, (x - mean) / std, rtol=1e-2, atol=1e-1)

    @pytest.mark.parametrize("shape", [(16,), (16, 16)])
    @pytest.mark.parametrize("dim", [(0,), (1,), (-1,), (2, 1), (-2, -1), (1, -1)])
    @pytest.mark.parametrize("mode", ["constant", "reflect"])
    def test_pad(self, shape: Tuple[int, ...], dim: Tuple[int, ...], mode: Literal["constant", "reflect"]) -> None:
        if len(shape) != len(dim):
            with pytest.raises(ValueError):
                tfs.Pad(shape=shape, dim=dim, mode=mode)
        elif mode != "constant":
            # "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now"
            # Constant padding is implemented for arbitrary dimensions. Circular, replicate and reflection padding are
            # implemented for padding the last 3 dimensions of a 4D or 5D input tensor, the last 2 dimensions of a 3D
            # or 4D input tensor, or the last dimension of a 2D or 3D input tensor.
            #
            # None of the test cases match a valid configuration because we always pad all dimensions of the 3D input
            # tensor (even if some dimensions get (0, 0) padding, it is still counted as padding)
            with pytest.raises(NotImplementedError):
                tf = tfs.Pad(shape=shape, dim=dim, mode=mode)
                tf(torch.zeros((3, 8, 13)))
        else:
            x = transform_create_input("numpy", shape=(3, 8, 13))

            tf = tfs.Pad(shape=shape, dim=dim, mode=mode)
            y_np = tf(x)
            y_pt = tf(torch.from_numpy(x))

            for s, d in zip(shape, dim):
                assert y_np.shape[d] == y_pt.shape[d] == s

            assert np.allclose(y_np, y_pt.numpy())

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("dims", [(1, 0, 2, 3, 4), (1, 0, 3, 2, 4)])
    def test_permute(self, backend: Literal["numpy", "torch"], dims: Tuple[int, ...]) -> None:
        x = transform_create_input(backend, shape=(2, 4, 6, 8, 10))

        tf = tfs.Permute(dims)
        y = tf(x)

        for k in range(len(y.shape)):
            assert y.shape[k] == x.shape[dims[k]]

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("shape", [(10, 3, 256), (-1, 24, 32), (10, 2, -1, 32)])
    def test_reshape(self, backend: Literal["numpy", "torch"], shape: Tuple[int, ...]) -> None:
        x = transform_create_input(backend, shape=(10, 3, 16, 16))

        tf = tfs.Reshape(shape=shape)
        y = tf(x)

        for d, s in enumerate(shape):
            if s > 0:
                assert y.shape[d] == s
