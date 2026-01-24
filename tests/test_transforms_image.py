from typing import Literal, Tuple, Union

import numpy as np
import pytest
import torch

import dltoolbox.transforms.image as tfs
from tests.utils import transform_create_input, transform_set_mode


class TestTransformsImage:
    @pytest.mark.parametrize("mode", ["train", "eval"])
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("shape", [(32, 32), (3, 32, 32)])
    @pytest.mark.parametrize("size", [16, (16, 16), 31, (31, 13)])
    def test_random_crop(
        self,
        mode: Literal["train", "eval"],
        backend: Literal["numpy", "torch"],
        shape: Tuple[int, ...],
        size: Union[int, Tuple[int, int]],
    ) -> None:
        tf = tfs.RandomCrop(size=size)
        transform_set_mode(tf, mode)

        x = transform_create_input(backend, shape)
        y = tf(x)

        assert y.shape[-1] == (size if isinstance(size, int) else size[-1])
        assert y.shape[-2] == (size if isinstance(size, int) else size[-2])

    @pytest.mark.parametrize("mode", ["train", "eval"])
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("dim", [(0,), (-1,), (0, 1)])
    @pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
    def test_random_flip(
        self, mode: Literal["train", "eval"], backend: Literal["numpy", "torch"], dim: Tuple[int, ...], prob: float
    ) -> None:
        tf = tfs.RandomFlip(dim=dim, prob=prob)
        transform_set_mode(tf, mode)

        x = transform_create_input(backend, (10, 10))
        y = tf(x)

        assert x.shape == y.shape
        if mode == "eval" or prob == 0.0:
            # no dimension should be flipped
            assert np.array_equal(y, x) if backend == "numpy" else torch.equal(y, x)
        elif mode == "train" and prob == 1.0:
            # all dimensions should be flipped
            assert np.array_equal(y, np.flip(x, dim)) if backend == "numpy" else torch.equal(y, torch.flip(x, dim))

    @pytest.mark.parametrize("mode", ["train", "eval"])
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("dim", [(0,), (-1,), (0, 1)])
    def test_random_noise(
        self, mode: Literal["train", "eval"], backend: Literal["numpy", "torch"], dim: Tuple[int, ...]
    ) -> None:
        tf = tfs.RandomNoise(dim=dim)
        transform_set_mode(tf, mode)

        x = transform_create_input(backend, (10, 10))
        y = tf(x)

        assert x.shape == y.shape
        assert x.dtype == y.dtype

        if mode == "eval":
            assert np.array_equal(x, y) if backend == "numpy" else torch.equal(y, x)
            return

        # checks for uint8
        if backend == "numpy":
            x_uint8 = (x * 256).astype(np.uint8)
            x_uint8[0, 0] = 0
            x_uint8[-1, -1] = 255
            y_uint8 = tf(x_uint8)
            # check if output type is input type
            assert y_uint8.dtype == np.uint8
            # check if no overflows occurred (for scale=1.0)
            assert not np.any(np.abs(x_uint8.astype(np.int16) - y_uint8.astype(np.int16)) > 1)
        elif backend == "torch":
            x_uint8 = (x * 256).to(torch.uint8)
            x_uint8[0, 0] = 0
            x_uint8[-1, -1] = 255
            y_uint8 = tf(x_uint8)
            # check if output type is input type
            assert y_uint8.dtype == torch.uint8
            # check if no overflows occurred (for scale=1.0)
            assert not torch.any(torch.abs(x_uint8.to(torch.int16) - y_uint8.to(torch.int16)) > 1)
