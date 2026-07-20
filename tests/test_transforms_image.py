from typing import Literal, Tuple

import numpy as np
import pytest
import torch

import dltoolbox.transforms.image as tfs
from tests.utils import transform_create_input, transform_set_mode


class TestTransformsImage:
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

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("scale", [0.25, 0.5, 1.0, 2.0])
    def test_resolution_downgrade(self, backend: Literal["numpy", "torch"], scale: float) -> None:
        tf = tfs.ResizeRoundTrip(dim=(0, 1), scale=scale)

        x = transform_create_input(backend, (16, 16))
        y = tf(x)

        assert x.shape == y.shape
        if scale == 1.0:
            # a round trip at scale 1.0 is a no-op
            assert np.allclose(x, y) if backend == "numpy" else torch.allclose(x, y)
        else:
            # a round trip at any other scale loses detail
            assert not (np.allclose(x, y) if backend == "numpy" else torch.allclose(x, y))

    def test_resolution_downgrade_invalid_scale(self) -> None:
        with pytest.raises(ValueError):
            tfs.ResizeRoundTrip(dim=(0, 1), scale=0.0)
        with pytest.raises(ValueError):
            tfs.ResizeRoundTrip(dim=(0, 1), scale=-1.0)

    @pytest.mark.parametrize("mode", ["train", "eval"])
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("dim", [(0, 1), (1, 2)])
    def test_random_resolution_downgrade(
        self, mode: Literal["train", "eval"], backend: Literal["numpy", "torch"], dim: Tuple[int, int]
    ) -> None:
        tf = tfs.RandomResizeRoundTrip(dim=dim, scale_range=(0.25, 0.75))
        transform_set_mode(tf, mode)

        x = transform_create_input(backend, (3, 16, 16))
        y = tf(x)

        assert x.shape == y.shape
        assert x.dtype == y.dtype

        if mode == "eval":
            assert np.array_equal(x, y) if backend == "numpy" else torch.equal(y, x)
        else:
            assert not (np.allclose(x, y) if backend == "numpy" else torch.allclose(x, y))

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("angle", [0.0, 45.0, 180.0])
    def test_rotate_any_degree(self, backend: Literal["numpy", "torch"], angle: float) -> None:
        tf = tfs.RotateAnyDegree(dim=(0, 1), angle=angle)

        x = transform_create_input(backend, (16, 16))
        y = tf(x)

        assert x.shape == y.shape
        assert x.dtype == y.dtype
        if angle == 0.0:
            # a rotation by 0 degrees is a no-op
            assert np.allclose(x, y) if backend == "numpy" else torch.allclose(x, y)
        else:
            assert not (np.allclose(x, y) if backend == "numpy" else torch.allclose(x, y))

    def test_rotate_any_degree_invalid_angle(self) -> None:
        with pytest.raises(ValueError):
            tfs.RotateAnyDegree(dim=(0, 1), angle=-1.0)
        with pytest.raises(ValueError):
            tfs.RotateAnyDegree(dim=(0, 1), angle=360.0)

    @pytest.mark.parametrize("mode", ["train", "eval"])
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("dim", [(0, 1), (1, 2)])
    def test_random_rotate_any_degree(
        self, mode: Literal["train", "eval"], backend: Literal["numpy", "torch"], dim: Tuple[int, int]
    ) -> None:
        tf = tfs.RandomRotateAnyDegree(dim=dim)
        transform_set_mode(tf, mode)

        x = transform_create_input(backend, (3, 16, 16))
        y = tf(x)

        assert x.shape == y.shape
        assert x.dtype == y.dtype

        if mode == "eval":
            assert np.array_equal(x, y) if backend == "numpy" else torch.equal(y, x)
        else:
            assert not (np.allclose(x, y) if backend == "numpy" else torch.allclose(x, y))
