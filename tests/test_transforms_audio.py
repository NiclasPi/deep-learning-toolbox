import pytest
from typing import Literal, Tuple

import dltoolbox.transforms as tfs
from tests.utils import transform_set_mode, transform_create_input


class TestTransformsAudio:
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("shape", [(1, 32), (2, 32)])
    def test_invert_phase(
            self,
            backend: Literal["numpy", "torch"],
            shape: Tuple[int, ...],
    ) -> None:
        tf = tfs.InvertPhase()

        x = transform_create_input(backend, shape)
        y = tf(x)

        assert x.dtype == y.dtype
        assert y.shape == x.shape
        assert x[0, 0] == -1 * y[0, 0]
        assert x[-1, -1] == -1 * y[-1, -1]

    @pytest.mark.skip
    def test_random_attenuation(self) -> None:
        # TODO
        pass

    @pytest.mark.parametrize("mode", ["train", "eval"])
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("shape", [(1, 32), (2, 32)])
    @pytest.mark.parametrize("size", [16, 9])
    def test_random_slice(
            self,
            mode: Literal["train", "eval"],
            backend: Literal["numpy", "torch"],
            shape: Tuple[int, ...],
            size: int,
    ) -> None:
        tf = tfs.RandomSlice(size=size)
        transform_set_mode(tf, mode)

        x = transform_create_input(backend, shape)
        y = tf(x)

        assert x.dtype == y.dtype
        assert y.shape[-1] == size
