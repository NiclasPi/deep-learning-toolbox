import numpy as np
import pytest
import torch
from typing import Tuple

import dltoolbox.transforms as tfs

from tests.utils import transform_create_input

class TestTransformsFrequency:
    @pytest.mark.parametrize("dim", [(0,), (1,), (0, 2), (-2, -1)])
    @pytest.mark.parametrize("log", [True, False])
    def test_fft(
            self,
            dim: Tuple[int, ...],
            log: bool,
    ) -> None:
        tf = tfs.FFT(dim=dim, log=log)

        x = transform_create_input("numpy", shape=(3, 128, 128))
        y = tf(x)
        z = tf(torch.from_numpy(x))

        assert x.shape == y.shape
        assert x.shape == z.shape
        assert y.dtype == np.float32 if log else np.complex64
        assert z.dtype == torch.float32 if log else torch.complex64
        assert torch.allclose(torch.from_numpy(y), z, rtol=1.3e-4, atol=1e-5)
