import pytest
import torch
from itertools import batched
from typing import Optional, Tuple

from dltoolbox.normalization import WelfordEstimator

class TestNormalization:
    @pytest.mark.parametrize("shape", [(100, 2, 8192), (50, 3, 128, 128), (50, 3, 64, 16, 16)])
    @pytest.mark.parametrize("dim", [None, (0,), (0, 1), (0, 2)])
    def test_welford(self, shape: Tuple[int, ...], dim: Optional[Tuple[int, ...]])-> None:
        dataset = torch.rand(shape, dtype=torch.float32)

        welford = WelfordEstimator(dim=dim)

        for batch_indices in batched(range(dataset.shape[0]), 8):
            batch = dataset[batch_indices, ...]
            welford.update(batch)

        # default dtypes are float64
        mean, std, permute = welford.finalize()
        assert mean.dtype == torch.float64
        assert std.dtype == torch.float64

        # return with tensors converted to float32
        mean, std, permute = welford.finalize(dtype=torch.float32)
        assert mean.dtype == torch.float32
        assert std.dtype == torch.float32

        # check computed mean and std values
        assert torch.allclose(mean, torch.mean(dataset, dim=dim), rtol=1e-2, atol=1e-3)  # generous tolerances
        assert torch.allclose(std, torch.std(dataset, dim=dim), rtol=1e-2, atol=1e-3)  # generous tolerances

        # can we apply the mean and std to a single sample?
        sample = dataset[0]
        assert permute is None or len(sample.shape) == len(permute)
        sample = torch.permute(sample, permute) if permute is not None else sample
        assert (sample - mean) / std is not None
