import os
import random
from typing import Sequence, Tuple, Union

import numpy as np
import torch


def seed_rngs(random_seed: int = 42) -> None:
    """Seed all random number generators for reproducibility"""

    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def count_parameters(model: torch.nn.Module, requires_grad: bool = False) -> int:
    """Count the number of parameters of a model"""

    return sum(p.numel() for p in model.parameters() if not requires_grad or p.requires_grad)


def inverse_permutation(perm: Sequence[int]) -> Tuple[int, ...]:
    inv = [0] * len(perm)
    for i in range(len(perm)):
        inv[perm[i]] = i
    return tuple(inv)


def inverse_permutation_np(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    return inv


def inverse_permutation_pt(perm: torch.Tensor) -> torch.Tensor:
    # https://discuss.pytorch.org/t/how-to-quickly-inverse-a-permutation-by-using-pytorch/116205/4
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


def get_tolerances(dtype: Union[np.dtype, torch.dtype]) -> Tuple[float, float]:
    """Return relative and absolute tolerance (rtol, atol) based on dtype"""
    # tolerances taken from https://pytorch.org/docs/stable/testing.html

    if dtype == np.float16 or dtype == torch.float16:
        return 1e-3, 1e-5
    elif dtype == np.float32 or dtype == torch.float32:
        return 1.3e-6, 1e-5
    elif dtype == np.float64 or dtype == torch.float64:
        return 1e-7, 1e-7
    else:
        return 0.0, 0.0
