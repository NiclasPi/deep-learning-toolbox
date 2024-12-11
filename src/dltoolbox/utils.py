import numpy as np
import os
import random
import torch
from typing import Sequence, Tuple


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
