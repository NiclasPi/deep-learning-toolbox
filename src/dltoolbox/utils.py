import numpy as np
import torch
from typing import Sequence, Tuple


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
