import numpy as np
import torch
from typing import Union

from .core import TransformerWithMode
from ._utils import make_slices


class RandomSlice(TransformerWithMode):
    """Extract a random fixed-size slice along the given dimension of the input, retaining all other dimensions."""

    def __init__(self, size: int, dim: int = -1) -> None:
        super().__init__()
        self._ts = size
        self._dim = dim

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        s = x.shape[self._dim]
        if s < self._ts:
            raise ValueError(f"slice size {self._ts} exceeds input size {tuple(x.shape)} at dim={self._dim} ({s})")
        if s == self._ts:
            return x

        i: int
        if self.is_eval_mode():
            # extract a centered slice in eval mode
            i = (s - self._ts - 1) // 2
        else:
            i = np.random.randint(0, s - self._ts + 1)

        return x[make_slices(tuple(x.shape), (self._dim,), (slice(i, i + self._ts),))]
