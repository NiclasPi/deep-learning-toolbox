import numpy as np
import torch
from typing import Union

from .core import TransformerBase, TransformerWithMode
from ._utils import make_slices


class ConvertToFloat32(TransformerBase):
    """Normalizes an integer PCM audio waveform to a float32 range of [-1, 1]."""

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, np.ndarray):
            if np.issubdtype(x.dtype, np.integer):
                x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
        elif isinstance(x, torch.Tensor):
            if not torch.is_floating_point(x) and not torch.is_complex(x):
                x = x.to(dtype=torch.float32) / float(torch.iinfo(x.dtype).max)
        else:
            raise ValueError(f"expected np.ndarray or torch.Tensor, got {type(x).__name__}")

        return x


class InvertPhase(TransformerBase):
    """Invert the phase of an audio waveform by negating its values."""

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if not (isinstance(x, np.ndarray) or isinstance(x, torch.Tensor)):
            raise ValueError(f"expected np.ndarray or torch.Tensor, got {type(x).__name__}")

        return x * -1


class RandomAttenuation(TransformerWithMode):
    """Attenuate the input values by multiplying an attenuation factor."""

    def __init__(self, attenuation: Union[float, tuple[float, float]]) -> None:
        super().__init__()
        self._attenuation = attenuation

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.is_eval_mode():
            return x

        attenuation: float
        if isinstance(self._attenuation, tuple):
            attenuation = np.random.uniform(self._attenuation[0], self._attenuation[1])
        else:
            attenuation = self._attenuation

        return x * attenuation


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
