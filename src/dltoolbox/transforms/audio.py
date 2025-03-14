import numpy as np
import torch
from typing import Literal, Union

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


class MixSample(TransformerWithMode):
    """Mixes a predefined audio sample into an input sample. Assumes both samples have equal shape."""

    def __init__(self, sample: Union[np.ndarray, torch.Tensor], level: float | tuple[float, float] = 0.5) -> None:
        super().__init__()
        self._sample = sample
        self._level = level

    def _get_sample(self, backend: Literal["numpy", "torch"], size: int) -> np.ndarray | torch.Tensor:
        sample = self._sample
        if backend == "numpy" and isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        elif backend == "torch" and isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample)

        s = sample.shape[-1]
        if s < size:
            # TODO: introduce padding here
            raise ValueError(f"sample length {s} is smaller than the requested size {size}")
        elif s == size:
            return sample
        else:  # sample needs slicing
            i: int = (s - size - 1) // 2 if self.is_eval_mode() else np.random.randint(0, s - size + 1)
            return sample[..., i:i + size]

    def _get_level(self) -> float:
        if self.is_eval_mode():
            return 0.5
        if isinstance(self._level, tuple):
            return np.random.uniform(low=self._level[0], high=self._level[1])
        return self._level

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        sample = self._get_sample("numpy" if isinstance(x, np.ndarray) else "torch", x.shape[-1])
        level = self._get_level()
        return x + sample * level


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
