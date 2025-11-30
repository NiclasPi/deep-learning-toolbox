import io
import numpy as np
import soundfile as sf
import torch
from typing import Literal, Union

from dltoolbox.transforms.core import TransformerBase, TransformerWithMode
from dltoolbox.transforms._utils import make_slices


class AudioCodecCompression(TransformerWithMode):
    """
    Apply compression of different audio codecs to the input audio waveform.
    Select a fixed audio codec or a random codec is selected on __call__.
    Select a fixed compression level between 0 (minimum compression) and 1 (highest compression)
    or a random value from a range is selected on __call__.
    """

    def __init__(
            self,
            codec: Literal["MP3", "OGG"] | tuple[Literal["MP3", "OGG"], ...] = "MP3",
            level: float | tuple[float, float] = 0.5,
    ) -> None:
        super().__init__()
        self._codec = (codec,) if isinstance(codec, str) else codec
        self._l_min, self._l_max = (level, level) if isinstance(level, float) else level

    def _get_format(self) -> str:
        if len(self._codec) > 1:
            if self.is_eval_mode():
                # always select the first codec in eval mode
                return self._codec[0]
            else:
                return self._codec[np.random.randint(len(self._codec))]

        return self._codec[0]

    def _get_compression_level(self) -> float:
        if self.is_eval_mode():
            # return the compression level between min and max in eval mode
            return self._l_min + (self._l_max - self._l_min) // 2
        # select a random compression level in the interval [min, max)
        return np.random.uniform(low=self._l_min, high=self._l_max)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        # input has shape (channels, frames)

        is_tensor = False
        if isinstance(x, torch.Tensor):
            is_tensor = True
            x = x.cpu().numpy()

        x = np.swapaxes(x, 0, 1)  # python-soundfile expects shape (frames, channels)

        # save using the lossy codec
        out = io.BytesIO()
        sf.write(file=out, data=x, samplerate=22050,
                 format=self._get_format(),
                 compression_level=self._get_compression_level())
        # read the compressed data
        out.seek(0)
        data, _ = sf.read(out, dtype=x.dtype, always_2d=True)

        x = np.swapaxes(data, 0, 1)  # swap back to (channels, frames)

        if is_tensor:
            x = torch.from_numpy(x)
        return x


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
