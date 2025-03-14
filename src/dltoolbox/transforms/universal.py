import numpy as np
import torch
from itertools import chain
from typing import Literal, Self, Tuple, Union

from dltoolbox.transforms.core import TransformerBase
from dltoolbox.normalization import Normalization, WelfordEstimator
from dltoolbox.utils import inverse_permutation


class Flip(TransformerBase):
    """Reverse the order of the input along the given dimensions."""

    def __init__(self, dim: int | Tuple[int, ...]) -> None:
        self._dim: Tuple[int, ...] = dim if isinstance(dim, tuple) else (dim,)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, np.ndarray):
            return np.flip(x, axis=self._dim)
        elif isinstance(x, torch.Tensor):
            return torch.flip(x, dims=self._dim)
        else:
            raise ValueError(f"expected np.ndarray or torch.Tensor, got {type(x).__name__}")


class Normalize(TransformerBase):
    """Normalize using the mean and standard deviation. Supports broadcasting to a common shape."""

    @classmethod
    def from_normalization(cls, n: Normalization, device: torch.device = torch.device("cpu")) -> Self:
        if isinstance(n.mean, torch.Tensor):  # if mean is tensor, then std should also be a tensor
            return cls(n.mean.to(dtype=torch.float32, device=device),
                       n.std.to(dtype=torch.float32, device=device),
                       n.perm)
        else:
            return cls(n.mean,
                       n.std,
                       n.perm)

    @classmethod
    def from_welford(cls, welford: WelfordEstimator, device: torch.device = torch.device("cpu")) -> Self:
        mean, std, permute = welford.finalize(dtype=torch.float32, device=device)
        return cls(mean, std, permute)

    def __init__(self,
                 mean: Union[float, np.ndarray, torch.Tensor],
                 std: Union[float, np.ndarray, torch.Tensor],
                 permute: Tuple[int, ...] | None = None
                 ) -> None:
        self._mean = mean
        self._std = std
        self._permute = permute

    def get_normalization(self, device: str | torch.device | None = None) -> Normalization:
        if isinstance(self._mean, torch.Tensor):  # if mean is tensor, then std should also be a tensor
            return Normalization(self._mean.to(device=device), self._std.to(device=device), self._permute)
        else:
            return Normalization(self._mean, self._std, self._permute)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self._permute is not None:
            if isinstance(x, np.ndarray):
                x = np.transpose(x, self._permute)
            elif isinstance(x, torch.Tensor):
                x = torch.permute(x, self._permute)
            else:
                raise ValueError(f"expected np.ndarray or torch.Tensor, got {type(x).__name__}")

        x = (x - self._mean) / (self._std + 1e-12)

        if self._permute is not None:
            if isinstance(x, np.ndarray):
                x = np.transpose(x, inverse_permutation(self._permute))  # undo permute
            elif isinstance(x, torch.Tensor):
                x = torch.permute(x, inverse_permutation(self._permute))  # undo permute

        return x


class Pad(TransformerBase):
    """Pad the given input along given dimensions."""

    def __init__(self,
                 shape: Tuple[int, ...],
                 dim: Tuple[int, ...],
                 mode: Literal["constant", "reflect", "replicate", "circular"] = "constant",
                 value: float | None = 0
                 ) -> None:
        if len(shape) != len(dim):
            raise ValueError(f"shape and dim tuples must have the same length")
        if mode == "constant" and value is None:
            raise ValueError("value must be provided for 'constant' padding mode")

        self._shape = shape
        self._dim = dim
        self._mode = mode
        self._value = value

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        padding = [(0, 0)] * x.ndim

        for dim, shape in zip(self._dim, self._shape):
            if x.shape[dim] < shape:
                missing = shape - x.shape[dim]
                padding[dim] = (missing // 2, (missing + 1) // 2)

        if isinstance(x, np.ndarray):
            # conditionally build kwargs because Numpy does not allow optional arguments to be None
            kwargs = {}
            if self._mode == "constant":
                kwargs["mode"] = "constant"
                kwargs["constant_values"] = self._value
            elif self._mode == "reflect":
                kwargs["mode"] = "reflect"
            elif self._mode == "replicate":
                kwargs["mode"] = "edge"
            elif self._mode == "circular":
                kwargs["mode"] = "wrap"

            return np.pad(x, padding, **kwargs)
        elif isinstance(x, torch.Tensor):
            # for PyTorch, the padding type requires flattening to tuple of ints and the order needs to be reversed
            padding = tuple(chain.from_iterable(padding[::-1]))

            # TODO: need to manually wrap around more than once, if needed for circular mode
            # see: https://github.com/v0lta/PyTorch-Wavelet-Toolbox/pull/84
            return torch.nn.functional.pad(x, padding, mode=self._mode, value=self._value)
        else:
            raise ValueError(f"expected np.ndarray or torch.Tensor, got {type(x).__name__}")


class Permute(TransformerBase):
    """Permute dimensions of the given input."""

    def __init__(self, dim: Tuple[int, ...]):
        self._dim = dim

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, np.ndarray):
            return np.transpose(x, self._dim)
        elif isinstance(x, torch.Tensor):
            return torch.permute(x, self._dim)
        else:
            raise ValueError(f"expected np.ndarray or torch.Tensor, got {type(x).__name__}")


class Reshape(TransformerBase):
    """Reshape the given input."""

    def __init__(self, shape: Tuple[int, ...]):
        self._shape = shape

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, np.ndarray):
            return np.reshape(x, self._shape)
        elif isinstance(x, torch.Tensor):
            return torch.reshape(x, self._shape)
        else:
            raise ValueError(f"expected np.ndarray or torch.Tensor, got {type(x).__name__}")
