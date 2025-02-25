import numpy as np
import torch
from abc import ABC, abstractmethod
from itertools import chain
from typing import Callable, Dict, Iterable, List, Literal, Optional, Self, Tuple, Union

from dltoolbox.normalization import Normalization, WelfordEstimator
from dltoolbox.utils import inverse_permutation

Transformer = Callable[
    [Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]],
    Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]
]


class TransformerBase(ABC):
    @abstractmethod
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError()


class TransformerMode(ABC):
    @abstractmethod
    def set_train_mode(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def set_eval_mode(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def is_train_mode(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def is_eval_mode(self) -> bool:
        raise NotImplementedError()


class TransformerWithMode(TransformerBase, TransformerMode):
    def __init__(self) -> None:
        self._eval = False

    @abstractmethod
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError()

    def set_train_mode(self) -> None:
        self._eval = False

    def set_eval_mode(self) -> None:
        self._eval = True

    def is_train_mode(self) -> bool:
        return not self._eval

    def is_eval_mode(self) -> bool:
        return self._eval


class Compose(TransformerBase):
    """Compose multiple transforms into one callable class."""

    def __init__(self, transforms: List[Transformer]):
        self._transforms = transforms

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        for transform in self._transforms:
            x = transform(x)
        return x

    def __len__(self) -> int:
        return len(self._transforms)

    def __setitem__(self, index: int, value: Transformer):
        self._transforms[index] = value

    def __getitem__(self, index: int) -> Transformer:
        return self._transforms[index]

    def append(self, transform: Transformer):
        self._transforms.append(transform)

    def extend(self, transforms: Iterable[Transformer]):
        self._transforms.extend(transforms)

    def insert(self, index: int, transform: Transformer):
        self._transforms.insert(index, transform)


class ComposeWithMode(Compose, TransformerMode):
    def set_train_mode(self) -> None:
        for transform in self._transforms:
            if isinstance(transform, TransformerMode):
                transform.set_train_mode()

    def set_eval_mode(self) -> None:
        for transform in self._transforms:
            if isinstance(transform, TransformerMode):
                transform.set_eval_mode()

    def is_train_mode(self) -> bool:
        return all(tf.is_train_mode() for tf in self._transforms if isinstance(tf, TransformerMode))

    def is_eval_mode(self) -> bool:
        return all(tf.is_eval_mode() for tf in self._transforms if isinstance(tf, TransformerMode))


class NoTransform(TransformerBase):
    """Does nothing."""

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return x


class ToTensor(TransformerBase):
    """Convert an input to a PyTorch tensor (dtype=float32)."""

    def __init__(self, device: Optional[torch.device] = None):
        self._device = device

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = np.ascontiguousarray(x)
            return torch.from_numpy(x).to(dtype=torch.float32, device=self._device)
        else:
            return x.to(dtype=torch.float32, device=self._device)


class DictTransformCreate:
    """Create named outputs from multiple transforms."""

    def __init__(self, transforms: Dict[str, Transformer]):
        self._transforms = transforms

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        return {key: tf(x) for key, tf in self._transforms.items()}


class DictTransformClone:
    """Clone a named input to a new named output. Performs a deep clone by default."""

    def __init__(self, source: str, clone: str, shallow: bool = False):
        self._source = source
        self._clone = clone
        self._shallow = shallow

    def __call__(self, x: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        if self._shallow:
            x[self._clone] = x[self._source]
        else:
            if isinstance(x[self._source], np.ndarray):
                x[self._clone] = np.copy(x[self._source])
            elif isinstance(x[self._source], torch.Tensor):
                x[self._clone] = torch.clone(x[self._source].detach())
            else:
                raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x[self._source])}")
        return x


class DictTransformApply:
    """Apply transform on one or multiple named inputs."""

    def __init__(self, keys: Union[str, Iterable[str]], transform: Transformer):
        self._keys = [keys] if isinstance(keys, str) else [key for key in keys]
        self._transform = transform

    def __call__(self, x: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        for key in self._keys:
            x[key] = self._transform(x[key])
        return x


class RandomChoices(TransformerWithMode):
    """Randomly chose one transform from a collection of transforms based on its probability."""

    def __init__(self, choices: Dict[Transformer, float]):
        super().__init__()
        self._transforms: Optional[Dict[int, Transformer]] = None
        if len(choices) > 0:
            self._transforms = {id(k): k for k in choices.keys()}
            # a-array for np.random.choice
            self._a = np.array(list(self._transforms.keys()))
            # normalize the sum of probabilities to 1.0
            alpha = 1.0 / sum(choices.values())
            # p-array for np.random.choice
            self._p = np.array(list(alpha * v for v in choices.values()))

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.is_eval_mode() or self._transforms is None:
            return x

        chosen_id = int(np.random.choice(self._a, p=self._p))
        return self._transforms[chosen_id](x)


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
            raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x)}")


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
            raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x)}")


class Pad(TransformerBase):
    """Pad the given input along given dimensions."""

    def __init__(self,
                 shape: Tuple[int, ...],
                 dim: Tuple[int, ...],
                 mode: Literal["constant", "reflect", "replicate", "circular"] = "constant",
                 value: Optional[float] = 0
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
            raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x)}")


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
                 permute: Optional[Tuple[int, ...]] = None
                 ) -> None:
        self._mean = mean
        self._std = std
        self._permute = permute

    def get_normalization(self, device: Optional[str | torch.device] = None) -> Normalization:
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
                raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x)}")

        x = (x - self._mean) / (self._std + 1e-12)

        if self._permute is not None:
            if isinstance(x, np.ndarray):
                x = np.transpose(x, inverse_permutation(self._permute))  # undo permute
            elif isinstance(x, torch.Tensor):
                x = torch.permute(x, inverse_permutation(self._permute))  # undo permute

        return x
