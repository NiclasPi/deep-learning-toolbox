import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

Transformer = Callable[[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]]


class TransformerBase(ABC):
    @abstractmethod
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError()


class TransformerWithMode(TransformerBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._eval = False

    @abstractmethod
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError()

    def train_mode(self) -> None:
        self._eval = False

    def eval_mode(self) -> None:
        self._eval = True

    def is_eval(self) -> bool:
        return self._eval


class Compose(TransformerBase):
    """Compose multiple transforms into one callable class"""

    def __init__(self, transforms: List[Transformer]):
        self._transforms = transforms

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        for transform in self._transforms:
            x = transform(x)
        return x

    def __len__(self) -> int:
        return len(self._transforms)


class ComposeWithMode(Compose):
    def set_train_mode(self) -> None:
        for transform in self._transforms:
            if isinstance(transform, TransformerWithMode):
                transform.train_mode()

    def set_eval_mode(self) -> None:
        for transform in self._transforms:
            if isinstance(transform, TransformerWithMode):
                transform.eval_mode()


class NoTransform(TransformerBase):
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return x


class ToTensor(TransformerBase):
    """Convert an input to a PyTorch tensor (dtype=float32)"""

    def __init__(self, device: Optional[torch.device] = None):
        self._device = device

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype=torch.float32, device=self._device)
        else:
            return x.to(dtype=torch.float32, device=self._device)
