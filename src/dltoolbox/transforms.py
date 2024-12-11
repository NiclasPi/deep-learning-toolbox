import numpy as np
import torch
from abc import ABC, abstractmethod
from math import sqrt
from typing import Callable, Dict, Iterable, List, Optional, Self, Tuple, Union
from scipy.fft import dctn

from dltoolbox.normalization import WelfordEstimator
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
            if isinstance(transform, TransformerWithMode):
                transform.set_train_mode()

    def set_eval_mode(self) -> None:
        for transform in self._transforms:
            if isinstance(transform, TransformerWithMode):
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
    """Apply transform on one named input."""

    def __init__(self, key: str, transform: Transformer):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        x[self._key] = self._transform(x[self._key])
        return x


class Permute(TransformerBase):
    """Permute dimensions of the given input."""

    def __init__(self, dims: Tuple[int, ...]):
        self.dims = dims

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, np.ndarray):
            return np.transpose(x, self.dims)
        elif isinstance(x, torch.Tensor):
            return torch.permute(x, self.dims)
        else:
            raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x)}")


class Normalize(TransformerBase):
    """Normalize using the mean and standard deviation. Supports broadcasting to a common shape."""

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

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self._permute is not None:
            x = torch.permute(x, self._permute)

        x = (x - self._mean) / self._std

        if self._permute is not None:
            x = torch.permute(x, inverse_permutation(self._permute))  # undo permute

        return x


class RandomCrop(TransformerWithMode):
    """Crop the given image at a random location. The input is expected to have shape [..., H, W]."""

    def __init__(self, size: Tuple[int, int]) -> None:
        super().__init__()
        self._th, self._tw = size

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        h, w = x.shape[-2:]
        if h < self._th or w < self._tw:
            raise ValueError(f"crop size {(self._th, self._tw)} is larger than input image size {(h, w)}")
        if h == self._th and w == self._tw:
            return x

        if self.is_eval_mode():
            # perform a center crop in eval mode
            i = h - (self._th + 1) // 2
            j = w - (self._tw + 1) // 2
            return x[..., i:i + self._th, j:j + self._tw]
        else:
            i = np.random.randint(0, h - self._th + 1)
            j = np.random.randint(0, w - self._tw + 1)
            return x[..., i:i + self._th, j:j + self._tw]


class RandomPatchesInGrid(TransformerWithMode):
    """
    Extract patches from a grid of the given image. The input is expected to have shape [..., H, W].
    Creates a new dimension for the resulting patches [..., N, size[0], size[1]].
    """

    def __init__(self,
                 size: Union[int, Tuple[int, int]],  # path size (height, width)
                 grid: Union[int, Tuple[int, int]],  # grid layout (rows, columns)
                 ) -> None:
        super().__init__()
        self._th, self._tw = size if isinstance(size, tuple) else (size, size)
        self._rows, self._cols = self._compute_grid_layout(grid)

    @staticmethod
    def _compute_grid_layout(grid: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(grid, int):
            # start from the largest possible factor near the square root of n
            for rows in range(int(sqrt(grid)), 0, -1):
                if grid % rows == 0:
                    columns = grid // rows
                    return rows, columns
            return 1, grid  # fallback for cases where n = 1
        else:
            return grid

    def _make_grid(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        *other_dims, h, w = x.shape
        if h // self._th < self._rows or w // self._tw < self._cols:
            raise ValueError(f"input of shape {x.shape} cannot be reshaped into a {self._rows}x{self._cols} grid "
                             f"with cells of at least size ({self._th}, {self._tw})")

        # compute the cell height and width
        cell_h = h // self._rows
        cell_w = h // self._cols

        # compute trimmed size to fit the grid
        trimmed_h = self._rows * cell_h
        trimmed_w = self._cols * cell_w
        x = x[..., :trimmed_h, :trimmed_w]

        # reshape into (..., rows, cols, cell_height, cell_width)
        return (x[..., :self._rows * cell_h, :self._cols * cell_w]
                .reshape(*other_dims, self._rows, cell_h, self._cols, cell_w)
                .swapaxes(-3, -2))

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        x_grid = self._make_grid(x)

        x_patches = []
        for row in range(self._rows):
            for col in range(self._cols):
                cell_ij = x_grid[..., row, col, :, :]
                cell_h, cell_w = cell_ij.shape[-2:]

                if self.is_eval_mode():
                    # select the center in eval mode
                    i = (cell_h - self._th + 1) // 2
                    j = (cell_w - self._tw + 1) // 2
                    x_patches.append(cell_ij[..., i:i + self._th, j:j + self._tw])
                else:
                    # select a random crop of the grid cell
                    i = np.random.randint(0, cell_h - self._th + 1)
                    j = np.random.randint(0, cell_w - self._tw + 1)
                    x_patches.append(cell_ij[..., i:i + self._th, j:j + self._tw])

        if isinstance(x_grid, np.ndarray):
            return np.stack(x_patches, axis=-3)
        elif isinstance(x_grid, torch.Tensor):
            return torch.stack(x_patches, dim=-3)


class FFT(TransformerBase):
    """Compute the Discrete Fourier Transform on k input dimensions using the Fast Fourier Transform (FFT) algorithm."""

    def __init__(self,
                 dim: Tuple[int, ...],
                 log: bool = True,
                 eps: float = 1e-12
                 ) -> None:
        self.dim = dim
        self.log = log
        self.eps = eps

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, np.ndarray):
            fft = np.fft.fftn(x, axes=self.dim)
            if self.log:
                fft = np.log(np.abs(fft) + self.eps)
            else:
                fft = np.concatenate([np.real(fft), np.imag(fft)], axis=-1)
            return fft
        elif isinstance(x, torch.Tensor):
            fft = torch.fft.fftn(x, dim=self.dim)
            if self.log:
                fft = torch.log(torch.abs(fft) + self.eps)
            else:
                fft = torch.cat([torch.real(fft), torch.imag(fft)], dim=-1)
            return fft
        else:
            raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x)}")


class DCT(TransformerBase):
    """Compute the Discrete Cosine Transform (DCT) on k input dimensions."""

    def __init__(self,
                 dim: Tuple[int, ...],
                 log: bool = True,
                 eps: float = 1e-12
                 ) -> None:
        self.dim = dim
        self.log = log
        self.eps = eps

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, np.ndarray):
            dct = dctn(x, axis=self.dim)
            if self.log:
                dct = np.log(np.abs(dct) + self.eps)
            return dct
        elif isinstance(x, torch.Tensor):
            raise NotImplemented("torch-based DCT not implemented yet")  # TODO
        else:
            raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x)}")
