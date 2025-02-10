import numpy as np
import torch
import torchvision
from io import BytesIO
from math import sqrt
from PIL import Image
from typing import Tuple, Union
from scipy.ndimage import gaussian_filter

from dltoolbox.transforms.core import TransformerBase, TransformerWithMode
from dltoolbox.transforms._image_utils_np import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue
from dltoolbox.transforms._utils import make_slices


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
        cell_w = w // self._cols

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
        else:
            raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x_grid)}")


class RandomFlip(TransformerWithMode):
    """
    Flip dimensions with given probabilities.
    The probability for a dimension to be flipped can be set individually for every dimension and defaults to 0.5.
    """

    def __init__(self, dim: Tuple[int, ...], prob: Union[float, Tuple[float, ...]] = 0.5) -> None:
        super().__init__()
        self._dim = dim
        if isinstance(prob, tuple):
            if len(prob) != len(dim):
                raise ValueError("tuples for dimensions and probabilities must have same length")
        elif isinstance(prob, float):
            prob = tuple(prob for _ in range(len(dim)))
        self._prob = np.array(prob, dtype=np.float64)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.is_eval_mode():
            return x

        # uniform distribution over [0, 1)
        dim_chosen = np.random.rand(len(self._dim)) <= self._prob
        dim_to_flip = tuple(d for i, d in enumerate(self._dim) if dim_chosen[i])

        if isinstance(x, np.ndarray):
            return np.flip(x, dim_to_flip)
        elif isinstance(x, torch.Tensor):
            return torch.flip(x, dim_to_flip)
        else:
            raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x)}")


class RandomRotate90(TransformerWithMode):
    """
    Rotate the input along 2 dimensions by k*90 degrees with a random k = {0, 1, 2, 3}.
    """

    def __init__(self, dim: Tuple[int, int]):
        super().__init__()
        self._dim = dim

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.is_eval_mode():
            return x

        k = np.random.randint(0, 4)
        if isinstance(x, np.ndarray):
            return np.rot90(x, k, axes=self._dim)
        elif isinstance(x, torch.Tensor):
            return torch.rot90(x, k, dims=self._dim)
        else:
            raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x)}")


class RandomErasing(TransformerWithMode):
    """
    Erase a randomly selected region with the specified fill value.
    """

    def __init__(self,
                 dim: Tuple[int, ...],
                 scale: Tuple[float, float] = (0.02, 0.33),
                 ratio: Tuple[float, float] = (0.3, 3.3),
                 value: Union[int, float, bool, complex] = 0,
                 ) -> None:
        super().__init__()
        self._dim = dim
        self._scale = scale
        self._ratio = ratio
        self._value = value

    def _get_region_slices(self, shape: Tuple[int, ...]) -> Tuple[slice, ...]:
        scales = np.random.uniform(low=self._scale[0], high=self._scale[1], size=len(shape))

        # compute region shape
        region = tuple(int(shape[i] * scales[i]) for i in range(len(shape)))

        # compute valid start indices
        indices = tuple(np.random.randint(0, shape[i] - region[i] + 1) for i in range(len(shape)))

        # return slices for the regions
        return tuple(slice(indices[i], indices[i] + region[i]) for i in range(len(shape)))

    def _get_region_values(self, region: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(region, np.ndarray):
            return np.full_like(region, fill_value=self._value)
        elif isinstance(region, torch.Tensor):
            return torch.full_like(region, fill_value=self._value)
        else:
            raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(region)}")

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        # get the slices of the randomly selected region
        region_slices = self._get_region_slices(tuple(x.shape[d] for d in self._dim))
        # get a view into that region
        region_view = x[make_slices(x.shape, self._dim, region_slices)]
        # assign new values to the region
        region_view[:] = self._get_region_values(region_view)
        return x


class RandomNoise(TransformerWithMode):
    """
    Add random noise to the input data.
    The random noise is taken from a normal distribution with std=1, then normalized to [-1, 1], and finally scaled.
    """

    def __init__(self, dim: Tuple[int, ...], scale: float = 1.0) -> None:
        super().__init__()
        self._dim = dim
        self._scale = scale

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.is_eval_mode():
            return x

        size = tuple(x.shape[d] for d in self._dim)
        noise = np.random.normal(0, 1, size)  # mean=0, std=1, dtype=float64
        noise = np.interp(noise, (noise.min(), noise.max()), (-1, 1))  # normalize to [-1, 1]
        noise *= self._scale

        if isinstance(x, np.ndarray):
            if issubclass(x.dtype.type, np.integer):
                # array has integer dtype and needs careful boundary handling
                dtype_info = np.iinfo(x.dtype)
                return np.clip(x.astype(np.float64) + noise,
                               a_min=dtype_info.min, a_max=dtype_info.max).astype(x.dtype)
            else:
                return x + noise.astype(x.dtype)
        elif isinstance(x, torch.Tensor):
            if not torch.is_floating_point(x) and not torch.is_complex(x):
                # tensor has integer dtype and needs careful boundary handling
                dtype_info = np.iinfo(np.dtype(str(x.dtype).lstrip("torch.")))
                return torch.clamp(x.to(torch.float64) + torch.from_numpy(noise),
                                   min=dtype_info.min, max=dtype_info.max).to(x.dtype)
            else:
                return x + torch.from_numpy(noise).to(x.dtype)
        else:
            raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x)}")


class GaussianBlur(TransformerBase):
    """
    Apply Gaussian blur on two input dimensions.
    """

    def __init__(self, dim: Tuple[int, int], sigma: Union[float, Tuple[float, float]] = 1.0, radius: int = 4):
        self._dim = dim
        self._sigma = sigma
        self._radius = radius

    def _get_sigma(self) -> float:
        if isinstance(self._sigma, tuple):
            return np.random.uniform(*self._sigma)
        else:
            return self._sigma

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, np.ndarray):
            return gaussian_filter(x, sigma=self._get_sigma(), radius=self._radius, axes=self._dim)
        elif isinstance(x, torch.Tensor):
            raise NotImplementedError()
        else:
            raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x)}")


class JPEGCompression(TransformerWithMode):
    """
    Apply JPEG compression on the input image.
    Select a fixed quality between 0 (worst) and 95 (best), or a range between a minimum and maximum quality.
    A random quality within the given range is selected upon call.
    """

    def __init__(self, quality: Union[int, Tuple[int, int]] = 75):
        super().__init__()
        self._q_min, self._q_max = (quality, quality) if isinstance(quality, int) else quality

    def _get_quality(self) -> int:
        if self.is_eval_mode():
            # return the quality between min and max in eval mode
            return self._q_min + (self._q_max - self._q_min) // 2
        # select a random quality setting in the interval [min, max]
        return np.random.randint(low=self._q_min, high=self._q_max + 1)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = False
        if isinstance(x, torch.Tensor):
            is_tensor = True
            x = x.cpu().numpy()

        # TODO: accept leading dimensions (-> utility function for permutations)

        x = np.transpose(x, (1, 2, 0))  # from (3, H, W) to (H, W, 3)
        img = Image.fromarray(x, mode="RGB")
        out = BytesIO()
        img.save(out, format="JPEG", quality=self._get_quality(), subsampling=0)
        x = np.transpose(np.array(Image.open(out)), (2, 0, 1))  # back to (3, H, W)

        if is_tensor:
            x = torch.from_numpy(x)
        return x


class ColorJitter(TransformerWithMode):
    """
    Randomly change the brightness, contrast, saturation and hue of an image.
    """

    def __init__(self,
                 brightness: Union[float, Tuple[float, float]] = 0.0,
                 contrast: Union[float, Tuple[float, float]] = 0.0,
                 saturation: Union[float, Tuple[float, float]] = 0.0,
                 hue: Union[float, Tuple[float, float]] = 0.0,
                 ):
        super().__init__()
        self._tvt = torchvision.transforms.ColorJitter(brightness=brightness,
                                                       contrast=contrast,
                                                       saturation=saturation,
                                                       hue=hue)
        self._brightness = brightness
        self._contrast = contrast
        self._saturation = saturation
        self._hue = hue

    def _get_params(self) -> Tuple[np.ndarray, float, float, float, float]:
        # indices defines the order of augmentations
        indices = np.random.permutation(4)

        b = np.random.uniform(self._brightness[0], self._brightness[1]) \
            if isinstance(self._brightness, tuple) else self._brightness
        c = np.random.uniform(self._contrast[0], self._contrast[1]) \
            if isinstance(self._contrast, tuple) else self._contrast
        s = np.random.uniform(self._saturation[0], self._saturation[1]) \
            if isinstance(self._saturation, tuple) else self._saturation
        h = np.random.uniform(self._hue[0], self._hue[1]) if isinstance(self._hue, tuple) else self._hue

        return indices, b, c, s, h

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.is_eval_mode():
            return x

        if isinstance(x, torch.Tensor):
            # use the torchvision implementation of ColorJitter for tensors
            return self._tvt(x)
        else:
            indices, b, c, s, h = self._get_params()
            for index in indices:
                if index == 0:
                    x = adjust_brightness(x, b)
                elif index == 1:
                    x = adjust_contrast(x, c)
                elif index == 2:
                    x = adjust_saturation(x, s)
                elif index == 3:
                    x = adjust_hue(x, h)
            return x
