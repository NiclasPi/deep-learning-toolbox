import numpy as np
import torch
from abc import ABC, abstractmethod
from io import BytesIO
from math import sqrt
from PIL import Image
from typing import Callable, Dict, Iterable, List, Optional, Self, Tuple, Union
from scipy.fft import dctn
from scipy.ndimage import convolve, gaussian_filter

from .core import TransformerBase


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
            dct = dctn(x, axes=self.dim)
            if self.log:
                dct = np.log(np.abs(dct) + self.eps)
            return dct
        elif isinstance(x, torch.Tensor):
            raise NotImplemented("torch-based DCT not implemented yet")  # TODO
        else:
            raise ValueError(f"expected torch.Tensor or np.ndarray, got {type(x)}")
