import numpy as np
from typing import Optional

"""
Utility functions for image manipulation.
All input images are expected to be numpy arrays of shape (..., C, H, W), where C denotes the number of color channels,
H denotes the height and W denotes the width of the image. Permitted number of color channels is 1 (grayscale) or 3 (rgb).
All operations are performed on the last three dimensions of the input.
Implementations closely follow the torchvision implementations.
"""


def assert_input_shape(img: np.ndarray) -> None:
    if img.ndim < 3:
        raise ValueError(f"input image array should have at least 3 dimensions, found {img.ndim}")
    if not (img.shape[-3] == 1 or img.shape[-3] == 3):
        raise ValueError(f"input image number of color channels should be 1 or 3, found {img.shape[0]}")


def convert_image_dtype(img: np.ndarray, dtype: np.dtype) -> np.ndarray:
    # int to float
    if np.issubdtype(img.dtype, np.integer) and np.issubdtype(dtype, np.floating):
        img_max = float(np.iinfo(img.dtype).max)
        return img.astype(dtype) / img_max
    # float to int
    elif np.issubdtype(img.dtype, np.floating) and np.issubdtype(dtype, np.integer):
        out_max = float(np.iinfo(dtype).max)
        return (img * (out_max + 1 - 1e-3)).astype(dtype)
    else:
        return img.astype(dtype, casting="same_kind")


def rgb2gray(img: np.ndarray) -> np.ndarray:
    # implementation closely follows PyTorch's torchvision implementation:
    # https://github.com/pytorch/vision/blob/release/2.0/torchvision/transforms/_functional_tensor.py#L146

    if img.shape[-3] == 3:
        out: np.ndarray = (0.2989 * img[..., -3, :, :] + 0.587 * img[..., -2, :, :] + 0.114 * img[..., -1, :, :])
        return np.stack((out, out, out), axis=-3, dtype=img.dtype, casting="unsafe")
    else:
        return img.copy()


def rgb2hsv(img: np.ndarray) -> np.ndarray:
    # implementation closely follows PyTorch's torchvision implementation:
    # https://github.com/pytorch/vision/blob/release/2.0/torchvision/transforms/_functional_tensor.py#L262

    if not img.shape[-3] == 3:
        raise ValueError(f"input image should have 3 color channels, found {img.shape[-3]}")

    if np.issubdtype(img.dtype, np.integer):
        img = convert_image_dtype(img, np.float32)

    r = img[..., -3, :, :]
    g = img[..., -2, :, :]
    b = img[..., -1, :, :]

    maxc = np.max(img, axis=-3)
    minc = np.min(img, axis=-3)

    eqc = maxc == minc
    cr = maxc - minc
    ones = np.ones_like(maxc)
    s = cr / np.where(eqc, ones, maxc)
    cr_divisor = np.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = np.fmod((h / 6.0 + 1.0), 1.0)
    return np.stack((h, s, maxc), axis=-3)


def hsv2rgb(img: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
    # implementation closely follows PyTorch's torchvision implementation:
    # https://github.com/pytorch/vision/blob/release/2.0/torchvision/transforms/_functional_tensor.py#L301

    if not img.shape[-3] == 3:
        raise ValueError(f"dim(-3) is expected to have the HSV channels, found {img.shape[-3]} != 3")

    h = img[..., -3, :, :]
    s = img[..., -2, :, :]
    v = img[..., -1, :, :]

    i = np.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.astype(np.int32)

    p = np.clip((v * (1.0 - s)), 0.0, 1.0)
    q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
    t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i[..., np.newaxis, :, :] == np.arange(6)[:, np.newaxis, np.newaxis]

    a1 = np.stack((v, q, p, p, t, v), axis=-3)
    a2 = np.stack((t, v, v, q, p, p), axis=-3)
    a3 = np.stack((p, p, t, v, v, q), axis=-3)
    a4 = np.stack((a1, a2, a3), axis=-4)

    out = np.einsum("...ijk, ...xijk -> ...xjk", mask, a4)
    return convert_image_dtype(out, dtype) if dtype is not None else out


def blend(img1: np.ndarray, img2: np.ndarray, ratio: float) -> np.ndarray:
    max_value: int | float
    if np.issubdtype(img1.dtype, np.integer):
        max_value = np.iinfo(img1.dtype).max
    elif np.issubdtype(img1.dtype, np.floating):
        max_value = np.finfo(img1.dtype).max
    else:
        raise ValueError(f"input images should have integer or floating dtype, got {img1.dtype}")

    out = img1.astype(np.float64) * ratio + img2.astype(np.float64) * (1.0 - ratio)
    return np.clip(out, 0, max_value).astype(img1.dtype)


def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return blend(img, np.zeros_like(img), factor)


def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    mean = np.mean(rgb2gray(img) if img.shape[-3] == 3 else img, axis=(-3, -2, -1), keepdims=True)
    return blend(img, mean, factor)


def adjust_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    return blend(img, rgb2gray(img), factor)


def adjust_hue(img: np.ndarray, factor: float) -> np.ndarray:
    if not (-0.5 <= factor <= 0.5):
        raise ValueError("hue factor is not in [-0.5 and 0.5]")

    # save original dtype
    dtype = img.dtype

    # convert to HSV (implicitly casts to float32)
    img = rgb2hsv(img)

    h = img[..., -3, :, :]
    s = img[..., -2, :, :]
    v = img[..., -1, :, :]

    h = (h + factor) % 1.0
    out = np.stack((h, s, v), axis=-3)

    # convert back to RGB (and to its original dtype)
    out = hsv2rgb(out, dtype)
    return out
