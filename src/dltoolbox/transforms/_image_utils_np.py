import cv2
import numpy as np
from typing import Literal

"""
Utility functions for image manipulation.
All input images are expected to be numpy arrays of shape (C, H, W), where C denotes the number of color channels,
H denotes the height and W denotes the width of the image. Permitted number of color channels is 1 (grayscale) or 3 (rgb).
Implementations use the OpenCV implementations.
"""


def assert_hsv_values(img: np.ndarray) -> None:
    h, s, v = np.split(img, 3, axis=0)
    if np.min(h) < 0 or np.max(h) > 360:
        raise ValueError("hue values must be between 0 and 360")
    if np.min(s) < 0 or np.max(s) > 1:
        raise ValueError("saturation values must be between 0 and 1")
    if np.min(v) < 0 or np.max(v) > 1:
        raise ValueError("brightness values must be between 0 and 1")


def assert_image(img: np.ndarray, colored: bool | None = None, mode: Literal["RGB", "HSV"] = "RGB") -> None:
    # check number of dimensions
    if img.ndim != 3:
        raise ValueError(f"input image should have exactly 3 dimensions, but got {img.ndim}")

    # check color channel dimension
    if colored is None and not (img.shape[0] == 1 or img.shape[0] == 3):
        raise ValueError(f"number of color channels should be 1 or 3, but got {img.shape[0]}")
    elif colored and img.shape[0] != 3:  # callee requests colored image
        raise ValueError(f"number of color channels should be 3, but got {img.shape[0]}")
    elif not colored and img.shape[0] != 1:  # callee requests grayscale image
        raise ValueError(f"number of color channels should be 1, but got {img.shape[0]}")

    # check image dtype
    if img.dtype == np.float32:
        if mode == "RGB" and (np.min(img) < 0 or np.max(img) > 1):
            raise ValueError("expected float32 image values in the range [0, 1]")
        elif mode == "HSV":
            assert_hsv_values(img)
    elif img.dtype != np.uint8:
        raise ValueError(f"input image dtype should be float32 or uint8, but got {img.dtype}")


def rgb2gray(img: np.ndarray) -> np.ndarray:
    assert_image(img, colored=True)
    return np.expand_dims(cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY), axis=0)


def rgb2hsv(img: np.ndarray) -> np.ndarray:
    assert_image(img, colored=True)
    return cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2HSV_FULL).transpose(2, 0, 1)


def hsv2rgb(img: np.ndarray) -> np.ndarray:
    assert_image(img, colored=True, mode="HSV")
    return cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_HSV2RGB_FULL).transpose(2, 0, 1)


def blend(img1: np.ndarray, img2: np.ndarray, ratio: float) -> np.ndarray:
    max_value: int | float
    if np.issubdtype(img1.dtype, np.integer):
        max_value = np.iinfo(img1.dtype).max
    elif np.issubdtype(img1.dtype, np.floating):
        max_value = 1
    else:
        raise ValueError(f"input images should have integer or floating dtype, but got {img1.dtype}")

    out = img1.astype(np.float64) * ratio + img2.astype(np.float64) * (1.0 - ratio)
    # TODO: clip to [0, 1] incorrect vor H values in HSV
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

    # convert to HSV
    img = rgb2hsv(img)
    h, s, v = np.split(img, 3, axis=0)

    # modify hue
    h = np.mod(h + (factor * 360), 180).astype(img.dtype)
    out = np.concatenate((h, s, v), axis=0)

    # convert back to RGB
    out = hsv2rgb(out)
    return out
