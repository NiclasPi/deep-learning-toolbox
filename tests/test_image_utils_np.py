from functools import partial
from typing import Callable

import numpy as np
import pytest
import torch

from dltoolbox.utils import get_tolerances


def create_image(dtype: np.dtype) -> np.ndarray:
    """Returns values in [0, 1) for a floating point dtype and values in [0, int_max] for any integer dtype."""

    img = np.random.rand(3, 32, 32)
    if np.issubdtype(dtype, np.integer):
        out_max = float(np.iinfo(dtype).max)
        return (img * (out_max + 1 - 1e-3)).astype(dtype)
    else:
        return img.astype(dtype)


def assert_with_torchvision(img: np.ndarray, out: np.ndarray, func: Callable[[...], torch.Tensor]) -> bool:
    x = torch.from_numpy(img)
    y = func(x)
    # TODO: be more tolerant for uint8: allow unit differences
    return np.allclose(y.numpy(), out, *get_tolerances(y.dtype))


class TestImageUtilsNp:
    @pytest.mark.parametrize("dtype", [np.float64, np.float32, np.uint16, np.uint8])
    def test_rgb2gray(self, dtype: np.dtype) -> None:
        from dltoolbox.transforms._image_utils_np import rgb2gray

        img = create_image(dtype)

        if dtype == np.float64 or dtype == np.uint16:
            with pytest.raises(ValueError):
                rgb2gray(img)
        else:
            out = rgb2gray(img)
            assert out.shape[0] == 1  # reduced to one color channel
            assert out.dtype == img.dtype
            from torchvision.transforms.functional import rgb_to_grayscale as rgb_to_grayscale_pt

            assert assert_with_torchvision(img, out, partial(rgb_to_grayscale_pt))

    @pytest.mark.parametrize("dtype", [np.float64, np.float32, np.uint16, np.uint8])
    def test_rgb2hsv(self, dtype: np.dtype) -> None:
        from dltoolbox.transforms._image_utils_np import rgb2hsv

        img = create_image(dtype)

        if dtype == np.float64 or dtype == np.uint16:
            with pytest.raises(ValueError):
                rgb2hsv(img)
        else:
            out = rgb2hsv(img)
            assert out.shape[0] == 3
            assert out.dtype == img.dtype
            from torchvision.transforms._functional_tensor import _rgb2hsv as rgb_to_hsv_pt

            assert assert_with_torchvision(img, out, partial(rgb_to_hsv_pt))

    @pytest.mark.parametrize("dtype", [np.float64, np.float32, np.uint16, np.uint8])
    def test_hsv2rgb(self, dtype: np.dtype) -> None:
        from dltoolbox.transforms._image_utils_np import hsv2rgb

        img = create_image(dtype)

        if dtype == np.float64 or dtype == np.uint16:
            with pytest.raises(ValueError):
                hsv2rgb(img)
        else:
            out = hsv2rgb(img)
            assert out.shape[0] == 3
            assert out.dtype == img.dtype
            from torchvision.transforms._functional_tensor import _hsv2rgb as hsv_to_rgb_pt

            assert assert_with_torchvision(img, out, partial(hsv_to_rgb_pt))

    @pytest.mark.parametrize("dtype", [np.float64, np.float32, np.uint16, np.uint8])
    def test_rgb2hsv2rgb(self, dtype: np.dtype) -> None:
        from dltoolbox.transforms._image_utils_np import hsv2rgb, rgb2hsv

        img = create_image(dtype)

        if dtype == np.float64 or dtype == np.uint16:
            with pytest.raises(ValueError):
                hsv2rgb(rgb2hsv(img))
        else:
            out = hsv2rgb(rgb2hsv(img))
            assert np.allclose(img, out, *get_tolerances(dtype))

    @pytest.mark.parametrize("dtype", [np.float32, np.uint8])
    def test_blend(self, dtype: np.dtype) -> None:
        from dltoolbox.transforms._image_utils_np import blend

        max_value = 255 if dtype == np.uint8 else 1
        img1 = np.zeros((3, 32, 32), dtype=dtype)
        img2 = np.full_like(img1, fill_value=max_value)

        out = blend(img1, img2, 0.5)
        assert out.shape == img1.shape
        assert out.dtype == img1.dtype
        assert np.allclose(out, np.full_like(out, fill_value=0.5 * max_value))

    @pytest.mark.parametrize("dtype", [np.float32, np.uint8])
    def test_adjust_brightness(self, dtype: np.dtype) -> None:
        from dltoolbox.transforms._image_utils_np import adjust_brightness

        img = create_image(dtype)
        out = adjust_brightness(img, 0.5)
        assert out.shape == img.shape
        assert out.dtype == img.dtype
        from torchvision.transforms.functional import adjust_brightness as adjust_brightness_pt

        assert assert_with_torchvision(img, out, partial(adjust_brightness_pt, brightness_factor=0.5))

    @pytest.mark.parametrize("dtype", [np.float32, np.uint8])
    def test_adjust_contrast(self, dtype: np.dtype) -> None:
        from dltoolbox.transforms._image_utils_np import adjust_contrast

        img = create_image(dtype)
        out = adjust_contrast(img, 0.5)
        assert out.shape == img.shape
        assert out.dtype == img.dtype
        from torchvision.transforms.functional import adjust_contrast as adjust_contrast_pt

        assert assert_with_torchvision(img, out, partial(adjust_contrast_pt, contrast_factor=0.5))

    @pytest.mark.parametrize("dtype", [np.float32, np.uint8])
    def test_adjust_saturation(self, dtype: np.dtype) -> None:
        from dltoolbox.transforms._image_utils_np import adjust_saturation

        img = create_image(dtype)
        out = adjust_saturation(img, 0.5)
        assert out.shape == img.shape
        assert out.dtype == img.dtype
        from torchvision.transforms.functional import adjust_saturation as adjust_saturation_pt

        assert assert_with_torchvision(img, out, partial(adjust_saturation_pt, saturation_factor=0.5))

    @pytest.mark.parametrize("dtype", [np.float32, np.uint8])
    def test_adjust_hue(self, dtype: np.dtype) -> None:
        from dltoolbox.transforms._image_utils_np import adjust_hue

        img = create_image(dtype)
        out = adjust_hue(img, 0.5)
        assert out.shape == img.shape
        assert out.dtype == img.dtype
        from torchvision.transforms.functional import adjust_hue as adjust_hue_pt

        assert_with_torchvision(img, out, partial(adjust_hue_pt, hue_factor=0.5))
