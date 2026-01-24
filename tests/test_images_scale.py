import numpy as np
import pytest
from PIL import Image

from dltoolbox.images.scale import scale_image


@pytest.mark.parametrize("image_size", [(48, 48), (48, 64), (64, 48), (64, 64)])
def test_scale_cover(image_size: tuple[int, int]) -> None:
    array = np.zeros((*image_size, 3), dtype=np.uint8)
    image = Image.fromarray(array)

    scaled = scale_image(image=image, target_size=(64, 64))

    # assert scaled dimensions cover the target size
    assert scaled.width >= 64
    assert scaled.height >= 64

    # assert aspect ratio is preserved
    original_ratio = image.width / image.height
    scaled_ratio = scaled.width / scaled.height
    assert abs(original_ratio - scaled_ratio) < 0.01, (
        f"Aspect ratio changed: original {original_ratio}, scaled {scaled_ratio}"
    )


@pytest.mark.parametrize("image_size", [(80, 80), (72, 80), (80, 72), (64, 80), (80, 64)])
def test_scale_fit(image_size: tuple[int, int]) -> None:
    array = np.zeros((*image_size, 3), dtype=np.uint8)
    image = Image.fromarray(array)

    scaled = scale_image(image=image, target_size=(64, 64))

    # assert smaller side exactly fits target
    assert min(*scaled.size) == 64, f"Smaller side should fit target: got {min(*scaled.size)}, expected {64}"

    # assert aspect ratio is preserved
    original_ratio = image.width / image.height
    scaled_ratio = scaled.width / scaled.height
    assert abs(original_ratio - scaled_ratio) < 0.01, (
        f"Aspect ratio changed: original {original_ratio}, scaled {scaled_ratio}"
    )
