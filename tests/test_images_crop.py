import numpy as np
import pytest
from PIL import Image

from dltoolbox.images.crop import crop_image


@pytest.mark.parametrize("image_size", [(48, 48), (48, 64), (64, 48), (37, 48)])
def test_crop_center(image_size: tuple[int, int]) -> None:
    array = np.zeros((*image_size, 3), dtype=np.uint8)
    image = Image.fromarray(array)

    cropped = crop_image(image=image, target_shape=(32, 32))

    # assert scaled dimensions cover the target size
    assert cropped.width == 32
    assert cropped.height == 32
