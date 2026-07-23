import numpy as np
import pytest
from PIL import Image

from dltoolbox.images.crop import calculate_crop_box, crop_image


@pytest.mark.parametrize("image_size", [(48, 48), (48, 64), (64, 48), (37, 48)])
def test_crop_center(image_size: tuple[int, int]) -> None:
    array = np.zeros((*image_size, 3), dtype=np.uint8)
    image = Image.fromarray(array)

    cropped = crop_image(image=image, target_shape=(32, 32))

    # assert scaled dimensions cover the target size
    assert cropped.width == 32
    assert cropped.height == 32


def test_calculate_crop_box_origin_anchors_box() -> None:
    box = calculate_crop_box(image_size=(64, 48), target_size=(32, 16), crop_origin=(10, 5))
    assert box == (10, 5, 42, 21)


def test_calculate_crop_box_origin_takes_precedence_over_randomized() -> None:
    # crop_origin wins even when randomized_crop is requested
    box = calculate_crop_box(image_size=(64, 48), target_size=(32, 16), randomized_crop=True, crop_origin=(0, 0))
    assert box == (0, 0, 32, 16)


@pytest.mark.parametrize("crop_origin", [(-1, 0), (0, -1), (33, 0), (0, 33)])
def test_calculate_crop_box_origin_out_of_bounds(crop_origin: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="exceeds image bounds"):
        calculate_crop_box(image_size=(64, 64), target_size=(32, 32), crop_origin=crop_origin)


def test_crop_image_with_origin_extracts_tile() -> None:
    # distinct value per row so we can verify the exact crop location
    array = np.zeros((48, 48, 3), dtype=np.uint8)
    array[16:32, 8:24] = 255
    image = Image.fromarray(array)

    cropped = crop_image(image=image, target_shape=(16, 16), crop_origin=(8, 16))

    assert cropped.size == (16, 16)
    assert np.array_equal(np.array(cropped), np.full((16, 16, 3), 255, dtype=np.uint8))
