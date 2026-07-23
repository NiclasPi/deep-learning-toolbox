import numpy as np
import pytest
from PIL import Image

from dltoolbox.ioutils import read_image


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "img.png"
    array = np.zeros((48, 48, 3), dtype=np.uint8)
    array[16:32, 8:24] = 255
    Image.fromarray(array).save(path)
    return str(path)


def test_read_image_with_crop_origin_extracts_tile(image_path: str) -> None:
    result = read_image(image_path, target_size=(16, 16), crop_origin=(8, 16))

    assert result.shape == (16, 16, 3)
    assert np.array_equal(result, np.full((16, 16, 3), 255, dtype=np.uint8))


@pytest.mark.parametrize(
    "kwargs", [{"randomized_crop": True}, {"scale_to_fit": True}, {"randomized_crop": True, "scale_to_fit": True}]
)
def test_read_image_rejects_crop_origin_combined_with_other_modes(image_path: str, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="crop_origin cannot be combined"):
        read_image(image_path, target_size=(16, 16), crop_origin=(8, 16), **kwargs)
