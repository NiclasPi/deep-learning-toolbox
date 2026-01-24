from random import randint

from PIL.Image import Image


def calculate_crop_box(
    *, image_size: int | tuple[int, int], target_size: int | tuple[int, int], randomized_crop: bool = False
) -> tuple[int, int, int, int]:
    """Compute the crop box for the provided target size.

    Args:
        image_size: The size of the image.
        target_size: The size of the target image.
        randomized_crop: Return a random crop location. If False, return the center crop box.

    Returns:
        The crop box, a 4-tuple defining the left, upper, right, and lower pixel coordinate.
    """
    if isinstance(image_size, int):
        image_width, image_height = image_size, image_size
    else:
        image_width, image_height = image_size
    if isinstance(target_size, int):
        target_width, target_height = target_size, target_size
    else:
        target_width, target_height = target_size

    if randomized_crop:
        # random crop
        max_x = image_width - target_width
        max_y = image_height - target_height
        x = randint(0, max_x)
        y = randint(0, max_y)
    else:
        # center crop
        x = (image_width - target_width) // 2
        y = (image_height - target_height) // 2

    return x, y, x + target_width, y + target_height


def crop_image(*, image: Image, target_shape: int | tuple[int, int], randomized_crop: bool = False) -> Image:
    """Crop the given image and return the cropped image.

    Args:
        image: The input image.
        target_shape:
            The desired target size. If a single integer is provided, both width and height
            are set to this value.
        randomized_crop (bool): Whether to crop at a random location.

    Returns:
        A new image resized to cover the target size while maintaining aspect ratio.
    """
    crop_box = calculate_crop_box(image_size=image.size, target_size=target_shape, randomized_crop=randomized_crop)
    return image.crop(crop_box)
