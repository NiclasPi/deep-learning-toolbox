from PIL.Image import Image, Resampling


def scale_image(
        *,
        image: Image,
        target_size: int | tuple[int, int],
) -> Image:
    """Scale an image to cover the target size while preserving aspect ratio.

    The image is scaled so that it completely covers the target width and height.
    If the image is smaller than the target size, it is scaled up so that the smaller side
    exactly matches the corresponding target dimension; if it is larger, it is scaled down so
    that the smaller side fits the target while preserving the original aspect ratio.

    Args:
        image: The input image.
        target_size:
            The desired target size. If a single integer is provided, both width and height
            are set to this value.

    Returns:
        A new image resized to cover the target size while maintaining aspect ratio.
    """
    if isinstance(target_size, int):
        target_width, target_height = target_size, target_size
    else:
        target_width, target_height = target_size

    # scale so smaller side fits target
    scale_factor = max(target_width / image.width, target_height / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)

    return image.resize((new_width, new_height), resample=Resampling.LANCZOS)
