import numpy as np
import soundfile as sf
from PIL import Image
from soxr import resample

from dltoolbox.errors import ImageTooSmallError
from dltoolbox.images.crop import crop_image
from dltoolbox.images.scale import scale_image


def read_audio(
        file_path: str,
        target_length: int,  # in samples
        target_sample_rate: int = 22050,
        load_offset: float = 0,  # in samples
        random_sample_slice: bool = False,
        dtype: np.dtype = np.int16,
) -> np.ndarray:
    """Load an audio file from disk and return it as a numpy array."""

    if dtype not in {np.int16, np.int32, np.float32, np.float64}:
        raise ValueError("dtype must be np.int16, np.int32, np.float32, or np.float64")

    # returns array of shape (samples, channels)
    audio_data, sample_rate = sf.read(
        file=file_path,
        start=load_offset,
        dtype=dtype,
        always_2d=True,
    )

    if sample_rate != target_sample_rate:
        # needs resampling
        audio_data = resample(audio_data, sample_rate, target_sample_rate)

    # transpose to shape (channels, samples)
    audio_data = np.transpose(audio_data, (1, 0))

    if audio_data.shape[0] > 1:
        # convert to mono by taking the left channel only
        audio_data = audio_data[0:1]  # index with 0:1 to keep the channel dimension

    # pad if more samples are requested than available
    audio_length = audio_data.shape[-1]
    if audio_length < target_length:
        # pad on both ends with zeros
        padl = (target_length - audio_length) // 2
        padr = (target_length - audio_length + 1) // 2
        assert audio_length + padl + padr == target_length
        audio_data = np.pad(audio_data, ((0, 0), (padl, padr)), "constant", constant_values=0)

    # resize to target size
    if random_sample_slice and audio_length > target_length:
        idx = np.random.randint(audio_length - target_length)
        audio_data = audio_data[:, idx: idx + target_length]
    else:
        audio_data = audio_data[:, :target_length]

    return audio_data


def read_image(
        file_path: str,
        target_size: int | tuple[int, int],
        *,
        randomized_crop: bool = False,
        scale_to_fit: bool = False,
) -> np.ndarray:
    """Load an image file from disk and return it as a numpy array.

    Args:
        file_path (str):
            Path to the image file
        target_size (int | tuple[int, int]):
            Target size as int (square) or tuple (width, height)
        randomized_crop (bool):
            If True, crop at random location. If False, center crop.
        scale_to_fit (bool):
            If True, scale image so smaller side fits target, then crop larger side.
            If False, crop directly from original image.

    Returns:
        Image as numpy array with shape (height, width, 3) and dtype uint8
    """
    image = Image.open(file_path).convert("RGB")

    if scale_to_fit:
        image = scale_image(image=image, target_size=target_size)

    if (
            image.width < (target_size[0] if isinstance(target_size, tuple) else target_size) or
            image.height < (target_size[1] if isinstance(target_size, tuple) else target_size)
    ):
        raise ImageTooSmallError(image.size, target_size)

    image = crop_image(image=image, target_shape=target_size, randomized_crop=randomized_crop)

    return np.array(image, dtype=np.uint8)
