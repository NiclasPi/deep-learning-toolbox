import numpy as np
import soundfile as sf
from PIL import Image
from soxr import resample


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
) -> np.ndarray:
    """Load an image file from disk and return it as a numpy array."""

    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    image = Image.open(file_path).convert("RGB")

    if image.width < target_size[0] or image.height < target_size[1]:
        raise RuntimeError(f"image '{file_path}' is smaller than the requested target size")

    fit_ratio = (image.size[0] / target_size[0], image.size[1] / target_size[1])
    if any(x > 1 for x in fit_ratio):
        # extract the largest possible box of the requested aspect ratio
        crop_size = (int(target_size[0] * min(fit_ratio)), int(target_size[1] * min(fit_ratio)))
        crop_box = (
            (image.size[0] - crop_size[0]) // 2,
            (image.size[1] - crop_size[1]) // 2,
            (image.size[0] + crop_size[0]) // 2,
            (image.size[1] + crop_size[1]) // 2
        )
        image = image.crop(crop_box)

        # resize image to requested size
        image = image.resize(target_size, resample=Image.Resampling.LANCZOS)

    return np.array(image, dtype=np.uint8)
