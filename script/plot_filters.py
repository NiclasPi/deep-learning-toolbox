import argparse
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from dltoolbox.filters import fft_radial_frequency_matrix, fft_low_pass_filter_mask, fft_high_pass_filter_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fft-shift",
        action="store_true",
        help="Shift the zero-frequency component to the center of the spectrum."
    )
    parser.add_argument(
        "--cutoff-frequency",
        type=float,
        default=0.5,
        help="The filter's cutoff frequency normalized to [0.0, 1.0] (default: 0.5)."
    )
    parser.add_argument(
        "--slope-type",
        choices=["linear", "gentle", "steep"],
        default="linear",
        help="The type of slope to use (default: linear)."
    )
    args = parser.parse_args()

    # image is from this helpful resource: https://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm
    image = Image.open("../assets/son3.gif").convert("RGB")
    fft = np.fft.fftn(np.array(image), axes=(0, 1))
    if args.fft_shift:
        fft = np.fft.fftshift(fft, axes=(0, 1))
    fft_magnitudes = np.log(np.abs(fft) + 1e-12)
    fft_magnitudes = fft_magnitudes / np.max(fft_magnitudes)  # normalize to [0, 1]

    radius = fft_radial_frequency_matrix(fft, (0, 1), is_centered=args.fft_shift is True)

    mask_LP = fft_low_pass_filter_mask(
        radial_matrix=radius,
        cutoff_freq=args.cutoff_frequency,
        slope_type=args.slope_type,
    )
    mask_HP = fft_high_pass_filter_mask(
        radial_matrix=radius,
        cutoff_freq=args.cutoff_frequency,
        slope_type=args.slope_type,
    )

    filtered_LP = fft * mask_LP
    filtered_HP = fft * mask_HP

    reconstructed_LP = np.fft.ifftn(
        np.fft.ifftshift(filtered_LP) if args.fft_shift else filtered_LP,
        axes=(0, 1),
    ).real.astype(np.uint8)
    reconstructed_HP = np.fft.ifftn(
        np.fft.ifftshift(filtered_HP) if args.fft_shift else filtered_HP,
        axes=(0, 1),
    ).real.astype(np.uint8)

    # plot the results

    fig, axs = plt.subplots(3, 3, layout="compressed", figsize=(9, 9))
    axs[0, 0].set_title("Radial Frequency Matrix")
    axs[0, 0].imshow(radius)
    axs[0, 1].set_title("Frequency Magnitudes of the Image")
    axs[0, 1].imshow(fft_magnitudes)
    axs[0, 2].set_title("Original image")
    axs[0, 2].imshow(image)

    axs[1, 0].set_title("Filter Mask for low-pass filter")
    axs[1, 0].imshow(mask_LP)
    axs[1, 1].set_title("Frequency Magnitudes of the low-pass filtered Image")
    filtered_LP_magnitudes = np.log(np.abs(filtered_LP) + 1e-12)
    filtered_LP_magnitudes /= np.max(filtered_LP_magnitudes)
    axs[1, 1].imshow(filtered_LP_magnitudes)
    axs[1, 2].set_title("Reconstructed Image from low-pass filter")
    axs[1, 2].imshow(reconstructed_LP)

    axs[2, 0].set_title("Filter Mask for high-pass filter")
    axs[2, 0].imshow(mask_HP)
    axs[2, 1].set_title("Frequency Magnitudes of the high-pass filtered Image")
    filtered_HP_magnitudes = np.log(np.abs(filtered_HP) + 1e-12)
    filtered_HP_magnitudes /= np.max(filtered_HP_magnitudes)
    axs[2, 1].imshow(filtered_HP_magnitudes)
    axs[2, 2].set_title("Reconstructed Image from high-pass filter")
    axs[2, 2].imshow(reconstructed_HP)

    plt.show()
