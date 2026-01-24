from typing import Literal, Tuple, Union

import numpy as np
import torch


def fft_radial_frequency_matrix(
    fft_data: Union[np.ndarray, torch.Tensor],
    fft_dim: Tuple[int, ...],
    is_centered: bool = False,  # frequencies have been zero centered (image mean is in the center)
) -> Union[np.ndarray, torch.Tensor]:
    radial_freq: Union[np.ndarray, torch.Tensor]
    if isinstance(fft_data, np.ndarray):
        freq_axes = [np.fft.fftfreq(fft_data.shape[d]) for d in fft_dim]
        if is_centered:
            freq_axes = [np.fft.fftshift(f) for f in freq_axes]
        freq_grids = np.meshgrid(*freq_axes, indexing="ij")
        radial_freq = np.sqrt(np.sum(np.stack(freq_grids, axis=0) ** 2, axis=0))
    elif isinstance(fft_data, torch.Tensor):
        freq_axes = [torch.fft.fftfreq(fft_data.shape[d], device=fft_data.device) for d in fft_dim]
        if is_centered:
            freq_axes = [torch.fft.fftshift(f) for f in freq_axes]
        freq_grids = torch.meshgrid(*freq_axes, indexing="ij")
        radial_freq = torch.sqrt(torch.sum(torch.stack(freq_grids, dim=0) ** 2, dim=0))
    else:
        raise ValueError("fft_data must be either a torch.Tensor or np.ndarray")

    # insert missing dimensions so that it can be multiplied with the FFT data
    new_shape = [1] * len(fft_data.shape)
    for d in fft_dim:
        new_shape[d] = fft_data.shape[d]
    radial_freq = radial_freq.reshape(new_shape)

    return radial_freq


def fft_get_maximum_radial_frequency(radial_matrix: Union[np.ndarray, torch.Tensor]) -> float:
    if isinstance(radial_matrix, np.ndarray):
        return np.max(radial_matrix)
    elif isinstance(radial_matrix, torch.Tensor):
        return torch.max(radial_matrix).item()
    else:
        raise ValueError("radial_matrix must be either a torch.Tensor or np.ndarray")


def fft_low_pass_filter_mask(
    *,
    radial_matrix: Union[np.ndarray, torch.Tensor],
    cutoff_freq: float,
    transition_width: float = 0.1,
    slope_type: Literal["linear", "gentle", "steep"] = "linear",
    slope_multiplier: float = 2.0,
) -> Union[np.ndarray, torch.Tensor]:
    """Returns filter mask for the low-pass filter."""

    upper_bound = cutoff_freq + transition_width  # no frequencies above the upper bound will be kept
    upper_bound *= fft_get_maximum_radial_frequency(radial_matrix)  # normalize frequency

    transition: Union[np.ndarray, torch.Tensor]
    if isinstance(radial_matrix, np.ndarray):
        transition = np.clip((upper_bound - radial_matrix) / (transition_width + 1e-8), 0, 1)
    elif isinstance(radial_matrix, torch.Tensor):
        transition = torch.clip((upper_bound - radial_matrix) / (transition_width + 1e-8), 0, 1)
    else:
        raise ValueError("radial_matrix must be either a torch.Tensor or np.ndarray")

    if slope_type == "gentle":
        transition = transition ** (1 / slope_multiplier)
    elif slope_type == "steep":
        transition = transition**slope_multiplier
    return transition


def fft_high_pass_filter_mask(
    *,
    radial_matrix: Union[np.ndarray, torch.Tensor],
    cutoff_freq: float,
    transition_width: float = 0.1,
    slope_type: Literal["linear", "gentle", "steep"] = "linear",
    slope_multiplier: float = 2.0,
) -> Union[np.ndarray, torch.Tensor]:
    """Returns filter mask for the high-pass filter."""

    lower_bound = cutoff_freq - transition_width  # no frequency below the lower bound will be kept
    lower_bound *= fft_get_maximum_radial_frequency(radial_matrix)  # normalize frequency

    transition: Union[np.ndarray, torch.Tensor]
    if isinstance(radial_matrix, np.ndarray):
        transition = np.clip((radial_matrix - lower_bound) / (transition_width + 1e-8), 0, 1)
    elif isinstance(radial_matrix, torch.Tensor):
        transition = torch.clip((radial_matrix - lower_bound) / (transition_width + 1e-8), 0, 1)
    else:
        raise ValueError("radial_matrix must be either a torch.Tensor or np.ndarray")

    if slope_type == "gentle":
        transition = transition ** (1 / slope_multiplier)
    elif slope_type == "steep":
        transition = transition**slope_multiplier
    return transition
