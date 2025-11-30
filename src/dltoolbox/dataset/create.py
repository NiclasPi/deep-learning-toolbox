from collections.abc import Callable, Iterable
from concurrent.futures import as_completed, ProcessPoolExecutor
from itertools import batched
from pathlib import Path

import h5py
import numpy as np

from dltoolbox.multiprocess import get_multiprocess_config


def process_batch(
        sample_paths: Iterable[Path],
        loader_func: Callable[[Path], np.ndarray],
) -> tuple[np.ndarray, Iterable[Path]]:
    loaded_samples: list[np.ndarray] = []
    for file_path in sample_paths:
        loaded_samples.append(loader_func(file_path))
    return np.stack(loaded_samples, axis=0), sample_paths


def create_dataset(
        *,
        h5_file: h5py.File,
        sample_paths: list[Path],
        sample_shape: tuple[int, ...],
        sample_dtype: np.dtype,
        loader_func: Callable[[Path], np.ndarray],
        max_memory: int | None = None,
        max_workers: int | None = None,
) -> list[Path]:
    """Populate an HDF5 dataset in parallel from a list of sample file paths.

    This function creates and fills a dataset inside the given HDF5 file using a
    process pool for efficient parallel loading. Each sample is loaded with
    `loader_func` and written into the dataset.
    Because parallel execution does not guarantee a deterministic processing order,
    the function returns the list of `sample_paths` reordered to reflect the exact
    index position of each sample in the final HDF5 dataset.

    Args:
        h5_file (h5py.File): Open HDF5 file in which the dataset will be created.
        sample_paths (list[Path]): Paths to the raw samples to be loaded.
        sample_shape (tuple[int, ...]): Shape of each sample in the dataset.
        sample_dtype (np.dtype): NumPy dtype of the dataset samples.
        loader_func (Callable[[Path], np.ndarray]): Function that loads and
            optionally preprocesses a sample from disk.
        max_memory (int | None, optional): Maximum memory (in bytes) to use for
            batching samples during processing. If None, memory usage is not
            constrained.
        max_workers (int | None, optional): Maximum number of worker processes.
            If None, uses a system-dependent default.

    Returns:
        list[Path]: The input `sample_paths` reordered to match the index order
        of samples stored in the resulting HDF5 dataset.
    """
    mp_cfg = get_multiprocess_config(max_available_memory=max_memory, max_workers=max_workers)

    num_samples = len(sample_paths)
    sample_bytes = np.empty(sample_shape, dtype=sample_dtype).nbytes
    batch_size = min(
        mp_cfg.max_memory_per_process // sample_bytes,  # hard memory per worker constraint
        num_samples // mp_cfg.max_processes  # even worker distribution
    )

    dataset = h5_file.create_dataset("data", shape=(num_samples, *sample_shape), dtype=sample_dtype)
    index = 0

    samples_order: list[Path] = []

    with ProcessPoolExecutor(max_workers=mp_cfg.max_processes) as executor:
        futures = {}
        for batch_paths in batched(sample_paths, batch_size):
            future = executor.submit(process_batch, batch_paths, loader_func)
            futures[future] = id(future)

        while len(futures) > 0:
            for future in as_completed(futures):
                samples_data, samples_paths = future.result()

                # store the result
                dataset.write_direct(samples_data, dest_sel=range(index, index + samples_data.shape[0]))
                index += samples_data.shape[0]

                # append the stored samples to the order
                samples_order.extend(samples_paths)

                # remove the future
                del futures[future]

    return samples_order
