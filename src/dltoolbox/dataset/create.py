import logging
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any
from uuid import uuid4

import h5py
import numpy as np

from dltoolbox.multiprocess import get_multiprocess_config


class BatchProcessItemError(Exception):
    """Exception raised when processing an individual item in a batch fails."""

    def __init__(self, file_path: Path, original_exception: Exception):
        self.file_path = file_path
        self.original_exception = original_exception
        super().__init__(f"error processing file '{file_path!s}': {original_exception!s}")

    def __reduce__(self):
        # tells pickle how to reconstruct the object
        return self.__class__, (self.file_path, self.original_exception)


def process_batch(
        *,
        free_slots_queue: Queue,
        worker_results_queue: Queue,
        batch_id: int,
        sample_paths: list[Path],
        sample_shape: tuple[int, ...],
        sample_dtype: np.dtype,
        loader_func: Callable[[Path], np.ndarray],
) -> None:
    slot_id: str | None = None
    shared_memory: SharedMemory | None = None
    try:
        slot_id = free_slots_queue.get()  # blocks until a slot is free
        if slot_id is None:
            worker_results_queue.put(("cancel", batch_id, slot_id))
            return
        shared_memory = SharedMemory(name=f"create_dataset_{slot_id}")
        batch_array = np.ndarray(
            shape=(len(sample_paths), *sample_shape),
            dtype=sample_dtype,
            buffer=shared_memory.buf,
        )
        current_index = 0
        encountered_errors = []
        for sample_path in sample_paths:
            try:
                batch_array[current_index] = loader_func(sample_path)
                current_index += 1
            except Exception as error:
                encountered_errors.append(
                    BatchProcessItemError(sample_path, error)
                )
        worker_results_queue.put(("ok", batch_id, slot_id, current_index, encountered_errors))
    except Exception as error:
        tb = traceback.format_exc()
        worker_results_queue.put(("err", batch_id, slot_id, str(error), tb))
    finally:
        if shared_memory is not None:
            shared_memory.close()


def create_dataset(
        *,
        h5_file: h5py.File,
        sample_paths: list[Path],
        sample_shape: tuple[int, ...],
        sample_dtype: np.dtype,
        loader_func: Callable[[Path], np.ndarray],
        dataset_key: str = "data",
        h5_compression: str | None = None,
        h5_compression_opts: Any | None = None,
        max_memory: int | None = None,
        max_workers: int | None = None,
) -> tuple[list[Path], list[BatchProcessItemError]]:
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
        loader_func (Callable[[Path], np.ndarray]):
            Function that loads and optionally preprocesses a sample from disk.
        dataset_key (str, default="data"):
            The HDF5 dataset key (i.e., the internal path/name under which the dataset is created).
            Must be a valid HDF5 object name (no whitespace or unsupported characters).
        h5_compression (str, optional):
            Name of the compression filter to use when creating the HDF5 dataset
            (e.g., "gzip", "lzf"). If None, no compression is applied.
        h5_compression_opts (Any, optional):
            Additional options passed to the chosen compression filter.
        max_memory (int | None, optional):
            Maximum memory (in bytes) to use for batching samples during processing.
            If None, memory usage is not constrained.
        max_workers (int | None, optional):
            Maximum number of worker processes. If None, uses a system-dependent default.

    Returns:
        tuple[list[Path], list[BatchProcessItemError]]:
            A tuple containing:
            - list[Path]: The input `sample_paths` reordered to match the index order
              of samples stored in the resulting HDF5 dataset.
            - list[BatchProcessItemError]: Any errors encountered while processing
              individual items.
    """
    mp_cfg = get_multiprocess_config(max_available_memory=max_memory, max_workers=max_workers)

    num_samples = len(sample_paths)
    sample_bytes = np.empty(sample_shape, dtype=sample_dtype).nbytes
    batch_size = min(
        mp_cfg.max_memory_per_process // sample_bytes,  # hard memory per worker constraint
        num_samples // mp_cfg.max_processes  # even worker distribution
    )

    dataset = h5_file.create_dataset(
        name=dataset_key,
        shape=(num_samples, *sample_shape),
        dtype=sample_dtype,
        maxshape=(None, *sample_shape),  # allow resiting along axis 0
        compression=h5_compression,
        compression_opts=h5_compression_opts,
    )
    index = 0

    manager = Manager()
    free_slots_queue = manager.Queue()
    worker_results_queue = manager.Queue()

    # for every worker process we create a shared memory that fits a batch sized samples array
    slot_ids: list[str] = [uuid4().hex[:6] for _ in range(mp_cfg.max_processes)]
    for slot_id in slot_ids:
        shared_memory = SharedMemory(
            name=f"create_dataset_{slot_id}",
            create=True,
            size=batch_size * sample_bytes,
        )
        shared_memory.close()
        free_slots_queue.put(slot_id)

    # batch start indices
    batch_start_map: dict[int, int] = {
        b_id: b_start
        for b_id, b_start in enumerate(range(0, num_samples, batch_size))
    }
    # batch item errors
    batch_item_errors: list[BatchProcessItemError] = []
    # sample paths that match the exact order in which samples were written to the dataset
    sample_paths_in_order: list[Path] = []

    try:
        with ProcessPoolExecutor(max_workers=mp_cfg.max_processes) as executor:
            num_batches_submitted: int = 0
            num_batches_completed: int = 0

            for batch_id, batch_start in batch_start_map.items():
                executor.submit(
                    process_batch,
                    free_slots_queue=free_slots_queue,
                    worker_results_queue=worker_results_queue,
                    batch_id=batch_id,
                    sample_paths=sample_paths[batch_start:batch_start + batch_size],
                    sample_shape=sample_shape,
                    sample_dtype=sample_dtype,
                    loader_func=loader_func,
                )
                num_batches_submitted += 1

            try:
                while num_batches_completed < num_batches_submitted:
                    worker_result = worker_results_queue.get()
                    num_batches_completed += 1

                    batch_id: int
                    slot_id: str | None

                    # check the status of this result
                    if worker_result[0] == "err":
                        _, batch_id, slot_id, err_str, err_tb = worker_result
                        logging.error(f"worker completed with error: {err_str}\n{err_tb}")
                    elif worker_result[0] == "ok":
                        actual_count: int
                        item_errors: list[BatchProcessItemError]
                        _, batch_id, slot_id, actual_count, item_errors = worker_result

                        # extend the ordered sample paths list
                        batch_start_index = batch_start_map[batch_id]
                        batch_samples = sample_paths[batch_start_index:batch_start_index + batch_size]
                        for batch_sample in batch_samples:
                            if not any(
                                    batch_sample.samefile(item_error.file_path)
                                    for item_error in item_errors
                            ):
                                sample_paths_in_order.append(batch_sample)

                        # extend item errors
                        batch_item_errors.extend(item_errors)

                        # get the view into the shared memory for that slot
                        shared_memory = SharedMemory(name=f"create_dataset_{slot_id}")
                        batch_array = np.ndarray(
                            shape=(actual_count, *sample_shape),
                            dtype=sample_dtype,
                            buffer=shared_memory.buf,
                        )

                        # store the result
                        dataset.write_direct(batch_array, dest_sel=range(index, index + actual_count))
                        index += actual_count

                    # make this slot available for writing again
                    if slot_id is not None:
                        free_slots_queue.put(slot_id)
            finally:
                # send sentinel value to all workers to unblock them
                for _ in range(mp_cfg.max_processes):
                    free_slots_queue.put(None)
    finally:
        for slot_id in slot_ids:
            try:
                shared_memory = SharedMemory(name=f"create_dataset_{slot_id}")
                shared_memory.unlink()
            except Exception:
                pass
        manager.shutdown()

    # resize the dataset to its actual size
    dataset.resize(index, axis=0)

    return sample_paths_in_order, batch_item_errors
