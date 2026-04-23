import json
import logging
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import h5py
import numpy as np

from dltoolbox.dataset.errors import BatchProcessItemError, SampleMetaPairingError, SampleSequenceLengthMismatchError
from dltoolbox.dataset.metadata.dataset_metadata import DatasetMetadata
from dltoolbox.dataset.metadata.sample_meta_protocols import SampleMetaEncoder
from dltoolbox.multiprocess import get_multiprocess_config


def _default_json_encoder(obj: Any) -> bytes:
    return json.dumps(obj).encode("utf-8")


def _prepare_user_block(user_block: DatasetMetadata | bytes | None) -> tuple[int, bytes | None]:
    if user_block is None:
        return 0, None
    ub_bytes = user_block.to_json_bytes() if isinstance(user_block, DatasetMetadata) else user_block
    ub_size = max(512, int(2 ** np.ceil(np.log2(len(ub_bytes)))))
    return ub_size, ub_bytes


def create_dataset_from_arrays(
    output_path: str,
    *,
    data: np.ndarray,
    labels: np.ndarray | None = None,
    sample_ids: Sequence[str] | None = None,
    sample_meta: Sequence[Any] | None = None,
    sample_meta_encoder: SampleMetaEncoder[Any] | None = None,
    user_block: DatasetMetadata | bytes | None = None,
    data_key: str = "data",
    labels_key: str = "labels",
    sample_ids_key: str = "metadata/sample_ids",
    sample_meta_key: str = "metadata/sample_meta",
    h5_chunk_length: int | None = None,
    h5_compression: str | None = None,
    h5_compression_opts: Any | None = None,
) -> None:
    """Write an HDF5 dataset file from in-memory arrays.

    Produces the same file layout as `create_dataset_from_paths` (same dataset
    keys, same userblock semantics, same sample-metadata encoding), but takes
    the sample array directly instead of loading from disk. All per-sample
    inputs (`labels`, `sample_ids`, `sample_meta`) must be index-matched with
    `data` along the first axis and are written in that order.

    Use for when the full dataset already fits in memory. For parallel loading
    from a list of source files, use `create_dataset_from_paths`.

    Args:
        output_path (str): Path where the HDF5 file will be written.
        data (np.ndarray): Sample array to write. The first axis is the sample
            dimension; remaining axes define the per-sample shape.
        labels (np.ndarray, optional):
            Per-sample labels, index-matched with `data` along axis 0.
        sample_ids (Sequence[str], optional):
            Per-sample string ids, index-matched with `data`. Must be provided
            together with `sample_meta`.
        sample_meta (Sequence[Any], optional):
            Per-sample metadata payloads, index-matched with `data`. Must be
            provided together with `sample_ids`.
        sample_meta_encoder (SampleMetaEncoder, optional):
            Callable converting each `sample_meta` entry to bytes. Defaults to
            a JSON encoder for JSON-native payloads (dict, list, primitives).
        user_block (DatasetMetadata | bytes, optional):
            Bytes to be written into the HDF5 userblock at the start of the file.
        data_key (str, default="data"):
            The HDF5 dataset key for the sample array.
        labels_key (str, default="labels"):
            The HDF5 dataset key for the labels array.
        sample_ids_key (str, default="metadata/sample_ids"):
            The HDF5 dataset key for sample ids.
        sample_meta_key (str, default="metadata/sample_meta"):
            The HDF5 dataset key for sample metadata.
        h5_chunk_length (int, optional):
            Write the data dataset in chunks of shape
            (chunk_length, *data.shape[1:]) along the first dimension; if None,
            the dataset is stored contiguously without chunking. Chunking can
            improve I/O performance and enable efficient partial reads.
        h5_compression (str, optional):
            Name of the compression filter to use when creating the data dataset
            (e.g., "gzip", "lzf"). If None, no compression is applied.
        h5_compression_opts (Any, optional):
            Additional options passed to the chosen compression filter.
    """
    if (sample_ids is None) != (sample_meta is None):
        raise SampleMetaPairingError()
    if sample_ids is not None and len(sample_ids) != len(sample_meta):
        raise SampleSequenceLengthMismatchError("sample_ids", len(sample_ids), "sample_meta", len(sample_meta))

    ub_size, ub_bytes = _prepare_user_block(user_block)

    with h5py.File(output_path, "w-", userblock_size=ub_size, libver="latest") as h5_file:
        h5_file.create_dataset(
            data_key,
            data=data,
            chunks=(h5_chunk_length, *data.shape[1:]) if h5_chunk_length is not None else None,
            compression=h5_compression,
            compression_opts=h5_compression_opts,
        )
        if labels is not None:
            h5_file.create_dataset(labels_key, data=labels)

        if sample_ids is not None:
            # default encoder handles JSON-native payloads (dict, list, primitives);
            # callers with typed payloads (e.g. dataclasses) supply their own SampleMetaEncoder
            encoder = sample_meta_encoder if sample_meta_encoder is not None else _default_json_encoder
            h5_file.create_dataset(sample_ids_key, data=list(sample_ids), dtype=h5py.string_dtype())
            h5_file.create_dataset(sample_meta_key, data=[encoder(m) for m in sample_meta], dtype=h5py.string_dtype())

    if ub_size > 0 and ub_bytes:
        with open(output_path, "br+") as h5_file:
            h5_file.write(ub_bytes)


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
        batch_array = np.ndarray(shape=(len(sample_paths), *sample_shape), dtype=sample_dtype, buffer=shared_memory.buf)
        current_index = 0
        encountered_errors = []
        for sample_path in sample_paths:
            try:
                batch_array[current_index] = loader_func(sample_path)
                current_index += 1
            except Exception as error:
                encountered_errors.append(BatchProcessItemError(sample_path, error))
        worker_results_queue.put(("ok", batch_id, slot_id, current_index, encountered_errors))
    except Exception as error:
        tb = traceback.format_exc()
        worker_results_queue.put(("err", batch_id, slot_id, str(error), tb))
    finally:
        if shared_memory is not None:
            shared_memory.close()


def create_dataset_from_paths(
    output_path: str,
    *,
    sample_paths: Sequence[Path],
    loader_func: Callable[[Path], np.ndarray],
    sample_shape: tuple[int, ...],
    sample_dtype: np.dtype,
    labels: np.ndarray | None = None,
    sample_ids: Sequence[str] | None = None,
    sample_meta: Sequence[Any] | None = None,
    sample_meta_encoder: SampleMetaEncoder[Any] | None = None,
    user_block: DatasetMetadata | bytes | None = None,
    data_key: str = "data",
    labels_key: str = "labels",
    sample_ids_key: str = "metadata/sample_ids",
    sample_meta_key: str = "metadata/sample_meta",
    h5_chunk_length: int | None = None,
    h5_compression: str | None = None,
    h5_compression_opts: Any | None = None,
    max_memory: int | None = None,
    max_workers: int | None = None,
) -> list[BatchProcessItemError]:
    """Build an HDF5 dataset file by loading samples from disk in parallel.

    Uses a process pool to apply `loader_func` to each path in `sample_paths`,
    writing the resulting arrays into the `data_key` dataset of a newly created
    HDF5 file. Because parallel execution does not preserve input order, any
    per-sample inputs (`labels`, `sample_ids`, `sample_meta`) are reordered
    internally to match the order in which samples land in the file; callers
    pass these as sequences index-matched with `sample_paths`.

    Samples whose `loader_func` raises are skipped and reported in the returned
    error list; the final dataset is resized to the number of successfully
    loaded samples.

    Use for when the source data lives on disk and is too large to fit into
    memory at once, or when loading many samples benefits from parallelism.
    For data already in memory, use `create_dataset_from_arrays`.

    Args:
        output_path (str): Path where the HDF5 file will be written.
        sample_paths (Sequence[Path]): Paths to the raw samples to be loaded.
        loader_func (Callable[[Path], np.ndarray]):
            Function that loads and optionally preprocesses a sample from disk.
        sample_shape (tuple[int, ...]): Shape of each sample in the dataset.
        sample_dtype (np.dtype): NumPy dtype of the dataset samples.
        labels (np.ndarray, optional):
            Per-sample labels, index-aligned with `sample_paths` along axis 0.
            Reordered internally to match the final on-disk sample order.
        sample_ids (Sequence[str], optional):
            Per-sample string ids, index-aligned with `sample_paths`. Must be
            provided together with `sample_meta`.
        sample_meta (Sequence[Any], optional):
            Per-sample metadata payloads, index-aligned with `sample_paths`.
            Must be provided together with `sample_ids`.
        sample_meta_encoder (SampleMetaEncoder, optional):
            Callable converting each `sample_meta` entry to bytes. Defaults to
            a JSON encoder for JSON-native payloads (dict, list, primitives).
        user_block (DatasetMetadata | bytes, optional):
            Bytes to be written into the HDF5 userblock at the start of the file.
        data_key (str, default="data"):
            The HDF5 dataset key for the sample array.
        labels_key (str, default="labels"):
            The HDF5 dataset key for the labels array.
        sample_ids_key (str, default="metadata/sample_ids"):
            The HDF5 dataset key for sample ids.
        sample_meta_key (str, default="metadata/sample_meta"):
            The HDF5 dataset key for sample metadata.
        h5_chunk_length (int, optional):
            Write the data dataset in chunks of shape (chunk_length, *sample_shape)
            along the first dimension; if None, the dataset is stored contiguously
            without chunking. Chunking can improve I/O performance and enable
            efficient partial reads.
        h5_compression (str, optional):
            Name of the compression filter to use when creating the data dataset
            (e.g., "gzip", "lzf"). If None, no compression is applied.
        h5_compression_opts (Any, optional):
            Additional options passed to the chosen compression filter.
        max_memory (int | None, optional):
            Maximum memory (in bytes) to use for batching samples during processing.
            If None, memory usage is not constrained.
        max_workers (int | None, optional):
            Maximum number of worker processes. If None, uses a system-dependent default.

    Returns:
        list[BatchProcessItemError]: Any errors encountered while loading
        individual items.
    """
    if (sample_ids is None) != (sample_meta is None):
        raise SampleMetaPairingError()
    if labels is not None and len(labels) != len(sample_paths):
        raise SampleSequenceLengthMismatchError("labels", len(labels), "sample_paths", len(sample_paths))
    if sample_ids is not None and len(sample_ids) != len(sample_paths):
        raise SampleSequenceLengthMismatchError("sample_ids", len(sample_ids), "sample_paths", len(sample_paths))
    if sample_meta is not None and len(sample_meta) != len(sample_paths):
        raise SampleSequenceLengthMismatchError("sample_meta", len(sample_meta), "sample_paths", len(sample_paths))

    ub_size, ub_bytes = _prepare_user_block(user_block)

    mp_cfg = get_multiprocess_config(max_available_memory=max_memory, max_workers=max_workers)

    num_samples = len(sample_paths)
    sample_bytes = np.empty(sample_shape, dtype=sample_dtype).nbytes
    batch_size = min(
        mp_cfg.max_memory_per_process // sample_bytes,  # hard memory per worker constraint
        num_samples // mp_cfg.max_processes,  # even worker distribution
    )

    manager = Manager()
    free_slots_queue = manager.Queue()
    worker_results_queue = manager.Queue()

    # for every worker process we create a shared memory that fits a batch sized samples array
    slot_ids: list[str] = [uuid4().hex[:6] for _ in range(mp_cfg.max_processes)]
    slot_shms: list[SharedMemory] = []
    for slot_id in slot_ids:
        slot_shms.append(SharedMemory(name=f"create_dataset_{slot_id}", create=True, size=batch_size * sample_bytes))
        free_slots_queue.put(slot_id)

    # batch start indices
    batch_start_map: dict[int, int] = {b_id: b_start for b_id, b_start in enumerate(range(0, num_samples, batch_size))}
    # batch item errors
    batch_item_errors: list[BatchProcessItemError] = []
    # original indices of samples in the order they were written to the dataset
    original_indices_in_order: list[int] = []

    try:
        with h5py.File(output_path, "w-", userblock_size=ub_size, libver="latest") as h5_file:
            dataset = h5_file.create_dataset(
                name=data_key,
                shape=(num_samples, *sample_shape),
                dtype=sample_dtype,
                chunks=(h5_chunk_length, *sample_shape) if h5_chunk_length is not None else None,
                maxshape=(None, *sample_shape),  # allow resizing along axis 0
                compression=h5_compression,
                compression_opts=h5_compression_opts,
            )
            index = 0

            with ProcessPoolExecutor(max_workers=mp_cfg.max_processes) as executor:
                num_batches_submitted: int = 0
                num_batches_completed: int = 0

                for batch_id, batch_start in batch_start_map.items():
                    executor.submit(
                        process_batch,
                        free_slots_queue=free_slots_queue,
                        worker_results_queue=worker_results_queue,
                        batch_id=batch_id,
                        sample_paths=list(sample_paths[batch_start : batch_start + batch_size]),
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

                            # record original indices for successful samples in this batch,
                            # preserving their within-batch order
                            batch_start_index = batch_start_map[batch_id]
                            batch_samples = sample_paths[batch_start_index : batch_start_index + batch_size]
                            for offset, batch_sample in enumerate(batch_samples):
                                if not any(batch_sample.samefile(item_error.file_path) for item_error in item_errors):
                                    original_indices_in_order.append(batch_start_index + offset)

                            # extend item errors
                            batch_item_errors.extend(item_errors)

                            # get the view into the shared memory for that slot
                            shared_memory = SharedMemory(name=f"create_dataset_{slot_id}")
                            try:
                                # read the data from shared memory
                                batch_array = np.ndarray(
                                    shape=(actual_count, *sample_shape), dtype=sample_dtype, buffer=shared_memory.buf
                                )
                                # store the result
                                dataset.write_direct(batch_array, dest_sel=range(index, index + actual_count))
                                index += actual_count
                                del batch_array  # release buffer reference before close()
                            finally:
                                shared_memory.close()
                        # make this slot available for writing again
                        if slot_id is not None:
                            free_slots_queue.put(slot_id)
                finally:
                    # send sentinel value to all workers to unblock them
                    for _ in range(mp_cfg.max_processes):
                        free_slots_queue.put(None)

            # resize the dataset to its actual size
            dataset.resize(index, axis=0)

            # write per-sample aligned inputs, reordered to match the on-disk sample order
            if labels is not None:
                ordered_labels = labels[original_indices_in_order]
                h5_file.create_dataset(labels_key, data=ordered_labels)

            if sample_ids is not None:
                encoder = sample_meta_encoder if sample_meta_encoder is not None else _default_json_encoder
                ordered_ids = [sample_ids[i] for i in original_indices_in_order]
                ordered_meta = [sample_meta[i] for i in original_indices_in_order]
                h5_file.create_dataset(sample_ids_key, data=ordered_ids, dtype=h5py.string_dtype())
                h5_file.create_dataset(
                    sample_meta_key, data=[encoder(m) for m in ordered_meta], dtype=h5py.string_dtype()
                )
    finally:
        for shm in slot_shms:
            try:
                shm.unlink()
            finally:
                shm.close()
        manager.shutdown()

    if ub_size > 0 and ub_bytes:
        with open(output_path, "br+") as f:
            f.write(ub_bytes)

    return batch_item_errors
