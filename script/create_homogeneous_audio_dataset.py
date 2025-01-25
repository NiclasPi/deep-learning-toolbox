import argparse
import h5py
import numpy as np
import pathlib
import psutil
import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import batched, chain
from librosa import load
from typing import Iterable, Iterator, Optional, Tuple

from dltoolbox.argutils import parse_dataset_size


def load_audio(
        file: pathlib.Path,
        target_size: int,  # length in samples
        load_sample_rate: int = 22050,
        load_sample_offset: float = 0,
        load_sample_duration: Optional[float] = None,
        random_sample_slice: bool = False,
) -> np.ndarray:
    data, _ = load(
        file.absolute(),
        sr=load_sample_rate,
        offset=load_sample_offset,
        duration=load_sample_duration,
        dtype=np.float32,
    )

    if len(data) == 0:
        raise RuntimeError(f"audio '{file.absolute()}' contains zero samples")

    if len(data) < target_size:
        # extend data array with zeros
        padl = (target_size - len(data)) // 2
        padr = (target_size - len(data) + 1) // 2
        assert len(data) + padl + padr == target_size
        data = np.pad(data, (padl, padr), "constant", constant_values=0)

    if random_sample_slice and len(data) > target_size:
        idx = np.random.randint(len(data) - target_size)
        data = data[idx: idx + target_size]
    else:
        data = data[:target_size]

    assert len(data) == target_size

    # normalize data to [-1,1]
    data = data / np.abs(data).max()

    return data


def process_batch(
        files: Iterable[pathlib.Path],
        **kwargs
) -> np.ndarray:
    samples = []

    for file in files:
        samples.append(load_audio(file, **kwargs))

    return np.stack(samples, axis=0)


def create_dataset(
        out_path: str,
        dataset_size: int,
        batch_iter: Iterator[Tuple[pathlib.Path, ...]],
        target_size: int,
        num_workers: int = 1,
        **kwargs
) -> None:
    with h5py.File(out_path, "w-") as h5:  # create file, fail if exists
        dataset = h5.create_dataset("data", shape=(dataset_size, target_size), dtype=np.float32)
        index = 0

        # parallel processing of batches
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for i in range(num_workers):
                    try:
                        future = executor.submit(process_batch, next(batch_iter), target_size=target_size, **kwargs)
                        futures[future] = id(future)
                    except StopIteration:
                        break

                while len(futures) > 0:
                    for future in as_completed(futures):
                        result = future.result()

                        # store the result
                        dataset.write_direct(result, dest_sel=range(index, index + result.shape[0]))
                        index += result.shape[0]

                        del result
                        del futures[future]

                        try:
                            future = executor.submit(process_batch, next(batch_iter), target_size=target_size, **kwargs)
                            futures[future] = id(future)
                        except StopIteration:
                            continue
        else:
            for batch in batch_iter:
                result = process_batch(batch, target_size=target_size, **kwargs)

                # store the result
                dataset.write_direct(result, dest_sel=range(index, index + result.shape[0]))
                index += result.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory",
        type=str,
        required=True,
        help="Path to the output directory."
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="dataset",
        help="Prefix for the output files."
    )
    parser.add_argument(
        "--data-directory",
        type=str,
        required=True,
        help="Path to the directory containing the audio files."
    )
    parser.add_argument(
        "--include-subdirs",
        action="store_true",
        help="Include subdirectories of the data directory."
    )
    parser.add_argument(
        "--file-extensions",
        type=str,
        nargs="+",
        help="File extensions to be included in the dataset."
    )
    parser.add_argument(
        "--target-size",
        type=int,
        required=True,
        help="Size of the audio after slicing."
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        nargs="+",
        help="Size of the output dataset(s). The first, second and third values denote the size of the train set, "
             "valid set, and test set, respectively. The size of the train set is required, others are optional. "
             "If missing, the respective sizes will be zero."
    )
    parser.add_argument(
        "--random-order",
        action="store_true",
        help="Process the files in the data directory in random order."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes."
    )
    parser.add_argument(
        "--max-memory",
        type=int,
        help="Maximum amount of memory to use (in MB)."
    )

    args = parser.parse_args()

    if args.num_workers is None or args.num_workers < 1:
        args.num_workers = 1

    root_path = pathlib.Path(args.data_directory)
    file_paths = list(
        chain.from_iterable(
            root_path.glob(f"{'**/' if args.include_subdirs else './'}*.{ext}", case_sensitive=False)
            for ext in args.file_extensions
        )
    )
    print(f"Number of files: {len(file_paths)}")

    # available memory in bytes
    available_memory = int(psutil.virtual_memory().available * 0.8)
    if args.max_memory is not None:
        if args.max_memory * 1e6 > available_memory:
            raise RuntimeError(f"requested memory ({args.max_memory} MB) is currently not available")
        available_memory = int(args.max_memory * 1e6)

    available_memory_per_worker = int(available_memory / args.num_workers)
    print(f"Available memory per worker: {available_memory_per_worker / 1e6:.02f}MB")

    # determine the train, valid, test splits
    train_size, valid_size, test_size = parse_dataset_size(args.dataset_size)
    if train_size + valid_size + test_size > len(file_paths):
        raise RuntimeError("dataset sizes exceed number of available files")
    if args.random_order:
        random.seed(args.seed)
        random.shuffle(file_paths)
    train_files = file_paths[:train_size]
    valid_files = file_paths[train_size:train_size + valid_size]
    test_files = file_paths[train_size + valid_size:train_size + valid_size + test_size]
    print(f"Dataset size: {train_size} (train), {valid_size} (valid), {test_size} (test)")

    # compute batch size
    sample_bytes = np.empty((args.target_size,), dtype=np.float32).nbytes
    batch_size = min(
        available_memory_per_worker // sample_bytes,  # hard memory per worker constraint
        min(k for k in [train_size, valid_size, test_size] if k > 0) // args.num_workers  # even worker distribution
    )
    print(f"Batch size: {batch_size}")

    # create datasets
    if train_size > 0:
        create_dataset(
            os.path.join(args.output_directory, f"{args.output_prefix}_train.hdf5"),
            train_size,
            batched(train_files, batch_size),
            target_size=args.target_size,
            random_sample_slice=True,
            num_workers=args.num_workers
        )
        print(f"Created train set, size: {train_size}")

    if valid_size > 0:
        create_dataset(
            os.path.join(args.output_directory, f"{args.output_prefix}_valid.hdf5"),
            valid_size,
            batched(valid_files, batch_size),
            target_size=args.target_size,
            random_sample_slice=True,
            num_workers=args.num_workers
        )
        print(f"Created valid set, size: {valid_size}")

    if test_size > 0:
        create_dataset(
            os.path.join(args.output_directory, f"{args.output_prefix}_test.hdf5"),
            test_size,
            batched(test_files, batch_size),
            target_size=args.target_size,
            random_sample_slice=True,
            num_workers=args.num_workers
        )
        print(f"Created test set, size: {test_size}")
