import argparse
import h5py
import numpy as np
import pathlib
import psutil
import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import batched, chain
from PIL import Image
from typing import Iterable, Iterator, Tuple

from dltoolbox.argutils import parse_image_size, parse_dataset_size


def load_image(
        file: pathlib.Path,
        size: Tuple[int, int]  # (width, height)
) -> Image:
    image = Image.open(file.absolute()).convert("RGB")

    if image.width < size[0] or image.height < size[1]:
        raise RuntimeError(f"image '{file.absolute()}' is smaller than the requested target size")

    fit_ratio = (image.size[0] / size[0], image.size[1] / size[1])
    if any(x > 1 for x in fit_ratio):
        # extract the largest possible box of the requested aspect ratio
        crop_size = (int(size[0] * min(fit_ratio)), int(size[1] * min(fit_ratio)))
        crop_box = (
            (image.size[0] - crop_size[0]) // 2,
            (image.size[1] - crop_size[1]) // 2,
            (image.size[0] + crop_size[0]) // 2,
            (image.size[1] + crop_size[1]) // 2
        )
        image = image.crop(crop_box)

        # resize image to requested size
        image = image.resize(size, resample=Image.Resampling.LANCZOS)

    return image


def process_batch(
        files: Iterable[pathlib.Path],
        size: Tuple[int, int]
) -> np.ndarray:
    images = []

    for file in files:
        # load image from disk and perform preprocessing
        image: Image = load_image(file, size)
        # convert image to numpy array and append
        images.append(np.array(image))

    return np.stack(images, axis=0)


def create_dataset(
        out_path: str,
        dataset_size: int,
        batch_iter: Iterator[Tuple[pathlib.Path, ...]],
        image_size: Tuple[int, int],
        num_workers: int = 1,
) -> None:
    with h5py.File(out_path, "w-") as h5:  # create file, fail if exists
        dataset = h5.create_dataset("data", shape=(dataset_size, *image_size, 3), dtype=np.uint8)
        index = 0

        # parallel processing of batches
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for i in range(num_workers):
                    try:
                        future = executor.submit(process_batch, next(batch_iter), image_size)
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
                            future = executor.submit(process_batch, next(batch_iter), image_size)
                            futures[future] = id(future)
                        except StopIteration:
                            continue
        else:
            for batch in batch_iter:
                result = process_batch(batch, image_size)

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
        help="Path to the directory containing the images."
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
        nargs="+",
        help="Size of the images after cropping and resizing. One value requests square sized images, two values "
             "request a specific width and height for the images."
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
    image_size = parse_image_size(args.target_size)
    image_bytes = np.empty((*image_size, 3), dtype=np.uint8).nbytes
    batch_size = min(
        available_memory_per_worker // image_bytes,  # hard memory per worker constraint
        min(k for k in [train_size, valid_size, test_size] if k > 0) // args.num_workers  # even worker distribution
    )
    print(f"Batch size: {batch_size}")

    # create datasets
    if train_size > 0:
        create_dataset(
            os.path.join(args.output_directory, f"{args.output_prefix}_train.hdf5"),
            train_size,
            batched(train_files, batch_size),
            image_size,
            num_workers=args.num_workers
        )
        print(f"Created train set, size: {train_size}")

    if valid_size > 0:
        create_dataset(
            os.path.join(args.output_directory, f"{args.output_prefix}_valid.hdf5"),
            valid_size,
            batched(valid_files, batch_size),
            image_size,
            num_workers=args.num_workers
        )
        print(f"Created valid set, size: {valid_size}")

    if test_size > 0:
        create_dataset(
            os.path.join(args.output_directory, f"{args.output_prefix}_test.hdf5"),
            test_size,
            batched(test_files, batch_size),
            image_size,
            num_workers=args.num_workers
        )
        print(f"Created test set, size: {test_size}")
