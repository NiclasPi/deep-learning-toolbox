from typing import Literal, Tuple, Union

import numpy as np
import torch

from dltoolbox.transforms import TransformerMode


def transform_set_mode(transform: TransformerMode, mode: Literal["train", "eval"]) -> None:
    if mode == "train":
        transform.set_train_mode()
    elif mode == "eval":
        transform.set_eval_mode()


def transform_create_input(
    backend: Literal["numpy", "torch"],
    shape: Tuple[int, ...],
    fill: Union[Literal["random", "zeros", "ones"], int, float] = "random",
) -> Union[np.ndarray, torch.Tensor]:
    if backend == "numpy":
        if fill == "random":
            return np.random.rand(*shape).astype(np.float32)
        elif fill == "zeros":
            return np.zeros(shape, dtype=np.float32)
        elif fill == "ones":
            return np.ones(shape, dtype=np.float32)
        elif isinstance(fill, int) or isinstance(fill, float):
            return np.full(shape, fill, dtype=np.float32)
        else:
            raise ValueError("Unrecognized fill")
    elif backend == "torch":
        if fill == "random":
            return torch.rand(shape, dtype=torch.float32)
        elif fill == "zeros":
            return torch.zeros(shape, dtype=torch.float32)
        elif fill == "ones":
            return torch.ones(shape, dtype=torch.float32)
        elif isinstance(fill, int) or isinstance(fill, float):
            return torch.full(shape, fill, dtype=torch.float32)
        else:
            raise ValueError("Unrecognized fill")
    else:
        raise ValueError(f"Backend {backend} not supported")
