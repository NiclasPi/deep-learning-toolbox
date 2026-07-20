from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def evaluation_mode(model: torch.nn.Module) -> Iterator[None]:
    """Context manager for save evaluation mode switching."""

    was_training = model.training

    model.eval()
    try:
        yield None
    finally:
        model.train(was_training)
