from typing import Optional


class EarlyStopping:
    """Early stops the training if loss doesn't improve."""

    def __init__(self, patience: int = 5, delta: float = 0.01) -> None:
        self._patience = patience
        self._delta = delta  # percent of minimum for required change
        self._minimum: Optional[float] = None  # current loss minimum
        self._counter = 0

    def __call__(self, loss: float) -> bool:
        if self._minimum is None:
            self._minimum = loss
            return False

        # loss change > minimum change?
        if self._minimum - loss > self._delta * self._minimum:
            # loss has improved -> reset counter
            self._counter = 0
        else:
            # loss has not improved -> increment counter
            self._counter += 1

        # update minimum loss value
        self._minimum = min(self._minimum, loss)
        return self._counter >= self._patience

    def __bool__(self) -> bool:
        return self._counter >= self._patience

    @property
    def counter(self) -> int:
        return self._counter
