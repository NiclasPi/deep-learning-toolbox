from typing import Callable, Iterable

WorkerInitFn = Callable[[int], None]


class CompositeWorkerInit:
    """Chain several worker init functions into one DataLoader ``worker_init_fn``.

    Each function is called with the worker ID, in list order. An exception stops the chain and is re-raised
    by the DataLoader in the main process. With the ``spawn`` or ``forkserver`` start method, the chain is
    pickled, so every function must be picklable (module-level functions or class instances, not lambdas).

    Args:
        init_fns: Functions taking the worker ID.
    """

    def __init__(self, init_fns: Iterable[WorkerInitFn]):
        self._init_fns = list(init_fns)

    def __call__(self, worker_id: int) -> None:
        for init_fn in self._init_fns:
            init_fn(worker_id)

    def __len__(self) -> int:
        return len(self._init_fns)

    def __getitem__(self, index: int) -> WorkerInitFn:
        return self._init_fns[index]

    def append(self, init_fn: WorkerInitFn):
        self._init_fns.append(init_fn)

    def extend(self, init_fns: Iterable[WorkerInitFn]):
        self._init_fns.extend(init_fns)

    def insert(self, index: int, init_fn: WorkerInitFn):
        self._init_fns.insert(index, init_fn)
