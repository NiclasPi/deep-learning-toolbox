import pytest
from torch.utils.data import DataLoader, Dataset

from dltoolbox.worker_init import CompositeWorkerInit

# per-process record of called init functions; each forked worker gets its own copy
_calls: list[tuple[str, int]] = []


def _record_first(worker_id: int) -> None:
    _calls.append(("first", worker_id))


def _record_second(worker_id: int) -> None:
    _calls.append(("second", worker_id))


class _CallsDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> tuple[tuple[str, int], ...]:
        return tuple(_calls)


class TestCompositeWorkerInit:
    def test_calls_in_order(self) -> None:
        calls = []
        init = CompositeWorkerInit([lambda i: calls.append(("a", i)), lambda i: calls.append(("b", i))])

        init(3)

        assert calls == [("a", 3), ("b", 3)]

    def test_list_mutation(self) -> None:
        calls = []
        init = CompositeWorkerInit([lambda i: calls.append("b")])
        init.append(lambda i: calls.append("c"))
        init.insert(0, lambda i: calls.append("a"))
        init.extend([lambda i: calls.append("d")])

        init(0)

        assert len(init) == 4
        assert calls == ["a", "b", "c", "d"]

    def test_exception_stops_chain(self) -> None:
        calls = []

        def fail(worker_id: int) -> None:
            raise ValueError("boom")

        init = CompositeWorkerInit([fail, lambda i: calls.append(i)])

        with pytest.raises(ValueError, match="boom"):
            init(0)
        assert calls == []

    def test_runs_in_dataloader_workers(self) -> None:
        init = CompositeWorkerInit([_record_first, _record_second])
        loader = DataLoader(_CallsDataset(), batch_size=None, num_workers=2, worker_init_fn=init)

        results = list(loader)

        worker_calls = {tuple(tuple(call) for call in calls) for calls in results}  # collation yields lists
        assert worker_calls == {(("first", 0), ("second", 0)), (("first", 1), ("second", 1))}
        assert _calls == []  # init functions ran only in the workers
