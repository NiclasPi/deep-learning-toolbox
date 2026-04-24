from abc import ABC, abstractmethod
from collections.abc import Sequence


class ISampleMetaStore[T](ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_by_index(self, index: int) -> T:
        raise NotImplementedError

    @abstractmethod
    def get_by_id(self, identifier: str) -> T:
        raise NotImplementedError

    def get_all_ids(self) -> Sequence[str]:
        raise NotImplementedError

    def get_all(self) -> Sequence[tuple[str, T]]:
        raise NotImplementedError
