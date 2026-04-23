from abc import ABC, abstractmethod


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
