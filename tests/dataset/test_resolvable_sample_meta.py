from dataclasses import dataclass

from dltoolbox.dataset.metadata.dataset_metadata import DatasetMetadata
from dltoolbox.dataset.metadata.resolvable_sample_meta import is_resolvable


@dataclass
class _WithResolve:
    name: str

    def resolve(self, sample_id: str, dataset_metadata: DatasetMetadata) -> "_WithResolve":
        return self


@dataclass
class _WithoutResolve:
    name: str


def test_is_resolvable_true_for_object_implementing_resolve() -> None:
    assert is_resolvable(_WithResolve(name="a")) is True


def test_is_resolvable_false_for_object_without_resolve() -> None:
    assert is_resolvable(_WithoutResolve(name="a")) is False
