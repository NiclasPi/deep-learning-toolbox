from typing import Tuple, Union

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import dltoolbox.transforms as tfs
from tests.utils import transform_create_input


class TestTransforms:
    def test_nested_transforms_with_mode(self) -> None:
        class SomeTransform(tfs.TransformerWithMode):
            def __init__(self):
                super().__init__()

            def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
                return x

        tf1 = SomeTransform()
        tf2 = SomeTransform()
        nest = tfs.ComposeWithMode([tfs.NoTransform(), tf1, tfs.ComposeWithMode([tfs.NoTransform(), tf2])])

        nest.set_eval_mode()
        assert tf1.is_eval_mode()
        assert tf2.is_eval_mode()
        assert nest.is_eval_mode()

        nest.set_train_mode()
        assert tf1.is_train_mode()
        assert tf2.is_train_mode()
        assert nest.is_train_mode()

    def test_transform_mutation_visible_to_dataloader_workers(self) -> None:
        """A Compose shared by reference with worker processes must reflect in-place slot swaps.

        DataLoader workers are (re-)forked from the main process on every fresh iteration, so a
        transform swapped into a shared Compose before that iteration must be visible to the
        workers, not just to the main process.
        """

        class MarkerDataset(Dataset):
            def __init__(self, data_transform: tfs.Compose):
                self.data_transform = data_transform

            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> str:
                return type(self.data_transform[1]).__name__

        compose = tfs.Compose([tfs.NoTransform(), tfs.NoTransform()])
        dataset = MarkerDataset(compose)
        loader = DataLoader(dataset, batch_size=4, num_workers=2)

        assert set(next(iter(loader))) == {"NoTransform"}

        compose[1] = tfs.ToTensor()

        observed = set(next(iter(loader)))
        assert observed == {"ToTensor"}, f"worker(s) did not see the mutated transform slot, saw {observed}"

    def test_transform_mode_visible_to_dataloader_workers(self) -> None:
        """A transform's train/eval mode toggled by reference must propagate to worker processes."""

        class ModeAwareTransform(tfs.TransformerWithMode):
            def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
                return x

        class ModeMarkerDataset(Dataset):
            def __init__(self, transform: tfs.TransformerWithMode):
                self.transform = transform

            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> bool:
                return self.transform.is_eval_mode()

        transform = ModeAwareTransform()
        dataset = ModeMarkerDataset(transform)
        loader = DataLoader(dataset, batch_size=4, num_workers=2)

        assert set(next(iter(loader)).tolist()) == {False}

        transform.set_eval_mode()

        observed = set(next(iter(loader)).tolist())
        assert observed == {True}, f"worker(s) did not see the mode change, saw {observed}"

    def test_dict_create(self) -> None:
        x = transform_create_input(backend="numpy", shape=(10, 10))

        tf = tfs.DictTransformCreate({"one": tfs.NoTransform(), "two": tfs.ToTensor()})
        y = tf(x)

        assert "one" in y and isinstance(y["one"], np.ndarray)
        assert "two" in y and isinstance(y["two"], torch.Tensor)

    def test_dict_clone(self) -> None:
        x = {"one": transform_create_input(backend="numpy", shape=(10, 10), fill="zeros")}

        tf = tfs.DictTransformClone("one", "two")
        y = tf(x)

        x["one"][0, :] = 1  # default clone is deep copy
        assert "two" in y and not np.any(y["two"] == 1)

        tf = tfs.DictTransformClone("one", "two", shallow=True)
        z = tf(x)

        x["one"][1, :] = 2
        assert "two" in y and np.any(y["two"] == 1)
        assert "two" in z and np.any(z["two"] == 2)  # shallow clone shares memory

    @pytest.mark.parametrize("keys", ["one", ("one", "two")])
    def test_dict_apply(self, keys: str | Tuple[str, ...]) -> None:
        if isinstance(keys, str):
            x = {keys: transform_create_input(backend="numpy", shape=(10, 10))}
        else:
            x = {key: transform_create_input(backend="numpy", shape=(10, 10)) for key in keys}

        tf = tfs.DictTransformApply(keys, tfs.ToTensor())
        y = tf(x)

        if isinstance(keys, str):
            assert keys in y and isinstance(y[keys], torch.Tensor)
        else:
            for key in keys:
                assert key in y and isinstance(y[key], torch.Tensor)

        tf = tfs.DictTransformApply("nan", tfs.ToTensor())
        with pytest.raises(KeyError):
            tf(x)
