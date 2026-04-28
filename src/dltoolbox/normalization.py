from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class Normalization:
    mean: Union[float, np.ndarray, torch.Tensor]
    std: Union[float, np.ndarray, torch.Tensor]
    perm: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True, eq=False)
class WelfordSnapshot:
    """Immutable snapshot of a WelfordEstimator's internal state.

    Captures everything needed to reconstruct or merge an estimator. Two snapshots
    are compatible for merging if and only if dim, permute, and reduced shape all match.

    Equality is value-based and uses torch.allclose with default tolerances, so two
    snapshots produced by different but numerically equivalent computation paths compare
    equal. Comparisons across devices are supported via implicit .cpu() promotion.

    Attributes:
        count: Scalar float64 tensor (shape (1,)) recording the total number of
            elements that have been accumulated into the statistics.
        mean: Float64 tensor holding the running mean. Shape is the "reduced shape":
            the input shape with all reduction dims removed.
        m2: Float64 tensor holding the running sum of squared deviations from the
            mean (M2 in Welford's algorithm). Same shape as mean.
        dim: Reduction dimensions passed to the originating WelfordEstimator, or
            None if the estimator reduces over all elements.
        permute: Permutation applied to input batches before accumulation, or None
            if no permutation is needed. Derived from dim and the input rank; two
            snapshots with the same dim will have the same permute.
    """

    count: torch.Tensor
    mean: torch.Tensor
    m2: torch.Tensor
    dim: Optional[Tuple[int, ...]]
    permute: Optional[Tuple[int, ...]]

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> WelfordSnapshot:
        """Reconstruct a snapshot from a dict produced by ``state_dict()``.

        Args:
            state_dict: Dict as returned by ``state_dict()`` or loaded via ``torch.load``.

        Returns:
            WelfordSnapshot: Reconstructed snapshot.
        """
        return cls(
            count=state_dict["count"],
            mean=state_dict["mean"],
            m2=state_dict["m2"],
            dim=tuple(state_dict["dim"]) if state_dict["dim"] is not None else None,
            permute=tuple(state_dict["permute"]) if state_dict["permute"] is not None else None,
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize to a flat dict suitable for ``torch.save``.

        ``dim`` and ``permute`` are stored as lists so the dict remains JSON-serialisable alongside the tensors.
        """
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "dim": list(self.dim) if self.dim is not None else None,
            "permute": list(self.permute) if self.permute is not None else None,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WelfordSnapshot):
            return NotImplemented
        return (
            torch.allclose(self.count.cpu(), other.count.cpu())
            and torch.allclose(self.mean.cpu(), other.mean.cpu())
            and torch.allclose(self.m2.cpu(), other.m2.cpu())
            and self.dim == other.dim
            and self.permute == other.permute
        )


class WelfordEstimator:
    """Compute the mean and standard deviation using Welford's iterative algorithm. Data is processed in batches
    which enables the computation on very large datasets which do not fit into main memory.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    _count: torch.Tensor
    _mean: torch.Tensor
    _m2: torch.Tensor

    def __init__(self, dim: Optional[Tuple[int, ...]] = None):
        self._initialized = False
        self._dim = dim
        self._permute: Optional[Tuple[int, ...]] = None

    def _try_initialize(self, shape: torch.Size, device: torch.device) -> None:
        if not self._initialized:
            self._count = torch.zeros(1, dtype=torch.float64, device=device)
            if self._dim is None:
                self._mean = torch.zeros(1, dtype=torch.float64, device=device)
                self._m2 = torch.zeros(1, dtype=torch.float64, device=device)
            else:
                reduced_shape = tuple([shape[i] for i in range(len(shape)) if i not in self._dim])
                self._mean = torch.zeros(reduced_shape, dtype=torch.float64, device=device)
                self._m2 = torch.zeros(reduced_shape, dtype=torch.float64, device=device)

                if any(v > i for i, v in enumerate(self._dim)):
                    # input needs permutation such that the requested dims are the leading dimensions
                    # otherwise computations cannot be applied to the intermediate tensors
                    self._permute = tuple([*self._dim] + [i for i in range(len(shape)) if i not in self._dim])

            self._initialized = True

    def _count_features(self, input_shape: torch.Size) -> torch.Tensor:
        if self._dim is None:
            return torch.prod(torch.tensor(input_shape))
        else:
            return torch.prod(torch.tensor([input_shape[i] for i in range(len(input_shape)) if i in self._dim]))

    def _empty(self) -> bool:
        return not self._initialized or self._count.item() == 0

    def _check_compatible(self, snap: WelfordSnapshot) -> None:
        if self._dim != snap.dim:
            raise ValueError(f"incompatible dim: {self._dim!r} vs {snap.dim!r}")
        if self._permute != snap.permute:
            raise ValueError(f"incompatible permute: {self._permute!r} vs {snap.permute!r}")
        if self._mean.shape != snap.mean.shape:
            raise ValueError(f"incompatible reduced shape: {tuple(self._mean.shape)} vs {tuple(snap.mean.shape)}")

    def _load_snapshot(self, snap: WelfordSnapshot) -> None:
        self._dim = snap.dim
        self._permute = snap.permute
        self._count = snap.count.detach().clone().to(dtype=torch.float64)
        self._mean = snap.mean.detach().clone().to(dtype=torch.float64)
        self._m2 = snap.m2.detach().clone().to(dtype=torch.float64)
        self._initialized = True

    def _align_device_dtype(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Move tensors to self's device and cast to float64."""
        return tuple(t.to(device=self._mean.device, dtype=torch.float64) for t in tensors)

    @classmethod
    @torch.no_grad()
    def from_snapshot(cls, snap: WelfordSnapshot) -> WelfordEstimator:
        """Construct an estimator preloaded with the given snapshot.

        Args:
            snap (WelfordSnapshot): Snapshot to restore.

        Returns:
            WelfordEstimator: A new estimator with the snapshot's state.
        """
        est = cls()
        est._load_snapshot(snap)
        return est

    @torch.no_grad()
    def snapshot(self) -> WelfordSnapshot:
        """Return a detached copy of the current estimator state.

        Returns:
            WelfordSnapshot: Immutable snapshot of the current state.

        Raises:
            RuntimeError: If the estimator has never been updated.
        """
        if not self._initialized:
            raise RuntimeError("cannot snapshot an estimator that has not been updated")
        return WelfordSnapshot(
            count=self._count.clone(),
            mean=self._mean.clone(),
            m2=self._m2.clone(),
            dim=self._dim,
            permute=self._permute,
        )

    @torch.no_grad()
    def merge(self, *snaps: WelfordSnapshot) -> None:
        """Merge one or more snapshots into this estimator in-place.

        Snapshots with count == 0 are skipped. When this estimator is
        uninitialized, state is copied from the first non-empty snapshot.
        Snapshot tensors are moved to this estimator's device before merging.

        Args:
            *snaps (WelfordSnapshot): Snapshots to merge.

        Raises:
            ValueError: If any snapshot is incompatible with the current state.
        """
        for snap in snaps:
            if snap.count.item() == 0:
                continue
            if self._empty():
                if self._initialized:
                    self._check_compatible(snap)
                self._load_snapshot(snap)
            else:
                self._check_compatible(snap)
                count_b, mean_b, m2_b = self._align_device_dtype(snap.count, snap.mean, snap.m2)
                count_a = self._count.clone()
                n = count_a + count_b
                delta = mean_b - self._mean
                self._mean = self._mean + delta * count_b / n
                self._m2 = self._m2 + m2_b + delta**2 * count_a * count_b / n
                self._count = n

    def __add__(self, other: WelfordSnapshot) -> WelfordEstimator:
        """Return a new estimator with other merged in, leaving self unchanged."""
        result = deepcopy(self)
        result.merge(other)
        return result

    @torch.no_grad()
    def update(self, data: torch.Tensor) -> None:
        """
        Update the internal state by incorporating the input data.

        Args:
            data (torch.Tensor): Input data.
        """

        self._try_initialize(data.shape, data.device)

        (data,) = self._align_device_dtype(data)

        self._count += self._count_features(data.shape)
        if self._permute is not None:
            data = torch.permute(data, self._permute)

        # Fused batch Chan merge: avoids materializing batch_mean / batch_m2 explicitly.
        delta1 = torch.sub(data, self._mean)
        self._mean += torch.sum(delta1 / self._count, dim=tuple(range(len(self._dim))) if self._dim else None)
        delta2 = torch.sub(data, self._mean)
        self._m2 += torch.sum(delta1 * delta2, dim=tuple(range(len(self._dim))) if self._dim else None)

    @torch.no_grad()
    def finalize(
        self, dtype: torch.dtype = torch.float64, device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[int, ...]]]:
        """
        Return the mean and standard deviation tensors.

        Additionally, if an input sample requires reordering of dimensions (permutation) before  applying the mean and
        standard deviation, the function returns the dimension ordering needed for this permutation.

        Args:
            dtype (torch.dtype): Data type of the output tensors Defaults to torch.float64.
            device (torch.device, optional): Location of output tensors. Defaults to the device of the input data.

        Returns:
            mean (torch.Tensor): The computed mean values.
            std (torch.Tensor): The computed standard deviation values.
            permutation (tuple, optional): The ordering of dimensions for permutation, if applicable.
        """

        return (
            self._mean.to(dtype=dtype, device=device),
            torch.sqrt(self._m2 / self._count).to(dtype=dtype, device=device),
            tuple(v - 1 for i, v in enumerate(self._permute) if i > 0) if self._permute is not None else None,
        )
