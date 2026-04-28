from itertools import batched
from typing import Optional, Tuple

import pytest
import torch

from dltoolbox.normalization import WelfordEstimator, WelfordSnapshot


class TestNormalization:
    @pytest.mark.parametrize("shape", [(100, 2, 8192), (50, 3, 128, 128), (50, 3, 64, 16, 16)])
    @pytest.mark.parametrize("dim", [None, (0,), (0, 1), (0, 2)])
    def test_welford(self, shape: Tuple[int, ...], dim: Optional[Tuple[int, ...]]) -> None:
        dataset = torch.rand(shape, dtype=torch.float32)

        welford = WelfordEstimator(dim=dim)

        for batch_indices in batched(range(dataset.shape[0]), 8):
            batch = dataset[batch_indices, ...]
            welford.update(batch)

        # default dtypes are float64
        mean, std, permute = welford.finalize()
        assert mean.dtype == torch.float64
        assert std.dtype == torch.float64

        # return with tensors converted to float32
        mean, std, permute = welford.finalize(dtype=torch.float32)
        assert mean.dtype == torch.float32
        assert std.dtype == torch.float32

        # check computed mean and std values
        assert torch.allclose(mean, torch.mean(dataset, dim=dim), rtol=1e-2, atol=1e-3)  # generous tolerances
        assert torch.allclose(std, torch.std(dataset, dim=dim), rtol=1e-2, atol=1e-3)  # generous tolerances

        # can we apply the mean and std to a single sample?
        sample = dataset[0]
        assert permute is None or len(sample.shape) == len(permute)
        sample = torch.permute(sample, permute) if permute is not None else sample
        assert (sample - mean) / std is not None


def _make_estimator(dataset: torch.Tensor, dim: Optional[Tuple[int, ...]]) -> WelfordEstimator:
    est = WelfordEstimator(dim=dim)
    for batch_indices in batched(range(dataset.shape[0]), 8):
        est.update(dataset[list(batch_indices)])
    return est


class TestWelfordSnapshot:
    def test_state_dict_round_trip(self) -> None:
        snap = _make_estimator(torch.rand(50, 3), (0,)).snapshot()
        assert WelfordSnapshot.from_state_dict(snap.state_dict()) == snap

    def test_cannot_snapshot_before_first_update(self) -> None:
        with pytest.raises(RuntimeError):
            WelfordEstimator(dim=(0,)).snapshot()

    @pytest.mark.parametrize(
        "shape,dim", [((100, 3), (0,)), ((50, 3, 128, 128), (0, 1)), ((50, 3, 64, 16, 16), (0, 2))]
    )
    def test_from_snapshot_reconstructs_equivalent_estimator(
        self, shape: Tuple[int, ...], dim: Tuple[int, ...]
    ) -> None:
        dataset = torch.rand(shape)
        est = _make_estimator(dataset, dim)
        snap = est.snapshot()
        assert snap == WelfordEstimator.from_snapshot(snap).snapshot()

    @pytest.mark.parametrize(
        "shape,dim", [((100, 3), (0,)), ((50, 3, 128, 128), (0, 1)), ((50, 3, 64, 16, 16), (0, 2))]
    )
    def test_merge_produces_same_statistics_as_full_single_pass(
        self, shape: Tuple[int, ...], dim: Tuple[int, ...]
    ) -> None:
        dataset = torch.rand(shape)
        mid = shape[0] // 2

        est_full = _make_estimator(dataset, dim)
        est_a = _make_estimator(dataset[:mid], dim)
        est_b = _make_estimator(dataset[mid:], dim)
        est_a.merge(est_b.snapshot())

        assert est_full.snapshot() == est_a.snapshot()

    def test_merge_order_does_not_affect_final_statistics(self) -> None:
        dataset = torch.rand(90, 3)
        dim = (0,)

        a = _make_estimator(dataset[:30], dim)
        b = _make_estimator(dataset[30:60], dim)
        c = _make_estimator(dataset[60:], dim)

        # (A merge B) merge C
        left = WelfordEstimator.from_snapshot(a.snapshot())
        left.merge(b.snapshot(), c.snapshot())

        # A merge (B merge C)
        bc = WelfordEstimator.from_snapshot(b.snapshot())
        bc.merge(c.snapshot())
        right = WelfordEstimator.from_snapshot(a.snapshot())
        right.merge(bc.snapshot())

        assert left.snapshot() == right.snapshot()

    def test_uninitialized_estimator_adopts_state_of_first_merged_snapshot(self) -> None:
        dataset = torch.rand(50, 3)
        est_src = _make_estimator(dataset, (0,))

        est_dst = WelfordEstimator()
        est_dst.merge(est_src.snapshot())

        assert est_dst.snapshot() == est_src.snapshot()

    def test_add_does_not_modify_the_original_estimator(self) -> None:
        dataset = torch.rand(100, 3)
        est_a = _make_estimator(dataset[:50], (0,))
        est_b = _make_estimator(dataset[50:], (0,))

        snap_before = est_a.snapshot()
        _ = est_a + est_b.snapshot()
        assert est_a.snapshot() == snap_before

    def test_merge_is_atomic_incompatible_snap_in_batch_leaves_state_unchanged(self) -> None:
        dataset = torch.rand(50, 3)
        est = _make_estimator(dataset, (0,))
        snap_compatible = _make_estimator(dataset, (0,)).snapshot()
        snap_incompatible = _make_estimator(dataset, None).snapshot()

        snap_before = est.snapshot()
        with pytest.raises(ValueError):
            est.merge(snap_compatible, snap_incompatible)
        assert est.snapshot() == snap_before

    def test_merge_rejects_snapshot_with_different_reduction_dim_without_corrupting_state(self) -> None:
        dataset = torch.rand(50, 3)
        est = _make_estimator(dataset, (0,))
        est_other = _make_estimator(dataset, None)

        snap_before = est.snapshot()
        with pytest.raises(ValueError):
            est.merge(est_other.snapshot())
        assert est.snapshot() == snap_before

    def test_merge_rejects_snapshot_with_different_reduced_shape_without_corrupting_state(self) -> None:
        est_a = _make_estimator(torch.rand(50, 3), (0,))
        est_b = _make_estimator(torch.rand(50, 5), (0,))
        snap_before = est_a.snapshot()
        with pytest.raises(ValueError):
            est_a.merge(est_b.snapshot())
        assert est_a.snapshot() == snap_before

    def test_add_result_has_statistics_equivalent_to_merged_single_pass(self) -> None:
        dataset = torch.rand(100, 3)
        est_full = _make_estimator(dataset, (0,))
        est_a = _make_estimator(dataset[:50], (0,))
        est_b = _make_estimator(dataset[50:], (0,))
        assert (est_a + est_b.snapshot()).snapshot() == est_full.snapshot()

    def test_snapshots_with_same_values_are_equal(self) -> None:
        count = torch.tensor([50.0], dtype=torch.float64)
        mean = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        m2 = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64)
        a = WelfordSnapshot(count=count.clone(), mean=mean.clone(), m2=m2.clone(), dim=(0,), permute=None)
        b = WelfordSnapshot(count=count.clone(), mean=mean.clone(), m2=m2.clone(), dim=(0,), permute=None)
        assert a == b

    def test_snapshots_with_different_values_are_not_equal(self) -> None:
        count = torch.tensor([50.0], dtype=torch.float64)
        m2 = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64)
        a = WelfordSnapshot(
            count=count.clone(),
            mean=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            m2=m2.clone(),
            dim=(0,),
            permute=None,
        )
        b = WelfordSnapshot(
            count=count.clone(),
            mean=torch.tensor([1.0, 2.0, 9.9], dtype=torch.float64),
            m2=m2.clone(),
            dim=(0,),
            permute=None,
        )
        assert a != b
