from __future__ import annotations

import numpy as np
import pytest

from forge.model.ugi_closure_placement import (
    UgiSparseClosureScorer,
    closure_set_loss,
    feasible_next_closures,
    sample_sparse_closures,
)

torch = pytest.importorskip("torch")


def _model() -> UgiSparseClosureScorer:
    return UgiSparseClosureScorer(
        maximum_children=3,
        maximum_component_atoms=16,
        maximum_cycle_rank=2,
        maximum_ring_size=6,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
    )


def test_feasible_candidates_retain_supported_ring_completion() -> None:
    # Canonical exterior tree for NC1CCNCC1; its source closure is (3, 5).
    offspring = np.asarray([2, 1, 1, 0, 1, 0], dtype=np.int64)
    candidates = feasible_next_closures(
        offspring,
        remaining_closures_including_next=1,
    )

    assert (3, 5) in candidates.edges
    assert set(candidates.ring_sizes.tolist()) <= {5, 6}
    assert all(left < right for left, right in candidates.edges)


def test_two_closure_set_loss_is_permutation_tolerant_and_differentiable() -> None:
    # Bicyclic NC1CN2CCC1CC2 exterior.
    offspring = torch.tensor([2, 1, 2, 0, 0, 2, 0, 0], dtype=torch.long)
    model = _model()
    loss = closure_set_loss(
        model,
        offspring,
        role_index=0,
        target_edges=((4, 7), (3, 6)),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_two_core_attached_head_branches_can_close_a_five_member_ring() -> None:
    offspring = np.asarray([1, 0, 1, 0], dtype=np.int64)
    candidates = feasible_next_closures(
        offspring,
        attachment_count=2,
        remaining_closures_including_next=1,
    )

    assert (1, 3) in candidates.edges
    index = candidates.edges.index((1, 3))
    assert int(candidates.ring_sizes[index]) == 5


def test_sparse_sampler_realizes_exact_cycle_rank_without_repair() -> None:
    offspring = np.asarray([2, 1, 2, 0, 0, 2, 0, 0], dtype=np.int64)
    left, right = sample_sparse_closures(
        _model(),
        offspring,
        role_index=0,
        cycle_rank=2,
        generator=torch.Generator().manual_seed(17),
        device="cpu",
    )

    pairs = list(zip(left.tolist(), right.tolist(), strict=True))
    assert len(pairs) == 2
    assert pairs == sorted(set(pairs))
    assert all(pair not in {(0, 1), (0, 4)} for pair in pairs)
