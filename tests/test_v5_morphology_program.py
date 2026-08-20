from __future__ import annotations

import numpy as np
import pytest

from forge.model.lipid_context import HEAD_REGION, INTERFACE_REGION, TAIL_REGION
from forge.model.v5_morphology_program import (
    MorphologyProgramError,
    V5GlobalMorphologyProgram,
    offspring_matches_program,
    regions_match_program,
    sample_offspring_with_exact_junction_budget,
    sample_regions_with_exact_program,
    tree_junction_contributions,
)

torch = pytest.importorskip("torch")


def test_tree_junction_budget_uses_tree_degree_and_not_child_count_alone() -> None:
    offspring = np.asarray([2, 2, 0, 0, 0], dtype=np.int64)

    assert tree_junction_contributions(offspring).tolist() == [0, 1, 0, 0, 0]


@pytest.mark.parametrize("node_count", [1, 2, 7, 32])
def test_zero_junction_sampler_returns_exact_valid_preorder_tree(node_count: int) -> None:
    logits = torch.randn(node_count, 5, generator=torch.Generator().manual_seed(node_count))
    offspring = sample_offspring_with_exact_junction_budget(
        logits,
        junction_budget=0,
        generator=torch.Generator().manual_seed(node_count + 100),
    )
    program = V5GlobalMorphologyProgram(
        n_head=1,
        n_interface=0,
        n_tail=node_count - 1,
        junction_budget_head_interface=0,
        junction_budget_tail=0,
        cycle_rank=0,
    )

    assert offspring_matches_program(offspring, program)


def test_offspring_sampler_exactly_spends_nonzero_junction_budget() -> None:
    logits = torch.zeros(12, 5)
    offspring = sample_offspring_with_exact_junction_budget(
        logits,
        junction_budget=3,
        generator=torch.Generator().manual_seed(11),
    )

    assert int(tree_junction_contributions(offspring).sum()) == 3


def test_region_sampler_honors_counts_and_separate_branch_budgets() -> None:
    offspring = np.asarray([3, 0, 2, 0, 0, 0], dtype=np.int64)
    contributions = tree_junction_contributions(offspring)
    assert contributions.tolist() == [1, 0, 1, 0, 0, 0]
    program = V5GlobalMorphologyProgram(
        n_head=2,
        n_interface=1,
        n_tail=3,
        junction_budget_head_interface=1,
        junction_budget_tail=1,
        cycle_rank=0,
    )
    logits = torch.zeros(6, 3)
    logits[:, TAIL_REGION] = 1.0
    regions = sample_regions_with_exact_program(
        logits,
        offspring,
        program,
        generator=torch.Generator().manual_seed(7),
    )

    assert regions[0] == HEAD_REGION
    assert regions_match_program(regions, offspring, program)
    assert np.count_nonzero(regions == INTERFACE_REGION) == 1


def test_region_sampler_rejects_tree_that_cannot_allocate_tail_branch_budget() -> None:
    offspring = np.asarray([3, 0, 0, 0], dtype=np.int64)
    program = V5GlobalMorphologyProgram(
        n_head=3,
        n_interface=0,
        n_tail=1,
        junction_budget_head_interface=0,
        junction_budget_tail=1,
        cycle_rank=0,
    )

    with pytest.raises(MorphologyProgramError, match="regional morphology program"):
        sample_regions_with_exact_program(
            torch.zeros(4, 3),
            offspring,
            program,
            generator=torch.Generator().manual_seed(3),
        )
