from __future__ import annotations

import numpy as np
import pytest

from forge.design.flow.v5_morphology_flow import (
    V5MorphologyFlow,
    sample_v5_morphologies,
)
from forge.design.flow.v5_morphology_program import (
    V5GlobalMorphologyProgram,
    offspring_matches_program,
    regions_match_program,
)

torch = pytest.importorskip("torch")


def test_v5_morphology_model_preserves_batch_and_sequence_shapes() -> None:
    model = V5MorphologyFlow(
        maximum_children=4,
        maximum_heavy_atoms=32,
        maximum_junction_budget=8,
        maximum_cycle_rank=3,
        hidden_dim=32,
        layers=2,
        dropout=0.0,
    )
    offspring = torch.zeros((2, 7), dtype=torch.long)
    regions = torch.zeros_like(offspring)
    programs = torch.tensor([[2, 1, 2, 0, 0, 0], [3, 1, 3, 1, 1, 1]])
    node_mask = torch.tensor([[True, True, True, True, True, False, False], [True] * 7])
    output = model(
        offspring,
        regions,
        programs,
        torch.tensor([0.2, 0.8]),
        node_mask,
    )

    assert output["offspring"].shape == (2, 7, 5)
    assert output["regions"].shape == (2, 7, 3)


def test_untrained_sampler_still_obeys_every_global_program_constraint() -> None:
    programs = (
        V5GlobalMorphologyProgram(2, 1, 5, 0, 0, 0),
        V5GlobalMorphologyProgram(3, 2, 5, 1, 1, 1),
        V5GlobalMorphologyProgram(2, 2, 8, 1, 2, 0),
    )
    model = V5MorphologyFlow(
        maximum_children=4,
        maximum_heavy_atoms=32,
        maximum_junction_budget=8,
        maximum_cycle_rank=3,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
    )
    samples, metrics = sample_v5_morphologies(
        model,
        programs,
        {
            "offspring": np.asarray([0.65, 0.25, 0.08, 0.015, 0.005]),
            "regions": np.asarray([0.2, 0.2, 0.6]),
        },
        sample_steps=4,
        batch_size=3,
        seed=20260731,
        device="cpu",
    )

    assert len(samples) == len(programs)
    assert metrics["terminal_tree_repairs"] == 0
    assert metrics["terminal_region_repairs"] == 0
    for sample in samples:
        assert offspring_matches_program(sample.offspring, sample.program)
        assert regions_match_program(sample.regions, sample.offspring, sample.program)
