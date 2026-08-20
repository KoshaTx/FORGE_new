from __future__ import annotations

import numpy as np
import pytest

from forge.design.flow.ugi_morphology_flow import (
    UgiMorphologyFlow,
    collate_ugi_morphology_records,
    noise_ugi_morphology_batch,
    sample_ugi_morphologies,
    ugi_morphology_flow_loss,
)
from forge.design.flow.ugi_morphology_program import (
    UgiComponentMorphology,
    UgiProductMorphology,
    attached_tree_matches_program,
)
from forge.potency.audit.ugi_semantic_annotations import ROLE_NAMES

torch = pytest.importorskip("torch")


def _record() -> UgiProductMorphology:
    components = []
    for role, offspring in zip(
        ROLE_NAMES,
        (
            np.asarray([1, 2, 0, 0], dtype=np.int64),
            np.asarray([1, 1, 1, 0], dtype=np.int64),
            np.asarray([1, 1, 0], dtype=np.int64),
        ),
        strict=True,
    ):
        components.append(
            UgiComponentMorphology(
                role=role,
                component_key=f"test/{role}",
                offspring=offspring,
                closure_left=np.zeros(0, dtype=np.int64),
                closure_right=np.zeros(0, dtype=np.int64),
            )
        )
    return UgiProductMorphology(
        product_id="test",
        components=tuple(components),  # type: ignore[arg-type]
    )


def _model() -> UgiMorphologyFlow:
    return UgiMorphologyFlow(
        maximum_children=3,
        maximum_component_atoms=16,
        maximum_total_atoms=48,
        maximum_junction_budget=4,
        maximum_cycle_rank=2,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
    )


def test_ugi_flow_uses_generated_role_layout_and_balanced_loss() -> None:
    record = _record()
    batch = collate_ugi_morphology_records(
        (record, record),
        maximum_nodes=16,
        maximum_children=3,
    )
    model = _model()
    generator = torch.Generator().manual_seed(5)
    sources = torch.full((3, 4), 0.25)
    noisy = noise_ugi_morphology_batch(
        batch,
        sources,
        torch.tensor([0.2, 0.8]),
        generator,
    )
    predictions = model(
        noisy["offspring"],
        batch["role_states"],
        batch["within_role_positions"],
        batch["programs"],
        torch.tensor([0.2, 0.8]),
        batch["node_mask"],
    )
    loss, metrics = ugi_morphology_flow_loss(predictions, batch)

    assert predictions["offspring"].shape == (2, 16, 4)
    assert torch.isfinite(loss)
    assert set(metrics) == {"total", *(f"{role}_offspring_ce" for role in ROLE_NAMES)}
    assert not hasattr(batch, "component_key")
    assert "distance_to_core" not in batch
    assert "node_states" not in batch


def test_untrained_ugi_sampler_obeys_all_three_exact_programs() -> None:
    record = _record()
    programs = (record.program, record.program, record.program)
    samples, metrics = sample_ugi_morphologies(
        _model(),
        programs,
        np.asarray(
            [
                [0.55, 0.30, 0.10, 0.05],
                [0.45, 0.40, 0.10, 0.05],
                [0.40, 0.45, 0.10, 0.05],
            ]
        ),
        sample_steps=3,
        batch_size=3,
        seed=23,
        device="cpu",
    )

    assert metrics["terminal_tree_repairs"] == 0
    assert metrics["clean_target_graph_distances_used"] is False
    assert metrics["component_catalog_ids_used"] is False
    for sample in samples:
        for role_index, offspring in enumerate(sample.offspring):
            assert attached_tree_matches_program(
                offspring,
                node_count=sample.program.node_counts[role_index],
                junction_budget=sample.program.junction_budgets[role_index],
            )
