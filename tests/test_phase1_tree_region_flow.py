from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forge.design.flow.phase1_tree_region_flow import (  # noqa: E402
    ConditionalTreeRegionFlow,
    collate_region_records,
    noise_region_batch,
    region_depth_marginal,
    region_flow_loss,
    region_transition_marginal,
    sample_tree_regions,
    tree_region_statistics,
)


@dataclass(frozen=True)
class _Record:
    parents: np.ndarray
    region_states: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.parents.size)


def _records() -> tuple[_Record, ...]:
    return (
        _Record(
            parents=np.asarray([0, 0, 1, 2, 3]),
            region_states=np.asarray([0, 0, 1, 2, 2]),
        ),
        _Record(
            parents=np.asarray([0, 0, 0, 1, 1]),
            region_states=np.asarray([0, 1, 1, 2, 2]),
        ),
    )


def test_region_sources_capture_depth_and_parent_transition() -> None:
    depth = region_depth_marginal(
        _records(),
        depth_buckets=8,
        region_classes=3,
    )
    transition = region_transition_marginal(
        _records(),
        region_classes=3,
    )

    assert depth.shape == (8, 3)
    assert transition.shape == (3, 3)
    assert np.allclose(depth.sum(axis=1), 1.0)
    assert np.allclose(transition.sum(axis=1), 1.0)
    assert depth[0, 0] > 0.99


def test_conditional_region_flow_trains_and_samples_on_fixed_tree() -> None:
    records = _records()
    clean = collate_region_records(records, maximum_nodes=5)
    depth_array = region_depth_marginal(
        records,
        depth_buckets=8,
        region_classes=3,
    )
    transition = region_transition_marginal(records, region_classes=3)
    depth = torch.as_tensor(depth_array, dtype=torch.float32)
    model = ConditionalTreeRegionFlow(
        region_classes=3,
        hidden_dim=32,
        layers=1,
        attention_heads=4,
        maximum_heavy_atoms=16,
        depth_buckets=8,
        maximum_degree=6,
        dropout=0.0,
    )
    generator = torch.Generator().manual_seed(29)
    t = torch.tensor([0.25, 0.75])
    noisy = noise_region_batch(clean, depth, t, generator)
    logits = model(
        noisy,
        clean["parents"],
        t,
        clean["node_mask"],
        clean["child_mask"],
    )
    loss, components = region_flow_loss(logits, clean)
    loss.backward()

    samples, runtime = sample_tree_regions(
        model,
        [record.parents for record in records],
        depth_array,
        transition,
        sample_steps=2,
        batch_size=2,
        transition_strength=1.0,
        seed=31,
        device="cpu",
    )
    statistics = tree_region_statistics(samples)

    assert components["total"] >= 0.0
    assert runtime["terminal_interventions"] == {}
    assert all(sample.regions[0] == 0 for sample in samples)
    assert statistics["samples"] == 2
