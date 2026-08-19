from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forge.product.phase1_tree_topology_flow import (  # noqa: E402
    OffspringTreeFlow,
    TreeTopologyFlowError,
    collate_tree_records,
    noise_tree_batch,
    offspring_marginal,
    offspring_to_parents,
    preorder_offspring_to_parents,
    preorder_record_offspring_counts,
    record_offspring_counts,
    sample_tree_topologies,
    sample_valid_offspring,
    tree_flow_loss,
    tree_topology_statistics,
)


@dataclass(frozen=True)
class _Record:
    parents: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.parents.size)


def test_bfs_offspring_encoding_roundtrips_chain_and_branch() -> None:
    chain = _Record(np.asarray([0, 0, 1, 2, 3]))
    branch = _Record(np.asarray([0, 0, 0, 1, 1]))

    for record in (chain, branch):
        offspring = record_offspring_counts(record)
        restored = offspring_to_parents(offspring)
        assert restored.tolist() == record.parents.tolist()

    assert record_offspring_counts(chain).tolist() == [1, 1, 1, 1, 0]
    assert record_offspring_counts(branch).tolist() == [2, 2, 0, 0, 0]


def test_non_bfs_parent_sequence_is_rejected() -> None:
    record = _Record(np.asarray([0, 0, 1, 0]))

    with pytest.raises(TreeTopologyFlowError, match="breadth first"):
        record_offspring_counts(record)


def test_preorder_offspring_encoding_roundtrips_nested_branches() -> None:
    record = _Record(np.asarray([0, 0, 1, 1, 0, 4, 4]))

    offspring = preorder_record_offspring_counts(record)
    restored = preorder_offspring_to_parents(offspring)

    assert offspring.tolist() == [2, 2, 0, 0, 2, 0, 0]
    assert restored.tolist() == record.parents.tolist()


def test_invalid_preorder_word_is_rejected() -> None:
    with pytest.raises(TreeTopologyFlowError, match="exhausts tree early"):
        preorder_offspring_to_parents(np.asarray([0, 2, 0]))


def test_preorder_decoder_is_exact_for_every_small_valid_word() -> None:
    for node_count in range(1, 7):
        valid_words = 0
        for raw_word in product(range(node_count), repeat=node_count):
            offspring = np.asarray(raw_word, dtype=np.int64)
            if int(offspring.sum()) != node_count - 1:
                continue
            pending = 1 + np.cumsum(offspring - 1)
            if np.any(pending[:-1] <= 0) or int(pending[-1]) != 0:
                continue
            parents = preorder_offspring_to_parents(offspring)
            restored = preorder_record_offspring_counts(_Record(parents))
            assert np.array_equal(restored, offspring)
            valid_words += 1
        assert valid_words >= 1


def test_singleton_offspring_word_decodes_in_both_languages() -> None:
    offspring = np.asarray([0], dtype=np.int64)

    assert offspring_to_parents(offspring).tolist() == [0]
    assert preorder_offspring_to_parents(offspring).tolist() == [0]


def test_structured_terminal_sampler_returns_one_valid_tree() -> None:
    logits = torch.full((6, 3), -8.0)
    logits[:-1, 1] = 8.0
    logits[-1, 0] = 8.0

    offspring = sample_valid_offspring(
        logits,
        generator=torch.Generator().manual_seed(11),
    )
    parents = offspring_to_parents(offspring)

    assert offspring.tolist() == [1, 1, 1, 1, 1, 0]
    assert parents.tolist() == [0, 0, 1, 2, 3, 4]


def test_tree_flow_trains_and_samples_valid_bfs_topologies() -> None:
    records = (
        _Record(np.asarray([0, 0, 1, 2, 3])),
        _Record(np.asarray([0, 0, 0, 1, 1])),
    )
    clean = collate_tree_records(
        records,
        maximum_nodes=5,
        maximum_children=2,
    )
    source_array = offspring_marginal(
        records,
        maximum_children=2,
    )
    source = torch.as_tensor(source_array, dtype=torch.float32)
    model = OffspringTreeFlow(
        maximum_children=2,
        hidden_dim=32,
        layers=1,
        attention_heads=4,
        maximum_heavy_atoms=16,
        dropout=0.0,
    )
    generator = torch.Generator().manual_seed(13)
    t = torch.tensor([0.25, 0.75])
    noisy = noise_tree_batch(clean, source, t, generator)
    predictions = model(noisy["offspring"], t, clean["node_mask"])
    loss, components = tree_flow_loss(predictions, clean)
    loss.backward()
    node_counts = np.zeros(17)
    node_counts[5] = 1.0

    samples, runtime = sample_tree_topologies(
        model,
        source_array,
        node_counts,
        sample_count=4,
        sample_steps=2,
        batch_size=2,
        seed=17,
        device="cpu",
    )
    statistics = tree_topology_statistics(samples, maximum_children=2)

    assert components["total"] >= 0.0
    assert runtime["terminal_interventions"] == {}
    assert len(samples) == 4
    assert all(sample.offspring.sum() == sample.node_count - 1 for sample in samples)
    assert all(np.all(sample.parents[1:] < np.arange(1, sample.node_count)) for sample in samples)
    assert 0.0 <= statistics["branch_atom_fraction"] <= 1.0
