from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.phase1.product_l1.sampling.phase1_generation import (  # noqa: E402
    _closure_pair_log_bias,
    _construct_product_terminal_graph,
    _degree_continuation_log_bias,
    _enforce_aromatic_cycle_consistency,
    load_product_checkpoint,
    maximum_likelihood_count_distributions,
    sample_product_endpoints,
)
from forge.model.defog_feasibility import AtomState, _model_state_sha256  # noqa: E402
from forge.model.phase1_flow import Phase1FlowError, SparseWholeLipidFlow  # noqa: E402


def _model() -> SparseWholeLipidFlow:
    return SparseWholeLipidFlow(
        node_classes=3,
        hidden_dim=12,
        layers=1,
        maximum_closures=2,
        maximum_heavy_atoms=8,
        dropout=0.0,
    )


def _package(model: SparseWholeLipidFlow) -> dict:
    return {
        "schema_version": "phase1_product_pretrain_checkpoint.v2",
        "trusted_local_checkpoint": True,
        "model_state_sha256": _model_state_sha256(model),
        "model_config": {
            "hidden_dim": 12,
            "layers": 1,
            "maximum_closure_slots": 2,
            "maximum_heavy_atoms": 8,
            "dropout": 0.0,
        },
        "atom_vocabulary": [
            {"symbol": "C", "formal_charge": 0, "aromatic": False},
            {"symbol": "N", "formal_charge": 0, "aromatic": False},
            {"symbol": "O", "formal_charge": 0, "aromatic": False},
        ],
        "node_marginal": [0.7, 0.2, 0.1],
        "bond_marginal": [0.8, 0.15, 0.05],
        "model_state_dict": model.state_dict(),
    }


def test_checkpoint_loader_reconstructs_exact_model(tmp_path: Path) -> None:
    model = _model()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(_package(model), checkpoint)

    loaded, vocabulary, node_marginal, bond_marginal, package = load_product_checkpoint(
        checkpoint,
        device="cpu",
    )

    assert _model_state_sha256(loaded) == package["model_state_sha256"]
    assert vocabulary == (
        AtomState("C", 0, False),
        AtomState("N", 0, False),
        AtomState("O", 0, False),
    )
    assert node_marginal.tolist() == pytest.approx([0.7, 0.2, 0.1])
    assert bond_marginal.tolist() == pytest.approx([0.8, 0.15, 0.05])


def test_checkpoint_loader_rejects_untrusted_package(tmp_path: Path) -> None:
    model = _model()
    package = _package(model)
    package["trusted_local_checkpoint"] = False
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(package, checkpoint)

    with pytest.raises(Phase1FlowError, match="trusted"):
        load_product_checkpoint(checkpoint, device="cpu")


def test_product_sampling_is_seed_deterministic_and_uses_learned_counts() -> None:
    model = _model()
    with torch.no_grad():
        model.node_count_logits.fill_(-20.0)
        model.node_count_logits[5] = 20.0
        model.closure_count_logits.fill_(-20.0)
        model.closure_count_logits[0] = 20.0
    vocabulary = (
        AtomState("C", 0, False),
        AtomState("N", 0, False),
        AtomState("O", 0, False),
    )
    kwargs = {
        "sample_count": 4,
        "sample_steps": 3,
        "batch_size": 2,
        "seed": 31,
        "device": "cpu",
    }

    first, first_profile = sample_product_endpoints(
        model,
        vocabulary,
        np.asarray([0.7, 0.2, 0.1]),
        np.asarray([0.8, 0.15, 0.05]),
        **kwargs,
    )
    second, second_profile = sample_product_endpoints(
        model,
        vocabulary,
        np.asarray([0.7, 0.2, 0.1]),
        np.asarray([0.8, 0.15, 0.05]),
        **kwargs,
    )

    assert [len(nodes) for nodes, _ in first] == [5, 5, 5, 5]
    assert all(np.array_equal(left[0], right[0]) for left, right in zip(first, second, strict=True))
    assert all(np.array_equal(left[1], right[1]) for left, right in zip(first, second, strict=True))
    for key in (
        "samples",
        "sampling_steps",
        "sampled_node_count_mean",
        "sampled_closure_count_mean",
        "terminal_constraint_events",
        "terminal_constraint_repair_fraction",
    ):
        assert first_profile[key] == second_profile[key]


def test_terminal_decoder_does_not_choose_high_valence_atoms_to_fit_topology() -> None:
    vocabulary = (
        AtomState("C", 0, False),
        AtomState("O", 0, False),
        AtomState("S", 0, False),
    )
    node_logits = torch.full((1, 4, 3), -20.0)
    node_logits[:, :, 1] = 20.0
    parent_logits = torch.full((1, 4, 4), -20.0)
    parent_logits[0, 1, 0] = 20.0
    parent_logits[0, 2, 1] = 20.0
    parent_logits[0, 3, 2] = 20.0
    predictions = {
        "nodes": node_logits,
        "parents": parent_logits,
        "parent_bonds": torch.tensor(
            [[[20.0, -20.0, -20.0]] * 4],
        ),
        "closure_left": torch.zeros((1, 1, 4)),
        "closure_right": torch.zeros((1, 1, 4)),
        "closure_bonds": torch.zeros((1, 1, 3)),
    }

    nodes, edges, repairs = _construct_product_terminal_graph(
        predictions,
        0,
        4,
        0,
        vocabulary,
        1,
        torch.Generator().manual_seed(7),
    )

    assert nodes.tolist() == [1, 1, 1, 1]
    assert np.count_nonzero(np.triu(edges, 1)) == 3
    assert repairs == {"requested_closures": 0, "realized_closures": 0}


def test_terminal_decoder_preserves_clean_bfs_parent_support() -> None:
    vocabulary = (AtomState("C", 0, False),)
    predictions = {
        "nodes": torch.zeros((1, 6, 1)),
        "parents": torch.tensor(
            [
                [
                    [0.0] * 6,
                    [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ),
        "parent_bonds": torch.tensor([[[20.0, -20.0, -20.0]] * 6]),
        "closure_left": torch.zeros((1, 1, 6)),
        "closure_right": torch.zeros((1, 1, 6)),
        "closure_bonds": torch.zeros((1, 1, 3)),
    }

    _, edges, _ = _construct_product_terminal_graph(
        predictions,
        0,
        6,
        0,
        vocabulary,
        1,
        torch.Generator().manual_seed(3),
    )
    parents = []
    for child in range(1, 6):
        candidates = np.flatnonzero(edges[child, :child])
        assert candidates.size == 1
        parents.append(int(candidates[0]))
    assert parents == sorted(parents)


def test_closure_pair_prior_corrects_endpoint_multiplicity() -> None:
    from forge.model.lipid_context import tree_pair_ring_sizes

    parents = np.asarray([0, 0, 1, 2, 3, 4], dtype=np.int64)
    node_count = len(parents)
    valid_pairs = torch.triu(
        torch.ones((node_count, node_count), dtype=torch.bool),
        diagonal=1,
    )
    for child in range(1, node_count):
        valid_pairs[parents[child], child] = False
    target = torch.full((9,), float(np.log(1e-4)))
    target[3] = float(np.log(0.7))
    target[4] = float(np.log(0.2))
    target[5] = float(np.log(0.08))
    target[6] = float(np.log(0.02))

    bias = _closure_pair_log_bias(
        parents,
        valid_pairs,
        target,
        strength=1.0,
    )
    probabilities = torch.softmax(bias[valid_pairs], dim=0)
    ring_sizes = torch.as_tensor(tree_pair_ring_sizes(parents))
    realized = {
        size: float(probabilities[ring_sizes[valid_pairs] == size].sum()) for size in (3, 4, 5, 6)
    }

    assert realized == pytest.approx({3: 0.7, 4: 0.2, 5: 0.08, 6: 0.02})
    assert torch.isfinite(bias[valid_pairs]).all()


def test_region_topology_bias_discourages_tail_branching() -> None:
    regions = np.asarray([0, 0, 2, 2], dtype=np.int64)
    degrees = np.asarray([2, 1, 2, 1], dtype=np.int64)
    degree_prior = np.log(
        np.asarray(
            [
                [1.0, 1.0, 0.8, 0.1, 0.01],
                [1.0, 1.0, 0.5, 0.1, 0.01],
                [1.0, 1.0, 0.05, 0.01, 0.01],
            ]
        )
    )
    degree_prior = np.pad(
        degree_prior,
        ((0, 0), (0, 1)),
        constant_values=-np.inf,
    )
    degree_bias = _degree_continuation_log_bias(
        regions,
        degrees,
        degree_prior,
        strength=1.0,
        device=torch.device("cpu"),
    )

    assert degree_bias[0] > degree_bias[2]


def test_aromaticity_is_decoded_as_one_consistent_cycle() -> None:
    vocabulary = (
        AtomState("C", 0, False),
        AtomState("C", 0, True),
    )
    parents = np.asarray([0, 0, 1, 2, 3, 4], dtype=np.int64)
    node_states = np.zeros(6, dtype=np.int64)
    edges = np.zeros((6, 6), dtype=np.int64)
    for child in range(1, 6):
        edges[child, parents[child]] = edges[parents[child], child] = 1
    edges[0, 5] = edges[5, 0] = 4
    predictions = {
        "nodes": torch.tensor([[[0.0, 8.0]] * 6]),
        "parent_bonds": torch.tensor([[[0.0, 0.0, 0.0, 8.0]] * 6]),
        "closure_bonds": torch.tensor([[[0.0, 0.0, 0.0, 8.0]]]),
    }
    repairs = Counter()

    _enforce_aromatic_cycle_consistency(
        node_states,
        edges,
        parents,
        [(0, 5, 0)],
        predictions,
        0,
        vocabulary,
        torch.Generator().manual_seed(5),
        allowed_cycle_sizes=frozenset({5, 6}),
        probability_threshold=0.5,
        repairs=repairs,
    )

    assert node_states.tolist() == [1] * 6
    assert set(edges[np.triu_indices(6, 1)]) == {0, 4}
    assert repairs == {"aromatic_cycle_size_6": 1}


def test_count_mle_uses_only_observed_support_and_overrides_slow_count_head() -> None:
    node_distribution, closure_distribution = maximum_likelihood_count_distributions(
        [4, 4, 5, 5],
        [0, 0, 1, 1],
        maximum_heavy_atoms=8,
        maximum_closures=2,
    )
    assert node_distribution.tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]
    )
    assert closure_distribution.tolist() == pytest.approx([0.5, 0.5, 0.0])

    model = _model()
    with torch.no_grad():
        model.node_count_logits.fill_(-20.0)
        model.node_count_logits[8] = 20.0
        model.closure_count_logits.fill_(-20.0)
        model.closure_count_logits[2] = 20.0
    vocabulary = (
        AtomState("C", 0, False),
        AtomState("N", 0, False),
        AtomState("O", 0, False),
    )
    _, profile = sample_product_endpoints(
        model,
        vocabulary,
        np.asarray([0.7, 0.2, 0.1]),
        np.asarray([0.8, 0.15, 0.05]),
        sample_count=16,
        sample_steps=3,
        batch_size=4,
        seed=37,
        device="cpu",
        node_count_distribution=node_distribution,
        closure_count_distribution=closure_distribution,
    )
    assert 4.0 <= profile["sampled_node_count_mean"] <= 5.0
    assert profile["node_count_source"] == "frozen_training_split_categorical_MLE"
    assert (
        profile["closure_count_source"] == "frozen_training_split_categorical_MLE_with_graph_bound"
    )
