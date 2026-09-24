"""Exact repeat supervision is invariant to branched serialization order."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from forge.model.repeat_supervision import align_qualified_repeats, aligned_repeat_consistency


def example(symmetric=False):
    # Two identical rooted fragments, but the second visits its two branches in reverse order.
    nodes = np.array([2, 0, 0 if symmetric else 1, 0, 0, 0, 0 if symmetric else 1])
    graph = SimpleNamespace(
        canonical_smiles="N(C(C)C)C(C)C" if symmetric else "N(C(O)C)C(O)C",
        node_states=nodes,
        parents=np.array([0, 0, 1, 1, 0, 4, 4]),
        parent_bonds=np.zeros(7, dtype=np.int64),
    )
    blocks = [
        SimpleNamespace(role="head", start=0, stop=1, atom_count=1),
        SimpleNamespace(role="tail", start=1, stop=4, atom_count=3),
        SimpleNamespace(role="tail", start=4, stop=7, atom_count=3),
    ]
    record = SimpleNamespace(
        graph=graph,
        node_count=7,
        component_blocks=blocks,
        canonical_atom_order=np.array([0, 1, 2, 3, 4, 6, 5]),
        core_position_states=np.array([2, 2, 1, 1, 2, 1, 1]),
    )
    return SimpleNamespace(record=record, source_quantities={"head": 1, "tail": 2})


def test_branched_serializations_match_atoms_and_edges_by_graph_not_position():
    e = example()
    atoms, bonds, counts = align_qualified_repeats(e)
    assert counts["aligned_roles"] == 1
    assert atoms[2] == atoms[6] > 0
    assert atoms[3] == atoms[5] > 0
    assert atoms[2] != atoms[3]
    assert atoms[[0, 1, 4]].tolist() == [0, 0, 0]
    assert bonds[2] == bonds[6] > 0
    assert bonds[3] == bonds[5] > 0
    assert e.record.graph.node_states.tolist() == [2, 0, 1, 0, 0, 0, 1]


def test_symmetric_fragment_abstains_without_removing_record():
    atoms, bonds, counts = align_qualified_repeats(example(symmetric=True))
    assert not atoms.any() and not bonds.any()
    assert len(atoms) == 7
    assert counts["ambiguous_correspondence"] == 1


def test_incompatible_product_atom_states_are_not_aligned():
    e = example()
    e.record.graph.node_states[6] = 3
    atoms, bonds, counts = align_qualified_repeats(e)
    assert not atoms.any() and not bonds.any()
    assert counts["atom_state_mismatch"] == 1


def test_quantity_mismatch_fails():
    e = example()
    e.source_quantities["tail"] = 3
    with pytest.raises(ValueError, match="quantity"):
        align_qualified_repeats(e)


def test_loss_matches_independent_pair_reference_and_has_finite_gradients():
    torch.manual_seed(72)
    atom_groups = torch.tensor([[0, 1, 1, 2, 2], [1, 1, 0, 0, 0]])
    bond_groups = torch.tensor([[0, 0, 2, 2, 0], [0, 0, 0, 0, 0]])
    predictions = {
        "nodes": torch.randn(2, 5, 4, requires_grad=True),
        "parent_bonds": torch.randn(2, 5, 3, requires_grad=True),
    }
    loss, atom_count, bond_count = aligned_repeat_consistency(predictions, atom_groups, bond_groups)
    p, q = predictions["nodes"].softmax(-1), predictions["parent_bonds"].softmax(-1)
    expected = (
        torch.stack(
            [
                (p[0, 1] - p[0, 2]).square().mean(),
                (p[0, 3] - p[0, 4]).square().mean(),
                (p[1, 0] - p[1, 1]).square().mean(),
            ]
        ).mean()
        + (q[0, 2] - q[0, 3]).square().mean()
    )
    torch.testing.assert_close(loss, expected)
    assert (atom_count, bond_count) == (3, 1)
    loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in predictions.values())
    assert predictions["nodes"].grad[0, 0].count_nonzero() == 0
    assert predictions["parent_bonds"].grad[1].count_nonzero() == 0


def test_gold_targets_have_zero_consistency_error_despite_order_difference():
    e = example()
    atoms, bonds, _ = align_qualified_repeats(e)
    logits = torch.full((1, 7, 3), -30.0)
    logits.scatter_(2, torch.tensor(e.record.graph.node_states)[None, :, None], 30)
    loss, _, _ = aligned_repeat_consistency(
        {"nodes": logits, "parent_bonds": torch.zeros(1, 7, 3)},
        torch.tensor(atoms)[None],
        torch.tensor(bonds)[None],
    )
    assert loss.item() == 0
    assert not torch.equal(logits[0, 2], logits[0, 5])  # Legacy positional pairing is wrong.


def test_empty_repeat_supervision_remains_differentiable():
    p = {
        "nodes": torch.randn(1, 3, 2, requires_grad=True),
        "parent_bonds": torch.randn(1, 3, 3, requires_grad=True),
    }
    groups = torch.zeros((1, 3), dtype=torch.long)
    loss, atoms, bonds = aligned_repeat_consistency(p, groups, groups)
    assert loss == atoms == bonds == 0
    loss.backward()
    assert all(v.grad.count_nonzero() == 0 for v in p.values())


def test_exact_supervision_loss_requires_explicit_sidecars():
    from forge.model.compose_lipid_training import compose_lipid_forward_loss

    with pytest.raises(ValueError, match="requires the transformer"):
        compose_lipid_forward_loss(
            None,
            {},
            architecture="sparse_mpnn",
            node_marginal=None,
            bond_marginal=None,
            times=None,
            generator=None,
            semantic_weights={},
            repeat_supervision="exact_fragment",
        )
