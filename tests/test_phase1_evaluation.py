from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")

from experiments.phase1.product_l1.sampling.phase1_evaluation import (  # noqa: E402
    evaluate_product_samples,
    training_record_edges,
)
from forge.model.defog_feasibility import AtomState  # noqa: E402
from forge.model.phase1_flow import TrainingGraphRecord  # noqa: E402


def _record(
    structure_id: str,
    smiles: str,
    nodes: list[int],
    parents: list[int],
) -> TrainingGraphRecord:
    return TrainingGraphRecord(
        structure_id=structure_id,
        canonical_smiles=smiles,
        node_states=np.asarray(nodes, dtype=np.int64),
        parents=np.asarray(parents, dtype=np.int64),
        parent_bonds=np.zeros(len(nodes), dtype=np.int64),
        closure_left=np.asarray([], dtype=np.int64),
        closure_right=np.asarray([], dtype=np.int64),
        closure_bonds=np.asarray([], dtype=np.int64),
    )


def test_training_record_edges_reconstructs_tree_and_closure() -> None:
    record = TrainingGraphRecord(
        structure_id="ring",
        canonical_smiles="C1CCC1",
        node_states=np.zeros(4, dtype=np.int64),
        parents=np.asarray([0, 0, 1, 2], dtype=np.int64),
        parent_bonds=np.zeros(4, dtype=np.int64),
        closure_left=np.asarray([0], dtype=np.int64),
        closure_right=np.asarray([3], dtype=np.int64),
        closure_bonds=np.asarray([0], dtype=np.int64),
    )

    edges = training_record_edges(record)

    assert np.count_nonzero(np.triu(edges, 1)) == 4
    assert edges[0, 3] == edges[3, 0] != 0


def test_product_evaluation_reports_novelty_and_branch_fidelity() -> None:
    vocabulary = (
        AtomState("C", 0, False),
        AtomState("N", 0, False),
    )
    train = (
        _record("chain", "CCCC", [0, 0, 0, 0], [0, 0, 1, 2]),
        _record("amine", "CCN", [0, 0, 1], [0, 0, 1]),
    )
    chain_edges = training_record_edges(train[0])
    novel_edges = np.zeros((4, 4), dtype=np.int64)
    novel_edges[0, 1] = novel_edges[1, 0] = 1
    novel_edges[0, 2] = novel_edges[2, 0] = 1
    novel_edges[0, 3] = novel_edges[3, 0] = 1

    result = evaluate_product_samples(
        (
            (np.asarray([0, 0, 0, 0]), chain_edges),
            (np.asarray([1, 0, 0, 0]), novel_edges),
        ),
        train,
        vocabulary,
    )

    assert result["validity"] == 1.0
    assert result["connectedness"] == 1.0
    assert result["exact_train_memorization_among_valid"] == 0.5
    assert result["exact_train_novelty_among_valid"] == 0.5
    assert result["generated"]["branch_atom_fraction"] == pytest.approx(1 / 8)
    assert set(result["generated"]["lipid_regions"]) == {"head", "interface", "tail"}
    assert "path_like_tail_component_fraction" in result["generated"]["morphology"]
    assert "regional_absolute_errors" in result["distribution_fidelity"]
    assert "morphology_absolute_errors" in result["distribution_fidelity"]
    assert result["distribution_fidelity"]["absolute_branch_atom_fraction_error"] > 0
    assert result["macrocycle_minimum_ring_size"] == 9


def test_product_evaluation_reports_nonaromatic_unsaturation() -> None:
    vocabulary = (AtomState("C", 0, False),)
    record = TrainingGraphRecord(
        structure_id="alkene",
        canonical_smiles="C=C",
        node_states=np.zeros(2, dtype=np.int64),
        parents=np.asarray([0, 0], dtype=np.int64),
        parent_bonds=np.asarray([0, 1], dtype=np.int64),
        closure_left=np.asarray([], dtype=np.int64),
        closure_right=np.asarray([], dtype=np.int64),
        closure_bonds=np.asarray([], dtype=np.int64),
    )
    edges = training_record_edges(record)

    result = evaluate_product_samples(((record.node_states, edges),), (record,), vocabulary)

    assert result["generated"]["nonaromatic_unsaturated_bond_fraction"] == 1.0
    assert result["generated"]["double_bond_fraction"] == 1.0
    assert result["generated"]["triple_bond_fraction"] == 0.0
    assert (
        result["distribution_fidelity"]["absolute_nonaromatic_unsaturated_bond_fraction_error"]
        == 0.0
    )
