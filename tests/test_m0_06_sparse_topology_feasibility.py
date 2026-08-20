from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

torch = pytest.importorskip("torch")

from forge.design.flow.defog_feasibility import AtomState, sha256_file  # noqa: E402
from forge.design.flow.sparse_topology_feasibility import (  # noqa: E402
    INDEX_TO_DENSE_BOND,
    SparseTopologyFlowProbe,
    _connected,
    _maximum_valence_units,
    _parent_candidate_mask,
    build_sparse_atom_vocabulary,
    collate_sparse_records,
    pointer_rstar_step,
    sample_pointer_interpolation,
    sample_sparse_endpoints,
    sparse_constitutional_roundtrip_exact,
    sparse_roundtrip_exact,
    tensorize_sparse_row,
)

REPO = Path(__file__).resolve().parents[1]


def _row(structure_id: str, smiles: str) -> dict[str, str]:
    molecule = Chem.MolFromSmiles(smiles)
    return {
        "r0_structure_id": structure_id,
        "canonical_isomeric_smiles": smiles,
        "heavy_atoms": str(molecule.GetNumHeavyAtoms()),
        "elements": "|".join(sorted({atom.GetSymbol() for atom in molecule.GetAtoms()})),
    }


def test_tree_plus_closures_roundtrips_ring_and_branch_graph() -> None:
    row = _row("ring", "CC1=CC=CC=C1N")
    vocabulary = (
        AtomState("C", 0, False),
        AtomState("N", 0, False),
    )
    record = tensorize_sparse_row(row, {state: index for index, state in enumerate(vocabulary)})

    assert record.node_count == 8
    assert record.closure_count == 1
    assert np.all(record.parents[1:] < np.arange(1, record.node_count))
    assert _connected(record.edges)
    assert sparse_roundtrip_exact(record)


@pytest.mark.parametrize(
    "smiles",
    (
        "FC1=CC=CC=C1N",
        "C[Si](C)(C)O[Si](C)(C)CCN",
        "C1=CC2=CC=CC=C2C=C1",
    ),
)
def test_constitutional_roundtrip_restores_aromatic_and_extended_elements(
    smiles: str,
) -> None:
    row = _row("extended", smiles)
    vocabulary = build_sparse_atom_vocabulary(
        [row],
        {"C", "N", "O", "F", "Si"},
    )
    record = tensorize_sparse_row(
        row,
        {state: index for index, state in enumerate(vocabulary)},
    )

    assert sparse_roundtrip_exact(record)
    assert sparse_constitutional_roundtrip_exact(record, vocabulary)


def test_explicit_aromatic_sparse_state_roundtrips_aromatic_nh() -> None:
    row = _row("pyrrole", "c1cc[nH]c1")
    vocabulary = build_sparse_atom_vocabulary(
        [row],
        {"C", "N"},
        preserve_aromaticity=True,
    )
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}
    record = tensorize_sparse_row(
        row,
        atom_to_index,
        preserve_aromaticity=True,
    )

    assert AtomState("N", 0, True, 1) in vocabulary
    assert 3 in record.parent_bonds or 3 in record.closure_bonds
    assert INDEX_TO_DENSE_BOND[3] == 4
    assert sparse_roundtrip_exact(record)
    assert sparse_constitutional_roundtrip_exact(record, vocabulary)


def test_pointer_noise_and_rstar_respect_variable_candidate_support() -> None:
    node_mask = torch.tensor([[True, True, True, True]])
    child_mask = node_mask.clone()
    child_mask[:, 0] = False
    candidates = _parent_candidate_mask(node_mask)
    clean = torch.tensor([[0, 0, 1, 1]])
    generator = torch.Generator().manual_seed(9)

    noisy = sample_pointer_interpolation(
        clean,
        candidates,
        child_mask,
        torch.zeros(1),
        generator,
    )
    logits = torch.randn((1, 4, 4), generator=generator)
    stepped = pointer_rstar_step(
        noisy,
        logits,
        candidates,
        child_mask,
        0.4,
        0.1,
        generator,
    )

    assert stepped[0, 0] == 0
    assert all(0 <= int(stepped[0, child]) < child for child in range(1, 4))


def test_pointer_rstar_never_transitions_to_masked_targets() -> None:
    node_mask = torch.tensor([[True, True, True, True]])
    child_mask = node_mask.clone()
    child_mask[:, 0] = False
    candidates = _parent_candidate_mask(node_mask)
    current = torch.tensor([[0, 0, 0, 0]])
    logits = torch.full((1, 4, 4), -20.0)
    logits[0, 2, 1] = 20.0
    logits[0, 3, 2] = 20.0

    for seed in range(64):
        stepped = pointer_rstar_step(
            current,
            logits,
            candidates,
            child_mask,
            0.4,
            0.9,
            torch.Generator().manual_seed(seed),
        )
        assert all(bool(candidates[0, child, int(stepped[0, child])]) for child in range(1, 4))


def test_sparse_model_shapes_do_not_create_dense_edge_hidden_states() -> None:
    model = SparseTopologyFlowProbe(
        node_classes=5,
        bond_classes=3,
        hidden_dim=16,
        layers=2,
        maximum_closures=4,
        dropout=0.0,
    )
    nodes = torch.zeros((2, 12), dtype=torch.long)
    parents = torch.arange(12)[None, :].repeat(2, 1)
    parents[:, 1:] -= 1
    bonds = torch.zeros_like(nodes)
    closure_left = torch.zeros((2, 4), dtype=torch.long)
    closure_right = torch.ones_like(closure_left)
    node_mask = torch.ones_like(nodes, dtype=torch.bool)
    child_mask = node_mask.clone()
    child_mask[:, 0] = False

    output = model(
        nodes,
        parents,
        bonds,
        closure_left,
        closure_right,
        torch.tensor([0.2, 0.7]),
        node_mask,
        child_mask,
    )

    assert output["nodes"].shape == (2, 12, 5)
    assert output["parents"].shape == (2, 12, 12)
    assert output["parent_bonds"].shape == (2, 12, 3)
    assert output["closure_left"].shape == (2, 4, 12)
    assert output["closure_right"].shape == (2, 4, 12)
    assert output["closure_bonds"].shape == (2, 4, 3)


def test_collation_preserves_all_sparse_state_masks() -> None:
    vocabulary = (
        AtomState("C", 0, False),
        AtomState("N", 0, False),
    )
    records = [
        tensorize_sparse_row(
            _row("chain", "CCCCN"),
            {state: index for index, state in enumerate(vocabulary)},
        ),
        tensorize_sparse_row(
            _row("ring", "C1CCNCC1"),
            {state: index for index, state in enumerate(vocabulary)},
        ),
    ]
    batch = collate_sparse_records(records, 12, 4)

    assert batch["node_mask"].sum() == 11
    assert batch["child_mask"].sum() == 9
    assert batch["closure_mask"].sum() == 1
    assert batch["closure_left"].shape == (2, 4)


def test_endpoint_sampling_handles_batches_with_zero_closures() -> None:
    vocabulary = (
        AtomState("C", 0, False),
        AtomState("N", 0, False),
    )
    record = tensorize_sparse_row(
        _row("chain", "CCCCN"),
        {state: index for index, state in enumerate(vocabulary)},
    )
    assert record.closure_count == 0
    model = SparseTopologyFlowProbe(
        node_classes=2,
        bond_classes=3,
        hidden_dim=8,
        layers=1,
        maximum_closures=2,
        dropout=0.0,
    )

    samples, metrics = sample_sparse_endpoints(
        model,
        [record],
        vocabulary,
        np.asarray([0.8, 0.2]),
        np.asarray([0.9, 0.09, 0.01]),
        n_max=8,
        maximum_closures=2,
        sample_count=2,
        sample_steps=2,
        batch_size=2,
        seed=17,
    )

    assert len(samples) == 2
    assert metrics["terminal_constraint_events"]["requested_closures"] == 0


def test_valence_policy_covers_declared_flat_atom_states() -> None:
    states = (
        AtomState("C", 0, False),
        AtomState("N", 0, False),
        AtomState("N", 1, False),
        AtomState("O", 0, False),
        AtomState("O", -1, False),
        AtomState("P", 0, False),
        AtomState("S", 0, False),
        AtomState("F", 0, False),
        AtomState("Si", 0, False),
    )
    assert [_maximum_valence_units(state) for state in states] == [
        8,
        6,
        8,
        4,
        2,
        10,
        12,
        2,
        8,
    ]


@pytest.mark.needs_vendor
def test_full_r0_sparse_representation_roundtrips_extended_elements_and_aromaticity() -> None:
    with gzip.open(
        REPO / "results/m0_03/r0_constitutional.csv.gz",
        "rt",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    declared_elements = {"C", "N", "O", "S", "P", "F", "Si"}
    vocabulary = build_sparse_atom_vocabulary(rows, declared_elements)
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}
    eligible = [row for row in rows if set(row["elements"].split("|")) <= declared_elements]
    records = [tensorize_sparse_row(row, atom_to_index) for row in eligible]
    aromatic_records = [
        record
        for record in records
        if any(
            atom.GetIsAromatic() for atom in Chem.MolFromSmiles(record.canonical_smiles).GetAtoms()
        )
    ]

    assert len(records) == 15_229
    assert all(sparse_roundtrip_exact(record) for record in records)
    assert all(sparse_constitutional_roundtrip_exact(record, vocabulary) for record in records)
    assert len(aromatic_records) == 1_891
    assert all(
        sparse_constitutional_roundtrip_exact(record, vocabulary) for record in aromatic_records
    )
    assert max(record.closure_count for record in records) == 12
    assert all(
        len(Chem.GetMolFrags(Chem.MolFromSmiles(row["canonical_isomeric_smiles"]))) == 1
        for row in eligible
    )


@pytest.mark.needs_vendor
def test_frozen_sparse_result_contract_when_present() -> None:
    result_path = REPO / "results/m0_06_sparse/result.json"
    if not result_path.exists():
        pytest.skip("bounded sparse result has not been generated")
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == ("m0_06_sparse_topology_feasibility_result.v1")
    assert result["representation_audit"]["roundtrip_fraction"] == 1.0
    assert result["representation_audit"]["maximum_observed_cycle_rank"] == 12
    assert {run["n_max"] for run in result["runs"]} == {64, 96}
    assert {row["n_max"] for row in result["scaling_dry_runs"]} == {
        128,
        192,
        282,
    }
    for record in result["inputs"].values():
        path = REPO / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]
    config_path = REPO / result["config"]["path"]
    assert sha256_file(config_path) == result["config"]["sha256"]


@pytest.mark.needs_vendor
def test_frozen_full_support_result_contract_when_present() -> None:
    result_path = REPO / "results/m0_06_sparse_full_support/result.json"
    if not result_path.exists():
        pytest.skip("bounded full-support sparse result has not been generated")
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == ("m0_06_sparse_topology_feasibility_result.v1")
    audit = result["representation_audit"]
    assert audit["eligible_single_fragment_rows"] == 15_229
    assert audit["roundtrip_exact_rows"] == 15_229
    assert audit["constitutional_roundtrip_exact_rows"] == 15_229
    assert audit["aromatic_rows"] == 1_891
    assert audit["aromatic_constitutional_roundtrip_exact_rows"] == 1_891
    assert result["support_audit"]["rows_outside_declared_element_vocabulary"] == 0
    assert {record["symbol"] for record in result["atom_vocabulary"]} == {
        "C",
        "N",
        "O",
        "S",
        "P",
        "F",
        "Si",
    }
    assert result["decision"]["accepted_for_production_architecture_implementation"]
    assert result["decision"]["checks"]["constitutional_roundtrip_pass"]
    for record in result["inputs"].values():
        path = REPO / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]
    config_path = REPO / result["config"]["path"]
    assert sha256_file(config_path) == result["config"]["sha256"]
