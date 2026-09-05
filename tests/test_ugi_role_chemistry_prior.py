from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from forge.model.defog_feasibility import AtomState
from forge.model.ugi_role_chemistry_prior import (
    UgiRoleChemistryPrior,
    UgiRoleChemistryPriorError,
)


def _write_gzip_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    assignments = tmp_path / "assignments.csv.gz"
    atoms = tmp_path / "semantic_atoms.csv.gz"
    bonds = tmp_path / "semantic_bonds.csv.gz"
    assignment_fields = [
        "product_id",
        "canonical_product_smiles",
        "primary_product_fold",
        "is_source_adjudicated_measured_product",
        "amine_head_smiles",
        "oxoester_aldehyde_body_tail_smiles",
        "isocyanide_tail_smiles",
    ]
    _write_gzip_csv(
        assignments,
        assignment_fields,
        [
            {
                "product_id": "train-product",
                "canonical_product_smiles": "CCNCCOCCC",
                "primary_product_fold": "train",
                "is_source_adjudicated_measured_product": "true",
                "amine_head_smiles": "CCN",
                "oxoester_aldehyde_body_tail_smiles": "CCO",
                "isocyanide_tail_smiles": "CCC",
            },
            {
                "product_id": "heldout-product",
                "canonical_product_smiles": "C#CCCCCCCC",
                "primary_product_fold": "heldout",
                "is_source_adjudicated_measured_product": "true",
                "amine_head_smiles": "C#C",
                "oxoester_aldehyde_body_tail_smiles": "CCCC",
                "isocyanide_tail_smiles": "CC",
            },
        ],
    )
    roles = (
        "amine_head",
        "amine_head",
        "amine_head",
        "oxoester_aldehyde_body_tail",
        "oxoester_aldehyde_body_tail",
        "oxoester_aldehyde_body_tail",
        "isocyanide_tail",
        "isocyanide_tail",
        "isocyanide_tail",
    )
    _write_gzip_csv(
        atoms,
        [
            "product_id",
            "product_atom_index",
            "origin_role",
            "is_ugi_core",
            "distance_to_nearest_core",
        ],
        [
            {
                "product_id": "train-product",
                "product_atom_index": str(index),
                "origin_role": role,
                "is_ugi_core": "False",
                "distance_to_nearest_core": str(index + 1),
            }
            for index, role in enumerate(roles)
        ],
    )
    _write_gzip_csv(
        bonds,
        [
            "product_id",
            "begin_atom_index",
            "end_atom_index",
            "bond_type",
            "is_in_ring",
        ],
        [
            {
                "product_id": "train-product",
                "begin_atom_index": str(index),
                "end_atom_index": str(index + 1),
                "bond_type": "SINGLE",
                "is_in_ring": "False",
            }
            for index in range(8)
        ],
    )
    return assignments, atoms, bonds


def test_role_chemistry_prior_uses_only_unique_measured_train_components(tmp_path: Path) -> None:
    assignments, atoms, bonds = _inputs(tmp_path)
    prior = UgiRoleChemistryPrior.from_training_data(
        assignments_path=assignments,
        semantic_atoms_path=atoms,
        semantic_bonds_path=bonds,
        minimum_context_count=1,
    )

    assert dict(prior.representative_components_by_role) == {
        "amine_head": 1,
        "isocyanide_tail": 1,
        "oxoester_aldehyde_body_tail": 1,
    }
    vocabulary = (
        AtomState("C", 0, False, 0),
        AtomState("N", 0, False, 0),
        AtomState("O", 0, False, 0),
    )
    atom_bias = prior.atom_log_bias(
        role="amine_head",
        depth=3,
        degree=2,
        in_ring=False,
        atom_vocabulary=vocabulary,
    )
    assert int(np.argmax(atom_bias)) == 1
    bond_bias = prior.bond_log_bias(
        role="isocyanide_tail",
        depth=1,
        in_ring=False,
        symbols=("C", "C"),
        bond_classes=4,
    )
    assert int(np.argmax(bond_bias)) == 0
    terminal_support = prior.bond_terminal_support_tiers(
        role="isocyanide_tail",
        terminal_offset=0,
        in_ring=False,
        symbols=("C", "C"),
        bond_classes=4,
    )
    assert terminal_support[0] == 4.0

    receipt = prior.to_mapping()
    serialized = json.dumps(receipt, sort_keys=True)
    assert "train-product" not in serialized
    assert "heldout-product" not in serialized
    assert "CCN" not in serialized
    assert receipt["component_identity_conditioning"] is False
    assert receipt["hard_support_changed"] is False
    assert receipt["bond_terminal_context_rows"] > 0
    assert receipt["bond_terminal_coordinate"].startswith("shortest role-induced distance")
    assert receipt["amine_head_arrangement_support"] == {
        "exact_signatures": 1,
        "coarse_signatures": 1,
        "basic_signatures": 1,
        "frequency_weighting": False,
    }
    assert receipt["amine_head_topology_support"] == {
        "exact_signatures": 1,
        "coarse_signatures": 1,
        "frequency_weighting": False,
    }
    assert receipt["role_unsaturation_count_support"] == {
        "amine_head": [[0, 0]],
        "oxoester_aldehyde_body_tail": [[0, 0]],
        "isocyanide_tail": [[0, 0]],
    }
    assert receipt["role_unsaturation_count_frequencies"] == {
        "amine_head": [[0, 0, 1]],
        "oxoester_aldehyde_body_tail": [[0, 0, 1]],
        "isocyanide_tail": [[0, 0, 1]],
    }
    assert receipt["role_unsaturation_pattern_frequencies"] == {
        "amine_head": [[[], [], 1]],
        "oxoester_aldehyde_body_tail": [[[], [], 1]],
        "isocyanide_tail": [[[], [], 1]],
    }
    assert prior.supported_unsaturation_counts("oxoester_aldehyde_body_tail") == ((0, 0),)
    assert prior.unsaturation_count_frequencies("oxoester_aldehyde_body_tail") == ((0, 0, 1),)
    assert prior.unsaturation_pattern_frequencies("oxoester_aldehyde_body_tail") == (
        ((), (), 1),
    )
    matching_topology_tier = prior.amine_head_topology_support_tier(
        nodes=(0, 1, 2),
        depths_by_node={0: 1, 1: 2, 2: 3},
        neighbors={0: {1}, 1: {0, 2}, 2: {1, 3}},
        ring_nodes=set(),
    )
    unsupported_topology_tier = prior.amine_head_topology_support_tier(
        nodes=(0, 1, 2),
        depths_by_node={0: 1, 1: 3, 2: 4},
        neighbors={0: {1}, 1: {0, 2}, 2: {1, 3}},
        ring_nodes=set(),
    )
    assert matching_topology_tier == 2.0
    assert unsupported_topology_tier == 0.0
    matching_tier = prior.amine_head_arrangement_support_tier(
        nodes=(0, 1, 2),
        symbols_by_node={0: "C", 1: "C", 2: "N", 3: "C"},
        depths_by_node={0: 1, 1: 2, 2: 3},
        neighbors={0: {1}, 1: {0, 2}, 2: {1, 3}},
        ring_nodes=set(),
    )
    moved_heteroatom_tier = prior.amine_head_arrangement_support_tier(
        nodes=(0, 1, 2),
        symbols_by_node={0: "C", 1: "N", 2: "C", 3: "C"},
        depths_by_node={0: 1, 1: 2, 2: 3},
        neighbors={0: {1}, 1: {0, 2}, 2: {1, 3}},
        ring_nodes=set(),
    )
    assert matching_tier == 4.0
    assert moved_heteroatom_tier == 0.0
    matching_similarity = prior.amine_head_arrangement_similarity(
        nodes=(0, 1, 2),
        symbols_by_node={0: "C", 1: "C", 2: "N", 3: "C"},
        depths_by_node={0: 1, 1: 2, 2: 3},
        neighbors={0: {1}, 1: {0, 2}, 2: {1, 3}},
        ring_nodes=set(),
    )
    moved_similarity = prior.amine_head_arrangement_similarity(
        nodes=(0, 1, 2),
        symbols_by_node={0: "C", 1: "N", 2: "C", 3: "C"},
        depths_by_node={0: 1, 1: 2, 2: 3},
        neighbors={0: {1}, 1: {0, 2}, 2: {1, 3}},
        ring_nodes=set(),
    )
    assert matching_similarity == 1.0
    assert 0.0 < moved_similarity < matching_similarity


def test_role_chemistry_prior_fails_closed_on_changed_input(tmp_path: Path) -> None:
    assignments, atoms, bonds = _inputs(tmp_path)
    with pytest.raises(UgiRoleChemistryPriorError, match="assignments changed"):
        UgiRoleChemistryPrior.from_training_data(
            assignments_path=assignments,
            semantic_atoms_path=atoms,
            semantic_bonds_path=bonds,
            expected_assignments_sha256="0" * 64,
        )


def test_role_chemistry_prior_preserves_fine_bond_depth_without_sparsifying_atoms(
    tmp_path: Path,
) -> None:
    assignments, atoms, bonds = _inputs(tmp_path)
    prior = UgiRoleChemistryPrior.from_training_data(
        assignments_path=assignments,
        semantic_atoms_path=atoms,
        semantic_bonds_path=bonds,
        maximum_depth_bucket=6,
        maximum_bond_depth_bucket=64,
        minimum_context_count=1,
    )

    assert max(int(entry[1]) for entry in prior.atom_counts) == 6
    assert max(int(entry[1]) for entry in prior.bond_counts) == 8
    assert prior.to_mapping()["maximum_depth_bucket"] == 6
    assert prior.to_mapping()["maximum_bond_depth_bucket"] == 64
