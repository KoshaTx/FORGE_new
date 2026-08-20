from __future__ import annotations

import csv
import gzip
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from experiments.archive.phase1.design_audits.canonical_representation_audit import (
    audit_canonical_representation,
    sparse_record_signature,
)
from forge.model.sparse_topology_feasibility import (
    build_sparse_atom_vocabulary,
)
from forge.model.v5_sparse_representation import (
    tensorize_v5_sparse_molecule,
    tensorize_v5_sparse_row,
    v5_constitutional_roundtrip_exact,
    v5_sparse_program_valid,
)


def _row(structure_id: str, smiles: str) -> dict[str, str]:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return {
        "r0_structure_id": structure_id,
        "canonical_isomeric_smiles": smiles,
        "heavy_atoms": str(molecule.GetNumHeavyAtoms()),
        "elements": "|".join(sorted({atom.GetSymbol() for atom in molecule.GetAtoms()})),
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = [
        _row("symmetric_ring", "NCC1CCCCC1"),
        _row("branched", "CN(C)CCOC(=O)C(C)CCCC"),
    ]
    r0_path = tmp_path / "r0.csv.gz"
    with gzip.open(r0_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    vocabulary = build_sparse_atom_vocabulary(
        rows,
        {"C", "N", "O"},
        preserve_aromaticity=True,
    )
    vocabulary_path = tmp_path / "vocabulary.json"
    vocabulary_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "atom_vocabulary": [
                    {
                        "index": index,
                        "symbol": state.symbol,
                        "formal_charge": state.formal_charge,
                        "aromatic": state.aromatic,
                        "explicit_hydrogens": state.explicit_hydrogens,
                    }
                    for index, state in enumerate(vocabulary)
                ],
            }
        )
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "preserve_aromaticity": True,
                    "root_strategy": "lipid_polar",
                    "region_scheme": "polar_structural_v2",
                    "degree_prior_maximum_degree": 8,
                    "region_classes": 3,
                    "maximum_heavy_atoms": 282,
                    "maximum_closure_slots": 12,
                }
            }
        )
    )
    return r0_path, vocabulary_path, config_path


@pytest.mark.parametrize(
    "tree_traversal",
    ("breadth_first", "breadth_first_tree_preorder", "depth_first_preorder"),
)
def test_direct_atom_renumbering_preserves_complete_encoded_state(tree_traversal: str) -> None:
    row = _row("symmetric", "NCC1CCCCC1")
    vocabulary = build_sparse_atom_vocabulary(
        [row],
        {"C", "N"},
        preserve_aromaticity=True,
    )
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}
    base = tensorize_v5_sparse_row(
        row,
        atom_to_index,
        preserve_aromaticity=True,
        root_strategy="lipid_polar",
        region_scheme="polar_structural_v2",
        tree_traversal=tree_traversal,
    )
    molecule = Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
    assert molecule is not None
    permuted = tensorize_v5_sparse_molecule(
        Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms())))),
        atom_to_index,
        structure_id=row["r0_structure_id"],
        canonical_smiles=row["canonical_isomeric_smiles"],
        preserve_aromaticity=True,
        root_strategy="lipid_polar",
        region_scheme="polar_structural_v2",
        tree_traversal=tree_traversal,
    )

    assert sparse_record_signature(permuted) == sparse_record_signature(base)


def test_full_audit_records_provenance_symmetry_and_determinism(tmp_path: Path) -> None:
    r0_path, vocabulary_path, config_path = _write_inputs(tmp_path)
    first = audit_canonical_representation(
        r0_path,
        vocabulary_path,
        config_path,
        seed=19,
        permutations_per_record=4,
    )
    second = audit_canonical_representation(
        r0_path,
        vocabulary_path,
        config_path,
        seed=19,
        permutations_per_record=4,
    )

    assert first["status"] == "pass"
    assert first["counts"]["records"] == 2
    assert first["counts"]["permutation_encodings"] == 8
    assert first["counts"]["exact_permutation_encodings"] == 8
    assert first["symmetry_strata"]["symmetric"]["records"] >= 1
    assert first["failure_counts"] == {}
    assert all(first["acceptance"].values())
    assert first["inputs"] == second["inputs"]
    assert first["counts"] == second["counts"]
    assert first["size_strata"] == second["size_strata"]
    assert first["symmetry_strata"] == second["symmetry_strata"]


def test_v5_record_is_linear_state_and_signature_covers_sparse_bonds() -> None:
    row = _row("chain", "CCCN")
    vocabulary = build_sparse_atom_vocabulary(
        [row],
        {"C", "N"},
        preserve_aromaticity=True,
    )
    record = tensorize_v5_sparse_row(
        row,
        {state: index for index, state in enumerate(vocabulary)},
        preserve_aromaticity=True,
    )
    assert not hasattr(record, "edges")
    assert "parents" not in record.__dataclass_fields__
    assert v5_sparse_program_valid(record)
    assert v5_constitutional_roundtrip_exact(record, vocabulary)
    modified_parent_bonds = record.parent_bonds.copy()
    modified_parent_bonds[1] = (int(modified_parent_bonds[1]) + 1) % 4
    modified = replace(record, parent_bonds=modified_parent_bonds)

    assert not np.array_equal(record.parent_bonds, modified.parent_bonds)
    assert sparse_record_signature(record) != sparse_record_signature(modified)


def test_v5_validator_rejects_noncanonical_or_out_of_range_state() -> None:
    row = _row("bicyclic", "C1CC2CCC1C2")
    vocabulary = build_sparse_atom_vocabulary(
        [row],
        {"C"},
        preserve_aromaticity=True,
    )
    record = tensorize_v5_sparse_row(
        row,
        {state: index for index, state in enumerate(vocabulary)},
        preserve_aromaticity=True,
        root_strategy="canonical",
        region_scheme="polar_structural_v2",
    )
    assert record.closure_count >= 2
    policy = {
        "atom_vocabulary_size": len(vocabulary),
        "region_classes": 3,
        "maximum_children": 8,
    }
    assert v5_sparse_program_valid(record, **policy)
    assert not v5_sparse_program_valid(
        replace(record, offspring=record.offspring.astype(np.float64)),
        **policy,
    )
    invalid_nodes = record.node_states.copy()
    invalid_nodes[0] = len(vocabulary)
    assert not v5_sparse_program_valid(
        replace(record, node_states=invalid_nodes),
        **policy,
    )
    assert record.region_states is not None
    invalid_regions = record.region_states.copy()
    invalid_regions[0] = 3
    assert not v5_sparse_program_valid(
        replace(record, region_states=invalid_regions),
        **policy,
    )
    assert not v5_sparse_program_valid(
        replace(
            record,
            closure_left=record.closure_left[::-1].copy(),
            closure_right=record.closure_right[::-1].copy(),
            closure_bonds=record.closure_bonds[::-1].copy(),
        ),
        **policy,
    )


@pytest.mark.parametrize(
    "tree_traversal",
    ("breadth_first", "breadth_first_tree_preorder", "depth_first_preorder"),
)
def test_single_node_tree_has_a_valid_zero_offspring_word(tree_traversal: str) -> None:
    row = _row("single", "C")
    vocabulary = build_sparse_atom_vocabulary(
        [row],
        {"C"},
        preserve_aromaticity=True,
    )
    record = tensorize_v5_sparse_row(
        row,
        {state: index for index, state in enumerate(vocabulary)},
        preserve_aromaticity=True,
        root_strategy="canonical",
        tree_traversal=tree_traversal,
    )

    assert record.offspring.tolist() == [0]
    assert record.parents.tolist() == [0]
    assert v5_sparse_program_valid(
        record,
        atom_vocabulary_size=1,
        maximum_children=0,
    )
