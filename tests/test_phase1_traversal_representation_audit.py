from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from rdkit import Chem

from forge.design.audit.traversal_representation_audit import audit_traversal_representations
from forge.design.flow.sparse_topology_feasibility import build_sparse_atom_vocabulary


def _row(structure_id: str, smiles: str) -> dict[str, str]:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return {
        "r0_structure_id": structure_id,
        "canonical_isomeric_smiles": smiles,
        "heavy_atoms": str(molecule.GetNumHeavyAtoms()),
        "elements": "|".join(sorted({atom.GetSymbol() for atom in molecule.GetAtoms()})),
    }


def _write_csv(path: Path, rows: list[dict[str, str]], *, compressed: bool = False) -> None:
    opener = gzip.open if compressed else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_traversal_audit_compares_exact_encodings_and_ugi_origins(tmp_path: Path) -> None:
    rows = [
        _row("train", "NCCO"),
        _row("cal", "CNCCOC(=O)C1CCCCC1"),
        _row("heldout", "CN(C)CCOC(=O)CCCC"),
    ]
    r0_path = tmp_path / "r0.csv.gz"
    _write_csv(r0_path, rows, compressed=True)
    assignments_path = tmp_path / "assignments.csv"
    _write_csv(
        assignments_path,
        [
            {"r0_structure_id": "train", "source_study_fold": "R0_train"},
            {"r0_structure_id": "cal", "source_study_fold": "R0_cal"},
            {"r0_structure_id": "heldout", "source_study_fold": "R0_heldout"},
        ],
    )

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
                }
            }
        )
    )

    ugi_smiles = Chem.MolToSmiles(
        Chem.MolFromSmiles("NCCO"),
        canonical=True,
        isomericSmiles=False,
    )
    ugi_molecule = Chem.MolFromSmiles(ugi_smiles)
    assert ugi_molecule is not None
    ugi_products_path = tmp_path / "ugi_products.csv.gz"
    _write_csv(
        ugi_products_path,
        [{"product_id": "ugi", "product_smiles": ugi_smiles}],
        compressed=True,
    )
    ugi_atoms_path = tmp_path / "ugi_atoms.csv.gz"
    _write_csv(
        ugi_atoms_path,
        [
            {
                "product_id": "ugi",
                "product_atom_index": str(index),
                "origin_role": "amine_head" if index < 2 else "aldehyde_tail",
            }
            for index in range(ugi_molecule.GetNumAtoms())
        ],
        compressed=True,
    )

    result = audit_traversal_representations(
        r0_path,
        assignments_path,
        vocabulary_path,
        config_path,
        ugi_products_path,
        ugi_atoms_path,
        seed=23,
        permutations_per_record=2,
    )

    assert result["status"] == "pass"
    assert result["failure_examples"] == []
    assert result["policy"]["selection_metrics_fold"] == "R0_cal"
    assert result["policy"]["r0_heldout_use"] == (
        "exact_invariance_checks_only_not_architecture_selection"
    )
    assert set(result["traversals"]) == {
        "breadth_first",
        "breadth_first_tree_preorder",
        "depth_first_preorder",
    }
    assert set(result["metric_winners"].values()).issubset(
        {"breadth_first", "breadth_first_tree_preorder", "depth_first_preorder", "tie"}
    )
    for traversal in result["traversals"].values():
        assert traversal["status"] == "pass"
        assert traversal["counts"]["records"] == 3
        assert traversal["counts"]["exact_permutation_encodings"] == 6
        assert traversal["fixed_bigram_model"]["evaluation_fold"] == "R0_cal"
        assert traversal["fixed_bigram_model"]["evaluation_cross_entropy_nats_per_token"] > 0
