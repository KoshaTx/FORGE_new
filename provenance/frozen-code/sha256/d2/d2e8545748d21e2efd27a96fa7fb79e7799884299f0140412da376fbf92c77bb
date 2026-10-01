from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path

from rdkit import Chem

from forge.corpus.ugi_component_expansion import (
    FOLDS,
    ROLES,
    _family_fold_map,
    _similarity_families,
    build_ugi_component_expansion,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_component_expansion.json"


def test_similarity_families_keep_close_homologs_together() -> None:
    fingerprint = {
        "radius": 2,
        "bits": 512,
        "include_chirality": False,
        "similarity_threshold": 0.70,
    }
    smiles = ["CCCCCC=O", "CCCCCCC=O", "NCCN", "NCCCN"]
    families = _similarity_families(smiles, fingerprint, "test_role")
    assert families["CCCCCC=O"] == families["CCCCCCC=O"]


def test_family_fold_map_never_splits_a_family() -> None:
    members = {
        "a": ["a1", "a2", "a3", "a4"],
        "b": ["b1", "b2"],
        "c": ["c1"],
        "d": ["d1"],
    }
    assignments = _family_fold_map(
        members,
        {"train": 0.7, "calibration": 0.15, "heldout": 0.15},
        7,
        "role",
    )
    assert set(assignments) == set(members)
    assert set(assignments.values()) == set(FOLDS)


def test_repository_component_expansion_contract() -> None:
    result, rows = build_ugi_component_expansion(CONFIG, REPO)
    assert result["status"] == "complete_l1_structural_expansion_census"
    assert result["claims_boundary"]["l1_admission_is_route_closure"] is False
    admitted = [row for row in rows if row["l1_structural_admission"] == "true"]
    assert admitted
    assert all(row["reference_forward_compatible"] == "true" for row in admitted)
    assert all(row["family_fold"] in FOLDS for row in admitted)
    assert set(row["role"] for row in admitted) == set(ROLES)
    assert all(
        all(
            atom.GetNumRadicalElectrons() == 0
            for atom in Chem.MolFromSmiles(row["canonical_smiles"]).GetAtoms()
        )
        for row in admitted
    )
    families: dict[str, set[str]] = defaultdict(set)
    for row in admitted:
        families[str(row["family_id"])].add(str(row["family_fold"]))
    assert all(len(folds) == 1 for folds in families.values())

    with gzip.open(
        REPO / "data/splits/phase1/ugi_l1_assignments.csv.gz", "rt", newline=""
    ) as handle:
        current_rows = list(csv.DictReader(handle))
    expected = {role: {row[f"{role}_smiles"] for row in current_rows} for role in ROLES}
    retained = {
        role: {
            row["canonical_smiles"]
            for row in admitted
            if row["role"] == role and row["is_current_catalog"] == "true"
        }
        for role in ROLES
    }
    assert {role: len(values) for role, values in retained.items()} == {
        role: len(values) for role, values in expected.items()
    }
