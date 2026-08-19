from __future__ import annotations

import numpy as np
from rdkit import Chem

from forge.bio.ugi_semantic_annotations import ROLE_NAMES
from forge.product.ugi_candidate_eligibility import (
    compile_smarts,
    declared_support_violations,
    heteroatom_extreme_flags,
    heteroatom_profile,
    heteroatom_thresholds,
    motif_hits,
)

SMARTS = {
    "alkyne": "[#6]#[#6]",
    "allene_or_cumulene": "[!#1]=[#6]=[!#1]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "peroxide": "[#8X2]-[#8X2]",
    "nitrogen_nitrogen": "[#7]~[#7]",
}


def test_frozen_motif_queries_detect_expected_chemistry() -> None:
    queries = compile_smarts(SMARTS)
    examples = {
        "alkyne": "CC#CC",
        "allene_or_cumulene": "C=C=C",
        "aldehyde": "CC=O",
        "peroxide": "COOC",
        "nitrogen_nitrogen": "CNN",
    }
    for name, smiles in examples.items():
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None
        assert motif_hits(molecule, queries)[name]
    sulfone = Chem.MolFromSmiles("CS(=O)(=O)C")
    assert sulfone is not None
    assert not motif_hits(sulfone, queries)["allene_or_cumulene"]


def test_heteroatom_profile_and_extreme_flags_are_strict() -> None:
    molecule = Chem.MolFromSmiles("CCO")
    assert molecule is not None
    count, fraction = heteroatom_profile(molecule)
    assert count == 1
    assert fraction == 1 / 3
    thresholds = heteroatom_thresholds(
        np.asarray([1, 2, 3, 4, 5], dtype=float),
        np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float),
        lower_quantile=0.01,
        upper_quantile=0.99,
    )
    assert heteroatom_extreme_flags(1, 0.1, thresholds)["any_extreme_heteroatom_pattern"]
    assert not heteroatom_extreme_flags(3, 0.3, thresholds)["any_extreme_heteroatom_pattern"]


def test_declared_support_violation_reports_atom_state_and_size() -> None:
    molecule = Chem.MolFromSmiles("CC")
    assert molecule is not None
    sample = {
        "program": {
            "node_counts": [1, 1, 1],
            "junction_budgets": [0, 0, 0],
            "cycle_ranks": [0, 0, 0],
            "attachment_counts": [1, 1, 1],
        },
        "offspring_by_role": {role: [0] for role in ROLE_NAMES},
    }
    config = {
        "maximum_total_atoms": 1,
        "maximum_component_atoms": 4,
        "maximum_junction_budget": 1,
        "maximum_cycle_rank": 1,
        "maximum_attachment_count": 1,
        "maximum_children": 2,
    }
    violations = declared_support_violations(sample, molecule, config, set())
    assert violations == ["atom_state_vocabulary", "maximum_total_atoms"]
