from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from forge.corpus.ugi_held_component_gate import (
    analyze_generated_components,
    exact_forward_reconstructs_ugi_product,
    held_role_class,
    load_ugi_reaction_contract,
)


def test_held_role_class_is_mutually_exclusive() -> None:
    assignment = {
        "amine_head_family_fold": "heldout",
        "oxoester_aldehyde_body_tail_family_fold": "train",
        "isocyanide_tail_family_fold": "heldout",
    }
    assert held_role_class(assignment) == "amine_head+isocyanide_tail"


def test_generated_component_analysis_uses_actual_component_graphs(tmp_path: Path) -> None:
    registry = tmp_path / "registry.csv.gz"
    fields = (
        "role",
        "canonical_smiles",
        "family_fold",
        "l1_structural_admission",
    )
    rows = (
        ("amine_head", "CN", "train", "true"),
        ("oxoester_aldehyde_body_tail", "CC=O", "train", "true"),
        ("isocyanide_tail", "[C-]#[N+]C", "train", "true"),
    )
    with gzip.open(registry, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    result = tmp_path / "samples.json"
    result.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "valid": True,
                        "smiles": "CNC(=O)C(C)NC",
                        "component_reconstruction_valid": True,
                        "held_role_class": "amine_head",
                        "component_smiles_by_role": {
                            "amine_head": "CN",
                            "oxoester_aldehyde_body_tail": "CC=O",
                            "isocyanide_tail": "[C-]#[N+]C",
                        },
                    },
                    {
                        "valid": True,
                        "smiles": "CCNC(C)C(=O)NC",
                        "component_reconstruction_valid": True,
                        "held_role_class": "amine_head",
                        "component_smiles_by_role": {
                            "amine_head": "CCN",
                            "oxoester_aldehyde_body_tail": "CC=O",
                            "isocyanide_tail": "[C-]#[N+]C",
                        },
                    },
                ],
                "statistics": {},
                "reference_comparison": {},
            }
        )
    )
    qualified_reactions = Path(__file__).parents[1] / "data/vendor/qualified_reactions_v1.json"
    observed = analyze_generated_components(result, registry, qualified_reactions)
    assert observed["product_component_novelty_counts"] == {
        "genuinely_generated_component": 1,
        "train_catalog_only": 1,
    }
    assert observed["exact_train_component_triples"] == 1
    assert observed["all_three_handles_pass_fraction"] == 1.0
    assert observed["exact_forward_product_reconstruction_fraction"] == 1.0
    assert observed["by_role"]["amine_head"]["membership_counts"] == {
        "outside_admitted_catalog": 1,
        "train": 1,
    }


def test_public_exact_forward_verifier_rejects_non_amine_precursor() -> None:
    reaction = load_ugi_reaction_contract(
        Path(__file__).parents[1] / "data/vendor/qualified_reactions_v1.json"
    )

    valid, _, valid_outcomes = exact_forward_reconstructs_ugi_product(
        reaction,
        {
            "amine_head": "CN",
            "oxoester_aldehyde_body_tail": "CC=O",
            "isocyanide_tail": "[C-]#[N+]C",
        },
        "CNC(=O)C(C)NC",
    )
    invalid, _, invalid_outcomes = exact_forward_reconstructs_ugi_product(
        reaction,
        {
            "amine_head": "C=N",
            "oxoester_aldehyde_body_tail": "CCCCCCCCCCCCCCCCCCCCCCC=O",
            "isocyanide_tail": "[C-]#[N+]CCCCCCCCCCCCCC",
        },
        "C=NC(CCCCCCCCCCCCCCCCCCCCCC)C(=O)NCCCCCCCCCCCCCC",
    )

    assert valid
    assert valid_outcomes > 0
    assert not invalid
    assert invalid_outcomes == 0
