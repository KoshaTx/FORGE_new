from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.phase1.synthesis_guidance.guidance.ugi_postselection_branching import (
    UgiPostselectionBranchingError,
    build_postselection_branching_audit,
    carbon_branch_metrics,
)
from forge.corpus.r1_prime_audit import sha256_file

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_postselection_branching_audit_v1.json"


def _paths() -> tuple[Path, ...]:
    config = json.loads(CONFIG.read_text())
    inputs = config["inputs"]
    return (
        CONFIG,
        REPO / inputs["production_manifest"]["path"],
        REPO / inputs["selected_sample"]["path"],
        REPO / inputs["selection_reference_assignments"]["path"],
    )


@pytest.mark.parametrize(
    ("smiles", "branches", "edges", "run"),
    [
        ("CCCCCC", 0, 0, 0),
        ("CC(C)CCC", 1, 0, 1),
        ("CC(C)C(C)CC", 2, 1, 2),
        ("CC(C)C(C)(C)C", 2, 1, 2),
    ],
)
def test_carbon_branch_metrics(smiles: str, branches: int, edges: int, run: int) -> None:
    metrics = carbon_branch_metrics(smiles)
    assert metrics["carbon_branch_atoms"] == branches
    assert metrics["adjacent_carbon_branch_edges"] == edges
    assert metrics["maximum_adjacent_branch_run"] == run


def test_invalid_component_fails_closed() -> None:
    with pytest.raises(UgiPostselectionBranchingError, match="valid connected graph"):
        carbon_branch_metrics("C.C")


def test_frozen_branching_audit_is_deterministic_and_detects_tail_spacing_shift() -> None:
    first = build_postselection_branching_audit(*_paths())
    second = build_postselection_branching_audit(*_paths())
    assert first == second
    summary = first["summary"]
    assert summary["generated_products_with_components"] == 1007
    assert summary["selection_reference_products"] == 82264
    assert summary["products_with_adjacent_tail_carbon_branch_atoms"] == 87
    assert (
        summary["generated"]["oxoester_aldehyde_body_tail"][
            "components_with_adjacent_carbon_branch_atoms"
        ]
        == 27
    )
    assert (
        summary["generated"]["isocyanide_tail"]["components_with_adjacent_carbon_branch_atoms"]
        == 63
    )
    assert (
        summary["selection_reference"]["oxoester_aldehyde_body_tail"][
            "components_with_adjacent_carbon_branch_atoms"
        ]
        == 0
    )
    assert (
        summary["selection_reference"]["isocyanide_tail"][
            "components_with_adjacent_carbon_branch_atoms"
        ]
        == 0
    )
    assert summary["challenger_required_before_candidate_lock"] is True


def test_frozen_config_hashes_match_inputs() -> None:
    config = json.loads(CONFIG.read_text())
    for specification in config["inputs"].values():
        assert sha256_file(REPO / specification["path"]) == specification["sha256"]
