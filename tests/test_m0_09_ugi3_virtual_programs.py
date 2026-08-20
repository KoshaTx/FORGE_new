from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from forge.corpus.r1_prime_audit import sha256_file
from forge.synthesis.terminals.ugi3_virtual_programs import (
    ESTER_PROGRAM,
    ISOCYANIDE_PROGRAM,
    Ugi3VirtualProgramError,
    _aldehyde_program,
    _isocyanide_program,
    _scope_features,
    build_ugi3_virtual_component_programs,
    write_ugi3_virtual_component_programs,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/m0_09_agile_virtual_ugi3_component_programs.json"
COMPONENT_LEDGER = REPO / "results/m0_09/agile_virtual_ugi3_component_ledger.csv.gz"
CAPABILITY = REPO / "results/m0_09/agile_virtual_ugi3_capability.json"
SOURCE_ROUTES = REPO / "results/m0_09/agile_component_routes.json"
RESULT = REPO / "results/m0_09/agile_virtual_ugi3_component_programs.json"


def _production_paths() -> tuple[Path, ...]:
    return CONFIG, COMPONENT_LEDGER, CAPABILITY, SOURCE_ROUTES


def test_ester_aldehyde_projection_builds_upstream_subcomponents() -> None:
    target = "CCCCCCCC(=O)OCCC=O"
    family, steps = _aldehyde_program(target)

    assert family == ESTER_PROGRAM
    assert steps == [
        {
            "step_index": 1,
            "transformation": "esterification",
            "reactants": ["CCCCCCCC(=O)O", "OCCCO"],
            "product": "CCCCCCCC(=O)OCCCO",
        },
        {
            "step_index": 2,
            "transformation": "alcohol_to_aldehyde_oxidation",
            "reactants": ["CCCCCCCC(=O)OCCCO"],
            "product": target,
        },
    ]


def test_isocyanide_projection_builds_amine_and_formamide() -> None:
    target = "[C-]#[N+]CCCCCCCCCCCCC"
    family, steps = _isocyanide_program(target)

    assert family == ISOCYANIDE_PROGRAM
    assert steps == [
        {
            "step_index": 1,
            "transformation": "amine_formylation",
            "reactants": ["CCCCCCCCCCCCCN"],
            "product": "CCCCCCCCCCCCCNC=O",
        },
        {
            "step_index": 2,
            "transformation": "formamide_dehydration_to_isocyanide",
            "reactants": ["CCCCCCCCCCCCCNC=O"],
            "product": target,
        },
    ]


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (
            "CCCCCC(C)CCC(=O)OCCC=O",
            {
                "carbon_branching": "branched",
                "carbon_unsaturation": "saturated",
                "ester_count": 1,
            },
        ),
        (
            "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)OCCC=O",
            {
                "carbon_branching": "unbranched",
                "carbon_unsaturation": "polyene",
                "ester_count": 1,
            },
        ),
        (
            "C#CCCCCCCCCC(=O)OCCC=O",
            {
                "carbon_branching": "unbranched",
                "carbon_unsaturation": "alkyne",
                "ester_count": 1,
            },
        ),
    ],
)
def test_scope_features_distinguish_tail_substrate_classes(
    smiles: str,
    expected: dict[str, str | int],
) -> None:
    observed = _scope_features(smiles, role="test_tail")

    assert {field: observed[field] for field in expected} == expected


def test_rejects_component_ledger_hash_mismatch(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["inputs"]["component_ledger"]["expected_sha256"] = "0" * 64
    bad_config = tmp_path / "config.json"
    bad_config.write_text(json.dumps(config))

    with pytest.raises(Ugi3VirtualProgramError, match="hash mismatch"):
        build_ugi3_virtual_component_programs(
            bad_config,
            *_production_paths()[1:],
        )


def test_writer_is_deterministic_for_committed_payloads(
    tmp_path: Path,
) -> None:
    result = json.loads(RESULT.read_text())
    details = result["artifacts"]["agile_virtual_ugi3_component_program_ledger.csv.gz"]
    ledger = (REPO / details["path"]).read_bytes()

    write_ugi3_virtual_component_programs(result, ledger, tmp_path)
    first = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    write_ugi3_virtual_component_programs(result, ledger, tmp_path)
    second = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    assert first == second
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_program_census_preserves_evidence_boundaries() -> None:
    result = json.loads(RESULT.read_text())
    summary = result["summary"]

    assert summary["source_components"] == 93
    assert summary["components_with_structural_program"] == 71
    assert summary["components_with_exact_source_program"] == 24
    assert summary["components_with_family_projected_program"] == 47
    assert summary["exact_source_programs_reproduced_structurally"] == 24
    assert summary["unique_proposed_intermediates"] == 65
    assert summary["unique_proposed_leaf_candidates"] == 33
    aldehyde_scope = summary["substrate_scope"]["oxoester_aldehyde_body_tail"]
    assert aldehyde_scope["carbon_unsaturation"] == {
        "alkyne": {
            "components": 4,
            "exact_source_programs": 1,
            "family_projected_programs": 3,
            "accepted_procurement_terminals": 0,
            "unresolved_components": 0,
        },
        "monoene": {
            "components": 13,
            "exact_source_programs": 6,
            "family_projected_programs": 7,
            "accepted_procurement_terminals": 0,
            "unresolved_components": 0,
        },
        "polyene": {
            "components": 4,
            "exact_source_programs": 1,
            "family_projected_programs": 3,
            "accepted_procurement_terminals": 0,
            "unresolved_components": 0,
        },
        "saturated": {
            "components": 41,
            "exact_source_programs": 9,
            "family_projected_programs": 32,
            "accepted_procurement_terminals": 0,
            "unresolved_components": 0,
        },
    }
    assert aldehyde_scope["carbon_branching"]["branched"]["exact_source_programs"] == 2
    assert aldehyde_scope["ester_bearing"]["yes"]["components"] == 56
    assert result["claims_boundary"]["family_projection_is_exact_substrate_evidence"] is False
    assert result["claims_boundary"]["proposed_leaf_is_procurement_closed"] is False

    details = result["artifacts"]["agile_virtual_ugi3_component_program_ledger.csv.gz"]
    ledger_path = REPO / details["path"]
    assert ledger_path.stat().st_size == details["bytes"]
    assert sha256_file(ledger_path) == details["sha256"]
    with gzip.open(ledger_path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 93
    assert all(json.loads(row["scope_features_json"]) for row in rows)
    assert (
        sum(row["structural_program_status"] == "exact_source_structure_reproduced" for row in rows)
        == 24
    )
    assert sum(row["structural_program_status"] == "family_projection_only" for row in rows) == 47
    assert all(
        row["route_closure"] != "computationally_complete"
        for row in rows
        if row["role"] != "amine_head"
    )
