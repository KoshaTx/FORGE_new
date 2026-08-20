from __future__ import annotations

import pytest

from experiments.archive.phase1.synthesis_audits.aizynthfinder_diagnostic import (
    AiZynthFinderDiagnosticError,
    select_full_search_targets,
    summarize_worker_records,
    targets_from_molecule_audit,
)


def _audit_result() -> dict:
    return {
        "schema_version": "phase1_ugi_high_potency_molecule_route_audit_result.v1",
        "summary": {
            "unresolved_component_diagnostics": {
                "by_role": {
                    "oxoester_aldehyde_body_tail": {
                        "nearest_reference_records": [
                            {
                                "unresolved_component": "CCCC(=O)OCC=O",
                                "nearest_route_ready_component": "CCCCC(=O)OCC=O",
                                "morgan_tanimoto": 0.91,
                            },
                            {
                                "unresolved_component": "CCCCCC(=O)OCCC=O",
                                "nearest_route_ready_component": "CCCCCCC(=O)OCCC=O",
                                "morgan_tanimoto": 0.88,
                            },
                        ]
                    },
                    "isocyanide_tail": {
                        "nearest_reference_records": [
                            {
                                "unresolved_component": "[C-]#[N+]CCCCCC",
                                "nearest_route_ready_component": "[C-]#[N+]CCCCC",
                                "morgan_tanimoto": 1.0,
                            }
                        ]
                    },
                }
            }
        },
    }


def test_builds_unique_unresolved_and_matched_targets() -> None:
    targets = targets_from_molecule_audit(_audit_result())
    assert len(targets) == 6
    assert len({target.target_id for target in targets}) == 6
    assert {target.cohort for target in targets} == {
        "unresolved",
        "matched_route_ready_control",
    }
    assert all(target.paired_smiles for target in targets)


def test_full_search_selection_includes_matched_controls() -> None:
    targets = targets_from_molecule_audit(_audit_result())
    selected = select_full_search_targets(targets, maximum_unresolved_per_role=1)
    chosen = [target for target in targets if target.target_id in selected]
    assert len(chosen) == 4
    assert sum(target.cohort == "unresolved" for target in chosen) == 2
    assert sum(target.cohort == "matched_route_ready_control" for target in chosen) == 2


def test_worker_summary_keeps_planner_solution_as_diagnostic_count() -> None:
    rows = [
        {
            "target": {
                "target_id": "a" * 64,
                "cohort": "unresolved",
                "role": "isocyanide_tail",
            },
            "single_step_proposals": [{"rank": 1}],
            "full_search": {"is_solved_to_public_stock": True},
        },
        {
            "target": {
                "target_id": "b" * 64,
                "cohort": "matched_route_ready_control",
                "role": "isocyanide_tail",
            },
            "single_step_proposals": [],
            "full_search": None,
        },
    ]
    summary = summarize_worker_records(rows)
    assert summary["total"] == {
        "targets": 2,
        "targets_with_single_step_proposals": 1,
        "single_step_proposals": 1,
        "planner_attempted": 1,
        "planner_solved_to_public_stock": 1,
    }


def test_rejects_unsupported_molecule_audit_schema() -> None:
    with pytest.raises(AiZynthFinderDiagnosticError, match="schema"):
        targets_from_molecule_audit({"schema_version": "wrong"})
