from __future__ import annotations

from dataclasses import dataclass

import pytest

from forge.product.ugi_program_matched_tail_chemistry import (
    compare_program_matched_tail_chemistry,
)


@dataclass(frozen=True)
class Program:
    node_counts: tuple[int, int, int]
    junction_budgets: tuple[int, int, int]
    cycle_ranks: tuple[int, int, int]
    attachment_counts: tuple[int, int, int]


@dataclass(frozen=True)
class Record:
    program: Program


def _program(aldehyde_nodes: int, aldehyde_junctions: int = 0) -> Program:
    return Program(
        node_counts=(3, aldehyde_nodes, 4),
        junction_budgets=(0, aldehyde_junctions, 0),
        cycle_ranks=(0, 0, 0),
        attachment_counts=(2, 1, 1),
    )


def _generated_row(aldehyde_nodes: int, aldehyde: str, aldehyde_junctions: int = 0) -> dict:
    program = _program(aldehyde_nodes, aldehyde_junctions)
    return {
        "valid": True,
        "component_reconstruction_valid": True,
        "l1_forward_verification": {"exact_product_reconstructed": True},
        "program": {
            "node_counts": list(program.node_counts),
            "junction_budgets": list(program.junction_budgets),
            "cycle_ranks": list(program.cycle_ranks),
            "attachment_counts": list(program.attachment_counts),
        },
        "component_smiles_by_role": {
            "oxoester_aldehyde_body_tail": aldehyde,
            "isocyanide_tail": "[C-]#[N+]CCCC",
        },
    }


def test_program_matching_removes_schedule_confounding() -> None:
    assignments = (
        {
            "oxoester_aldehyde_body_tail_smiles": "CCCC=O",
            "isocyanide_tail_smiles": "[C-]#[N+]CCCC",
        },
        {
            "oxoester_aldehyde_body_tail_smiles": "CCCCCC=O",
            "isocyanide_tail_smiles": "[C-]#[N+]CCCC",
        },
    )
    records = (Record(_program(3)), Record(_program(5)))
    sample = {"samples": [_generated_row(3, "CCCC=O")]}

    result = compare_program_matched_tail_chemistry(
        sample=sample,
        assignments=assignments,
        joint_records=records,
        training_weights=(0.01, 0.99),
    )

    aldehyde = result["by_role"]["oxoester_aldehyde_body_tail"]
    assert aldehyde["exact_match_fraction"] == 1.0
    assert aldehyde["exact_joint_cell_sensitivity"][
        "mean_absolute_error_all_features"
    ] == pytest.approx(0.0)
    assert aldehyde["marginally_balanced_coordinate_supported_cohort"][
        "mean_absolute_error_all_features"
    ] == pytest.approx(0.0)


def test_unmatched_programs_are_reported_not_silently_dropped() -> None:
    assignments = (
        {
            "oxoester_aldehyde_body_tail_smiles": "CCCC=O",
            "isocyanide_tail_smiles": "[C-]#[N+]CCCC",
        },
        {
            "oxoester_aldehyde_body_tail_smiles": "CCCCCC=O",
            "isocyanide_tail_smiles": "[C-]#[N+]CCCC",
        },
        {
            "oxoester_aldehyde_body_tail_smiles": "CCCCCCCC=O",
            "isocyanide_tail_smiles": "[C-]#[N+]CCCC",
        },
    )
    records = (
        Record(_program(3, 0)),
        Record(_program(6, 1)),
        Record(_program(8, 0)),
    )
    sample = {
        "samples": [
            _generated_row(3, "CCCC=O"),
            _generated_row(6, "CCCCCCC=O"),
            _generated_row(8, "CCCCCCCCC=O", aldehyde_junctions=1),
        ]
    }

    result = compare_program_matched_tail_chemistry(
        sample=sample,
        assignments=assignments,
        joint_records=records,
        training_weights=(1.0, 1.0, 1.0),
    )

    aldehyde = result["by_role"]["oxoester_aldehyde_body_tail"]
    assert aldehyde["exact_match_fraction"] == pytest.approx(1 / 3)
    assert aldehyde["unmatched_generated_program_cells"] == [
        {
            "program": {
                "role_node_count": 6,
                "role_junction_budget": 0,
                "role_cycle_rank": 0,
                "role_attachment_count": 1,
            },
            "generated_rows": 1,
        },
        {
            "program": {
                "role_node_count": 8,
                "role_junction_budget": 1,
                "role_cycle_rank": 0,
                "role_attachment_count": 1,
            },
            "generated_rows": 1,
        },
    ]
    assert (
        aldehyde["marginally_balanced_coordinate_supported_cohort"]["raking_diagnostics"][
            "maximum_absolute_marginal_error"
        ]
        < 1e-8
    )
