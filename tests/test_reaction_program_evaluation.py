from __future__ import annotations

import pytest

from forge.model.reaction_program_evaluation import (
    ReactionProgramEvaluationError,
    evaluate_reaction_program_samples,
)

TRAINING_PRODUCTS = {"aza": {"CCN"}, "reductive": {"CN"}}
TRAINING_COMPONENTS = {
    "aza": {"terminal_head": {"N"}, "repeat_component": {"CC"}},
    "reductive": {"terminal_head": {"CN"}, "repeat_component": {"C=O"}},
}


def _exact_row(program_id: str = "aza") -> dict[str, object]:
    return {
        "program_id": program_id,
        "valid": True,
        "canonical_smiles": "CCCN",
        "exact_l1_program": True,
        "exact_l1_trace_count": 1,
        "forward_verified_trace_count": 1,
        "exact_l1_traces": [
            {
                "terminal_head_smiles": "N",
                "repeated_component_smiles": ["CCC"],
            }
        ],
    }


def test_evaluation_reports_nonselecting_family_and_role_specific_metrics() -> None:
    rows = [
        _exact_row(),
        {
            "program_id": "aza",
            "valid": False,
            "canonical_smiles": None,
            "exact_l1_program": False,
            "exact_l1_trace_count": 0,
            "forward_verified_trace_count": 0,
            "exact_l1_traces": [],
        },
        {
            "program_id": "reductive",
            "valid": True,
            "canonical_smiles": "CO",
            "exact_l1_program": False,
            "exact_l1_trace_count": 0,
            "forward_verified_trace_count": 0,
            "exact_l1_traces": [],
        },
    ]

    result = evaluate_reaction_program_samples(
        rows,
        training_products=TRAINING_PRODUCTS,
        training_components=TRAINING_COMPONENTS,
    )

    assert result["coverage_and_precision_reported"] is True
    assert result["reductive_amination_substructure_rate_reported"] is False
    assert result["overall"]["valid_fraction"] == pytest.approx(2 / 3)
    assert result["overall"]["retro_decomposition_coverage_among_valid"] == 0.5
    assert result["overall"]["retro_transform_precision"] == 1.0
    assert result["per_program"]["aza"]["novel_terminal_head_fraction"] == 0.0
    assert result["per_program"]["aza"]["novel_repeat_component_fraction"] == 1.0
    assert result["per_program"]["aza"]["exact_l1_yield_per_attempt"] == 0.5
    assert (
        result["per_program"]["aza"]["unique_open_ended_exact_l1_products_per_1000_attempts"]
        == 500.0
    )
    assert result["per_program"]["reductive"]["retro_transform_precision"] is None


def test_evaluation_measures_connectedness_instead_of_copying_validity() -> None:
    row = {
        "program_id": "aza",
        "valid": True,
        "canonical_smiles": "C.C",
        "exact_l1_program": False,
        "exact_l1_trace_count": 0,
        "forward_verified_trace_count": 0,
        "exact_l1_traces": [],
    }

    result = evaluate_reaction_program_samples(
        [row],
        training_products=TRAINING_PRODUCTS,
        training_components=TRAINING_COMPONENTS,
    )

    assert result["overall"]["valid_fraction"] == 1.0
    assert result["overall"]["connected_fraction"] == 0.0


def test_evaluation_rejects_a_decomposition_that_does_not_forward_replay() -> None:
    row = _exact_row()
    row["forward_verified_trace_count"] = 0

    with pytest.raises(ReactionProgramEvaluationError, match="failed exact forward replay"):
        evaluate_reaction_program_samples(
            [row],
            training_products=TRAINING_PRODUCTS,
            training_components=TRAINING_COMPONENTS,
        )
