from __future__ import annotations

from experiments.phase1.hela_potency.applicability import (
    _flatten_metadata,
    _prepare_branch_rows,
)


def _row(draw: int, *, admitted: bool = True):
    return {
        "draw_index": draw,
        "scheduled_branch_class": "aldehyde_origin_branched",
        "realized_carbon_branch_class": ("aldehyde_origin_branched" if admitted else None),
        "exact_refit_corpus_product": False,
        "native_terminal": {"valid": True},
        "tail_component_descriptors": (
            {
                "oxoester_aldehyde_body_tail": {
                    "heavy_atoms": 18,
                    "carbon_atoms": 16,
                    "carbon_branch_points": 1,
                    "adjacent_carbon_branch_edges": 0,
                    "carbon_carbon_double_bonds": 0,
                    "ester_carbonyls": 1,
                },
                "isocyanide_tail": {
                    "heavy_atoms": 12,
                    "carbon_atoms": 12,
                    "carbon_branch_points": 0,
                    "adjacent_carbon_branch_edges": 0,
                    "carbon_carbon_double_bonds": 0,
                    "ester_carbonyls": 0,
                },
            }
            if admitted
            else None
        ),
        "terminal_chemical_admission": {
            "admitted": admitted,
            "exact_l1": True,
            "raw_molecule_valid": True,
            "reason": "admitted" if admitted else "handle_policy_failure:amine_head",
        },
    }


def test_prepare_branch_rows_masks_only_inadmissible_terminals() -> None:
    prepared, metadata = _prepare_branch_rows([_row(0), _row(1, admitted=False)], expected_draws=2)

    assert prepared[0]["arm_id"] == "branch_exploration"
    assert prepared[0]["native_terminal"] == {"valid": True}
    assert prepared[1]["native_terminal"] is None
    assert metadata[0]["realized_carbon_branch_class"] == "aldehyde_origin_branched"


def test_flatten_metadata_preserves_realized_branch_chemistry() -> None:
    _, metadata = _prepare_branch_rows([_row(0)], expected_draws=1)
    classified = [{"draw_index": 0, "reason": "interpolative"}]

    _flatten_metadata(classified, metadata)

    assert classified[0]["aldehyde_carbon_atoms"] == 16
    assert classified[0]["aldehyde_carbon_branch_points"] == 1
    assert classified[0]["aldehyde_ester_carbonyls"] == 1
