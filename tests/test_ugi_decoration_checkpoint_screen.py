from __future__ import annotations

from forge.product.ugi_decoration_checkpoint_screen import (
    pareto_frontier,
    select_confirmed_training_duration,
)


def _arm(validity: float, error: float, branch: float, *, passes: bool = True) -> dict:
    return {
        "passes_all_hard_gates": passes,
        "pareto_axes": {
            "maximize_valid_fraction": validity,
            "minimize_program_balanced_local_chemistry_mae": error,
            "minimize_program_matched_branch_geometry_jsd_bits": branch,
        },
    }


def test_pareto_frontier_retains_tradeoffs_and_excludes_dominated_arms() -> None:
    arms = {
        "high_validity": _arm(0.99, 0.05, 0.04),
        "high_chemistry": _arm(0.96, 0.02, 0.03),
        "high_branch": _arm(0.95, 0.04, 0.01),
        "dominated": _arm(0.95, 0.06, 0.05),
        "failed_gate": _arm(1.0, 0.0, 0.0, passes=False),
    }

    assert pareto_frontier(arms) == ["high_branch", "high_chemistry", "high_validity"]


def test_confirmed_duration_uses_frozen_lexicographic_noninferiority() -> None:
    arms = {
        "1500": {
            "passes_all_hard_gates": True,
            "mean_axes": {
                "program_balanced_local_chemistry_mae": 0.040,
                "program_matched_branch_geometry_jsd_bits": 0.120,
                "valid_fraction": 0.981,
            },
        },
        "3000": {
            "passes_all_hard_gates": True,
            "mean_axes": {
                "program_balanced_local_chemistry_mae": 0.034,
                "program_matched_branch_geometry_jsd_bits": 0.099,
                "valid_fraction": 0.958,
            },
        },
        "4000": {
            "passes_all_hard_gates": True,
            "mean_axes": {
                "program_balanced_local_chemistry_mae": 0.038,
                "program_matched_branch_geometry_jsd_bits": 0.094,
                "valid_fraction": 0.965,
            },
        },
    }

    decision = select_confirmed_training_duration(
        arms,
        chemistry_noninferiority_margin=0.01,
        branch_jsd_noninferiority_margin_bits=0.01,
        validity_tie_margin=0.01,
    )

    assert decision["chemistry_noninferior_steps"] == [1500, 3000, 4000]
    assert decision["branch_noninferior_steps"] == [3000, 4000]
    assert decision["validity_tied_steps"] == [3000, 4000]
    assert decision["selected_checkpoint_step"] == 3000
