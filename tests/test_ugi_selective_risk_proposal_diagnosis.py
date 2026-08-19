from __future__ import annotations

import json

import numpy as np
import pytest

from forge.potency.ugi_selective_risk_proposal_diagnosis import (
    VIEWS,
    _finite_quantile,
    _spearman,
    gate_diagnostics,
)


def _broad_row(
    roles: tuple[str, ...],
    *,
    bins: dict[str, str] | None = None,
    measured: bool = False,
) -> dict[str, object]:
    selected = bins or {view: "interpolative" for view in VIEWS}
    return {
        "exact_identity_provenance": (
            "exact_measured_combination" if measured else "exact_new_component_identity"
        ),
        "exact_unseen_roles_json": json.dumps(list(roles), separators=(",", ":")),
        "branch_class": "linear_tail_origins",
        **{f"{view}_distribution_bin": selected[view] for view in VIEWS},
    }


def test_gate_diagnostics_separates_chemistry_from_role_policy() -> None:
    rows = [
        _broad_row((), measured=True),
        _broad_row(("amine",)),
        _broad_row(("aldehyde",)),
        _broad_row(("amine", "aldehyde", "isocyanide")),
        _broad_row(
            ("aldehyde", "isocyanide"),
            bins={
                "product": "boundary",
                "amine": "interpolative",
                "aldehyde": "interpolative",
                "isocyanide": "interpolative",
            },
        ),
    ]

    result = gate_diagnostics(rows, source="broad_census")

    assert result["valid_exact_l1"]["count"] == 5
    assert result["exact_measured_neutral"]["count"] == 1
    assert result["novel_all_four_views_interpolative"]["count"] == 3
    assert result["active_intersection"]["among_chemically_supported_novel"]["count"] == 1
    assert result["chemically_supported_but_role_policy_abstains"]["count"] == 2
    assert result["leave_one_view_out"]["product"]["increment_over_all_four"] == 1
    assert result["leave_one_view_out"]["product"]["active_increment_over_all_four"] == 1


def test_lambda_zero_invalid_terminal_is_not_counted_as_chemically_unsupported() -> None:
    valid = {
        "terminal_present": "True",
        "terminal_valid": "True",
        "exact_l1": "True",
        "exact_measured_combination": "false",
        "exact_unseen_roles_json": '["amine"]',
        "bounded_neighborhood": "local_smoothed_neighborhood",
        "branch_class": "linear_tail_origins",
        "program_index": "0",
        **{f"{view}_distribution_bin": "interpolative" for view in VIEWS},
    }
    invalid = {
        **valid,
        "terminal_valid": "False",
        **{f"{view}_distribution_bin": "" for view in VIEWS},
    }

    result = gate_diagnostics([valid, invalid], source="lambda_zero_seed")

    assert result["valid_exact_l1"]["count"] == 1
    assert result["active_intersection"]["among_attempts_or_records"]["count"] == 1
    assert result["novel_all_four_views_interpolative"]["denominator"] == 1


def test_finite_quantile_uses_finite_sample_upper_index() -> None:
    assert _finite_quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 3.0
    assert _finite_quantile([1.0, 2.0, 3.0, 4.0], 0.9) == 4.0


def test_spearman_handles_ties_and_constant_predictions() -> None:
    assert _spearman(np.asarray([1.0, 2.0, 3.0]), np.asarray([3.0, 2.0, 1.0])) == pytest.approx(
        -1.0
    )
    assert _spearman(np.asarray([1.0, 1.0]), np.asarray([1.0, 2.0])) is None
