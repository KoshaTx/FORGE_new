from __future__ import annotations

import pytest

from forge.potency.ranking.ugi_production_full_support_rescoring_v3 import (
    UgiProductionFullSupportRescoringV3Error,
    _attach_admission_metadata,
    _prepare_rows,
)


def _row(arm: str, draw: int, *, admitted: bool = True):
    return {
        "arm_id": arm,
        "draw_index": draw,
        "native_terminal": {"valid": True},
        "terminal_chemical_admission": {
            "admitted": admitted,
            "exact_l1": True,
            "raw_molecule_valid": True,
            "reason": "admitted" if admitted else "handle_policy_failure:amine_head",
        },
    }


def test_prepare_rows_masks_only_chemically_inadmissible_terminals():
    rows = [_row("broad_prior", 0), _row("support_enriched", 0, admitted=False)]
    prepared, metadata = _prepare_rows(rows, expected_draws_per_arm=1)
    assert prepared[0]["native_terminal"] == {"valid": True}
    assert prepared[1]["native_terminal"] is None
    assert metadata[("broad_prior", 0)]["terminal_chemical_admitted"] is True
    assert metadata[("support_enriched", 0)]["terminal_chemical_admitted"] is False


def test_attach_admission_metadata_preserves_admitted_reason_and_overrides_rejection():
    rows = [_row("broad_prior", 0), _row("support_enriched", 0, admitted=False)]
    _, metadata = _prepare_rows(rows, expected_draws_per_arm=1)
    classified = [
        {"arm_id": "broad_prior", "draw_index": 0, "reason": "interpolative"},
        {"arm_id": "support_enriched", "draw_index": 0, "reason": "invalid"},
    ]
    _attach_admission_metadata(classified, metadata)
    assert classified[0]["reason"] == "interpolative"
    assert classified[1]["reason"] == "terminal_chemical_admission:handle_policy_failure:amine_head"


def test_prepare_rows_rejects_inconsistent_admission():
    rows = [_row("broad_prior", 0), _row("support_enriched", 0)]
    rows[0]["terminal_chemical_admission"]["exact_l1"] = False
    with pytest.raises(UgiProductionFullSupportRescoringV3Error):
        _prepare_rows(rows, expected_draws_per_arm=1)
