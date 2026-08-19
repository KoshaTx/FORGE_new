from __future__ import annotations

import json

import pytest

from forge.potency.ugi_generated_role_applicability_census import (
    ROLES,
    VIEWS,
    _policy_classification,
    _radius_ratio,
    summarize_applicability_rows,
)


def _source(index: int, *, roles: tuple[str, ...]) -> dict[str, str]:
    return {
        "sample_index": str(index),
        "product_smiles": "CCNC(=O)C(NC)C",
        "amine_smiles": "CN",
        "aldehyde_smiles": "CC=O",
        "isocyanide_smiles": "[C-]#[N+]C",
        "exact_unseen_roles_json": json.dumps(list(roles), separators=(",", ":")),
    }


def _derived(
    index: int,
    *,
    roles: tuple[str, ...],
    bins: dict[str, str] | None = None,
    provenance: str = "exact_new_component_identity",
) -> dict[str, object]:
    selected_bins = bins or {view: "interpolative" for view in VIEWS}
    row: dict[str, object] = {
        "sample_index": index,
        "product_id": f"p{index}",
        "exact_identity_provenance": provenance,
        "exact_unseen_roles_json": json.dumps(list(roles), separators=(",", ":")),
        "branch_class": "linear_tail_origins",
        "current_policy_action": (
            "abstain" if roles == ROLES else "applicability_supported_no_score"
        ),
        "current_policy_reason": (
            "unsupported_all_three_new"
            if roles == ROLES
            else "supported_pattern_all_views_interpolative"
        ),
        "noninterpolative_views_json": json.dumps(
            [view for view in VIEWS if selected_bins[view] != "interpolative"],
            separators=(",", ":"),
        ),
        "extrapolative_views_json": json.dumps(
            [view for view in VIEWS if selected_bins[view] == "extrapolative"],
            separators=(",", ":"),
        ),
    }
    for view in VIEWS:
        row[f"{view}_distribution_bin"] = selected_bins[view]
        row[f"{view}_fingerprint_distance"] = 0.1
        row[f"{view}_descriptor_distance"] = 0.2
        row[f"{view}_interpolative_radius_ratio"] = 0.8
        row[f"{view}_boundary_radius_ratio"] = 0.4
    return row


def test_all_three_new_remains_abstained_even_when_every_view_is_interpolative() -> None:
    row = {
        "exact_unseen_roles_json": json.dumps(list(ROLES)),
        "exact_identity_provenance": "exact_new_component_identity",
        **{f"{view}_distribution_bin": "interpolative" for view in VIEWS},
    }
    supported = {
        ("amine",): "amine_only",
        ("aldehyde", "isocyanide"): "aldehyde_isocyanide_pair",
    }

    assert _policy_classification(row, supported) == (
        "",
        "abstain",
        "unsupported_all_three_new",
    )


def test_radius_ratio_uses_worst_fingerprint_or_descriptor_view() -> None:
    row = {
        "amine_fingerprint_distance": 0.25,
        "amine_descriptor_distance": 0.4,
    }
    thresholds = {
        "amine": {
            "fingerprint": {"interpolative_max": 0.5},
            "descriptor": {"interpolative_max": 0.5},
        }
    }

    assert _radius_ratio(row, thresholds, "amine", "interpolative_max") == pytest.approx(0.8)


def test_summary_retains_fully_generated_all_three_new_as_support_only() -> None:
    derived = [
        _derived(0, roles=ROLES),
        _derived(1, roles=("amine",)),
    ]
    sources = [
        _source(0, roles=ROLES),
        _source(1, roles=("amine",)),
    ]

    summary = summarize_applicability_rows(derived, sources)

    assert summary["records"] == 2
    assert summary["joint"]["all_three_roles_exact_new"] == 1
    assert summary["joint"]["all_three_roles_exact_new_all_four_views_interpolative"] == 1
    assert summary["joint"]["current_policy_actions"] == {
        "abstain": 1,
        "applicability_supported_no_score": 1,
    }
    assert summary["roles"]["amine"]["exact_new"]["records"] == 2
    assert summary["roles"]["aldehyde"]["exact_new"]["records"] == 1
    assert summary["roles"]["isocyanide"]["exact_new"]["records"] == 1
