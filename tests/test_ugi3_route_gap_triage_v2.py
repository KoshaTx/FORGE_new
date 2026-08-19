from __future__ import annotations

from forge.route.ugi3_precursor_leaf_closure import ALDEHYDE_ROLE, HEAD_ROLE
from forge.route.ugi3_route_gap_triage_v2 import (
    _family_discovery_cluster,
    classify_route_gap,
)


def test_current_terminal_precedes_stale_outside_support() -> None:
    result = classify_route_gap(
        role=HEAD_ROLE,
        target_smiles="NC1CCNCC1",
        assessment_outcome="outside_support",
        current_terminal_keys={"NC1CCNCC1"},
        historical_leaf_keys=set(),
    )
    assert result["triage_class"] == "current_target_terminal_available_value_refresh"
    assert result["target_current_terminal"] is True


def test_outside_support_without_current_terminal_remains_separate() -> None:
    result = classify_route_gap(
        role=ALDEHYDE_ROLE,
        target_smiles="CCC(C)CC(C)CC=O",
        assessment_outcome="outside_support",
        current_terminal_keys=set(),
        historical_leaf_keys=set(),
    )
    assert result["triage_class"] == "outside_declared_route_support_review"
    assert result["family_discovery_cluster"] == ""


def test_carbonate_aldehydes_share_one_candidate_family_cluster() -> None:
    first = _family_discovery_cluster(ALDEHYDE_ROLE, "CCCCCCCCOC(=O)OCCCCCC=O")
    second = _family_discovery_cluster(ALDEHYDE_ROLE, "CCCCCCCCCCCCCCOC(=O)OCCCCCC=O")
    assert first == "carbonate_linked_aldehyde_program_candidate"
    assert second == first


def test_unclustered_targets_do_not_aggregate_by_default() -> None:
    first = _family_discovery_cluster(HEAD_ROLE, "CN")
    second = _family_discovery_cluster(HEAD_ROLE, "CCN")
    assert first.startswith("unclustered_exact_identity_")
    assert second.startswith("unclustered_exact_identity_")
    assert first != second
