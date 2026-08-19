from __future__ import annotations

from forge.route.ugi3_precursor_leaf_closure import ALDEHYDE_ROLE, HEAD_ROLE, ISOCYANIDE_ROLE
from forge.route.ugi3_route_gap_triage import classify_route_gap


def test_current_exact_target_precedes_upstream_planning() -> None:
    result = classify_route_gap(
        role=HEAD_ROLE,
        target_smiles="CCN",
        assessment_outcome="missing_knowledge",
        current_terminal_keys={"CCN"},
        historical_leaf_keys=set(),
    )
    assert result["triage_class"] == "current_target_terminal_available_value_refresh"
    assert result["new_reaction_family_indicated_now"] is False


def test_existing_isocyanide_family_with_current_leaf_needs_exact_scope() -> None:
    result = classify_route_gap(
        role=ISOCYANIDE_ROLE,
        target_smiles="CCCCCCCCCCCC[N+]#[C-]",
        assessment_outcome="missing_knowledge",
        current_terminal_keys={"CCCCCCCCCCCCN"},
        historical_leaf_keys=set(),
    )
    assert result["triage_class"] == "projected_family_all_leaves_current_exact_scope_missing"
    assert result["program_family"] == "primary_amine_formylation_then_formamide_dehydration"


def test_existing_aldehyde_family_preserves_leaf_gap() -> None:
    result = classify_route_gap(
        role=ALDEHYDE_ROLE,
        target_smiles="CCCCCCCC=O",
        assessment_outcome="missing_knowledge",
        current_terminal_keys=set(),
        historical_leaf_keys=set(),
    )
    assert result["triage_class"] == "projected_family_leaf_and_exact_scope_gap"
    assert result["unresolved_projected_leaves_json"] != "[]"


def test_outside_support_is_not_promoted_to_family_mining() -> None:
    result = classify_route_gap(
        role=ALDEHYDE_ROLE,
        target_smiles="CCC(C)CC(C)CC=O",
        assessment_outcome="outside_support",
        current_terminal_keys=set(),
        historical_leaf_keys=set(),
    )
    assert result["triage_class"] == "outside_declared_route_support_review"
    assert result["new_reaction_family_indicated_now"] is False
