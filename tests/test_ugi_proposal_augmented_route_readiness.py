from __future__ import annotations

from pathlib import Path

from forge.synthesis.value.proposal_readiness import (
    EXACT,
    FAMILY_ALL,
    FAMILY_PARTIAL,
    UNRESOLVED,
    build_proposal_augmented_route_readiness,
    component_readiness_class,
    product_readiness_class,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi_proposal_augmented_route_readiness_v1.json"


def test_readiness_policy_is_minimum_across_roles() -> None:
    assert component_readiness_class(EXACT) == EXACT
    assert component_readiness_class("missing_knowledge") == UNRESOLVED
    assert product_readiness_class((EXACT, FAMILY_ALL, EXACT)) == FAMILY_ALL
    assert product_readiness_class((EXACT, FAMILY_ALL, FAMILY_PARTIAL)) == FAMILY_PARTIAL
    assert product_readiness_class((EXACT, UNRESOLVED, EXACT)) == UNRESOLVED


def test_frozen_proposal_augmented_readiness_is_deterministic() -> None:
    first_result, first_ledger = build_proposal_augmented_route_readiness(REPO, CONFIG)
    second_result, second_ledger = build_proposal_augmented_route_readiness(REPO, CONFIG)
    assert first_result == second_result
    assert first_ledger == second_ledger
    assert first_result["decision"]["graded_route_readiness_available_for_controller_qualification"]
    assert first_result["policy"]["proposal_model_score_used"] is False
    assert first_result["policy"]["utility_is_synthesis_success_probability"] is False
