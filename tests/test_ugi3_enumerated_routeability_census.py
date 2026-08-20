from __future__ import annotations

import pytest

from forge.route.audit.ugi3_enumerated_routeability_census import (
    EXACT_COMPLETE,
    EXACT_L3_OPEN,
    FAMILY_PROJECTED,
    MISSING_KNOWLEDGE,
    OUTSIDE_SUPPORT,
    Ugi3EnumeratedRouteabilityCensusError,
    classify_product_routeability,
    component_route_tier,
)


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"hybrid_outcome": "complete", "search_channel": "exact_baseline"}, EXACT_COMPLETE),
        (
            {"hybrid_outcome": "missing_knowledge", "search_channel": "family_projection"},
            FAMILY_PROJECTED,
        ),
        (
            {"hybrid_outcome": "missing_knowledge", "search_channel": "exact_evidence_l3_open"},
            EXACT_L3_OPEN,
        ),
        (
            {"hybrid_outcome": "missing_knowledge", "search_channel": "provenance_priority"},
            MISSING_KNOWLEDGE,
        ),
        (
            {"hybrid_outcome": "outside_support", "search_channel": "no_admitted_route"},
            OUTSIDE_SUPPORT,
        ),
    ],
)
def test_component_route_tier_is_evidence_bounded(record: dict[str, str], expected: str) -> None:
    assert component_route_tier(record) == expected


@pytest.mark.parametrize(
    ("tiers", "expected"),
    [
        ((EXACT_COMPLETE, EXACT_COMPLETE, EXACT_COMPLETE), "all_exact_route_complete"),
        ((EXACT_COMPLETE, FAMILY_PROJECTED, EXACT_COMPLETE), "family_projected_no_other_gap"),
        (
            (EXACT_COMPLETE, EXACT_COMPLETE, EXACT_L3_OPEN),
            "exact_route_l3_open_no_family_projection",
        ),
        (
            (EXACT_COMPLETE, FAMILY_PROJECTED, EXACT_L3_OPEN),
            "family_projected_plus_l3_open",
        ),
        (
            (EXACT_COMPLETE, MISSING_KNOWLEDGE, FAMILY_PROJECTED),
            "missing_route_knowledge_present",
        ),
        (
            (EXACT_COMPLETE, MISSING_KNOWLEDGE, OUTSIDE_SUPPORT),
            "outside_current_route_support_present",
        ),
    ],
)
def test_product_class_uses_most_conservative_component(
    tiers: tuple[str, str, str], expected: str
) -> None:
    assert classify_product_routeability(tiers) == expected


def test_unknown_hybrid_outcome_fails_loudly() -> None:
    with pytest.raises(Ugi3EnumeratedRouteabilityCensusError, match="unsupported hybrid"):
        component_route_tier({"hybrid_outcome": "maybe", "search_channel": "guess"})
