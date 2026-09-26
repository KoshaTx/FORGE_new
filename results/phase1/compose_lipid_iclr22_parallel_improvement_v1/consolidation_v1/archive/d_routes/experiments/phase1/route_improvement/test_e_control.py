"""Control removes only the named admitted intervention routes, never other evidence."""

from e_full import without_new_routes

from forge.synthesis.assessment.computational_makeability import (
    BranchEvidence,
    BranchRequirement,
    ComponentEvidence,
    ComputationalRoute,
    Receipt,
    RouteBasis,
    RouteNode,
)


def fixture():
    receipt = Receipt("unit-test-only", "0" * 64)
    original = ComputationalRoute("original", RouteBasis.PLANNER_SOLVED, RouteNode("CC"), receipt)
    added = ComputationalRoute("D:0", RouteBasis.PLANNER_SOLVED, RouteNode("CC"), receipt)
    requirement = BranchRequirement("0:0", "unit-role", "CC", 2, ("unit-stage",))
    return (BranchEvidence(requirement, ComponentEvidence("CC", routes=(original, added))),)


def test_only_exact_admitted_id_is_removed():
    source = fixture()
    actual = without_new_routes(source, {"D:0"})
    assert actual[0].requirement is source[0].requirement
    assert actual[0].component.routes == source[0].component.routes[:1]
    assert len(source[0].component.routes) == 2
    assert actual[0].component.direct == source[0].component.direct
    assert actual[0].component.search_outcome == source[0].component.search_outcome
    assert actual[0].component.search_receipt == source[0].component.search_receipt


def test_absent_route_and_empty_intervention_are_identity_controls():
    source = fixture()
    assert without_new_routes(source, {"D:other"}) == source
    assert without_new_routes(source, set()) == source


def test_full_requested_branch_tuple_retained():
    source = fixture()
    actual = without_new_routes(source, {"D:0", "original"})
    assert len(actual) == len(source) == 1
    assert actual[0].component.routes == ()
    assert actual[0].requirement.quantity == 2
    assert actual[0].requirement.stages == ("unit-stage",)
