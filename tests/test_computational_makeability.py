"""Pure evidence classification: no network, chemistry replay, or models."""

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from forge.synthesis.assessment.computational_makeability import (
    BranchEvidence,
    BranchRequirement,
    ComponentEvidence,
    ComponentState,
    ComputationalRoute,
    EvidenceAxes,
    ListingState,
    LookupOutcome,
    MakeabilityError,
    MakeabilityPolicy,
    ProductState,
    Receipt,
    RouteBasis,
    RouteNode,
    RouteStep,
    SearchOutcome,
    VendorListing,
    classify_component,
    classify_listing,
    classify_product,
)

NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)
POLICY = MakeabilityPolicy.from_mapping(
    {
        "schema": "forge.computational_makeability_policy.v1",
        "policy_id": "compose_lipid_computational_makeability_v1",
        "vendor_snapshot_max_age_days": 30,
    }
)
PIN = Receipt("https://supplier.example/exact-item", hashlib.sha256(b"source").hexdigest())


def listing(identity, count=1, observed=NOW):
    return VendorListing(identity, LookupOutcome.SUCCESS, count, observed, PIN)


def leaf(identity, count=1):
    return RouteNode(identity, listings=(listing(identity, count),))


def route(route_id="r1", children=None, *, basis=RouteBasis.PLANNER_SOLVED, target="CCO"):
    return ComputationalRoute(
        route_id,
        basis,
        RouteNode(
            target,
            RouteStep(
                children if children is not None else (leaf("CC=O"),),
                PIN,
                forward_products=(target, "CO"),
                transform_receipt=PIN,
                axes=EvidenceAxes(independent_forward_replay=False, unique_forward_product=False),
            ),
        ),
        PIN,
    )


def component(*, direct=(), routes=(), **kwargs):
    return classify_component(
        ComponentEvidence("CCO", direct, routes, **kwargs), policy=POLICY, as_of=NOW
    )


def product(required=None, supplied=None, **kwargs):
    requirement = BranchRequirement("tail", "tail", "CCO", 2, ("assembly",))
    return classify_product(
        "request-1",
        l1_exact=kwargs.pop("l1_exact", True),
        l1_receipt=kwargs.pop("l1_receipt", PIN),
        required_branches=(requirement,) if required is None else required,
        evidence=(
            (BranchEvidence(requirement, ComponentEvidence("CCO", (listing("CCO"),))),)
            if supplied is None
            else supplied
        ),
        policy=POLICY,
        as_of=NOW,
        **kwargs,
    )


def test_direct_positive_listing_needs_no_sku_purity_stock_or_shipping():
    result = component(direct=(listing("CCO"),))
    assert result.state is ComponentState.BUY
    assert result.makeable
    assert product().state is ProductState.MAKEABLE


def test_planner_solved_path_needs_no_independent_forward_or_unique_product():
    candidate = route()
    candidate = replace(
        candidate,
        root=replace(
            candidate.root,
            step=replace(candidate.root.step, transform_receipt=None, forward_products=None),
        ),
    )
    result = component(routes=(candidate,))
    assert result.state is ComponentState.MAKE_FROM_VENDOR_LISTED
    assert result.routes[0].complete
    assert result.routes[0].evidence.root.step.axes.independent_forward_replay is False


def test_constructed_path_accepts_target_among_multiple_products():
    candidate = route(basis=RouteBasis.FORWARD_APPLIED)
    assert component(routes=(candidate,)).makeable
    assert (
        component(direct=(listing("CCO"),), routes=(candidate,)).state
        is ComponentState.BUY_AND_MAKE
    )


def test_multistep_constructed_path_checks_every_application():
    intermediate = RouteNode("CC=O", RouteStep((leaf("CC"),), PIN, ("CC=O",), PIN))
    candidate = route(children=(intermediate,), basis=RouteBasis.FORWARD_APPLIED)
    assert component(routes=(candidate,)).makeable
    bad = replace(intermediate, step=replace(intermediate.step, forward_products=("CO",)))
    result = component(routes=(route(children=(bad,), basis=RouteBasis.FORWARD_APPLIED),))
    assert not result.makeable
    assert "target_not_in_step_forward_products" in result.routes[0].reasons


def test_constructed_path_requires_known_transform_receipt():
    candidate = route(basis=RouteBasis.FORWARD_APPLIED)
    candidate = replace(
        candidate,
        root=replace(candidate.root, step=replace(candidate.root.step, transform_receipt=None)),
    )
    result = component(routes=(candidate,))
    assert not result.makeable
    assert result.routes[0].reasons == ("missing_known_or_source_transform_receipt",)


def test_partial_path_retains_every_leaf_and_cannot_pass():
    result = component(routes=(route(children=(leaf("CC"), RouteNode("O"))),))
    assert result.state is ComponentState.ROUTE_ONLY
    assert not result.makeable
    assert [item.identity for item in result.routes[0].leaves] == ["CC", "O"]
    assert result.routes[0].leaves[1].listings == ()


def test_alternatives_are_or_but_leaves_cannot_mix_between_paths():
    first = route("first", (leaf("CC"), RouteNode("O")))
    second = route("second", (RouteNode("CC"), leaf("O")))
    assert not component(routes=(first, second)).makeable
    complete = route("third", (leaf("CC"), leaf("O")))
    result = component(routes=(first, second, complete))
    assert result.makeable
    assert [item.complete for item in result.routes] == [False, False, True]


def test_exact_identity_required_in_direct_leaf_and_route_root():
    assert component(direct=(listing("CO"),)).direct[0].state is ListingState.IDENTITY_MISMATCH
    bad_leaf = RouteNode("CC", listings=(listing("CCC"),))
    assert not component(routes=(route(children=(bad_leaf,)),)).makeable
    result = component(routes=(route(target="CO"),))
    assert not result.makeable
    assert "target_identity_mismatch" in result.routes[0].reasons


@pytest.mark.parametrize(
    "age,state",
    [
        (-1, ListingState.FUTURE),
        (0, ListingState.LISTED),
        (29.999, ListingState.LISTED),
        (30, ListingState.STALE),
        (31, ListingState.STALE),
    ],
)
def test_snapshot_interval_is_half_open_and_never_accepts_future(age, state):
    result = classify_listing(
        "CCO", listing("CCO", observed=NOW - timedelta(days=age)), policy=POLICY, as_of=NOW
    )
    assert result.state is state


def test_lookup_failure_unknown_and_actual_zero_are_distinct():
    failed = VendorListing("CCO", LookupOutcome.FAILED, receipt=PIN)
    unknown = VendorListing("CCO", LookupOutcome.UNASSESSED)
    assert component(direct=(failed,)).direct[0].state is ListingState.FAILED
    assert component(direct=(unknown,)).direct[0].state is ListingState.UNKNOWN
    assert component(direct=(listing("CCO", 0),)).state is ComponentState.UNKNOWN
    blocked = component(
        direct=(listing("CCO", 0),), search_outcome=SearchOutcome.COMPLETED, search_receipt=PIN
    )
    assert blocked.state is ComponentState.BLOCKED
    assert (
        component(
            direct=(failed,), search_outcome=SearchOutcome.COMPLETED, search_receipt=PIN
        ).state
        is ComponentState.UNKNOWN
    )


def test_empty_route_and_cyclic_route_do_not_count():
    empty = ComputationalRoute("empty", RouteBasis.PLANNER_SOLVED, leaf("CCO"), PIN)
    cycle = route(children=(leaf("CCO"),))
    assert not component(routes=(empty,)).makeable
    assert "cyclic_route" in component(routes=(cycle,)).routes[0].reasons


def test_required_branch_omission_retains_denominator():
    result = product(supplied=())
    assert result.state is ProductState.UNKNOWN
    assert result.branches[0].reason == "missing_branch"
    output = result.to_dict()
    assert (
        output["required_branches"],
        output["represented_branches"],
        output["makeable_branches"],
    ) == (1, 0, 0)
    assert not output["strict_dossier_admission"]
    assert not output["synthesis_guidance_admission"]


@pytest.mark.parametrize(
    "change",
    [
        {"role": "head"},
        {"identity": "CO"},
        {"quantity": 1},
        {"stages": ("other",)},
    ],
)
def test_full_branch_binding_not_identity_only(change):
    required = BranchRequirement("tail", "tail", "CCO", 2, ("assembly",))
    supplied = BranchEvidence(
        replace(required, **change), ComponentEvidence("CCO", (listing("CCO"),))
    )
    result = product(required=(required,), supplied=(supplied,))
    assert result.state is ProductState.UNKNOWN
    assert result.branches[0].reason == "branch_binding_mismatch"


def test_l1_nonexact_missing_and_empty_are_not_success():
    assert product(l1_exact=False).state is ProductState.NONEXACT_L1
    assert product(l1_exact=None).state is ProductState.UNKNOWN
    assert product(l1_receipt=None).state is ProductState.UNKNOWN
    assert product(required=(), supplied=()).state is ProductState.UNKNOWN
    assert product(l1_exact=False).to_dict()["required_branches"] == 1


def test_no_bare_strict_closed_shortcut():
    with pytest.raises(TypeError):
        ComponentEvidence("CCO", strict_closed=True)
    assert not component().makeable


@pytest.mark.parametrize("count", [True, False, 1.5, "1", -1])
def test_malformed_vendor_counts_fail_loudly(count):
    with pytest.raises(MakeabilityError, match="vendor_count"):
        listing("CCO", count)


def test_receipts_timestamps_policy_and_duplicate_branches_validate():
    with pytest.raises(MakeabilityError, match="timestamp"):
        VendorListing("CCO", LookupOutcome.SUCCESS, 1)
    with pytest.raises(MakeabilityError, match="timezone-aware"):
        component(direct=(listing("CCO", observed=NOW.replace(tzinfo=None)),))
    with pytest.raises(MakeabilityError, match="SHA-256"):
        Receipt("source", "not-a-hash")
    PIN.verify_bytes(b"source")
    with pytest.raises(MakeabilityError, match="digest mismatch"):
        PIN.verify_bytes(b"changed")
    with pytest.raises(MakeabilityError, match="schema"):
        MakeabilityPolicy.from_mapping({"schema": "other"})
    required = BranchRequirement("tail", "tail", "CCO")
    with pytest.raises(MakeabilityError, match="duplicate"):
        product(required=(required, required), supplied=())
    with pytest.raises(MakeabilityError, match="undeclared"):
        product(required=())


def test_serialization_is_complete_json_and_secondary_axes_do_not_gate():
    required = BranchRequirement("tail", "tail", "CCO", 2, ("assembly",))
    candidate = route(basis=RouteBasis.FORWARD_APPLIED)
    axes = EvidenceAxes(False, False, False, False, False)
    candidate = replace(
        candidate, root=replace(candidate.root, step=replace(candidate.root.step, axes=axes))
    )
    output = product(
        supplied=(BranchEvidence(required, ComponentEvidence("CCO", routes=(candidate,))),)
    ).to_dict()
    saved = json.loads(json.dumps(output))
    assert saved["state"] == "computationally_makeable"
    step = saved["branches"][0]["assessment"]["routes"][0]["evidence"]["root"]["step"]
    assert step["axes"]["substrate_applicability"] is False
    assert step["receipt"]["sha256"] == PIN.sha256


def test_all_product_branches_required_even_with_one_success():
    first = BranchRequirement("head", "head", "N")
    second = BranchRequirement("tail", "tail", "CCO")
    supplied = (
        BranchEvidence(first, ComponentEvidence("N", (listing("N"),))),
        BranchEvidence(
            second, ComponentEvidence("CCO", routes=(route(children=(RouteNode("CC"),)),))
        ),
    )
    result = product(required=(first, second), supplied=supplied)
    assert result.state is ProductState.UNKNOWN
    assert result.to_dict()["makeable_branches"] == 1
    assert result.to_dict()["required_branches"] == 2
