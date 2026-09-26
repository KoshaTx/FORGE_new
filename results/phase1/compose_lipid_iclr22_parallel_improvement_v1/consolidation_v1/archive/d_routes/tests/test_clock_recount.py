"""Counterfactual removes only exact newly admitted evidence, including internal nodes."""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments/phase1/route_improvement")
)
import clock_recount as module  # noqa: E402
from computational_makeability_v2 import (  # noqa: E402
    ComponentEvidence,
    ComputationalRoute,
    LookupOutcome,
    Receipt,
    RouteBasis,
    RouteNode,
    RouteStep,
    VendorListing,
)


def fixture():
    receipt = Receipt("test/old", "1" * 64)
    latest = Receipt("test/new", "2" * 64)
    old = VendorListing(
        "C", LookupOutcome.SUCCESS, 1, module.routes.old.time(module.routes.AS_OF), receipt
    )
    new = replace(old, receipt=latest)
    leaf = RouteNode("C", listings=(old, new))
    root = RouteNode("CC", RouteStep((leaf,), receipt), (new,))
    path = ComputationalRoute("old", RouteBasis.PLANNER_SOLVED, root, receipt)
    evidence = ComponentEvidence("CC", (old, new), (path, replace(path, route_id="new")))
    return evidence, old, new


def test_identity_control():
    evidence, _, _ = fixture()
    assert module.strip_new_component(evidence, set(), set()) == evidence


def test_only_exact_new_receipts_and_paths_removed():
    evidence, old, new = fixture()
    result = module.strip_new_component(evidence, {"new"}, {module.listing_key(new)})
    assert result.direct == (old,)
    assert len(result.routes) == 1 and result.routes[0].route_id == "old"
    assert result.routes[0].root.listings == ()
    assert result.routes[0].root.step.reactants[0].listings == (old,)
    assert result.routes[0].root.step.receipt == evidence.routes[0].root.step.receipt
    assert len(evidence.direct) == 2 and len(evidence.routes) == 2


def test_similar_but_wrong_receipt_does_not_remove():
    evidence, _, new = fixture()
    key = (new.receipt.location, "3" * 64)
    assert module.strip_new_component(evidence, {"new_other"}, {key}) == evidence
