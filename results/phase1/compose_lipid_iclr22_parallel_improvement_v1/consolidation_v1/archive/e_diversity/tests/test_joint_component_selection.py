"""Per-request quality and route safety boundaries for a saved candidate join."""

from dataclasses import replace

import pytest

from experiments.phase1.multireaction.component_concentration_selector import (
    ComponentIdentity,
    SelectionCandidate,
)
from experiments.phase1.multireaction.joint_component_selection import (
    ROUTE_KEYS,
    bind_route,
    eligibility,
)


def fixture():
    c = SelectionCandidate(
        0, 0, "CN", True, True, True, True, (ComponentIdentity("head", "N", True, True),)
    )
    gate = {"qualified_design_pass": True, "chemical": {"flags": []}}
    route = {k: True for k in ROUTE_KEYS}
    return c, gate, route


@pytest.mark.parametrize("key", ROUTE_KEYS)
def test_each_old_route_positive_requires_new_evidence(key):
    c, g, r = fixture()
    changed = {**r, key: False}
    assert eligibility(c, replace(c, ordinal=1), g, g, r, changed) == ["lost_" + key]


def test_does_not_inherit_strict_or_waive_design_for_diversity():
    c, g, r = fixture()
    new = {"qualified_design_pass": False, "chemical": {"flags": []}}
    assert eligibility(c, c, g, new, r, r) == ["lost_current_design_pass"]


def test_unknown_to_supported_and_context_removal_are_allowed():
    c, g, r = fixture()
    old = {k: False for k in ROUTE_KEYS}
    flag = {
        "qualified_design_pass": False,
        "chemical": {"flags": [{"tier": "context_required", "code": "old"}]},
    }
    assert eligibility(c, c, flag, g, old, r) == []


def test_new_context_code_is_refused():
    c, g, r = fixture()
    new = {
        "qualified_design_pass": True,
        "chemical": {"flags": [{"tier": "context_required", "code": "new"}]},
    }
    assert eligibility(c, c, g, new, r, r) == ["new_context_code"]


def test_route_identity_join_refuses_different_product_or_component():
    c, _, r = fixture()
    q = {
        "index": 0,
        "family": "family",
        "canonical_product": "CN",
        "selected_ordinal": 0,
        "exact_L1": True,
        "requirements": [{"role": "head", "identity": "N", "branch_id": "0:0"}],
    }
    v = {
        **r,
        "index": 0,
        "family": "family",
        "exact_L1": True,
        "branches": [{"branch_id": "0:0", "identity": "N"}],
    }
    assert bind_route(c, "family", q, v)
    with pytest.raises(ValueError, match="product identity"):
        bind_route(c, "family", {**q, "canonical_product": "CC"}, v)
    with pytest.raises(ValueError, match="components|component"):
        bind_route(
            c,
            "family",
            {**q, "requirements": [{"role": "head", "identity": "O", "branch_id": "0:0"}]},
            v,
        )
