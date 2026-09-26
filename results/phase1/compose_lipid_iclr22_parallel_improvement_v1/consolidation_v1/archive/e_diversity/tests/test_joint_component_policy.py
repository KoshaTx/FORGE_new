"""Per-request quality and route safety boundaries for a saved candidate join."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from experiments.phase1.multireaction.component_concentration_selector import (
    ComponentIdentity,
    SelectionCandidate,
)
from experiments.phase1.multireaction.joint_component_policy import (
    ROUTE_KEYS,
    bind_route,
    eligibility,
    require_same_route_context,
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


def test_nonexact_baseline_has_no_fabricated_canonical_decomposition():
    c = SelectionCandidate(0, 0, "CN", True, False, False, True, ())
    q = {
        "index": 0,
        "family": "family",
        "selected_smiles": "CN",
        "selected_ordinal": 0,
        "exact_L1": False,
        "requirements": [],
    }
    v = {
        "index": 0,
        "family": "family",
        "exact_L1": False,
        "branches": [],
        **{k: False for k in ROUTE_KEYS},
    }
    assert bind_route(c, "family", q, v)
    with pytest.raises(ValueError, match="identity"):
        bind_route(c, "family", {**q, "selected_smiles": "CC"}, v)


@pytest.mark.parametrize("value", [1, "true", None])
def test_untyped_route_values_are_not_evidence(value):
    c, g, r = fixture()
    with pytest.raises(ValueError, match="Untyped"):
        eligibility(c, c, g, g, r, {**r, "combined_primary": value})


def test_alternative_cannot_belong_to_another_request():
    c, g, r = fixture()
    with pytest.raises(ValueError, match="different request"):
        eligibility(c, replace(c, request=2), g, g, r, r)


@pytest.mark.parametrize("where", ["candidate", "requirements", "branches"])
def test_duplicate_identity_rows_cannot_be_hidden_by_dictionary_join(where):
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
    if where == "candidate":
        c = SimpleNamespace(**{**vars(c), "components": c.components * 2})
    elif where == "requirements":
        q["requirements"] *= 2
    else:
        v["branches"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        bind_route(c, "family", q, v)


@pytest.mark.parametrize("field,value", [("quantity", 2), ("stages", ["another_stage"])])
def test_source_quantities_and_stages_do_not_change_with_identity(field, value):
    part = {"role": "head", "quantity": 1, "stages": ["assembly"]}
    before = {"requirements": [part]}
    require_same_route_context(before, before)
    with pytest.raises(ValueError, match="Changed source"):
        require_same_route_context(before, {"requirements": [{**part, field: value}]})


def test_nonexact_cannot_inherit_positive_proof():
    c = SelectionCandidate(0, 0, "CN", True, False, False, True, ())
    q = {
        "index": 0,
        "family": "family",
        "selected_smiles": "CN",
        "selected_ordinal": 0,
        "exact_L1": False,
        "requirements": [],
    }
    v = {
        "index": 0,
        "family": "family",
        "exact_L1": False,
        "branches": [],
        **{k: False for k in ROUTE_KEYS},
        "strict_secondary": True,
    }
    with pytest.raises(ValueError, match="Nonexact"):
        bind_route(c, "family", q, v)
