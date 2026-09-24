"""The bounded exact-source review must not inflate documentary component evidence."""

from __future__ import annotations

import copy
import json

import pytest

from forge.core.hashing import PinError
from results.phase1.compose_lipid_route_evidence_v2.adjudicate import (
    HERE,
    adjudicate_aema,
    adjudicate_cached_vendor,
    assess_admitted_step,
)


@pytest.fixture
def review():
    return json.loads((HERE / "source_review.json").read_text())


def test_exact_source_step_preserves_alternative_inverse_and_missing_leaves(review):
    result = adjudicate_aema(review)
    assert result["record"]["disposition"] == "admit_exact"
    assert all(result["mechanical_checks"].values())
    assert len(result["all_inverse_candidates"]) == 2
    assert result["registered_net_byproducts"] == {"H": 1, "Cl": 1, "formal_charge": 0}
    assert result["complete_component_dossier"] is None
    assert not result["admitted_complete_component_dossier"]
    assert len(result["remaining_leaves"]) == 2
    assert all(r["current_l3"] is None for r in result["remaining_leaves"])
    assert not result["computed_quantity_audit"][
        "numeric_stoichiometry_and_yield_supervision_admitted"
    ]
    integrated = assess_admitted_step(result)["assessment"]
    assert integrated["outcome"] == "missing_knowledge"
    assert integrated["route_tree"]["step"]["forward_product_count"] == 1
    assert len(integrated["route_tree"]["children"]) == 2
    assert all(
        child["outcome"] == "missing_knowledge" for child in integrated["route_tree"]["children"]
    )


def test_source_identity_conflict_cannot_be_admitted(review):
    review["claim"]["source_structure_status"] = "unresolved_source_identity"
    result = adjudicate_aema(review)
    assert result["record"]["disposition"] == "abstain"
    assert result["record"]["evidence_basis"] == "source_conflict"


def test_failed_forward_product_cannot_be_admitted(review):
    review["claim"]["canonical_smiles"] = "C=CC(=O)OCCOC(=O)C=C"
    result = adjudicate_aema(review)
    assert result["record"]["disposition"] == "abstain"
    assert not result["mechanical_checks"]["unique_unfiltered_forward_product"]


def test_source_tamper_missing_procedure_and_unreviewed_pages_fail_closed(review):
    damaged = copy.deepcopy(review)
    damaged["source"]["sha256"] = "0" * 64
    with pytest.raises(PinError, match="pinned input for Zhou SI changed"):
        adjudicate_aema(damaged)
    damaged = copy.deepcopy(review)
    damaged["claim"]["procedure"].pop("purification")
    with pytest.raises(ValueError, match="Incomplete source procedure"):
        adjudicate_aema(damaged)
    damaged = copy.deepcopy(review)
    damaged["source"]["visually_reviewed_physical_pages"] = [19]
    with pytest.raises(ValueError, match="neighboring pages"):
        adjudicate_aema(damaged)


def test_smiles_atom_order_does_not_change_admission(review):
    original = adjudicate_aema(review)
    review["claim"]["canonical_smiles"] = "CC(=C)C(=O)OCCOC(=O)C=C"
    review["claim"]["precursors"]["acyl_tail"]["canonical_smiles"] = "ClC(=O)C=C"
    review["claim"]["precursors"]["aminoalcohol_head"]["canonical_smiles"] = "OCCOC(=O)C(C)=C"
    reordered = adjudicate_aema(review)
    assert original["record"] == reordered["record"]
    assert original["mechanical_checks"] == reordered["mechanical_checks"]
    assert original["all_inverse_candidates"] == reordered["all_inverse_candidates"]


def test_cached_stock_and_later_review_timestamp_never_create_current_l3():
    rows = json.loads((HERE / "vendor_observations.json").read_text())["observations"]
    for row in rows:
        assert not adjudicate_cached_vendor(row)["admitted_current_terminal"]
        row["reviewed_at_utc"] = "2030-01-01T00:00:00Z"
        assert adjudicate_cached_vendor(row)["current_l3"] is None
    rows[0]["stock_observed_at_utc"] = "2026-09-24T00:00:00Z"
    with pytest.raises(ValueError, match="cannot authenticate a live stock timestamp"):
        adjudicate_cached_vendor(rows[0])


def test_wrong_vendor_structure_cannot_be_admitted():
    row = json.loads((HERE / "vendor_observations.json").read_text())["observations"][0]
    row["source_smiles"] = "CN(C)CCN"
    with pytest.raises(ValueError, match="constitution disagree"):
        adjudicate_cached_vendor(row)


def test_saved_all_family_impact_never_promotes_partial_closure():
    result = json.loads((HERE / "impact.json").read_text())
    assert len(result["families"]) == 22
    assert result["summary"]["exact_l1"] == 1288
    assert result["summary"]["requests_with_new_exact_L2_step"] == 47
    assert result["summary"]["requests_with_selected_identity_for_all_roles"] == 0
    assert result["summary"]["current_complete_products_admitted"] == 0
    assert all(r["current_complete_dossiers"] is None for r in result["families"])
