import copy

import pytest

from experiments.phase1.multireaction.augment_saved_pool_gates import joined_assessment


def fixture():
    original = {
        "chemical": {"canonical_smiles": "CC"},
        "head": {"status": "not_applicable"},
        "source_exact": True,
        "status_by_axis": {"ring": "abstain"},
    }
    after = {
        **original,
        "ring": {"tree_basis": "admitted_fixture_basis"},
        "status_by_axis": {"ring": "pass"},
        "qualified_design_pass": True,
    }
    candidate = {"request": 4, "ordinal": 2, "smiles": "CC"}
    target = {
        "admitted": True,
        "ambiguity": False,
        "errors": [],
        "candidate_lineages": 1,
        "index": 4,
        "ordinal": 2,
        "original_status": {"ring": "abstain"},
        "resulting_status": {"ring": "pass"},
        "resulting_design": True,
        "recovered": [
            {
                "assessment": after,
                "exact_graph_roundtrip": True,
                "roles_core_origins_layout_invariant": True,
            }
        ],
    }
    return target, candidate, original


def test_exact_bound_assessment_is_reused_without_mutating_original():
    target, candidate, original = fixture()
    before = copy.deepcopy(original)
    output = joined_assessment(target, candidate, original, "admitted_fixture_basis")
    assert output is target["recovered"][0]["assessment"]
    assert original == before


@pytest.mark.parametrize("mutation", ["ordinal", "smiles", "basis", "not_admitted", "changed_head"])
def test_identity_or_admission_mismatch_fails(mutation):
    target, candidate, original = fixture()
    if mutation == "ordinal":
        candidate["ordinal"] = 3
    elif mutation == "smiles":
        candidate["smiles"] = "CCC"
    elif mutation == "basis":
        target["recovered"][0]["assessment"]["ring"]["tree_basis"] = "invented"
    elif mutation == "not_admitted":
        target["admitted"] = False
    else:
        target["recovered"][0]["assessment"]["head"] = {"status": "pass"}
    with pytest.raises(ValueError):
        joined_assessment(target, candidate, original, "admitted_fixture_basis")
