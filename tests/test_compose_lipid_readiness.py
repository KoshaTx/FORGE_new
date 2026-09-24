"""Accounting must never promote ambiguous, mismatched or protected replay evidence."""

import copy
import hashlib

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.corpus.compose_lipid_readiness import validate_replay_row


def fixture():
    prepared = {
        "target_id": "t",
        "family": "family",
        "constitution_id": hashlib.sha256(b"CCN").hexdigest(),
        "component_instances": [["a", "id1", 1], ["b", "id2", 2]],
        "construction_basis": "source",
        "eligible_for_program_preparation": True,
        "old_split": "train",
        "corrected_split": "train",
        "exclusion_reasons": [],
        "pending_reasons": [],
    }
    row = {
        k: copy.deepcopy(prepared[k])
        for k in [
            "target_id",
            "family",
            "constitution_id",
            "component_instances",
            "construction_basis",
        ]
    }
    row.update(
        training_admitted=False,
        experimental_execution_admitted=False,
        replay={
            "computed_consistency_pass": True,
            "forward_layers": [["CN"], ["CCN"]],
            "checks": {"complete_search": True, "unique_forward_exact": True},
            "bound_reasons": [],
            "disposition": "exact_computed_reconstruction",
        },
    )
    return row, prepared


def test_exact_identity_and_complete_checks_are_required():
    row, prepared = fixture()
    assert validate_replay_row(row, prepared)


@pytest.mark.parametrize(
    "key,value",
    [
        ("old_split", "test"),
        ("corrected_split", "unassigned"),
        ("exclusion_reasons", ["held_component"]),
        ("pending_reasons", ["partition"]),
        ("eligible_for_program_preparation", False),
    ],
)
def test_protected_or_unassigned_cannot_be_admitted(key, value):
    row, prepared = fixture()
    prepared[key] = value
    with pytest.raises(ComposeLipidError, match="protected or unresolved"):
        validate_replay_row(row, prepared)


@pytest.mark.parametrize(
    "key", ["target_id", "family", "constitution_id", "component_instances", "construction_basis"]
)
def test_replay_cannot_change_any_source_recipe_field(key):
    row, prepared = fixture()
    row[key] = "changed"
    with pytest.raises(ComposeLipidError, match="changed source"):
        validate_replay_row(row, prepared)


@pytest.mark.parametrize(
    "change",
    [
        "wrong_product",
        "ambiguous_product",
        "empty_checks",
        "failed_check",
        "saturated",
        "training",
        "execution",
    ],
)
def test_incomplete_chemistry_and_unsupported_admission_fail(change):
    row, prepared = fixture()
    if change == "wrong_product":
        row["replay"]["forward_layers"][-1] = ["CCC"]
    if change == "ambiguous_product":
        row["replay"]["forward_layers"][-1] = ["CCN", "CCC"]
    if change == "empty_checks":
        row["replay"]["checks"] = {}
    if change == "failed_check":
        row["replay"]["checks"]["complete_search"] = False
    if change == "saturated":
        row["replay"]["bound_reasons"] = ["outcome_bound"]
    if change == "training":
        row["training_admitted"] = True
    if change == "execution":
        row["experimental_execution_admitted"] = True
    with pytest.raises(ComposeLipidError):
        validate_replay_row(row, prepared)


def test_unresolved_evidence_stays_unresolved():
    row, prepared = fixture()
    row["replay"] = {
        "computed_consistency_pass": False,
        "disposition": "unsupported_source_role_tuple",
    }
    assert not validate_replay_row(row, prepared)
