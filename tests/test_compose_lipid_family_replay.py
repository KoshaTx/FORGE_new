"""Source recipe identity and quantity must remain authoritative during replay."""

import hashlib

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import LibraryAssemblyError
from forge.corpus.compose_lipid_family_replay import fixed_components, replay_record


def item(**changes):
    preparation = {
        "eligible_for_program_preparation": True,
        "component_instances": [["head", "a", 1], ["tail", "b", 1]],
        "constitution_id": hashlib.sha256(b"CCO").hexdigest(),
    }
    preparation.update(changes)
    return {"source": {"constitution": "CCO"}, "preparation": preparation}


STRUCTURES = {"a": "CC", "b": "O", "c": "N"}
MAPPING = {"amine": "head", "body": "tail"}


def test_exact_components_are_resolved_by_global_identity_and_explicit_binding():
    result, reason = fixed_components(
        item()["preparation"]["component_instances"], STRUCTURES, MAPPING
    )
    assert result == {"amine": "CC", "body": "O"}
    assert reason is None
    assert fixed_components(
        list(reversed(item()["preparation"]["component_instances"])), STRUCTURES, MAPPING
    ) == (result, reason)


@pytest.mark.parametrize(
    "instances,reason",
    [
        ([["head", "a", 1], ["tail", "b", 2]], "outside_qualified_program_multiplicity"),
        ([["head", "a", 1], ["tail", "b", 1], ["head", "c", 1]], "unsupported_source_role_tuple"),
        ([["amine", "a", 1], ["tail", "b", 1]], "unsupported_source_role_tuple"),
        ([["head", "a", 1]], "unsupported_source_role_tuple"),
    ],
)
def test_unsupported_recipe_abstains_before_accessing_target(instances, reason):
    supplied = item(component_instances=instances)
    supplied["source"] = {}
    result = replay_record(supplied, STRUCTURES, {"kind": "fixed", "mapping": MAPPING, "run": None})
    assert result == {"computed_consistency_pass": False, "disposition": reason}


@pytest.mark.parametrize(
    "instances",
    [
        [["head", "missing", 1]],
        [["head", "a", True]],
        [["head", "a", 0]],
    ],
)
def test_invalid_or_unresolved_components_fail_closed(instances):
    with pytest.raises(ComposeLipidError):
        fixed_components(instances, STRUCTURES, MAPPING)


def test_protected_product_never_reaches_executor():
    supplied = item(eligible_for_program_preparation=False)
    supplied["source"] = {}
    with pytest.raises(ComposeLipidError, match="Protected or unassigned"):
        replay_record(supplied, {}, None)


def test_unqualified_family_is_explicit_and_does_not_read_product():
    supplied = item()
    supplied["source"] = {}
    assert (
        replay_record(supplied, {}, None)["disposition"] == "pending_source_program_qualification"
    )


def test_competing_outcomes_cannot_be_relabelled_exact():
    def run(components, target):
        assert components == {"amine": "CC", "body": "O"}
        return {
            "computed_consistency_pass": False,
            "checks": {"unique": False},
            "forward_products": [target, "CCN"],
        }

    result = replay_record(item(), STRUCTURES, {"kind": "fixed", "mapping": MAPPING, "run": run})
    assert result["disposition"] == "unresolved_computed_reconstruction"
    assert len(result["forward_products"]) == 2


def test_saturated_search_is_recorded_as_unsupported():
    def run(*args):
        raise LibraryAssemblyError("search saturated")

    result = replay_record(item(), STRUCTURES, {"kind": "fixed", "mapping": MAPPING, "run": run})
    assert result["disposition"] == "unsupported_registry_replay"
    assert result["reason"] == "search saturated"


@pytest.mark.parametrize("checks", [{}, {"complete": False}])
def test_executor_cannot_claim_pass_without_complete_checks(checks):
    with pytest.raises(ComposeLipidError, match="passing checks"):
        replay_record(
            item(),
            STRUCTURES,
            {
                "kind": "fixed",
                "mapping": MAPPING,
                "run": lambda *_: {"computed_consistency_pass": True, "checks": checks},
            },
        )


def test_passing_target_must_match_prior_authenticated_identity():
    executor = {
        "kind": "fixed",
        "mapping": MAPPING,
        "run": lambda *_: {"computed_consistency_pass": True, "checks": {"exact": True}},
    }
    result = replay_record(item(), STRUCTURES, executor)
    assert result["verified_target_constitution_id"] == item()["preparation"]["constitution_id"]
    with pytest.raises(ComposeLipidError, match="authenticated TRAIN identity"):
        replay_record(item(constitution_id="wrong"), STRUCTURES, executor)
