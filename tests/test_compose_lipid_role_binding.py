"""Role translation must be supported by every precursor witness, never its target."""

import json

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.corpus import compose_lipid_role_binding as module
from forge.corpus.compose_lipid_role_binding import bind_recipe, fixed_binding


def examples():
    return (
        [["first_side", "global-a", 1], ["second_side", "global-b", 1]],
        {"global-a": "CCN", "global-b": "CC=O"},
        [
            {"role": "amine", "canonical_smiles": "CCN"},
            {"role": "aldehyde", "canonical_smiles": "CC=O"},
        ],
        {"amine", "aldehyde"},
    )


def test_binding_is_exact_and_permutation_invariant_without_product_input():
    instances, structures, prior, roles = examples()
    expected = {"first_side": "amine", "second_side": "aldehyde"}
    assert bind_recipe(instances, structures, prior, roles) == expected
    assert (
        bind_recipe(list(reversed(instances)), structures, list(reversed(prior)), roles) == expected
    )


@pytest.mark.parametrize(
    "kind", ["quantity", "missing", "duplicate", "ambiguous", "changed", "role"]
)
def test_incomplete_ambiguous_or_changed_witness_cannot_establish_binding(kind):
    instances, structures, prior, roles = examples()
    if kind == "quantity":
        instances[0][2] = 2
    if kind == "missing":
        del structures["global-a"]
    if kind == "duplicate":
        instances[1][0] = instances[0][0]
    if kind == "ambiguous":
        prior[1]["canonical_smiles"] = prior[0]["canonical_smiles"]
    if kind == "changed":
        structures["global-a"] = "CCO"
    if kind == "role":
        prior[1]["role"] = prior[0]["role"]
    with pytest.raises(ComposeLipidError):
        bind_recipe(instances, structures, prior, roles)


def test_no_majority_vote_or_empty_binding_is_allowed():
    with pytest.raises(ComposeLipidError, match="conflicts"):
        fixed_binding([{"a": "amine"}] * 10 + [{"a": "aldehyde"}])
    with pytest.raises(ComposeLipidError, match="needs exact"):
        fixed_binding([])
    assert fixed_binding([{"a": "amine"}] * 3) == ({"a": "amine"}, 3)


def test_prior_authentication_never_replays_the_superseded_population(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("An older population reached chemistry evaluation")

    monkeypatch.setattr(module.event, "_evaluate", forbidden)
    monkeypatch.setattr(module.event, "verify_source_event", forbidden)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"inputs": {}}))
    evidence = tmp_path / "prior.json"
    payload = {
        "schema_version": module.event.RESULT_SCHEMA,
        "status": "source_event_evaluated_training_unqualified",
        "policy": module.event.POLICY,
        "config": {"path": "config.json"},
        "inputs": {},
        "implementation": {name: {"path": name} for name in module.event.IMPLEMENTATION},
        "artifacts": {"programs.jsonl.gz": {"path": "programs.jsonl.gz"}},
    }
    evidence.write_text(json.dumps(payload))
    calls = []

    def resolve(value, repo, *, label):
        calls.append(label)
        return repo / value["path"]

    monkeypatch.setattr(module, "resolve_pin", resolve)
    assert module.authenticate_prior(tmp_path, evidence) == payload
    assert "programs.jsonl.gz" in calls
