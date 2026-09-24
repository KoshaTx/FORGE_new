"""Full-program rejection, fixed donor choice and novelty/diversity preservation."""

import copy

import pytest

from forge.corpus.qualified_program_cache import QualifiedProgramExample
from forge.model.compose_lipid_completion import (
    admit_generated_repeat_completions,
    check_generated_repeat_completion,
)
from forge.model.compose_lipid_layout import build_layout, summarize_layout
from tests.test_source_instance_coordinates import record


def example():
    r, _, atoms = record(
        "N(CC)CO",
        ["head", "tail", "tail", "tail", "tail"],
        ["amine", "carbon", "exterior", "carbon", "exterior"],
    )
    e = QualifiedProgramExample(r, "f", (("head", "head-id", 1), ("tail", "same-id", 2)), ())
    bundle, choices, rings = summarize_layout(e)
    layout = build_layout(bundle, choices, rings, identity="generated")
    state = dict(
        nodes=r.graph.node_states.tolist(),
        parents=r.graph.parents.tolist(),
        parent_bonds=r.graph.parent_bonds.tolist(),
        closure_left=[],
        closure_right=[],
        closure_bonds=[],
    )
    return layout, state, atoms


def checker(calls, *, donor_exact=True):
    def check(layout, smiles):
        calls.append(smiles)
        return dict(status="evaluated", exact=donor_exact and smiles != "CCNCO")

    return check


def test_first_generated_donor_checked_and_original_retained():
    layout, state, atoms = example()
    before = copy.deepcopy(state)
    calls = []
    row = check_generated_repeat_completion(layout, state, atoms, check_product=checker(calls))
    assert state == before
    assert calls[0] == "CCNCO"
    assert len(calls) == row["source_check_calls"] == 2
    assert calls[1] in {"CCNCC", "OCNCO"}
    assert row["proposals"][0]["donors"] == [0]
    assert len(row["proposals"]) == 1
    selected = admit_generated_repeat_completions([row], is_train_product=lambda _: False)[0]
    assert selected["accepted"] and selected["changed"]
    assert selected["original_smiles"] == "CCNCO"
    assert selected["selected_smiles"] == calls[1]


def test_rejected_first_donor_does_not_trigger_second_donor_or_pass():
    layout, state, atoms = example()
    calls = []
    row = check_generated_repeat_completion(
        layout, state, atoms, check_product=checker(calls, donor_exact=False)
    )
    selected = admit_generated_repeat_completions([row], is_train_product=lambda _: False)[0]
    assert len(calls) == 2
    assert not selected["accepted"] and not selected["changed"]
    assert selected["selected_smiles"] == "CCNCO"
    assert selected["proposals"][0]["status"] == "program_check_failed"


def test_exact_original_is_immutable():
    layout, state, atoms = example()
    calls = []

    def exact(layout, smiles):
        calls.append(smiles)
        return dict(status="evaluated", exact=True)

    row = check_generated_repeat_completion(layout, state, atoms, check_product=exact)
    assert calls == ["CCNCO"] and not row["proposals"]
    selected = admit_generated_repeat_completions([row], is_train_product=lambda _: False)[0]
    assert selected["accepted"] and not selected["changed"]
    assert selected["decision"]["disposition"] == "original_exact"


def test_decode_abstention_retains_attempt_without_checks():
    layout, state, atoms = example()
    calls = []
    row = check_generated_repeat_completion(
        layout, state, atoms, check_product=checker(calls), reason="closure_budget"
    )
    selected = admit_generated_repeat_completions([row], is_train_product=lambda _: False)[0]
    assert not calls and not selected["accepted"]
    assert selected["selected_smiles"] is None and selected["reason"] == "closure_budget"


def test_source_checker_failure_and_malformed_assessment_fail_loudly():
    layout, state, atoms = example()

    def fail(layout, smiles):
        raise RuntimeError("registry mismatch")

    with pytest.raises(RuntimeError, match="registry mismatch"):
        check_generated_repeat_completion(layout, state, atoms, check_product=fail)
    with pytest.raises(ValueError, match="full-program assessment"):
        check_generated_repeat_completion(
            layout, state, atoms, check_product=lambda *_: dict(status="abstained", exact=True)
        )


def test_novel_original_is_not_replaced_by_train_product():
    layout, state, atoms = example()
    row = check_generated_repeat_completion(layout, state, atoms, check_product=checker([]))
    candidate = row["proposals"][0]["smiles"]
    selected = admit_generated_repeat_completions(
        [row], is_train_product=lambda smiles: smiles == candidate
    )[0]
    assert not selected["changed"] and not selected["accepted"]
    assert selected["decision"]["proposal_checks"][0]["status"] == "would_reduce_train_novelty"


def test_completion_does_not_concentrate_population_on_existing_product():
    layout, state, atoms = example()
    row = check_generated_repeat_completion(layout, state, atoms, check_product=checker([]))
    exact = copy.deepcopy(row)
    exact.update(
        original_smiles=row["proposals"][0]["smiles"],
        original_check=dict(status="evaluated", exact=True),
        proposals=[],
    )
    selected = admit_generated_repeat_completions([row, exact], is_train_product=lambda _: False)
    assert len(selected) == 2 and not any(r["changed"] for r in selected)
    assert selected[0]["decision"]["proposal_checks"][0]["status"] == "would_concentrate_products"
    assert selected[1]["accepted"]


def test_forged_donor_or_exact_status_is_rejected():
    layout, state, atoms = example()
    row = check_generated_repeat_completion(layout, state, atoms, check_product=checker([]))
    row["proposals"][0]["donors"] = [1]
    with pytest.raises(ValueError, match="donor choice"):
        admit_generated_repeat_completions([row], is_train_product=lambda _: False)
    row["proposals"][0]["donors"] = [0]
    row["proposals"][0]["reaction_check"]["exact"] = False
    with pytest.raises(ValueError, match="full-program assessment"):
        admit_generated_repeat_completions([row], is_train_product=lambda _: False)
