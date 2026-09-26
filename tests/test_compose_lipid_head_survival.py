"""The new design predicate stays separate from unchanged complete-source admission."""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from forge.core.hashing import resolve_pin
from forge.model.compose_lipid_head_survival import assess_head_survival, propose_head_survival
from forge.model.precursor_reuse_projection import fixed_graph_preserved, state_graph
from results.phase1.compose_lipid_structure_repair_v1.head.run_case import HERE, policy, read


@pytest.fixture(scope="module")
def sample():
    receipt = HERE / "a17_fixture.json"
    if not receipt.exists():
        pytest.skip("Pinned saved A17 diagnostic fixture is absent")
    path = resolve_pin(read(receipt)["fixture"], Path.cwd(), label="head regression input")
    return torch.load(path, map_location="cpu", weights_only=False), policy()


def test_lost_head_regression_does_not_redefine_source_l1(sample):
    value, p = sample
    nodes, edges = state_graph(value["state"])
    result = assess_head_survival(value["layout"], nodes, edges, value["atoms"], p)
    assert result["status"] == "fail" and result["applicable"]
    assert result["retained_head_atoms"] == []
    assert result["consumed_head_atoms"]
    assert "source_l1" not in result


def test_positive_source_heads_and_cross_family_prior_are_explicit(sample):
    _, p = sample
    assert p.controls["by_family"] == {
        "aldehyde_ugi4": {"heads": 20, "failures": 0, "abstentions": 0},
        "ketone_ugi4": {"heads": 44, "failures": 0, "abstentions": 0},
    }
    assert any(
        source["smiles"] == "NC1CCNCC1"
        for sources in p.controls["environment_sources"].values()
        for source in sources
    )
    assert all(
        set(c["retained_atoms"]).isdisjoint(c["reactive_atoms"])
        for c in p.controls["unique_train_heads"]
    )


def test_other_families_are_not_applicable(sample):
    value, p = sample
    nodes, edges = state_graph(value["state"])
    layout = replace(value["layout"], family="ketone_ugi4")
    result = assess_head_survival(layout, nodes, edges, value["atoms"], p)
    assert result["status"] == "not_applicable" and not result["applicable"]


def test_charged_class_abstains(sample):
    value, p = sample
    nodes, edges = state_graph(value["state"])
    index = next(
        i
        for b in value["layout"].record.component_blocks
        if b.role == p.head_role
        for i in range(b.start, b.stop)
        if not value["layout"].record.fixed_atom_mask[i]
    )
    charged_n = next(
        i for i, a in enumerate(value["atoms"]) if a.symbol == "N" and a.formal_charge == 1
    )
    nodes[index] = charged_n
    result = assess_head_survival(value["layout"], nodes, edges, value["atoms"], p)
    assert result["status"] == "abstain" and not result["applicable"]


def test_design_match_cannot_override_source_failure_and_costs_are_bounded(sample):
    value, p = sample
    original = copy.deepcopy(value["state"])
    calls = []

    def reject(smiles):
        calls.append(smiles)
        return {"exact": False, "status": "unit_test_forced_source_failure"}

    result = propose_head_survival(
        value["layout"],
        value["state"],
        value["predictions"],
        value["atoms"],
        p,
        source_assessor=reject,
        maximum_candidates=2,
    )
    assert len(calls) == result["costs"]["source_checks"] == 2
    assert result["search_censored"]
    assert not result["changed"] and result["state"] == original == value["state"]
    assert all(
        not r["accepted"] and r["design_assessment"]["status"] == "pass"
        for r in result["proposals"]
    )
    for r in result["proposals"]:
        nodes, edges = state_graph(original)
        changed = np.flatnonzero(nodes != np.asarray(r["nodes"]))
        assert len(changed) == 1
        assert edges.tolist() == r["edges"]
        assert fixed_graph_preserved(np.asarray(r["nodes"]), edges, value["layout"].record)
        assert r["local_environment_sources"]


def test_missing_local_support_retains_original_without_source_calls(sample):
    value, p = sample

    def unexpected(_):
        raise AssertionError("Unsupported chemistry must not call source executor")

    result = propose_head_survival(
        value["layout"],
        value["state"],
        value["predictions"],
        value["atoms"],
        replace(p, train_environments=()),
        source_assessor=unexpected,
    )
    assert result["state"] == value["state"] and not result["changed"]
    assert result["costs"]["source_checks"] == 0


def test_nonfinite_predictions_reject_before_source_work(sample):
    value, p = sample
    predictions = copy.deepcopy(value["predictions"])
    predictions["nodes"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite node logits"):
        propose_head_survival(
            value["layout"],
            value["state"],
            predictions,
            value["atoms"],
            p,
            source_assessor=lambda _: {"exact": False},
        )
