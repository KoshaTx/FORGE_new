"""Atom-aware constructive readouts retain registry controls and graph conservation."""

import copy
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from forge.corpus.compose_lipid_aema_ester import load_contract
from forge.model.compose_lipid_quality_generation import with_retained_component_domain
from forge.model.compose_lipid_scaffold_construction import construct_scaffold_proposals
from forge.model.compose_lipid_symmetric_arms import construct_symmetric_role_arms
from forge.model.precursor_reuse_projection import graph_smiles, state_graph
from results.phase1.compose_lipid_quality_decode_v2.run import serial_state
from tests import test_compose_lipid_symmetric_arms as symmetric_controls
from tests.test_compose_lipid_scaffold_construction import fixture


@pytest.mark.needs_vendor
@pytest.mark.parametrize("index", [0, 1, 2])
def test_atom_aware_scaffold_preserves_source_controls_and_serializes_losslessly(index):
    layout, predictions, atoms, reaction, _ = fixture(index)
    old = construct_scaffold_proposals(layout, predictions, atoms, reaction)
    new = construct_scaffold_proposals(
        layout, predictions, atoms, reaction, atom_aware_topology=True
    )
    assert new == old
    for proposal in new["proposals"]:
        if not proposal.get("smiles"):
            continue
        state = serial_state(proposal["nodes"], proposal["edges"], layout)
        assert state is not None
        nodes, edges = state_graph(state)
        np.testing.assert_array_equal(nodes, proposal["nodes"])
        np.testing.assert_array_equal(edges, proposal["edges"])
        assert graph_smiles(nodes, edges, atoms) == proposal["smiles"]


@pytest.mark.needs_vendor
def test_atom_aware_symmetric_arms_preserve_complete_source_replay(monkeypatch):
    monkeypatch.setattr(
        symmetric_controls,
        "construct_symmetric_role_arms",
        partial(construct_symmetric_role_arms, atom_aware_topology=True),
    )
    symmetric_controls.test_registry_symmetric_internal_arms_preserve_size_and_full_replay()


@pytest.mark.needs_vendor
def test_ester_domain_projection_uses_registry_counts_and_retained_query():
    root = Path(__file__).resolve().parents[1]
    _, _, executors, _ = load_contract(
        root, root / "results/phase1/compose_lipid_fast_qualification_v1/aema-config.json"
    )
    for executor in executors.values():
        executor = {**executor, "mapping": {role: role for role in executor["program"].roles}}
        before = copy.deepcopy(executor["ester_thiol_domain"])
        output = with_retained_component_domain(executor, [])
        assert len(output) == 1
        assert output[0]["product_element_counts"] == before["constraints"]["exact_element_counts"]
        assert output[0]["required_queries"] == [
            {**query, "whole_component": True}
            for query in before["constraints"]["required_queries"]
        ]
        assert executor["ester_thiol_domain"] == before
        assert with_retained_component_domain(executor, output) == output
        changed = {**executor, "ester_thiol_domain": copy.deepcopy(before)}
        stage = executor["program"].specification["stages"][-1]
        adapter = executor["program"].adapters[-1]
        role = stage["added_roles"][0]
        reacted_query = adapter.reaction.forward.GetReactantTemplate(adapter.roles.index(role))
        from rdkit import Chem

        changed["ester_thiol_domain"]["constraints"]["required_queries"][0]["smarts"] = (
            Chem.MolToSmarts(reacted_query)
        )
        with pytest.raises(ValueError, match="disjoint"):
            with_retained_component_domain(changed, [])


def test_absent_domain_does_not_change_or_alias_existing_policies():
    policies = [{"role": "head", "product_element_counts": {"N": 1}}]
    output = with_retained_component_domain({}, policies)
    assert output == policies and output is not policies
    output[0]["product_element_counts"]["N"] = 2
    assert policies[0]["product_element_counts"]["N"] == 1
