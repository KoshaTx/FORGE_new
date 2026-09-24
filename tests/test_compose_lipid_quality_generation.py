"""Atom-aware constructive readouts retain registry controls and graph conservation."""

from functools import partial

import numpy as np
import pytest

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
