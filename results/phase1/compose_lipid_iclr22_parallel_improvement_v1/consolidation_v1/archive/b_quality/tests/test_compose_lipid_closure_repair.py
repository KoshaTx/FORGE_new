"""Closure repair protects topology, source core and full candidate denominators."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from forge.model.compose_lipid_closure_repair import (
    propose_closure_endpoints,
    validate_unchanged_tree,
)
from forge.model.defog_feasibility import AtomState


@pytest.fixture
def ring():
    state = dict(
        nodes=[0] * 6,
        parents=[0, 0, 1, 2, 3, 4],
        parent_bonds=[0] * 6,
        closure_left=[0],
        closure_right=[5],
        closure_bonds=[0],
    )
    block = SimpleNamespace(start=0, stop=6, role="test_component", role_state=1)
    graph = SimpleNamespace(
        node_states=np.zeros(6, dtype=int),
        parents=np.array(state["parents"]),
        parent_bond_states=np.zeros(6, dtype=int),
        closure_left=np.array([0]),
        closure_right=np.array([5]),
        closure_bond_states=np.array([0]),
    )
    record = SimpleNamespace(
        node_count=6,
        graph=graph,
        fixed_atom_mask=np.zeros(6, dtype=bool),
        fixed_parent_bond_mask=np.zeros(6, dtype=bool),
        fixed_closure_bond_mask=np.zeros(1, dtype=bool),
        component_blocks=[block],
        core_position_states=np.ones(6, dtype=int),
        role_states=np.ones(6, dtype=int),
    )
    layout = SimpleNamespace(
        record=record,
        core_units=np.full(6, -1),
        ring_sizes_by_role={1: (3,)},
        variable_closures_by_role={1: 1},
        quantities={"test_component": 1},
    )
    return (
        layout,
        state,
        [AtomState("C", 0, False)],
        dict(closure_left=np.zeros((1, 6)), closure_right=np.zeros((1, 6))),
    )


def test_repairs_ring_without_changing_tree_or_composition(ring):
    layout, state, atoms, logits = ring
    result = propose_closure_endpoints(layout, state, logits, atoms)
    assert result["status"] == "proposed" and 1 <= len(result["proposals"]) <= 2
    for row in result["proposals"]:
        assert not row["ring"]["fundamental_size_mismatch"]
        assert row["route_verdict"] is row["model_likelihood"] is None
        for key in ("nodes", "parents", "parent_bonds", "closure_bonds"):
            assert row["tree_state"][key] == state[key]


def test_requested_ring_is_exact_noop_without_logits(ring):
    layout, state, atoms, _ = ring
    layout.ring_sizes_by_role = {1: (6,)}
    result = propose_closure_endpoints(layout, state, {}, atoms)
    assert result["status"] == "already_matches_request_noop"
    assert result["proposals"] == [] and result["costs"]["states_checked"] == 0


def test_duplicate_tree_edge_rejected(ring):
    layout, state, atoms, _ = ring
    trial = copy.deepcopy(state)
    trial["closure_right"] = [1]
    with pytest.raises(ValueError, match="invalid generated closure"):
        validate_unchanged_tree(layout, state, trial, atoms)


def test_cross_component_closure_rejected(ring):
    layout, state, atoms, _ = ring
    layout.record.component_blocks = [
        SimpleNamespace(start=0, stop=3),
        SimpleNamespace(start=3, stop=6),
    ]
    trial = copy.deepcopy(state)
    trial["closure_right"] = [4]
    with pytest.raises(ValueError, match="Cross-origin"):
        validate_unchanged_tree(layout, state, trial, atoms)


def test_atom_or_tree_edits_rejected(ring):
    layout, state, atoms, _ = ring
    for field in ("nodes", "parents", "parent_bonds", "closure_bonds"):
        trial = copy.deepcopy(state)
        trial[field][-1] += 1
        with pytest.raises(ValueError, match="changed " + field):
            validate_unchanged_tree(layout, state, trial, atoms)


def test_nonfinite_logits_rejected(ring):
    layout, state, atoms, logits = ring
    logits["closure_left"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        propose_closure_endpoints(layout, state, logits, atoms)


def test_deterministic_complete_ledger(ring):
    layout, state, atoms, logits = ring
    assert propose_closure_endpoints(layout, state, logits, atoms) == propose_closure_endpoints(
        layout, state, logits, atoms
    )
