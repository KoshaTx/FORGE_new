"""Atom-aware construction preserves learned elements and complete size support."""

import copy
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem

from forge.corpus.qualified_program_cache import QualifiedProgramExample
from forge.model.compose_lipid_atom_aware import topology_atom_states
from forge.model.compose_lipid_generation import constrained_readout
from forge.model.compose_lipid_layout import (
    build_layout,
    collate_generated_layouts,
    summarize_layout,
)
from forge.model.precursor_reuse_projection import graph_smiles, state_graph
from forge.model.synthesis_program_sampling import _atom_capacity_table
from tests.test_compose_lipid_generation import gold_predictions
from tests.test_source_instance_coordinates import record


def fixture(smiles):
    count = Chem.MolFromSmiles(smiles).GetNumAtoms()
    r, _, atoms = record(smiles, ["one"] * count, ["core"] + ["exterior"] * (count - 1))
    e = QualifiedProgramExample(r, "fixture", (("one", "anonymous", 1),), ())
    bundle, choices, rings = summarize_layout(e)
    layout = build_layout(bundle, choices, rings, identity="atom-aware-test")
    return r, atoms, layout, gold_predictions(r, atoms)


def output_smiles(state, count, closures, atoms):
    values = {
        k: v[0, : closures if k.startswith("closure") else count].tolist() for k, v in state.items()
    }
    return graph_smiles(*state_graph(values), atoms)


def test_overconnected_carbon_is_rewired_without_forcing_negligible_sulfur():
    r, atoms, layout, pred = fixture("CSCCCCC")
    carbon = next(i for i, atom in enumerate(atoms) if atom.symbol == "C")
    sulfur = next(i for i, atom in enumerate(atoms) if atom.symbol == "S")
    pred["nodes"].fill_(-12)
    pred["nodes"][:, :, carbon] = 0
    pred["parents"][0, 2:, 1] = 1000
    batch = collate_generated_layouts([layout], maximum_closures=12)
    before = copy.deepcopy(pred)
    old, reasons = constrained_readout(pred, batch, [layout], atoms)
    assert reasons == (None,)
    assert int(old["nodes"][0, 1]) == sulfur
    assert sum(int(x) == 1 for x in old["parents"][0, 2:]) == 5
    new, reasons = constrained_readout(pred, batch, [layout], atoms, atom_aware_topology=True)
    assert reasons == (None,)
    assert set(new["nodes"][0].tolist()) == {carbon}
    assert sum(int(x) == 1 for x in new["parents"][0, 2:]) <= 3
    assert Chem.MolFromSmiles(output_smiles(new, r.node_count, 0, atoms)).GetNumAtoms() == 7
    assert all(np.array_equal(pred[k].numpy(), before[k].numpy()) for k in pred)


@pytest.mark.parametrize("smiles", ["CSSC", "CS(=O)(=O)C", "COP(=O)(O)OC", "CC1CCCC1"])
def test_gold_sulfur_phosphorus_and_rings_are_not_forbidden(smiles):
    r, atoms, layout, pred = fixture(smiles)
    batch = collate_generated_layouts([layout], maximum_closures=12)
    state, reasons = constrained_readout(
        pred, batch, [layout], atoms, rings=True, atom_aware_topology=True
    )
    assert reasons == (None,)
    assert output_smiles(state, r.node_count, r.graph.closure_count, atoms) == Chem.MolToSmiles(
        Chem.MolFromSmiles(smiles)
    )


def test_full_254_atom_and_12_closure_support_is_preserved():
    smiles = "C" * 182 + "C1CCCCC1" * 12
    r, atoms, layout, pred = fixture(smiles)
    assert r.node_count == 254 and r.graph.closure_count == 12
    state, reasons = constrained_readout(
        pred,
        collate_generated_layouts([layout], maximum_closures=12),
        [layout],
        atoms,
        rings=True,
        atom_aware_topology=True,
    )
    assert reasons == (None,)
    assert output_smiles(state, 254, 12, atoms) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_atom_first_abstains_without_substitution_when_topology_is_impossible():
    r, atoms, layout, pred = fixture("COCC")
    oxygen = next(i for i, atom in enumerate(atoms) if atom.symbol == "O")
    pred["nodes"].fill_(0)
    pred["nodes"][:, :, oxygen] = 1000
    # Extra closure requires more capacity than the chosen all-oxygen exterior supplies.
    _, _, cyclic_layout, _ = fixture("CC1CC1")
    state, reasons = constrained_readout(
        pred,
        collate_generated_layouts([cyclic_layout], maximum_closures=12),
        [cyclic_layout],
        atoms,
        atom_aware_topology=True,
    )
    assert reasons[0] is not None


def test_unavoidable_fixed_bonds_are_reserved_and_bad_logits_rejected():
    r, atoms, _, pred = fixture("CS(=O)(=O)C")
    # Make all source edges immutable, while atom identities remain model decisions.
    mask = np.ones(r.node_count, dtype=bool)
    mask[0] = False
    fixed_atoms = np.ones(r.node_count, dtype=bool)
    fixed_atoms[
        [i for i, state in enumerate(r.graph.node_states) if atoms[state].symbol == "S"]
    ] = False
    reserved = replace(r, fixed_parent_bond_mask=mask, fixed_atom_mask=fixed_atoms)
    states, reason = topology_atom_states(
        pred["nodes"][0].numpy(), reserved, _atom_capacity_table(atoms), (2, 4, 6)
    )
    assert reason is None and np.array_equal(states, r.graph.node_states)
    broken = pred["nodes"][0].numpy().copy()
    broken[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        topology_atom_states(broken, reserved, _atom_capacity_table(atoms), (2, 4, 6))
