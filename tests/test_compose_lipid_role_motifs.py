"""Joint motif proposals conserve topology and cannot overwrite source constraints."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from forge.model.compose_lipid_role_motifs import compile_role_motif_prior, propose_role_motifs
from forge.model.precursor_reuse_projection import state_graph
from tests.test_compose_lipid_atom_aware import fixture
from tests.test_compose_lipid_ring_quality import prepared


def prior(components=("CC(=O)OCCC",)):
    registry = json.loads(
        Path("data/vendor/qualified_ester_thiol_yne_source_program_v1.json").read_text()
    )
    return compile_role_motif_prior(
        "fixture", "one", registry["ester_thiol_domain"]["ester_query"], components, {}
    )


def test_joint_atom_and_bond_edit_restores_ester_without_rewiring():
    atoms, layout, state, predictions = prepared("CC(C)OCCC")
    original = copy.deepcopy(state)
    result = propose_role_motifs(layout, state, predictions, atoms, prior())
    assert len(result["proposals"]) == 1
    proposal = result["proposals"][0]
    assert proposal["smiles"] == Chem.MolToSmiles(Chem.MolFromSmiles("CC(=O)OCCC"))
    assert proposal["atom_changes"] == proposal["bond_changes"] == 1
    assert np.array_equal(np.asarray(proposal["edges"]) > 0, state_graph(state)[1] > 0)
    assert state == original


def test_observed_control_has_no_changes_or_search():
    atoms, layout, state, predictions = prepared("CC(=O)OCCC")
    result = propose_role_motifs(layout, state, predictions, atoms, prior())
    assert result["status"] == "prior_already_satisfied"
    assert result["proposals"] == [] and result["candidates_evaluated"] == 0


def test_two_joint_motifs_are_possible_in_one_bounded_beam():
    atoms, layout, state, predictions = prepared("CC(C)OCCCC(C)OCCC")
    value = prior(("CC(=O)OCCCC(=O)OCCC",))
    result = propose_role_motifs(layout, state, predictions, atoms, value)
    assert result["candidates_evaluated"] <= 64 * 4 * 2
    assert result["proposals"]
    for candidate in result["proposals"]:
        molecule = Chem.MolFromSmiles(candidate["smiles"])
        assert len(molecule.GetSubstructMatches(Chem.MolFromSmarts(value.query))) >= 2
        assert sum(a.GetSymbol() == "O" for a in molecule.GetAtoms()) == 4


def test_fixed_atom_and_bond_masks_cannot_be_overwritten():
    atoms, layout, state, predictions = prepared("CC(C)OCCC")
    original_record, _, _, _ = fixture("CC(C)OCCC")
    layout = replace(
        layout,
        record=replace(
            layout.record,
            graph=original_record.graph,
            fixed_atom_mask=np.ones(len(state["nodes"]), dtype=bool),
            fixed_parent_bond_mask=np.array([False] + [True] * (len(state["nodes"]) - 1)),
        ),
    )
    result = propose_role_motifs(layout, state, predictions, atoms, prior())
    assert result["proposals"] == []


def test_unqualified_element_count_variation_rejected():
    with pytest.raises(ValueError, match="invariant"):
        prior(("CC(=O)OCCC", "CC(=O)OCCO"))


def test_ring_atoms_are_not_modified_by_nonring_prior():
    atoms, layout, state, predictions = prepared("CC1CCCOC1")
    result = propose_role_motifs(layout, state, predictions, atoms, prior())
    assert result["proposals"] == []


def test_nonfinite_predictions_and_invalid_budget_fail_loudly():
    atoms, layout, state, predictions = prepared("CC(C)OCCC")
    predictions["nodes"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        propose_role_motifs(layout, state, predictions, atoms, prior())
    with pytest.raises(ValueError, match="positive integers"):
        propose_role_motifs(layout, state, predictions, atoms, prior(), maximum_steps=0)


@pytest.mark.parametrize("value", [-1, 0.5, 0.0, True, 999])
def test_atom_class_bounds_are_checked_before_integer_coercion(value):
    atoms, layout, state, predictions = prepared("CC(C)OCCC")
    state["nodes"][-1] = value
    with pytest.raises(ValueError, match="integer indices within vocabulary"):
        propose_role_motifs(layout, state, predictions, atoms, prior())


def test_carbonate_double_match_is_not_two_separate_esters():
    atoms, layout, state, predictions = prepared("CCCOC(=O)OCCO")
    value = prior(("CC(=O)OCCCC(=O)OCCC",))
    result = propose_role_motifs(layout, state, predictions, atoms, value)
    assert result["initial_prior_deficit"] == 1
    assert result["status"] != "prior_already_satisfied"
