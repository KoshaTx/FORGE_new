"""Ring priors preserve full graphs and distinguish aromatic and aliphatic support."""

import copy
import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.model.compose_lipid_ring_quality import propose_ring_bonds, ring_patterns
from forge.model.precursor_reuse_projection import graph_smiles, state_graph
from forge.model.qualified_vocabulary import QualifiedAtomVocabulary
from tests.test_compose_lipid_atom_aware import fixture


def prepared(smiles):
    record, atoms, layout, predictions = fixture(smiles)
    graph = record.graph
    state = {
        key: getattr(graph, "node_states" if key == "nodes" else key).tolist()
        for key in (
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        )
    }
    return atoms, layout, state, {key: value[0].numpy() for key, value in predictions.items()}


def test_joint_bonds_recover_aromatic_ring_without_replacing_component():
    atoms, layout, state, predictions = prepared("CCC1=CCCC=C1")
    original = copy.deepcopy(state)
    result = propose_ring_bonds(
        layout, state, predictions, atoms, {"one": ring_patterns(["CCc1ccccc1"])}
    )
    assert result["changed_rings"] == 1
    molecule = Chem.MolFromSmiles(result["smiles"])
    assert rdMolDescriptors.CalcNumAromaticRings(molecule) == 1
    assert molecule.GetNumAtoms() == len(state["nodes"])
    for key in ("nodes", "parents", "closure_left", "closure_right"):
        assert result["state"][key] == state[key]
    assert state == original


@pytest.mark.parametrize("smiles", ["CCc1ncccc1", "CCc1cc[nH]c1", "CCC1CCCCC1", "CCC1=CCCCC1"])
def test_source_pattern_positive_aromatic_heterocycles_and_aliphatic_rings_invariant(smiles):
    atoms, layout, state, predictions = prepared(smiles)
    result = propose_ring_bonds(layout, state, predictions, atoms, {"one": ring_patterns([smiles])})
    assert result["changed_rings"] == 0
    assert result["state"] == state
    assert result["smiles"] == graph_smiles(*state_graph(state), atoms)


@pytest.mark.parametrize("smiles", ["CCC1CCC1", "O=C1CCCCC1", "CCC12CC3CC(CC(C3)C1)C2"])
def test_unmatched_valence_incompatible_and_polycyclic_rings_are_retained(smiles):
    atoms, layout, state, predictions = prepared(smiles)
    result = propose_ring_bonds(
        layout, state, predictions, atoms, {"one": ring_patterns(["CCc1ccccc1"])}
    )
    assert result["changed_rings"] == 0
    assert result["state"] == state
    assert result["smiles"] is not None


def test_full_254_atoms_12_closures_and_bromine_remain_supported(monkeypatch):
    monkeypatch.setattr(
        "tests.test_source_instance_coordinates.QualifiedAtomVocabulary",
        partial(QualifiedAtomVocabulary, neutral_monovalent_extensions=("Br",)),
    )
    smiles = "C" * 181 + "C1CCCCC1" * 12 + "Br"
    atoms, layout, state, predictions = prepared(smiles)
    assert len(state["nodes"]) == 254 and len(state["closure_left"]) == 12
    result = propose_ring_bonds(
        layout, state, predictions, atoms, {"one": ring_patterns(["CC1CCCCC1"])}
    )
    assert result["state"] == state
    assert Chem.MolFromSmiles(result["smiles"]).GetNumHeavyAtoms() == 254
    assert "Br" in result["smiles"]


def test_nonfinite_predictions_fail_before_proposal():
    atoms, layout, state, predictions = prepared("CCC1CCCCC1")
    predictions["parent_bonds"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        propose_ring_bonds(layout, state, predictions, atoms, {})


def test_actual_train_role_patterns_preserve_aromatic_vs_terpenoid_difference():
    path = Path("results/phase1/compose_lipid_quality_v1/reference.json")
    if not path.exists():
        pytest.skip("Pinned TRAIN reference artifact absent")
    reference = json.loads(path.read_text())
    patterns = {}
    for family, role in (
        ("a3_amine_aldehyde_alkyne", "aldehyde"),
        ("ketone_ugi4", "coupled_ketone"),
    ):
        smiles = [
            reference["component_smiles_by_identity"][key]
            for key in reference["component_ids_by_family_role"][family][role]
        ]
        patterns[family] = ring_patterns(smiles)
    assert patterns["a3_amine_aldehyde_alkyne"]
    assert all(
        sum(bond == 1 for bond in p.bonds) == 3 for p in patterns["a3_amine_aldehyde_alkyne"]
    )
    assert {sum(bond == 1 for bond in p.bonds) for p in patterns["ketone_ugi4"]} == {0, 1}
