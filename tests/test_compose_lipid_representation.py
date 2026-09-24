import numpy as np
import pytest

from forge.corpus.compose_lipid_representation import capacity_covered, kekule_states
from forge.model.defog_feasibility import AtomState
from forge.model.qualified_vocabulary import QualifiedAtomVocabulary
from forge.model.sparse_topology_feasibility import (
    sparse_constitutional_roundtrip_exact,
    tensorize_sparse_row,
)


@pytest.mark.parametrize(
    "smiles",
    ["c1cc[nH]c1", "Cn1ccnc1", "c1ccc2ccccc2c1", "[nH+]1ccccc1", "CCBr", "CCN(C)CC"],
)
def test_kekule_roundtrip_preserves_aromatic_chemistry_without_larger_capacities(smiles):
    states = tuple(sorted(set(kekule_states(smiles)), key=AtomState.key))
    vocabulary = QualifiedAtomVocabulary(states, ("Br",) if "Br" in smiles else ())
    record = tensorize_sparse_row(
        {"r0_structure_id": "test", "canonical_isomeric_smiles": smiles},
        {s: i for i, s in enumerate(vocabulary)},
        preserve_aromaticity=False,
    )
    assert sparse_constitutional_roundtrip_exact(record, vocabulary)
    assert capacity_covered(record, vocabulary)
    assert all(
        0 <= bond < 3 for bond in np.concatenate((record.parent_bonds, record.closure_bonds))
    )
    assert all(not state.aromatic for state in vocabulary)


def test_aromatic_hydrogen_accounting_problem_is_visible_under_original_encoding():
    # Aromatic bond tokens consume three half-bond units each; explicit N-H consumes two
    # more. Kekule single/double bonds remove that representation mismatch, not a chemical gate.
    states = (AtomState("C", 0, True), AtomState("N", 0, True, 1))
    graph = tensorize_sparse_row(
        {"r0_structure_id": "test", "canonical_isomeric_smiles": "c1cc[nH]c1"},
        {s: i for i, s in enumerate(states)},
        preserve_aromaticity=True,
    )
    assert not capacity_covered(graph, QualifiedAtomVocabulary(states))
