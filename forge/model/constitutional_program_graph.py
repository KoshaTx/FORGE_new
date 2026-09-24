"""Constitutional Kekule tensors with the shared synthesis-program layout and masks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from rdkit import Chem

from forge.model.defog_feasibility import AtomState
from forge.model.qualified_vocabulary import QualifiedAtomVocabulary
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.sparse_topology_feasibility import (
    SPARSE_BOND_TO_INDEX,
    _kekulized_molecule,
    sparse_constitutional_roundtrip_exact,
    sparse_roundtrip_exact,
)
from forge.model.synthesis_program_graph import (
    SynthesisProgramGraphError,
    SynthesisProgramGraphRecord,
    tensorize_synthesis_program_product,
)


def tensorize_constitutional_program_product(
    *,
    record_id: str,
    program_id: str,
    canonical_product_smiles: str,
    atom_roles: Sequence[str],
    atom_core_positions: Sequence[str],
    program_depth: int,
    vocabulary: ReactionProgramVocabulary,
    atom_vocabulary: QualifiedAtomVocabulary,
    fixed_atom_indices: Sequence[int] = (),
) -> SynthesisProgramGraphRecord:
    """Keep qualified semantic coordinates while using the three Kekule bond classes.

    The shared serializer defines component order and fixed masks on the original
    chemical graph. Its temporary aromatic indices are discarded: all returned atom
    indices belong to the caller's explicit constitutional vocabulary. This function
    neither qualifies the supplied atom origins nor fits a training vocabulary.
    """
    if any(state.aromatic or state.explicit_hydrogens for state in atom_vocabulary):
        raise SynthesisProgramGraphError("Constitutional program vocabulary must use Kekule states")
    molecule = Chem.MolFromSmiles(canonical_product_smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise SynthesisProgramGraphError(f"Invalid connected program product: {record_id}")
    if any(atom.GetIsotope() or atom.GetAtomMapNum() for atom in molecule.GetAtoms()):
        raise SynthesisProgramGraphError("Program products must be isotope-free and unmapped")
    kekule = _kekulized_molecule(canonical_product_smiles)
    atom_to_index = {state: index for index, state in enumerate(atom_vocabulary)}
    try:
        canonical_states = np.asarray(
            [
                atom_to_index[AtomState(atom.GetSymbol(), atom.GetFormalCharge(), False, 0)]
                for atom in kekule.GetAtoms()
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise SynthesisProgramGraphError(
            f"{record_id} has a state outside the qualified constitutional vocabulary"
        ) from error
    # Use the established layout/mask contract without changing its frozen aromatic
    # implementation. Unsupported final atom states have already failed above.
    layout_vocabulary = tuple(
        dict.fromkeys(
            (
                *atom_vocabulary,
                *(
                    AtomState(
                        atom.GetSymbol(),
                        atom.GetFormalCharge(),
                        atom.GetIsAromatic(),
                        atom.GetNumExplicitHs(),
                    )
                    for atom in molecule.GetAtoms()
                ),
            )
        )
    )
    record = tensorize_synthesis_program_product(
        record_id=record_id,
        program_id=program_id,
        canonical_product_smiles=canonical_product_smiles,
        atom_roles=atom_roles,
        atom_core_positions=atom_core_positions,
        program_depth=program_depth,
        vocabulary=vocabulary,
        atom_vocabulary=layout_vocabulary,
        fixed_atom_indices=fixed_atom_indices,
    )
    order = record.canonical_atom_order
    parent_bonds = np.zeros(record.node_count, dtype=np.int64)
    for child in range(1, record.node_count):
        bond = kekule.GetBondBetweenAtoms(
            int(order[child]), int(order[record.graph.parents[child]])
        )
        parent_bonds[child] = SPARSE_BOND_TO_INDEX[bond.GetBondType()]
    closure_bonds = np.asarray(
        [
            SPARSE_BOND_TO_INDEX[
                kekule.GetBondBetweenAtoms(int(order[left]), int(order[right])).GetBondType()
            ]
            for left, right in zip(
                record.graph.closure_left, record.graph.closure_right, strict=True
            )
        ],
        dtype=np.int64,
    )
    if (
        np.any(parent_bonds < 0)
        or np.any(parent_bonds >= 3)
        or np.any(closure_bonds < 0)
        or np.any(closure_bonds >= 3)
    ):
        raise SynthesisProgramGraphError("Kekule graph contains an unsupported bond token")
    dense = np.asarray(Chem.GetAdjacencyMatrix(kekule, useBO=True))[np.ix_(order, order)]
    if np.any(dense != np.floor(dense)):
        raise SynthesisProgramGraphError("Kekule graph contains a fractional bond order")
    graph = replace(
        record.graph,
        node_states=canonical_states[order],
        parent_bonds=parent_bonds,
        closure_bonds=closure_bonds,
        edges=dense.astype(np.int64),
    )
    used = 2 * graph.edges.sum(axis=1)
    capacities = np.asarray(atom_vocabulary.capacities(), dtype=np.int64)[graph.node_states]
    if np.any(used > capacities):
        raise SynthesisProgramGraphError(f"Constitutional valence capacity exceeded: {record_id}")
    if not sparse_roundtrip_exact(graph) or not sparse_constitutional_roundtrip_exact(
        graph, atom_vocabulary
    ):
        raise SynthesisProgramGraphError(f"Constitutional semantic round trip failed: {record_id}")
    return replace(record, graph=graph)
