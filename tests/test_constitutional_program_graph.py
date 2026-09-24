from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem

from forge.model.constitutional_program_graph import tensorize_constitutional_program_product
from forge.model.defog_feasibility import AtomState
from forge.model.qualified_vocabulary import QualifiedAtomVocabulary
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.sparse_topology_feasibility import (
    sparse_constitutional_roundtrip_exact,
    sparse_roundtrip_exact,
)
from forge.model.synthesis_program_graph import (
    SynthesisProgramGraphError,
    tensorize_synthesis_program_product,
)


def arguments(smiles):
    mol = Chem.MolFromSmiles(smiles)
    roles = ["precursor"] * mol.GetNumAtoms()
    positions = ["program:core", *(["exterior"] * (mol.GetNumAtoms() - 1))]
    return dict(
        record_id="encoding-control",
        program_id="program",
        canonical_product_smiles=smiles,
        atom_roles=roles,
        atom_core_positions=positions,
        program_depth=1,
        vocabulary=ReactionProgramVocabulary.from_semantics(
            program_ids=("program",),
            roles=roles,
            core_positions=("program:core",),
            maximum_steps=3,
        ),
        atom_vocabulary=QualifiedAtomVocabulary(
            tuple(
                sorted(
                    {
                        AtomState(a.GetSymbol(), a.GetFormalCharge(), False, 0)
                        for a in mol.GetAtoms()
                    },
                    key=AtomState.key,
                )
            ),
            neutral_monovalent_extensions=("Br",) if "Br" in smiles else (),
        ),
    )


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1CN(C)CC(=O)NCC",
        "c1cc[nH]c1",
        "C[n+]1ccccc1",
        "Brc1ccccc1",
        "C[N+](C)(C)CC(=O)[O-]",
        "CCOP(=O)(OCC)OCC",
        "C" * 254,
    ],
)
def test_complete_graph_survives_encoding_with_explicit_vocabulary(smiles):
    args = arguments(smiles)
    vocabulary_before = args["atom_vocabulary"]
    record = tensorize_constitutional_program_product(**args)
    assert record.node_count == Chem.MolFromSmiles(smiles).GetNumAtoms()
    assert sparse_roundtrip_exact(record.graph)
    assert sparse_constitutional_roundtrip_exact(record.graph, vocabulary_before)
    assert args["atom_vocabulary"] == vocabulary_before
    assert record.graph.node_states.max() < len(vocabulary_before)
    assert set(record.graph.parent_bonds) <= {0, 1, 2}
    assert set(record.graph.closure_bonds) <= {0, 1, 2}


def test_aromatic_layout_and_masks_keep_original_chemical_meaning():
    args = arguments("c1ccccc1CN")
    args["fixed_atom_indices"] = (0,)
    actual = tensorize_constitutional_program_product(**args)
    mol = Chem.MolFromSmiles(args["canonical_product_smiles"])
    old_args = dict(
        args,
        atom_vocabulary=tuple(
            sorted(
                {
                    AtomState(
                        a.GetSymbol(), a.GetFormalCharge(), a.GetIsAromatic(), a.GetNumExplicitHs()
                    )
                    for a in mol.GetAtoms()
                },
                key=AtomState.key,
            )
        ),
    )
    prior = tensorize_synthesis_program_product(**old_args)
    for name in (
        "canonical_atom_order",
        "role_states",
        "core_position_states",
        "fixed_atom_mask",
        "fixed_parent_bond_mask",
        "fixed_closure_bond_mask",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(prior, name))
    assert actual.component_blocks == prior.component_blocks
    # An arbitrary Kekule single bond at the core does not become a fixed attachment.
    assert not actual.fixed_parent_bond_mask.any()


def test_nonaromatic_arrays_are_unchanged_and_repeated_roles_remain_separate():
    args = arguments("CCN(CC)CC")
    args["atom_roles"] = ["head" if i == 2 else "tail" for i in range(7)]
    args["atom_core_positions"] = ["program:core" if i == 2 else "exterior" for i in range(7)]
    args["vocabulary"] = ReactionProgramVocabulary.from_semantics(
        program_ids=("program",),
        roles=("head", "tail"),
        core_positions=("program:core",),
        maximum_steps=3,
    )
    actual = tensorize_constitutional_program_product(**args)
    prior = tensorize_synthesis_program_product(**args)
    for name in (
        "node_states",
        "parents",
        "parent_bonds",
        "closure_left",
        "closure_right",
        "closure_bonds",
        "edges",
    ):
        np.testing.assert_array_equal(getattr(actual.graph, name), getattr(prior.graph, name))
    assert actual.component_blocks == prior.component_blocks
    assert sum(block.role == "tail" for block in actual.component_blocks) == 3


def test_unknown_final_state_cannot_enter_through_temporary_layout_vocabulary():
    args = arguments("Brc1ccccc1")
    args["atom_vocabulary"] = QualifiedAtomVocabulary((AtomState("C", 0, False, 0),))
    with pytest.raises(SynthesisProgramGraphError, match="outside the qualified"):
        tensorize_constitutional_program_product(**args)


@pytest.mark.parametrize("smiles", ["[13CH3]N", "[CH3:1]N", "C.N"])
def test_unqualified_identity_is_rejected(smiles):
    with pytest.raises(SynthesisProgramGraphError):
        tensorize_constitutional_program_product(**arguments(smiles))


def test_aromatic_final_vocabulary_is_rejected():
    args = arguments("c1ccccc1")
    args["atom_vocabulary"] = replace(args["atom_vocabulary"], states=(AtomState("C", 0, True, 0),))
    with pytest.raises(SynthesisProgramGraphError, match="must use Kekule"):
        tensorize_constitutional_program_product(**args)
