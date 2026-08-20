from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from forge.model.defog_feasibility import FeasibilityError
from forge.model.lipid_support_skeleton import (
    CARBON_INDUCED,
    FULL_HEAVY,
    FUNCTIONAL_SUPPORT,
    encode_lipid_support_skeleton,
    selected_component_count,
    skeleton_roundtrip_exact,
    skeleton_signature,
    skeleton_statistics,
)


@pytest.mark.parametrize("variant", [FULL_HEAVY, CARBON_INDUCED, FUNCTIONAL_SUPPORT])
def test_skeleton_roundtrip_is_exact(variant: str) -> None:
    molecule = Chem.MolFromSmiles("CCCCCCCC(=O)OCCN(C)CCOC(=O)CCCCCC")
    encoding = encode_lipid_support_skeleton(
        molecule,
        structure_id="ester-lipid",
        variant=variant,
    )
    assert skeleton_roundtrip_exact(encoding)


def test_functional_support_prunes_carbonyl_oxygen_but_retains_ester_bridge() -> None:
    molecule = Chem.MolFromSmiles("CCCC(=O)OCCCC")
    encoding = encode_lipid_support_skeleton(
        molecule,
        structure_id="ester",
        variant=FUNCTIONAL_SUPPORT,
    )
    retained_symbols = [encoding.atom_states[index].symbol for index in encoding.retained_atoms]
    removed_symbols = [encoding.atom_states[index].symbol for index in encoding.removed_atoms]
    assert retained_symbols.count("O") == 1
    assert removed_symbols == ["O"]
    stats = skeleton_statistics(encoding)
    assert stats["connected"]
    assert stats["resolved_false_junctions"] == 1
    assert stats["removed_apparent_leaves"] == 1


def test_functional_support_keeps_terminal_amine_and_ring_heteroatoms() -> None:
    molecule = Chem.MolFromSmiles("NCCN1CCOCC1")
    encoding = encode_lipid_support_skeleton(
        molecule,
        structure_id="head",
        variant=FUNCTIONAL_SUPPORT,
    )
    retained_symbols = [encoding.atom_states[index].symbol for index in encoding.retained_atoms]
    assert retained_symbols.count("N") == 2
    assert retained_symbols.count("O") == 1
    assert not encoding.removed_atoms


def test_carbon_induced_exposes_heteroatom_disconnection() -> None:
    molecule = Chem.MolFromSmiles("CCCCOCCCC")
    encoding = encode_lipid_support_skeleton(
        molecule,
        structure_id="ether",
        variant=CARBON_INDUCED,
    )
    assert selected_component_count(encoding) == 2
    assert skeleton_statistics(encoding)["connected"] is False


def test_reaction_core_protection_overrides_terminal_pruning() -> None:
    molecule = Chem.MolFromSmiles("CC(=O)NCC")
    normalized = Chem.MolFromSmiles(Chem.MolToSmiles(molecule, canonical=True))
    core_oxygen = next(
        atom.GetIdx()
        for atom in normalized.GetAtoms()
        if atom.GetSymbol() == "O" and atom.GetDegree() == 1
    )
    encoding = encode_lipid_support_skeleton(
        molecule,
        structure_id="protected",
        variant=FUNCTIONAL_SUPPORT,
        protected_atoms={core_oxygen},
    )
    assert core_oxygen in encoding.retained_atoms


@pytest.mark.parametrize("variant", [FULL_HEAVY, CARBON_INDUCED, FUNCTIONAL_SUPPORT])
def test_signature_is_stable_to_input_atom_permutation(variant: str) -> None:
    molecule = Chem.MolFromSmiles("CCCN1CCN(CC1)CCOC(=O)CCCCCC")
    order = np.random.default_rng(7).permutation(molecule.GetNumAtoms()).tolist()
    permuted = Chem.RenumberAtoms(molecule, order)
    first = encode_lipid_support_skeleton(
        molecule,
        structure_id="original",
        variant=variant,
    )
    second = encode_lipid_support_skeleton(
        permuted,
        structure_id="permuted",
        variant=variant,
    )
    assert skeleton_signature(first) == skeleton_signature(second)


def test_invalid_variant_is_rejected() -> None:
    molecule = Chem.MolFromSmiles("CCCC")
    with pytest.raises(FeasibilityError, match="unsupported"):
        encode_lipid_support_skeleton(
            molecule,
            structure_id="bad",
            variant="invented",
        )
