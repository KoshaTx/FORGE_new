from __future__ import annotations

import numpy as np
from rdkit import Chem

from forge.model.lipid_context import (
    HEAD_REGION,
    INTERFACE_REGION,
    TAIL_REGION,
    assign_lipid_regions,
    rooted_distances,
    select_lipid_polar_root,
    tree_pair_ring_sizes,
)


def test_lipid_root_prefers_non_amide_head_nitrogen() -> None:
    molecule = Chem.MolFromSmiles("CCCCNC(=O)C(CCCC)N(C)CC")
    assert molecule is not None

    root = select_lipid_polar_root(molecule)
    atom = molecule.GetAtomWithIdx(root)

    assert atom.GetSymbol() == "N"
    assert not any(
        neighbor.GetSymbol() == "C"
        and any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(neighbor).GetSymbol() == "O"
            for bond in neighbor.GetBonds()
        )
        for neighbor in atom.GetNeighbors()
    )


def test_lipid_root_is_invariant_to_atom_permutation() -> None:
    molecule = Chem.MolFromSmiles("CCCCNC(=O)C(CCCC)N(C)CC")
    assert molecule is not None
    original_root = select_lipid_polar_root(molecule)
    permutation = np.random.default_rng(19).permutation(molecule.GetNumAtoms()).tolist()
    permuted = Chem.RenumberAtoms(molecule, permutation)
    permuted_root = select_lipid_polar_root(permuted)

    assert permutation[permuted_root] == original_root
    assert rooted_distances(molecule, original_root)[original_root] == 0


def test_tree_pair_ring_sizes_are_path_distance_plus_one() -> None:
    parents = np.asarray([0, 0, 1, 2, 1], dtype=np.int64)
    sizes = tree_pair_ring_sizes(parents)

    assert sizes[0, 2] == 3
    assert sizes[0, 3] == 4
    assert sizes[3, 4] == 4
    assert np.array_equal(sizes, sizes.T)


def test_structural_lipid_regions_keep_head_ring_and_distal_tail_distinct() -> None:
    molecule = Chem.MolFromSmiles("CCCCCCCCCC(=O)OCCCCCC(N)CN1CCCCC1")
    assert molecule is not None
    root = select_lipid_polar_root(molecule)
    regions = assign_lipid_regions(molecule, root)

    assert regions[root] == HEAD_REGION
    assert all(
        regions[index] == HEAD_REGION
        for ring in molecule.GetRingInfo().AtomRings()
        for index in ring
    )
    assert regions[0] == TAIL_REGION
    assert any(region == INTERFACE_REGION for region in regions)
