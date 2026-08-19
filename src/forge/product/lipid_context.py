"""Lipid-native topology context that does not depend on fragment identities."""

from __future__ import annotations

from collections import deque

import numpy as np
from rdkit import Chem

from forge.product.defog_feasibility import FeasibilityError

LIPID_REGION_NAMES = ("head", "interface", "tail")
HEAD_REGION = 0
INTERFACE_REGION = 1
TAIL_REGION = 2


def _is_carbonyl_carbon(atom: Chem.Atom) -> bool:
    if atom.GetSymbol() != "C":
        return False
    return any(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetOtherAtom(atom).GetSymbol() in {"O", "S"}
        for bond in atom.GetBonds()
    )


def _is_amide_like_nitrogen(atom: Chem.Atom) -> bool:
    if atom.GetSymbol() != "N":
        return False
    return any(_is_carbonyl_carbon(neighbor) for neighbor in atom.GetNeighbors())


def _distances_from(molecule: Chem.Mol, start: int) -> list[int]:
    distances = [-1] * molecule.GetNumAtoms()
    distances[start] = 0
    queue = deque([start])
    while queue:
        atom_index = queue.popleft()
        for neighbor in molecule.GetAtomWithIdx(atom_index).GetNeighbors():
            neighbor_index = neighbor.GetIdx()
            if distances[neighbor_index] < 0:
                distances[neighbor_index] = distances[atom_index] + 1
                queue.append(neighbor_index)
    if any(distance < 0 for distance in distances):
        raise FeasibilityError("lipid-native root selection requires one connected molecule")
    return distances


def _local_hetero_count(molecule: Chem.Mol, start: int, radius: int = 2) -> int:
    distances = _distances_from(molecule, start)
    return sum(
        distance <= radius and molecule.GetAtomWithIdx(index).GetSymbol() not in {"C", "H", "F"}
        for index, distance in enumerate(distances)
    )


def select_lipid_polar_root(molecule: Chem.Mol) -> int:
    """Select a deterministic polar head anchor without using a component catalog.

    The priority favors ionizable, non-amide nitrogens and phosphorus atoms,
    then local heteroatom context and graph centrality. Canonical ranks resolve
    symmetry and make the result independent of input atom order.
    """

    if molecule.GetNumAtoms() == 0:
        raise FeasibilityError("cannot select a root for an empty molecule")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise FeasibilityError("lipid-native root selection requires one connected molecule")
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    primary_roles: list[int] = []
    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        non_amide_nitrogen = symbol == "N" and not _is_amide_like_nitrogen(atom)
        primary_roles.append(
            5
            if non_amide_nitrogen and charge > 0
            else (
                4
                if non_amide_nitrogen
                else (
                    3 if symbol == "P" else 2 if symbol == "N" else 1 if symbol in {"O", "S"} else 0
                )
            )
        )
    best_role = max(primary_roles)
    candidates = [
        index for index, primary_role in enumerate(primary_roles) if primary_role == best_role
    ]

    def tie_break(index: int) -> tuple[int, int, int, int, int]:
        atom = molecule.GetAtomWithIdx(index)
        distances = _distances_from(molecule, atom.GetIdx())
        eccentricity = max(distances)
        return (
            _local_hetero_count(molecule, atom.GetIdx()),
            int(atom.GetFormalCharge() != 0),
            atom.GetDegree(),
            -eccentricity,
            -ranks[atom.GetIdx()],
        )

    return max(candidates, key=tie_break)


def rooted_distances(molecule: Chem.Mol, root: int) -> tuple[int, ...]:
    """Return exact graph distances from a validated root."""

    if not 0 <= root < molecule.GetNumAtoms():
        raise FeasibilityError("root index lies outside the molecule")
    return tuple(_distances_from(molecule, root))


def assign_lipid_regions(molecule: Chem.Mol, root: int) -> np.ndarray:
    """Assign generic structural regions without component or fragment identities.

    The frozen rule was calibrated against atom-mapped Ugi products. It treats
    the polar-root neighborhood and its small rings as head-like, heteroatoms
    and carbonyl centers as interfacial, and remote carbon-rich atoms as tails.
    """

    distances = rooted_distances(molecule, root)
    head_ring_atoms: set[int] = set()
    for ring in molecule.GetRingInfo().AtomRings():
        if any(distances[index] <= 3 for index in ring):
            head_ring_atoms.update(ring)
    regions = np.full(molecule.GetNumAtoms(), TAIL_REGION, dtype=np.int64)
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        hard_linker = (
            atom.GetSymbol() in {"O", "S", "P"}
            or _is_carbonyl_carbon(atom)
            or _is_amide_like_nitrogen(atom)
        )
        if (distances[index] <= 3 or index in head_ring_atoms) and not hard_linker:
            regions[index] = HEAD_REGION
        elif hard_linker or (4 <= distances[index] <= 7 and atom.GetDegree() >= 3):
            regions[index] = INTERFACE_REGION
    return regions


def tree_pair_ring_sizes(parents: np.ndarray) -> np.ndarray:
    """Return the ring size formed by adding every possible tree closure edge."""

    if parents.ndim != 1 or len(parents) == 0 or int(parents[0]) != 0:
        raise FeasibilityError("tree parents must be one-dimensional and root at zero")
    node_count = len(parents)
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for child in range(1, node_count):
        parent = int(parents[child])
        if not 0 <= parent < child:
            raise FeasibilityError("tree parent must precede its child")
        adjacency[child].append(parent)
        adjacency[parent].append(child)
    sizes = np.zeros((node_count, node_count), dtype=np.int64)
    for start in range(node_count):
        distances = [-1] * node_count
        distances[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        sizes[start] = np.asarray(distances, dtype=np.int64) + 1
    return sizes
