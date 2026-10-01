"""Atom-indexed structural diagnostics, independent of TRAIN-neighborhood support.

Alerts identify graph motifs for chemical review, not experimental failure or a
universal lipid-quality score. Reaction SMARTS are supplied by pinned registries.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem, rdBase

from forge.model.compose_lipid_layout import tree_ring_size


@dataclass(frozen=True)
class RegistryQuery:
    name: str
    smarts: str
    source: str

    def compile(self) -> Chem.Mol:
        query = Chem.MolFromSmarts(self.smarts)
        if query is None or query.GetNumAtoms() == 0:
            raise ValueError(f"Invalid pinned registry query: {self.name}")
        return query


def _path(adjacency, start, target, *, excluded_atom=None, excluded_edge=None):
    previous = {start: None}
    todo = deque([start])
    while todo:
        node = todo.popleft()
        if node == target:
            path = [node]
            while previous[path[-1]] is not None:
                path.append(previous[path[-1]])
            return tuple(reversed(path))
        for other in adjacency[node]:
            if other == excluded_atom or frozenset((node, other)) == excluded_edge:
                continue
            if other not in previous:
                previous[other] = node
                todo.append(other)
    return None


def _components(adjacency, members):
    remaining = set(members)
    groups = []
    while remaining:
        todo, group = [min(remaining)], set()
        while todo:
            node = todo.pop()
            if node not in remaining:
                continue
            remaining.remove(node)
            group.add(node)
            todo.extend(adjacency[node])
        groups.append(sorted(group))
    return groups


def _distances(adjacency, root):
    distances = {root: 0}
    todo = deque([root])
    while todo:
        node = todo.popleft()
        for other in adjacency[node]:
            if other not in distances:
                distances[other] = distances[node] + 1
                todo.append(other)
    return distances


def audit_layout_rings(edges, layout, *, tree_state, tree_basis):
    """Separate basis-invariant role cycle counts from tree-basis ring sizes.

    This checks the sampled coarse condition, not chemical feasibility. A dense
    constructor graph does not uniquely determine its original spanning tree; the
    caller must label any reconstruction rather than pretend it is an observed one.
    """
    edges = np.asarray(edges)
    record = layout.record
    if (
        edges.shape != (record.node_count, record.node_count)
        or not np.array_equal(edges, edges.T)
        or np.any(np.diag(edges))
    ):
        raise ValueError("Layout ring audit needs a symmetric simple graph of the declared size")
    adjacency = [np.flatnonzero(row).tolist() for row in edges]
    roles = np.asarray(record.role_states)
    core = np.asarray(record.core_position_states) > 1

    def rank(members):
        if not len(members):
            return 0
        local = edges[np.ix_(members, members)]
        return (
            int(np.count_nonzero(np.triu(local)))
            - len(members)
            + len(_components(adjacency, members))
        )

    role_names = {int(block.role_state): block.role for block in record.component_blocks}
    output = {}
    for role in sorted(set(roles.tolist())):
        members = np.flatnonzero(roles == role).tolist()
        core_members = [i for i in members if core[i]]
        observed = rank(members) - rank(core_members)
        expected = int(layout.variable_closures_by_role.get(role, 0))
        output[role] = {
            "role": role_names.get(role),
            "atoms": members,
            "observed_variable_cycle_rank": observed,
            "declared_variable_closures": expected,
            "cycle_allocation_matches": observed == expected,
            "declared_fundamental_ring_sizes": list(layout.ring_sizes_by_role.get(role, ())),
            "observed_fundamental_rings": [],
        }
    cross_origin = [
        [a, b]
        for a, b in zip(*np.nonzero(np.triu(edges)), strict=True)
        if roles[a] != roles[b] and not (core[a] and core[b])
    ]
    if tree_state is not None:
        for a, b in zip(tree_state["closure_left"], tree_state["closure_right"], strict=True):
            if core[a] and core[b]:
                continue
            if roles[a] != roles[b]:
                continue
            size = tree_ring_size(tree_state["parents"], int(a), int(b))
            local = output[int(roles[a])]
            local["observed_fundamental_rings"].append(
                {
                    "closure_atoms": [int(a), int(b)],
                    "size": size,
                    "within_declared_role_sizes": size in local["declared_fundamental_ring_sizes"],
                }
            )
    return {
        "atom_index_basis": "pinned generated graph node order",
        "tree_basis": tree_basis,
        "roles": output,
        "unexpected_cross_origin_edges": [[int(a), int(b)] for a, b in cross_origin],
        "cycle_allocation_mismatch": any(
            not v["cycle_allocation_matches"] for v in output.values()
        ),
        "fundamental_size_mismatch": (
            any(
                not r["within_declared_role_sizes"]
                for v in output.values()
                for r in v["observed_fundamental_rings"]
            )
            if tree_state is not None
            else None
        ),
        "interpretation": "Declared coarse-condition diagnostic; not a universal chemical-validity test. Fundamental ring size depends on the stated tree basis.",
    }


def audit_smiles(
    smiles: str | None,
    *,
    registry_queries: Sequence[tuple[RegistryQuery, Chem.Mol]],
    basic_query: tuple[RegistryQuery, Chem.Mol],
    policy: dict[str, Any],
    allowed_elements: set[str],
    allowed_atom_charges: set[tuple[str, int]] | None = None,
    allowed_kekule_bonds: set[str] | None = None,
    maximum_query_matches: int = 4096,
) -> dict[str, Any]:
    """Audit one unchanged output; all atom indices refer to canonical SMILES.

    The caller supplies an externally frozen rubric and registry query provenance.
    No unknown motif is repaired, filtered, or promoted to a chemistry violation.
    """
    if maximum_query_matches < 1:
        raise ValueError("maximum_query_matches must be positive")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return {
            "canonical_smiles": None,
            "atom_index_basis": "unavailable",
            "valid": False,
            "flags": [
                {
                    "code": "unparseable_or_empty",
                    "tier": "demonstrated_representation_issue",
                    "atoms": [],
                }
            ],
        }
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    mol = Chem.MolFromSmiles(canonical)
    atoms = list(mol.GetAtoms())
    adjacency = [sorted(n.GetIdx() for n in atom.GetNeighbors()) for atom in atoms]
    flags: list[dict[str, Any]] = []

    def flag(code, tier, indices, **detail):
        flags.append(dict(code=code, tier=tier, atoms=sorted(set(indices)), **detail))

    connected = len(Chem.GetMolFrags(mol)) == 1
    cycles = mol.GetNumBonds() - mol.GetNumAtoms() + len(Chem.GetMolFrags(mol))
    if not connected:
        flag("disconnected_product", "demonstrated_representation_issue", range(len(atoms)))
    if mol.GetNumHeavyAtoms() > policy["maximum_heavy_atoms"]:
        flag("outside_declared_size", "demonstrated_representation_issue", range(len(atoms)))
    if cycles > policy["maximum_independent_cycles"]:
        flag("outside_declared_cycles", "demonstrated_representation_issue", range(len(atoms)))
    for atom in atoms:
        i = atom.GetIdx()
        if atom.GetSymbol() not in allowed_elements:
            flag("outside_declared_elements", "demonstrated_representation_issue", [i])
        if (
            allowed_atom_charges is not None
            and (atom.GetSymbol(), atom.GetFormalCharge()) not in allowed_atom_charges
        ):
            flag("outside_declared_atom_charge_states", "demonstrated_representation_issue", [i])
        if atom.GetNumRadicalElectrons():
            flag(
                "radical_state",
                "demonstrated_representation_issue",
                [i],
                electrons=atom.GetNumRadicalElectrons(),
            )
        doubles = [
            b.GetOtherAtomIdx(i) for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
        ]
        if atom.GetSymbol() == "C" and len(doubles) == 2:
            neighbors = [atoms[j].GetSymbol() for j in doubles]
            if neighbors == ["C", "C"]:
                path = _path(adjacency, *doubles, excluded_atom=i)
                size = len(path) + 1 if path else None
                small = (
                    size is not None and size <= policy["small_endocyclic_allene_maximum_ring_size"]
                )
                flag(
                    "small_endocyclic_allene" if small else "other_allene",
                    "justified_chemical_alert" if small else "context_required",
                    [i, *doubles, *(path or ())],
                    central_atom=i,
                    minimum_containing_ring_size=size,
                )
            else:
                flag(
                    "heteroatom_cumulene",
                    "context_required",
                    [i, *doubles],
                    terminal_symbols=neighbors,
                )
        oxygen = [
            n.GetIdx()
            for n in atom.GetNeighbors()
            if n.GetSymbol() == "O"
            and mol.GetBondBetweenAtoms(i, n.GetIdx()).GetBondType() == Chem.BondType.SINGLE
        ]
        if (
            atom.GetSymbol() == "C"
            and not atom.GetIsAromatic()
            and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds())
        ):
            hydroxy = [j for j in oxygen if atoms[j].GetTotalNumHs() > 0]
            if len(oxygen) >= 3:
                flag(
                    "saturated_carbon_at_least_three_oxygens",
                    "context_required",
                    [i, *oxygen],
                    oxygen_neighbors=len(oxygen),
                    hydroxyl_neighbors=len(hydroxy),
                )
            elif len(oxygen) == 2 and hydroxy:
                flag(
                    "hydrated_or_hemiacetal_carbon",
                    "context_required",
                    [i, *oxygen],
                    hydroxyl_neighbors=len(hydroxy),
                )
        if atom.GetSymbol() in {"P", "S"} and atom.GetDegree() >= 5:
            flag(
                "high_coordination_phosphorus_or_sulfur",
                "context_required",
                [i, *adjacency[i]],
                element=atom.GetSymbol(),
                degree=atom.GetDegree(),
            )
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (
            bond.GetBondType() == Chem.BondType.TRIPLE
            and atoms[a].GetSymbol() == atoms[b].GetSymbol() == "C"
        ):
            path = _path(adjacency, a, b, excluded_edge=frozenset((a, b)))
            if path:
                small = len(path) <= policy["small_endocyclic_alkyne_maximum_ring_size"]
                flag(
                    "small_endocyclic_alkyne" if small else "other_cyclic_alkyne",
                    "justified_chemical_alert" if small else "context_required",
                    path,
                    minimum_containing_ring_size=len(path),
                )
        if (
            bond.GetBondType() == Chem.BondType.SINGLE
            and atoms[a].GetSymbol() == atoms[b].GetSymbol() == "O"
        ):
            flag("peroxide_bond", "context_required", [a, b])
    if allowed_kekule_bonds is not None:
        kekule = Chem.Mol(mol)
        Chem.Kekulize(kekule, clearAromaticFlags=True)
        for bond in kekule.GetBonds():
            if str(bond.GetBondType()) not in allowed_kekule_bonds:
                flag(
                    "outside_declared_kekule_bonds",
                    "demonstrated_representation_issue",
                    [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
                )
    for ring in mol.GetRingInfo().AtomRings():
        bonds = [
            mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))
        ]
        orders = [b.GetBondTypeAsDouble() for b in bonds]
        if (
            len(ring) == 4
            and all(atoms[i].GetSymbol() == "C" and atoms[i].GetFormalCharge() == 0 for i in ring)
            and orders in ([1.0, 2.0, 1.0, 2.0], [2.0, 1.0, 2.0, 1.0])
        ):
            flag("cyclobutadiene_like_ring", "justified_chemical_alert", ring)
        elif (
            len(ring) <= 4
            and any(v == 2 for v in orders)
            and not any(atoms[i].GetIsAromatic() for i in ring)
        ):
            flag("other_small_unsaturated_ring", "context_required", ring, ring_size=len(ring))

    query_results = []
    for definition, query in registry_queries:
        matches = mol.GetSubstructMatches(
            query, uniquify=True, maxMatches=maximum_query_matches + 1
        )
        query_results.append(
            {
                "name": definition.name,
                "source": definition.source,
                "matches": [list(v) for v in matches[:maximum_query_matches]],
                "censored": len(matches) > maximum_query_matches,
            }
        )
    basic_definition, basic = basic_query
    basic_matches = mol.GetSubstructMatches(
        basic, uniquify=True, maxMatches=maximum_query_matches + 1
    )
    basic_atoms = sorted({i for match in basic_matches for i in match})
    carbon_atoms = [a.GetIdx() for a in atoms if a.GetSymbol() == "C"]
    carbon_domains = _components(adjacency, carbon_atoms)
    distances = [_distances(adjacency, root) for root in basic_atoms]
    domains = []
    for members in carbon_domains:
        near = [values[a] for a in members for values in distances if a in values]
        domains.append(
            {
                "atoms": members,
                "carbon_atoms": len(members),
                "minimum_distance_to_basic_candidate": min(near) if near else None,
                "maximum_distance_to_basic_candidate": max(near) if near else None,
            }
        )
    return {
        "canonical_smiles": canonical,
        "atom_index_basis": "canonical_smiles",
        "valid": True,
        "connected": connected,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "independent_cycles": cycles,
        "rings": [
            {
                "atoms": list(ring),
                "size": len(ring),
                "symbols": [atoms[i].GetSymbol() for i in ring],
                "all_aromatic": all(atoms[i].GetIsAromatic() for i in ring),
                "bond_orders": [
                    mol.GetBondBetweenAtoms(
                        ring[i], ring[(i + 1) % len(ring)]
                    ).GetBondTypeAsDouble()
                    for i in range(len(ring))
                ],
            }
            for ring in mol.GetRingInfo().AtomRings()
        ],
        "ring_inventory_basis": "RDKit symmetrized SSSR; descriptive, not layout fundamental cycles",
        "flags": flags,
        "registry_handles": query_results,
        "organization": {
            "basic_query_source": basic_definition.source,
            "basic_candidate_atoms": basic_atoms,
            "basic_query_censored": len(basic_matches) > maximum_query_matches,
            "carbon_domains": sorted(domains, key=lambda d: (-d["carbon_atoms"], d["atoms"])),
            "carbon_branch_atoms": [
                i
                for i in carbon_atoms
                if sum(atoms[j].GetSymbol() == "C" for j in adjacency[i]) >= 3
            ],
            "carbon_terminal_atoms": [i for i in carbon_atoms if len(adjacency[i]) == 1],
            "heteroatom_indices": [a.GetIdx() for a in atoms if a.GetSymbol() not in {"C", "H"}],
        },
        "qualification": "Limited structural diagnostics; no high-quality or delivery certification",
        "representation_coverage": {
            "atom_charge_states": allowed_atom_charges is not None,
            "kekule_bonds": allowed_kekule_bonds is not None,
            "stereochemistry": False,
            "explicit_hydrogen_state_encoding": False,
        },
    }
