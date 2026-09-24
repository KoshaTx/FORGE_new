"""Opt-in joint bond proposals from isolated TRAIN ring motifs.

This is an explicit motif prior, not a validity filter: unfamiliar or polycyclic
rings remain unchanged and unqualified. Atom identities, every edge endpoint and
all non-ring bonds remain generated. Complete source assessment is still required.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, rdBase

from forge.model.defog_feasibility import BOND_TYPE_TO_INDEX, graph_to_molecule
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph
from forge.model.sparse_topology_feasibility import BOND_VALENCE_UNITS, INDEX_TO_DENSE_BOND
from forge.model.synthesis_program_sampling import _atom_capacity_table

DENSE_TO_SPARSE = {dense: sparse for sparse, dense in INDEX_TO_DENSE_BOND.items()}


@dataclass(frozen=True, order=True)
class RingPattern:
    atoms: tuple[tuple[str, int], ...]
    bonds: tuple[int, ...]


def isolated_cycles(molecule: Chem.Mol) -> tuple[tuple[int, ...], ...]:
    """Return ordered, atom-disjoint simple rings; fused and spiro systems abstain."""
    rings = tuple(frozenset(ring) for ring in molecule.GetRingInfo().AtomRings())
    output = []
    for index, atoms in enumerate(rings):
        if any(atoms & other for j, other in enumerate(rings) if index != j):
            continue
        neighbors = {
            atom: sorted(
                n.GetIdx()
                for n in molecule.GetAtomWithIdx(atom).GetNeighbors()
                if n.GetIdx() in atoms
            )
            for atom in atoms
        }
        if any(len(value) != 2 for value in neighbors.values()):
            continue
        path = [min(atoms)]
        while len(path) < len(atoms):
            choices = [atom for atom in neighbors[path[-1]] if atom not in path]
            if not choices:
                break
            path.append(min(choices))
        if len(path) == len(atoms) and path[0] in neighbors[path[-1]]:
            output.append(tuple(path))
    return tuple(output)


def ring_patterns(smiles: Sequence[str]) -> tuple[RingPattern, ...]:
    """Extract unweighted isolated ring patterns from caller-pinned TRAIN components."""
    patterns = set()
    for value in sorted(set(smiles)):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError("Ring reference requires valid connected source components")
        cycles = isolated_cycles(molecule)
        Chem.Kekulize(molecule, clearAromaticFlags=True)
        for cycle in cycles:
            # Both directions and all starts are needed: serialization is not ring chemistry.
            for direction in (cycle, tuple(reversed(cycle))):
                for start in range(len(cycle)):
                    ordered = direction[start:] + direction[:start]
                    patterns.add(
                        RingPattern(
                            atoms=tuple(
                                (
                                    molecule.GetAtomWithIdx(i).GetSymbol(),
                                    molecule.GetAtomWithIdx(i).GetFormalCharge(),
                                )
                                for i in ordered
                            ),
                            bonds=tuple(
                                DENSE_TO_SPARSE[
                                    BOND_TYPE_TO_INDEX[
                                        molecule.GetBondBetweenAtoms(a, b).GetBondType()
                                    ]
                                ]
                                for a, b in zip(ordered, ordered[1:] + ordered[:1], strict=True)
                            ),
                        )
                    )
    return tuple(sorted(patterns))


def propose_ring_bonds(
    layout,
    state: Mapping[str, Sequence[int]],
    predictions: Mapping,
    atoms: Sequence,
    patterns_by_role: Mapping[str, Sequence[RingPattern]],
) -> dict:
    """Jointly choose a feasible observed bond pattern on each unfamiliar simple ring.

    Rings already matching an admitted pattern are invariant. The proposal changes
    no element, atom count, topology, fixed state or origin. Disjoint rings have
    independent bond budgets, so their maximum-logit assignments do not need a beam.
    """
    local = {key: [int(value) for value in values] for key, values in state.items()}
    nodes, edges = state_graph(local)
    if len(nodes) != layout.record.node_count or not fixed_graph_preserved(
        nodes, edges, layout.record
    ):
        raise ValueError("Ring proposal input changed atom budget or fixed source graph")
    logits = {key: np.asarray(predictions[key]) for key in ("parent_bonds", "closure_bonds")}
    if any(not np.isfinite(value).all() for value in logits.values()):
        raise ValueError("Ring proposal requires finite bond predictions")
    for key, count in (
        ("parent_bonds", len(nodes)),
        ("closure_bonds", len(local["closure_bonds"])),
    ):
        if (
            logits[key].ndim != 2
            or logits[key].shape[0] < count
            or not 1 <= logits[key].shape[1] <= len(BOND_VALENCE_UNITS)
        ):
            raise ValueError("Ring proposal bond prediction shape differs from graph")
    with rdBase.BlockLogs():
        molecule = graph_to_molecule(nodes, edges, atoms)
    edge_slots = {}
    for child in range(1, len(nodes)):
        edge_slots[tuple(sorted((child, local["parents"][child])))] = (
            "parent_bonds",
            child,
            bool(layout.record.fixed_parent_bond_mask[child]),
        )
    for slot, (a, b) in enumerate(zip(local["closure_left"], local["closure_right"], strict=True)):
        edge_slots[tuple(sorted((a, b)))] = (
            "closure_bonds",
            slot,
            bool(layout.record.fixed_closure_bond_mask[slot]),
        )
    blocks = {
        node: index
        for index, block in enumerate(layout.record.component_blocks)
        for node in range(block.start, block.stop)
    }
    capacities = _atom_capacity_table(atoms)[nodes]
    decisions = []
    for cycle in isolated_cycles(molecule):
        decision = dict(nodes=list(cycle), status="retained_unknown")
        decisions.append(decision)
        if len({blocks[i] for i in cycle}) != 1 or any(
            layout.record.core_position_states[i] > 1 for i in cycle
        ):
            decision["status"] = "fixed_core_or_cross_origin_unchanged"
            continue
        role = layout.record.component_blocks[blocks[cycle[0]]].role
        decision["role"] = role
        pairs = [tuple(sorted((a, b))) for a, b in zip(cycle, cycle[1:] + cycle[:1], strict=True)]
        if any(edge_slots[pair][2] for pair in pairs):
            decision["status"] = "fixed_bond_unchanged"
            continue
        labels = tuple(
            (atoms[int(nodes[i])].symbol, atoms[int(nodes[i])].formal_charge) for i in cycle
        )
        patterns = sorted(
            {pattern.bonds for pattern in patterns_by_role.get(role, ()) if pattern.atoms == labels}
        )
        current = tuple(DENSE_TO_SPARSE[int(edges[a, b])] for a, b in pairs)
        decision["compatible_patterns"] = len(patterns)
        if current in patterns:
            decision["status"] = "observed_ring_unchanged"
            continue
        best = None
        for bonds in patterns:
            if any(
                bond >= logits[edge_slots[pair][0]].shape[1]
                for pair, bond in zip(pairs, bonds, strict=True)
            ):
                continue
            trial = edges.copy()
            score = 0.0
            for pair, bond in zip(pairs, bonds, strict=True):
                a, b = pair
                trial[a, b] = trial[b, a] = INDEX_TO_DENSE_BOND[bond]
                field, slot, _ = edge_slots[pair]
                score += float(logits[field][slot, bond])
            units = np.asarray([0, *BOND_VALENCE_UNITS.tolist()])[trial].sum(1)
            if np.any(units > capacities):
                continue
            with rdBase.BlockLogs():
                smiles = graph_smiles(nodes, trial, atoms)
            if smiles is None:
                continue
            candidate = (score, bonds)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            continue
        score, bonds = best
        old_score = sum(
            float(logits[edge_slots[pair][0]][edge_slots[pair][1], bond])
            for pair, bond in zip(pairs, current, strict=True)
        )
        for pair, bond in zip(pairs, bonds, strict=True):
            field, slot, _ = edge_slots[pair]
            local[field][slot] = bond
            a, b = pair
            edges[a, b] = edges[b, a] = INDEX_TO_DENSE_BOND[bond]
        decision.update(
            status="TRAIN_ring_motif_proposal",
            before=list(current),
            after=list(bonds),
            bond_logit_change=score - old_score,
        )
    assert fixed_graph_preserved(nodes, edges, layout.record)
    assert all(
        local[key] == list(state[key])
        for key in ("nodes", "parents", "closure_left", "closure_right")
    )
    return dict(
        state=local,
        smiles=graph_smiles(nodes, edges, atoms),
        decisions=decisions,
        changed_rings=sum(d["status"] == "TRAIN_ring_motif_proposal" for d in decisions),
        scope="Post-training isolated-ring motif proposal; complete source checks still required",
    )
