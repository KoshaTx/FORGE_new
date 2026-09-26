"""Bounded, opt-in TRAIN ring-system topology proposals on generated atom slots.

Only ring connectivity and typed ring bonds are supplied by the prior. No atoms,
whole precursors, external bonds or fixed reaction-core state are substituted.
Unknown originals remain available to the caller and are never declared invalid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, rdBase

from forge.model.compose_lipid_generation import _repeat_correspondences
from forge.model.defog_feasibility import graph_to_molecule
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles
from forge.model.sparse_topology_feasibility import BOND_VALENCE_UNITS
from forge.model.synthesis_program_sampling import _atom_capacity_table


@dataclass(frozen=True)
class RingSystemMotif:
    identity: str
    atoms: tuple[tuple[str, int], ...]
    bonds: tuple[tuple[int, ...], ...]
    cycle_rank: int


@dataclass(frozen=True)
class RingSystemLimits:
    maximum_proposals: int = 2
    beam_width: int = 32
    maximum_states_per_system: int = 256

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in vars(self).values()):
            raise ValueError("Ring-system limits must be positive integers")


def ring_systems(molecule):
    """Atom sets connected by ring bonds; bridges between rings stay exterior."""
    adjacency = {}
    for bond in molecule.GetBonds():
        if bond.IsInRing():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
    remaining, output = set(adjacency), []
    while remaining:
        todo, found = [min(remaining)], set()
        while todo:
            atom = todo.pop()
            if atom not in found:
                found.add(atom)
                todo.extend(adjacency[atom] - found)
        remaining -= found
        output.append(tuple(sorted(found)))
    return tuple(output)


def _fragment(mol, members):
    editable = Chem.RWMol()
    mapping = {}
    for old in members:
        atom = mol.GetAtomWithIdx(old)
        new = Chem.Atom(atom.GetSymbol())
        new.SetFormalCharge(atom.GetFormalCharge())
        mapping[old] = editable.AddAtom(new)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in mapping and b in mapping and bond.IsInRing():
            editable.AddBond(mapping[a], mapping[b], bond.GetBondType())
    out = editable.GetMol()
    Chem.SanitizeMol(out)
    return out


def extract_ring_system_motifs(smiles: Sequence[str]) -> tuple[RingSystemMotif, ...]:
    motifs = {}
    for value in sorted(set(smiles)):
        mol = Chem.MolFromSmiles(value)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("TRAIN ring reference must contain connected valid components")
        systems = ring_systems(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        for members in systems:
            fragment = _fragment(mol, members)
            identity = Chem.MolToSmiles(fragment)
            canonical = Chem.MolFromSmiles(identity)
            Chem.Kekulize(canonical, clearAromaticFlags=True)
            bonds = np.zeros((len(members), len(members)), dtype=int)
            for bond in canonical.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bonds[a, b] = bonds[b, a] = int(bond.GetBondTypeAsDouble())
            motifs[identity] = RingSystemMotif(
                identity,
                tuple((a.GetSymbol(), a.GetFormalCharge()) for a in canonical.GetAtoms()),
                tuple(tuple(map(int, row)) for row in bonds),
                canonical.GetNumBonds() - canonical.GetNumAtoms() + 1,
            )
    return tuple(motifs[k] for k in sorted(motifs))


def ordered_state(nodes, edges, layout):
    """Choose an explicit model-compatible ordered tree; never fabricate provenance."""
    record = layout.record
    parents, bonds, tree = [0] * len(nodes), [0] * len(nodes), set()
    for child in range(1, len(nodes)):
        options = np.flatnonzero(edges[child, :child])
        if not len(options):
            return None
        parent = (
            int(record.graph.parents[child])
            if record.fixed_parent_bond_mask[child]
            else int(options[0])
        )
        if not edges[parent, child]:
            return None
        parents[child], bonds[child] = parent, int(edges[parent, child]) - 1
        tree.add((parent, child))
    closure = [
        (int(a), int(b))
        for a, b in zip(*np.nonzero(np.triu(edges)), strict=True)
        if (a, b) not in tree
    ]
    return dict(
        nodes=list(map(int, nodes)),
        parents=parents,
        parent_bonds=bonds,
        closure_left=[a for a, _ in closure],
        closure_right=[b for _, b in closure],
        closure_bonds=[int(edges[a, b]) - 1 for a, b in closure],
    )


def _cycle_order(edges, members):
    if any(np.count_nonzero(edges[i, list(members)]) != 2 for i in members):
        return None
    path = [min(members)]
    while len(path) < len(members):
        choices = [j for j in members if edges[path[-1], j] and j not in path]
        if not choices:
            return None
        path.append(min(choices))
    return path


def _score(edges, predictions):
    a, b = np.nonzero(np.triu(edges))
    return float(
        sum(
            predictions["parents"][j, i] + predictions["parent_bonds"][j, int(edges[i, j]) - 1]
            for i, j in zip(a, b, strict=True)
        )
    )


def ring_system_diagnostics(layout, nodes, edges, atoms, motifs_by_role):
    """Describe role-local motif support; absence is unknown, never invalidity."""
    molecule = graph_to_molecule(np.asarray(nodes), np.asarray(edges), atoms)
    records = []
    for members in ring_systems(molecule):
        blocks = [
            b
            for b in layout.record.component_blocks
            if set(members).issubset(range(b.start, b.stop))
        ]
        role = blocks[0].role if len(blocks) == 1 else None
        try:
            identity = Chem.MolToSmiles(_fragment(molecule, members))
        except Chem.MolSanitizeException:
            # Cutting exterior bonds/explicit-H context can make a valid full
            # molecule's isolated aromatic fragment unrepresentable. Preserve
            # the product and record unknown support; never invent a motif.
            records.append(
                dict(
                    atoms=list(members),
                    role=role,
                    identity=None,
                    observed=False,
                    reason="ring_fragment_requires_external_context",
                )
            )
            continue
        records.append(
            dict(
                atoms=list(members),
                role=role,
                identity=identity,
                observed=identity in {m.identity for m in motifs_by_role.get(role, ())},
            )
        )
    return dict(
        ring_systems=records,
        complete_typed_ring_systems_observed=all(r["observed"] for r in records),
        interpretation="TRAIN motif support; not chemistry validity, sampled-size compliance or synthesis admission",
    )


def _simple_trials(edges, labels, members, motif, budget):
    cycle = _cycle_order(edges, members)
    template = np.asarray(motif.bonds)
    order = _cycle_order(template, tuple(range(len(template))))
    if cycle is None or order is None or len(order) > len(cycle):
        return
    count = 0
    for direction in (cycle, list(reversed(cycle))):
        for start in range(len(cycle)):
            path = (direction[start:] + direction[:start])[: len(order)]
            for shift in range(len(order)):
                source = order[shift:] + order[:shift]
                if any(labels[a] != motif.atoms[b] for a, b in zip(path, source, strict=True)):
                    continue
                for cut_end in ((0, 1) if len(path) < len(cycle) else (0,)):
                    if count >= budget:
                        return
                    count += 1
                    trial = edges.copy()
                    if len(path) < len(cycle):
                        endpoint = path[-1] if cut_end == 0 else path[0]
                        outside = next(j for j in members if edges[endpoint, j] and j not in path)
                        trial[endpoint, outside] = trial[outside, endpoint] = 0
                    for k, (a, b) in enumerate(zip(path, path[1:] + path[:1], strict=True)):
                        bond = template[source[k], source[(k + 1) % len(source)]]
                        trial[a, b] = trial[b, a] = bond
                    yield trial


def _cage_trials(edges, labels, pool, motif, predictions, capacities, limits):
    """Map a typed motif onto the same slots, enforcing ordered-tree connectivity."""
    n = len(pool)
    if n != len(motif.atoms) or n * (n + 1) // 2 > limits.maximum_states_per_system:
        return [], 0
    width = min(
        limits.beam_width,
        max(1, (limits.maximum_states_per_system - n) // max(1, n * (n - 1) // 2)),
    )
    template = np.asarray(motif.bonds)
    external = edges.copy()
    external[np.ix_(pool, pool)] = 0
    units = np.asarray([0, *BOND_VALENCE_UNITS.tolist()])
    outside = units[external].sum(1)
    degree_units = units[template].sum(1)
    beam, explored = [(0.0, ())], 0
    for position, atom in enumerate(pool):
        choices = []
        for score, mapping in beam:
            for source in range(n):
                if source in mapping or labels[atom] != motif.atoms[source]:
                    continue
                if outside[atom] + degree_units[source] > capacities[atom]:
                    continue
                if not np.any(external[atom, :atom]) and not any(
                    template[source, t] for t in mapping
                ):
                    continue
                explored += 1
                if explored > limits.maximum_states_per_system:
                    raise AssertionError("Ring assignment exceeded frozen state budget")
                delta = sum(
                    predictions["parents"][atom, pool[j]]
                    + predictions["parent_bonds"][atom, template[source, t] - 1]
                    for j, t in enumerate(mapping)
                    if template[source, t]
                )
                choices.append((score + float(delta), mapping + (source,)))
        beam = sorted(choices, key=lambda x: (-x[0], x[1]))[:width]
        if not beam:
            return [], explored
    output = []
    for _, mapping in beam:
        trial = external.copy()
        trial[np.ix_(pool, pool)] = template[np.ix_(mapping, mapping)]
        output.append(trial)
    return output, explored


def propose_ring_systems(
    layout,
    nodes,
    edges,
    atoms,
    predictions: Mapping,
    motifs_by_role,
    *,
    limits: RingSystemLimits = RingSystemLimits(),
):
    """Return at most two finalized graph proposals; the unchanged original is explicit."""
    nodes, edges = np.asarray(nodes), np.asarray(edges)
    if not np.issubdtype(nodes.dtype, np.integer) or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("Ring-system nodes and bonds must contain integer classes")
    if nodes.shape != (layout.record.node_count,) or edges.shape != (len(nodes), len(nodes)):
        raise ValueError("Ring-system input differs from declared atom budget")
    if not len(nodes) or nodes.min() < 0 or nodes.max() >= len(atoms):
        raise ValueError("Ring-system atom class is outside vocabulary")
    if (
        not np.array_equal(edges, edges.T)
        or np.any(np.diag(edges))
        or not fixed_graph_preserved(nodes, edges, layout.record)
    ):
        raise ValueError("Ring-system input changed graph symmetry or fixed source core")
    predictions = {k: np.asarray(predictions[k]) for k in ("parents", "parent_bonds")}
    if (
        any(value.ndim != 2 for value in predictions.values())
        or predictions["parents"].shape[0] < len(nodes)
        or predictions["parents"].shape[1] < len(nodes)
        or predictions["parent_bonds"].shape[0] < len(nodes)
    ):
        raise ValueError("Ring-system logits have insufficient atom support")
    if any(not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError("Ring-system logits must be finite")
    if edges.max() > predictions["parent_bonds"].shape[1] or edges.min() < 0:
        raise ValueError("Ring-system bond class is outside prediction vocabulary")
    labels = [(atoms[int(i)].symbol, atoms[int(i)].formal_charge) for i in nodes]
    capacities = _atom_capacity_table(atoms)[nodes]
    units = np.asarray([0, *BOND_VALENCE_UNITS.tolist()])
    original_smiles = graph_smiles(nodes, edges, atoms)
    if original_smiles is None:
        return dict(
            status="invalid_input_retained",
            original=dict(smiles=None),
            proposals=[],
            accounting={},
            limits=vars(limits),
        )
    molecule = graph_to_molecule(nodes, edges, atoms)
    systems = ring_systems(molecule)
    groups, _ = _repeat_correspondences(layout)
    ties, skip = {}, set()
    for group in groups or []:
        first = group[0]
        if all(
            np.array_equal(nodes[first], nodes[other])
            and np.array_equal(edges[np.ix_(first, first)], edges[np.ix_(other, other)])
            for other in group[1:]
        ):
            ties[frozenset(first)] = group
            skip.update(i for other in group[1:] for i in other)
    proposals = [(edges.copy(), [], [])]
    accounting = dict(systems_considered=0, states_explored=0, graphs_sanitized=0)
    abstentions = []
    for block in layout.record.component_blocks:
        members = list(range(block.start, block.stop))
        if any(i in skip for i in members):
            continue
        local_systems = [s for s in systems if set(s).issubset(members)]
        for system in local_systems:
            if any(
                layout.record.core_position_states[i] > 1 or layout.record.fixed_atom_mask[i]
                for i in system
            ):
                abstentions.append("ring_intersects_fixed_core")
                continue
            try:
                fragment = _fragment(molecule, system)
            except Chem.MolSanitizeException:
                abstentions.append("ring_fragment_requires_external_context")
                continue
            identity = Chem.MolToSmiles(fragment)
            motifs = tuple(motifs_by_role.get(block.role, ()))
            simple = _cycle_order(edges, system) is not None
            allowed_sizes = layout.ring_sizes_by_role.get(block.role_state, ())
            if identity in {m.identity for m in motifs} and (
                not simple or len(system) in allowed_sizes
            ):
                continue
            accounting["systems_considered"] += 1
            rank = int(np.count_nonzero(np.triu(edges[np.ix_(system, system)]))) - len(system) + 1
            compatible = [
                m
                for m in motifs
                if m.cycle_rank == rank
                and (rank != 1 or len(m.atoms) in allowed_sizes)
                and all(b <= predictions["parent_bonds"].shape[1] for row in m.bonds for b in row)
            ]
            candidates = []
            remaining = limits.maximum_states_per_system
            for previous, history, motif_ids in proposals:
                for motif in compatible:
                    if remaining <= 0:
                        break
                    if rank == 1:
                        trials = list(_simple_trials(previous, labels, system, motif, remaining))
                        explored = len(trials)
                    else:
                        pool = [
                            i
                            for i in members
                            if layout.record.core_position_states[i] == 1
                            and not layout.record.fixed_atom_mask[i]
                        ]
                        bounded = RingSystemLimits(
                            limits.maximum_proposals, limits.beam_width, remaining
                        )
                        trials, explored = _cage_trials(
                            previous, labels, pool, motif, predictions, capacities, bounded
                        )
                    remaining -= explored
                    accounting["states_explored"] += explored
                    for trial in trials:
                        changed = np.argwhere(np.triu(trial != previous))
                        group = ties.get(frozenset(members), [members])
                        if len(group) > 1:
                            reverse = {value: j for j, value in enumerate(group[0])}
                            for other in group[1:]:
                                for a, b in changed:
                                    if a not in reverse or b not in reverse:
                                        raise AssertionError("Ring projection escaped component")
                                    c, d = other[reverse[a]], other[reverse[b]]
                                    trial[c, d] = trial[d, c] = trial[a, b]
                        if np.any(units[trial].sum(1) > capacities) or not fixed_graph_preserved(
                            nodes, trial, layout.record
                        ):
                            continue
                        if np.count_nonzero(np.triu(trial)) != np.count_nonzero(np.triu(edges)):
                            continue
                        state = ordered_state(nodes, trial, layout)
                        if (
                            state is None
                            or len(state["closure_left"]) != layout.record.graph.closure_count
                        ):
                            continue
                        accounting["graphs_sanitized"] += 1
                        with rdBase.BlockLogs():
                            smiles = graph_smiles(nodes, trial, atoms)
                        if smiles is None:
                            continue
                        edits = [
                            dict(
                                atoms=[int(a), int(b)],
                                before=int(previous[a, b]),
                                after=int(trial[a, b]),
                            )
                            for a, b in np.argwhere(np.triu(previous != trial))
                        ]
                        candidates.append(
                            (
                                _score(trial, predictions),
                                smiles,
                                trial,
                                history + edits,
                                motif_ids
                                + [
                                    dict(
                                        role=block.role,
                                        identity=motif.identity,
                                        ring_atom_count=len(motif.atoms),
                                        cycle_rank=rank,
                                        tied_occurrences=len(group),
                                    )
                                ],
                            )
                        )
            unique = {}
            for value in sorted(candidates, key=lambda x: (-x[0], x[1])):
                unique.setdefault(value[1], value)
            if unique:
                proposals = [
                    (v[2], v[3], v[4]) for v in list(unique.values())[: limits.maximum_proposals]
                ]
            else:
                abstentions.append("no_compatible_bounded_ring_assignment")
    emitted = []
    for trial, edits, ids in proposals:
        if not edits:
            continue
        smiles = graph_smiles(nodes, trial, atoms)
        emitted.append(
            dict(
                nodes=nodes.tolist(),
                edges=trial.tolist(),
                smiles=smiles,
                tree_state=ordered_state(nodes, trial, layout),
                tree_basis="ordered_tree_after_ring_system_repair",
                design_audit=ring_system_diagnostics(layout, nodes, trial, atoms, motifs_by_role),
                edits=edits,
                motif_ids=ids,
                training_prior=True,
            )
        )
    return dict(
        status="proposed" if emitted else "unchanged",
        original=dict(
            smiles=original_smiles,
            design_audit=ring_system_diagnostics(layout, nodes, edges, atoms, motifs_by_role),
        ),
        proposals=emitted,
        accounting=accounting,
        abstentions=abstentions,
        limits=vars(limits),
    )
