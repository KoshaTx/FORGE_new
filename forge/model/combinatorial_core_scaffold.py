"""Sample reaction-core alternatives from qualified TRAIN semantics, without precursor interiors."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace

import numpy as np

from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord


class CoreScaffoldError(ValueError):
    """A core or layout violates the declared conditioning contract."""


def core_coordinates(record):
    """Return anonymous block/core labels and their node correspondence, without size or identity."""
    blocks = []
    for block in record.component_blocks:
        indices = sorted(
            (i for i in range(block.start, block.stop) if record.core_position_states[i] > 1),
            key=lambda i: (int(record.core_position_states[i]), i),
        )
        labels = tuple(int(record.core_position_states[i]) for i in indices)
        blocks.append((int(block.role_state), labels, tuple(indices)))
    blocks.sort(key=lambda b: (b[0], b[1], b[2]))
    key = (record.program_id, record.program_depth, tuple((r, c) for r, c, _ in blocks))
    return key, tuple(i for _, _, indices in blocks for i in indices)


@dataclass(frozen=True, order=True)
class CoreScaffold:
    semantic_key: tuple
    node_states: tuple[int, ...]
    edges: tuple[tuple[int, int, int], ...]


def extract_core(record):
    key, indices = core_coordinates(record)
    positions = {node: i for i, node in enumerate(indices)}
    edges = []
    pairs = [
        (int(record.graph.parents[i]), i, int(record.graph.parent_bonds[i]))
        for i in range(1, record.node_count)
    ] + list(zip(record.graph.closure_left, record.graph.closure_right, record.graph.closure_bonds))
    for left, right, bond in pairs:
        if left in positions and right in positions:
            a, b = sorted((positions[left], positions[right]))
            edges.append((a, b, int(bond)))
    return key, CoreScaffold(
        key, tuple(int(record.graph.node_states[i]) for i in indices), tuple(sorted(edges))
    )


class CoreScaffoldPrior:
    """Retain every observed typed core alternative, weighted within semantic bundles."""

    def __init__(self, cache):
        counts = defaultdict(Counter)
        self.training_records = 0
        for index in cache.indices(fold="train"):
            key, core = extract_core(cache.record(int(index)))
            weight = float(cache.arrays["source_weights"][index])
            if not np.isfinite(weight) or weight <= 0 or not core.node_states:
                raise CoreScaffoldError("core support has invalid TRAIN weight or empty semantics")
            counts[key][core] += weight
            self.training_records += 1
        self.support = {}
        for key, values in counts.items():
            choices = tuple(sorted(values))
            weights = np.asarray([values[c] for c in choices], dtype=np.float64)
            self.support[key] = (choices, weights / weights.sum())

    def sample(self, record, *, seed):
        key, _ = core_coordinates(record)
        if key not in self.support:
            raise CoreScaffoldError("sampled semantic bundle lacks a qualified TRAIN core")
        choices, weights = self.support[key]
        index = int(np.random.default_rng(seed).choice(len(choices), p=weights))
        return choices[index]

    def report(self):
        return {
            "training_records": self.training_records,
            "semantic_bundles": len(self.support),
            "typed_core_alternatives": sum(len(c) for c, _ in self.support.values()),
            "by_family": {
                family: {
                    "bundles": sum(k[0] == family for k in self.support),
                    "alternatives": sum(
                        len(c) for k, (c, _) in self.support.items() if k[0] == family
                    ),
                }
                for family in sorted({k[0] for k in self.support})
            },
        }


def canonical_core_slots(record):
    """Place anonymous core slots first within each otherwise empty block, in both paired arms."""
    if np.any(record.fixed_atom_mask):
        return record
    if (
        record.graph.canonical_smiles
        or np.any(record.graph.node_states)
        or np.any(record.graph.parent_bonds)
        or np.any(record.fixed_parent_bond_mask)
        or np.any(record.fixed_closure_bond_mask)
    ):
        raise CoreScaffoldError("core slot normalization requires an empty count-only layout")
    positions = record.core_position_states.copy()
    for block in record.component_blocks:
        core = sorted(int(c) for c in positions[block.start : block.stop] if c > 1)
        positions[block.start : block.stop] = core + [1] * (block.atom_count - len(core))
    return replace(record, core_position_states=positions)


def apply_core_scaffold(record: SynthesisProgramGraphRecord, core: CoreScaffold):
    """Fix only the sampled core; abstain if its sparse serialization exceeds existing slots."""
    key, indices = core_coordinates(record)
    if (
        key != core.semantic_key
        or len(indices) != len(core.node_states)
        or any(s < 0 for s in core.node_states)
    ):
        raise CoreScaffoldError("scaffold node support differs from layout")
    graph = record.graph
    if graph.canonical_smiles or np.any(graph.node_states[~record.fixed_atom_mask]):
        raise CoreScaffoldError("scaffold requires a target-free layout")
    nodes, fixed_atoms = graph.node_states.copy(), record.fixed_atom_mask.copy()
    for node, state in zip(indices, core.node_states, strict=True):
        if fixed_atoms[node] and nodes[node] != state:
            raise CoreScaffoldError("sampled core contradicts existing fixed atom")
        nodes[node], fixed_atoms[node] = state, True
    edges = {}

    def add(left, right, bond):
        pair = tuple(sorted((int(left), int(right))))
        if pair[0] == pair[1] or bond < 0 or (pair in edges and edges[pair] != bond):
            raise CoreScaffoldError("core has a contradictory edge")
        edges[pair] = int(bond)

    for i in np.flatnonzero(record.fixed_parent_bond_mask):
        add(graph.parents[i], i, graph.parent_bonds[i])
    for i in np.flatnonzero(record.fixed_closure_bond_mask):
        add(graph.closure_left[i], graph.closure_right[i], graph.closure_bonds[i])
    for left, right, bond in core.edges:
        if not 0 <= left < right < len(indices):
            raise CoreScaffoldError("core edge has invalid endpoints")
        add(indices[left], indices[right], bond)
    parents, bonds = graph.parents.copy(), graph.parent_bonds.copy()
    fixed_parents = record.fixed_parent_bond_mask.copy()
    lefts, rights, cbonds = (
        graph.closure_left.copy(),
        graph.closure_right.copy(),
        graph.closure_bonds.copy(),
    )
    fixed_closures = record.fixed_closure_bond_mask.copy()
    existing_closures = {
        tuple(sorted((int(lefts[i]), int(rights[i])))) for i in np.flatnonzero(fixed_closures)
    }
    for (left, right), bond in sorted(edges.items()):
        if (left, right) in existing_closures:
            continue
        if not fixed_parents[right]:
            parents[right], bonds[right], fixed_parents[right] = left, bond, True
        elif parents[right] == left and bonds[right] == bond:
            continue
        else:
            available = np.flatnonzero(~fixed_closures)
            if not len(available):
                return None, "core_serialization_exceeds_sampled_closure_slots"
            slot = available[0]
            if not fixed_atoms[left] or not fixed_atoms[right]:
                return None, "closure_would_fix_an_exterior_atom"
            lefts[slot], rights[slot], cbonds[slot], fixed_closures[slot] = left, right, bond, True
    updated = replace(
        graph,
        node_states=nodes,
        parents=parents,
        parent_bonds=bonds,
        closure_left=lefts,
        closure_right=rights,
        closure_bonds=cbonds,
    )
    return (
        replace(
            record,
            graph=updated,
            fixed_atom_mask=fixed_atoms,
            fixed_parent_bond_mask=fixed_parents,
            fixed_closure_bond_mask=fixed_closures,
        ),
        None,
    )
