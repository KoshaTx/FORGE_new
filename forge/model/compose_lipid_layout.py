"""Anonymous, factorized coarse layouts for qualified multireaction generation.

Only core chemistry and count/position conditions are stored. No exterior atom,
bond, parent pointer, product identity or precursor identity enters the prior.
Multiple connected regions of one source precursor and introduced atoms remain
explicit. Role draws preserve their joint repeated-occurrence size patterns.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from forge.model.introduced_source_coordinates import collate_source_and_introduced_records
from forge.model.reaction_program_flow import derive_role_morphology_states
from forge.model.sparse_topology_feasibility import BOND_VALENCE_UNITS, SparseGraphRecord
from forge.model.synthesis_program_graph import (
    SynthesisProgramComponentBlock,
    SynthesisProgramGraphRecord,
)


def encoded(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def tree_ring_size(parents, left, right):
    """Length of the fundamental ring formed by one closure on the declared tree."""
    paths = []
    for node in (left, right):
        path = [int(node)]
        while path[-1] != 0:
            parent = int(parents[path[-1]])
            if parent >= path[-1] or parent < 0:
                raise ValueError("Ring support requires an ordered rooted tree")
            path.append(parent)
        paths.append(path)
    distances = {node: i for i, node in enumerate(paths[0])}
    return next(i + distances[node] + 1 for i, node in enumerate(paths[1]) if node in distances)


def summarize_layout(example):
    """Extract an anonymous assembly bundle and independent per-role count options."""
    record, graph = example.record, example.record.graph
    core = record.core_position_states > 1
    if np.any(record.fixed_atom_mask & ~core):
        raise ValueError("A fixed exterior atom requires a separately qualified layout contract")
    indices = np.flatnonzero(core)
    inverse = {int(node): i for i, node in enumerate(indices)}
    used = np.zeros(record.node_count, dtype=np.int64)
    tree, closures = [], []
    variable_closures, rings = defaultdict(int), defaultdict(set)
    block_index = np.zeros(record.node_count, dtype=np.int64)
    for i, block in enumerate(record.component_blocks):
        block_index[block.start : block.stop] = i
    edges = [
        (int(graph.parents[i]), i, int(graph.parent_bonds[i]), False)
        for i in range(1, record.node_count)
    ] + [
        (int(a), int(b), int(c), True)
        for a, b, c in zip(graph.closure_left, graph.closure_right, graph.closure_bonds)
    ]
    for a, b, bond, closure in edges:
        used[[a, b]] += int(BOND_VALENCE_UNITS[bond])
        if core[a] and core[b]:
            (closures if closure else tree).append([inverse[a], inverse[b], bond])
        elif block_index[a] != block_index[b]:
            raise ValueError("Unmapped cross-origin edge cannot enter a generated layout")
        elif closure:
            role = int(record.role_states[a])
            variable_closures[role] += 1
            rings[role].add(tree_ring_size(graph.parents, a, b))
    morphology = derive_role_morphology_states(record)
    shapes = defaultdict(list)
    skeleton = []
    for block in record.component_blocks:
        positions = record.core_position_states[block.start : block.stop].tolist()
        skeleton.append([block.role, block.role_state, [p for p in positions if p > 1]])
        shapes[block.role_state].append(positions)
    bundle = dict(
        family=example.family,
        program=record.program_id,
        program_state=record.program_state,
        depth=record.program_depth,
        quantities=example.source_quantities,
        introduced=list(example.introduced_roles),
        blocks=skeleton,
        core_nodes=graph.node_states[indices].tolist(),
        core_units=used[indices].tolist(),
        core_tree=tree,
        core_closures=closures,
    )
    options = {
        role: dict(
            positions=positions,
            morphology=morphology[np.flatnonzero(record.role_states == role)[0]].tolist(),
            closures=variable_closures[role],
        )
        for role, positions in shapes.items()
    }
    return bundle, options, {role: sorted(values) for role, values in rings.items()}


@dataclass(frozen=True)
class SampledProgramLayout:
    record: SynthesisProgramGraphRecord
    family: str
    quantities: dict[str, int]
    introduced_roles: tuple[str, ...]
    core_units: np.ndarray
    variable_closures_by_role: dict[int, int]
    ring_sizes_by_role: dict[int, tuple[int, ...]]


def collate_generated_layouts(layouts, *, maximum_closures, maximum_nodes=None):
    return collate_source_and_introduced_records(
        [x.record for x in layouts],
        [x.quantities for x in layouts],
        introduced_roles=[x.introduced_roles for x in layouts],
        maximum_nodes=maximum_nodes,
        maximum_closures=maximum_closures,
    )


def build_layout(bundle, choices, rings, *, identity):
    """Instantiate only coarse conditions and the sampled anonymous assembly core."""
    blocks, roles, positions, morphology = [], [], [], []
    occurrences = defaultdict(int)
    for role, state, signature in bundle["blocks"]:
        choice = choices[int(state)]
        local = choice["positions"][occurrences[state]]
        occurrences[state] += 1
        if [p for p in local if p > 1] != signature:
            raise ValueError("Generated block changed its core coordinates")
        start = len(roles)
        roles.extend([state] * len(local))
        positions.extend(local)
        morphology.extend([choice["morphology"]] * len(local))
        blocks.append(SynthesisProgramComponentBlock(role, state, start, len(roles)))
    n = len(roles)
    core = np.flatnonzero(np.asarray(positions) > 1)
    nodes, parents, bonds = (np.zeros(n, dtype=np.int64) for _ in range(3))
    units = np.full(n, -1, dtype=np.int64)
    nodes[core], units[core] = bundle["core_nodes"], bundle["core_units"]
    fixed_atoms, fixed_parents = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    fixed_atoms[core] = True
    for a, b, bond in bundle["core_tree"]:
        parent, child = int(core[a]), int(core[b])
        if not 0 <= parent < child or fixed_parents[child]:
            raise ValueError("Core template does not have ordered unique parents")
        parents[child], bonds[child], fixed_parents[child] = parent, bond, True
    k = len(bundle["core_closures"]) + sum(c["closures"] for c in choices.values())
    left, right, cbond = (np.zeros(k, dtype=np.int64) for _ in range(3))
    fixed_closures = np.zeros(k, dtype=bool)
    for slot, (a, b, bond) in enumerate(bundle["core_closures"]):
        left[slot], right[slot], cbond[slot], fixed_closures[slot] = core[a], core[b], bond, True
    graph = SparseGraphRecord(
        identity, "", nodes, parents, bonds, left, right, cbond, np.empty((0, 0), dtype=np.int8)
    )
    record = SynthesisProgramGraphRecord(
        graph,
        np.arange(n),
        bundle["program"],
        bundle["program_state"],
        bundle["depth"],
        np.asarray(roles, dtype=np.int64),
        np.asarray(positions, dtype=np.int64),
        tuple(blocks),
        fixed_atoms,
        fixed_parents,
        fixed_closures,
        np.asarray(morphology, dtype=np.int64),
    )
    return SampledProgramLayout(
        record,
        bundle["family"],
        dict(bundle["quantities"]),
        tuple(bundle["introduced"]),
        units,
        {int(role): c["closures"] for role, c in choices.items()},
        {int(role): tuple(values) for role, values in rings.items()},
    )


class ComposeLipidLayoutPrior:
    """Sample role counts independently, exactly conditioned on atom/closure support."""

    def __init__(self, document):
        if document.get("schema_version") != "forge.compose_lipid_layout_prior.v1":
            raise ValueError("Unknown layout prior schema")
        self.document = document
        self.maximum_atoms = document["support"]["maximum_atoms"]
        self.maximum_closures = document["support"]["maximum_closures"]
        self._compiled = {}
        self._families = defaultdict(list)
        for i, item in enumerate(document["bundles"]):
            if not np.isfinite(item["mass"]) or item["mass"] <= 0:
                raise ValueError("Layout bundles require positive finite mass")
            self._families[item["bundle"]["family"]].append(i)

    def _compile(self, index):
        if index in self._compiled:
            return self._compiled[index]
        item = self.document["bundles"][index]
        roles = sorted(item["roles"], key=int)
        choices, probabilities, sizes = [], [], []
        nmax = self.maximum_atoms
        cmax = self.maximum_closures - len(item["bundle"]["core_closures"])
        if cmax < 0:
            raise ValueError("Core exceeds declared closure support")
        for role in roles:
            options = item["roles"][role]
            weights = np.array([v["mass"] for v in options], dtype=np.float64)
            if not np.isfinite(weights).all() or np.any(weights <= 0):
                raise ValueError("Role options require positive finite mass")
            choices.append([v["shape"] for v in options])
            probabilities.append(weights / weights.sum())
            sizes.append(
                [(sum(map(len, v["shape"]["positions"])), v["shape"]["closures"]) for v in options]
            )
        suffix = [np.zeros((nmax + 1, cmax + 1)) for _ in range(len(roles) + 1)]
        suffix[-1][0, 0] = 1
        for j in range(len(roles) - 1, -1, -1):
            for (n, c), weight in zip(sizes[j], probabilities[j], strict=True):
                if n <= nmax and c <= cmax:
                    suffix[j][n:, c:] += weight * suffix[j + 1][: nmax + 1 - n, : cmax + 1 - c]
        cumulative = [v.cumsum(0).cumsum(1) for v in suffix]
        if cumulative[0][-1, -1] <= 0:
            raise ValueError("No layout mass within declared full molecular support")
        result = roles, choices, probabilities, sizes, cumulative, cmax
        self._compiled[index] = result
        return result

    def sample(self, family, *, rng, identity):
        indices = self._families[family]
        weights = np.array([self.document["bundles"][i]["mass"] for i in indices])
        index = indices[int(rng.choice(len(indices), p=weights / weights.sum()))]
        item = self.document["bundles"][index]
        roles, choices, probabilities, sizes, cumulative, cmax = self._compile(index)
        nleft, cleft, selected = self.maximum_atoms, cmax, {}
        for j, role in enumerate(roles):
            feasible = np.array(
                [
                    (
                        weight * cumulative[j + 1][nleft - n, cleft - c]
                        if n <= nleft and c <= cleft
                        else 0
                    )
                    for (n, c), weight in zip(sizes[j], probabilities[j], strict=True)
                ]
            )
            if not np.isfinite(feasible).all() or feasible.sum() <= 0:
                raise ValueError("Conditioned layout probability is invalid")
            pick = int(rng.choice(len(feasible), p=feasible / feasible.sum()))
            selected[int(role)] = choices[j][pick]
            n, c = sizes[j][pick]
            nleft, cleft = nleft - n, cleft - c
        return build_layout(item["bundle"], selected, item["rings"], identity=identity)
