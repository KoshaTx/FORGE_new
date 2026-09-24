"""Transfer only a qualified core traversal to sampled exterior counts."""

from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np

from forge.model.combinatorial_core_scaffold import (
    CoreScaffold,
    CoreScaffoldError,
    core_coordinates,
    extract_core,
)
from forge.model.synthesis_program_graph import SynthesisProgramComponentBlock


class CoreOrderPrior:
    """Retain weighted core-only traversal alternatives; no exterior indices or graphs survive."""

    def __init__(self, cache):
        support = defaultdict(Counter)
        for index in cache.indices(fold="train"):
            record = cache.record(int(index))
            _, core = extract_core(record)
            _, nodes = core_coordinates(record)
            order = tuple(sorted(range(len(nodes)), key=lambda i: nodes[i]))
            support[core][order] += float(cache.arrays["source_weights"][index])
        self.support = {}
        for core, counts in support.items():
            orders = tuple(sorted(counts))
            weights = np.asarray([counts[o] for o in orders])
            if not np.isfinite(weights).all() or np.any(weights <= 0):
                raise CoreScaffoldError("core ordering weights are invalid")
            self.support[core] = (orders, weights / weights.sum())

    def sample(self, core, *, seed):
        if core not in self.support:
            raise CoreScaffoldError("core has no qualified traversal")
        orders, weights = self.support[core]
        return orders[int(np.random.default_rng(seed).choice(len(orders), p=weights))]


def order_core_slots(record, core, order):
    """Reorder entire blocks and their core slots, preserving exterior counts and core identity."""
    key, nodes = core_coordinates(record)
    if key != core.semantic_key or sorted(order) != list(range(len(nodes))):
        raise CoreScaffoldError("core traversal differs from sampled core")
    if np.any(record.fixed_atom_mask):
        return record, core
    if (
        record.graph.canonical_smiles
        or np.any(record.graph.node_states)
        or np.any(record.graph.parents)
        or np.any(record.graph.parent_bonds)
        or record.role_morphology_states is not None
        or np.any(record.fixed_parent_bond_mask)
        or np.any(record.fixed_closure_bond_mask)
    ):
        raise CoreScaffoldError("core ordering requires an empty count layout")
    rank = {nodes[i]: position for position, i in enumerate(order)}
    blocks = sorted(
        record.component_blocks,
        key=lambda b: min(
            (rank[i] for i in range(b.start, b.stop) if i in rank), default=len(nodes) + b.start
        ),
    )
    permutation, new_blocks = [], []
    for block in blocks:
        start = len(permutation)
        permutation.extend(
            sorted((i for i in range(block.start, block.stop) if i in rank), key=rank.get)
        )
        permutation.extend(i for i in range(block.start, block.stop) if i not in rank)
        new_blocks.append(
            SynthesisProgramComponentBlock(block.role, block.role_state, start, len(permutation))
        )
    permutation = np.asarray(permutation, dtype=np.int64)
    inverse = np.argsort(permutation)
    graph = replace(record.graph, node_states=record.graph.node_states[permutation])
    after = replace(
        record,
        graph=graph,
        role_states=record.role_states[permutation],
        core_position_states=record.core_position_states[permutation],
        component_blocks=tuple(new_blocks),
        fixed_atom_mask=record.fixed_atom_mask[permutation],
        canonical_atom_order=np.arange(record.node_count),
    )
    new_key, new_nodes = core_coordinates(after)
    new_coordinate = {node: i for i, node in enumerate(new_nodes)}
    mapping = {i: new_coordinate[int(inverse[node])] for i, node in enumerate(nodes)}
    states = [0] * len(nodes)
    for old, new in mapping.items():
        states[new] = core.node_states[old]
    edges = tuple(sorted((*sorted((mapping[a], mapping[b])), bond) for a, b, bond in core.edges))
    return after, CoreScaffold(new_key, tuple(states), edges)
