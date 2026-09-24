"""Qualified TRAIN core conditions, with every precursor interior kept variable.

Typed cores are sampled conditions, not invariants inferred for unseen chemistry.
The bank contains anonymous reaction cores only, never complete component graphs.
"""

import json
from dataclasses import replace

import numpy as np

from forge.corpus.qualified_program_cache import QualifiedProgramExample
from forge.model.combinatorial_core_scaffold import (
    CoreScaffold,
    apply_core_scaffold,
    core_coordinates,
    extract_core,
)
from forge.model.sparse_topology_feasibility import BOND_VALENCE_UNITS


def core_condition(record):
    """Anonymous typed core plus heavy-atom valence at its mapped positions."""
    key, core = extract_core(record)
    _, indices = core_coordinates(record)
    used = np.zeros(record.node_count, dtype=np.int64)
    graph = record.graph
    for left, right, bond in [
        *(
            (int(graph.parents[i]), i, int(graph.parent_bonds[i]))
            for i in range(1, record.node_count)
        ),
        *zip(graph.closure_left, graph.closure_right, graph.closure_bonds),
    ]:
        used[[left, right]] += int(BOND_VALENCE_UNITS[bond])
    return key, {
        "nodes": core.node_states,
        "edges": core.edges,
        "units": tuple(int(used[i]) for i in indices),
    }


def sample_core_condition(record, bank, rng):
    """Draw a core with the conditional mass fitted on qualified TRAIN records."""
    key, indices = core_coordinates(record)
    values = bank.get(json.dumps(key, separators=(",", ":")))
    if not values:
        raise ValueError("Layout has no qualified TRAIN core support")
    weights = np.array([item["mass"] for item in values], dtype=np.float64)
    item = values[int(rng.choice(len(values), p=weights / weights.sum()))]
    core = CoreScaffold(key, tuple(item["nodes"]), tuple(tuple(e) for e in item["edges"]))
    units = np.full(record.node_count, -1, dtype=np.int64)
    units[list(indices)] = item["units"]
    return core, units


def condition_on_qualified_core(example: QualifiedProgramExample) -> QualifiedProgramExample:
    """Mask the admitted source core in its existing, lossless sparse serialization."""
    record, graph = example.record, example.record.graph
    core = record.core_position_states > 1
    parents = core & core[graph.parents]
    parents[0] = False
    closures = core[graph.closure_left] & core[graph.closure_right]
    return replace(
        example,
        record=replace(
            record,
            fixed_atom_mask=record.fixed_atom_mask | core,
            fixed_parent_bond_mask=record.fixed_parent_bond_mask | parents,
            fixed_closure_bond_mask=record.fixed_closure_bond_mask | closures,
        ),
    )


def layout_with_sampled_core(example: QualifiedProgramExample, core):
    """Scrub target values before installing a separately sampled TRAIN core."""
    record = example.record
    graph = record.graph
    blank = replace(
        graph,
        canonical_smiles="",
        node_states=np.where(record.fixed_atom_mask, graph.node_states, 0),
        parents=np.where(record.fixed_parent_bond_mask, graph.parents, 0),
        parent_bonds=np.where(record.fixed_parent_bond_mask, graph.parent_bonds, 0),
        closure_left=np.where(record.fixed_closure_bond_mask, graph.closure_left, 0),
        closure_right=np.where(record.fixed_closure_bond_mask, graph.closure_right, 0),
        closure_bonds=np.where(record.fixed_closure_bond_mask, graph.closure_bonds, 0),
        edges=np.empty((0, 0), dtype=np.int8),
    )
    result, reason = apply_core_scaffold(replace(record, graph=blank), core)
    return (None if result is None else replace(example, record=result)), reason
