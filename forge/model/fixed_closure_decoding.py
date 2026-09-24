"""Reserve every immutable edge before choosing variable sparse-tree parents.

The frozen decoder remains the chemistry/validity authority. This front-end selects the same
greedy parent scores under capacity that already includes fixed closures, then sends that
topology through the unchanged bond, atom, and graph checks. No-fixed-closure inputs are identity.
"""

from __future__ import annotations

import numpy as np

from forge.model.sparse_topology_feasibility import BOND_VALENCE_UNITS
from forge.model.synthesis_program_sampling import (
    _argmax_allowed,
    _atom_capacity_table,
    _bond_unit_table,
    decode_synthesis_program_strict_argmax,
)


def _reserved_parents(logits, record, atom_vocabulary, bond_classes):
    count = record.node_count
    capacities = _atom_capacity_table(atom_vocabulary)
    units = _bond_unit_table(bond_classes)
    capacity = np.full(count, int(capacities.max()), dtype=np.int64)
    for node in np.flatnonzero(record.fixed_atom_mask):
        state = int(record.graph.node_states[node])
        if not 0 <= state < len(capacities):
            return None, "fixed_atom_state_outside_vocabulary"
        capacity[node] = capacities[state]
    used = np.zeros(count, dtype=np.int64)
    parents = np.zeros(count, dtype=np.int64)
    occupied = set()

    def reserve(left, right, bond):
        pair = tuple(sorted((int(left), int(right))))
        if not 0 <= pair[0] < pair[1] < count or pair in occupied or not 0 <= bond < bond_classes:
            return False
        occupied.add(pair)
        used[list(pair)] += int(units[bond])
        return True

    for child in np.flatnonzero(record.fixed_parent_bond_mask):
        parent, bond = int(record.graph.parents[child]), int(record.graph.parent_bonds[child])
        if not 0 <= parent < child or not reserve(parent, child, bond):
            return None, "invalid_fixed_parent_edge"
        parents[child] = parent
    for slot in np.flatnonzero(record.fixed_closure_bond_mask):
        if not reserve(
            int(record.graph.closure_left[slot]),
            int(record.graph.closure_right[slot]),
            int(record.graph.closure_bonds[slot]),
        ):
            return None, "invalid_fixed_closure"
    if np.any(used > capacity):
        return None, "immutable_edge_valence_exceeds_support"
    minimum_bond = int(units.min())
    for child in range(1, count):
        if record.fixed_parent_bond_mask[child]:
            continue
        allowed = np.zeros(count, dtype=np.bool_)
        if used[child] + minimum_bond <= capacity[child]:
            allowed[:child] = used[:child] + minimum_bond <= capacity[:child]
            for parent in np.flatnonzero(allowed):
                if (int(parent), child) in occupied:
                    allowed[parent] = False
        parent = _argmax_allowed(logits[child, :count], allowed)
        if parent is None:
            return None, "parent_capacity_exhausted_after_fixed_closure_reservation"
        parents[child] = parent
        used[[child, parent]] += minimum_bond
        occupied.add((parent, child))
    return parents, None


def decode_fixed_closure_reserved_argmax(predictions, layout, records, atom_vocabulary):
    """Strict base-policy decoding with fixed-closure capacity reserved before parent choices."""
    if len(records) != int(layout["node_mask"].shape[0]):
        raise ValueError("closure-reserved decoder batch and records differ")
    active = [i for i, r in enumerate(records) if np.any(r.fixed_closure_bond_mask)]
    if not active or predictions["parent_bonds"].shape[-1] > len(BOND_VALENCE_UNITS):
        return decode_synthesis_program_strict_argmax(predictions, layout, records, atom_vocabulary)
    import torch

    revised = {**predictions, "parents": predictions["parents"].clone()}
    parent_logits = predictions["parents"].detach().cpu().numpy()
    paths, failures = {}, {}
    for index in active:
        record = records[index]
        parents, reason = _reserved_parents(
            parent_logits[index], record, atom_vocabulary, predictions["parent_bonds"].shape[-1]
        )
        if reason:
            failures[index] = reason
            continue
        paths[index] = parents
        # Encode the selected feasible topology for the frozen decoder. Atom and bond scores
        # are untouched. Selection uses the original parent scores, not the sentinel magnitudes.
        bound = torch.finfo(revised["parents"].dtype)
        for child in range(1, record.node_count):
            if not record.fixed_parent_bond_mask[child]:
                revised["parents"][index, child].fill_(bound.min)
                revised["parents"][index, child, parents[child]] = bound.max
    terminal, reasons = decode_synthesis_program_strict_argmax(
        revised, layout, records, atom_vocabulary
    )
    reasons = list(reasons)
    for index, reason in failures.items():
        reasons[index] = reason
    for index, parents in paths.items():
        if reasons[index] is None and not np.array_equal(
            terminal["parents"][index, : len(parents)].cpu().numpy(), parents
        ):
            raise ValueError("strict decoder departed from the reserved parent choices")
    return terminal, tuple(reasons)
