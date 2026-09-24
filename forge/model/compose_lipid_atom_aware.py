"""Choose atom identities before spending their capacity on generated topology."""

from collections.abc import Sequence

import numpy as np

from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord


def topology_atom_states(
    logits: np.ndarray,
    record: SynthesisProgramGraphRecord,
    atom_capacities: np.ndarray,
    bond_units: Sequence[int],
) -> tuple[np.ndarray | None, str | None]:
    """Deterministic atom MAP subject to unavoidable incident-bond requirements.

    Fixed bonds and every non-root node's incoming tree edge are unavoidable.
    Later variable edges must fit the chosen atom; they cannot change its element
    to acquire extra capacity. This bounded factorization can abstain even when a
    different joint assignment exists. It neither forbids vocabulary states nor
    establishes chemical credibility beyond the supplied valence capacities.
    """
    count = record.node_count
    if logits.shape != (count, len(atom_capacities)) or not np.isfinite(logits).all():
        raise ValueError("Atom logits must be finite and match record and vocabulary")
    required = np.zeros(count, dtype=np.int64)
    for child in range(1, count):
        if record.fixed_parent_bond_mask[child]:
            parent = int(record.graph.parents[child])
            bond = int(record.graph.parent_bonds[child])
            if not 0 <= parent < child or not 0 <= bond < len(bond_units):
                return None, "invalid_fixed_parent_edge"
            units = int(bond_units[bond])
            required[[child, parent]] += units
        else:
            required[child] += 2
    for slot in np.flatnonzero(record.fixed_closure_bond_mask):
        left, right = int(record.graph.closure_left[slot]), int(record.graph.closure_right[slot])
        bond = int(record.graph.closure_bonds[slot])
        if not 0 <= left < right < count or not 0 <= bond < len(bond_units):
            return None, "invalid_fixed_closure"
        required[[left, right]] += int(bond_units[bond])
    if count > 1:
        required[0] = max(2, required[0])
    states = np.asarray(record.graph.node_states, dtype=np.int64).copy()
    for node in range(count):
        if record.fixed_atom_mask[node]:
            continue
        allowed = atom_capacities >= required[node]
        if not allowed.any():
            return None, "atom_aware_unavoidable_valence_unavailable"
        states[node] = int(np.argmax(np.where(allowed, logits[node], -np.inf)))
    return states, None
