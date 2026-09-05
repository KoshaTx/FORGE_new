"""Attempt-weighted coarse topology diversity for generated Ugi products."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from forge.model.ugi_morphology_program import preorder_attached_forest_to_parents
from forge.potency.annotations import ROLE_NAMES


class UgiTopologyDiversityError(ValueError):
    """A sampled topology cannot be interpreted under the declared Ugi program."""


def _effective_count(values: Sequence[tuple[Any, ...]]) -> float:
    if not values:
        return 0.0
    counts = np.asarray(tuple(Counter(values).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(math.exp(-float(np.dot(probabilities, np.log(probabilities)))))


def role_topology_signature(
    offspring: Sequence[int],
    *,
    attachment_count: int,
    cycle_rank: int,
) -> tuple[Any, ...]:
    """Summarize visual tree shape without atom, bond or component identity."""

    word = np.asarray(tuple(offspring), dtype=np.int64)
    try:
        parents = preorder_attached_forest_to_parents(
            word,
            attachment_count=int(attachment_count),
        )
    except (RuntimeError, ValueError) as error:
        raise UgiTopologyDiversityError("sampled role offspring word is invalid") from error
    depths = np.ones(word.size, dtype=np.int64)
    children: list[list[int]] = [[] for _ in range(word.size)]
    roots: list[int] = []
    for node, parent in enumerate(parents.tolist()):
        if parent >= 0:
            depths[node] = depths[parent] + 1
            children[parent].append(node)
        else:
            roots.append(node)

    def canonical_subtree(node: int) -> tuple[Any, ...]:
        return tuple(sorted(canonical_subtree(child) for child in children[node]))

    canonical_forest = tuple(sorted(canonical_subtree(root) for root in roots))
    leaves = tuple(sorted(int(depths[index]) for index in np.flatnonzero(word == 0)))
    branches = tuple(sorted(int(depths[index]) for index in np.flatnonzero(word > 1)))
    return (
        int(word.size),
        int(attachment_count),
        canonical_forest,
        int(depths.max()),
        leaves,
        branches,
        int(word.max()),
        int(cycle_rank),
    )


def topology_signature_from_attempt(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Return one complete role-topology signature, or ``None`` for a failed attempt."""

    if row.get("valid") is not True:
        return None
    program = row.get("program")
    topology = row.get("sampled_topology")
    if not isinstance(program, Mapping) or not isinstance(topology, Mapping):
        raise UgiTopologyDiversityError("valid attempt omits its program or sampled topology")
    expected_program_fields = {
        "attachment_counts",
        "cycle_ranks",
        "junction_budgets",
        "node_counts",
    }
    if set(program) != expected_program_fields:
        raise UgiTopologyDiversityError("attempt morphology program fields changed")
    offspring_by_role = topology.get("offspring_by_role")
    attachments = program.get("attachment_counts")
    cycles = program.get("cycle_ranks")
    if (
        not isinstance(offspring_by_role, Sequence)
        or isinstance(offspring_by_role, (str, bytes))
        or not isinstance(attachments, Sequence)
        or not isinstance(cycles, Sequence)
        or len(offspring_by_role) != len(ROLE_NAMES)
        or len(attachments) != len(ROLE_NAMES)
        or len(cycles) != len(ROLE_NAMES)
    ):
        raise UgiTopologyDiversityError("attempt role topology dimensions changed")
    return tuple(
        role_topology_signature(
            offspring_by_role[index],
            attachment_count=int(attachments[index]),
            cycle_rank=int(cycles[index]),
        )
        for index in range(len(ROLE_NAMES))
    )


def summarize_topology_diversity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report attempt-weighted complete and role-local topology concentration."""

    if not rows:
        raise UgiTopologyDiversityError("topology-diversity assessment is empty")
    signatures = [topology_signature_from_attempt(row) for row in rows]
    valid = [value for value in signatures if value is not None]
    role_values = {role: [value[index] for value in valid] for index, role in enumerate(ROLE_NAMES)}
    return {
        "attempts": len(rows),
        "valid_topologies": len(valid),
        "unique_complete_topology_signatures": len(set(valid)),
        "unique_complete_topology_signatures_per_attempt": len(set(valid)) / len(rows),
        "effective_complete_topology_count": _effective_count(valid),
        "role_effective_topology_counts": {
            role: _effective_count(values) for role, values in role_values.items()
        },
        "role_unique_topology_signatures": {
            role: len(set(values)) for role, values in role_values.items()
        },
        "identity_fields_used": False,
        "invalid_attempts_retained_in_denominator": True,
    }


__all__ = [
    "UgiTopologyDiversityError",
    "role_topology_signature",
    "summarize_topology_diversity",
    "topology_signature_from_attempt",
]
