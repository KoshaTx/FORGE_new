"""Grammar-normalized Ugi topology supervision for a frozen graph Transformer.

The shared model supplies contextual node representations.  A small residual head scores child
counts and closure endpoints, while this module normalizes those scores only over complete
role-local topologies admitted by the frozen Ugi morphology grammar.  Product or component
identifiers never enter the objective.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

from forge.model.reaction_program_transformer import synthesis_program_offspring_targets
from forge.model.ugi_closure_placement import feasible_next_closures
from forge.model.ugi_morphology_program import (
    UgiMorphologyProgramError,
    attached_forest_structured_nll,
)
from forge.model.ugi_transformer_topology import UgiTransformerTopologyPolicy
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


class UgiStructuredTopologyError(ValueError):
    """Structured Ugi topology supervision violates its frozen support."""


def _role_training_rows(
    clean: Mapping[str, Any],
    offspring_targets: Any,
    *,
    role_indices: Mapping[str, int],
    policy: UgiTransformerTopologyPolicy,
) -> tuple[
    dict[tuple[int, int, int, int, int], list[tuple[int, Any, Any]]],
    list[dict[str, Any]],
]:
    """Collect same-signature tree batches and cyclic closure records."""

    if torch is None:
        raise UgiStructuredTopologyError("structured Ugi topology supervision requires torch")
    required = {
        "node_mask",
        "role_states",
        "core_position_states",
        "component_instance_states",
        "role_morphology_states",
        "closure_left",
        "closure_right",
        "closure_mask",
    }
    if not required.issubset(clean):
        raise UgiStructuredTopologyError("structured topology batch is missing graph semantics")
    if set(role_indices) != set(ROLE_NAMES) or len(set(role_indices.values())) != len(ROLE_NAMES):
        raise UgiStructuredTopologyError("Ugi role indices are incomplete or ambiguous")

    node_mask = clean["node_mask"].detach().cpu().numpy().astype(bool)
    role_states = clean["role_states"].detach().cpu().numpy()
    core_states = clean["core_position_states"].detach().cpu().numpy()
    components = clean["component_instance_states"].detach().cpu().numpy()
    morphology = clean["role_morphology_states"].detach().cpu().numpy()
    closure_left = clean["closure_left"].detach().cpu().numpy()
    closure_right = clean["closure_right"].detach().cpu().numpy()
    closure_mask = clean["closure_mask"].detach().cpu().numpy().astype(bool)

    groups: dict[tuple[int, int, int, int, int], list[tuple[int, Any, Any]]] = defaultdict(list)
    closure_rows: list[dict[str, Any]] = []
    batch_size = int(node_mask.shape[0])
    for batch_index in range(batch_size):
        for role_position, role in enumerate(ROLE_NAMES):
            role_index = int(role_indices[role])
            exterior = np.flatnonzero(
                node_mask[batch_index]
                & (role_states[batch_index] == role_index)
                & (core_states[batch_index] == 1)
                & (components[batch_index] > 0)
            ).astype(np.int64)
            if exterior.size < 1:
                raise UgiStructuredTopologyError(f"training row has no {role} exterior")
            values = np.unique(morphology[batch_index, exterior], axis=0)
            if values.shape != (1, 4) or np.any(values[0] < 1):
                raise UgiStructuredTopologyError(
                    f"training row has inconsistent {role} morphology states"
                )
            node_count, junction_budget, cycle_rank, attachment_count = (
                int(value) - 1 for value in values[0]
            )
            if node_count != int(exterior.size):
                raise UgiStructuredTopologyError(f"{role} node-count condition changed")
            branch_limit = int(policy.maximum_adjacent_branch_run_by_role[role_position])
            index_tensor = torch.as_tensor(
                exterior, dtype=torch.long, device=offspring_targets.device
            )
            target = offspring_targets[batch_index].index_select(0, index_tensor)
            key = (
                role_position,
                node_count,
                junction_budget,
                attachment_count,
                branch_limit,
            )
            groups[key].append((batch_index, index_tensor, target))

            if cycle_rank:
                local_by_global = {
                    int(global_index): local_index
                    for local_index, global_index in enumerate(exterior.tolist())
                }
                target_edges: list[tuple[int, int]] = []
                target_slots: list[int] = []
                for slot in np.flatnonzero(closure_mask[batch_index]).tolist():
                    left = int(closure_left[batch_index, slot])
                    right = int(closure_right[batch_index, slot])
                    if left in local_by_global and right in local_by_global:
                        target_edges.append(
                            tuple(sorted((local_by_global[left], local_by_global[right])))
                        )
                        target_slots.append(int(slot))
                if len(target_edges) != cycle_rank or len(set(target_edges)) != cycle_rank:
                    raise UgiStructuredTopologyError(
                        f"{role} closure targets do not match the conditioned cycle rank"
                    )
                closure_rows.append(
                    {
                        "batch_index": batch_index,
                        "role_position": role_position,
                        "offspring": target,
                        "exterior": exterior,
                        "attachment_count": attachment_count,
                        "target_edges": tuple(target_edges),
                        "target_slots": tuple(target_slots),
                    }
                )
    return dict(groups), closure_rows


def _structured_closure_loss(
    predictions: Mapping[str, Any],
    closure_rows: list[dict[str, Any]],
    *,
    policy: UgiTransformerTopologyPolicy,
) -> tuple[Any, int]:
    left_logits = predictions["structured_closure_left"]
    right_logits = predictions["structured_closure_right"]
    losses = []
    decisions = 0
    for row in closure_rows:
        offspring_tensor = row["offspring"]
        offspring = offspring_tensor.detach().cpu().numpy().astype(np.int64)
        exterior = row["exterior"]
        target_slots = tuple(int(slot) for slot in row["target_slots"])
        target_edges = frozenset(row["target_edges"])

        def complete_target_paths(
            slot_index: int,
            selected: tuple[tuple[int, int], ...],
            remaining: frozenset[tuple[int, int]],
        ) -> list[Any]:
            if slot_index == len(target_slots):
                return [left_logits.new_zeros(())] if not remaining else []
            slot = target_slots[slot_index]
            candidates = feasible_next_closures(
                offspring,
                selected=selected,
                remaining_closures_including_next=len(remaining),
                allowed_ring_sizes=policy.allowed_ring_sizes,
                maximum_heavy_degree=policy.maximum_heavy_degree,
                attachment_count=int(row["attachment_count"]),
            )
            if not candidates.edges:
                return []
            edge_scores = []
            edge_to_index = {}
            for candidate_index, (local_left, local_right) in enumerate(candidates.edges):
                global_left = int(exterior[local_left])
                global_right = int(exterior[local_right])
                direct = (
                    left_logits[row["batch_index"], slot, global_left]
                    + right_logits[row["batch_index"], slot, global_right]
                )
                reverse = (
                    left_logits[row["batch_index"], slot, global_right]
                    + right_logits[row["batch_index"], slot, global_left]
                )
                edge_scores.append(torch.logsumexp(torch.stack((direct, reverse)), dim=0))
                edge_to_index[(local_left, local_right)] = candidate_index
            log_probabilities = torch.log_softmax(torch.stack(edge_scores), dim=0)
            paths = []
            for edge in sorted(remaining):
                candidate_index = edge_to_index.get(edge)
                if candidate_index is None:
                    continue
                suffixes = complete_target_paths(
                    slot_index + 1,
                    (*selected, edge),
                    remaining - {edge},
                )
                paths.extend(log_probabilities[candidate_index] + suffix for suffix in suffixes)
            return paths

        path_scores = complete_target_paths(0, (), target_edges)
        if not path_scores:
            raise UgiStructuredTopologyError(
                "training closure set lies outside the declared sequential ring support"
            )
        decisions += len(target_slots)
        losses.append(-torch.logsumexp(torch.stack(path_scores), dim=0) / len(target_slots))
    if not losses:
        return left_logits.new_zeros(()), 0
    return torch.stack(losses).mean(), decisions


def ugi_structured_topology_loss(
    predictions: Mapping[str, Any],
    clean: Mapping[str, Any],
    *,
    role_indices: Mapping[str, int],
    policy: UgiTransformerTopologyPolicy,
    tree_weight: float,
    closure_weight: float,
) -> tuple[Any, dict[str, Any]]:
    """Train only the Ugi structured head under the exact constructive support."""

    if torch is None:
        raise UgiStructuredTopologyError("structured Ugi topology supervision requires torch")
    if tree_weight <= 0.0 or closure_weight < 0.0:
        raise UgiStructuredTopologyError("structured topology loss weights are invalid")
    required = {
        "structured_offspring",
        "structured_closure_left",
        "structured_closure_right",
    }
    if not required.issubset(predictions):
        raise UgiStructuredTopologyError("model has no structured Ugi topology outputs")
    maximum_children = int(predictions["structured_offspring"].shape[-1]) - 1
    offspring_targets, _ = synthesis_program_offspring_targets(
        clean, maximum_children=maximum_children
    )
    groups, closure_rows = _role_training_rows(
        clean,
        offspring_targets,
        role_indices=role_indices,
        policy=policy,
    )
    group_losses = []
    tree_records = 0
    for (_, _, junction_budget, attachment_count, branch_limit), rows in sorted(groups.items()):
        local_logits = torch.stack(
            [
                predictions["structured_offspring"][batch_index].index_select(0, indices)
                for batch_index, indices, _ in rows
            ]
        )
        local_targets = torch.stack([target for _, _, target in rows])
        try:
            losses = attached_forest_structured_nll(
                local_logits,
                local_targets,
                junction_budget=junction_budget,
                attachment_count=attachment_count,
                maximum_adjacent_branch_run=branch_limit,
            )
        except UgiMorphologyProgramError as error:
            raise UgiStructuredTopologyError(str(error)) from error
        group_losses.append(losses.sum())
        tree_records += len(rows)
    if tree_records < 1:
        raise UgiStructuredTopologyError("structured topology batch contains no Ugi trees")
    tree_loss = torch.stack(group_losses).sum() / tree_records
    closure_loss, closure_decisions = _structured_closure_loss(
        predictions, closure_rows, policy=policy
    )
    total = tree_weight * tree_loss + closure_weight * closure_loss
    return total, {
        "structured_tree_nll": tree_loss.detach(),
        "structured_closure_nll": closure_loss.detach(),
        "structured_tree_records": tree_loss.new_tensor(tree_records, dtype=torch.int64),
        "structured_closure_decisions": tree_loss.new_tensor(closure_decisions, dtype=torch.int64),
        "structured_topology_total": total.detach(),
    }


__all__ = [
    "UgiStructuredTopologyError",
    "ugi_structured_topology_loss",
]
