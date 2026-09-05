"""Exact Ugi topology coupling for the shared reaction-program Transformer.

The Transformer supplies local child-count and closure-endpoint scores.  This module conditions
those scores on the coarse Ugi program and returns one exact feasible topology.  It never selects a
component, copies a fragment, repairs an accepted molecule, or changes the requested program.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord
from forge.model.ugi_all_role_semantic_program import UgiAllRoleSemanticTarget
from forge.model.ugi_amine_semantic_program import (
    UgiAmineSemanticTarget,
    amine_local_substitution_metrics,
)
from forge.model.ugi_closure_placement import feasible_next_closures
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_mog_semantic_guidance import (
    UgiMogSemanticGuidancePolicy,
    replace_amine_semantics,
    replace_directional_ester_semantics,
)
from forge.model.ugi_morphology_program import (
    UgiMorphologyProgram,
    UgiMorphologyProgramError,
    enumerate_attached_offspring_with_exact_budget,
    preorder_attached_forest_to_parents,
    sample_attached_offspring_with_exact_budget,
    sample_attached_offspring_with_exact_budget_and_cycle_rank,
)
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


UGI_PROGRAM_ID = "ugi_3cr_agile"


class UgiTransformerTopologyError(ValueError):
    """A Transformer topology request violates the frozen Ugi program contract."""


@dataclass(frozen=True)
class UgiTransformerTopologyPolicy:
    """Training-supported structural bounds used by the exact conditional decoder."""

    allowed_ring_sizes: tuple[int, ...]
    maximum_heavy_degree: int
    maximum_adjacent_branch_run_by_role: tuple[int, int, int]

    def __post_init__(self) -> None:
        if (
            not self.allowed_ring_sizes
            or tuple(sorted(set(self.allowed_ring_sizes))) != self.allowed_ring_sizes
            or self.allowed_ring_sizes[0] < 3
            or self.maximum_heavy_degree < 2
            or len(self.maximum_adjacent_branch_run_by_role) != len(ROLE_NAMES)
            or any(value < 0 for value in self.maximum_adjacent_branch_run_by_role)
        ):
            raise UgiTransformerTopologyError("invalid exact Ugi topology policy")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> UgiTransformerTopologyPolicy:
        branch = value.get("maximum_adjacent_branch_run_by_role")
        if not isinstance(branch, Mapping) or set(branch) != set(ROLE_NAMES):
            raise UgiTransformerTopologyError(
                "Ugi topology policy must name every precursor role exactly once"
            )
        raw_ring_sizes = value.get("allowed_ring_sizes")
        if not isinstance(raw_ring_sizes, Sequence) or isinstance(raw_ring_sizes, (str, bytes)):
            raise UgiTransformerTopologyError("Ugi topology ring-size support is missing")
        return cls(
            allowed_ring_sizes=tuple(sorted(set(int(item) for item in raw_ring_sizes))),
            maximum_heavy_degree=int(value.get("maximum_heavy_degree", -1)),
            maximum_adjacent_branch_run_by_role=tuple(int(branch[role]) for role in ROLE_NAMES),
        )

    @classmethod
    def from_support_documents(
        cls,
        closure_config: Mapping[str, Any],
        morphology_config: Mapping[str, Any],
    ) -> UgiTransformerTopologyPolicy:
        """Read topology support from the two frozen Ugi training contracts."""

        try:
            support = closure_config["support"]
            branch = morphology_config["sampling"]["maximum_adjacent_branch_run_by_role"]
            return cls.from_mapping(
                {
                    "allowed_ring_sizes": support["ring_sizes"],
                    "maximum_heavy_degree": support["maximum_heavy_degree"],
                    "maximum_adjacent_branch_run_by_role": branch,
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise UgiTransformerTopologyError(
                "frozen Ugi topology support documents are malformed"
            ) from error

    def to_mapping(self) -> dict[str, Any]:
        return {
            "allowed_ring_sizes": list(self.allowed_ring_sizes),
            "maximum_heavy_degree": self.maximum_heavy_degree,
            "maximum_adjacent_branch_run_by_role": dict(
                zip(ROLE_NAMES, self.maximum_adjacent_branch_run_by_role, strict=True)
            ),
        }


@dataclass(frozen=True)
class UgiExactTopology:
    """One exact global sparse topology plus its role-local offspring words."""

    parents: np.ndarray
    closure_left: np.ndarray
    closure_right: np.ndarray
    offspring_by_role: tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class UgiLocalSemanticTopology:
    """One complete local topology satisfying an identity-free semantic coordinate."""

    offspring: np.ndarray
    closures: tuple[tuple[int, int], ...]


def _component_graph_diameter(
    offspring: np.ndarray,
    closures: Sequence[Sequence[int]],
    *,
    attachment_count: int,
) -> int:
    """Return node-count diameter after adding the implicit precursor handle atom."""

    parents = preorder_attached_forest_to_parents(offspring, attachment_count=attachment_count)
    virtual_core = len(parents)
    adjacency = [set() for _ in range(len(parents) + 1)]
    for child, parent in enumerate(parents.tolist()):
        target = virtual_core if parent < 0 else int(parent)
        adjacency[child].add(target)
        adjacency[target].add(child)
    for raw_left, raw_right in closures:
        left, right = int(raw_left), int(raw_right)
        adjacency[left].add(right)
        adjacency[right].add(left)
    maximum_edges = 0
    for start in range(len(adjacency)):
        distances = {start: 0}
        queue = [start]
        for node in queue:
            for neighbor in adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        maximum_edges = max(maximum_edges, max(distances.values()))
    return maximum_edges + 1


def _semantic_topology_adjacency(
    offspring: np.ndarray,
    closures: Sequence[Sequence[int]],
    *,
    attachment_count: int,
) -> tuple[tuple[frozenset[int], ...], int]:
    """Return the precursor graph with the adapter-fixed amine N as a virtual node."""

    parents = preorder_attached_forest_to_parents(offspring, attachment_count=attachment_count)
    virtual_core = len(parents)
    adjacency = [set() for _ in range(virtual_core + 1)]
    for child, parent in enumerate(parents.tolist()):
        target = virtual_core if parent < 0 else int(parent)
        adjacency[child].add(target)
        adjacency[target].add(child)
    for raw_left, raw_right in closures:
        left, right = int(raw_left), int(raw_right)
        adjacency[left].add(right)
        adjacency[right].add(left)
    return tuple(frozenset(neighbors) for neighbors in adjacency), virtual_core


def _induced_node_diameter(adjacency: Sequence[Sequence[int]], nodes: frozenset[int]) -> int:
    """Return node-count diameter of an induced, potentially disconnected subgraph."""

    if not nodes:
        return 0
    maximum_edges = 0
    for start in nodes:
        distances = {start: 0}
        queue = [start]
        for current in queue:
            for neighbor in adjacency[current]:
                if neighbor in nodes and neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        maximum_edges = max(maximum_edges, max(distances.values()))
    return maximum_edges + 1


def _amine_semantic_symbol_assignments(
    topology: UgiLocalSemanticTopology,
    *,
    target: UgiAmineSemanticTarget,
    attachment_count: int,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate exact identity-free element assignments supported by one topology.

    The implicit reaction-core atom is the precursor's amine nitrogen.  Exterior N/O positions are
    enumerated without consulting any component or fragment identity.  The returned tuple is in
    exterior-node order followed by the implicit core N.
    """

    exterior_count = len(topology.offspring)
    if (
        target.nitrogen_atoms + target.oxygen_atoms > exterior_count + 1
        or target.nitrogen_atoms < 1
    ):
        return ()
    exterior_nitrogens = target.nitrogen_atoms - 1
    exterior_oxygens = target.oxygen_atoms
    exterior_carbons = exterior_count - exterior_nitrogens - exterior_oxygens
    if exterior_carbons < 0:
        return ()
    adjacency, virtual_core = _semantic_topology_adjacency(
        topology.offspring,
        topology.closures,
        attachment_count=attachment_count,
    )
    local_degrees = tuple(len(neighbors) for neighbors in adjacency)
    if (
        target.heavy_branch_atoms is not None
        and sum(degree >= 3 for degree in local_degrees) != target.heavy_branch_atoms
    ):
        return ()
    exterior = tuple(range(exterior_count))
    output: list[tuple[str, ...]] = []
    for nitrogen_nodes in combinations(exterior, exterior_nitrogens):
        nitrogen_set = frozenset((*nitrogen_nodes, virtual_core))
        remaining = tuple(node for node in exterior if node not in nitrogen_set)
        for oxygen_nodes in combinations(remaining, exterior_oxygens):
            oxygen_set = frozenset(oxygen_nodes)
            carbon_set = frozenset(
                node for node in exterior if node not in nitrogen_set and node not in oxygen_set
            )
            if len(carbon_set) != exterior_carbons:
                continue
            if any(not 1 <= len(adjacency[node]) <= 3 for node in nitrogen_set):
                continue
            if any(not 1 <= len(adjacency[node]) <= 2 for node in oxygen_set):
                continue
            if _induced_node_diameter(adjacency, carbon_set) != target.carbon_skeleton_diameter:
                continue
            reactive_sites = sum(1 <= len(adjacency[node]) <= 2 for node in nitrogen_set)
            if not 1 <= reactive_sites <= 2:
                continue
            symbols = tuple(
                "N" if node in nitrogen_set else "O" if node in oxygen_set else "C"
                for node in range(exterior_count + 1)
            )
            if target.hydrogen_bond_donors is not None:
                donors, branches = amine_local_substitution_metrics(symbols, local_degrees)
                if donors != target.hydrogen_bond_donors or branches != target.heavy_branch_atoms:
                    continue
            output.append(symbols)
    return tuple(output)


def _semantic_topology_tree_path(
    topology: UgiLocalSemanticTopology,
    *,
    left: int,
    right: int,
    attachment_count: int,
) -> tuple[int, ...]:
    """Return the unique path in the local tree, including the implicit core node."""

    parents = preorder_attached_forest_to_parents(
        topology.offspring, attachment_count=attachment_count
    )
    virtual_core = len(parents)
    adjacency = [set() for _ in range(virtual_core + 1)]
    for child, parent in enumerate(parents.tolist()):
        target = virtual_core if parent < 0 else int(parent)
        adjacency[child].add(target)
        adjacency[target].add(child)
    previous: dict[int, int | None] = {left: None}
    queue = [left]
    for node in queue:
        if node == right:
            break
        for neighbor in sorted(adjacency[node]):
            if neighbor not in previous:
                previous[neighbor] = node
                queue.append(neighbor)
    if right not in previous:
        raise UgiTransformerTopologyError("semantic topology tree is disconnected")
    path = [right]
    while previous[path[-1]] is not None:
        path.append(int(previous[path[-1]]))
    return tuple(reversed(path))


def _amine_topology_support_context(
    topology: UgiLocalSemanticTopology,
    *,
    attachment_count: int,
) -> tuple[tuple[int, ...], dict[int, int], dict[int, set[int]], frozenset[int]]:
    """Describe one complete rooted head shape in the train-reference coordinate system."""

    adjacency, virtual_core = _semantic_topology_adjacency(
        topology.offspring,
        topology.closures,
        attachment_count=attachment_count,
    )
    nodes = tuple(range(len(topology.offspring)))
    depths = {virtual_core: 0}
    queue = [virtual_core]
    for current in queue:
        for neighbor in sorted(adjacency[current]):
            if neighbor not in depths:
                depths[neighbor] = depths[current] + 1
                queue.append(neighbor)
    if any(node not in depths for node in nodes):
        raise UgiTransformerTopologyError("amine topology is disconnected from its core anchor")
    ring_nodes: set[int] = set()
    for left, right in topology.closures:
        ring_nodes.update(
            _semantic_topology_tree_path(
                topology,
                left=int(left),
                right=int(right),
                attachment_count=attachment_count,
            )
        )
    return (
        nodes,
        {node: depths[node] for node in nodes},
        {node: set(adjacency[node]) for node in range(len(adjacency))},
        frozenset(ring_nodes).intersection(nodes),
    )


def _assignment_has_local_chemistry_support(
    topology: UgiLocalSemanticTopology,
    symbols: Sequence[str],
    *,
    attachment_count: int,
    support: LocalChemistrySupport,
    program_id: str,
    role: str,
) -> bool:
    """Apply the same train-fold local edge, triangle and cycle support used at decode time."""

    adjacency, _ = _semantic_topology_adjacency(
        topology.offspring,
        topology.closures,
        attachment_count=attachment_count,
    )
    if len(symbols) != len(adjacency):
        raise UgiTransformerTopologyError("semantic atom assignment changed size")
    if any(
        not support.allows_role_edge_for_any_bond(
            program_id,
            role,
            symbols[left],
            role,
            symbols[right],
        )
        for left, neighbors in enumerate(adjacency)
        for right in neighbors
        if left < right
    ):
        return False
    triangles = (
        (left, middle, right)
        for left in range(len(adjacency))
        for middle in range(left + 1, len(adjacency))
        for right in range(middle + 1, len(adjacency))
        if middle in adjacency[left] and right in adjacency[left] and right in adjacency[middle]
    )
    if any(
        not support.allows_role_triangle(
            program_id,
            ((role, symbols[node]) for node in triangle),
        )
        for triangle in triangles
    ):
        return False
    if support.enforces_role_cycles and any(
        not support.allows_role_cycle(
            program_id,
            (
                (role, symbols[node])
                for node in _semantic_topology_tree_path(
                    topology,
                    left=int(left),
                    right=int(right),
                    attachment_count=attachment_count,
                )
            ),
        )
        for left, right in topology.closures
    ):
        return False
    return True


def amine_semantic_topology_supports_target(
    topology: UgiLocalSemanticTopology,
    *,
    target: UgiAmineSemanticTarget,
    attachment_count: int,
    local_chemistry_support: LocalChemistrySupport | None = None,
    program_id: str = UGI_PROGRAM_ID,
    role: str = "amine_head",
) -> bool:
    """Check exact semantic, registry-handle and optional train-fold chemistry support.

    With ``local_chemistry_support``, this is the complete topology-side feasibility condition used
    by terminal decoding.  It conditions one topology draw on nonempty chemistry support; it does
    not retry, repair, inspect a completed molecule, or select a stored component.
    """

    assignments = _amine_semantic_symbol_assignments(
        topology,
        target=target,
        attachment_count=attachment_count,
    )
    if local_chemistry_support is None:
        return bool(assignments)
    exterior_count = len(topology.offspring)
    if not local_chemistry_support.component_is_within_observed_support(
        program_id,
        role,
        heavy_atoms=exterior_count + 1,
        carbon_atoms=(exterior_count + 1 - target.nitrogen_atoms - target.oxygen_atoms),
        heteroatoms=(target.nitrogen_atoms + target.oxygen_atoms),
    ):
        return False
    return any(
        _assignment_has_local_chemistry_support(
            topology,
            symbols,
            attachment_count=attachment_count,
            support=local_chemistry_support,
            program_id=program_id,
            role=role,
        )
        for symbols in assignments
    )


def _enumerate_closure_sets(
    offspring: np.ndarray,
    *,
    cycle_rank: int,
    attachment_count: int,
    allowed_ring_sizes: Sequence[int],
    maximum_heavy_degree: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if cycle_rank == 0:
        return ((),)
    output: list[tuple[tuple[int, int], ...]] = []

    def visit(selected: tuple[tuple[int, int], ...]) -> None:
        if len(selected) == cycle_rank:
            output.append(selected)
            return
        candidates = feasible_next_closures(
            offspring,
            selected=selected,
            remaining_closures_including_next=cycle_rank - len(selected),
            attachment_count=attachment_count,
            allowed_ring_sizes=allowed_ring_sizes,
            maximum_heavy_degree=maximum_heavy_degree,
        )
        for edge in candidates.edges:
            normalized = tuple(sorted((int(edge[0]), int(edge[1]))))
            if selected and normalized <= selected[-1]:
                continue
            visit((*selected, normalized))

    visit(())
    return tuple(output)


def enumerate_amine_semantic_topologies(
    *,
    node_count: int,
    junction_budget: int,
    cycle_rank: int,
    attachment_count: int,
    target: UgiAmineSemanticTarget,
    maximum_children: int,
    policy: UgiTransformerTopologyPolicy,
    allowed_ring_sizes: Sequence[int] | None = None,
    local_chemistry_support: LocalChemistrySupport | None = None,
    maximum_enumerated_nodes: int = 10,
) -> tuple[UgiLocalSemanticTopology, ...]:
    """Enumerate complete small-head support matching one measured semantic target."""

    if not 1 <= target.heavy_atom_graph_diameter <= node_count + 1:
        raise UgiTransformerTopologyError("amine semantic diameter is outside graph support")
    role_index = ROLE_NAMES.index("amine_head")
    try:
        words = enumerate_attached_offspring_with_exact_budget(
            node_count=node_count,
            junction_budget=junction_budget,
            maximum_children=maximum_children,
            attachment_count=attachment_count,
            maximum_adjacent_branch_run=(policy.maximum_adjacent_branch_run_by_role[role_index]),
            maximum_enumerated_nodes=maximum_enumerated_nodes,
        )
    except UgiMorphologyProgramError as error:
        raise UgiTransformerTopologyError(str(error)) from error
    ring_sizes = policy.allowed_ring_sizes if allowed_ring_sizes is None else allowed_ring_sizes
    output: list[UgiLocalSemanticTopology] = []
    for offspring in words:
        for closures in _enumerate_closure_sets(
            offspring,
            cycle_rank=cycle_rank,
            attachment_count=attachment_count,
            allowed_ring_sizes=ring_sizes,
            maximum_heavy_degree=policy.maximum_heavy_degree,
        ):
            topology = UgiLocalSemanticTopology(offspring=offspring.copy(), closures=closures)
            if (
                _component_graph_diameter(offspring, closures, attachment_count=attachment_count)
                != target.heavy_atom_graph_diameter
            ):
                continue
            if amine_semantic_topology_supports_target(
                topology,
                target=target,
                attachment_count=attachment_count,
                local_chemistry_support=local_chemistry_support,
            ):
                output.append(topology)
    return tuple(output)


def _sample_amine_semantic_topology(
    *,
    logits: Any,
    target: UgiAmineSemanticTarget,
    junction_budget: int,
    cycle_rank: int,
    attachment_count: int,
    fixed_roots: Sequence[int],
    exterior: np.ndarray,
    closure_left_logits: np.ndarray,
    closure_right_logits: np.ndarray,
    first_slot: int,
    policy: UgiTransformerTopologyPolicy,
    allowed_ring_sizes: Sequence[int] | None,
    local_chemistry_support: LocalChemistrySupport | None,
    semantic_guidance_policy: UgiMogSemanticGuidancePolicy | None,
    program: UgiMorphologyProgram,
    all_role_target: UgiAllRoleSemanticTarget | None,
    generator: Any,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Draw once from the neural topology law conditioned on measured head semantics."""

    target_band = (
        (target,)
        if semantic_guidance_policy is None
        else semantic_guidance_policy.amine_targets_within_band(target)
    )
    by_key: dict[
        tuple[tuple[int, ...], tuple[tuple[int, int], ...]],
        tuple[UgiLocalSemanticTopology, float, float | None],
    ] = {}
    if (
        semantic_guidance_policy is not None
        and semantic_guidance_policy.uses_joint_realism
        and all_role_target is None
    ):
        raise UgiTransformerTopologyError(
            "joint-realism topology guidance requires a complete all-role target"
        )
    for candidate_target in target_band:
        distance = (
            0.0
            if semantic_guidance_policy is None
            else semantic_guidance_policy.amine_distance(candidate_target, target)
        )
        for candidate in enumerate_amine_semantic_topologies(
            node_count=int(logits.shape[0]),
            junction_budget=junction_budget,
            cycle_rank=cycle_rank,
            attachment_count=attachment_count,
            target=candidate_target,
            maximum_children=int(logits.shape[1]) - 1,
            policy=policy,
            allowed_ring_sizes=allowed_ring_sizes,
            local_chemistry_support=local_chemistry_support,
        ):
            key = (
                tuple(int(value) for value in candidate.offspring.tolist()),
                tuple((int(left), int(right)) for left, right in candidate.closures),
            )
            joint_score = (
                None
                if semantic_guidance_policy is None
                or not semantic_guidance_policy.uses_joint_realism
                else semantic_guidance_policy.joint_realism_scores(
                    program,
                    (replace_amine_semantics(all_role_target, candidate_target),),
                )[0]
            )
            previous = by_key.get(key)
            if previous is None or (distance, -(joint_score or 0.0)) < (
                previous[1],
                -(previous[2] or 0.0),
            ):
                by_key[key] = (candidate, distance, joint_score)
    ordered = tuple(by_key[key] for key in sorted(by_key))
    candidates = tuple(candidate for candidate, _, _ in ordered)
    semantic_distances = tuple(distance for _, distance, _ in ordered)
    joint_realism_scores = (
        None
        if semantic_guidance_policy is None or not semantic_guidance_policy.uses_joint_realism
        else tuple(float(score) for _, _, score in ordered if score is not None)
    )
    topology_support_scores = None
    if (
        semantic_guidance_policy is not None
        and semantic_guidance_policy.whole_head_topology_support
    ):
        topology_support_scores = tuple(
            semantic_guidance_policy.amine_head_topology_support_score(
                nodes=context[0],
                depths_by_node=context[1],
                neighbors=context[2],
                ring_nodes=context[3],
            )
            for context in (
                _amine_topology_support_context(
                    candidate,
                    attachment_count=attachment_count,
                )
                for candidate in candidates
            )
        )
    if not candidates:
        raise UgiTransformerTopologyError(
            "coarse amine program has no joint semantic and local-chemistry support"
        )
    scores = logits.to(torch.float64)
    positions = torch.arange(int(logits.shape[0]), device=scores.device)
    candidate_scores: list[Any] = []
    oriented_edges: list[tuple[list[int], list[int]]] = []
    for candidate in candidates:
        local_parents = preorder_attached_forest_to_parents(
            candidate.offspring, attachment_count=attachment_count
        )
        permutation = _root_aligned_permutation(local_parents, fixed_roots)
        score = scores[
            positions,
            torch.as_tensor(candidate.offspring, dtype=torch.long, device=scores.device),
        ].sum()
        left_output: list[int] = []
        right_output: list[int] = []
        for local_slot, (left, right) in enumerate(candidate.closures):
            global_left = int(exterior[int(permutation[left])])
            global_right = int(exterior[int(permutation[right])])
            slot = first_slot + local_slot
            direct = float(
                closure_left_logits[slot, global_left] + closure_right_logits[slot, global_right]
            )
            reverse = float(
                closure_left_logits[slot, global_right] + closure_right_logits[slot, global_left]
            )
            if reverse > direct:
                global_left, global_right = global_right, global_left
                score = score + reverse
            else:
                score = score + direct
            left_output.append(global_left)
            right_output.append(global_right)
        candidate_scores.append(score)
        oriented_edges.append((left_output, right_output))
    if semantic_guidance_policy is None:
        probabilities = torch.stack(candidate_scores).softmax(dim=0)
    else:
        probabilities = torch.as_tensor(
            semantic_guidance_policy.probabilities(
                [float(value) for value in candidate_scores],
                semantic_distances,
                joint_realism_scores,
                topology_support_scores,
                local_chemistry_rank_weight=(
                    semantic_guidance_policy.effective_whole_head_topology_rank_weight
                ),
            ),
            dtype=torch.float64,
        )
    selected = int(torch.multinomial(probabilities, 1, generator=generator).item())
    left, right = oriented_edges[selected]
    return candidates[selected].offspring.copy(), left, right


def _role_targets(record: SynthesisProgramGraphRecord) -> dict[str, tuple[int, int, int, int]]:
    states = record.role_morphology_states
    if states is None:
        raise UgiTransformerTopologyError("exact Ugi topology requires role morphology states")
    output: dict[str, tuple[int, int, int, int]] = {}
    for block in record.component_blocks:
        if block.role not in ROLE_NAMES:
            continue
        values = np.unique(states[block.start : block.stop], axis=0)
        if values.shape != (1, 4) or np.any(values[0] < 1):
            raise UgiTransformerTopologyError(f"role morphology is not constant for {block.role}")
        output[block.role] = tuple(int(item) - 1 for item in values[0])
    if set(output) != set(ROLE_NAMES):
        raise UgiTransformerTopologyError("Ugi layout does not contain all precursor roles")
    return output


def _root_aligned_permutation(
    parents: np.ndarray,
    fixed_root_positions: Sequence[int],
) -> np.ndarray:
    """Map forest roots onto anonymous fixed attachment slots while preserving parent order."""

    roots = np.flatnonzero(parents < 0).tolist()
    fixed_roots = [int(value) for value in fixed_root_positions]
    if len(roots) != len(fixed_roots) or len(set(fixed_roots)) != len(fixed_roots):
        raise UgiTransformerTopologyError("decoded and adapter-fixed attachment counts disagree")
    node_count = len(parents)
    remaining_source = [index for index in range(node_count) if index not in set(roots)]
    remaining_target = [index for index in range(node_count) if index not in set(fixed_roots)]
    permutation = np.empty(node_count, dtype=np.int64)
    for source, target in zip(roots, fixed_roots, strict=True):
        permutation[source] = target
    for source, target in zip(remaining_source, remaining_target, strict=True):
        permutation[source] = target
    for child, parent in enumerate(parents.tolist()):
        if parent >= 0 and int(permutation[parent]) >= int(permutation[child]):
            raise UgiTransformerTopologyError("root alignment would violate sparse parent ordering")
    return permutation


def _select_role_closures(
    *,
    offspring: np.ndarray,
    attachment_count: int,
    cycle_rank: int,
    exterior: np.ndarray,
    permutation: np.ndarray,
    closure_left_logits: np.ndarray,
    closure_right_logits: np.ndarray,
    first_slot: int,
    policy: UgiTransformerTopologyPolicy,
    allowed_ring_sizes: tuple[int, ...] | None = None,
) -> tuple[list[int], list[int]]:
    selected: list[tuple[int, int]] = []
    oriented: list[tuple[int, int]] = []
    for local_slot in range(cycle_rank):
        candidates = feasible_next_closures(
            offspring,
            selected=selected,
            remaining_closures_including_next=cycle_rank - local_slot,
            allowed_ring_sizes=(
                policy.allowed_ring_sizes if allowed_ring_sizes is None else allowed_ring_sizes
            ),
            maximum_heavy_degree=policy.maximum_heavy_degree,
            attachment_count=attachment_count,
        )
        if not candidates.edges:
            raise UgiTransformerTopologyError(
                "exact Ugi tree cannot realize its declared cycle rank"
            )
        slot = first_slot + local_slot
        scored: list[tuple[float, int, int, int, int]] = []
        for left, right in candidates.edges:
            global_left = int(exterior[int(permutation[left])])
            global_right = int(exterior[int(permutation[right])])
            direct = float(
                closure_left_logits[slot, global_left] + closure_right_logits[slot, global_right]
            )
            reverse = float(
                closure_left_logits[slot, global_right] + closure_right_logits[slot, global_left]
            )
            if reverse > direct:
                scored.append((reverse, -left, -right, global_right, global_left))
            else:
                scored.append((direct, -left, -right, global_left, global_right))
        _, negative_left, negative_right, global_left, global_right = max(scored)
        selected.append((-negative_left, -negative_right))
        oriented.append((global_left, global_right))
    return [left for left, _ in oriented], [right for _, right in oriented]


def decode_ugi_exact_topology(
    predictions: Mapping[str, Any],
    *,
    index: int,
    record: SynthesisProgramGraphRecord,
    policy: UgiTransformerTopologyPolicy,
    generator: Any,
    ester_chemotype_policy: UgiEsterChemotypePolicy | None = None,
    amine_semantic_target: UgiAmineSemanticTarget | None = None,
    all_role_semantic_target: UgiAllRoleSemanticTarget | None = None,
    local_chemistry_support: LocalChemistrySupport | None = None,
    semantic_guidance_policy: UgiMogSemanticGuidancePolicy | None = None,
) -> UgiExactTopology:
    """Condition one Transformer endpoint on an exact feasible Ugi morphology program."""

    if torch is None or record.program_id != UGI_PROGRAM_ID:
        raise UgiTransformerTopologyError("exact coupled topology currently supports Ugi only")
    if amine_semantic_target is not None and all_role_semantic_target is not None:
        raise UgiTransformerTopologyError(
            "amine-only and all-role semantic targets are mutually exclusive"
        )
    effective_amine_target = (
        all_role_semantic_target.amine
        if all_role_semantic_target is not None
        else amine_semantic_target
    )
    offspring_field = (
        "structured_offspring" if "structured_offspring" in predictions else "offspring"
    )
    closure_left_field = (
        "structured_closure_left" if "structured_closure_left" in predictions else "closure_left"
    )
    closure_right_field = (
        "structured_closure_right" if "structured_closure_right" in predictions else "closure_right"
    )
    if offspring_field not in predictions:
        raise UgiTransformerTopologyError("Transformer checkpoint has no offspring topology head")
    targets = _role_targets(record)
    program = UgiMorphologyProgram(
        node_counts=tuple(targets[role][0] for role in ROLE_NAMES),
        junction_budgets=tuple(targets[role][1] for role in ROLE_NAMES),
        cycle_ranks=tuple(targets[role][2] for role in ROLE_NAMES),
        attachment_counts=tuple(targets[role][3] for role in ROLE_NAMES),
    )
    parents = record.graph.parents.copy()
    closure_left = np.zeros(record.graph.closure_count, dtype=np.int64)
    closure_right = np.zeros(record.graph.closure_count, dtype=np.int64)
    offspring_rows: list[np.ndarray] = []
    closure_cursor = 0
    block_by_role = {
        block.role: block for block in record.component_blocks if block.role in ROLE_NAMES
    }
    for role_index, role in enumerate(ROLE_NAMES):
        block = block_by_role[role]
        exterior = np.flatnonzero(
            (np.arange(record.node_count) >= block.start)
            & (np.arange(record.node_count) < block.stop)
            & (record.core_position_states == 1)
        ).astype(np.int64)
        node_count, junction_budget, cycle_rank, attachment_count = targets[role]
        if len(exterior) != node_count or node_count < 1:
            raise UgiTransformerTopologyError(f"{role} exterior size changed")
        fixed_roots = [
            local
            for local, node in enumerate(exterior.tolist())
            if bool(record.fixed_parent_bond_mask[node])
            and int(record.core_position_states[int(record.graph.parents[node])]) > 1
        ]
        if len(fixed_roots) != attachment_count:
            raise UgiTransformerTopologyError(f"{role} attachment contract changed")
        logits = predictions[offspring_field][index, exterior].detach().to("cpu")
        require_ester_topology = (
            ester_chemotype_policy is not None
            and record.program_id == ester_chemotype_policy.reaction_id
            and role == ester_chemotype_policy.aldehyde_role
        )
        minimum_exterior = (
            ester_chemotype_policy.minimum_topology_exterior_atoms(role)
            if ester_chemotype_policy is not None
            else 1
        )
        maximum_exterior = (
            ester_chemotype_policy.maximum_exterior_atoms(role)
            if ester_chemotype_policy is not None
            else node_count
        )
        if not minimum_exterior <= node_count <= maximum_exterior:
            raise UgiTransformerTopologyError(
                f"{role} exterior size is outside the requested chemotype support"
            )
        if require_ester_topology and (
            junction_budget != 1 or cycle_rank != 0 or attachment_count != 1
        ):
            raise UgiTransformerTopologyError(
                "requested ester chemotype is incompatible with the sampled morphology program"
            )
        semantic_closures: tuple[list[int], list[int]] | None = None
        if effective_amine_target is not None and role == "amine_head":
            offspring, semantic_left, semantic_right = _sample_amine_semantic_topology(
                logits=logits,
                target=effective_amine_target,
                junction_budget=junction_budget,
                cycle_rank=cycle_rank,
                attachment_count=attachment_count,
                fixed_roots=fixed_roots,
                exterior=exterior,
                closure_left_logits=predictions[closure_left_field][index]
                .detach()
                .to("cpu")
                .numpy(),
                closure_right_logits=predictions[closure_right_field][index]
                .detach()
                .to("cpu")
                .numpy(),
                first_slot=closure_cursor,
                policy=policy,
                allowed_ring_sizes=(
                    ester_chemotype_policy.allowed_amine_cycle_sizes(node_count)
                    if ester_chemotype_policy is not None and cycle_rank > 0
                    else None
                ),
                local_chemistry_support=local_chemistry_support,
                semantic_guidance_policy=semantic_guidance_policy,
                program=program,
                all_role_target=all_role_semantic_target,
                generator=generator,
            )
            semantic_closures = (semantic_left, semantic_right)
        elif require_ester_topology:
            try:
                offspring = _sample_constructive_ester_offspring(
                    logits,
                    generator=generator,
                    minimum_side_carbons=(ester_chemotype_policy.minimum_ester_side_carbons),
                    minimum_long_side_carbons=(
                        ester_chemotype_policy.minimum_ester_long_side_carbons
                    ),
                    exact_full_side_carbons=(
                        None
                        if all_role_semantic_target is None
                        else (
                            all_role_semantic_target.tail_pair.aldehyde_ester_short_side_carbons,
                            all_role_semantic_target.tail_pair.aldehyde_ester_long_side_carbons,
                        )
                    ),
                    exact_alkoxy_handle_and_acyl_side_carbons=(
                        None
                        if all_role_semantic_target is None
                        or all_role_semantic_target.tail_pair.aldehyde_alkoxy_handle_side_carbons
                        == 0
                        else (
                            all_role_semantic_target.tail_pair.aldehyde_alkoxy_handle_side_carbons,
                            all_role_semantic_target.tail_pair.aldehyde_acyl_side_carbons,
                        )
                    ),
                    semantic_guidance_policy=semantic_guidance_policy,
                    program=program,
                    all_role_target=all_role_semantic_target,
                )
            except UgiMorphologyProgramError as error:
                raise UgiTransformerTopologyError(str(error)) from error
        else:
            try:
                if cycle_rank:
                    offspring = sample_attached_offspring_with_exact_budget_and_cycle_rank(
                        logits,
                        junction_budget=junction_budget,
                        cycle_rank=cycle_rank,
                        attachment_count=attachment_count,
                        generator=generator,
                        allowed_ring_sizes=policy.allowed_ring_sizes,
                        maximum_heavy_degree=policy.maximum_heavy_degree,
                        maximum_adjacent_branch_run=(
                            policy.maximum_adjacent_branch_run_by_role[role_index]
                        ),
                    )
                else:
                    offspring = sample_attached_offspring_with_exact_budget(
                        logits,
                        junction_budget=junction_budget,
                        attachment_count=attachment_count,
                        generator=generator,
                        maximum_adjacent_branch_run=(
                            policy.maximum_adjacent_branch_run_by_role[role_index]
                        ),
                    )
            except UgiMorphologyProgramError as error:
                raise UgiTransformerTopologyError(str(error)) from error
        local_parents = preorder_attached_forest_to_parents(
            offspring,
            attachment_count=attachment_count,
        )
        permutation = _root_aligned_permutation(local_parents, fixed_roots)
        for source_child, source_parent in enumerate(local_parents.tolist()):
            target_child = int(exterior[int(permutation[source_child])])
            if source_parent < 0:
                if not bool(record.fixed_parent_bond_mask[target_child]):
                    raise UgiTransformerTopologyError("decoded root lost its fixed core attachment")
                continue
            if bool(record.fixed_parent_bond_mask[target_child]):
                raise UgiTransformerTopologyError("decoded non-root overlaps a fixed attachment")
            target_parent = int(exterior[int(permutation[source_parent])])
            if target_parent >= target_child:
                raise UgiTransformerTopologyError("decoded Ugi parent does not precede its child")
            parents[target_child] = target_parent
        if semantic_closures is None:
            local_left, local_right = _select_role_closures(
                offspring=offspring,
                attachment_count=attachment_count,
                cycle_rank=cycle_rank,
                exterior=exterior,
                permutation=permutation,
                closure_left_logits=predictions[closure_left_field][index]
                .detach()
                .to("cpu")
                .numpy(),
                closure_right_logits=predictions[closure_right_field][index]
                .detach()
                .to("cpu")
                .numpy(),
                first_slot=closure_cursor,
                policy=policy,
                allowed_ring_sizes=(
                    ester_chemotype_policy.allowed_amine_cycle_sizes(node_count)
                    if ester_chemotype_policy is not None
                    and role == ester_chemotype_policy.amine_role
                    and cycle_rank > 0
                    else None
                ),
            )
        else:
            local_left, local_right = semantic_closures
        if cycle_rank:
            closure_left[closure_cursor : closure_cursor + cycle_rank] = local_left
            closure_right[closure_cursor : closure_cursor + cycle_rank] = local_right
        closure_cursor += cycle_rank
        offspring_rows.append(offspring)
    if closure_cursor != record.graph.closure_count:
        raise UgiTransformerTopologyError("role cycle ranks do not conserve closure slots")
    return UgiExactTopology(
        parents=parents,
        closure_left=closure_left,
        closure_right=closure_right,
        offspring_by_role=tuple(offspring_rows),  # type: ignore[arg-type]
    )


def _sample_constructive_ester_offspring(
    logits: Any,
    *,
    generator: Any,
    minimum_side_carbons: int,
    minimum_long_side_carbons: int,
    exact_full_side_carbons: tuple[int, int] | None = None,
    exact_alkoxy_handle_and_acyl_side_carbons: tuple[int, int] | None = None,
    semantic_guidance_policy: UgiMogSemanticGuidancePolicy | None = None,
    program: UgiMorphologyProgram | None = None,
    all_role_target: UgiAllRoleSemanticTarget | None = None,
) -> np.ndarray:
    """Sample exactly from one-junction trees that can host the requested ester chemotype.

    A one-root, one-junction exterior contains one bifurcation and otherwise only chains. An
    ester-capable member has a one-node carbonyl-oxygen branch; the other child is a chain. We
    enumerate every preorder word with that property, retain the words satisfying the two carbon
    arm floors, and draw once under the Transformer's factorized offspring probabilities. This is
    exact conditional sampling on the declared topology support, not rejection or repair.
    """

    if torch is None or logits.ndim != 2 or logits.shape[1] < 3:
        raise UgiMorphologyProgramError(
            "constructive ester topology requires [nodes, at least three child classes] logits"
        )
    node_count = int(logits.shape[0])
    # The two precursor carbon arms share the ester cut and include the fixed
    # aldehyde-derived reaction-core carbon, which is absent from this exterior.
    # The exterior does contain the two ester oxygens, hence C_short + C_long + 1.
    required_nodes = int(minimum_side_carbons + minimum_long_side_carbons + 1)
    if node_count < required_nodes:
        raise UgiMorphologyProgramError(
            "aldehyde exterior is too small for the requested ester carbon-arm support"
        )

    if semantic_guidance_policy is not None and (
        exact_full_side_carbons is None or exact_alkoxy_handle_and_acyl_side_carbons is None
    ):
        raise UgiMorphologyProgramError(
            "semantic ester guidance requires one complete directional target"
        )
    raw_candidates = _enumerate_constructive_ester_offspring(
        node_count=node_count,
        minimum_side_carbons=minimum_side_carbons,
        minimum_long_side_carbons=minimum_long_side_carbons,
        exact_full_side_carbons=(
            exact_full_side_carbons if semantic_guidance_policy is None else None
        ),
        exact_alkoxy_handle_and_acyl_side_carbons=(
            exact_alkoxy_handle_and_acyl_side_carbons if semantic_guidance_policy is None else None
        ),
    )
    candidates: list[np.ndarray] = []
    semantic_distances: list[float] = []
    joint_realism_scores: list[float] = []
    if semantic_guidance_policy is None:
        candidates.extend(raw_candidates)
        semantic_distances.extend(0.0 for _ in raw_candidates)
    else:
        assert exact_alkoxy_handle_and_acyl_side_carbons is not None
        allowed = set(
            semantic_guidance_policy.aldehyde_directional_pairs_within_band(
                alkoxy_handle_carbons=exact_alkoxy_handle_and_acyl_side_carbons[0],
                acyl_carbons=exact_alkoxy_handle_and_acyl_side_carbons[1],
            )
        )
        allowed = {
            pair
            for pair in allowed
            if min(pair) >= minimum_side_carbons and max(pair) >= minimum_long_side_carbons
        }
        for candidate in raw_candidates:
            supported = _full_directional_ester_side_carbon_counts_for_tree(
                preorder_attached_forest_to_parents(candidate, attachment_count=1)
            )
            eligible = sorted(allowed.intersection(supported))
            if not eligible:
                continue
            candidates.append(candidate)
            semantic_distances.append(
                min(
                    semantic_guidance_policy.aldehyde_distance(
                        pair, exact_alkoxy_handle_and_acyl_side_carbons
                    )
                    for pair in eligible
                )
            )
            if semantic_guidance_policy.uses_joint_realism:
                if program is None or all_role_target is None:
                    raise UgiMorphologyProgramError(
                        "joint-realism ester guidance requires a complete program and target"
                    )
                scores = semantic_guidance_policy.joint_realism_scores(
                    program,
                    tuple(
                        replace_directional_ester_semantics(all_role_target, pair)
                        for pair in eligible
                    ),
                )
                assert scores is not None
                joint_realism_scores.append(max(scores))
    if not candidates:
        raise UgiMorphologyProgramError(
            "constructive ester topology has no support for the requested carbon arms"
        )

    scores = logits.to(torch.float64)
    positions = torch.arange(node_count, device=scores.device)
    candidate_scores = torch.stack(
        [
            scores[positions, torch.as_tensor(value, device=scores.device)].sum()
            for value in candidates
        ]
    )
    if semantic_guidance_policy is None:
        probabilities = candidate_scores.softmax(dim=0)
    else:
        probabilities = torch.as_tensor(
            semantic_guidance_policy.probabilities(
                [float(value) for value in candidate_scores],
                semantic_distances,
                (joint_realism_scores if semantic_guidance_policy.uses_joint_realism else None),
            ),
            dtype=torch.float64,
        )
    selected = int(torch.multinomial(probabilities, 1, generator=generator).item())
    return candidates[selected]


def _enumerate_constructive_ester_offspring(
    *,
    node_count: int,
    minimum_side_carbons: int,
    minimum_long_side_carbons: int,
    exact_full_side_carbons: tuple[int, int] | None = None,
    exact_alkoxy_handle_and_acyl_side_carbons: tuple[int, int] | None = None,
) -> tuple[np.ndarray, ...]:
    """Enumerate all one-junction exterior trees supporting the requested ester arms."""

    candidates: list[np.ndarray] = []
    for prefix_nodes in range(1, node_count - 2):
        chain_nodes = node_count - prefix_nodes - 2
        if chain_nodes < 2:
            continue
        chain = [1] * (chain_nodes - 1) + [0]
        for first_child, second_child in (([0], chain), (chain, [0])):
            offspring = np.asarray(
                [1] * prefix_nodes + [2] + first_child + second_child,
                dtype=np.int64,
            )
            parents = preorder_attached_forest_to_parents(offspring, attachment_count=1)
            if (
                _has_ester_capable_tree(
                    parents,
                    minimum_side_carbons=minimum_side_carbons,
                    minimum_long_side_carbons=minimum_long_side_carbons,
                )
                and (
                    exact_full_side_carbons is None
                    or exact_full_side_carbons in _full_ester_side_carbon_counts_for_tree(parents)
                )
                and (
                    exact_alkoxy_handle_and_acyl_side_carbons is None
                    or exact_alkoxy_handle_and_acyl_side_carbons
                    in _full_directional_ester_side_carbon_counts_for_tree(parents)
                )
            ):
                candidates.append(offspring)
    return tuple(candidates)


def _full_ester_side_carbon_counts_for_tree(
    parents: np.ndarray,
) -> frozenset[tuple[int, int]]:
    """Return precursor-level ester carbon-arm counts realizable by one exterior tree.

    The exterior tree omits the aldehyde carbonyl oxygen and keeps the aldehyde carbon as a fixed
    reaction-core atom.  The latter is represented here as one virtual carbon attached to the
    tree root, so the returned counts match the recovered precursor rather than the product-only
    exterior.
    """

    roots = np.flatnonzero(parents < 0).tolist()
    if len(roots) != 1:
        return frozenset()
    neighbors = [set() for _ in range(len(parents))]
    for child, parent in enumerate(parents.tolist()):
        if parent < 0:
            continue
        neighbors[child].add(parent)
        neighbors[parent].add(child)
    output: set[tuple[int, int]] = set()
    root = int(roots[0])

    def complete_degree(node: int) -> int:
        # The local exterior omits the fixed reaction-core neighbour of its root.  Ester motif
        # roles are classified on the complete product graph, so account for that neighbour here.
        return len(neighbors[node]) + int(node == root)

    for center, adjacent in enumerate(neighbors):
        if complete_degree(center) != 3:
            continue
        leaves = [node for node in adjacent if complete_degree(node) == 1]
        bridges = [node for node in adjacent if complete_degree(node) == 2]
        for leaf in leaves:
            for bridge in bridges:
                alkoxy = next(iter(neighbors[bridge] - {center}), -1)
                substituents = adjacent - {leaf, bridge}
                if alkoxy < 0 or len(substituents) != 1 or alkoxy in substituents:
                    continue
                blocked = frozenset((center, bridge))

                def component(start: int) -> set[int]:
                    visited = {start}
                    frontier = [start]
                    while frontier:
                        node = frontier.pop()
                        for target in neighbors[node]:
                            if frozenset((node, target)) == blocked or target in visited:
                                continue
                            visited.add(target)
                            frontier.append(target)
                    return visited

                sides = (component(center), component(bridge))
                counts = [len(sides[0]) - 1, len(sides[1]) - 1]
                counts[0 if root in sides[0] else 1] += 1
                output.add(tuple(sorted((int(counts[0]), int(counts[1])))))
    return frozenset(output)


def _full_directional_ester_side_carbon_counts_for_tree(
    parents: np.ndarray,
) -> frozenset[tuple[int, int]]:
    """Return alkoxy-handle/acyl carbon counts supported by one exterior tree.

    The attachment root is adjacent to the fixed aldehyde-derived reaction-core carbon.  A tuple
    is emitted only when that root lies on the alkoxy side of the ester, matching the directional
    arrangement observed in the measured AGILE-type training aldehydes.
    """

    roots = np.flatnonzero(parents < 0).tolist()
    if len(roots) != 1:
        return frozenset()
    root = int(roots[0])
    neighbors = [set() for _ in range(len(parents))]
    for child, parent in enumerate(parents.tolist()):
        if parent < 0:
            continue
        neighbors[child].add(parent)
        neighbors[parent].add(child)

    def complete_degree(node: int) -> int:
        return len(neighbors[node]) + int(node == root)

    output: set[tuple[int, int]] = set()
    for center, adjacent in enumerate(neighbors):
        if complete_degree(center) != 3:
            continue
        leaves = [node for node in adjacent if complete_degree(node) == 1]
        bridges = [node for node in adjacent if complete_degree(node) == 2]
        for leaf in leaves:
            for bridge in bridges:
                alkoxy = next(iter(neighbors[bridge] - {center}), -1)
                substituents = adjacent - {leaf, bridge}
                if alkoxy < 0 or len(substituents) != 1 or alkoxy in substituents:
                    continue
                blocked = frozenset((center, bridge))

                def component(start: int) -> set[int]:
                    visited = {start}
                    frontier = [start]
                    while frontier:
                        node = frontier.pop()
                        for target in neighbors[node]:
                            if frozenset((node, target)) == blocked or target in visited:
                                continue
                            visited.add(target)
                            frontier.append(target)
                    return visited

                acyl_side = component(center)
                alkoxy_side = component(bridge)
                if root not in alkoxy_side:
                    continue
                acyl_carbons = len(acyl_side) - 1
                alkoxy_handle_carbons = len(alkoxy_side) - 1 + 1
                output.add((int(alkoxy_handle_carbons), int(acyl_carbons)))
    return frozenset(output)


def _has_ester_capable_tree(
    parents: np.ndarray,
    *,
    minimum_side_carbons: int,
    minimum_long_side_carbons: int,
) -> bool:
    """Return whether one tree can host a precursor-level C(=O)-O-C motif.

    The local exterior omits one fixed aldehyde-derived reaction-core carbon.  Reuse the exact
    precursor-level counter so topology support and terminal chemistry cannot disagree by one atom.
    """

    return any(
        short_side >= minimum_side_carbons and long_side >= minimum_long_side_carbons
        for short_side, long_side in _full_ester_side_carbon_counts_for_tree(parents)
    )


__all__ = [
    "UGI_PROGRAM_ID",
    "UgiExactTopology",
    "UgiLocalSemanticTopology",
    "UgiTransformerTopologyError",
    "UgiTransformerTopologyPolicy",
    "decode_ugi_exact_topology",
    "enumerate_amine_semantic_topologies",
    "_enumerate_constructive_ester_offspring",
    "_full_directional_ester_side_carbon_counts_for_tree",
    "_full_ester_side_carbon_counts_for_tree",
]
