"""Support-constrained graph and Ugi terminal assignments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np

from forge.model._synthesis_sampling.contracts import SynthesisProgramSamplingError
from forge.model.defog_feasibility import AtomState
from forge.model.local_chemistry_support import LocalChemistrySupport, tree_path_indices
from forge.model.qualified_vocabulary import QualifiedAtomVocabulary
from forge.model.sparse_topology_feasibility import BOND_VALENCE_UNITS, _maximum_valence_units
from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord
from forge.model.ugi_all_role_semantic_program import UgiAllRoleSemanticTarget
from forge.model.ugi_amine_semantic_program import (
    UgiAmineSemanticTarget,
    amine_local_substitution_metrics,
)
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_mog_semantic_guidance import (
    UgiMogSemanticGuidancePolicy,
    replace_amine_semantics,
    replace_directional_ester_semantics,
    replace_tail_unsaturation_semantics,
)
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.model.ugi_role_chemistry_prior import UgiRoleChemistryPrior
from forge.model.ugi_transformer_topology import UGI_PROGRAM_ID, UgiTransformerTopologyPolicy
from forge.potency.annotations import ROLE_NAMES

_ATOM_CAPACITY_CACHE: dict[int, tuple[Sequence[AtomState], np.ndarray]] = {}
_BOND_UNIT_CACHE: dict[int, np.ndarray] = {}


def _available_valence_units(state: AtomState) -> int:
    return _maximum_valence_units(state) - 2 * int(state.explicit_hydrogens)


def _atom_capacity_table(atom_vocabulary: Sequence[AtomState]) -> np.ndarray:
    """Return the frozen per-state valence capacities for one atom vocabulary.

    The table is a pure function of the vocabulary, but the strict decoder rebuilt it once per
    decoded record.  The vocabulary is held by reference in the cache value so its ``id`` cannot be
    recycled onto a different vocabulary while the entry is live.
    """

    entry = _ATOM_CAPACITY_CACHE.get(id(atom_vocabulary))
    if entry is None:
        table = np.asarray(
            (
                atom_vocabulary.capacities()
                if isinstance(atom_vocabulary, QualifiedAtomVocabulary)
                else [_available_valence_units(state) for state in atom_vocabulary]
            ),
            dtype=np.int64,
        )
        table.setflags(write=False)
        entry = (atom_vocabulary, table)
        _ATOM_CAPACITY_CACHE[id(atom_vocabulary)] = entry
    return entry[1]


def _bond_unit_table(bond_classes: int) -> np.ndarray:
    """Return bond valence units on the host without a per-record device transfer.

    ``BOND_VALENCE_UNITS`` is a small constant torch tensor.  Reading it per decoded record forced
    a device-to-host copy for every sample in the batch, which on an accelerator is a full
    synchronization each time.  It is the same four numbers on every call.
    """

    table = _BOND_UNIT_CACHE.get(bond_classes)
    if table is None:
        table = BOND_VALENCE_UNITS[:bond_classes].cpu().numpy().astype(np.int64)
        table.setflags(write=False)
        _BOND_UNIT_CACHE[bond_classes] = table
    return table


def _argmax_allowed(logits: np.ndarray, valid: np.ndarray) -> int | None:
    if logits.ndim != 1 or valid.shape != logits.shape or not np.any(valid):
        return None
    masked = np.where(valid, logits, -np.inf)
    return int(np.argmax(masked))


def _sample_allowed(
    logits: np.ndarray,
    valid: np.ndarray,
    *,
    generator: np.random.Generator,
    temperature: float,
) -> int | None:
    """Draw once from model probabilities restricted to the declared support."""

    if (
        logits.ndim != 1
        or valid.shape != logits.shape
        or not np.any(valid)
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        return None
    allowed = np.flatnonzero(valid)
    scaled = np.asarray(logits[allowed], dtype=np.float64) / temperature
    scaled -= np.max(scaled)
    probabilities = np.exp(scaled)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0:
        return None
    probabilities /= total
    return int(generator.choice(allowed, p=probabilities))


def _sample_mog_chemistry_allowed(
    logits: np.ndarray,
    valid: np.ndarray,
    local_scores: np.ndarray,
    *,
    policy: UgiMogSemanticGuidancePolicy,
    generator: np.random.Generator,
    rank_weight: float | None = None,
) -> int | None:
    """Draw one hard-supported chemistry state from the complete MOG rank law."""

    allowed = np.flatnonzero(valid)
    if allowed.size == 0:
        return None
    model_scores = np.asarray(logits, dtype=np.float64)[allowed]
    semantic_distances = np.zeros(allowed.size, dtype=np.float64)
    chemistry_scores = np.asarray(local_scores, dtype=np.float64)[allowed]
    joint_scores = np.zeros(allowed.size, dtype=np.float64) if policy.uses_joint_realism else None
    probabilities = (
        policy.atom_chemistry_probabilities(
            model_scores,
            semantic_distances,
            chemistry_scores,
            joint_scores,
        )
        if rank_weight is None
        else policy.chemistry_probabilities(
            model_scores,
            semantic_distances,
            chemistry_scores,
            joint_scores,
            rank_weight=rank_weight,
        )
    )
    return int(generator.choice(allowed, p=probabilities))


def _distances_to_reaction_core(
    neighbors: Sequence[set[int]], core_position_states: np.ndarray
) -> np.ndarray:
    """Return unweighted graph distance to the nearest reaction-core atom."""

    distances = np.full(len(neighbors), -1, dtype=np.int64)
    frontier = [int(node) for node in np.flatnonzero(core_position_states > 1)]
    if not frontier:
        return distances
    distances[frontier] = 0
    while frontier:
        node = frontier.pop(0)
        for target in neighbors[node]:
            if distances[target] >= 0:
                continue
            distances[target] = distances[node] + 1
            frontier.append(target)
    return distances


def _ugi_program_from_record(record: SynthesisProgramGraphRecord) -> UgiMorphologyProgram:
    """Recover the complete coarse Ugi program without reading component identities."""

    states = record.role_morphology_states
    if states is None:
        raise SynthesisProgramSamplingError(
            "joint-realism guidance requires explicit role morphology states"
        )
    values_by_role: dict[str, tuple[int, int, int, int]] = {}
    for role in ROLE_NAMES:
        blocks = [block for block in record.component_blocks if block.role == role]
        if len(blocks) != 1:
            raise SynthesisProgramSamplingError(
                "joint-realism guidance requires each Ugi role exactly once"
            )
        block = blocks[0]
        values = np.unique(states[block.start : block.stop], axis=0)
        if values.shape != (1, 4) or np.any(values[0] < 1):
            raise SynthesisProgramSamplingError(
                "joint-realism guidance encountered an invalid role morphology"
            )
        values_by_role[role] = tuple(int(value) - 1 for value in values[0])
    return UgiMorphologyProgram(
        node_counts=tuple(values_by_role[role][0] for role in ROLE_NAMES),
        junction_budgets=tuple(values_by_role[role][1] for role in ROLE_NAMES),
        cycle_ranks=tuple(values_by_role[role][2] for role in ROLE_NAMES),
        attachment_counts=tuple(values_by_role[role][3] for role in ROLE_NAMES),
    )


def _induced_graph_diameter(nodes: Sequence[int], neighbors: Sequence[set[int]]) -> int:
    """Return the number of vertices on the longest shortest path in one induced component."""

    selected = {int(node) for node in nodes}
    if not selected:
        return 0
    maximum_edges = 0
    for start in selected:
        distances = {start: 0}
        queue = [start]
        for current in queue:
            for neighbor in neighbors[current]:
                if neighbor in selected and neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        if len(distances) != len(selected):
            return 0
        maximum_edges = max(maximum_edges, max(distances.values()))
    return maximum_edges + 1


def _generated_ring_support(
    generated_cycles: Sequence[Sequence[int]],
) -> tuple[frozenset[int], frozenset[tuple[int, int]]]:
    """Return generated ring nodes and edges from tree paths closed by each closure."""

    nodes: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for raw_cycle in generated_cycles:
        cycle = tuple(int(node) for node in raw_cycle)
        if len(cycle) < 3:
            continue
        nodes.update(cycle)
        edges.update(tuple(sorted(pair)) for pair in zip(cycle[:-1], cycle[1:], strict=True))
        edges.add(tuple(sorted((cycle[0], cycle[-1]))))
    return frozenset(nodes), frozenset(edges)


def _ugi_ester_motif_constraints(
    predictions: Mapping[str, np.ndarray],
    index: int,
    record: SynthesisProgramGraphRecord,
    atom_vocabulary: Sequence[AtomState],
    parents: np.ndarray,
    closure_left: np.ndarray,
    closure_right: np.ndarray,
    neighbors: Sequence[set[int]],
    policy: UgiEsterChemotypePolicy,
    distances_to_core: np.ndarray | None = None,
    ring_nodes: frozenset[int] = frozenset(),
    ring_edges: frozenset[tuple[int, int]] = frozenset(),
    all_role_semantic_target: UgiAllRoleSemanticTarget | None = None,
    semantic_guidance_policy: UgiMogSemanticGuidancePolicy | None = None,
    terminal_generator: np.random.Generator | None = None,
) -> tuple[dict[int, str], dict[tuple[int, int], int]] | None:
    """Select the highest-scoring feasible C(=O)-O-C placement in the requested role."""

    if record.program_id != policy.reaction_id:
        return {}, {}
    blocks = [block for block in record.component_blocks if block.role == policy.aldehyde_role]
    if len(blocks) != 1:
        return None
    block = blocks[0]
    exterior = {
        node
        for node in range(block.start, block.stop)
        if int(record.core_position_states[node]) == 1 and not bool(record.fixed_atom_mask[node])
    }
    if len(exterior) < 5:
        return None
    attachment_roots = {
        node
        for node in exterior
        if bool(record.fixed_parent_bond_mask[node])
        and int(record.core_position_states[int(record.graph.parents[node])]) > 1
    }
    if len(attachment_roots) != 1:
        return None
    attachment_root = next(iter(attachment_roots))

    edge_logits: dict[tuple[int, int], np.ndarray] = {}
    edge_fixed: dict[tuple[int, int], bool] = {}
    for child in range(1, record.node_count):
        pair = tuple(sorted((child, int(parents[child]))))
        edge_logits[pair] = predictions["parent_bonds"][index, child]
        edge_fixed[pair] = bool(record.fixed_parent_bond_mask[child])
    for slot, (left, right) in enumerate(zip(closure_left, closure_right, strict=True)):
        pair = tuple(sorted((int(left), int(right))))
        edge_logits[pair] = predictions["closure_bonds"][index, slot]
        edge_fixed[pair] = bool(record.fixed_closure_bond_mask[slot])

    capacities = _atom_capacity_table(atom_vocabulary)

    def atom_score(node: int, symbol: str, required_units: int) -> float | None:
        eligible = [
            state
            for state, atom in enumerate(atom_vocabulary)
            if atom.symbol == symbol
            and atom.formal_charge == 0
            and not atom.aromatic
            and int(capacities[state]) >= required_units
        ]
        if not eligible:
            return None
        return max(float(predictions["nodes"][index, node, state]) for state in eligible)

    candidates: list[
        tuple[
            float,
            float,
            float | None,
            float | None,
            dict[int, str],
            dict[tuple[int, int], int],
        ]
    ] = []
    joint_program = (
        None
        if semantic_guidance_policy is None or not semantic_guidance_policy.uses_joint_realism
        else _ugi_program_from_record(record)
    )
    if joint_program is not None and all_role_semantic_target is None:
        return None
    if (
        joint_program is not None
        and all_role_semantic_target.tail_pair.aldehyde_alkoxy_handle_side_carbons == 0
    ):
        return None
    for center in sorted(exterior):
        center_neighbors = sorted(neighbors[center] & exterior)
        if len(center_neighbors) != 3:
            continue
        leaf_neighbors = [node for node in center_neighbors if len(neighbors[node]) == 1]
        bridge_neighbors = [node for node in center_neighbors if len(neighbors[node]) == 2]
        for carbonyl_oxygen in leaf_neighbors:
            for ester_oxygen in bridge_neighbors:
                alkoxy = next(iter(neighbors[ester_oxygen] - {center}), -1)
                carbon_substituents = [
                    node for node in center_neighbors if node not in {carbonyl_oxygen, ester_oxygen}
                ]
                if (
                    alkoxy not in exterior
                    or len(carbon_substituents) != 1
                    or len({center, carbonyl_oxygen, ester_oxygen, alkoxy, *carbon_substituents})
                    != 5
                ):
                    continue
                blocked = frozenset((center, ester_oxygen))

                def exterior_component(start: int) -> set[int]:
                    visited = {start}
                    frontier = [start]
                    while frontier:
                        node = frontier.pop()
                        for target in neighbors[node] & exterior:
                            if frozenset((node, target)) == blocked or target in visited:
                                continue
                            visited.add(target)
                            frontier.append(target)
                    return visited

                sides = (exterior_component(center), exterior_component(ester_oxygen))
                exterior_carbon_counts = [len(sides[0]) - 1, len(sides[1]) - 1]
                # The measured ester-side policy is defined on recovered aldehyde precursors.
                # The exterior omits the fixed aldehyde-derived reaction-core carbon, so restore
                # that carbon before applying either the measured support floor or the exact target.
                full_carbon_counts = exterior_carbon_counts.copy()
                full_carbon_counts[0 if attachment_root in sides[0] else 1] += 1
                carbon_counts = tuple(sorted(full_carbon_counts))
                if (
                    carbon_counts[0] < policy.minimum_ester_side_carbons
                    or carbon_counts[1] < policy.minimum_ester_long_side_carbons
                ):
                    continue
                requested_sides = (
                    None
                    if all_role_semantic_target is None
                    else (
                        all_role_semantic_target.tail_pair.aldehyde_ester_short_side_carbons,
                        all_role_semantic_target.tail_pair.aldehyde_ester_long_side_carbons,
                    )
                )
                if (
                    requested_sides is not None
                    and semantic_guidance_policy is None
                    and carbon_counts != requested_sides
                ):
                    continue
                requested_directional_sides = (
                    None
                    if all_role_semantic_target is None
                    or all_role_semantic_target.tail_pair.aldehyde_alkoxy_handle_side_carbons == 0
                    else (
                        all_role_semantic_target.tail_pair.aldehyde_alkoxy_handle_side_carbons,
                        all_role_semantic_target.tail_pair.aldehyde_acyl_side_carbons,
                    )
                )
                if requested_directional_sides is not None:
                    root_on_alkoxy_side = attachment_root in sides[1]
                    directional_sides = (
                        int(full_carbon_counts[1]),
                        int(full_carbon_counts[0]),
                    )
                    allowed_directional = (
                        {requested_directional_sides}
                        if semantic_guidance_policy is None
                        else set(
                            semantic_guidance_policy.aldehyde_directional_pairs_within_band(
                                alkoxy_handle_carbons=requested_directional_sides[0],
                                acyl_carbons=requested_directional_sides[1],
                            )
                        )
                    )
                    if not root_on_alkoxy_side or directional_sides not in allowed_directional:
                        continue
                    semantic_distance = (
                        0.0
                        if semantic_guidance_policy is None
                        else semantic_guidance_policy.aldehyde_distance(
                            directional_sides, requested_directional_sides
                        )
                    )
                else:
                    semantic_distance = 0.0
                carbon_substituent = carbon_substituents[0]
                forced_atoms = {
                    center: "C",
                    carbonyl_oxygen: "O",
                    ester_oxygen: "O",
                    alkoxy: "C",
                    carbon_substituent: "C",
                }
                forced_bonds = {
                    tuple(sorted((center, carbonyl_oxygen))): 1,
                    tuple(sorted((center, ester_oxygen))): 0,
                    tuple(sorted((ester_oxygen, alkoxy))): 0,
                    tuple(sorted((center, carbon_substituent))): 0,
                }
                if any(pair not in edge_logits or edge_fixed[pair] for pair in forced_bonds):
                    continue
                node_requirements = {
                    center: 8,
                    carbonyl_oxygen: 4,
                    ester_oxygen: 4,
                    alkoxy: 2 * len(neighbors[alkoxy]),
                    carbon_substituent: 2 * len(neighbors[carbon_substituent]),
                }
                scores = [
                    atom_score(node, symbol, node_requirements[node])
                    for node, symbol in forced_atoms.items()
                ]
                if any(value is None for value in scores):
                    continue
                score = sum(float(value) for value in scores if value is not None) + sum(
                    float(edge_logits[pair][bond]) for pair, bond in forced_bonds.items()
                )
                joint_score = (
                    None
                    if joint_program is None
                    else semantic_guidance_policy.joint_realism_scores(
                        joint_program,
                        (
                            replace_directional_ester_semantics(
                                all_role_semantic_target,
                                directional_sides,
                            ),
                        ),
                    )[0]
                )
                local_score = None
                if semantic_guidance_policy is not None and (
                    semantic_guidance_policy.uses_local_chemistry_bonds
                ):
                    chemistry_prior = semantic_guidance_policy.local_chemistry_prior
                    if chemistry_prior is None or distances_to_core is None:
                        return None
                    local_evidence: list[float] = []
                    if semantic_guidance_policy.uses_coordinate_local_chemistry_atoms:
                        for node, symbol in forced_atoms.items():
                            bias = semantic_guidance_policy.atom_local_scores(
                                role=policy.aldehyde_role,
                                depth=int(distances_to_core[node]),
                                degree=len(neighbors[node]),
                                in_ring=node in ring_nodes,
                                atom_vocabulary=atom_vocabulary,
                            )
                            eligible = [
                                state
                                for state, atom in enumerate(atom_vocabulary)
                                if atom.symbol == symbol
                                and atom.formal_charge == 0
                                and not atom.aromatic
                                and int(capacities[state]) >= node_requirements[node]
                            ]
                            if not eligible:
                                return None
                            local_evidence.append(max(float(bias[state]) for state in eligible))
                    for pair, bond in forced_bonds.items():
                        left, right = pair
                        local_evidence.append(
                            float(
                                semantic_guidance_policy.bond_local_scores(
                                    role=policy.aldehyde_role,
                                    depth=min(
                                        int(distances_to_core[left]),
                                        int(distances_to_core[right]),
                                    ),
                                    in_ring=pair in ring_edges,
                                    symbols=(forced_atoms[left], forced_atoms[right]),
                                    bond_classes=len(edge_logits[pair]),
                                )[bond]
                            )
                        )
                    local_score = semantic_guidance_policy.aggregate_local_scores(local_evidence)
                candidates.append(
                    (
                        score,
                        semantic_distance,
                        joint_score,
                        local_score,
                        forced_atoms,
                        forced_bonds,
                    )
                )
    if not candidates:
        return None
    if (
        semantic_guidance_policy is None
        or not semantic_guidance_policy.uses_ranked_terminal_chemistry
    ):
        selected = max(range(len(candidates)), key=lambda value: candidates[value][0])
    else:
        if terminal_generator is None:
            return None
        model_scores = [candidate[0] for candidate in candidates]
        semantic_distances = [candidate[1] for candidate in candidates]
        joint_scores = (
            [float(candidate[2]) for candidate in candidates]
            if semantic_guidance_policy.uses_joint_realism
            else None
        )
        local_scores = (
            [float(candidate[3]) for candidate in candidates]
            if semantic_guidance_policy.uses_local_chemistry_bonds
            else None
        )
        probabilities = (
            semantic_guidance_policy.bond_chemistry_probabilities(
                model_scores,
                semantic_distances,
                local_scores,
                joint_scores,
            )
            if local_scores is not None
            else semantic_guidance_policy.probabilities(
                model_scores,
                semantic_distances,
                joint_scores,
            )
        )
        selected = int(terminal_generator.choice(len(candidates), p=probabilities))
    candidate = candidates[selected]
    return candidate[4], candidate[5]


def _select_ugi_all_role_tail_bonds(
    predictions: Mapping[str, np.ndarray],
    index: int,
    record: SynthesisProgramGraphRecord,
    atom_vocabulary: Sequence[AtomState],
    node_states: np.ndarray,
    parents: np.ndarray,
    closure_left: np.ndarray,
    closure_right: np.ndarray,
    minimum_used: np.ndarray,
    capacities: np.ndarray,
    bond_units: np.ndarray,
    role_names: Sequence[str],
    local_chemistry_support: LocalChemistrySupport,
    existing_forced_bonds: Mapping[tuple[int, int], int],
    target: UgiAllRoleSemanticTarget,
    distances_to_core: np.ndarray | None = None,
    ring_edges: frozenset[tuple[int, int]] = frozenset(),
    semantic_guidance_policy: UgiMogSemanticGuidancePolicy | None = None,
    terminal_generator: np.random.Generator | None = None,
) -> tuple[dict[tuple[int, int], int] | None, str | None]:
    """Choose exact tail unsaturation globally under model scores and valence support.

    Every eligible C--C edge receives exactly one state in the returned map.  The search is small:
    measured Ugi tails contain at most two optional C--C unsaturations, so enumeration is bounded by
    pairs of edges rather than by the full bond-state product space.
    """

    if record.program_id != UGI_PROGRAM_ID:
        return dict(existing_forced_bonds), None
    if len(bond_units) < 3 or tuple(int(value) for value in bond_units[:3]) != (2, 4, 6):
        return None, "ugi_all_role_tail_bond_vocabulary_changed"

    # Distances from genuine hydrophobic termini complement the existing distance-to-core
    # coordinate.  A role attachment can be degree one in the induced role graph, so termini are
    # identified in the complete generated topology and must be carbon.  This excludes the
    # reaction-core attachment and the ester oxygen without using a component identity.
    adjacency = [set() for _ in range(record.node_count)]
    for child in range(1, record.node_count):
        parent = int(parents[child])
        if 0 <= parent < record.node_count:
            adjacency[parent].add(child)
            adjacency[child].add(parent)
    for left, right in zip(closure_left, closure_right, strict=True):
        left_index = int(left)
        right_index = int(right)
        if 0 <= left_index < record.node_count and 0 <= right_index < record.node_count:
            adjacency[left_index].add(right_index)
            adjacency[right_index].add(left_index)
    symbols = tuple(atom_vocabulary[int(state)].symbol for state in node_states)
    terminal_offsets = np.full(record.node_count, -1, dtype=np.int64)
    for role in sorted(set(role_names)):
        frontier = [
            node
            for node in range(record.node_count)
            if role_names[node] == role and symbols[node] == "C" and len(adjacency[node]) == 1
        ]
        for node in frontier:
            terminal_offsets[node] = 0
        cursor = 0
        while cursor < len(frontier):
            node = frontier[cursor]
            cursor += 1
            for neighbor in sorted(adjacency[node]):
                if role_names[neighbor] != role or terminal_offsets[neighbor] >= 0:
                    continue
                terminal_offsets[neighbor] = terminal_offsets[node] + 1
                frontier.append(neighbor)

    metadata: dict[tuple[int, int], tuple[str, int, np.ndarray]] = {}
    for child in range(1, record.node_count):
        if record.fixed_parent_bond_mask[child]:
            continue
        parent = int(parents[child])
        pair = tuple(sorted((parent, child)))
        metadata[pair] = ("parent", child, predictions["parent_bonds"][index, child])
    for slot, (left, right) in enumerate(zip(closure_left, closure_right, strict=True)):
        if record.fixed_closure_bond_mask[slot]:
            continue
        pair = tuple(sorted((int(left), int(right))))
        metadata[pair] = ("closure", slot, predictions["closure_bonds"][index, slot])

    desired = {
        "oxoester_aldehyde_body_tail": (
            int(target.tail_pair.aldehyde_carbon_carbon_double_bonds),
            int(target.tail_pair.aldehyde_carbon_carbon_triple_bonds),
        ),
        "isocyanide_tail": (
            int(target.tail_pair.isocyanide_carbon_carbon_double_bonds),
            int(target.tail_pair.isocyanide_carbon_carbon_triple_bonds),
        ),
    }
    selected_forced = dict(existing_forced_bonds)

    def assignment_is_feasible(candidate: Mapping[tuple[int, int], int]) -> bool:
        used = minimum_used.copy()
        for pair, bond in candidate.items():
            if pair not in metadata or bond < 0 or bond >= len(bond_units):
                return False
            left, right = pair
            extra = int(bond_units[bond]) - 2
            used[[left, right]] += extra
            if used[left] > capacities[left] or used[right] > capacities[right]:
                return False
            if not local_chemistry_support.allows_role_edge(
                record.program_id,
                role_names[left],
                symbols[left],
                bond,
                role_names[right],
                symbols[right],
            ):
                return False
        return True

    for role, (double_count, triple_count) in desired.items():
        eligible = tuple(
            pair
            for pair in sorted(metadata)
            if role_names[pair[0]] == role_names[pair[1]] == role
            and int(record.core_position_states[pair[0]]) == 1
            and int(record.core_position_states[pair[1]]) == 1
            and symbols[pair[0]] == symbols[pair[1]] == "C"
            and pair not in selected_forced
        )
        count_options = (
            ((double_count, triple_count),)
            if semantic_guidance_policy is None
            or (
                semantic_guidance_policy.tail_unsaturation_count_tolerance == 0
                and semantic_guidance_policy.tail_unsaturation_count_strategy
                not in {"frequency_resampled", "smoothed_frequency_resampled"}
            )
            else semantic_guidance_policy.tail_unsaturation_count_options(
                role=role,
                requested_double_count=double_count,
                requested_triple_count=triple_count,
            )
        )
        # score, semantic distance, local support, C=C count, C#C count, assignment
        candidates: list[
            tuple[float, float, float | None, int, int, dict[tuple[int, int], int]]
        ] = []
        for candidate_double_count, candidate_triple_count in count_options:
            if candidate_double_count + candidate_triple_count > len(eligible):
                continue
            semantic_distance = float(
                abs(candidate_double_count - double_count)
                + abs(candidate_triple_count - triple_count)
            )
            for double_edges in combinations(eligible, candidate_double_count):
                double_set = frozenset(double_edges)
                remaining = tuple(pair for pair in eligible if pair not in double_set)
                triple_edge_sets = combinations(remaining, candidate_triple_count)
                for triple_edges in triple_edge_sets:
                    triple_set = frozenset(triple_edges)
                    assignment = {
                        pair: (1 if pair in double_set else 2 if pair in triple_set else 0)
                        for pair in eligible
                    }
                    complete = {**selected_forced, **assignment}
                    if not assignment_is_feasible(complete):
                        continue
                    score = sum(float(metadata[pair][2][bond]) for pair, bond in assignment.items())
                    local_score = None
                    if semantic_guidance_policy is not None and (
                        semantic_guidance_policy.uses_local_chemistry_bonds
                    ):
                        chemistry_prior = semantic_guidance_policy.local_chemistry_prior
                        if chemistry_prior is None or distances_to_core is None:
                            return None, "ugi_local_chemistry_reference_unavailable"
                        scored_assignment = tuple(
                            (pair, bond)
                            for pair, bond in assignment.items()
                            if not (
                                semantic_guidance_policy.local_chemistry_unsaturation_position_only
                                and bond == 0
                            )
                        )
                        # A saturated assignment is itself measured-supported.  When only the
                        # unsaturation positions are scored there are no selected non-single edges,
                        # so it receives the strongest support tier rather than an empty minimum.
                        local_score = (
                            4.0
                            if not scored_assignment
                            else semantic_guidance_policy.aggregate_local_scores(
                                [
                                    float(
                                        semantic_guidance_policy.bond_local_scores(
                                            role=role,
                                            depth=min(
                                                int(distances_to_core[pair[0]]),
                                                int(distances_to_core[pair[1]]),
                                            ),
                                            in_ring=pair in ring_edges,
                                            symbols=(symbols[pair[0]], symbols[pair[1]]),
                                            bond_classes=len(bond_units),
                                            terminal_offset=(
                                                min(
                                                    int(terminal_offsets[pair[0]]),
                                                    int(terminal_offsets[pair[1]]),
                                                )
                                                if terminal_offsets[pair[0]] >= 0
                                                and terminal_offsets[pair[1]] >= 0
                                                else None
                                            ),
                                        )[bond]
                                    )
                                    for pair, bond in scored_assignment
                                ]
                            )
                        )
                        minimum_tier = (
                            semantic_guidance_policy.local_chemistry_unsaturation_minimum_support_tier
                        )
                        if (
                            minimum_tier is not None
                            and scored_assignment
                            and local_score < float(minimum_tier)
                        ):
                            continue
                    candidates.append(
                        (
                            score,
                            semantic_distance,
                            local_score,
                            candidate_double_count,
                            candidate_triple_count,
                            assignment,
                        )
                    )
        if not candidates:
            return None, f"ugi_all_role_tail_unsaturation_assignment_unavailable:{role}"
        if (
            semantic_guidance_policy is None
            or not semantic_guidance_policy.uses_ranked_terminal_bonds
        ):
            selected = max(range(len(candidates)), key=lambda value: candidates[value][0])
        else:
            if terminal_generator is None:
                return None, "ugi_all_role_tail_guidance_generator_unavailable"
            model_scores = [candidate[0] for candidate in candidates]
            semantic_distances = [candidate[1] for candidate in candidates]
            joint_scores = (
                [
                    float(
                        semantic_guidance_policy.joint_realism_scores(
                            _ugi_program_from_record(record),
                            (
                                replace_tail_unsaturation_semantics(
                                    target,
                                    role=role,
                                    double_count=candidate[3],
                                    triple_count=candidate[4],
                                ),
                            ),
                        )[0]
                    )
                    for candidate in candidates
                ]
                if semantic_guidance_policy.uses_joint_realism
                else None
            )
            local_scores = (
                [float(candidate[2]) for candidate in candidates]
                if semantic_guidance_policy.uses_local_chemistry_bonds
                else None
            )

            def candidate_terminal_pattern(
                candidate_index: int,
            ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
                double_offsets: list[int] = []
                triple_offsets: list[int] = []
                for pair, bond in candidates[candidate_index][5].items():
                    if bond not in {1, 2}:
                        continue
                    if terminal_offsets[pair[0]] < 0 or terminal_offsets[pair[1]] < 0:
                        return None
                    offset_value = min(
                        int(terminal_offsets[pair[0]]),
                        int(terminal_offsets[pair[1]]),
                    )
                    (double_offsets if bond == 1 else triple_offsets).append(offset_value)
                return tuple(sorted(double_offsets)), tuple(sorted(triple_offsets))

            measured_terminal_patterns: (
                dict[tuple[tuple[int, ...], tuple[int, ...]], int] | None
            ) = None
            if semantic_guidance_policy.tail_unsaturation_position_strategy in {
                "measured_joint_terminal_pattern",
                "measured_joint_terminal_pattern_ranked",
            }:
                chemistry_prior = semantic_guidance_policy.local_chemistry_prior
                if chemistry_prior is None:
                    return None, "ugi_local_chemistry_reference_unavailable"
                measured_terminal_patterns = {
                    (double_offsets, triple_offsets): count
                    for double_offsets, triple_offsets, count in (
                        chemistry_prior.unsaturation_pattern_frequencies(role)
                    )
                }
                if not measured_terminal_patterns:
                    return None, f"ugi_tail_unsaturation_pattern_unavailable:{role}"
                if (
                    semantic_guidance_policy.tail_unsaturation_position_strategy
                    == "measured_joint_terminal_pattern_ranked"
                ):
                    assert local_scores is not None
                    # Five is one rank above the strongest coordinatewise-support tier.  It changes
                    # only the ordering within a measured count class; the existing entropy mixture
                    # keeps every chemically valid assignment eligible.
                    local_scores = [
                        (
                            5.0
                            if candidate_terminal_pattern(candidate_index)
                            in measured_terminal_patterns
                            else score
                        )
                        for candidate_index, score in enumerate(local_scores)
                    ]

            def ranked_probabilities(candidate_indices: Sequence[int]) -> np.ndarray:
                subset_model = [model_scores[value] for value in candidate_indices]
                subset_semantic = [semantic_distances[value] for value in candidate_indices]
                subset_joint = (
                    None
                    if joint_scores is None
                    else [joint_scores[value] for value in candidate_indices]
                )
                if local_scores is not None:
                    return semantic_guidance_policy.bond_chemistry_probabilities(
                        subset_model,
                        subset_semantic,
                        [local_scores[value] for value in candidate_indices],
                        subset_joint,
                    )
                return semantic_guidance_policy.probabilities(
                    subset_model,
                    subset_semantic,
                    subset_joint,
                )

            if (
                semantic_guidance_policy.tail_unsaturation_position_strategy
                == "measured_joint_terminal_pattern"
            ):
                assert measured_terminal_patterns is not None
                indices_by_pattern: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
                for candidate_index in range(len(candidates)):
                    pattern = candidate_terminal_pattern(candidate_index)
                    if pattern is None:
                        continue
                    if pattern in measured_terminal_patterns:
                        indices_by_pattern.setdefault(pattern, []).append(candidate_index)
                if not indices_by_pattern:
                    return None, f"ugi_tail_unsaturation_pattern_unavailable:{role}"
                pattern_keys = tuple(sorted(indices_by_pattern))
                pattern_probabilities = np.asarray(
                    [float(measured_terminal_patterns[key]) for key in pattern_keys],
                    dtype=np.float64,
                )
                if not np.isfinite(pattern_probabilities).all() or not np.all(
                    pattern_probabilities > 0
                ):
                    return None, f"ugi_tail_unsaturation_pattern_frequency_invalid:{role}"
                pattern_probabilities /= pattern_probabilities.sum()
                selected_pattern = pattern_keys[
                    int(terminal_generator.choice(len(pattern_keys), p=pattern_probabilities))
                ]
                selected_indices = tuple(indices_by_pattern[selected_pattern])
                selected = int(
                    terminal_generator.choice(
                        selected_indices,
                        p=ranked_probabilities(selected_indices),
                    )
                )
            elif semantic_guidance_policy.tail_unsaturation_count_strategy in {
                "grouped_downward",
                "frequency_downward",
                "frequency_resampled",
                "smoothed_frequency_resampled",
            }:
                group_keys = tuple(
                    sorted({(candidate[3], candidate[4]) for candidate in candidates})
                )
                indices_by_group = {
                    key: tuple(
                        candidate_index
                        for candidate_index, candidate in enumerate(candidates)
                        if (candidate[3], candidate[4]) == key
                    )
                    for key in group_keys
                }

                def log_mean_exp(values: Sequence[float]) -> float:
                    array = np.asarray(values, dtype=np.float64)
                    maximum = float(np.max(array))
                    return maximum + float(np.log(np.exp(array - maximum).mean()))

                group_model_scores = [
                    log_mean_exp([model_scores[value] for value in indices_by_group[key]])
                    for key in group_keys
                ]
                group_semantic_distances = [
                    semantic_distances[indices_by_group[key][0]] for key in group_keys
                ]
                group_joint_scores = (
                    None
                    if joint_scores is None
                    else [joint_scores[indices_by_group[key][0]] for key in group_keys]
                )
                group_local_scores = (
                    None
                    if local_scores is None
                    else [
                        max(local_scores[value] for value in indices_by_group[key])
                        for key in group_keys
                    ]
                )
                if semantic_guidance_policy.tail_unsaturation_count_strategy in {
                    "frequency_downward",
                    "frequency_resampled",
                    "smoothed_frequency_resampled",
                }:
                    chemistry_prior = semantic_guidance_policy.local_chemistry_prior
                    if chemistry_prior is None:
                        return None, "ugi_local_chemistry_reference_unavailable"
                    measured_counts = {
                        (double_count, triple_count): count
                        for double_count, triple_count, count in (
                            chemistry_prior.unsaturation_count_frequencies(role)
                        )
                    }
                    smoothing = (
                        float(
                            chemistry_prior.smoothing
                            if semantic_guidance_policy.tail_unsaturation_frequency_pseudocount
                            is None
                            else semantic_guidance_policy.tail_unsaturation_frequency_pseudocount
                        )
                        if semantic_guidance_policy.tail_unsaturation_count_strategy
                        == "smoothed_frequency_resampled"
                        else 0.0
                    )
                    group_probabilities = np.asarray(
                        [float(measured_counts.get(key, 0)) + smoothing for key in group_keys],
                        dtype=np.float64,
                    )
                    if not np.isfinite(group_probabilities).all() or not np.any(
                        group_probabilities > 0
                    ):
                        return None, f"ugi_tail_unsaturation_frequency_unavailable:{role}"
                    group_probabilities /= group_probabilities.sum()
                else:
                    group_probabilities = (
                        semantic_guidance_policy.bond_chemistry_probabilities(
                            group_model_scores,
                            group_semantic_distances,
                            group_local_scores,
                            group_joint_scores,
                        )
                        if group_local_scores is not None
                        else semantic_guidance_policy.probabilities(
                            group_model_scores,
                            group_semantic_distances,
                            group_joint_scores,
                        )
                    )
                selected_group = group_keys[
                    int(terminal_generator.choice(len(group_keys), p=group_probabilities))
                ]
                selected_indices = indices_by_group[selected_group]
                selected = int(
                    terminal_generator.choice(
                        selected_indices,
                        p=ranked_probabilities(selected_indices),
                    )
                )
            else:
                all_indices = tuple(range(len(candidates)))
                selected = int(
                    terminal_generator.choice(
                        all_indices,
                        p=ranked_probabilities(all_indices),
                    )
                )
        selected_forced.update(candidates[selected][5])
    return selected_forced, None


def _exact_role_morphology_targets(
    record: SynthesisProgramGraphRecord,
) -> dict[int, tuple[int, int, int, int]]:
    """Read one explicit, internally consistent morphology target per semantic role."""

    states = record.role_morphology_states
    if states is None:
        raise SynthesisProgramSamplingError(
            "exact program-topology decoding requires explicit role morphology states"
        )
    targets: dict[int, tuple[int, int, int, int]] = {}
    for role_state in sorted(set(int(value) for value in record.role_states)):
        if role_state <= 0:
            continue
        values = states[record.role_states == role_state]
        unique = np.unique(values, axis=0)
        if unique.shape != (1, 4) or np.any(unique[0] < 1):
            raise SynthesisProgramSamplingError(
                "exact role morphology must be positive and constant within each role"
            )
        targets[role_state] = tuple(int(value) - 1 for value in unique[0])
    return targets


def _terminal_role_morphology(
    record: SynthesisProgramGraphRecord,
    *,
    parents: np.ndarray,
    closure_left: np.ndarray,
    closure_right: np.ndarray,
    parent_edge_mask: np.ndarray | None = None,
) -> dict[int, tuple[int, int, int, int]]:
    """Measure the same four coarse coordinates used by the conditioning tensor."""

    core = record.core_position_states > 1
    output: dict[int, tuple[int, int, int, int]] = {}
    for role_state in sorted(set(int(value) for value in record.role_states)):
        if role_state <= 0:
            continue
        role = record.role_states == role_state
        exterior_indices = set(np.flatnonzero(role & ~core).tolist())
        child_counts = {node: 0 for node in exterior_indices}
        attachments = 0
        for child in range(1, record.node_count):
            if parent_edge_mask is not None and not bool(parent_edge_mask[child]):
                continue
            parent = int(parents[child])
            if child in exterior_indices and parent in exterior_indices:
                child_counts[parent] += 1
            elif (
                child in exterior_indices
                and bool(core[parent])
                and int(record.role_states[parent]) == role_state
            ) or (
                parent in exterior_indices
                and bool(core[child])
                and int(record.role_states[child]) == role_state
            ):
                attachments += 1
        cycles = 0
        for left, right in zip(closure_left, closure_right, strict=True):
            left_index = int(left)
            right_index = int(right)
            if left_index in exterior_indices and right_index in exterior_indices:
                cycles += 1
            elif (
                left_index in exterior_indices
                and bool(core[right_index])
                and int(record.role_states[right_index]) == role_state
            ) or (
                right_index in exterior_indices
                and bool(core[left_index])
                and int(record.role_states[left_index]) == role_state
            ):
                attachments += 1
        output[role_state] = (
            len(exterior_indices),
            sum(max(children - 1, 0) for children in child_counts.values()),
            cycles,
            attachments,
        )
    return output


def _ugi_topology_support_failure(
    record: SynthesisProgramGraphRecord,
    *,
    parents: np.ndarray,
    closure_left: np.ndarray,
    closure_right: np.ndarray,
    policy: UgiTransformerTopologyPolicy,
    ester_policy: UgiEsterChemotypePolicy | None,
) -> str | None:
    """Validate a learned Ugi topology against the frozen structural support.

    The learned parent and closure heads remain responsible for selecting the topology. This
    check only removes structures outside measured training support; it does not construct an
    alternative tree, retry an attempt, or copy a component graph.
    """

    if record.program_id != UGI_PROGRAM_ID:
        return "learned_ugi_topology_applied_to_other_program"
    neighbors = [set() for _ in range(record.node_count)]
    tree_neighbors = [set() for _ in range(record.node_count)]
    exterior_children = np.zeros(record.node_count, dtype=np.int64)
    for child in range(1, record.node_count):
        parent = int(parents[child])
        neighbors[child].add(parent)
        neighbors[parent].add(child)
        tree_neighbors[child].add(parent)
        tree_neighbors[parent].add(child)
        if (
            int(record.core_position_states[child]) == 1
            and int(record.core_position_states[parent]) == 1
            and int(record.role_states[child]) == int(record.role_states[parent])
        ):
            exterior_children[parent] += 1
    for left, right in zip(closure_left, closure_right, strict=True):
        left_index = int(left)
        right_index = int(right)
        neighbors[left_index].add(right_index)
        neighbors[right_index].add(left_index)
    exterior = record.core_position_states == 1
    if any(len(neighbors[node]) > policy.maximum_heavy_degree for node in np.flatnonzero(exterior)):
        return "learned_ugi_topology_heavy_degree_outside_support"

    role_by_state = {block.role_state: block.role for block in record.component_blocks}
    role_limits = dict(zip(ROLE_NAMES, policy.maximum_adjacent_branch_run_by_role, strict=True))
    branched = exterior & (exterior_children >= 2)
    for role_state, role in role_by_state.items():
        pending = set(np.flatnonzero(branched & (record.role_states == int(role_state))).tolist())
        maximum_run = 0
        while pending:
            start = pending.pop()
            stack = [start]
            size = 0
            while stack:
                node = stack.pop()
                size += 1
                adjacent = pending.intersection(tree_neighbors[node])
                pending.difference_update(adjacent)
                stack.extend(adjacent)
            maximum_run = max(maximum_run, size)
        if maximum_run > role_limits.get(role, 0):
            return f"learned_ugi_topology_adjacent_branch_run_outside_support:{role}"

    for slot, (left, right) in enumerate(zip(closure_left, closure_right, strict=True)):
        cycle = tree_path_indices(parents, int(left), int(right))
        ring_size = len(cycle)
        allowed = policy.allowed_ring_sizes
        left_role_state = int(record.role_states[int(left)])
        left_role = role_by_state.get(left_role_state)
        if (
            ester_policy is not None
            and left_role == ester_policy.amine_role
            and left_role_state == int(record.role_states[int(right)])
        ):
            exterior_count = sum(
                int(value) == left_role_state and int(core_state) == 1
                for value, core_state in zip(
                    record.role_states, record.core_position_states, strict=True
                )
            )
            allowed = ester_policy.allowed_amine_cycle_sizes(exterior_count)
        if ring_size not in allowed:
            return f"learned_ugi_topology_ring_size_outside_support:{slot}"
    return None


def _carbon_skeleton_diameter_from_states(
    nodes: Sequence[int],
    *,
    known: Mapping[int, int],
    neighbors: Sequence[set[int]],
    atom_vocabulary: Sequence[AtomState],
) -> int:
    carbon = {
        int(node)
        for node in nodes
        if int(node) in known and atom_vocabulary[int(known[int(node)])].symbol == "C"
    }
    if not carbon:
        return 0
    maximum_edges = 0
    for start in carbon:
        distances = {start: 0}
        queue = [start]
        for current in queue:
            for neighbor in neighbors[current]:
                if neighbor in carbon and neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        maximum_edges = max(maximum_edges, max(distances.values()))
    return maximum_edges + 1


def _select_ugi_amine_atom_states(
    predictions: Mapping[str, np.ndarray],
    index: int,
    record: SynthesisProgramGraphRecord,
    atom_vocabulary: Sequence[AtomState],
    atom_capacities: np.ndarray,
    minimum_used: np.ndarray,
    neighbors: Sequence[set[int]],
    triangles: Sequence[tuple[int, int, int]],
    generated_cycles: Sequence[Sequence[int]],
    role_names: Sequence[str],
    local_chemistry_support: LocalChemistrySupport,
    policy: UgiEsterChemotypePolicy,
    chemistry_prior: UgiRoleChemistryPrior | None = None,
    chemistry_prior_strength: float = 0.0,
    distances_to_core: np.ndarray | None = None,
    ring_nodes: frozenset[int] = frozenset(),
    ring_edges: frozenset[tuple[int, int]] = frozenset(),
    semantic_target: UgiAmineSemanticTarget | None = None,
    all_role_semantic_target: UgiAllRoleSemanticTarget | None = None,
    semantic_guidance_policy: UgiMogSemanticGuidancePolicy | None = None,
    terminal_generator: np.random.Generator | None = None,
) -> tuple[dict[int, int] | None, str | None]:
    """Choose one globally feasible measured-like amine exterior under model scores.

    Without an explicit semantic target, the legacy measured-quantile C/N policy is retained.  With
    a target, every exact C/N/O placement matching its total composition and carbon-skeleton
    diameter is enumerated.  The head exterior has at most eight atoms, so this is a bounded exact
    conditional readout rather than rejection, repair or component lookup.
    """

    if record.program_id != policy.reaction_id:
        return {}, None
    blocks = [block for block in record.component_blocks if block.role == policy.amine_role]
    if len(blocks) != 1:
        return None, "ugi_amine_role_block_unavailable"
    block = blocks[0]
    exterior = tuple(
        node
        for node in range(block.start, block.stop)
        if int(record.core_position_states[node]) == 1 and not bool(record.fixed_atom_mask[node])
    )
    if not exterior:
        return None, "ugi_amine_exterior_unavailable"
    fixed_states = {
        int(node): int(record.graph.node_states[node])
        for node in np.flatnonzero(record.fixed_atom_mask)
    }
    candidates: list[
        tuple[float, float, float | None, float | None, object | None, dict[int, int]]
    ] = []
    joint_program = (
        None
        if semantic_guidance_policy is None or not semantic_guidance_policy.uses_joint_realism
        else _ugi_program_from_record(record)
    )
    if joint_program is not None and all_role_semantic_target is None:
        return None, "ugi_joint_realism_complete_target_unavailable"
    symbol_assignments: list[dict[int, str]] = []
    if semantic_target is None:
        minimum_n = int(policy.minimum_amine_exterior_nitrogens)
        maximum_n = min(int(policy.maximum_amine_exterior_nitrogens), len(exterior))
        for nitrogen_count in range(minimum_n, maximum_n + 1):
            for nitrogen_nodes in combinations(exterior, nitrogen_count):
                nitrogen_set = set(nitrogen_nodes)
                symbol_assignments.append(
                    {node: ("N" if node in nitrogen_set else "C") for node in exterior}
                )
    else:
        fixed_block_symbols = Counter(
            atom_vocabulary[fixed_states[node]].symbol
            for node in range(block.start, block.stop)
            if node in fixed_states
        )
        target_counts = {
            "N": int(semantic_target.nitrogen_atoms),
            "O": int(semantic_target.oxygen_atoms),
        }
        target_counts["C"] = block.atom_count - target_counts["N"] - target_counts["O"]
        required = {
            symbol: target_counts[symbol] - int(fixed_block_symbols[symbol])
            for symbol in ("C", "N", "O")
        }
        if (
            any(count < 0 for count in required.values())
            or sum(required.values()) != len(exterior)
            or any(
                symbol not in {"C", "N", "O"} and count > 0
                for symbol, count in fixed_block_symbols.items()
            )
        ):
            return None, "ugi_amine_semantic_composition_unavailable"
        for nitrogen_nodes in combinations(exterior, required["N"]):
            nitrogen_set = set(nitrogen_nodes)
            remaining = tuple(node for node in exterior if node not in nitrogen_set)
            for oxygen_nodes in combinations(remaining, required["O"]):
                oxygen_set = set(oxygen_nodes)
                symbol_assignments.append(
                    {
                        node: ("N" if node in nitrogen_set else "O" if node in oxygen_set else "C")
                        for node in exterior
                    }
                )
    stages: Counter[str] = Counter()
    for symbols_by_node in symbol_assignments:
        selected: dict[int, int] = {}
        score = 0.0
        local_evidence_scores: list[float] = []
        feasible = True
        for node in exterior:
            symbol = symbols_by_node[node]
            valid = np.asarray(
                [
                    atom.symbol == symbol
                    and atom.formal_charge == 0
                    and not atom.aromatic
                    and int(atom_capacities[state]) >= int(minimum_used[node])
                    for state, atom in enumerate(atom_vocabulary)
                ],
                dtype=np.bool_,
            )
            state = _argmax_allowed(predictions["nodes"][index, node], valid)
            if state is None:
                feasible = False
                break
            selected[node] = state
            score += float(predictions["nodes"][index, node, state])
            if semantic_guidance_policy is not None and (
                semantic_guidance_policy.uses_coordinate_local_chemistry_atoms
            ):
                rank_prior = semantic_guidance_policy.local_chemistry_prior
                if rank_prior is None or distances_to_core is None:
                    feasible = False
                    break
                rank_bias = semantic_guidance_policy.atom_local_scores(
                    role=block.role,
                    depth=int(distances_to_core[node]),
                    degree=len(neighbors[node]),
                    in_ring=node in ring_nodes,
                    atom_vocabulary=atom_vocabulary,
                )
                local_evidence_scores.append(float(rank_bias[state]))
            if chemistry_prior is not None and chemistry_prior_strength > 0:
                if distances_to_core is None or int(distances_to_core[node]) < 1:
                    feasible = False
                    break
                bias = chemistry_prior.atom_log_bias(
                    role=block.role,
                    depth=int(distances_to_core[node]),
                    degree=len(neighbors[node]),
                    in_ring=node in ring_nodes,
                    atom_vocabulary=atom_vocabulary,
                )
                score += chemistry_prior_strength * float(bias[state])
        if not feasible:
            continue
        stages["atom_capacity"] += 1
        known = {**fixed_states, **selected}
        if any(node not in known for node in range(block.start, block.stop)):
            continue
        symbols = [atom_vocabulary[known[node]].symbol for node in range(block.start, block.stop)]
        carbon_atoms = symbols.count("C")
        heavy_atoms = len(symbols)
        carbon_diameter = _carbon_skeleton_diameter_from_states(
            range(block.start, block.stop),
            known=known,
            neighbors=neighbors,
            atom_vocabulary=atom_vocabulary,
        )
        actual_donors: int | None = None
        actual_branches: int | None = None
        if semantic_target is not None:
            tolerance = (
                0
                if semantic_guidance_policy is None
                else semantic_guidance_policy.amine_carbon_skeleton_diameter_tolerance
            )
            if abs(carbon_diameter - int(semantic_target.carbon_skeleton_diameter)) > tolerance:
                continue
        stages["carbon_diameter"] += 1
        if semantic_target is not None:
            block_nodes = set(range(block.start, block.stop))
            reactive_amine_sites = sum(
                atom_vocabulary[known[node]].symbol == "N"
                and 1 <= sum(neighbor in block_nodes for neighbor in neighbors[node]) <= 2
                for node in block_nodes
            )
            if not 1 <= reactive_amine_sites <= 2:
                continue
            if semantic_target.hydrogen_bond_donors is not None:
                ordered_nodes = tuple(range(block.start, block.stop))
                actual_donors, actual_branches = amine_local_substitution_metrics(
                    tuple(atom_vocabulary[known[node]].symbol for node in ordered_nodes),
                    tuple(
                        sum(neighbor in block_nodes for neighbor in neighbors[node])
                        for node in ordered_nodes
                    ),
                )
                if abs(actual_donors - semantic_target.hydrogen_bond_donors) > (
                    0
                    if semantic_guidance_policy is None
                    else semantic_guidance_policy.amine_hydrogen_bond_donors_tolerance
                ) or abs(actual_branches - semantic_target.heavy_branch_atoms) > (
                    0
                    if semantic_guidance_policy is None
                    else semantic_guidance_policy.amine_heavy_branch_atoms_tolerance
                ):
                    continue
        stages["registry_handle"] += 1
        if not local_chemistry_support.component_is_within_observed_support(
            record.program_id,
            block.role,
            heavy_atoms=heavy_atoms,
            carbon_atoms=carbon_atoms,
            heteroatoms=heavy_atoms - carbon_atoms,
        ):
            continue
        stages["component_support"] += 1
        if any(
            left in known
            and right in known
            and not local_chemistry_support.allows_role_edge_for_any_bond(
                record.program_id,
                role_names[left],
                atom_vocabulary[known[left]].symbol,
                role_names[right],
                atom_vocabulary[known[right]].symbol,
            )
            for left in range(record.node_count)
            for right in neighbors[left]
            if left < right
        ):
            continue
        stages["edge_support"] += 1
        head_arrangement_label: object | None = None
        if (
            semantic_guidance_policy is not None
            and semantic_guidance_policy.uses_local_chemistry_atoms
            and semantic_guidance_policy.local_chemistry_score_mode == "support_tier"
        ):
            exterior_set = set(exterior)
            for left in exterior:
                for right in neighbors[left]:
                    if right not in exterior_set or left >= right:
                        continue
                    adjacency_score = semantic_guidance_policy.edge_symbol_local_score(
                        role=block.role,
                        depth=min(
                            int(distances_to_core[left]),
                            int(distances_to_core[right]),
                        ),
                        in_ring=tuple(sorted((left, right))) in ring_edges,
                        symbols=(
                            atom_vocabulary[known[left]].symbol,
                            atom_vocabulary[known[right]].symbol,
                        ),
                    )
                    if adjacency_score is None:
                        raise RuntimeError("support-tier adjacency score is unavailable")
                    local_evidence_scores.append(float(adjacency_score))
        if any(
            all(node in known for node in triangle)
            and not local_chemistry_support.allows_role_triangle(
                record.program_id,
                ((role_names[node], atom_vocabulary[known[node]].symbol) for node in triangle),
            )
            for triangle in triangles
        ):
            continue
        stages["triangle_support"] += 1
        if local_chemistry_support.enforces_role_cycles and any(
            all(node in known for node in cycle)
            and not local_chemistry_support.allows_role_cycle(
                record.program_id,
                ((role_names[node], atom_vocabulary[known[node]].symbol) for node in cycle),
            )
            for cycle in generated_cycles
        ):
            continue
        stages["cycle_support"] += 1
        if (
            semantic_guidance_policy is not None
            and semantic_guidance_policy.local_chemistry_score_mode
            in {
                "whole_head_support_tier",
                "whole_head_support_binary",
                "whole_head_support_distance",
            }
        ):
            if distances_to_core is None:
                return None, "ugi_head_arrangement_reference_unavailable"
            whole_head_score = semantic_guidance_policy.amine_head_arrangement_support_score(
                nodes=exterior,
                symbols_by_node={
                    node: atom_vocabulary[state].symbol for node, state in known.items()
                },
                depths_by_node={node: int(distances_to_core[node]) for node in exterior},
                neighbors=neighbors,
                ring_nodes=ring_nodes,
            )
            local_evidence_scores.append(whole_head_score)
            chemistry_reference = semantic_guidance_policy.local_chemistry_prior
            if chemistry_reference is None:
                return None, "ugi_head_arrangement_reference_unavailable"
            head_arrangement_label = chemistry_reference.amine_head_arrangement_signatures(
                nodes=exterior,
                symbols_by_node={
                    node: atom_vocabulary[state].symbol for node, state in known.items()
                },
                depths_by_node={node: int(distances_to_core[node]) for node in exterior},
                neighbors=neighbors,
                ring_nodes=ring_nodes,
            )[0]
        semantic_distance = (
            0.0
            if semantic_target is None
            else float(abs(carbon_diameter - semantic_target.carbon_skeleton_diameter))
            + (
                0.0
                if semantic_target.hydrogen_bond_donors is None
                else float(
                    abs(actual_donors - semantic_target.hydrogen_bond_donors)
                    + abs(actual_branches - semantic_target.heavy_branch_atoms)
                )
            )
        )
        joint_score = None
        if joint_program is not None:
            heavy_diameter = _induced_graph_diameter(
                range(block.start, block.stop),
                neighbors,
            )
            if heavy_diameter < 1 or semantic_target is None:
                continue
            candidate_amine = UgiAmineSemanticTarget(
                heavy_atom_graph_diameter=heavy_diameter,
                carbon_skeleton_diameter=carbon_diameter,
                nitrogen_atoms=semantic_target.nitrogen_atoms,
                oxygen_atoms=semantic_target.oxygen_atoms,
                hydrogen_bond_donors=actual_donors,
                heavy_branch_atoms=actual_branches,
            )
            joint_score = semantic_guidance_policy.joint_realism_scores(
                joint_program,
                (replace_amine_semantics(all_role_semantic_target, candidate_amine),),
            )[0]
        candidates.append(
            (
                score,
                semantic_distance,
                joint_score,
                (
                    semantic_guidance_policy.aggregate_local_scores(local_evidence_scores)
                    if semantic_guidance_policy is not None
                    and semantic_guidance_policy.uses_local_chemistry_atoms
                    else None
                ),
                head_arrangement_label,
                selected,
            )
        )
    if candidates:
        if (
            semantic_guidance_policy is None
            or not semantic_guidance_policy.uses_ranked_terminal_chemistry
        ):
            choice = max(range(len(candidates)), key=lambda value: candidates[value][0])
        else:
            if terminal_generator is None:
                return None, "ugi_amine_semantic_guidance_generator_unavailable"
            model_scores = [candidate[0] for candidate in candidates]
            semantic_distances = [candidate[1] for candidate in candidates]
            joint_scores = (
                [float(candidate[2]) for candidate in candidates]
                if semantic_guidance_policy.uses_joint_realism
                else None
            )
            local_scores = (
                [float(candidate[3]) for candidate in candidates]
                if semantic_guidance_policy.uses_local_chemistry_atoms
                else None
            )
            if local_scores is None:
                probabilities = semantic_guidance_policy.probabilities(
                    model_scores,
                    semantic_distances,
                    joint_scores,
                )
            elif semantic_guidance_policy.local_chemistry_score_mode in {
                "whole_head_support_tier",
                "whole_head_support_binary",
                "whole_head_support_distance",
            }:
                probabilities = semantic_guidance_policy.whole_head_chemistry_probabilities(
                    model_scores,
                    semantic_distances,
                    local_scores,
                    joint_scores,
                    group_labels=[candidate[4] for candidate in candidates],
                )
            else:
                probabilities = semantic_guidance_policy.atom_chemistry_probabilities(
                    model_scores,
                    semantic_distances,
                    local_scores,
                    joint_scores,
                )
            choice = int(terminal_generator.choice(len(candidates), p=probabilities))
        return candidates[choice][5], None
    if not symbol_assignments:
        return None, "ugi_amine_semantic_composition_unavailable"
    stage_reasons = (
        ("atom_capacity", "ugi_amine_atom_capacity_unavailable"),
        ("carbon_diameter", "ugi_amine_carbon_diameter_unavailable"),
        ("registry_handle", "ugi_amine_registry_handle_unavailable"),
        ("component_support", "ugi_amine_component_support_unavailable"),
        ("edge_support", "ugi_amine_edge_support_unavailable"),
        ("triangle_support", "ugi_amine_triangle_support_unavailable"),
        ("cycle_support", "ugi_amine_cycle_support_unavailable"),
    )
    for stage, reason in stage_reasons:
        if stages[stage] == 0:
            return None, reason
    return None, "ugi_amine_global_atom_assignment_unavailable"
