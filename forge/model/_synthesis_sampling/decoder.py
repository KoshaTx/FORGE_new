"""Strict terminal decoding with explicit abstention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from forge.model._synthesis_sampling.constraints import (
    _argmax_allowed,
    _atom_capacity_table,
    _bond_unit_table,
    _distances_to_reaction_core,
    _exact_role_morphology_targets,
    _generated_ring_support,
    _sample_allowed,
    _sample_mog_chemistry_allowed,
    _select_ugi_all_role_tail_bonds,
    _select_ugi_amine_atom_states,
    _terminal_role_morphology,
    _ugi_ester_motif_constraints,
    _ugi_topology_support_failure,
)
from forge.model._synthesis_sampling.contracts import (
    SynthesisProgramSamplingError,
    TerminalArrayState,
    TerminalTensorState,
    TopologyArrayState,
)
from forge.model.compose_lipid_atom_aware import topology_atom_states
from forge.model.defog_feasibility import AtomState
from forge.model.local_chemistry_support import LocalChemistrySupport, tree_path_indices
from forge.model.reaction_core_saturation import BoundReactionCoreSaturation
from forge.model.reaction_program_flow import restore_synthesis_program_fixed_states
from forge.model.sparse_topology_feasibility import BOND_VALENCE_UNITS
from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord
from forge.model.ugi_all_role_semantic_program import UgiAllRoleSemanticTarget
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_mog_semantic_guidance import UgiMogSemanticGuidancePolicy
from forge.model.ugi_role_chemistry_prior import UgiRoleChemistryPrior
from forge.model.ugi_transformer_topology import UGI_PROGRAM_ID, UgiTransformerTopologyPolicy

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


def _strict_terminal_record(
    predictions: Mapping[str, np.ndarray],
    index: int,
    record: SynthesisProgramGraphRecord,
    atom_vocabulary: Sequence[AtomState],
    local_chemistry_support: LocalChemistrySupport | None = None,
    *,
    enforce_program_topology: bool = False,
    enforce_program_cycles: bool = False,
    confine_generated_edges: bool = False,
    core_saturation: BoundReactionCoreSaturation | None = None,
    terminal_generator: np.random.Generator | None = None,
    terminal_temperature: float = 1.0,
    ugi_ester_chemotype_policy: UgiEsterChemotypePolicy | None = None,
    ugi_role_chemistry_prior: UgiRoleChemistryPrior | None = None,
    ugi_role_chemistry_prior_strength: float = 0.0,
    ugi_topology_policy: UgiTransformerTopologyPolicy | None = None,
    ugi_amine_semantic_target: UgiAmineSemanticTarget | None = None,
    ugi_all_role_semantic_target: UgiAllRoleSemanticTarget | None = None,
    ugi_mog_semantic_guidance_policy: UgiMogSemanticGuidancePolicy | None = None,
    topology_only: bool = False,
    qualified_core_units: np.ndarray | None = None,
    confine_origin_edges: bool = False,
    reserve_fixed_closures: bool = False,
    origin_closure_roles: Sequence[int] | None = None,
    origin_ring_sizes: Mapping[int, Sequence[int]] | None = None,
    exact_origin_morphology: bool = False,
    atom_aware_topology: bool = False,
) -> tuple[TerminalArrayState | TopologyArrayState | None, str | None]:
    """Decode one exact-size graph under topology and valence support, without fallback."""

    if ugi_amine_semantic_target is not None and ugi_all_role_semantic_target is not None:
        return None, "ugi_semantic_target_modes_are_mutually_exclusive"
    effective_amine_semantic_target = (
        ugi_all_role_semantic_target.amine
        if ugi_all_role_semantic_target is not None
        else ugi_amine_semantic_target
    )

    count = record.node_count
    closure_count = record.graph.closure_count
    bond_classes = predictions["parent_bonds"].shape[-1]
    if BOND_VALENCE_UNITS is None or bond_classes > len(BOND_VALENCE_UNITS):
        return None, "unsupported_bond_vocabulary"
    bond_units = _bond_unit_table(bond_classes)
    atom_capacities = _atom_capacity_table(atom_vocabulary)
    maximum_capacity = int(atom_capacities.max())
    maximum_capacities = np.full(record.node_count, maximum_capacity, dtype=np.int64)
    for node in np.flatnonzero(record.fixed_atom_mask):
        state = int(record.graph.node_states[node])
        if state >= len(atom_vocabulary):
            return None, "fixed_atom_state_outside_vocabulary"
        maximum_capacities[node] = atom_capacities[state]
    topology_states = None
    if atom_aware_topology:
        if terminal_generator is not None or any(
            value is not None
            for value in (
                ugi_ester_chemotype_policy,
                ugi_role_chemistry_prior,
                effective_amine_semantic_target,
                ugi_mog_semantic_guidance_policy,
            )
        ):
            return None, "atom_aware_topology_incompatible_chemistry_policy"
        topology_states, reason = topology_atom_states(
            predictions["nodes"][index, :count], record, atom_capacities, bond_units
        )
        if topology_states is None:
            return None, reason
        maximum_capacities = atom_capacities[topology_states].copy()
    parents = np.zeros(count, dtype=np.int64)
    parent_bonds = np.zeros(count, dtype=np.int64)
    closure_left = np.zeros(closure_count, dtype=np.int64)
    closure_right = np.zeros(closure_count, dtype=np.int64)
    closure_bonds = np.zeros(closure_count, dtype=np.int64)
    minimum_used = np.zeros(count, dtype=np.int64)
    degrees = np.zeros(count, dtype=np.int64)
    occupied: set[tuple[int, int]] = set()
    component_instances = np.zeros(count, dtype=np.int64)
    for component_index, block in enumerate(record.component_blocks, start=1):
        component_instances[block.start : block.stop] = component_index
    core = record.core_position_states
    role_by_state = {block.role_state: block.role for block in record.component_blocks}
    try:
        role_names = tuple(role_by_state[int(value)] for value in record.role_states)
    except KeyError:
        return None, "unnamed_semantic_role"
    morphology_targets = (
        _exact_role_morphology_targets(record)
        if enforce_program_topology or enforce_program_cycles or exact_origin_morphology
        else None
    )
    if origin_closure_roles is not None:
        if len(origin_closure_roles) != closure_count or any(
            (role != -1 if record.fixed_closure_bond_mask[slot] else role not in role_by_state)
            for slot, role in enumerate(origin_closure_roles)
        ):
            return None, "invalid_origin_closure_allocation"
        if origin_ring_sizes is None:
            return None, "missing_origin_ring_support"

    # The qualified transform pins the hydrogen count, and therefore the exact heavy-atom
    # valence, of some reaction-core positions.  Reduce their capacity to that requirement so the
    # admissible sets below simply cannot place another neighbour there, and record the target so
    # under-saturation is caught too.  Precursor components meet only at the core, so a generated
    # edge stays inside one component block and a generated closure joins two exterior atoms.
    required_core_units: np.ndarray | None = None
    core_constrained = core_saturation is not None and core_saturation.applies_to(record)
    component_confined = confine_generated_edges or (
        core_constrained and core_saturation.policy.component_confined_generated_edges
    )
    exterior_only_closures = confine_generated_edges or (
        core_constrained and core_saturation.policy.exterior_only_generated_closures
    )
    if core_constrained:
        required_core_units = core_saturation.required_units(record)
        pinned = required_core_units >= 0
        if np.any(required_core_units[pinned] > maximum_capacities[pinned]):
            return None, "reaction_core_saturation_exceeds_atom_support"
        maximum_capacities[pinned] = required_core_units[pinned]
    if qualified_core_units is not None:
        if qualified_core_units.shape != (count,) or np.any(
            (qualified_core_units >= 0) & (record.core_position_states <= 1)
        ):
            return None, "invalid_qualified_core_valence_condition"
        required_core_units = qualified_core_units
        pinned = required_core_units >= 0
        if np.any(required_core_units[pinned] > maximum_capacities[pinned]):
            return None, "qualified_core_valence_exceeds_atom_support"
        maximum_capacities[pinned] = required_core_units[pinned]

    if reserve_fixed_closures:
        for slot in np.flatnonzero(record.fixed_closure_bond_mask):
            left, right = sorted(
                (int(record.graph.closure_left[slot]), int(record.graph.closure_right[slot]))
            )
            bond = int(record.graph.closure_bonds[slot])
            if (
                not 0 <= left < right < count
                or (left, right) in occupied
                or not 0 <= bond < bond_classes
            ):
                return None, "invalid_fixed_closure"
            closure_left[slot], closure_right[slot], closure_bonds[slot] = left, right, bond
            minimum_used[[left, right]] += int(bond_units[bond])
            degrees[[left, right]] += 1
            occupied.add((left, right))

    # Reserve immutable adapter edges first so variable choices cannot consume their capacity.
    for child in np.flatnonzero(record.fixed_parent_bond_mask):
        child = int(child)
        if child == 0:
            return None, "fixed_root_parent"
        parent = int(record.graph.parents[child])
        bond = int(record.graph.parent_bonds[child])
        if parent < 0 or parent >= child or bond >= bond_classes:
            return None, "invalid_fixed_parent_edge"
        units = int(bond_units[bond])
        parents[child] = parent
        parent_bonds[child] = bond
        degrees[[child, parent]] += 1
        minimum_used[[child, parent]] += units
        occupied.add((parent, child))
    if np.any(minimum_used > maximum_capacities):
        return None, "fixed_parent_valence_exceeds_support"

    node_positions = np.arange(count, dtype=np.int64)
    parent_headroom = minimum_used + 2 <= maximum_capacities
    for child in range(1, count):
        if record.fixed_parent_bond_mask[child]:
            continue
        # The child's own headroom does not depend on the candidate parent, so the whole
        # admissible-parent row is one vectorized comparison rather than a Python scan over every
        # earlier node.  ``parent_headroom`` tracks ``minimum_used + 2 <= maximum_capacities``
        # incrementally; only the two endpoints of an accepted edge can change it.
        valid = np.zeros(count, dtype=np.bool_)
        if parent_headroom[child]:
            valid[:child] = parent_headroom[:child]
            # Core saturation confines generated tree edges to one component just as program
            # topology does, so both gates mask the admissible row the same way.
            if (
                enforce_program_topology
                or component_confined
                or confine_origin_edges
                or exact_origin_morphology
            ):
                valid[:child] &= component_instances[:child] == component_instances[child]
            if reserve_fixed_closures:
                for parent in np.flatnonzero(valid):
                    if (int(parent), child) in occupied:
                        valid[parent] = False
        if enforce_program_topology or exact_origin_morphology:
            for parent in np.flatnonzero(valid).tolist():
                trial_parents = parents.copy()
                trial_parents[child] = parent
                observed = _terminal_role_morphology(
                    record,
                    parents=trial_parents,
                    closure_left=(
                        closure_left[record.fixed_closure_bond_mask]
                        if exact_origin_morphology
                        else np.empty(0, dtype=np.int64)
                    ),
                    closure_right=(
                        closure_right[record.fixed_closure_bond_mask]
                        if exact_origin_morphology
                        else np.empty(0, dtype=np.int64)
                    ),
                    parent_edge_mask=(record.fixed_parent_bond_mask | (node_positions <= child)),
                )
                assert morphology_targets is not None
                role_state = int(record.role_states[child])
                target = morphology_targets[role_state]
                # Counts and cycles are layout-level invariants.  During tree construction only
                # junction and core-attachment budgets can increase.  The upper-bound checks
                # prevent overshoot; the remaining-child bounds also prevent the greedy model
                # score from consuming the last opportunity to meet a positive budget.  This is
                # constrained MAP decoding of the learned parent logits, not a retry or a
                # constructive replacement topology.
                remaining_variable_children = sum(
                    not bool(record.fixed_parent_bond_mask[future])
                    and int(record.role_states[future]) == role_state
                    for future in range(child + 1, count)
                )
                current = observed[role_state]
                if (
                    current[1] > target[1]
                    or current[3] > target[3]
                    or current[1] + remaining_variable_children < target[1]
                    or current[3]
                    + remaining_variable_children
                    + (
                        int((~record.fixed_closure_bond_mask).sum())
                        if exact_origin_morphology
                        else 0
                    )
                    < target[3]
                ):
                    valid[parent] = False
        parent = _argmax_allowed(predictions["parents"][index, child, :count], valid)
        if parent is None:
            if enforce_program_topology:
                reason = "program_topology_parent_unavailable"
            elif component_confined:
                reason = "reaction_core_component_parent_unavailable"
            else:
                reason = "parent_capacity_exhausted"
            return None, reason
        parents[child] = parent
        degrees[[child, parent]] += 1
        minimum_used[[child, parent]] += 2
        parent_headroom[[child, parent]] = (
            minimum_used[[child, parent]] + 2 <= maximum_capacities[[child, parent]]
        )
        occupied.add((parent, child))

    for slot in np.flatnonzero(record.fixed_closure_bond_mask):
        if reserve_fixed_closures:
            continue
        slot = int(slot)
        left = int(record.graph.closure_left[slot])
        right = int(record.graph.closure_right[slot])
        bond = int(record.graph.closure_bonds[slot])
        pair = tuple(sorted((left, right)))
        if left < 0 or right >= count or left == right or pair in occupied or bond >= bond_classes:
            return None, "invalid_fixed_closure"
        units = int(bond_units[bond])
        closure_left[slot] = left
        closure_right[slot] = right
        closure_bonds[slot] = bond
        degrees[[left, right]] += 1
        minimum_used[[left, right]] += units
        occupied.add(pair)
    if np.any(minimum_used > maximum_capacities):
        return None, "fixed_closure_valence_exceeds_support"

    topology_neighbors = [set() for _ in range(count)]
    for left, right in occupied:
        topology_neighbors[left].add(right)
        topology_neighbors[right].add(left)
    for slot in range(closure_count):
        if record.fixed_closure_bond_mask[slot]:
            continue
        pair_scores = (
            predictions["closure_left"][index, slot, :count, None]
            + predictions["closure_right"][index, slot, None, :count]
        )
        best: tuple[float, int, int] | None = None
        for left in range(count):
            for right in range(left + 1, count):
                if (left, right) in occupied:
                    continue
                semantic_pair = component_instances[left] == component_instances[right] or (
                    int(core[left]) > 1 and int(core[right]) > 1
                )
                if confine_origin_edges:
                    semantic_pair = component_instances[left] == component_instances[right]
                if exterior_only_closures:
                    # A ring that reaches a reaction-core atom, or crosses two precursor
                    # components, cannot be cut back into that transform's precursors.
                    semantic_pair = (
                        component_instances[left] == component_instances[right]
                        and int(core[left]) == 1
                        and int(core[right]) == 1
                    )
                if enforce_program_topology or enforce_program_cycles:
                    role_state = int(record.role_states[left])
                    current = _terminal_role_morphology(
                        record,
                        parents=parents,
                        closure_left=closure_left[:slot],
                        closure_right=closure_right[:slot],
                    )
                    assert morphology_targets is not None
                    semantic_pair = (
                        component_instances[left] == component_instances[right]
                        and int(core[left]) == 1
                        and int(core[right]) == 1
                        and role_state == int(record.role_states[right])
                        and current[role_state][2] < morphology_targets[role_state][2]
                    )
                if (
                    not semantic_pair
                    or minimum_used[left] + 2 > maximum_capacities[left]
                    or minimum_used[right] + 2 > maximum_capacities[right]
                ):
                    continue
                if origin_closure_roles is not None:
                    requested_role = origin_closure_roles[slot]
                    if not (
                        int(record.role_states[left]) == requested_role
                        and int(record.role_states[right]) == requested_role
                        and component_instances[left] == component_instances[right]
                    ):
                        continue
                    ring_size = len(tree_path_indices(parents, left, right))
                    if ring_size not in origin_ring_sizes.get(requested_role, ()):
                        continue
                if exact_origin_morphology:
                    trial = _terminal_role_morphology(
                        record,
                        parents=parents,
                        closure_left=np.append(closure_left[:slot], left),
                        closure_right=np.append(closure_right[:slot], right),
                    )
                    assert morphology_targets is not None
                    if any(
                        trial[r][j] > target[j]
                        for r, target in morphology_targets.items()
                        for j in (2, 3)
                    ):
                        continue
                if local_chemistry_support is not None:
                    if local_chemistry_support.enforces_role_cycles:
                        cycle = tree_path_indices(parents, left, right)
                        if not local_chemistry_support.allows_any_role_cycle(
                            record.program_id,
                            (role_names[node] for node in cycle),
                        ):
                            continue
                    elif any(
                        not local_chemistry_support.allows_any_role_triangle(
                            record.program_id,
                            (role_names[left], role_names[middle], role_names[right]),
                        )
                        for middle in topology_neighbors[left].intersection(
                            topology_neighbors[right]
                        )
                    ):
                        continue
                direct = float(pair_scores[left, right])
                reverse = float(pair_scores[right, left])
                candidate = (max(direct, reverse), left, right)
                if best is None or candidate > best:
                    best = candidate
                    if reverse > direct:
                        closure_left[slot], closure_right[slot] = right, left
                    else:
                        closure_left[slot], closure_right[slot] = left, right
        if best is None:
            return None, (
                "program_role_closure_unavailable"
                if enforce_program_cycles and not enforce_program_topology
                else (
                    "reaction_core_exterior_closure_unavailable"
                    if exterior_only_closures and not enforce_program_topology
                    else "closure_pair_unavailable"
                )
            )
        _, left, right = best
        degrees[[left, right]] += 1
        minimum_used[[left, right]] += 2
        occupied.add((left, right))
        topology_neighbors[left].add(right)
        topology_neighbors[right].add(left)

    if enforce_program_topology or exact_origin_morphology:
        assert morphology_targets is not None
        observed_morphology = _terminal_role_morphology(
            record,
            parents=parents,
            closure_left=closure_left,
            closure_right=closure_right,
        )
        if observed_morphology != morphology_targets:
            return None, "program_morphology_exactness_failure"
    elif enforce_program_cycles:
        assert morphology_targets is not None
        observed_morphology = _terminal_role_morphology(
            record,
            parents=parents,
            closure_left=closure_left,
            closure_right=closure_right,
        )
        if any(
            observed_morphology[role_state][2] != target[2]
            for role_state, target in morphology_targets.items()
        ):
            return None, "program_role_cycle_exactness_failure"

    neighbors = [set() for _ in range(count)]
    for left, right in occupied:
        neighbors[left].add(right)
        neighbors[right].add(left)
    triangles: list[tuple[int, int, int]] = []
    for left in range(count):
        for middle in sorted(value for value in neighbors[left] if value > left):
            for right in sorted(
                value for value in neighbors[left].intersection(neighbors[middle]) if value > middle
            ):
                triangles.append((left, middle, right))
    generated_cycles = [
        tree_path_indices(parents, int(closure_left[slot]), int(closure_right[slot]))
        for slot in range(closure_count)
        if not record.fixed_closure_bond_mask[slot]
    ]
    distances_to_core = _distances_to_reaction_core(neighbors, record.core_position_states)
    if (
        (ugi_role_chemistry_prior is not None and ugi_role_chemistry_prior_strength > 0)
        or (
            ugi_mog_semantic_guidance_policy is not None
            and ugi_mog_semantic_guidance_policy.uses_local_chemistry
        )
    ) and np.any(distances_to_core < 0):
        return None, "reaction_core_distance_unavailable"
    ring_nodes, ring_edges = _generated_ring_support(generated_cycles)

    forced_atom_symbols: dict[int, str] = {}
    forced_bonds: dict[tuple[int, int], int] = {}
    if ugi_ester_chemotype_policy is not None:
        motif = _ugi_ester_motif_constraints(
            predictions,
            index,
            record,
            atom_vocabulary,
            parents,
            closure_left,
            closure_right,
            neighbors,
            ugi_ester_chemotype_policy,
            distances_to_core=distances_to_core,
            ring_nodes=ring_nodes,
            ring_edges=ring_edges,
            all_role_semantic_target=ugi_all_role_semantic_target,
            semantic_guidance_policy=ugi_mog_semantic_guidance_policy,
            terminal_generator=terminal_generator,
        )
        if motif is None:
            return None, "ugi_aldehyde_ester_topology_unavailable"
        forced_atom_symbols, forced_bonds = motif

    if ugi_topology_policy is not None:
        topology_failure = _ugi_topology_support_failure(
            record,
            parents=parents,
            closure_left=closure_left,
            closure_right=closure_right,
            policy=ugi_topology_policy,
            ester_policy=ugi_ester_chemotype_policy,
        )
        if topology_failure is not None:
            return None, topology_failure
    if topology_only:
        return {
            "parents": parents,
            "closure_left": closure_left,
            "closure_right": closure_right,
        }, None

    preselected_atom_states: dict[int, int] = {}
    if ugi_ester_chemotype_policy is not None and local_chemistry_support is not None:
        selected_amine, amine_failure = _select_ugi_amine_atom_states(
            predictions,
            index,
            record,
            atom_vocabulary,
            atom_capacities,
            minimum_used,
            neighbors,
            triangles,
            generated_cycles,
            role_names,
            local_chemistry_support,
            ugi_ester_chemotype_policy,
            chemistry_prior=ugi_role_chemistry_prior,
            chemistry_prior_strength=ugi_role_chemistry_prior_strength,
            distances_to_core=distances_to_core,
            ring_nodes=ring_nodes,
            ring_edges=ring_edges,
            semantic_target=effective_amine_semantic_target,
            all_role_semantic_target=ugi_all_role_semantic_target,
            semantic_guidance_policy=ugi_mog_semantic_guidance_policy,
            terminal_generator=terminal_generator,
        )
        if selected_amine is None:
            return None, amine_failure or "ugi_amine_global_atom_assignment_unavailable"
        preselected_atom_states.update(selected_amine)

    node_states = np.zeros(count, dtype=np.int64)
    capacities = np.zeros(count, dtype=np.int64)
    assigned = np.zeros(count, dtype=np.bool_)
    block_by_node = {
        node: block for block in record.component_blocks for node in range(block.start, block.stop)
    }
    if local_chemistry_support is not None:
        for block in record.component_blocks:
            bounds = local_chemistry_support.component_support_bounds(record.program_id, block.role)
            if not bounds.heavy_atoms_min <= block.atom_count <= bounds.heavy_atoms_max:
                return None, "component_heavy_atoms_outside_observed_local_support"
    for node in range(count):
        valid = atom_capacities >= minimum_used[node]
        if (
            ugi_ester_chemotype_policy is not None
            and record.program_id == ugi_ester_chemotype_policy.reaction_id
            and int(record.core_position_states[node]) == 1
        ):
            role = role_names[node]
            allowed_symbols = (
                ({"C", "N", "O"} if effective_amine_semantic_target is not None else {"C", "N"})
                if role == ugi_ester_chemotype_policy.amine_role
                else (
                    {forced_atom_symbols.get(node, "C")}
                    if role == ugi_ester_chemotype_policy.aldehyde_role
                    else ({"C"} if role == ugi_ester_chemotype_policy.isocyanide_role else None)
                )
            )
            if allowed_symbols is not None:
                valid &= np.asarray(
                    [
                        atom.symbol in allowed_symbols
                        and atom.formal_charge == 0
                        and not atom.aromatic
                        for atom in atom_vocabulary
                    ],
                    dtype=np.bool_,
                )
        if node in forced_atom_symbols:
            required_symbol = forced_atom_symbols[node]
            valid &= np.asarray(
                [
                    atom.symbol == required_symbol and atom.formal_charge == 0 and not atom.aromatic
                    for atom in atom_vocabulary
                ],
                dtype=np.bool_,
            )
        pre_local_valid = valid.copy()
        if local_chemistry_support is not None:
            block = block_by_node[node]
            bounds = local_chemistry_support.component_support_bounds(record.program_id, block.role)
            assigned_symbols = [
                atom_vocabulary[int(node_states[index])].symbol
                for index in range(block.start, node)
            ]
            assigned_carbons = assigned_symbols.count("C")
            assigned_heteroatoms = len(assigned_symbols) - assigned_carbons
            remaining_after_node = block.stop - node - 1
            for state, atom in enumerate(atom_vocabulary):
                if not valid[state]:
                    continue
                carbon_atoms = assigned_carbons + int(atom.symbol == "C")
                heteroatoms = assigned_heteroatoms + int(atom.symbol != "C")
                if (
                    carbon_atoms > bounds.carbon_atoms_max
                    or carbon_atoms + remaining_after_node < bounds.carbon_atoms_min
                    or heteroatoms > bounds.heteroatoms_max
                    or heteroatoms + remaining_after_node < bounds.heteroatoms_min
                ):
                    valid[state] = False
                    continue
                for neighbor in neighbors[node]:
                    if not assigned[neighbor]:
                        continue
                    neighbor_atom = atom_vocabulary[int(node_states[neighbor])]
                    if not local_chemistry_support.allows_role_edge_for_any_bond(
                        record.program_id,
                        role_names[node],
                        atom.symbol,
                        role_names[neighbor],
                        neighbor_atom.symbol,
                    ):
                        valid[state] = False
                        break
                if not valid[state]:
                    continue
                for triangle in triangles:
                    if node not in triangle:
                        continue
                    others = tuple(value for value in triangle if value != node)
                    if not all(assigned[value] for value in others):
                        continue
                    signature = (
                        (role_names[node], atom.symbol),
                        *(
                            (
                                role_names[value],
                                atom_vocabulary[int(node_states[value])].symbol,
                            )
                            for value in others
                        ),
                    )
                    if not local_chemistry_support.allows_role_triangle(
                        record.program_id, signature
                    ):
                        valid[state] = False
                        break
                if not valid[state] or not local_chemistry_support.enforces_role_cycles:
                    continue
                for cycle in generated_cycles:
                    if node not in cycle:
                        continue
                    others = tuple(value for value in cycle if value != node)
                    if not all(assigned[value] for value in others):
                        continue
                    signature = (
                        (role_names[node], atom.symbol),
                        *(
                            (
                                role_names[value],
                                atom_vocabulary[int(node_states[value])].symbol,
                            )
                            for value in others
                        ),
                    )
                    if not local_chemistry_support.allows_role_cycle(record.program_id, signature):
                        valid[state] = False
                        break
        if record.fixed_atom_mask[node]:
            state = int(record.graph.node_states[node])
            if state >= len(atom_vocabulary) or not valid[state]:
                reason = (
                    "fixed_atom_local_chemistry_exceeds_support"
                    if local_chemistry_support is not None
                    else "fixed_atom_valence_exceeds_support"
                )
                return None, reason
        elif topology_states is not None:
            state = int(topology_states[node])
            if not valid[state]:
                return None, "atom_aware_chosen_state_exceeds_chemistry_support"
        elif node in preselected_atom_states:
            state = int(preselected_atom_states[node])
            if state >= len(atom_vocabulary) or not valid[state]:
                return None, "ugi_amine_global_atom_assignment_invariant_failure"
        else:
            node_logits = predictions["nodes"][index, node]
            mog_local_scores: np.ndarray | None = None
            if (
                ugi_mog_semantic_guidance_policy is not None
                and ugi_mog_semantic_guidance_policy.uses_coordinate_local_chemistry_atoms
                and record.program_id == UGI_PROGRAM_ID
                and int(record.core_position_states[node]) == 1
            ):
                chemistry_prior = ugi_mog_semantic_guidance_policy.local_chemistry_prior
                if chemistry_prior is None:
                    return None, "ugi_local_chemistry_reference_unavailable"
                mog_local_scores = ugi_mog_semantic_guidance_policy.atom_local_scores(
                    role=role_names[node],
                    depth=int(distances_to_core[node]),
                    degree=len(neighbors[node]),
                    in_ring=node in ring_nodes,
                    atom_vocabulary=atom_vocabulary,
                )
            if (
                ugi_role_chemistry_prior is not None
                and ugi_role_chemistry_prior_strength > 0
                and record.program_id == ugi_role_chemistry_prior.reaction_id
                and int(record.core_position_states[node]) == 1
            ):
                node_logits = node_logits + ugi_role_chemistry_prior_strength * (
                    ugi_role_chemistry_prior.atom_log_bias(
                        role=role_names[node],
                        depth=int(distances_to_core[node]),
                        degree=len(neighbors[node]),
                        in_ring=node in ring_nodes,
                        atom_vocabulary=atom_vocabulary,
                    )
                )
            if mog_local_scores is not None:
                if terminal_generator is None:
                    return None, "ugi_local_chemistry_guidance_generator_unavailable"
                selected = _sample_mog_chemistry_allowed(
                    node_logits,
                    valid,
                    mog_local_scores,
                    policy=ugi_mog_semantic_guidance_policy,
                    generator=terminal_generator,
                )
            else:
                deterministic_terminal_chemistry = (
                    ugi_mog_semantic_guidance_policy is not None
                    and not ugi_mog_semantic_guidance_policy.uses_ranked_terminal_chemistry
                )
                selected = (
                    _argmax_allowed(node_logits, valid)
                    if terminal_generator is None or deterministic_terminal_chemistry
                    else _sample_allowed(
                        node_logits,
                        valid,
                        generator=terminal_generator,
                        temperature=terminal_temperature,
                    )
                )
            if selected is None:
                if local_chemistry_support is None:
                    reason = "atom_valence_state_unavailable"
                elif not np.any(pre_local_valid):
                    reason = "atom_chemotype_or_valence_state_unavailable"
                else:
                    block = block_by_node[node]
                    bounds = local_chemistry_support.component_support_bounds(
                        record.program_id, block.role
                    )
                    assigned_symbols = [
                        atom_vocabulary[int(node_states[index])].symbol
                        for index in range(block.start, node)
                    ]
                    assigned_carbons = assigned_symbols.count("C")
                    assigned_heteroatoms = len(assigned_symbols) - assigned_carbons
                    remaining_after_node = block.stop - node - 1
                    count_feasible = pre_local_valid.copy()
                    for candidate_state, atom in enumerate(atom_vocabulary):
                        if not count_feasible[candidate_state]:
                            continue
                        carbon_atoms = assigned_carbons + int(atom.symbol == "C")
                        heteroatoms = assigned_heteroatoms + int(atom.symbol != "C")
                        if (
                            carbon_atoms > bounds.carbon_atoms_max
                            or carbon_atoms + remaining_after_node < bounds.carbon_atoms_min
                            or heteroatoms > bounds.heteroatoms_max
                            or heteroatoms + remaining_after_node < bounds.heteroatoms_min
                        ):
                            count_feasible[candidate_state] = False
                    reason = (
                        f"atom_component_count_state_unavailable:{block.role}"
                        if not np.any(count_feasible)
                        else f"atom_local_graph_state_unavailable:{block.role}"
                    )
                return None, reason
            state = selected
        node_states[node] = state
        capacities[node] = atom_capacities[state]
        assigned[node] = True

    if local_chemistry_support is not None:
        for block in record.component_blocks:
            symbols = [
                atom_vocabulary[int(node_states[node])].symbol
                for node in range(block.start, block.stop)
            ]
            carbon_atoms = symbols.count("C")
            heavy_atoms = len(symbols)
            if not local_chemistry_support.component_is_within_observed_support(
                record.program_id,
                block.role,
                heavy_atoms=heavy_atoms,
                carbon_atoms=carbon_atoms,
                heteroatoms=heavy_atoms - carbon_atoms,
            ):
                return None, "component_outside_observed_local_support"

    if required_core_units is not None:
        # Bond-order selection below must not spend a pinned core position's stated valence on a
        # higher bond order either, so the realized capacities carry the same ceiling.
        capacities = np.minimum(capacities, maximum_capacities)
    if ugi_all_role_semantic_target is not None:
        if local_chemistry_support is None:
            return None, "ugi_all_role_semantic_local_support_unavailable"
        selected_tail_bonds, tail_bond_failure = _select_ugi_all_role_tail_bonds(
            predictions,
            index,
            record,
            atom_vocabulary,
            node_states,
            parents,
            closure_left,
            closure_right,
            minimum_used,
            capacities,
            bond_units,
            role_names,
            local_chemistry_support,
            forced_bonds,
            ugi_all_role_semantic_target,
            distances_to_core=distances_to_core,
            ring_edges=ring_edges,
            semantic_guidance_policy=ugi_mog_semantic_guidance_policy,
            terminal_generator=terminal_generator,
        )
        if selected_tail_bonds is None:
            return None, tail_bond_failure or "ugi_all_role_tail_bond_assignment_unavailable"
        forced_bonds = selected_tail_bonds
    used = minimum_used.copy()
    variable_edges: list[tuple[str, int, int, int]] = []
    for child in range(1, count):
        if not record.fixed_parent_bond_mask[child]:
            variable_edges.append(("parent", child, int(parents[child]), child))
    for slot in range(closure_count):
        if not record.fixed_closure_bond_mask[slot]:
            variable_edges.append(
                ("closure", slot, int(closure_left[slot]), int(closure_right[slot]))
            )
    for kind, slot, left, right in variable_edges:
        spare = min(int(capacities[left] - used[left]), int(capacities[right] - used[right]))
        valid = bond_units <= 2 + spare
        forced_bond = forced_bonds.get(tuple(sorted((left, right))))
        if forced_bond is not None:
            valid &= np.arange(len(valid), dtype=np.int64) == forced_bond
        for bond, units in enumerate(bond_units):
            if units == 3 and not (
                atom_vocabulary[node_states[left]].aromatic
                and atom_vocabulary[node_states[right]].aromatic
            ):
                valid[bond] = False
        if local_chemistry_support is not None:
            left_atom = atom_vocabulary[int(node_states[left])]
            right_atom = atom_vocabulary[int(node_states[right])]
            for bond in range(len(valid)):
                if valid[bond] and not local_chemistry_support.allows_role_edge(
                    record.program_id,
                    role_names[left],
                    left_atom.symbol,
                    bond,
                    role_names[right],
                    right_atom.symbol,
                ):
                    valid[bond] = False
        logits = predictions["parent_bonds" if kind == "parent" else "closure_bonds"][index, slot]
        mog_local_scores: np.ndarray | None = None
        whole_head_amine_bond_support = (
            ugi_mog_semantic_guidance_policy is not None
            and ugi_mog_semantic_guidance_policy.local_chemistry_score_mode
            in {
                "whole_head_support_tier",
                "whole_head_support_binary",
                "whole_head_support_distance",
            }
            and ugi_mog_semantic_guidance_policy.whole_head_hard_bond_support
            and record.program_id == UGI_PROGRAM_ID
            and role_names[left] == role_names[right] == "amine_head"
            and int(record.core_position_states[left]) == 1
            and int(record.core_position_states[right]) == 1
        )
        ranked_local_bond = (
            ugi_mog_semantic_guidance_policy is not None
            and ugi_mog_semantic_guidance_policy.uses_local_chemistry_bonds
            and record.program_id == UGI_PROGRAM_ID
            and role_names[left] == role_names[right]
            and int(record.core_position_states[left]) == 1
            and int(record.core_position_states[right]) == 1
        )
        if whole_head_amine_bond_support or ranked_local_bond:
            assert ugi_mog_semantic_guidance_policy is not None
            chemistry_prior = ugi_mog_semantic_guidance_policy.local_chemistry_prior
            if chemistry_prior is None:
                return None, "ugi_local_chemistry_reference_unavailable"
            local_bond_support = ugi_mog_semantic_guidance_policy.bond_local_scores(
                role=role_names[left],
                depth=min(int(distances_to_core[left]), int(distances_to_core[right])),
                in_ring=tuple(sorted((left, right))) in ring_edges,
                symbols=(
                    atom_vocabulary[int(node_states[left])].symbol,
                    atom_vocabulary[int(node_states[right])].symbol,
                ),
                bond_classes=len(valid),
            )
            if whole_head_amine_bond_support:
                valid &= local_bond_support >= (
                    ugi_mog_semantic_guidance_policy.minimum_whole_head_support_tier
                )
            if ranked_local_bond:
                mog_local_scores = local_bond_support
        if (
            ugi_role_chemistry_prior is not None
            and ugi_role_chemistry_prior_strength > 0
            and record.program_id == ugi_role_chemistry_prior.reaction_id
            and role_names[left] == role_names[right]
            and int(record.core_position_states[left]) == 1
            and int(record.core_position_states[right]) == 1
        ):
            logits = logits + ugi_role_chemistry_prior_strength * (
                ugi_role_chemistry_prior.bond_log_bias(
                    role=role_names[left],
                    depth=min(int(distances_to_core[left]), int(distances_to_core[right])),
                    in_ring=tuple(sorted((left, right))) in ring_edges,
                    symbols=(
                        atom_vocabulary[int(node_states[left])].symbol,
                        atom_vocabulary[int(node_states[right])].symbol,
                    ),
                    bond_classes=len(valid),
                )
            )
        if mog_local_scores is not None:
            if terminal_generator is None:
                return None, "ugi_local_chemistry_guidance_generator_unavailable"
            bond = _sample_mog_chemistry_allowed(
                logits,
                valid,
                mog_local_scores,
                policy=ugi_mog_semantic_guidance_policy,
                generator=terminal_generator,
                rank_weight=(
                    ugi_mog_semantic_guidance_policy.effective_local_chemistry_bond_rank_weight
                ),
            )
        else:
            deterministic_terminal_chemistry = (
                ugi_mog_semantic_guidance_policy is not None
                and not ugi_mog_semantic_guidance_policy.uses_ranked_terminal_chemistry
            )
            deterministic_terminal_bonds = (
                ugi_mog_semantic_guidance_policy is not None
                and not ugi_mog_semantic_guidance_policy.uses_ranked_terminal_bonds
            )
            bond = (
                _argmax_allowed(logits, valid)
                if (
                    terminal_generator is None
                    or deterministic_terminal_chemistry
                    or deterministic_terminal_bonds
                )
                else _sample_allowed(
                    logits,
                    valid,
                    generator=terminal_generator,
                    temperature=terminal_temperature,
                )
            )
        if bond is None:
            qualifier = "local_chemistry" if local_chemistry_support is not None else "valence"
            return None, f"{kind}_bond_{qualifier}_unavailable"
        extra = int(bond_units[bond]) - 2
        used[[left, right]] += extra
        if kind == "parent":
            parent_bonds[slot] = bond
        else:
            closure_bonds[slot] = bond
    if np.any(used > capacities):
        return None, "terminal_valence_overflow"
    if required_core_units is not None:
        pinned = required_core_units >= 0
        if np.any(used[pinned] != required_core_units[pinned]):
            # Masking makes over-substitution unreachable; this catches under-substitution, which
            # would leave the core atom with an extra hydrogen and equally break the transform.
            return None, "reaction_core_saturation_unmet"
    if local_chemistry_support is not None:
        symbols = tuple(atom_vocabulary[int(state)].symbol for state in node_states)
        terminal_edges = [
            (int(parents[child]), child, int(parent_bonds[child])) for child in range(1, count)
        ]
        terminal_edges.extend(
            (int(closure_left[slot]), int(closure_right[slot]), int(closure_bonds[slot]))
            for slot in range(closure_count)
        )
        if any(
            not local_chemistry_support.allows_role_edge(
                record.program_id,
                role_names[left],
                symbols[left],
                bond,
                role_names[right],
                symbols[right],
            )
            for left, right, bond in terminal_edges
        ):
            return None, "terminal_edge_outside_observed_local_support"
        if any(
            not local_chemistry_support.allows_role_triangle(
                record.program_id,
                ((role_names[index], symbols[index]) for index in triangle),
            )
            for triangle in triangles
        ):
            return None, "terminal_triangle_outside_observed_local_support"
        if local_chemistry_support.enforces_role_cycles and any(
            not local_chemistry_support.allows_role_cycle(
                record.program_id,
                ((role_names[index], symbols[index]) for index in cycle),
            )
            for cycle in generated_cycles
        ):
            return None, "terminal_cycle_outside_observed_role_morphology_support"
    return {
        "nodes": node_states,
        "parents": parents,
        "parent_bonds": parent_bonds,
        "closure_left": closure_left,
        "closure_right": closure_right,
        "closure_bonds": closure_bonds,
    }, None


def decode_synthesis_program_strict_argmax(
    predictions: Mapping[str, Any],
    layout: Mapping[str, Any],
    records: Sequence[SynthesisProgramGraphRecord],
    atom_vocabulary: Sequence[AtomState],
    local_chemistry_support: LocalChemistrySupport | None = None,
    *,
    enforce_program_topology: bool = False,
    enforce_program_cycles: bool = False,
    confine_generated_edges: bool = False,
    core_saturation: BoundReactionCoreSaturation | None = None,
    terminal_generator: np.random.Generator | None = None,
    terminal_temperature: float = 1.0,
    ugi_ester_chemotype_policy: UgiEsterChemotypePolicy | None = None,
    ugi_role_chemistry_prior: UgiRoleChemistryPrior | None = None,
    ugi_role_chemistry_prior_strength: float = 0.0,
    ugi_topology_policy: UgiTransformerTopologyPolicy | None = None,
    ugi_amine_semantic_targets: Sequence[UgiAmineSemanticTarget | None] | None = None,
    ugi_all_role_semantic_targets: Sequence[UgiAllRoleSemanticTarget | None] | None = None,
    ugi_mog_semantic_guidance_policy: UgiMogSemanticGuidancePolicy | None = None,
    qualified_core_units: Sequence[np.ndarray] | None = None,
    confine_origin_edges: bool = False,
    reserve_fixed_closures: bool = False,
    origin_closure_roles: Sequence[Sequence[int]] | None = None,
    origin_ring_sizes: Sequence[Mapping[int, Sequence[int]]] | None = None,
    exact_origin_morphology: bool = False,
    topology_only: bool = False,
    atom_aware_topology: bool = False,
) -> tuple[TerminalTensorState, tuple[str | None, ...]]:
    """Decode once under strict support; infeasible attempts abstain and are never repaired."""

    if len(records) != int(layout["node_mask"].shape[0]):
        raise SynthesisProgramSamplingError("strict decoder batch and record counts disagree")
    for value in (origin_closure_roles, origin_ring_sizes):
        if value is not None and len(value) != len(records):
            raise SynthesisProgramSamplingError("Origin constraints and records disagree")
    if qualified_core_units is not None and len(qualified_core_units) != len(records):
        raise SynthesisProgramSamplingError("qualified core conditions and records differ")
    has_amine_targets = ugi_amine_semantic_targets is not None and any(
        target is not None for target in ugi_amine_semantic_targets
    )
    has_all_role_targets = ugi_all_role_semantic_targets is not None and any(
        target is not None for target in ugi_all_role_semantic_targets
    )
    if has_amine_targets and has_all_role_targets:
        raise SynthesisProgramSamplingError(
            "amine-only and all-role semantic target batches are mutually exclusive"
        )
    if ugi_amine_semantic_targets is None:
        semantic_targets: tuple[UgiAmineSemanticTarget | None, ...] = (None,) * len(records)
    else:
        semantic_targets = tuple(ugi_amine_semantic_targets)
        if len(semantic_targets) != len(records):
            raise SynthesisProgramSamplingError(
                "strict decoder semantic-target and record counts disagree"
            )
    if ugi_all_role_semantic_targets is None:
        all_role_targets: tuple[UgiAllRoleSemanticTarget | None, ...] = (None,) * len(records)
    else:
        all_role_targets = tuple(ugi_all_role_semantic_targets)
        if len(all_role_targets) != len(records):
            raise SynthesisProgramSamplingError(
                "strict decoder all-role-target and record counts disagree"
            )
    cpu_predictions = {
        key: value.detach().to("cpu").numpy()
        for key, value in predictions.items()
        if key
        in {
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        }
    }
    # Strict decoding is an RDKit/NumPy terminal operation.  Keep its output on CPU: the previous
    # implementation copied predictions GPU->CPU, then performed thousands of tiny decoded-state
    # copies CPU->GPU only for the caller to immediately copy every state GPU->CPU again.
    cpu_layout = {
        key: value.detach().to("cpu")
        for key, value in layout.items()
        if key
        in {
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
            "fixed_atom_mask",
            "fixed_parent_mask",
            "fixed_parent_bond_mask",
            "fixed_closure_endpoint_mask",
            "fixed_closure_bond_mask",
        }
    }
    terminal = {
        field: torch.zeros_like(cpu_layout[field])
        for field in (
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        )
    }
    terminal = restore_synthesis_program_fixed_states(terminal, cpu_layout)
    reasons: list[str | None] = []
    for index, (record, semantic_target, all_role_target) in enumerate(
        zip(records, semantic_targets, all_role_targets, strict=True)
    ):
        decoded, reason = _strict_terminal_record(
            cpu_predictions,
            index,
            record,
            atom_vocabulary,
            local_chemistry_support,
            enforce_program_topology=enforce_program_topology,
            enforce_program_cycles=enforce_program_cycles,
            confine_generated_edges=confine_generated_edges,
            core_saturation=core_saturation,
            terminal_generator=terminal_generator,
            terminal_temperature=terminal_temperature,
            ugi_ester_chemotype_policy=ugi_ester_chemotype_policy,
            ugi_role_chemistry_prior=ugi_role_chemistry_prior,
            ugi_role_chemistry_prior_strength=ugi_role_chemistry_prior_strength,
            ugi_topology_policy=ugi_topology_policy,
            ugi_amine_semantic_target=semantic_target,
            ugi_all_role_semantic_target=all_role_target,
            ugi_mog_semantic_guidance_policy=ugi_mog_semantic_guidance_policy,
            qualified_core_units=(
                None if qualified_core_units is None else qualified_core_units[index]
            ),
            confine_origin_edges=confine_origin_edges,
            reserve_fixed_closures=reserve_fixed_closures,
            origin_closure_roles=(
                None if origin_closure_roles is None else origin_closure_roles[index]
            ),
            origin_ring_sizes=None if origin_ring_sizes is None else origin_ring_sizes[index],
            exact_origin_morphology=exact_origin_morphology,
            topology_only=topology_only,
            atom_aware_topology=atom_aware_topology,
        )
        reasons.append(reason)
        if decoded is None:
            continue
        for field, values in cast(Mapping[str, np.ndarray], decoded).items():
            terminal[field][index, : len(values)] = torch.as_tensor(
                values, dtype=terminal[field].dtype
            )
    return cast(
        TerminalTensorState, restore_synthesis_program_fixed_states(terminal, cpu_layout)
    ), tuple(reasons)


def decode_synthesis_program_strict_topology(
    predictions: Mapping[str, Any],
    records: Sequence[SynthesisProgramGraphRecord],
    atom_vocabulary: Sequence[AtomState],
    local_chemistry_support: LocalChemistrySupport,
    *,
    core_saturation: BoundReactionCoreSaturation,
    ugi_topology_policy: UgiTransformerTopologyPolicy,
    ugi_ester_chemotype_policy: UgiEsterChemotypePolicy,
) -> tuple[dict[str, Any], tuple[str | None, ...]]:
    """Select one learned, support-constrained topology per Ugi attempt without chemistry.

    Parent and closure scores come from the Transformer after its ordinary joint flow. The
    provisional chemistry is discarded. Invalid topology attempts abstain once; no alternate
    topology is proposed and no accepted state is repaired.
    """

    cpu_predictions = {
        key: value.detach().to("cpu").numpy()
        for key, value in predictions.items()
        if key
        in {
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        }
    }
    maximum_nodes = max(record.node_count for record in records)
    maximum_closures = max(record.graph.closure_count for record in records)
    topology = {
        "parents": torch.zeros((len(records), maximum_nodes), dtype=torch.long),
        "closure_left": torch.zeros((len(records), maximum_closures), dtype=torch.long),
        "closure_right": torch.zeros((len(records), maximum_closures), dtype=torch.long),
    }
    reasons: list[str | None] = []
    for index, record in enumerate(records):
        decoded, reason = _strict_terminal_record(
            cpu_predictions,
            index,
            record,
            atom_vocabulary,
            local_chemistry_support,
            enforce_program_topology=True,
            enforce_program_cycles=True,
            confine_generated_edges=True,
            core_saturation=core_saturation,
            ugi_ester_chemotype_policy=ugi_ester_chemotype_policy,
            ugi_topology_policy=ugi_topology_policy,
            topology_only=True,
        )
        reasons.append(reason)
        if decoded is None:
            continue
        for field in topology:
            values = cast(Mapping[str, np.ndarray], decoded)[field]
            topology[field][index, : len(values)] = torch.as_tensor(values, dtype=torch.long)
    return topology, tuple(reasons)
