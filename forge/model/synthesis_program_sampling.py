"""Fixed-state-safe sampling for the shared Ugi/BL/LX whole-product flow."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from forge.core.io import read_json_object
from forge.flow import rstar_step
from forge.model.defog_feasibility import AtomState, _model_state_sha256, graph_to_molecule
from forge.model.local_chemistry_support import LocalChemistrySupport, tree_path_indices
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_flow import (
    collate_synthesis_program_layouts,
    decode_synthesis_program_argmax,
    restore_synthesis_program_fixed_states,
)
from forge.model.sparse_topology_feasibility import (
    BOND_VALENCE_UNITS,
    INDEX_TO_DENSE_BOND,
    _endpoint_candidate_mask,
    _maximum_valence_units,
    _parent_candidate_mask,
    pointer_rstar_step,
)
from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord
from forge.model.synthesis_program_training import build_synthesis_program_flow
from forge.model.tensor_checkpoint import TensorCheckpointError, decode_tensor_state

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]

CHECKPOINT_SCHEMA = "forge.synthesis_program_sparse_flow_checkpoint.v1"
TERMINAL_DECODE_POLICIES = (
    "unconstrained_argmax",
    "strict_valence_topology_argmax",
)
LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY = "strict_local_chemistry_argmax"
SUPPORTED_TERMINAL_DECODE_POLICIES = (
    *TERMINAL_DECODE_POLICIES,
    LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY,
)


class SynthesisProgramSamplingError(ValueError):
    """A shared synthesis-program sampling request violates its fixed-state contract."""


def synthesis_program_source_marginals(
    records: Sequence[SynthesisProgramGraphRecord],
    weights: np.ndarray,
    *,
    node_classes: int,
    bond_classes: int,
    probability_floor: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit full-support sources under explicit record weights, never raw family counts."""

    if (
        not records
        or weights.shape != (len(records),)
        or np.any(weights <= 0)
        or not np.isfinite(weights).all()
        or not np.isclose(weights.sum(), 1.0)
        or probability_floor <= 0
    ):
        raise SynthesisProgramSamplingError("source marginals require a finite normalized measure")
    nodes = np.full(node_classes, probability_floor, dtype=np.float64)
    bonds = np.full(bond_classes, probability_floor, dtype=np.float64)
    for record, weight in zip(records, weights, strict=True):
        variable_atoms = record.graph.node_states[~record.fixed_atom_mask]
        variable_parent_bonds = record.graph.parent_bonds[1:][~record.fixed_parent_bond_mask[1:]]
        variable_closure_bonds = record.graph.closure_bonds[~record.fixed_closure_bond_mask]
        nodes += float(weight) * np.bincount(variable_atoms, minlength=node_classes)
        bonds += float(weight) * np.bincount(variable_parent_bonds, minlength=bond_classes)
        bonds += float(weight) * np.bincount(variable_closure_bonds, minlength=bond_classes)
    return nodes / nodes.sum(), bonds / bonds.sum()


def load_synthesis_program_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
) -> tuple[
    Any,
    ReactionProgramVocabulary,
    tuple[AtomState, ...],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Load an authenticated non-executable tensor checkpoint."""

    if torch is None:
        raise SynthesisProgramSamplingError("checkpoint loading requires torch")
    package = read_json_object(
        checkpoint_path,
        error=SynthesisProgramSamplingError,
        label="shared synthesis-program checkpoint",
    )
    if (
        package.get("schema_version") != CHECKPOINT_SCHEMA
        or package.get("trusted_local_checkpoint") is not True
    ):
        raise SynthesisProgramSamplingError("checkpoint is not a trusted shared-flow checkpoint")
    raw_vocabulary = package.get("program_vocabulary")
    raw_atoms = package.get("atom_vocabulary")
    model_config = package.get("model_config")
    if (
        not isinstance(raw_vocabulary, Mapping)
        or not isinstance(raw_atoms, list)
        or not isinstance(model_config, Mapping)
    ):
        raise SynthesisProgramSamplingError("checkpoint is missing its model contract")
    vocabulary = ReactionProgramVocabulary(
        program_states=tuple(str(value) for value in raw_vocabulary["program_states"]),
        role_states=tuple(str(value) for value in raw_vocabulary["role_states"]),
        core_position_states=tuple(str(value) for value in raw_vocabulary["core_position_states"]),
        maximum_steps=int(raw_vocabulary["maximum_steps"]),
    )
    atom_vocabulary = tuple(
        AtomState(
            symbol=str(row["symbol"]),
            formal_charge=int(row["formal_charge"]),
            aromatic=bool(row["aromatic"]),
            explicit_hydrogens=int(row["explicit_hydrogens"]),
        )
        for row in raw_atoms
    )
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise SynthesisProgramSamplingError("CUDA sampling requested but unavailable")
    if resolved_device.type == "mps" and not torch.backends.mps.is_available():
        raise SynthesisProgramSamplingError("MPS sampling requested but unavailable")
    model = build_synthesis_program_flow(
        vocabulary=vocabulary,
        node_classes=len(atom_vocabulary),
        model_config=model_config,
        device=resolved_device,
    )
    raw_state = package.get("model_state")
    if not isinstance(raw_state, Mapping):
        raise SynthesisProgramSamplingError("checkpoint has no deterministic tensor state")
    try:
        state = decode_tensor_state(raw_state)
    except TensorCheckpointError as error:
        raise SynthesisProgramSamplingError(str(error)) from error
    model.load_state_dict(state, strict=True)
    if _model_state_sha256(model) != package.get("model_state_sha256"):
        raise SynthesisProgramSamplingError("checkpoint model-state hash mismatch")
    node_marginal = np.asarray(package.get("node_marginal"), dtype=np.float64)
    bond_marginal = np.asarray(package.get("bond_marginal"), dtype=np.float64)
    if (
        node_marginal.shape != (len(atom_vocabulary),)
        or bond_marginal.shape != (int(model_config["bond_classes"]),)
        or np.any(node_marginal <= 0)
        or np.any(bond_marginal <= 0)
        or not np.isclose(node_marginal.sum(), 1.0)
        or not np.isclose(bond_marginal.sum(), 1.0)
    ):
        raise SynthesisProgramSamplingError("checkpoint source marginals are invalid")
    model.eval()
    return model, vocabulary, atom_vocabulary, node_marginal, bond_marginal, package


def _move(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def _initial_state(
    layout: Mapping[str, Any],
    node_marginal: Any,
    bond_marginal: Any,
    generator: Any,
) -> dict[str, Any]:
    batch, nodes = layout["node_mask"].shape
    closures = layout["closure_mask"].shape[1]
    state = {
        "nodes": torch.multinomial(
            node_marginal, batch * nodes, replacement=True, generator=generator
        ).reshape(batch, nodes),
        "parent_bonds": torch.multinomial(
            bond_marginal, batch * nodes, replacement=True, generator=generator
        ).reshape(batch, nodes),
        "closure_bonds": torch.multinomial(
            bond_marginal, batch * closures, replacement=True, generator=generator
        ).reshape(batch, closures),
    }
    parent_candidates = _parent_candidate_mask(layout["node_mask"])
    parent_probabilities = parent_candidates.to(torch.float32)
    parent_probabilities /= parent_probabilities.sum(dim=-1, keepdim=True).clamp(min=1)
    state["parents"] = torch.zeros((batch, nodes), dtype=torch.long, device=node_marginal.device)
    active_parents = layout["parent_variable_mask"]
    state["parents"][active_parents] = torch.multinomial(
        parent_probabilities[active_parents], 1, generator=generator
    ).squeeze(1)
    endpoint_candidates = _endpoint_candidate_mask(layout["node_mask"], closures)
    endpoint_probabilities = endpoint_candidates.to(torch.float32)
    endpoint_probabilities /= endpoint_probabilities.sum(dim=-1, keepdim=True).clamp(min=1)
    state["closure_left"] = torch.zeros(
        (batch, closures), dtype=torch.long, device=node_marginal.device
    )
    state["closure_right"] = torch.zeros_like(state["closure_left"])
    active_closures = layout["closure_endpoint_variable_mask"]
    for field in ("closure_left", "closure_right"):
        state[field][active_closures] = torch.multinomial(
            endpoint_probabilities[active_closures], 1, generator=generator
        ).squeeze(1)
    state["parent_bonds"][:, 0] = 0
    return restore_synthesis_program_fixed_states(state, layout)


def _fixed_state_exact(state: Mapping[str, Any], layout: Mapping[str, Any]) -> bool:
    fields = {
        "nodes": "fixed_atom_mask",
        "parents": "fixed_parent_mask",
        "parent_bonds": "fixed_parent_bond_mask",
        "closure_left": "fixed_closure_endpoint_mask",
        "closure_right": "fixed_closure_endpoint_mask",
        "closure_bonds": "fixed_closure_bond_mask",
    }
    return all(
        torch.equal(state[field][layout[mask]], layout[field][layout[mask]])
        for field, mask in fields.items()
    )


def _terminal_smiles(
    state: Mapping[str, Any],
    index: int,
    node_count: int,
    closure_count: int,
    atom_vocabulary: Sequence[AtomState],
) -> str | None:
    nodes = state["nodes"][index, :node_count].detach().cpu().numpy().astype(np.int64)
    parents = state["parents"][index, :node_count].detach().cpu().numpy().astype(np.int64)
    parent_bonds = state["parent_bonds"][index, :node_count].detach().cpu().numpy().astype(np.int64)
    left = state["closure_left"][index, :closure_count].detach().cpu().numpy().astype(np.int64)
    right = state["closure_right"][index, :closure_count].detach().cpu().numpy().astype(np.int64)
    closure_bonds = (
        state["closure_bonds"][index, :closure_count].detach().cpu().numpy().astype(np.int64)
    )
    edges = np.zeros((node_count, node_count), dtype=np.int64)
    try:
        for child in range(1, node_count):
            parent = int(parents[child])
            if parent < 0 or parent >= child:
                return None
            dense = INDEX_TO_DENSE_BOND[int(parent_bonds[child])]
            edges[child, parent] = edges[parent, child] = dense
        occupied = {tuple(sorted((child, int(parents[child])))) for child in range(1, node_count)}
        for slot in range(closure_count):
            pair = tuple(sorted((int(left[slot]), int(right[slot]))))
            if pair[0] < 0 or pair[1] >= node_count or pair[0] == pair[1] or pair in occupied:
                return None
            occupied.add(pair)
            dense = INDEX_TO_DENSE_BOND[int(closure_bonds[slot])]
            edges[pair[0], pair[1]] = edges[pair[1], pair[0]] = dense
        molecule = graph_to_molecule(nodes, edges, atom_vocabulary)
    except Exception:
        return None
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    # A graph may survive RDKit construction and serialization yet fail sanitizing parse on the
    # resulting SMILES.  Downstream metrics parse canonical strings again, so the sampler must not
    # label such a row valid and then fail the entire fixed-checkpoint evaluation later.
    reparsed = Chem.MolFromSmiles(smiles)
    if reparsed is None:
        return None
    return Chem.MolToSmiles(reparsed, canonical=True, isomericSmiles=False)


def _available_valence_units(state: AtomState) -> int:
    return _maximum_valence_units(state) - 2 * int(state.explicit_hydrogens)


def _argmax_allowed(logits: np.ndarray, valid: np.ndarray) -> int | None:
    if logits.ndim != 1 or valid.shape != logits.shape or not np.any(valid):
        return None
    masked = np.where(valid, logits, -np.inf)
    return int(np.argmax(masked))


def _strict_terminal_record(
    predictions: Mapping[str, np.ndarray],
    index: int,
    record: SynthesisProgramGraphRecord,
    atom_vocabulary: Sequence[AtomState],
    local_chemistry_support: LocalChemistrySupport | None = None,
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """Decode one exact-size graph under topology and valence support, without fallback."""

    count = record.node_count
    closure_count = record.graph.closure_count
    bond_classes = predictions["parent_bonds"].shape[-1]
    if BOND_VALENCE_UNITS is None or bond_classes > len(BOND_VALENCE_UNITS):
        return None, "unsupported_bond_vocabulary"
    bond_units = BOND_VALENCE_UNITS[:bond_classes].cpu().numpy().astype(np.int64)
    atom_capacities = np.asarray(
        [_available_valence_units(state) for state in atom_vocabulary], dtype=np.int64
    )
    maximum_capacity = int(atom_capacities.max())
    maximum_capacities = np.full(record.node_count, maximum_capacity, dtype=np.int64)
    for node in np.flatnonzero(record.fixed_atom_mask):
        state = int(record.graph.node_states[node])
        if state >= len(atom_vocabulary):
            return None, "fixed_atom_state_outside_vocabulary"
        maximum_capacities[node] = atom_capacities[state]
    parents = np.zeros(count, dtype=np.int64)
    parent_bonds = np.zeros(count, dtype=np.int64)
    closure_left = np.zeros(closure_count, dtype=np.int64)
    closure_right = np.zeros(closure_count, dtype=np.int64)
    closure_bonds = np.zeros(closure_count, dtype=np.int64)
    minimum_used = np.zeros(count, dtype=np.int64)
    degrees = np.zeros(count, dtype=np.int64)
    occupied: set[tuple[int, int]] = set()

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

    for child in range(1, count):
        if record.fixed_parent_bond_mask[child]:
            continue
        valid = np.zeros(count, dtype=np.bool_)
        for parent in range(child):
            valid[parent] = (
                minimum_used[child] + 2 <= maximum_capacities[child]
                and minimum_used[parent] + 2 <= maximum_capacities[parent]
            )
        parent = _argmax_allowed(predictions["parents"][index, child, :count], valid)
        if parent is None:
            return None, "parent_capacity_exhausted"
        parents[child] = parent
        degrees[[child, parent]] += 1
        minimum_used[[child, parent]] += 2
        occupied.add((parent, child))

    component_instances = np.zeros(count, dtype=np.int64)
    for component_index, block in enumerate(record.component_blocks, start=1):
        component_instances[block.start : block.stop] = component_index
    core = record.core_position_states
    role_by_state = {block.role_state: block.role for block in record.component_blocks}
    try:
        role_names = tuple(role_by_state[int(value)] for value in record.role_states)
    except KeyError:
        return None, "unnamed_semantic_role"

    for slot in np.flatnonzero(record.fixed_closure_bond_mask):
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
                if (
                    not semantic_pair
                    or minimum_used[left] + 2 > maximum_capacities[left]
                    or minimum_used[right] + 2 > maximum_capacities[right]
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
            return None, "closure_pair_unavailable"
        _, left, right = best
        degrees[[left, right]] += 1
        minimum_used[[left, right]] += 2
        occupied.add((left, right))
        topology_neighbors[left].add(right)
        topology_neighbors[right].add(left)

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
        else:
            selected = _argmax_allowed(predictions["nodes"][index, node], valid)
            if selected is None:
                reason = (
                    "atom_local_chemistry_state_unavailable"
                    if local_chemistry_support is not None
                    else "atom_valence_state_unavailable"
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
        bond = _argmax_allowed(logits, valid)
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
) -> tuple[dict[str, Any], tuple[str | None, ...]]:
    """Decode once under strict support; infeasible attempts abstain and are never repaired."""

    if len(records) != int(layout["node_mask"].shape[0]):
        raise SynthesisProgramSamplingError("strict decoder batch and record counts disagree")
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
    terminal = {
        field: torch.zeros_like(layout[field])
        for field in (
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        )
    }
    terminal = restore_synthesis_program_fixed_states(terminal, layout)
    reasons: list[str | None] = []
    for index, record in enumerate(records):
        decoded, reason = _strict_terminal_record(
            cpu_predictions,
            index,
            record,
            atom_vocabulary,
            local_chemistry_support,
        )
        reasons.append(reason)
        if decoded is None:
            continue
        for field, values in decoded.items():
            terminal[field][index, : len(values)] = torch.as_tensor(
                values, dtype=terminal[field].dtype, device=terminal[field].device
            )
    return restore_synthesis_program_fixed_states(terminal, layout), tuple(reasons)


def sample_synthesis_program_products(
    model: Any,
    records: Sequence[SynthesisProgramGraphRecord],
    atom_vocabulary: Sequence[AtomState],
    node_marginal: np.ndarray,
    bond_marginal: np.ndarray,
    *,
    samples_per_program: int,
    sample_steps: int,
    batch_size: int,
    seed: int,
    device: str,
    conditioning_mode: str = "program",
    program_state_mapping: Mapping[int, int] | None = None,
    role_state_mapping: Mapping[int, int] | None = None,
    terminal_decode_policy: str = "unconstrained_argmax",
    local_chemistry_support: LocalChemistrySupport | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate from semantic layouts while exposing only Ugi adapter-fixed graph states."""

    if (
        torch is None
        or not records
        or samples_per_program < 1
        or sample_steps < 2
        or batch_size < 1
        or terminal_decode_policy not in SUPPORTED_TERMINAL_DECODE_POLICIES
    ):
        raise SynthesisProgramSamplingError("invalid shared synthesis-program sampling request")
    if (terminal_decode_policy == LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY) != (
        local_chemistry_support is not None
    ):
        raise SynthesisProgramSamplingError(
            "strict local-chemistry decoding and its support policy must be supplied together"
        )
    resolved_device = torch.device(device)
    repeated = tuple(record for record in records for _ in range(samples_per_program))
    maximum_closures = int(model.maximum_closures)
    node_p0 = torch.as_tensor(node_marginal, dtype=torch.float32, device=resolved_device)
    bond_p0 = torch.as_tensor(bond_marginal, dtype=torch.float32, device=resolved_device)
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    outputs: list[dict[str, Any]] = []
    fixed_failures = 0
    strict_abstentions: Counter[str] = Counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(repeated), batch_size):
            local = repeated[offset : offset + batch_size]
            layout = _move(
                collate_synthesis_program_layouts(
                    local,
                    maximum_closures=maximum_closures,
                    conditioning_mode=conditioning_mode,
                    program_state_mapping=program_state_mapping,
                    role_state_mapping=role_state_mapping,
                ),
                resolved_device,
            )
            state = _initial_state(layout, node_p0, bond_p0, generator)
            fixed_failures += int(not _fixed_state_exact(state, layout))
            parent_candidates = _parent_candidate_mask(layout["node_mask"])
            endpoint_candidates = _endpoint_candidate_mask(layout["node_mask"], maximum_closures)
            for step in range(sample_steps):
                t_value = step / sample_steps
                t = torch.full((len(local),), t_value, device=resolved_device)
                predictions = model(
                    nodes=state["nodes"],
                    parents=state["parents"],
                    parent_bonds=state["parent_bonds"],
                    closure_left=state["closure_left"],
                    closure_right=state["closure_right"],
                    closure_bonds=state["closure_bonds"],
                    t=t,
                    node_mask=layout["node_mask"],
                    child_mask=layout["child_mask"],
                    closure_mask=layout["closure_mask"],
                    program_states=layout["program_states"],
                    role_states=layout["role_states"],
                    core_position_states=layout["core_position_states"],
                    program_depths=layout["program_depths"],
                    adapter_mask=layout["adapter_mask"],
                    repeat_group_states=layout["repeat_group_states"],
                    component_position_states=layout["component_position_states"],
                    component_instance_states=layout["component_instance_states"],
                )
                state["nodes"] = rstar_step(
                    state["nodes"],
                    predictions["nodes"].softmax(dim=-1),
                    node_p0,
                    t_value,
                    1.0 / sample_steps,
                    layout["atom_variable_mask"],
                    generator,
                )
                if bool(layout["parent_variable_mask"].any()):
                    state["parents"] = pointer_rstar_step(
                        state["parents"],
                        predictions["parents"],
                        parent_candidates,
                        layout["parent_variable_mask"],
                        t_value,
                        1.0 / sample_steps,
                        generator,
                    )
                for field, mask_name in (
                    ("parent_bonds", "parent_bond_variable_mask"),
                    ("closure_bonds", "closure_bond_variable_mask"),
                ):
                    state[field] = rstar_step(
                        state[field],
                        predictions[field].softmax(dim=-1),
                        bond_p0,
                        t_value,
                        1.0 / sample_steps,
                        layout[mask_name],
                        generator,
                    )
                if bool(layout["closure_endpoint_variable_mask"].any()):
                    for field in ("closure_left", "closure_right"):
                        state[field] = pointer_rstar_step(
                            state[field],
                            predictions[field],
                            endpoint_candidates,
                            layout["closure_endpoint_variable_mask"],
                            t_value,
                            1.0 / sample_steps,
                            generator,
                        )
                state = restore_synthesis_program_fixed_states(state, layout)
                fixed_failures += int(not _fixed_state_exact(state, layout))
            terminal_predictions = model(
                nodes=state["nodes"],
                parents=state["parents"],
                parent_bonds=state["parent_bonds"],
                closure_left=state["closure_left"],
                closure_right=state["closure_right"],
                closure_bonds=state["closure_bonds"],
                t=torch.ones(len(local), device=resolved_device),
                node_mask=layout["node_mask"],
                child_mask=layout["child_mask"],
                closure_mask=layout["closure_mask"],
                program_states=layout["program_states"],
                role_states=layout["role_states"],
                core_position_states=layout["core_position_states"],
                program_depths=layout["program_depths"],
                adapter_mask=layout["adapter_mask"],
                repeat_group_states=layout["repeat_group_states"],
                component_position_states=layout["component_position_states"],
                component_instance_states=layout["component_instance_states"],
            )
            if terminal_decode_policy in {
                "strict_valence_topology_argmax",
                LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY,
            }:
                terminal, abstention_reasons = decode_synthesis_program_strict_argmax(
                    terminal_predictions,
                    layout,
                    local,
                    atom_vocabulary,
                    local_chemistry_support,
                )
            else:
                terminal = decode_synthesis_program_argmax(terminal_predictions, layout)
                abstention_reasons = (None,) * len(local)
            fixed_failures += int(not _fixed_state_exact(terminal, layout))
            for index, record in enumerate(local):
                count = record.node_count
                closure_count = record.graph.closure_count
                abstention_reason = abstention_reasons[index]
                if abstention_reason is not None:
                    strict_abstentions[abstention_reason] += 1
                    smiles = None
                else:
                    smiles = _terminal_smiles(
                        terminal,
                        index,
                        count,
                        closure_count,
                        atom_vocabulary,
                    )
                exact_fields = {
                    "nodes": bool(
                        torch.equal(
                            terminal["nodes"][index, :count].cpu(),
                            torch.from_numpy(record.graph.node_states),
                        )
                    ),
                    "parents": bool(
                        torch.equal(
                            terminal["parents"][index, :count].cpu(),
                            torch.from_numpy(record.graph.parents),
                        )
                    ),
                    "parent_bonds": bool(
                        torch.equal(
                            terminal["parent_bonds"][index, :count].cpu(),
                            torch.from_numpy(record.graph.parent_bonds),
                        )
                    ),
                    "closure_left": bool(
                        torch.equal(
                            terminal["closure_left"][index, :closure_count].cpu(),
                            torch.from_numpy(record.graph.closure_left),
                        )
                    ),
                    "closure_right": bool(
                        torch.equal(
                            terminal["closure_right"][index, :closure_count].cpu(),
                            torch.from_numpy(record.graph.closure_right),
                        )
                    ),
                    "closure_bonds": bool(
                        torch.equal(
                            terminal["closure_bonds"][index, :closure_count].cpu(),
                            torch.from_numpy(record.graph.closure_bonds),
                        )
                    ),
                }
                target_values = {
                    "nodes": record.graph.node_states,
                    "parents": record.graph.parents,
                    "parent_bonds": record.graph.parent_bonds,
                    "closure_left": record.graph.closure_left,
                    "closure_right": record.graph.closure_right,
                    "closure_bonds": record.graph.closure_bonds,
                }
                mismatch_positions = {}
                for field, exact in exact_fields.items():
                    if exact:
                        continue
                    size = closure_count if field.startswith("closure") else count
                    observed = terminal[field][index, :size].cpu().numpy()
                    mismatch_positions[field] = np.flatnonzero(
                        observed != target_values[field]
                    ).tolist()
                outputs.append(
                    {
                        "sample_index": offset + index,
                        "program_id": record.program_id,
                        "layout_record_id": record.graph.structure_id,
                        "canonical_smiles": smiles,
                        "valid": smiles is not None,
                        "constraint_abstention_reason": abstention_reason,
                        "local_chemistry_policy_applied": local_chemistry_support is not None,
                        "exact_target_graph": smiles == record.graph.canonical_smiles,
                        "exact_tensor": all(exact_fields.values()),
                        "exact_fields": exact_fields,
                        "mismatch_positions": mismatch_positions,
                    }
                )
    by_program: dict[str, dict[str, int]] = {}
    for program_id in sorted({record.program_id for record in records}):
        program_rows = [row for row in outputs if row["program_id"] == program_id]
        by_program[program_id] = {
            "samples": len(program_rows),
            "valid": sum(bool(row["valid"]) for row in program_rows),
            "exact_target_graph": sum(bool(row["exact_target_graph"]) for row in program_rows),
            "exact_tensor": sum(bool(row["exact_tensor"]) for row in program_rows),
            "strict_constraint_abstentions": sum(
                row["constraint_abstention_reason"] is not None for row in program_rows
            ),
        }
    return outputs, {
        "samples": len(outputs),
        "valid": sum(bool(row["valid"]) for row in outputs),
        "exact_target_graph": sum(bool(row["exact_target_graph"]) for row in outputs),
        "exact_tensor": sum(bool(row["exact_tensor"]) for row in outputs),
        "fixed_state_failures": fixed_failures,
        "terminal_decode_policy": terminal_decode_policy,
        "strict_constraint_abstentions": sum(strict_abstentions.values()),
        "strict_constraint_abstention_reasons": dict(sorted(strict_abstentions.items())),
        "local_chemistry_policy_applied": local_chemistry_support is not None,
        "by_program": by_program,
        "repairs": dict(Counter()),
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY",
    "SUPPORTED_TERMINAL_DECODE_POLICIES",
    "TERMINAL_DECODE_POLICIES",
    "SynthesisProgramSamplingError",
    "load_synthesis_program_checkpoint",
    "decode_synthesis_program_strict_argmax",
    "sample_synthesis_program_products",
    "synthesis_program_source_marginals",
]
