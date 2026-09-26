"""Sampling state initialization, fixed-state preservation, and graph conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from rdkit import Chem

from forge.flow import rstar_step
from forge.model._synthesis_sampling.contracts import SynthesisProgramSamplingError
from forge.model.defog_feasibility import AtomState, graph_to_molecule
from forge.model.reaction_program_flow import resolve_synthesis_program_source_marginals
from forge.model.sparse_topology_feasibility import (
    INDEX_TO_DENSE_BOND,
    _endpoint_candidate_mask,
    _parent_candidate_mask,
)
from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


def _move(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def _restore_fixed_states_in_place(
    state: dict[str, Any], layout: Mapping[str, Any]
) -> dict[str, Any]:
    """Restore immutable adapter states without cloning six complete state tensors."""

    fields = {
        "nodes": "fixed_atom_mask",
        "parents": "fixed_parent_mask",
        "parent_bonds": "fixed_parent_bond_mask",
        "closure_left": "fixed_closure_endpoint_mask",
        "closure_right": "fixed_closure_endpoint_mask",
        "closure_bonds": "fixed_closure_bond_mask",
    }
    for field, mask_name in fields.items():
        mask = layout[mask_name]
        state[field][mask] = layout[field][mask]
    return state


def _program_conditioning(model: Any, layout: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze one batch's clean program context, encoding it once where the model allows reuse.

    Every coordinate here is layout semantics: masks, program state, precursor roles, reaction-core
    positions, depth, repeat groups, component positions and role morphology.  None of them is
    denoised, so they are identical at all `sample_steps` calls and at the terminal call.  Models
    that expose ``prepare_program_memory`` additionally return their encoded program tokens, the
    per-block cross-attention projections of those tokens and the per-adapter routing weights, all
    of which are functions of these same fixed coordinates.  The model validates that the memory
    was built from these exact tensors and fails closed otherwise.
    """

    conditioning = {
        field: layout[field]
        for field in (
            "node_mask",
            "child_mask",
            "closure_mask",
            "program_states",
            "role_states",
            "core_position_states",
            "program_depths",
            "adapter_mask",
            "repeat_group_states",
            "component_position_states",
            "component_instance_states",
            "role_morphology_states",
        )
    }
    prepare = getattr(model, "prepare_program_memory", None)
    if prepare is not None:
        conditioning["program_memory"] = prepare(
            program_states=layout["program_states"],
            role_states=layout["role_states"],
            core_position_states=layout["core_position_states"],
            program_depths=layout["program_depths"],
            adapter_mask=layout["adapter_mask"],
            repeat_group_states=layout["repeat_group_states"],
            component_position_states=layout["component_position_states"],
            role_morphology_states=layout["role_morphology_states"],
        )
    return conditioning


def _initial_state(
    layout: Mapping[str, Any],
    node_marginal: Any,
    bond_marginal: Any,
    generator: Any,
) -> dict[str, Any]:
    batch, nodes = layout["node_mask"].shape
    closures = layout["closure_mask"].shape[1]
    node_source, parent_bond_source, closure_bond_source = (
        resolve_synthesis_program_source_marginals(layout, node_marginal, bond_marginal)
    )

    def draw(source: Any, shape: tuple[int, int]) -> Any:
        if source.ndim == 1:
            probabilities = source[None].expand(shape[0] * shape[1], -1)
        else:
            probabilities = source.reshape(shape[0] * shape[1], -1)
        return torch.multinomial(probabilities, 1, generator=generator).reshape(shape)

    state = {
        "nodes": draw(node_source, (batch, nodes)),
        "parent_bonds": draw(parent_bond_source, (batch, nodes)),
        "closure_bonds": draw(closure_bond_source, (batch, closures)),
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
    return _restore_fixed_states_in_place(state, layout)


def _draw_source_categorical(
    source: Any,
    shape: tuple[int, int],
    *,
    generator: Any,
) -> Any:
    """Draw one categorical source state per requested coordinate."""

    count = shape[0] * shape[1]
    if count == 0:
        return torch.zeros(shape, dtype=torch.long, device=source.device)
    if source.ndim == 1:
        probabilities = source[None].expand(count, -1)
    elif source.ndim == 3 and tuple(source.shape[:2]) == shape:
        probabilities = source.reshape(count, -1)
    else:
        raise SynthesisProgramSamplingError(
            "topology-conditioned chemistry source has incompatible support"
        )
    return torch.multinomial(probabilities, 1, generator=generator).reshape(shape)


def _initial_topology_conditioned_chemistry_state(
    topology_state: Mapping[str, Any],
    layout: Mapping[str, Any],
    node_source: Any,
    parent_bond_source: Any,
    closure_bond_source: Any,
    active_examples: Any,
    *,
    generator: Any,
) -> dict[str, Any]:
    """Reset only variable chemistry while preserving one already constructed topology."""

    state = {field: values.clone() for field, values in topology_state.items()}
    example_mask = active_examples[:, None]
    for field, source, mask_name in (
        ("nodes", node_source, "atom_variable_mask"),
        ("parent_bonds", parent_bond_source, "parent_bond_variable_mask"),
        ("closure_bonds", closure_bond_source, "closure_bond_variable_mask"),
    ):
        drawn = _draw_source_categorical(
            source,
            tuple(state[field].shape),
            generator=generator,
        )
        active = layout[mask_name] & example_mask
        state[field][active] = drawn[active]
    state["parent_bonds"][:, 0] = 0
    return _restore_fixed_states_in_place(state, layout)


def _topology_conditioned_chemistry_flow(
    model: Any,
    topology_state: Mapping[str, Any],
    layout: Mapping[str, Any],
    conditioning: Mapping[str, Any],
    node_source: Any,
    parent_bond_source: Any,
    closure_bond_source: Any,
    active_examples: Any,
    *,
    steps: int,
    generator: Any,
    potency_condition: Any | None,
) -> tuple[dict[str, Any], list[Any]]:
    """Denoise chemistry repeatedly while projecting onto one fixed sampled topology.

    This is the topology-first factorization used by the native Ugi model: parents and closure
    endpoints are constructed once, variable atom and bond states are redrawn from their declared
    source, and the neural denoiser is refreshed before every chemistry-only R-star transition.
    """

    if steps < 2 or not bool(active_examples.any()):
        raise SynthesisProgramSamplingError(
            "topology-conditioned chemistry flow requires at least two steps and one active row"
        )
    state = _initial_topology_conditioned_chemistry_state(
        topology_state,
        layout,
        node_source,
        parent_bond_source,
        closure_bond_source,
        active_examples,
        generator=generator,
    )
    topology_fields = ("parents", "closure_left", "closure_right")
    topology_snapshot = {field: state[field].clone() for field in topology_fields}
    active = {
        "nodes": layout["atom_variable_mask"] & active_examples[:, None],
        "parent_bonds": layout["parent_bond_variable_mask"] & active_examples[:, None],
        "closure_bonds": layout["closure_bond_variable_mask"] & active_examples[:, None],
    }
    sources = {
        "nodes": node_source,
        "parent_bonds": parent_bond_source,
        "closure_bonds": closure_bond_source,
    }
    fixed_failure_checks = [~_fixed_state_exact_tensor(state, layout)]
    topology_exact = torch.ones((), dtype=torch.bool, device=state["nodes"].device)
    for step in range(steps):
        t_value = step / steps
        t = torch.full(
            (state["nodes"].shape[0],),
            t_value,
            dtype=torch.float32,
            device=state["nodes"].device,
        )
        predictions = model(
            nodes=state["nodes"],
            parents=state["parents"],
            parent_bonds=state["parent_bonds"],
            closure_left=state["closure_left"],
            closure_right=state["closure_right"],
            closure_bonds=state["closure_bonds"],
            t=t,
            potency_condition=potency_condition,
            **conditioning,
        )
        for field in ("nodes", "parent_bonds", "closure_bonds"):
            state[field] = rstar_step(
                state[field],
                predictions[field].softmax(dim=-1),
                sources[field],
                t_value,
                1.0 / steps,
                active[field],
                generator,
            )
        state = _restore_fixed_states_in_place(state, layout)
        fixed_failure_checks.append(~_fixed_state_exact_tensor(state, layout))
        for field in topology_fields:
            topology_exact = topology_exact & torch.equal(state[field], topology_snapshot[field])
    if not bool(topology_exact.item()):
        raise SynthesisProgramSamplingError(
            "topology-conditioned chemistry flow changed a parent or closure endpoint"
        )
    terminal_time = torch.ones(
        (state["nodes"].shape[0],), dtype=torch.float32, device=state["nodes"].device
    )
    terminal_predictions = model(
        nodes=state["nodes"],
        parents=state["parents"],
        parent_bonds=state["parent_bonds"],
        closure_left=state["closure_left"],
        closure_right=state["closure_right"],
        closure_bonds=state["closure_bonds"],
        t=terminal_time,
        potency_condition=potency_condition,
        **conditioning,
    )
    return terminal_predictions, fixed_failure_checks


def _fixed_state_exact(state: Mapping[str, Any], layout: Mapping[str, Any]) -> bool:
    return bool(_fixed_state_exact_tensor(state, layout).item())


def _fixed_state_exact_tensor(state: Mapping[str, Any], layout: Mapping[str, Any]) -> Any:
    """Return the fixed-state audit as a device scalar without forcing synchronization."""

    fields = {
        "nodes": "fixed_atom_mask",
        "parents": "fixed_parent_mask",
        "parent_bonds": "fixed_parent_bond_mask",
        "closure_left": "fixed_closure_endpoint_mask",
        "closure_right": "fixed_closure_endpoint_mask",
        "closure_bonds": "fixed_closure_bond_mask",
    }
    exact = torch.ones((), dtype=torch.bool, device=state["nodes"].device)
    for field, mask in fields.items():
        exact = exact & torch.all(state[field][layout[mask]] == layout[field][layout[mask]])
    return exact


def _fixed_state_exact_records(
    state: Mapping[str, Any], records: Sequence[SynthesisProgramGraphRecord]
) -> bool:
    """Audit a CPU terminal batch directly against its record-level immutable states."""

    fields = {
        "nodes": ("fixed_atom_mask", "node_states"),
        "parents": ("fixed_parent_bond_mask", "parents"),
        "parent_bonds": ("fixed_parent_bond_mask", "parent_bonds"),
        "closure_left": ("fixed_closure_bond_mask", "closure_left"),
        "closure_right": ("fixed_closure_bond_mask", "closure_right"),
        "closure_bonds": ("fixed_closure_bond_mask", "closure_bonds"),
    }
    # One host view per field, then per-record slicing of that view.  The audit previously built a
    # fresh NumPy array for every field of every record, which is six conversions per sample.
    views = {field: state[field].numpy() for field in fields}
    for index, record in enumerate(records):
        for field, (mask_name, target_name) in fields.items():
            mask = getattr(record, mask_name)
            target = (
                record.graph.node_states
                if target_name == "node_states"
                else getattr(record.graph, target_name)
            )
            observed = views[field][index, : len(target)]
            if not np.array_equal(observed[mask], target[mask]):
                return False
    return True


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
