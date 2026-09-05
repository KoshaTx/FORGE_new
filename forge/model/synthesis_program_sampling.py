"""Fixed-state-safe sampling for the shared Ugi/BL/LX whole-product flow."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from forge.core.io import read_json_object
from forge.flow import rstar_step
from forge.model.defog_feasibility import AtomState, _model_state_sha256, graph_to_molecule
from forge.model.local_chemistry_support import LocalChemistrySupport, tree_path_indices
from forge.model.potency_conditioning import PotencyCondition
from forge.model.reaction_core_saturation import (
    BoundReactionCoreSaturation,
    ReactionCoreSaturationPolicy,
)
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_flow import (
    collate_synthesis_program_layouts,
    decode_synthesis_program_argmax,
    resolve_synthesis_program_source_marginals,
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
from forge.model.ugi_transformer_topology import (
    UGI_PROGRAM_ID,
    UgiTransformerTopologyError,
    UgiTransformerTopologyPolicy,
    decode_ugi_exact_topology,
)
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]

CHECKPOINT_SCHEMA = "forge.synthesis_program_sparse_flow_checkpoint.v1"
JOINT_SAMPLING_FACTORIZATION = "joint"
LEARNED_TOPOLOGY_THEN_CHEMISTRY_FACTORIZATION = "learned_topology_then_chemistry"
SUPPORTED_SAMPLING_FACTORIZATIONS = frozenset(
    {
        JOINT_SAMPLING_FACTORIZATION,
        LEARNED_TOPOLOGY_THEN_CHEMISTRY_FACTORIZATION,
    }
)
TERMINAL_DECODE_POLICIES = (
    "unconstrained_argmax",
    "strict_valence_topology_argmax",
)
LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY = "strict_local_chemistry_argmax"
PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY = "strict_program_topology_argmax"
COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY = "strict_ugi_program_coupled_conditional"
CORE_SATURATION_TERMINAL_DECODE_POLICY = "strict_reaction_core_saturation_argmax"
ROLE_LOCAL_SUPPORTED_TERMINAL_DECODE_POLICY = (
    "strict_role_local_closure_chemistry_core_saturation_argmax"
)
UGI_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY = (
    "strict_ugi_topology_role_local_chemistry_core_saturation"
)
UGI_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY = (
    "strict_ugi_topology_role_local_chemistry_core_saturation_stochastic"
)
UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY = (
    "strict_ugi_ester_topology_role_local_chemistry_core_saturation"
)
UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY = (
    "strict_ugi_ester_topology_role_local_chemistry_core_saturation_stochastic"
)
UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY = (
    "strict_ugi_ester_topology_role_local_chemistry_core_saturation_mog_guided"
)
UGI_TOPOLOGY_COUPLED_TERMINAL_DECODE_POLICIES = frozenset(
    {
        COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
        UGI_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
        UGI_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY,
    }
)
LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICIES = frozenset(
    {
        LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY,
        ROLE_LOCAL_SUPPORTED_TERMINAL_DECODE_POLICY,
        UGI_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
        UGI_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY,
    }
)
PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICIES = frozenset(
    {
        PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY,
        COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
        UGI_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
        UGI_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY,
    }
)
PROGRAM_CYCLE_TERMINAL_DECODE_POLICIES = frozenset(
    {
        *PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICIES,
        ROLE_LOCAL_SUPPORTED_TERMINAL_DECODE_POLICY,
    }
)
COMPONENT_CONFINED_TERMINAL_DECODE_POLICIES = frozenset(
    {
        *PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICIES,
    }
)
CORE_SATURATION_TERMINAL_DECODE_POLICIES = frozenset(
    {
        CORE_SATURATION_TERMINAL_DECODE_POLICY,
        ROLE_LOCAL_SUPPORTED_TERMINAL_DECODE_POLICY,
        UGI_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
        UGI_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY,
    }
)
SUPPORTED_TERMINAL_DECODE_POLICIES = (
    *TERMINAL_DECODE_POLICIES,
    LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY,
    PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY,
    COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
    CORE_SATURATION_TERMINAL_DECODE_POLICY,
    ROLE_LOCAL_SUPPORTED_TERMINAL_DECODE_POLICY,
    UGI_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
    UGI_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
    UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
    UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
    UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY,
)
STOCHASTIC_TERMINAL_DECODE_POLICIES = frozenset(
    {
        UGI_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY,
    }
)
UGI_ESTER_TERMINAL_DECODE_POLICIES = frozenset(
    {
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY,
    }
)
STRICT_TERMINAL_DECODE_POLICIES = frozenset(
    {
        "strict_valence_topology_argmax",
        LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY,
        PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY,
        COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
        CORE_SATURATION_TERMINAL_DECODE_POLICY,
        ROLE_LOCAL_SUPPORTED_TERMINAL_DECODE_POLICY,
        UGI_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
        UGI_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
        UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY,
    }
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
    expected_prefix = (len(vocabulary.program_states), len(vocabulary.role_states))
    global_sources = node_marginal.shape == (len(atom_vocabulary),) and bond_marginal.shape == (
        int(model_config["bond_classes"]),
    )
    program_role_sources = node_marginal.shape == (
        *expected_prefix,
        len(atom_vocabulary),
    ) and bond_marginal.shape == (*expected_prefix, int(model_config["bond_classes"]))
    if (
        not (global_sources or program_role_sources)
        or np.any(node_marginal <= 0)
        or np.any(bond_marginal <= 0)
        or not np.allclose(node_marginal.sum(axis=-1), 1.0)
        or not np.allclose(bond_marginal.sum(axis=-1), 1.0)
    ):
        raise SynthesisProgramSamplingError("checkpoint source marginals are invalid")
    model.eval()
    return model, vocabulary, atom_vocabulary, node_marginal, bond_marginal, package


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


def _available_valence_units(state: AtomState) -> int:
    return _maximum_valence_units(state) - 2 * int(state.explicit_hydrogens)


_ATOM_CAPACITY_CACHE: dict[int, tuple[Sequence[AtomState], np.ndarray]] = {}
_BOND_UNIT_CACHE: dict[int, np.ndarray] = {}


def _atom_capacity_table(atom_vocabulary: Sequence[AtomState]) -> np.ndarray:
    """Return the frozen per-state valence capacities for one atom vocabulary.

    The table is a pure function of the vocabulary, but the strict decoder rebuilt it once per
    decoded record.  The vocabulary is held by reference in the cache value so its ``id`` cannot be
    recycled onto a different vocabulary while the entry is live.
    """

    entry = _ATOM_CAPACITY_CACHE.get(id(atom_vocabulary))
    if entry is None:
        table = np.asarray(
            [_available_valence_units(state) for state in atom_vocabulary], dtype=np.int64
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
                        minimum_tier = semantic_guidance_policy.local_chemistry_unsaturation_minimum_support_tier
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

            measured_terminal_patterns: dict[
                tuple[tuple[int, ...], tuple[int, ...]], int
            ] | None = None
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
                        5.0
                        if candidate_terminal_pattern(candidate_index)
                        in measured_terminal_patterns
                        else score
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
                indices_by_pattern: dict[
                    tuple[tuple[int, ...], tuple[int, ...]], list[int]
                ] = {}
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
                    int(
                        terminal_generator.choice(
                            len(pattern_keys), p=pattern_probabilities
                        )
                    )
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
                if (
                    semantic_guidance_policy.tail_unsaturation_count_strategy
                    in {
                        "frequency_downward",
                        "frequency_resampled",
                        "smoothed_frequency_resampled",
                    }
                ):
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
) -> tuple[dict[str, np.ndarray] | None, str | None]:
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
        if enforce_program_topology or enforce_program_cycles
        else None
    )

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
            if enforce_program_topology or component_confined:
                valid[:child] &= component_instances[:child] == component_instances[child]
        if enforce_program_topology:
            for parent in np.flatnonzero(valid).tolist():
                trial_parents = parents.copy()
                trial_parents[child] = parent
                observed = _terminal_role_morphology(
                    record,
                    parents=trial_parents,
                    closure_left=np.empty(0, dtype=np.int64),
                    closure_right=np.empty(0, dtype=np.int64),
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
                    or current[3] + remaining_variable_children < target[3]
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

    if enforce_program_topology:
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
) -> tuple[dict[str, Any], tuple[str | None, ...]]:
    """Decode once under strict support; infeasible attempts abstain and are never repaired."""

    if len(records) != int(layout["node_mask"].shape[0]):
        raise SynthesisProgramSamplingError("strict decoder batch and record counts disagree")
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
        )
        reasons.append(reason)
        if decoded is None:
            continue
        for field, values in decoded.items():
            terminal[field][index, : len(values)] = torch.as_tensor(
                values, dtype=terminal[field].dtype
            )
    return restore_synthesis_program_fixed_states(terminal, cpu_layout), tuple(reasons)


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
            values = decoded[field]
            topology[field][index, : len(values)] = torch.as_tensor(values, dtype=torch.long)
    return topology, tuple(reasons)


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
    sampling_factorization: str = JOINT_SAMPLING_FACTORIZATION,
    terminal_decode_policy: str = "unconstrained_argmax",
    terminal_seed: int | None = None,
    terminal_temperature: float = 1.0,
    topology_conditioned_chemistry_steps: int = 0,
    topology_conditioned_chemistry_seed: int | None = None,
    local_chemistry_support: LocalChemistrySupport | None = None,
    ugi_topology_policy: UgiTransformerTopologyPolicy | None = None,
    reaction_core_saturation_policy: ReactionCoreSaturationPolicy | None = None,
    ugi_ester_chemotype_policy: UgiEsterChemotypePolicy | None = None,
    ugi_role_chemistry_prior: UgiRoleChemistryPrior | None = None,
    ugi_role_chemistry_prior_strength: float = 0.0,
    ugi_amine_semantic_targets: Sequence[UgiAmineSemanticTarget] | None = None,
    ugi_all_role_semantic_targets: Sequence[UgiAllRoleSemanticTarget] | None = None,
    ugi_mog_semantic_guidance_policy: UgiMogSemanticGuidancePolicy | None = None,
    potency_condition: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate from semantic layouts while exposing only Ugi adapter-fixed graph states."""

    if (
        torch is None
        or not records
        or samples_per_program < 1
        or sample_steps < 2
        or batch_size < 1
        or sampling_factorization not in SUPPORTED_SAMPLING_FACTORIZATIONS
        or terminal_decode_policy not in SUPPORTED_TERMINAL_DECODE_POLICIES
    ):
        raise SynthesisProgramSamplingError("invalid shared synthesis-program sampling request")
    if potency_condition is not None and not isinstance(potency_condition, PotencyCondition):
        raise SynthesisProgramSamplingError(
            "sampling accepts only one scalar potency condition for the complete batch"
        )
    stochastic_terminal = terminal_decode_policy in STOCHASTIC_TERMINAL_DECODE_POLICIES
    if stochastic_terminal != (terminal_seed is not None):
        raise SynthesisProgramSamplingError(
            "stochastic terminal decoding requires one explicit terminal seed, and argmax "
            "decoding must not receive one"
        )
    if not np.isfinite(terminal_temperature) or terminal_temperature <= 0:
        raise SynthesisProgramSamplingError("terminal temperature must be finite and positive")
    topology_conditioned_chemistry = topology_conditioned_chemistry_steps > 0
    if topology_conditioned_chemistry_steps == 1 or topology_conditioned_chemistry_steps < 0:
        raise SynthesisProgramSamplingError(
            "topology-conditioned chemistry flow must be disabled or use at least two steps"
        )
    if topology_conditioned_chemistry != (topology_conditioned_chemistry_seed is not None):
        raise SynthesisProgramSamplingError(
            "topology-conditioned chemistry flow requires one explicit seed, and the disabled "
            "flow must not receive one"
        )
    if (
        not np.isfinite(ugi_role_chemistry_prior_strength)
        or ugi_role_chemistry_prior_strength < 0
        or (ugi_role_chemistry_prior_strength > 0 and ugi_role_chemistry_prior is None)
    ):
        raise SynthesisProgramSamplingError(
            "a nonzero Ugi chemistry-prior strength and one explicit prior must be supplied together"
        )
    if ugi_role_chemistry_prior is not None and (
        ugi_ester_chemotype_policy is None
        or ugi_role_chemistry_prior.reaction_id != ugi_ester_chemotype_policy.reaction_id
    ):
        raise SynthesisProgramSamplingError(
            "the soft Ugi chemistry prior must bind the active ester-chemotype reaction"
        )
    if ugi_amine_semantic_targets is not None and (
        len(ugi_amine_semantic_targets) != len(records)
        or any(record.program_id != UGI_PROGRAM_ID for record in records)
        or ugi_ester_chemotype_policy is None
        or ugi_topology_policy is None
    ):
        raise SynthesisProgramSamplingError(
            "Ugi amine semantic targets require one target per Ugi record and the exact Ugi "
            "topology/chemotype policies"
        )
    if ugi_all_role_semantic_targets is not None and (
        len(ugi_all_role_semantic_targets) != len(records)
        or any(record.program_id != UGI_PROGRAM_ID for record in records)
        or ugi_ester_chemotype_policy is None
        or ugi_topology_policy is None
        or local_chemistry_support is None
    ):
        raise SynthesisProgramSamplingError(
            "Ugi all-role semantic targets require one target per Ugi record and the exact Ugi "
            "topology, chemotype, and local-chemistry policies"
        )
    if ugi_amine_semantic_targets is not None and ugi_all_role_semantic_targets is not None:
        raise SynthesisProgramSamplingError(
            "amine-only and all-role semantic targets are mutually exclusive"
        )
    mog_guidance = (
        terminal_decode_policy == UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY
    )
    if mog_guidance != (ugi_mog_semantic_guidance_policy is not None):
        raise SynthesisProgramSamplingError(
            "the MOG-style terminal decoder and one explicit semantic-guidance policy must be "
            "supplied together"
        )
    if mog_guidance and ugi_all_role_semantic_targets is None:
        raise SynthesisProgramSamplingError(
            "MOG-style semantic guidance requires one all-role target per Ugi attempt"
        )
    if (
        mog_guidance
        and ugi_mog_semantic_guidance_policy.uses_joint_realism
        and ugi_mog_semantic_guidance_policy.joint_realism_scorer is None
    ):
        raise SynthesisProgramSamplingError(
            "joint-realism MOG guidance requires one bound measured-train reference"
        )
    if (
        mog_guidance
        and ugi_mog_semantic_guidance_policy.uses_local_reference
        and ugi_mog_semantic_guidance_policy.local_chemistry_prior is None
    ):
        raise SynthesisProgramSamplingError(
            "local-chemistry MOG guidance requires one bound measured-train reference"
        )
    if (
        mog_guidance
        and ugi_mog_semantic_guidance_policy.uses_local_reference
        and (
            ugi_ester_chemotype_policy is None
            or ugi_mog_semantic_guidance_policy.local_chemistry_prior.reaction_id
            != ugi_ester_chemotype_policy.reaction_id
        )
    ):
        raise SynthesisProgramSamplingError(
            "local-chemistry MOG guidance must bind the active Ugi reaction"
        )
    if (terminal_decode_policy in LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICIES) != (
        local_chemistry_support is not None
    ):
        raise SynthesisProgramSamplingError(
            "strict local-chemistry decoding and its support policy must be supplied together"
        )
    coupled_ugi_topology = terminal_decode_policy in UGI_TOPOLOGY_COUPLED_TERMINAL_DECODE_POLICIES
    if coupled_ugi_topology != (ugi_topology_policy is not None):
        raise SynthesisProgramSamplingError(
            "coupled Ugi topology decoding and its explicit support policy must be supplied together"
        )
    if topology_conditioned_chemistry and not coupled_ugi_topology:
        raise SynthesisProgramSamplingError(
            "topology-conditioned chemistry flow requires a Ugi topology-aware decoder"
        )
    learned_topology_then_chemistry = (
        sampling_factorization == LEARNED_TOPOLOGY_THEN_CHEMISTRY_FACTORIZATION
    )
    if learned_topology_then_chemistry and (
        not topology_conditioned_chemistry
        or terminal_decode_policy not in UGI_ESTER_TERMINAL_DECODE_POLICIES
        or any(record.program_id != UGI_PROGRAM_ID for record in records)
    ):
        raise SynthesisProgramSamplingError(
            "learned topology-then-chemistry sampling requires only Ugi records, the strict Ugi "
            "ester decoder, and at least two chemistry-flow steps"
        )
    if (terminal_decode_policy in CORE_SATURATION_TERMINAL_DECODE_POLICIES) != (
        reaction_core_saturation_policy is not None
    ):
        raise SynthesisProgramSamplingError(
            "strict reaction-core saturation decoding and its registry contract must be supplied "
            "together"
        )
    if (terminal_decode_policy in UGI_ESTER_TERMINAL_DECODE_POLICIES) != (
        ugi_ester_chemotype_policy is not None
    ):
        raise SynthesisProgramSamplingError(
            "the Ugi ester-stratum decoder and its registry-bound policy must be supplied together"
        )
    core_saturation: BoundReactionCoreSaturation | None = None
    if reaction_core_saturation_policy is not None:
        vocabulary = getattr(model, "vocabulary", None)
        core_position_states = getattr(vocabulary, "core_position_states", None)
        if core_position_states is None:
            raise SynthesisProgramSamplingError(
                "reaction-core saturation decoding requires a model that declares its "
                "core-position vocabulary"
            )
        core_saturation = reaction_core_saturation_policy.bind(core_position_states)
    resolved_device = torch.device(device)
    repeated = tuple(record for record in records for _ in range(samples_per_program))
    repeated_semantic_targets: tuple[UgiAmineSemanticTarget | None, ...] = (
        (None,) * len(repeated)
        if ugi_amine_semantic_targets is None
        else tuple(
            target for target in ugi_amine_semantic_targets for _ in range(samples_per_program)
        )
    )
    repeated_all_role_targets: tuple[UgiAllRoleSemanticTarget | None, ...] = (
        (None,) * len(repeated)
        if ugi_all_role_semantic_targets is None
        else tuple(
            target for target in ugi_all_role_semantic_targets for _ in range(samples_per_program)
        )
    )
    maximum_closures = int(model.maximum_closures)
    node_p0 = torch.as_tensor(node_marginal, dtype=torch.float32, device=resolved_device)
    bond_p0 = torch.as_tensor(bond_marginal, dtype=torch.float32, device=resolved_device)
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    topology_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    topology_conditioned_chemistry_generator = (
        None
        if topology_conditioned_chemistry_seed is None
        else torch.Generator(device=resolved_device).manual_seed(
            int(topology_conditioned_chemistry_seed)
        )
    )
    terminal_generator = (
        None if terminal_seed is None else np.random.default_rng(int(terminal_seed))
    )
    outputs: list[dict[str, Any]] = []
    fixed_failures = 0
    strict_abstentions: Counter[str] = Counter()
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(repeated), batch_size):
            local = repeated[offset : offset + batch_size]
            local_semantic_targets = repeated_semantic_targets[offset : offset + batch_size]
            local_all_role_targets = repeated_all_role_targets[offset : offset + batch_size]
            cpu_layout = collate_synthesis_program_layouts(
                local,
                maximum_closures=maximum_closures,
                conditioning_mode=conditioning_mode,
                program_state_mapping=program_state_mapping,
                role_state_mapping=role_state_mapping,
            )
            has_variable_parents = bool(cpu_layout["parent_variable_mask"].any())
            has_variable_closure_endpoints = bool(
                cpu_layout["closure_endpoint_variable_mask"].any()
            )
            layout = _move(cpu_layout, resolved_device)
            node_source, parent_bond_source, closure_bond_source = (
                resolve_synthesis_program_source_marginals(layout, node_p0, bond_p0)
            )
            state = _initial_state(layout, node_p0, bond_p0, generator)
            fixed_failure_checks = [~_fixed_state_exact_tensor(state, layout)]
            parent_candidates = _parent_candidate_mask(layout["node_mask"])
            endpoint_candidates = _endpoint_candidate_mask(layout["node_mask"], maximum_closures)
            time_grid = torch.arange(
                sample_steps, dtype=torch.float32, device=resolved_device
            ) / float(sample_steps)
            terminal_time = torch.ones((len(local),), device=resolved_device)
            conditioning = _program_conditioning(model, layout)
            for step in range(sample_steps):
                t_value = step / sample_steps
                t = time_grid[step].expand(len(local))
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
                state["nodes"] = rstar_step(
                    state["nodes"],
                    predictions["nodes"].softmax(dim=-1),
                    node_source,
                    t_value,
                    1.0 / sample_steps,
                    layout["atom_variable_mask"],
                    generator,
                )
                if has_variable_parents:
                    state["parents"] = pointer_rstar_step(
                        state["parents"],
                        predictions["parents"],
                        parent_candidates,
                        layout["parent_variable_mask"],
                        t_value,
                        1.0 / sample_steps,
                        generator,
                    )
                for field, mask_name, source in (
                    ("parent_bonds", "parent_bond_variable_mask", parent_bond_source),
                    ("closure_bonds", "closure_bond_variable_mask", closure_bond_source),
                ):
                    state[field] = rstar_step(
                        state[field],
                        predictions[field].softmax(dim=-1),
                        source,
                        t_value,
                        1.0 / sample_steps,
                        layout[mask_name],
                        generator,
                    )
                if has_variable_closure_endpoints:
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
                state = _restore_fixed_states_in_place(state, layout)
                fixed_failure_checks.append(~_fixed_state_exact_tensor(state, layout))
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
            topology_reasons: list[str | None] = [None] * len(local)
            topology_diagnostics: list[dict[str, Any] | None] = [None] * len(local)
            batch_has_coupled_ugi = coupled_ugi_topology and any(
                record.program_id == UGI_PROGRAM_ID for record in local
            )
            if batch_has_coupled_ugi:
                assert ugi_topology_policy is not None
                topology_state = {key: value.clone() for key, value in state.items()}
                if learned_topology_then_chemistry:
                    assert local_chemistry_support is not None
                    assert core_saturation is not None
                    assert ugi_ester_chemotype_policy is not None
                    learned_topology, learned_reasons = decode_synthesis_program_strict_topology(
                        terminal_predictions,
                        local,
                        atom_vocabulary,
                        local_chemistry_support,
                        core_saturation=core_saturation,
                        ugi_topology_policy=ugi_topology_policy,
                        ugi_ester_chemotype_policy=ugi_ester_chemotype_policy,
                    )
                    topology_reasons = list(learned_reasons)
                    for index, record in enumerate(local):
                        if topology_reasons[index] is not None:
                            continue
                        count = record.node_count
                        closure_count = record.graph.closure_count
                        for field in ("parents", "closure_left", "closure_right"):
                            size = count if field == "parents" else closure_count
                            topology_state[field][index, :size] = learned_topology[field][
                                index, :size
                            ].to(resolved_device)
                        topology_diagnostics[index] = {
                            "source": "transformer_parent_and_closure_heads",
                            "parents": learned_topology["parents"][index, :count].tolist(),
                            "closure_left": learned_topology["closure_left"][
                                index, :closure_count
                            ].tolist(),
                            "closure_right": learned_topology["closure_right"][
                                index, :closure_count
                            ].tolist(),
                        }
                else:
                    for index, record in enumerate(local):
                        if record.program_id != UGI_PROGRAM_ID:
                            continue
                        try:
                            decoded_topology = decode_ugi_exact_topology(
                                terminal_predictions,
                                index=index,
                                record=record,
                                policy=ugi_topology_policy,
                                generator=topology_generator,
                                ester_chemotype_policy=ugi_ester_chemotype_policy,
                                amine_semantic_target=local_semantic_targets[index],
                                all_role_semantic_target=local_all_role_targets[index],
                                local_chemistry_support=local_chemistry_support,
                                semantic_guidance_policy=ugi_mog_semantic_guidance_policy,
                            )
                        except UgiTransformerTopologyError as error:
                            topology_reasons[index] = f"ugi_topology_coupling_failure:{error}"
                            continue
                        count = record.node_count
                        closure_count = record.graph.closure_count
                        topology_state["parents"][index, :count] = torch.as_tensor(
                            decoded_topology.parents,
                            dtype=topology_state["parents"].dtype,
                            device=resolved_device,
                        )
                        if closure_count:
                            topology_state["closure_left"][index, :closure_count] = torch.as_tensor(
                                decoded_topology.closure_left,
                                dtype=topology_state["closure_left"].dtype,
                                device=resolved_device,
                            )
                            topology_state["closure_right"][index, :closure_count] = (
                                torch.as_tensor(
                                    decoded_topology.closure_right,
                                    dtype=topology_state["closure_right"].dtype,
                                    device=resolved_device,
                                )
                            )
                        topology_diagnostics[index] = {
                            "source": "constructive_offspring_head",
                            "parents": decoded_topology.parents.tolist(),
                            "closure_left": decoded_topology.closure_left.tolist(),
                            "closure_right": decoded_topology.closure_right.tolist(),
                            "offspring_by_role": [
                                values.tolist() for values in decoded_topology.offspring_by_role
                            ],
                        }
                topology_state = _restore_fixed_states_in_place(topology_state, layout)
                if topology_conditioned_chemistry:
                    assert topology_conditioned_chemistry_generator is not None
                    active_examples = torch.as_tensor(
                        [
                            record.program_id == UGI_PROGRAM_ID
                            and (
                                not learned_topology_then_chemistry
                                or topology_reasons[index] is None
                            )
                            for index, record in enumerate(local)
                        ],
                        dtype=torch.bool,
                        device=resolved_device,
                    )
                    if bool(active_examples.any()):
                        terminal_predictions, chemistry_fixed_checks = (
                            _topology_conditioned_chemistry_flow(
                                model,
                                topology_state,
                                layout,
                                conditioning,
                                node_source,
                                parent_bond_source,
                                closure_bond_source,
                                active_examples,
                                steps=topology_conditioned_chemistry_steps,
                                generator=topology_conditioned_chemistry_generator,
                                potency_condition=potency_condition,
                            )
                        )
                        fixed_failure_checks.extend(chemistry_fixed_checks)
                else:
                    # Chemistry is predicted after, and therefore conditional on, the exact tree
                    # and feasible closure endpoints.  The optional flow above strengthens this
                    # one-pass factorization by repeatedly refreshing the denoiser on that tree.
                    terminal_predictions = model(
                        nodes=topology_state["nodes"],
                        parents=topology_state["parents"],
                        parent_bonds=topology_state["parent_bonds"],
                        closure_left=topology_state["closure_left"],
                        closure_right=topology_state["closure_right"],
                        closure_bonds=topology_state["closure_bonds"],
                        t=terminal_time,
                        potency_condition=potency_condition,
                        **conditioning,
                    )
                # The exact topology is already decoded.  One-hot pointer scores let the common
                # strict chemistry decoder preserve it while retaining all valence checks.
                pointer_predictions = dict(terminal_predictions)
                parent_logits = torch.full_like(pointer_predictions["parents"], -1e9)
                parent_logits.scatter_(-1, topology_state["parents"].unsqueeze(-1), 1e9)
                pointer_predictions["parents"] = parent_logits
                for field in ("closure_left", "closure_right"):
                    endpoint_logits = torch.full_like(pointer_predictions[field], -1e9)
                    endpoint_logits.scatter_(-1, topology_state[field].unsqueeze(-1), 1e9)
                    pointer_predictions[field] = endpoint_logits
                terminal_predictions = pointer_predictions
            if terminal_decode_policy in STRICT_TERMINAL_DECODE_POLICIES:
                terminal, abstention_reasons = decode_synthesis_program_strict_argmax(
                    terminal_predictions,
                    layout,
                    local,
                    atom_vocabulary,
                    local_chemistry_support,
                    core_saturation=core_saturation,
                    terminal_generator=terminal_generator,
                    terminal_temperature=terminal_temperature,
                    ugi_ester_chemotype_policy=ugi_ester_chemotype_policy,
                    ugi_role_chemistry_prior=ugi_role_chemistry_prior,
                    ugi_role_chemistry_prior_strength=ugi_role_chemistry_prior_strength,
                    ugi_topology_policy=(
                        ugi_topology_policy if learned_topology_then_chemistry else None
                    ),
                    ugi_amine_semantic_targets=local_semantic_targets,
                    ugi_all_role_semantic_targets=local_all_role_targets,
                    ugi_mog_semantic_guidance_policy=ugi_mog_semantic_guidance_policy,
                    enforce_program_topology=(
                        terminal_decode_policy in PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICIES
                        and (not coupled_ugi_topology or batch_has_coupled_ugi)
                    ),
                    enforce_program_cycles=(
                        terminal_decode_policy in PROGRAM_CYCLE_TERMINAL_DECODE_POLICIES
                    ),
                    confine_generated_edges=(
                        terminal_decode_policy in COMPONENT_CONFINED_TERMINAL_DECODE_POLICIES
                        and (not coupled_ugi_topology or batch_has_coupled_ugi)
                    ),
                )
                if coupled_ugi_topology:
                    abstention_reasons = tuple(
                        topology_reason if topology_reason is not None else chemistry_reason
                        for topology_reason, chemistry_reason in zip(
                            topology_reasons, abstention_reasons, strict=True
                        )
                    )
            else:
                terminal = decode_synthesis_program_argmax(terminal_predictions, layout)
                abstention_reasons = (None,) * len(local)
            if terminal_decode_policy in STRICT_TERMINAL_DECODE_POLICIES:
                fixed_failures += int(not _fixed_state_exact_records(terminal, local))
            else:
                fixed_failure_checks.append(~_fixed_state_exact_tensor(terminal, layout))
            fixed_failures += int(torch.stack(fixed_failure_checks).sum().item())
            terminal_cpu = {field: values.detach().to("cpu") for field, values in terminal.items()}
            for index, record in enumerate(local):
                count = record.node_count
                closure_count = record.graph.closure_count
                abstention_reason = abstention_reasons[index]
                if abstention_reason is not None:
                    strict_abstentions[abstention_reason] += 1
                    smiles = None
                else:
                    smiles = _terminal_smiles(
                        terminal_cpu,
                        index,
                        count,
                        closure_count,
                        atom_vocabulary,
                    )
                exact_fields = {
                    "nodes": np.array_equal(
                        terminal_cpu["nodes"][index, :count].numpy(),
                        record.graph.node_states,
                    ),
                    "parents": np.array_equal(
                        terminal_cpu["parents"][index, :count].numpy(), record.graph.parents
                    ),
                    "parent_bonds": np.array_equal(
                        terminal_cpu["parent_bonds"][index, :count].numpy(),
                        record.graph.parent_bonds,
                    ),
                    "closure_left": np.array_equal(
                        terminal_cpu["closure_left"][index, :closure_count].numpy(),
                        record.graph.closure_left,
                    ),
                    "closure_right": np.array_equal(
                        terminal_cpu["closure_right"][index, :closure_count].numpy(),
                        record.graph.closure_right,
                    ),
                    "closure_bonds": np.array_equal(
                        terminal_cpu["closure_bonds"][index, :closure_count].numpy(),
                        record.graph.closure_bonds,
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
                    observed = terminal_cpu[field][index, :size].numpy()
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
                        "program_topology_policy_applied": (
                            terminal_decode_policy in PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICIES
                            and (
                                terminal_decode_policy
                                not in UGI_TOPOLOGY_COUPLED_TERMINAL_DECODE_POLICIES
                                or record.program_id == UGI_PROGRAM_ID
                            )
                        ),
                        "program_role_cycle_policy_applied": (
                            terminal_decode_policy in PROGRAM_CYCLE_TERMINAL_DECODE_POLICIES
                        ),
                        "component_confined_policy_applied": (
                            terminal_decode_policy in COMPONENT_CONFINED_TERMINAL_DECODE_POLICIES
                            and (
                                terminal_decode_policy
                                not in UGI_TOPOLOGY_COUPLED_TERMINAL_DECODE_POLICIES
                                or record.program_id == UGI_PROGRAM_ID
                            )
                        ),
                        "topology_coupling_second_pass_applied": (
                            coupled_ugi_topology and record.program_id == UGI_PROGRAM_ID
                        ),
                        "topology_conditioned_chemistry_flow_applied": (
                            topology_conditioned_chemistry
                            and record.program_id == UGI_PROGRAM_ID
                            and (
                                not learned_topology_then_chemistry
                                or topology_reasons[index] is None
                            )
                        ),
                        "sampling_factorization": sampling_factorization,
                        "learned_topology_flow_applied": (
                            learned_topology_then_chemistry
                            and record.program_id == UGI_PROGRAM_ID
                            and topology_reasons[index] is None
                        ),
                        "reaction_core_saturation_policy_applied": (
                            core_saturation is not None and core_saturation.applies_to(record)
                        ),
                        "ugi_ester_chemotype_policy_applied": (
                            ugi_ester_chemotype_policy is not None
                            and record.program_id == ugi_ester_chemotype_policy.reaction_id
                        ),
                        "ugi_role_chemistry_prior_applied": (
                            ugi_role_chemistry_prior is not None
                            and ugi_role_chemistry_prior_strength > 0
                            and record.program_id == ugi_role_chemistry_prior.reaction_id
                        ),
                        "ugi_amine_semantic_target_applied": (
                            local_semantic_targets[index] is not None
                        ),
                        "requested_ugi_amine_semantic_target": (
                            None
                            if local_semantic_targets[index] is None
                            else local_semantic_targets[index].to_mapping()
                        ),
                        "ugi_all_role_semantic_target_applied": (
                            local_all_role_targets[index] is not None
                        ),
                        "ugi_mog_semantic_guidance_applied": (
                            ugi_mog_semantic_guidance_policy is not None
                        ),
                        "ugi_joint_semantic_realism_guidance_applied": (
                            ugi_mog_semantic_guidance_policy is not None
                            and ugi_mog_semantic_guidance_policy.uses_joint_realism
                        ),
                        "ugi_local_chemistry_mog_guidance_applied": (
                            ugi_mog_semantic_guidance_policy is not None
                            and ugi_mog_semantic_guidance_policy.uses_local_chemistry
                        ),
                        "ugi_local_atom_chemistry_mog_guidance_applied": (
                            ugi_mog_semantic_guidance_policy is not None
                            and ugi_mog_semantic_guidance_policy.uses_local_chemistry_atoms
                        ),
                        "ugi_local_bond_chemistry_mog_guidance_applied": (
                            ugi_mog_semantic_guidance_policy is not None
                            and ugi_mog_semantic_guidance_policy.uses_local_chemistry_bonds
                        ),
                        "ugi_whole_head_topology_support_applied": (
                            ugi_mog_semantic_guidance_policy is not None
                            and ugi_mog_semantic_guidance_policy.whole_head_topology_support
                        ),
                        "ugi_local_atom_chemistry_total_variation_radius": (
                            None
                            if ugi_mog_semantic_guidance_policy is None
                            else ugi_mog_semantic_guidance_policy.local_chemistry_atom_total_variation_radius
                        ),
                        "requested_ugi_all_role_semantic_target": (
                            None
                            if local_all_role_targets[index] is None
                            else local_all_role_targets[index].to_mapping()
                        ),
                        "requested_role_morphology": (
                            {
                                next(
                                    block.role
                                    for block in record.component_blocks
                                    if block.role_state == state
                                ): list(values)
                                for state, values in _exact_role_morphology_targets(record).items()
                            }
                            if record.role_morphology_states is not None
                            else None
                        ),
                        "sampled_topology": topology_diagnostics[index],
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
        "sampling_factorization": sampling_factorization,
        "learned_topology_flow_applied": any(
            bool(row["learned_topology_flow_applied"]) for row in outputs
        ),
        "topology_proposal_context": (
            "joint_flow_with_provisional_chemistry_then_discard_chemistry"
            if learned_topology_then_chemistry
            else None
        ),
        "terminal_decode_policy": terminal_decode_policy,
        "terminal_seed": terminal_seed,
        "terminal_temperature": terminal_temperature,
        "topology_conditioned_chemistry_steps": topology_conditioned_chemistry_steps,
        "topology_conditioned_chemistry_seed": topology_conditioned_chemistry_seed,
        "topology_conditioned_chemistry_flow_applied": any(
            bool(row["topology_conditioned_chemistry_flow_applied"]) for row in outputs
        ),
        "topology_conditioned_chemistry_neural_evaluations_per_batch": (
            topology_conditioned_chemistry_steps + 1 if topology_conditioned_chemistry else 0
        ),
        "topology_conditioned_chemistry_source_reset": topology_conditioned_chemistry,
        "terminal_chemistry_readout": (
            (
                "support_constrained_topology_ranked_atom_and_bond_argmax"
                if not ugi_mog_semantic_guidance_policy.uses_ranked_terminal_chemistry
                else (
                    "support_constrained_mog_ranked_categorical_atom_draw_and_bond_argmax"
                    if not ugi_mog_semantic_guidance_policy.uses_ranked_terminal_bonds
                    else "support_constrained_mog_ranked_categorical_atom_and_bond_draw"
                )
            )
            if mog_guidance and ugi_mog_semantic_guidance_policy is not None
            else (
                "support_constrained_soft_prior_categorical_atom_and_bond_draw"
                if stochastic_terminal and ugi_role_chemistry_prior_strength > 0
                else (
                    "support_constrained_categorical_atom_and_bond_draw"
                    if stochastic_terminal
                    else (
                        "support_constrained_soft_prior_atom_and_bond_argmax"
                        if ugi_role_chemistry_prior_strength > 0
                        else "support_constrained_atom_and_bond_argmax"
                    )
                )
            )
        ),
        "potency_condition": (
            None
            if potency_condition is None
            else {
                "endpoint_id": potency_condition.endpoint_id,
                "target_quantile": potency_condition.target_quantile,
                "policy_id": potency_condition.policy_id,
            }
        ),
        "strict_constraint_abstentions": sum(strict_abstentions.values()),
        "strict_constraint_abstention_reasons": dict(sorted(strict_abstentions.items())),
        "local_chemistry_policy_applied": local_chemistry_support is not None,
        "program_topology_policy_applied": any(
            bool(row["program_topology_policy_applied"]) for row in outputs
        ),
        "program_role_cycle_policy_applied": (
            terminal_decode_policy in PROGRAM_CYCLE_TERMINAL_DECODE_POLICIES
        ),
        "component_confined_policy_applied": any(
            bool(row["component_confined_policy_applied"]) for row in outputs
        ),
        "topology_coupling_second_pass_applied": any(
            bool(row["topology_coupling_second_pass_applied"]) for row in outputs
        ),
        "topology_selection": (
            (
                "transformer_parent_and_closure_flow_then_support_constrained_topology_"
                f"then_{topology_conditioned_chemistry_steps}_step_conditioned_chemistry_flow_"
                "then_argmax_chemistry"
            )
            if learned_topology_then_chemistry
            else (
                (
                    "measured_joint_program_then_constructive_ester_and_role_local_cycle_topology_"
                    + (
                        f"then_{topology_conditioned_chemistry_steps}_step_conditioned_chemistry_flow_"
                        "then_sequential_masked_categorical_chemistry"
                        if topology_conditioned_chemistry and stochastic_terminal
                        else (
                            f"then_{topology_conditioned_chemistry_steps}_step_conditioned_"
                            "chemistry_flow_then_argmax_chemistry"
                            if topology_conditioned_chemistry
                            else (
                                "then_sequential_masked_categorical_chemistry"
                                if stochastic_terminal
                                else "then_argmax_chemistry"
                            )
                        )
                    )
                )
                if (
                    ugi_ester_chemotype_policy is not None
                    and any(bool(row["topology_coupling_second_pass_applied"]) for row in outputs)
                )
                else (
                    (
                        "exact_program_conditional_sample_"
                        f"then_{topology_conditioned_chemistry_steps}_step_conditioned_chemistry_flow_"
                        "then_sequential_masked_categorical_chemistry"
                        if topology_conditioned_chemistry and stochastic_terminal
                        else (
                            "exact_program_conditional_sample_"
                            f"then_{topology_conditioned_chemistry_steps}_step_conditioned_"
                            "chemistry_flow_then_argmax_chemistry"
                            if topology_conditioned_chemistry
                            else (
                                "exact_program_conditional_sample_then_sequential_masked_"
                                "categorical_chemistry"
                                if stochastic_terminal
                                else "exact_program_conditional_sample_then_argmax_chemistry"
                            )
                        )
                    )
                    if any(bool(row["topology_coupling_second_pass_applied"]) for row in outputs)
                    else None
                )
            )
        ),
        "terminal_neural_refresh_after_each_choice": False,
        "topology_seed": (
            seed
            if learned_topology_then_chemistry
            else (
                seed + 1
                if any(bool(row["topology_coupling_second_pass_applied"]) for row in outputs)
                else None
            )
        ),
        "ugi_topology_policy": (
            None if ugi_topology_policy is None else ugi_topology_policy.to_mapping()
        ),
        "reaction_core_saturation_policy_applied": reaction_core_saturation_policy is not None,
        "reaction_core_saturation_policy": (
            None
            if reaction_core_saturation_policy is None
            else reaction_core_saturation_policy.to_mapping()
        ),
        "ugi_ester_chemotype_policy_applied": ugi_ester_chemotype_policy is not None,
        "ugi_ester_chemotype_policy": (
            None if ugi_ester_chemotype_policy is None else ugi_ester_chemotype_policy.to_mapping()
        ),
        "ugi_role_chemistry_prior_applied": (
            ugi_role_chemistry_prior is not None and ugi_role_chemistry_prior_strength > 0
        ),
        "ugi_role_chemistry_prior_strength": ugi_role_chemistry_prior_strength,
        "ugi_mog_semantic_guidance_applied": mog_guidance,
        "ugi_mog_semantic_guidance_policy": (
            None
            if ugi_mog_semantic_guidance_policy is None
            else ugi_mog_semantic_guidance_policy.to_mapping()
        ),
        "ugi_joint_semantic_realism_guidance_applied": (
            ugi_mog_semantic_guidance_policy is not None
            and ugi_mog_semantic_guidance_policy.uses_joint_realism
        ),
        "ugi_joint_semantic_realism_reference": (
            None
            if ugi_mog_semantic_guidance_policy is None
            else ugi_mog_semantic_guidance_policy.joint_realism_audit()
        ),
        "ugi_local_chemistry_mog_guidance_applied": (
            ugi_mog_semantic_guidance_policy is not None
            and ugi_mog_semantic_guidance_policy.uses_local_chemistry
        ),
        "ugi_local_atom_chemistry_mog_guidance_applied": (
            ugi_mog_semantic_guidance_policy is not None
            and ugi_mog_semantic_guidance_policy.uses_local_chemistry_atoms
        ),
        "ugi_local_bond_chemistry_mog_guidance_applied": (
            ugi_mog_semantic_guidance_policy is not None
            and ugi_mog_semantic_guidance_policy.uses_local_chemistry_bonds
        ),
        "ugi_whole_head_topology_support_applied": (
            ugi_mog_semantic_guidance_policy is not None
            and ugi_mog_semantic_guidance_policy.whole_head_topology_support
        ),
        "ugi_local_atom_chemistry_total_variation_radius": (
            None
            if ugi_mog_semantic_guidance_policy is None
            else ugi_mog_semantic_guidance_policy.local_chemistry_atom_total_variation_radius
        ),
        "ugi_local_chemistry_mog_reference": (
            None
            if ugi_mog_semantic_guidance_policy is None
            else ugi_mog_semantic_guidance_policy.local_chemistry_audit()
        ),
        "ugi_role_chemistry_prior": (
            None if ugi_role_chemistry_prior is None else ugi_role_chemistry_prior.to_mapping()
        ),
        "ugi_amine_semantic_target_applied": ugi_amine_semantic_targets is not None,
        "ugi_amine_semantic_joint_support_applied": (
            ugi_amine_semantic_targets is not None and local_chemistry_support is not None
        ),
        "ugi_amine_semantic_target_policy": (
            None
            if ugi_amine_semantic_targets is None
            else {
                "source": "identity-free measured-Ugi train-fold conditional program",
                "fields": list(ugi_amine_semantic_targets[0].to_mapping()),
                "sampling": "one semantic draw and one exact conditional decode without retry",
                "topology_conditioning": "nonempty train-fold local-chemistry support",
            }
        ),
        "ugi_all_role_semantic_target_applied": ugi_all_role_semantic_targets is not None,
        "ugi_all_role_semantic_joint_support_applied": (
            ugi_all_role_semantic_targets is not None and local_chemistry_support is not None
        ),
        "ugi_all_role_semantic_target_policy": (
            None
            if ugi_all_role_semantic_targets is None
            else {
                "source": "identity-free measured-Ugi train-fold joint conditional program",
                "fields": list(ugi_all_role_semantic_targets[0].to_mapping()),
                "sampling": (
                    "one all-role semantic draw and one exact conditional decode without retry"
                ),
                "topology_conditioning": "nonempty train-fold local-chemistry support",
            }
        ),
        "by_program": by_program,
        "repairs": dict(Counter()),
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "COMPONENT_CONFINED_TERMINAL_DECODE_POLICIES",
    "CORE_SATURATION_TERMINAL_DECODE_POLICIES",
    "CORE_SATURATION_TERMINAL_DECODE_POLICY",
    "COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY",
    "LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICIES",
    "LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY",
    "PROGRAM_CYCLE_TERMINAL_DECODE_POLICIES",
    "PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICIES",
    "PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY",
    "ROLE_LOCAL_SUPPORTED_TERMINAL_DECODE_POLICY",
    "STRICT_TERMINAL_DECODE_POLICIES",
    "SUPPORTED_TERMINAL_DECODE_POLICIES",
    "TERMINAL_DECODE_POLICIES",
    "UGI_TOPOLOGY_COUPLED_TERMINAL_DECODE_POLICIES",
    "UGI_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY",
    "UGI_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY",
    "UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY",
    "UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY",
    "UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY",
    "UGI_ESTER_TERMINAL_DECODE_POLICIES",
    "STOCHASTIC_TERMINAL_DECODE_POLICIES",
    "SynthesisProgramSamplingError",
    "load_synthesis_program_checkpoint",
    "decode_synthesis_program_strict_argmax",
    "sample_synthesis_program_products",
    "synthesis_program_source_marginals",
]
