"""Public sampling API and fixed-state-safe whole-product flow loop."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from forge.flow import rstar_step
from forge.model._synthesis_sampling.checkpoint import (
    load_synthesis_program_checkpoint as load_synthesis_program_checkpoint,
)
from forge.model._synthesis_sampling.checkpoint import (
    synthesis_program_source_marginals as synthesis_program_source_marginals,
)
from forge.model._synthesis_sampling.constraints import _ATOM_CAPACITY_CACHE as _ATOM_CAPACITY_CACHE
from forge.model._synthesis_sampling.constraints import _BOND_UNIT_CACHE as _BOND_UNIT_CACHE
from forge.model._synthesis_sampling.constraints import _argmax_allowed as _argmax_allowed
from forge.model._synthesis_sampling.constraints import _atom_capacity_table as _atom_capacity_table
from forge.model._synthesis_sampling.constraints import (
    _available_valence_units as _available_valence_units,
)
from forge.model._synthesis_sampling.constraints import _bond_unit_table as _bond_unit_table
from forge.model._synthesis_sampling.constraints import (
    _carbon_skeleton_diameter_from_states as _carbon_skeleton_diameter_from_states,
)
from forge.model._synthesis_sampling.constraints import (
    _distances_to_reaction_core as _distances_to_reaction_core,
)
from forge.model._synthesis_sampling.constraints import (
    _exact_role_morphology_targets as _exact_role_morphology_targets,
)
from forge.model._synthesis_sampling.constraints import (
    _generated_ring_support as _generated_ring_support,
)
from forge.model._synthesis_sampling.constraints import (
    _induced_graph_diameter as _induced_graph_diameter,
)
from forge.model._synthesis_sampling.constraints import _sample_allowed as _sample_allowed
from forge.model._synthesis_sampling.constraints import (
    _sample_mog_chemistry_allowed as _sample_mog_chemistry_allowed,
)
from forge.model._synthesis_sampling.constraints import (
    _select_ugi_all_role_tail_bonds as _select_ugi_all_role_tail_bonds,
)
from forge.model._synthesis_sampling.constraints import (
    _select_ugi_amine_atom_states as _select_ugi_amine_atom_states,
)
from forge.model._synthesis_sampling.constraints import (
    _terminal_role_morphology as _terminal_role_morphology,
)
from forge.model._synthesis_sampling.constraints import (
    _ugi_ester_motif_constraints as _ugi_ester_motif_constraints,
)
from forge.model._synthesis_sampling.constraints import (
    _ugi_program_from_record as _ugi_program_from_record,
)
from forge.model._synthesis_sampling.constraints import (
    _ugi_topology_support_failure as _ugi_topology_support_failure,
)
from forge.model._synthesis_sampling.contracts import CHECKPOINT_SCHEMA as CHECKPOINT_SCHEMA
from forge.model._synthesis_sampling.contracts import (
    COMPONENT_CONFINED_TERMINAL_DECODE_POLICIES as COMPONENT_CONFINED_TERMINAL_DECODE_POLICIES,
)
from forge.model._synthesis_sampling.contracts import (
    CORE_SATURATION_TERMINAL_DECODE_POLICIES as CORE_SATURATION_TERMINAL_DECODE_POLICIES,
)
from forge.model._synthesis_sampling.contracts import (
    CORE_SATURATION_TERMINAL_DECODE_POLICY as CORE_SATURATION_TERMINAL_DECODE_POLICY,
)
from forge.model._synthesis_sampling.contracts import (
    COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY as COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
)
from forge.model._synthesis_sampling.contracts import (
    JOINT_SAMPLING_FACTORIZATION as JOINT_SAMPLING_FACTORIZATION,
)
from forge.model._synthesis_sampling.contracts import (
    LEARNED_TOPOLOGY_THEN_CHEMISTRY_FACTORIZATION as LEARNED_TOPOLOGY_THEN_CHEMISTRY_FACTORIZATION,
)
from forge.model._synthesis_sampling.contracts import (
    LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICIES as LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICIES,
)
from forge.model._synthesis_sampling.contracts import (
    LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY as LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY,
)
from forge.model._synthesis_sampling.contracts import (
    PROGRAM_CYCLE_TERMINAL_DECODE_POLICIES as PROGRAM_CYCLE_TERMINAL_DECODE_POLICIES,
)
from forge.model._synthesis_sampling.contracts import (
    PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICIES as PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICIES,
)
from forge.model._synthesis_sampling.contracts import (
    PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY as PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY,
)
from forge.model._synthesis_sampling.contracts import (
    ROLE_LOCAL_SUPPORTED_TERMINAL_DECODE_POLICY as ROLE_LOCAL_SUPPORTED_TERMINAL_DECODE_POLICY,
)
from forge.model._synthesis_sampling.contracts import (
    STOCHASTIC_TERMINAL_DECODE_POLICIES as STOCHASTIC_TERMINAL_DECODE_POLICIES,
)
from forge.model._synthesis_sampling.contracts import (
    STRICT_TERMINAL_DECODE_POLICIES as STRICT_TERMINAL_DECODE_POLICIES,
)
from forge.model._synthesis_sampling.contracts import (
    SUPPORTED_SAMPLING_FACTORIZATIONS as SUPPORTED_SAMPLING_FACTORIZATIONS,
)
from forge.model._synthesis_sampling.contracts import (
    SUPPORTED_TERMINAL_DECODE_POLICIES as SUPPORTED_TERMINAL_DECODE_POLICIES,
)
from forge.model._synthesis_sampling.contracts import (
    TERMINAL_DECODE_POLICIES as TERMINAL_DECODE_POLICIES,
)
from forge.model._synthesis_sampling.contracts import (
    UGI_ESTER_TERMINAL_DECODE_POLICIES as UGI_ESTER_TERMINAL_DECODE_POLICIES,
)
from forge.model._synthesis_sampling.contracts import (
    UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY as UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY,
)
from forge.model._synthesis_sampling.contracts import (
    UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY as UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
)
from forge.model._synthesis_sampling.contracts import (
    UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY as UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
)
from forge.model._synthesis_sampling.contracts import (
    UGI_TOPOLOGY_COUPLED_TERMINAL_DECODE_POLICIES as UGI_TOPOLOGY_COUPLED_TERMINAL_DECODE_POLICIES,
)
from forge.model._synthesis_sampling.contracts import (
    UGI_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY as UGI_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
)
from forge.model._synthesis_sampling.contracts import (
    UGI_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY as UGI_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
)
from forge.model._synthesis_sampling.contracts import (
    SynthesisProgramSamplingError as SynthesisProgramSamplingError,
)
from forge.model._synthesis_sampling.decoder import (
    _strict_terminal_record as _strict_terminal_record,
)
from forge.model._synthesis_sampling.decoder import (
    decode_synthesis_program_strict_argmax as decode_synthesis_program_strict_argmax,
)
from forge.model._synthesis_sampling.decoder import (
    decode_synthesis_program_strict_topology as decode_synthesis_program_strict_topology,
)
from forge.model._synthesis_sampling.state import (
    _draw_source_categorical as _draw_source_categorical,
)
from forge.model._synthesis_sampling.state import _fixed_state_exact as _fixed_state_exact
from forge.model._synthesis_sampling.state import (
    _fixed_state_exact_records as _fixed_state_exact_records,
)
from forge.model._synthesis_sampling.state import (
    _fixed_state_exact_tensor as _fixed_state_exact_tensor,
)
from forge.model._synthesis_sampling.state import _initial_state as _initial_state
from forge.model._synthesis_sampling.state import (
    _initial_topology_conditioned_chemistry_state as _initial_topology_conditioned_chemistry_state,
)
from forge.model._synthesis_sampling.state import _move as _move
from forge.model._synthesis_sampling.state import _program_conditioning as _program_conditioning
from forge.model._synthesis_sampling.state import (
    _restore_fixed_states_in_place as _restore_fixed_states_in_place,
)
from forge.model._synthesis_sampling.state import _terminal_smiles as _terminal_smiles
from forge.model._synthesis_sampling.state import (
    _topology_conditioned_chemistry_flow as _topology_conditioned_chemistry_flow,
)
from forge.model.defog_feasibility import AtomState
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.potency_conditioning import PotencyCondition
from forge.model.reaction_core_saturation import (
    BoundReactionCoreSaturation,
    ReactionCoreSaturationPolicy,
)
from forge.model.reaction_program_flow import (
    collate_synthesis_program_layouts,
    decode_synthesis_program_argmax,
    resolve_synthesis_program_source_marginals,
)
from forge.model.sparse_topology_feasibility import (
    _endpoint_candidate_mask,
    _parent_candidate_mask,
    pointer_rstar_step,
)
from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord
from forge.model.ugi_all_role_semantic_program import UgiAllRoleSemanticTarget
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_mog_semantic_guidance import UgiMogSemanticGuidancePolicy
from forge.model.ugi_role_chemistry_prior import UgiRoleChemistryPrior
from forge.model.ugi_transformer_topology import (
    UGI_PROGRAM_ID,
    UgiTransformerTopologyError,
    UgiTransformerTopologyPolicy,
    decode_ugi_exact_topology,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


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
            terminal: Mapping[str, Any]
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
            terminal_cpu = {
                field: values.detach().to("cpu")
                for field, values in cast(Mapping[str, torch.Tensor], terminal).items()
            }
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
