"""Matched Ugi comparison of the frozen base and one authenticated specialist.

Both methods receive the same ordered morphology programs and sampling seed.  The only permitted
difference is the authenticated specialist delta; reaction-core decoding, assessment and attempt
budgets remain shared.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from experiments.phase1.product_l1.evaluation.ugi_v0_current_program_comparison import (
    _load_or_assess_current,
    _load_or_sample_current,
    _load_programs,
    _metric_deltas,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.reaction_core_saturation import ReactionCoreSaturationPolicy
from forge.model.synthesis_program_layout import SynthesisProgramLayoutPrior
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.model.ugi_transformer_topology import UgiTransformerTopologyPolicy
from forge.potency.annotations import ROLE_NAMES

CONFIG_SCHEMA_V3 = "forge.ugi_chemistry_specialist_comparison_config.v3"
CONFIG_SCHEMA = "forge.ugi_chemistry_specialist_comparison_config.v4"
CONFIG_SCHEMA_V5 = "forge.ugi_joint_lipid_specialist_comparison_config.v5"
CONFIG_SCHEMA_V6 = "forge.ugi_role_local_specialist_comparison_config.v6"
CONFIG_SCHEMA_V7 = "forge.ugi_measured_only_full_model_comparison_config.v7"
CONFIG_SCHEMA_V8 = "forge.ugi_structured_topology_specialist_comparison_config.v8"
RESULT_SCHEMA_V3 = "forge.ugi_chemistry_specialist_comparison_result.v3"
RESULT_SCHEMA = "forge.ugi_chemistry_specialist_comparison_result.v4"
RESULT_SCHEMA_V5 = "forge.ugi_joint_lipid_specialist_comparison_result.v5"
RESULT_SCHEMA_V6 = "forge.ugi_role_local_specialist_comparison_result.v6"
RESULT_SCHEMA_V7 = "forge.ugi_measured_only_full_model_comparison_result.v7"
RESULT_SCHEMA_V8 = "forge.ugi_structured_topology_specialist_comparison_result.v8"
PROGRAM_DRAW_SCHEMA = "forge.ugi_measured_joint_morphology_program_draw.v1"
SPECIALIST_RESULT_SCHEMA = "forge.reaction_program_chemistry_specialization_result.v3"
CONTEXTUAL_SPECIALIST_RESULT_SCHEMA = (
    "forge.reaction_program_contextual_chemistry_specialization_result.v4"
)
JOINT_LIPID_SPECIALIST_RESULT_SCHEMA = "forge.reaction_program_joint_lipid_specialization_result.v5"
ROLE_LOCAL_SPECIALIST_RESULT_SCHEMA = "forge.reaction_program_role_local_specialization_result.v6"
MEASURED_ONLY_FULL_RESULT_SCHEMA = "forge.reaction_program_measured_only_full_finetune_result.v7"
STRUCTURED_TOPOLOGY_RESULT_SCHEMA = (
    "forge.reaction_program_structured_topology_specialization_result.v8"
)
INPUT_LABELS_V3 = {
    "base_checkpoint_archive",
    "base_design",
    "base_training_result",
    "common_ugi_assessment_config",
    "lipid_realism_config",
    "local_chemistry_config",
    "production_cache",
    "program_draw",
    "qualified_reactions",
    "role_morphology_policy",
    "topology_closure_config",
    "topology_morphology_config",
    "ugi_assignments",
}
INPUT_LABELS = INPUT_LABELS_V3 - {"program_draw"}
RECOVERY_INPUT_LABELS = {"specialist_checkpoint", "specialist_result"}


class UgiChemistrySpecialistComparisonError(ValueError):
    """The matched chemistry-specialist assessment contract changed."""


def _validate(config: Mapping[str, Any], *, profile: str, device: str) -> Mapping[str, Any]:
    schema = config.get("schema_version")
    if schema not in {
        CONFIG_SCHEMA_V3,
        CONFIG_SCHEMA,
        CONFIG_SCHEMA_V5,
        CONFIG_SCHEMA_V6,
        CONFIG_SCHEMA_V7,
        CONFIG_SCHEMA_V8,
    }:
        raise UgiChemistrySpecialistComparisonError("unsupported comparison config")
    expected_inputs = INPUT_LABELS_V3 if schema == CONFIG_SCHEMA_V3 else INPUT_LABELS
    admitted_inputs = {frozenset(expected_inputs)}
    if schema == CONFIG_SCHEMA:
        admitted_inputs.add(frozenset(expected_inputs | RECOVERY_INPUT_LABELS))
    if (
        not isinstance(config.get("inputs"), Mapping)
        or frozenset(config["inputs"]) not in admitted_inputs
    ):
        raise UgiChemistrySpecialistComparisonError("comparison inputs changed")
    profiles = config.get("profiles")
    if not isinstance(profiles, Mapping) or not isinstance(profiles.get(profile), Mapping):
        raise UgiChemistrySpecialistComparisonError("comparison profile is missing")
    runtime = profiles[profile]
    if (
        runtime.get("device") != device
        or int(runtime.get("program_count", 0)) < 4
        or int(runtime.get("sample_steps", 0)) < 2
        or int(runtime.get("batch_size", 0)) < 1
        or (
            schema
            in {
                CONFIG_SCHEMA,
                CONFIG_SCHEMA_V5,
                CONFIG_SCHEMA_V6,
                CONFIG_SCHEMA_V7,
                CONFIG_SCHEMA_V8,
            }
            and int(runtime.get("layout_seed", -1)) < 0
        )
        or runtime.get("terminal_decode_policy")
        != "strict_ugi_ester_topology_role_local_chemistry_core_saturation"
    ):
        raise UgiChemistrySpecialistComparisonError("comparison runtime changed")
    if config.get("policy") != {
        "candidate_selection": False,
        "heldout_training_access": False,
        "method_blind_assessment": True,
        "paired_program_order": True,
        "paired_random_stream": True,
        "repairs_or_retries": False,
        "route_or_oracle_calls": 0,
        "training_calls": 0,
    }:
        raise UgiChemistrySpecialistComparisonError("comparison policy changed")
    decision = config.get("promotion_gate")
    if not isinstance(decision, Mapping) or dict(decision) != {
        "exact_l1_noninferiority_margin": 0.02,
        "local_support_noninferiority_margin": 0.02,
        "realism_c2st_required_reduction": 0.02,
        "validity_noninferiority_margin": 0.02,
    }:
        raise UgiChemistrySpecialistComparisonError("promotion gate changed")
    if schema in {
        CONFIG_SCHEMA,
        CONFIG_SCHEMA_V5,
        CONFIG_SCHEMA_V6,
        CONFIG_SCHEMA_V7,
        CONFIG_SCHEMA_V8,
    } and config.get("preflight_gate") != {
        "minimum_base_valid_fraction": 0.75,
        "minimum_specialist_valid_fraction": 0.75,
        "require_estimable_c2st": True,
        "require_zero_program_support_abstentions": True,
    }:
        raise UgiChemistrySpecialistComparisonError("comparison preflight gate changed")
    return runtime


def _program_from_layout(record: Any, *, sample_index: int) -> UgiMorphologyProgram:
    """Recover one role-ordered count program from an explicit sampled layout."""

    states = record.role_morphology_states
    if states is None:
        raise UgiChemistrySpecialistComparisonError(
            f"measured Ugi program row {sample_index} lacks role morphology"
        )
    values_by_role: dict[str, tuple[int, int, int, int]] = {}
    for role in ROLE_NAMES:
        try:
            role_state = next(
                block.role_state for block in record.component_blocks if block.role == role
            )
        except StopIteration as error:
            raise UgiChemistrySpecialistComparisonError(
                f"measured Ugi program row {sample_index} lacks role {role!r}"
            ) from error
        rows = states[record.role_states == int(role_state)]
        unique = np.unique(rows, axis=0)
        if unique.shape != (1, 4) or (unique < 1).any():
            raise UgiChemistrySpecialistComparisonError(
                f"measured Ugi program row {sample_index} has malformed role morphology"
            )
        values_by_role[role] = tuple(int(value) - 1 for value in unique[0])
    columns = tuple(
        tuple(values_by_role[role][column] for role in ROLE_NAMES) for column in range(4)
    )
    return UgiMorphologyProgram(
        node_counts=columns[0],
        junction_budgets=columns[1],
        cycle_ranks=columns[2],
        attachment_counts=columns[3],
    )


def _sample_supported_programs(
    cache_path: Path,
    ester_policy: UgiEsterChemotypePolicy,
    *,
    count: int,
    seed: int,
) -> tuple[tuple[UgiMorphologyProgram, ...], dict[str, Any]]:
    """Draw the paired program sequence from the decoder's own measured joint support."""

    cache = SynthesisProgramProductionCache(cache_path)
    try:
        prior = SynthesisProgramLayoutPrior(
            cache,
            ugi_ester_chemotype_policy=ester_policy,
        )
        layouts = prior.sample(
            "ugi_3cr_agile",
            sample_count=count,
            seed=seed,
            role_morphology_conditioning=True,
            exact_program_topology=True,
            ugi_ester_chemotype_policy=ester_policy,
        )
    finally:
        cache.close()
    programs = tuple(
        _program_from_layout(record, sample_index=index) for index, record in enumerate(layouts)
    )
    document = {
        "schema_version": PROGRAM_DRAW_SCHEMA,
        "source": "measured_training_joint_count_prior",
        "reaction_id": "ugi_3cr_agile",
        "seed": seed,
        "rows": len(programs),
        "component_identity_conditioning": False,
        "samples": [
            {
                "sample_index": index,
                "program": {
                    "attachment_counts": list(program.attachment_counts),
                    "cycle_ranks": list(program.cycle_ranks),
                    "junction_budgets": list(program.junction_budgets),
                    "node_counts": list(program.node_counts),
                },
            }
            for index, program in enumerate(programs)
        ],
    }
    return programs, document


def _program_support_abstentions(sampling: Mapping[str, Any]) -> int:
    reasons = sampling.get("strict_constraint_abstention_reasons", {})
    if not isinstance(reasons, Mapping):
        return 0
    return sum(
        int(count)
        for reason, count in reasons.items()
        if str(reason).startswith("ugi_topology_coupling_failure:")
    )


def _topology_policy(inputs: Mapping[str, Path]) -> UgiTransformerTopologyPolicy:
    closure = read_json_object(
        inputs["topology_closure_config"],
        error=UgiChemistrySpecialistComparisonError,
        label="Ugi closure topology support",
    )
    morphology = read_json_object(
        inputs["topology_morphology_config"],
        error=UgiChemistrySpecialistComparisonError,
        label="Ugi morphology topology support",
    )
    try:
        return UgiTransformerTopologyPolicy.from_support_documents(closure, morphology)
    except ValueError as error:
        raise UgiChemistrySpecialistComparisonError(
            "pinned Ugi topology policy is malformed"
        ) from error


def run_ugi_chemistry_specialist_comparison(
    config_path: Path,
    specialist_checkpoint_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    """Compare the chemistry delta with its exact base on paired Ugi programs."""

    config = read_json_object(
        config_path,
        error=UgiChemistrySpecialistComparisonError,
        label="Ugi chemistry-specialist comparison config",
    )
    runtime = _validate(config, profile=profile, device=device)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise UgiChemistrySpecialistComparisonError(
            f"comparison output directory is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        label: resolve_pin(pin, repo, label=label)
        for label, pin in sorted(config["inputs"].items())
    }
    if RECOVERY_INPUT_LABELS.issubset(inputs):
        if inputs["specialist_checkpoint"] != specialist_checkpoint_path.resolve():
            raise UgiChemistrySpecialistComparisonError(
                "pinned specialist checkpoint differs from the runtime checkpoint"
            )
        if (
            inputs["specialist_result"]
            != specialist_checkpoint_path.with_name("result.json").resolve()
        ):
            raise UgiChemistrySpecialistComparisonError(
                "pinned specialist result is not adjacent to its checkpoint"
            )
    specialist_result_path = specialist_checkpoint_path.with_name("result.json")
    specialist_result = read_json_object(
        specialist_result_path,
        error=UgiChemistrySpecialistComparisonError,
        label="Ugi chemistry-specialist result",
    )
    specialist_schema = specialist_result.get("schema_version")
    admitted_delta_policies = {
        SPECIALIST_RESULT_SCHEMA: "chemistry_output_adapters_only",
        CONTEXTUAL_SPECIALIST_RESULT_SCHEMA: "contextual_chemistry_refinement_only",
        JOINT_LIPID_SPECIALIST_RESULT_SCHEMA: "joint_lipid_adapter_only",
        ROLE_LOCAL_SPECIALIST_RESULT_SCHEMA: "role_local_tree_decoder_only",
        MEASURED_ONLY_FULL_RESULT_SCHEMA: "full_model_measured_only",
        STRUCTURED_TOPOLOGY_RESULT_SCHEMA: "structured_ugi_topology_head_only",
    }
    joint_lipid_specialist = specialist_schema == JOINT_LIPID_SPECIALIST_RESULT_SCHEMA
    role_local_specialist = specialist_schema == ROLE_LOCAL_SPECIALIST_RESULT_SCHEMA
    measured_only_full_model = specialist_schema == MEASURED_ONLY_FULL_RESULT_SCHEMA
    structured_topology_specialist = specialist_schema == STRUCTURED_TOPOLOGY_RESULT_SCHEMA
    if (
        specialist_schema not in admitted_delta_policies
        or specialist_result.get("status") != "pass"
        or specialist_result.get("target_program") != "ugi_3cr_agile"
        or specialist_result.get("specialist", {}).get("delta_parameter_policy")
        != admitted_delta_policies.get(specialist_schema)
        or (
            not (
                joint_lipid_specialist
                or role_local_specialist
                or measured_only_full_model
                or structured_topology_specialist
            )
            and specialist_result.get("gates", {}).get("topology_outputs_frozen") is not True
        )
        or (
            specialist_schema == CONTEXTUAL_SPECIALIST_RESULT_SCHEMA
            and specialist_result.get("gates", {}).get("contextual_features_declared") is not True
        )
        or (
            (joint_lipid_specialist or role_local_specialist or measured_only_full_model)
            and specialist_result.get("gates", {}).get("joint_lipid_measure_exact") is not True
        )
        or (
            measured_only_full_model
            and specialist_result.get("gates", {}).get("full_model_parameter_scope_exact")
            is not True
        )
        or (
            structured_topology_specialist
            and (
                specialist_result.get("gates", {}).get("structured_topology_head_supervised")
                is not True
                or specialist_result.get("gates", {}).get("structured_grammar_policy_pinned")
                is not True
            )
        )
        or specialist_result.get("checkpoint", {}).get("sha256")
        != sha256_file(specialist_checkpoint_path)
    ):
        raise UgiChemistrySpecialistComparisonError("chemistry-specialist result is not admissible")

    local_support = LocalChemistrySupport.from_mapping(
        read_json_object(
            inputs["role_morphology_policy"],
            error=UgiChemistrySpecialistComparisonError,
            label="role-local chemistry support",
        )
    )
    topology_policy = _topology_policy(inputs)
    core_policy = ReactionCoreSaturationPolicy.from_qualified_registry(
        inputs["qualified_reactions"],
        reaction_id="ugi_3cr_agile",
        expected_sha256=sha256_file(inputs["qualified_reactions"]),
    )
    ester_policy = UgiEsterChemotypePolicy.from_qualified_registry(
        inputs["qualified_reactions"],
        training_assignments_path=inputs["ugi_assignments"],
        reaction_id="ugi_3cr_agile",
        expected_sha256=sha256_file(inputs["qualified_reactions"]),
        expected_training_assignments_sha256=sha256_file(inputs["ugi_assignments"]),
    )
    if config["schema_version"] == CONFIG_SCHEMA_V3:
        programs = _load_programs(inputs["program_draw"], count=int(runtime["program_count"]))
        program_draw_path = inputs["program_draw"]
    else:
        programs, program_draw = _sample_supported_programs(
            inputs["production_cache"],
            ester_policy,
            count=int(runtime["program_count"]),
            seed=int(runtime["layout_seed"]),
        )
        program_draw_path = output_dir / "paired_program_draw.json"
        if program_draw_path.is_file():
            persisted = read_json_object(
                program_draw_path,
                error=UgiChemistrySpecialistComparisonError,
                label="paired measured-Ugi program draw",
            )
            if persisted != program_draw:
                raise UgiChemistrySpecialistComparisonError(
                    "persisted measured-Ugi program draw changed"
                )
        else:
            write_json(program_draw_path, program_draw)
    current_inputs = {
        "checkpoint_archive": inputs["base_checkpoint_archive"],
        "common_ugi_assessment_config": inputs["common_ugi_assessment_config"],
        "current_training_result": inputs["base_training_result"],
        "lipid_realism_config": inputs["lipid_realism_config"],
        "local_chemistry_config": inputs["local_chemistry_config"],
        "production_cache": inputs["production_cache"],
        "production_design": inputs["base_design"],
        "role_morphology_policy": inputs["role_morphology_policy"],
    }
    if config["schema_version"] == CONFIG_SCHEMA_V3:
        current_inputs["program_draw"] = inputs["program_draw"]
    sampling_runtime = {
        "batch_size": int(runtime["batch_size"]),
        "current_terminal_decode_policy": str(runtime["terminal_decode_policy"]),
        "flow_seed": int(runtime["flow_seed"]),
        "program_count": int(runtime["program_count"]),
        "sample_steps": int(runtime["sample_steps"]),
        "terminal_decoder_seed": None,
    }
    shared_arguments = {
        "repo": repo,
        "inputs": current_inputs,
        "runtime": sampling_runtime,
        "config_sha256": sha256_file(config_path),
        "device": device,
        "arm_id": str(config["base_checkpoint"]["arm_id"]),
        "checkpoint_step": int(config["base_checkpoint"]["step"]),
        "programs": programs,
        "program_draw_sha256": sha256_file(program_draw_path),
        "ugi_topology_policy": topology_policy,
        "local_chemistry_support": local_support,
        "reaction_core_saturation_policy": core_policy,
        "ugi_ester_chemotype_policy": ester_policy,
    }
    base_rows, base_sampling, base_sampling_path = _load_or_sample_current(
        output_dir=output_dir / "base_sampling",
        **shared_arguments,
    )
    specialist_rows, specialist_sampling, specialist_sampling_path = _load_or_sample_current(
        output_dir=output_dir / "specialist_sampling",
        specialist_checkpoint=specialist_checkpoint_path,
        **shared_arguments,
    )
    base_assessment = _load_or_assess_current(
        rows=base_rows,
        sampling_path=base_sampling_path,
        output_dir=output_dir / "base_assessment",
        inputs=current_inputs,
        repo=repo,
        method_id="forge_seed0_base_topology_first",
    )
    specialist_assessment = _load_or_assess_current(
        rows=specialist_rows,
        sampling_path=specialist_sampling_path,
        output_dir=output_dir / "specialist_assessment",
        inputs=current_inputs,
        repo=repo,
        method_id=(
            "forge_seed0_ugi_joint_lipid_specialist_topology_first"
            if joint_lipid_specialist
            else "forge_seed0_ugi_role_local_specialist_topology_first"
            if role_local_specialist
            else "forge_seed0_ugi_measured_only_full_model_topology_first"
            if measured_only_full_model
            else "forge_seed0_ugi_structured_topology_specialist"
            if structured_topology_specialist
            else "forge_seed0_ugi_chemistry_specialist_topology_first"
        ),
    )
    base_metrics = dict(base_assessment["assessment"]["metrics"])
    specialist_metrics = dict(specialist_assessment["assessment"]["metrics"])
    deltas = _metric_deltas(specialist_metrics, base_metrics)
    promotion = config["promotion_gate"]
    required = {
        "exact_l1_yield_per_attempt",
        "local_support_qualified_exact_l1_yield_per_attempt",
        "realism_c2st_auc",
        "valid_fraction_per_attempt",
    }
    if role_local_specialist or measured_only_full_model:
        promotion_decision = "defer_to_role_realism_panel"
    elif any(deltas.get(name) is None for name in required):
        promotion_decision = "not_estimable"
    elif (
        float(deltas["exact_l1_yield_per_attempt"])
        >= -float(promotion["exact_l1_noninferiority_margin"])
        and float(deltas["valid_fraction_per_attempt"])
        >= -float(promotion["validity_noninferiority_margin"])
        and float(deltas["local_support_qualified_exact_l1_yield_per_attempt"])
        >= -float(promotion["local_support_noninferiority_margin"])
        and float(deltas["realism_c2st_auc"])
        <= -float(promotion["realism_c2st_required_reduction"])
    ):
        promotion_decision = (
            "promote_joint_lipid_specialist"
            if joint_lipid_specialist
            else "promote_structured_topology_specialist"
            if structured_topology_specialist
            else "promote_chemistry_specialist"
        )
    else:
        promotion_decision = "do_not_promote"

    paired_programs = len(base_rows) == len(specialist_rows) == len(programs) and all(
        base_rows[index]["program"] == specialist_rows[index]["program"]
        for index in range(len(programs))
    )
    specialist_method_key = (
        "joint_lipid_specialist"
        if joint_lipid_specialist
        else "role_local_specialist"
        if role_local_specialist
        else "measured_only_full_model"
        if measured_only_full_model
        else "structured_topology_specialist"
        if structured_topology_specialist
        else "chemistry_specialist"
    )
    delta_key = f"{specialist_method_key}_minus_base"
    structural_gates = {
        "attempt_denominator_matched": len(base_rows)
        == len(specialist_rows)
        == int(runtime["program_count"]),
        "candidate_selection_absent": True,
        "specialist_delta_scope_exact": (
            specialist_result["gates"]["joint_lipid_measure_exact"]
            and specialist_result["gates"]["full_model_parameter_scope_exact"]
            if measured_only_full_model
            else specialist_result["gates"]["joint_lipid_measure_exact"]
            if joint_lipid_specialist or role_local_specialist
            else specialist_result["gates"]["structured_topology_head_supervised"]
            and specialist_result["gates"]["structured_grammar_policy_pinned"]
            if structured_topology_specialist
            else specialist_result["gates"]["topology_outputs_frozen"]
        ),
        "method_blind_assessment_shared": True,
        "no_repairs_or_retries": not base_sampling.get("repairs")
        and not specialist_sampling.get("repairs"),
        "paired_program_order": paired_programs,
        "route_or_oracle_calls_zero": True,
        "training_calls_zero": True,
    }
    preflight_gate = config.get("preflight_gate")
    preflight_gates: dict[str, bool] | None = None
    if isinstance(preflight_gate, Mapping):
        preflight_gates = {
            "base_valid_fraction_sufficient": float(base_metrics["valid_fraction_per_attempt"])
            >= float(preflight_gate["minimum_base_valid_fraction"]),
            "specialist_valid_fraction_sufficient": float(
                specialist_metrics["valid_fraction_per_attempt"]
            )
            >= float(preflight_gate["minimum_specialist_valid_fraction"]),
            "c2st_estimable": (
                not bool(preflight_gate["require_estimable_c2st"])
                or (
                    base_metrics["realism_c2st_auc"] is not None
                    and specialist_metrics["realism_c2st_auc"] is not None
                )
            ),
            "program_support_abstentions_zero": (
                not bool(preflight_gate["require_zero_program_support_abstentions"])
                or (
                    _program_support_abstentions(base_sampling) == 0
                    and _program_support_abstentions(specialist_sampling) == 0
                )
            ),
        }
    result_schema = (
        RESULT_SCHEMA_V3
        if config["schema_version"] == CONFIG_SCHEMA_V3
        else RESULT_SCHEMA_V5
        if config["schema_version"] == CONFIG_SCHEMA_V5
        else RESULT_SCHEMA_V6
        if config["schema_version"] == CONFIG_SCHEMA_V6
        else RESULT_SCHEMA_V7
        if config["schema_version"] == CONFIG_SCHEMA_V7
        else RESULT_SCHEMA_V8
        if config["schema_version"] == CONFIG_SCHEMA_V8
        else RESULT_SCHEMA
    )
    result = {
        "schema_version": result_schema,
        "status": (
            "complete"
            if all(structural_gates.values())
            and (profile != "h100_preflight" or all((preflight_gates or {}).values()))
            else "fail"
        ),
        "profile": profile,
        "programs_per_method": len(programs),
        "methods": {
            "base": {
                "metrics": base_metrics,
                "sampling_summary": base_sampling,
                "sampling": artifact_record(base_sampling_path),
                "assessment": artifact_record(output_dir / "base_assessment" / "result_index.json"),
            },
            specialist_method_key: {
                "metrics": specialist_metrics,
                "sampling_summary": specialist_sampling,
                "sampling": artifact_record(specialist_sampling_path),
                "assessment": artifact_record(
                    output_dir / "specialist_assessment" / "result_index.json"
                ),
            },
        },
        delta_key: deltas,
        "specialist_method_key": specialist_method_key,
        "promotion_gate": dict(promotion),
        "promotion_decision": promotion_decision,
        "structural_gates": structural_gates,
        "preflight_gates": preflight_gates,
        "program_draw": artifact_record(program_draw_path),
        "specialist_checkpoint": artifact_record(specialist_checkpoint_path),
        "specialist_result": artifact_record(specialist_result_path),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(inputs.items())},
        "candidate_selection": False,
        "calls": {"training": 0, "route": 0, "oracle": 0},
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    if result["status"] != "complete":
        raise UgiChemistrySpecialistComparisonError(
            f"comparison gates failed: structural={structural_gates}, preflight={preflight_gates}"
        )
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "CONFIG_SCHEMA_V5",
    "CONFIG_SCHEMA_V6",
    "CONFIG_SCHEMA_V7",
    "CONFIG_SCHEMA_V8",
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_V5",
    "RESULT_SCHEMA_V6",
    "RESULT_SCHEMA_V7",
    "RESULT_SCHEMA_V8",
    "UgiChemistrySpecialistComparisonError",
    "run_ugi_chemistry_specialist_comparison",
]
