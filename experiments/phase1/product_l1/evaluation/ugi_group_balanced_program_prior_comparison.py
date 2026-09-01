"""Matched comparison of occurrence- and group-balanced measured-Ugi program priors.

The molecular checkpoint, flow seed, decoder, verifier and attempt budget are identical.  The
intervention changes only the categorical law over identity-free coarse morphology programs.  Both
laws are fitted from source-adjudicated measured training products admitted by the same frozen Ugi
ester decoder support.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
    _load_or_assess_current,
    _load_or_sample_current,
    _metric_deltas,
    _program_support_abstentions,
    _sample_supported_programs,
    _topology_policy,
)
from experiments.phase1.product_l1.evaluation.ugi_v0_current_program_comparison import (
    _load_programs,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.reaction_core_saturation import ReactionCoreSaturationPolicy
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy

CONFIG_SCHEMA = "forge.ugi_group_balanced_program_prior_comparison_config.v1"
RESULT_SCHEMA = "forge.ugi_group_balanced_program_prior_comparison_result.v1"
PROGRAM_DRAW_SCHEMA = "forge.ugi_measured_joint_morphology_program_draw.v2"
INPUT_LABELS = {
    "base_checkpoint_archive",
    "base_design",
    "base_training_result",
    "common_ugi_assessment_config",
    "group_balanced_program_draw",
    "lipid_realism_config",
    "local_chemistry_config",
    "production_cache",
    "qualified_reactions",
    "role_morphology_policy",
    "topology_closure_config",
    "topology_morphology_config",
    "ugi_assignments",
}


class UgiGroupBalancedProgramPriorComparisonError(ValueError):
    """The matched morphology-prior comparison contract changed."""


def _runtime(config: Mapping[str, Any], *, profile: str, device: str) -> Mapping[str, Any]:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not isinstance(config.get("inputs"), Mapping)
        or set(config["inputs"]) != INPUT_LABELS
        or not isinstance(config.get("profiles"), Mapping)
        or not isinstance(config["profiles"].get(profile), Mapping)
    ):
        raise UgiGroupBalancedProgramPriorComparisonError(
            "unsupported group-balanced program-prior comparison config"
        )
    runtime = config["profiles"][profile]
    if (
        runtime.get("device") != device
        or int(runtime.get("program_count", 0)) < 4
        or int(runtime.get("sample_steps", 0)) < 2
        or int(runtime.get("batch_size", 0)) < 1
        or int(runtime.get("program_seed", -1)) < 0
        or int(runtime.get("flow_seed", -1)) < 0
        or runtime.get("terminal_decode_policy")
        != "strict_ugi_ester_topology_role_local_chemistry_core_saturation"
    ):
        raise UgiGroupBalancedProgramPriorComparisonError("comparison runtime changed")
    if config.get("policy") != {
        "candidate_selection": False,
        "component_identity_conditioning": False,
        "heldout_training_access": False,
        "method_blind_assessment": True,
        "program_laws_intentionally_different": True,
        "program_random_seed_matched": True,
        "paired_flow_random_stream": True,
        "repairs_or_retries": False,
        "route_or_oracle_calls": 0,
        "training_calls": 0,
    }:
        raise UgiGroupBalancedProgramPriorComparisonError("comparison scientific policy changed")
    if config.get("promotion_gate") != {
        "effective_component_count_minimum_retained_ratio": 0.8,
        "exact_l1_noninferiority_margin": 0.02,
        "local_support_noninferiority_margin": 0.02,
        "mean_pairwise_ecfp4_distance_noninferiority_margin": 0.05,
        "realism_c2st_required_reduction": 0.02,
        "unique_exact_l1_noninferiority_margin": 0.05,
        "validity_noninferiority_margin": 0.02,
    }:
        raise UgiGroupBalancedProgramPriorComparisonError("promotion gate changed")
    if config.get("preflight_gate") != {
        "minimum_control_valid_fraction": 0.75,
        "minimum_group_balanced_valid_fraction": 0.75,
        "require_estimable_c2st": True,
        "require_zero_program_support_abstentions": True,
    }:
        raise UgiGroupBalancedProgramPriorComparisonError("preflight gate changed")
    base = config.get("base_checkpoint")
    if (
        not isinstance(base, Mapping)
        or not isinstance(base.get("arm_id"), str)
        or not base["arm_id"]
        or isinstance(base.get("step"), bool)
        or int(base.get("step", 0)) < 1
    ):
        raise UgiGroupBalancedProgramPriorComparisonError("base checkpoint changed")
    return runtime


def _passes_promotion(
    deltas: Mapping[str, float | None],
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> tuple[bool, dict[str, bool | float | None]]:
    required = {
        "exact_l1_yield_per_attempt",
        "local_support_qualified_exact_l1_yield_per_attempt",
        "mean_pairwise_ecfp4_distance",
        "realism_c2st_auc",
        "unique_exact_l1_products_per_attempt",
        "valid_fraction_per_attempt",
    }
    estimable = not any(deltas.get(metric) is None for metric in required)
    control_effective = control.get("effective_component_count")
    treatment_effective = treatment.get("effective_component_count")
    effective_ratio = (
        None
        if control_effective is None or treatment_effective is None or float(control_effective) <= 0
        else float(treatment_effective) / float(control_effective)
    )
    checks: dict[str, bool | float | None] = {
        "required_metrics_estimable": estimable,
        "effective_component_count_retained_ratio": effective_ratio,
        "effective_component_count_preserved": (
            None
            if effective_ratio is None
            else effective_ratio >= float(gate["effective_component_count_minimum_retained_ratio"])
        ),
    }
    if not estimable:
        return False, checks
    checks.update(
        {
            "exact_l1_preserved": float(deltas["exact_l1_yield_per_attempt"])
            >= -float(gate["exact_l1_noninferiority_margin"]),
            "local_support_preserved": float(
                deltas["local_support_qualified_exact_l1_yield_per_attempt"]
            )
            >= -float(gate["local_support_noninferiority_margin"]),
            "validity_preserved": float(deltas["valid_fraction_per_attempt"])
            >= -float(gate["validity_noninferiority_margin"]),
            "realism_improved": float(deltas["realism_c2st_auc"])
            <= -float(gate["realism_c2st_required_reduction"]),
            "pairwise_diversity_preserved": float(deltas["mean_pairwise_ecfp4_distance"])
            >= -float(gate["mean_pairwise_ecfp4_distance_noninferiority_margin"]),
            "unique_exact_l1_preserved": float(deltas["unique_exact_l1_products_per_attempt"])
            >= -float(gate["unique_exact_l1_noninferiority_margin"]),
        }
    )
    return (
        all(
            value is True
            for key, value in checks.items()
            if key != "effective_component_count_retained_ratio"
        ),
        checks,
    )


def run_ugi_group_balanced_program_prior_comparison(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    """Compare two train-fold program laws using one frozen molecular generator."""

    config = read_json_object(
        config_path,
        error=UgiGroupBalancedProgramPriorComparisonError,
        label="group-balanced Ugi program-prior comparison config",
    )
    runtime = _runtime(config, profile=profile, device=device)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise UgiGroupBalancedProgramPriorComparisonError(
            f"comparison output directory is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        label: resolve_pin(pin, repo, label=label)
        for label, pin in sorted(config["inputs"].items())
    }
    group_document = read_json_object(
        inputs["group_balanced_program_draw"],
        error=UgiGroupBalancedProgramPriorComparisonError,
        label="group-balanced measured-Ugi program draw",
    )
    if (
        group_document.get("schema_version") != PROGRAM_DRAW_SCHEMA
        or group_document.get("status") != "pass"
        or group_document.get("source") != "group_balanced_measured_training_joint_count_prior"
        or group_document.get("reaction_id") != "ugi_3cr_agile"
        or group_document.get("seed") != int(runtime["program_seed"])
        or int(group_document.get("rows", 0)) < int(runtime["program_count"])
        or group_document.get("component_identity_conditioning") is not False
        or group_document.get("heldout_access") is not False
        or group_document.get("training_generation_repair_retry_route_or_oracle_calls") != 0
        or group_document.get("candidate_selection") is not False
    ):
        raise UgiGroupBalancedProgramPriorComparisonError(
            "group-balanced program draw is not admissible"
        )

    local_support = LocalChemistrySupport.from_mapping(
        read_json_object(
            inputs["role_morphology_policy"],
            error=UgiGroupBalancedProgramPriorComparisonError,
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
    control_programs, control_draw = _sample_supported_programs(
        inputs["production_cache"],
        ester_policy,
        count=int(runtime["program_count"]),
        seed=int(runtime["program_seed"]),
    )
    control_draw_path = output_dir / "occurrence_weighted_program_draw.json"
    if control_draw_path.is_file():
        persisted = read_json_object(
            control_draw_path,
            error=UgiGroupBalancedProgramPriorComparisonError,
            label="persisted occurrence-weighted program draw",
        )
        if persisted != control_draw:
            raise UgiGroupBalancedProgramPriorComparisonError(
                "persisted occurrence-weighted program draw changed"
            )
    else:
        write_json(control_draw_path, control_draw)
    treatment_programs = _load_programs(
        inputs["group_balanced_program_draw"], count=int(runtime["program_count"])
    )

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
    sampling_runtime = {
        "batch_size": int(runtime["batch_size"]),
        "current_terminal_decode_policy": str(runtime["terminal_decode_policy"]),
        "flow_seed": int(runtime["flow_seed"]),
        "program_count": int(runtime["program_count"]),
        "sample_steps": int(runtime["sample_steps"]),
        "terminal_decoder_seed": None,
    }
    common_arguments = {
        "repo": repo,
        "inputs": current_inputs,
        "runtime": sampling_runtime,
        "config_sha256": sha256_file(config_path),
        "device": device,
        "arm_id": str(config["base_checkpoint"]["arm_id"]),
        "checkpoint_step": int(config["base_checkpoint"]["step"]),
        "ugi_topology_policy": topology_policy,
        "local_chemistry_support": local_support,
        "reaction_core_saturation_policy": core_policy,
        "ugi_ester_chemotype_policy": ester_policy,
    }
    arms = {
        "occurrence_weighted": (
            control_programs,
            control_draw_path,
            "forge_seed0_occurrence_weighted_measured_program_prior",
        ),
        "group_balanced": (
            treatment_programs,
            inputs["group_balanced_program_draw"],
            "forge_seed0_group_balanced_measured_program_prior",
        ),
    }
    methods: dict[str, Any] = {}
    for label, (programs, draw_path, method_id) in arms.items():
        rows, sampling, sampling_path = _load_or_sample_current(
            output_dir=output_dir / label / "sampling",
            programs=programs,
            program_draw_sha256=sha256_file(draw_path),
            **common_arguments,
        )
        assessment = _load_or_assess_current(
            rows=rows,
            sampling_path=sampling_path,
            output_dir=output_dir / label / "assessment",
            inputs=current_inputs,
            repo=repo,
            method_id=method_id,
        )
        methods[label] = {
            "metrics": dict(assessment["assessment"]["metrics"]),
            "sampling": artifact_record(sampling_path),
            "sampling_summary": sampling,
            "assessment": artifact_record(output_dir / label / "assessment" / "result_index.json"),
            "program_draw": artifact_record(draw_path),
            "program_support_abstentions": _program_support_abstentions(sampling),
        }
    control = methods["occurrence_weighted"]["metrics"]
    treatment = methods["group_balanced"]["metrics"]
    deltas = _metric_deltas(treatment, control)
    promotes, promotion_checks = _passes_promotion(
        deltas, control, treatment, config["promotion_gate"]
    )
    preflight = config["preflight_gate"]
    preflight_checks = {
        "control_valid_fraction_sufficient": float(control["valid_fraction_per_attempt"])
        >= float(preflight["minimum_control_valid_fraction"]),
        "group_balanced_valid_fraction_sufficient": float(treatment["valid_fraction_per_attempt"])
        >= float(preflight["minimum_group_balanced_valid_fraction"]),
        "c2st_estimable": (
            not bool(preflight["require_estimable_c2st"])
            or (
                control.get("realism_c2st_auc") is not None
                and treatment.get("realism_c2st_auc") is not None
            )
        ),
        "program_support_abstentions_zero": (
            not bool(preflight["require_zero_program_support_abstentions"])
            or all(method["program_support_abstentions"] == 0 for method in methods.values())
        ),
    }
    structural_checks = {
        "attempt_denominator_matched": all(
            int(method["sampling_summary"]["samples"]) == int(runtime["program_count"])
            for method in methods.values()
        ),
        "candidate_selection_absent": True,
        "component_identity_conditioning_absent": True,
        "method_blind_assessment_shared": True,
        "no_repairs_or_retries": all(
            not method["sampling_summary"].get("repairs") for method in methods.values()
        ),
        "route_or_oracle_calls_zero": True,
        "training_calls_zero": True,
    }
    status = (
        "complete"
        if all(structural_checks.values())
        and (profile != "h100_preflight" or all(preflight_checks.values()))
        else "fail"
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "profile": profile,
        "programs_per_method": int(runtime["program_count"]),
        "methods": methods,
        "group_balanced_minus_occurrence_weighted": deltas,
        "promotion_gate": dict(config["promotion_gate"]),
        "promotion_checks": promotion_checks,
        "promotion_decision": (
            "diagnostic_only"
            if profile == "smoke"
            else (
                "eligible_for_full_comparison"
                if profile == "h100_preflight" and status == "complete"
                else (
                    "promote_group_balanced_program_prior"
                    if profile == "full" and promotes
                    else "do_not_promote"
                )
            )
        ),
        "preflight_checks": preflight_checks,
        "structural_checks": structural_checks,
        "program_pairing": {
            "same_attempt_count": True,
            "same_program_random_seed": True,
            "same_flow_random_seed": True,
            "identical_programs": False,
            "reason": "the categorical program law is the intervention",
        },
        "inputs": {
            "config": pin_record(config_path, repo),
            **{label: pin_record(path, repo) for label, path in sorted(inputs.items())},
        },
        "policy": dict(config["policy"]),
        "candidate_selection": False,
        "calls": {"training": 0, "route": 0, "oracle": 0},
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    if status != "complete":
        raise UgiGroupBalancedProgramPriorComparisonError(
            f"comparison gates failed: structural={structural_checks}, preflight={preflight_checks}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "h100_preflight", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    result = run_ugi_group_balanced_program_prior_comparison(
        args.config,
        repo,
        args.output_dir,
        profile=args.profile,
        device=args.device,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiGroupBalancedProgramPriorComparisonError",
    "run_ugi_group_balanced_program_prior_comparison",
]
