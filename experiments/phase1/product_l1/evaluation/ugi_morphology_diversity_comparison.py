"""Compare the frozen and entropy-calibrated Ugi morphology-program laws."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.phase1.product_l1.evaluation.ugi_all_role_semantic_program_comparison import (
    _load_draw,
    _semantic_support_audit,
)
from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
    _program_support_abstentions,
    _topology_policy,
)
from experiments.phase1.product_l1.evaluation.ugi_v0_current_program_comparison import (
    _load_or_assess_current,
    _load_or_sample_current,
    _metric_deltas,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.reaction_core_saturation import ReactionCoreSaturationPolicy
from forge.model.synthesis_program_sampling import (
    UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY,
)
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_mog_semantic_guidance import UgiMogSemanticGuidancePolicy
from forge.model.ugi_role_chemistry_prior import UgiRoleChemistryPrior
from forge.model.ugi_topology_diversity import summarize_topology_diversity

CONFIG_SCHEMA = "forge.ugi_morphology_diversity_comparison_config.v1"
RESULT_SCHEMA = "forge.ugi_morphology_diversity_comparison_result.v1"

INPUT_LABELS = {
    "base_checkpoint_archive",
    "base_design",
    "base_training_result",
    "baseline_program_draw",
    "common_ugi_assessment_config",
    "lipid_realism_config",
    "local_chemistry_config",
    "production_cache",
    "qualified_reactions",
    "role_morphology_policy",
    "semantic_atoms",
    "semantic_bonds",
    "topology_closure_config",
    "topology_morphology_config",
    "treatment_program_draw",
    "ugi_assignments",
}


class UgiMorphologyDiversityComparisonError(ValueError):
    """The morphology-diversity experiment contract changed."""


def _runtime(config: Mapping[str, Any], *, profile: str, device: str) -> Mapping[str, Any]:
    required = {
        "schema_version",
        "scientific_question",
        "inputs",
        "base_checkpoint",
        "semantic_guidance",
        "profiles",
        "policy",
        "preflight_gate",
        "full_promotion_gate",
        "nonclaims",
    }
    if (
        set(config) != required
        or config.get("schema_version") != CONFIG_SCHEMA
        or not isinstance(config.get("inputs"), Mapping)
        or set(config["inputs"]) != INPUT_LABELS
        or not isinstance(config.get("profiles"), Mapping)
        or set(config["profiles"]) != {"smoke", "h100_preflight", "full"}
        or not isinstance(config["profiles"].get(profile), Mapping)
    ):
        raise UgiMorphologyDiversityComparisonError("morphology-diversity config changed")
    if config.get("base_checkpoint") != {
        "arm_id": "shared_bias_program_role_source",
        "step": 9143,
    }:
        raise UgiMorphologyDiversityComparisonError("morphology-diversity checkpoint changed")
    if not isinstance(config.get("semantic_guidance"), Mapping):
        raise UgiMorphologyDiversityComparisonError("semantic-guidance policy is missing")
    if not isinstance(config.get("nonclaims"), list) or not config["nonclaims"]:
        raise UgiMorphologyDiversityComparisonError("morphology-diversity nonclaims changed")
    runtime = config["profiles"][profile]
    if (
        runtime.get("device") != device
        or int(runtime.get("program_count", 0)) < 8
        or int(runtime.get("sample_steps", 0)) < 2
        or int(runtime.get("batch_size", 0)) < 1
        or int(runtime.get("flow_seed", -1)) < 0
        or int(runtime.get("terminal_decoder_seed", -1)) < 0
        or runtime.get("terminal_decode_policy")
        != UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY
    ):
        raise UgiMorphologyDiversityComparisonError("morphology-diversity runtime changed")
    if config.get("policy") != {
        "candidate_selection": False,
        "component_identity_conditioning": False,
        "heldout_training_access": False,
        "method_blind_assessment": True,
        "program_laws_intentionally_different": True,
        "repairs_or_retries": False,
        "route_or_oracle_calls": 0,
        "same_checkpoint_and_decoder": True,
        "same_flow_random_stream": True,
        "same_local_chemistry_guidance": True,
        "training_calls": 0,
    }:
        raise UgiMorphologyDiversityComparisonError("morphology-diversity policy changed")
    if set(config.get("preflight_gate", {})) != {
        "minimum_effective_topology_count_ratio",
        "minimum_exact_l1_yield",
        "minimum_valid_fraction",
    }:
        raise UgiMorphologyDiversityComparisonError("morphology-diversity preflight gate changed")
    if set(config.get("full_promotion_gate", {})) != {
        "descriptor_precision_noninferiority_margin",
        "effective_component_count_retention_minimum",
        "effective_topology_count_required_ratio",
        "exact_l1_absolute_minimum",
        "realism_c2st_required_reduction",
        "unique_exact_l1_retention_minimum",
        "valid_fraction_minimum",
    }:
        raise UgiMorphologyDiversityComparisonError("morphology-diversity promotion gate changed")
    return runtime


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0 else numerator / denominator


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _promotion_checks(
    *,
    baseline: Mapping[str, Any],
    treatment: Mapping[str, Any],
    baseline_topology: Mapping[str, Any],
    treatment_topology: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    effective_component_ratio = _ratio(
        float(treatment["effective_component_count"]),
        float(baseline["effective_component_count"]),
    )
    unique_exact_ratio = _ratio(
        float(treatment["unique_exact_l1_products_per_attempt"]),
        float(baseline["unique_exact_l1_products_per_attempt"]),
    )
    topology_ratio = _ratio(
        float(treatment_topology["effective_complete_topology_count"]),
        float(baseline_topology["effective_complete_topology_count"]),
    )
    baseline_c2st = _numeric(baseline["realism_c2st_auc"])
    treatment_c2st = _numeric(treatment["realism_c2st_auc"])
    baseline_fingerprint = _numeric(baseline["fingerprint_manifold_precision_per_attempt"])
    treatment_fingerprint = _numeric(treatment["fingerprint_manifold_precision_per_attempt"])
    baseline_descriptor = _numeric(baseline["descriptor_manifold_precision_per_attempt"])
    treatment_descriptor = _numeric(treatment["descriptor_manifold_precision_per_attempt"])
    return {
        "exact_l1_absolute_minimum": float(treatment["exact_l1_yield_per_attempt"])
        >= float(gate["exact_l1_absolute_minimum"]),
        "validity_absolute_minimum": float(treatment["valid_fraction_per_attempt"])
        >= float(gate["valid_fraction_minimum"]),
        "effective_component_count_retained_ratio": effective_component_ratio,
        "effective_component_count_preserved": (
            effective_component_ratio is not None
            and effective_component_ratio
            >= float(gate["effective_component_count_retention_minimum"])
        ),
        "unique_exact_l1_retained_ratio": unique_exact_ratio,
        "unique_exact_l1_preserved": (
            unique_exact_ratio is not None
            and unique_exact_ratio >= float(gate["unique_exact_l1_retention_minimum"])
        ),
        "effective_topology_count_ratio": topology_ratio,
        "coarse_topology_diversity_improved": (
            topology_ratio is not None
            and topology_ratio >= float(gate["effective_topology_count_required_ratio"])
        ),
        "realism_c2st_improved": (
            baseline_c2st is not None
            and treatment_c2st is not None
            and treatment_c2st <= baseline_c2st - float(gate["realism_c2st_required_reduction"])
        ),
        "fingerprint_manifold_precision_improved": (
            baseline_fingerprint is not None
            and treatment_fingerprint is not None
            and treatment_fingerprint > baseline_fingerprint
        ),
        "descriptor_manifold_precision_noninferior": (
            baseline_descriptor is not None
            and treatment_descriptor is not None
            and treatment_descriptor
            >= baseline_descriptor - float(gate["descriptor_precision_noninferiority_margin"])
        ),
    }


def _passes(checks: Mapping[str, Any]) -> bool:
    ignored = {
        "effective_component_count_retained_ratio",
        "effective_topology_count_ratio",
        "unique_exact_l1_retained_ratio",
    }
    return all(value is True for key, value in checks.items() if key not in ignored)


def run_ugi_morphology_diversity_comparison(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    """Run a matched program-law comparison with identical chemistry guidance."""

    config = read_json_object(
        config_path,
        error=UgiMorphologyDiversityComparisonError,
        label="Ugi morphology-diversity comparison config",
    )
    runtime = _runtime(config, profile=profile, device=device)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise UgiMorphologyDiversityComparisonError(
            f"comparison output directory is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        label: resolve_pin(pin, repo, label=label)
        for label, pin in sorted(config["inputs"].items())
    }
    count = int(runtime["program_count"])
    baseline_programs, _, baseline_targets = _load_draw(
        inputs["baseline_program_draw"], count=count
    )
    treatment_programs, _, treatment_targets = _load_draw(
        inputs["treatment_program_draw"], count=count
    )
    baseline_draw = read_json_object(
        inputs["baseline_program_draw"],
        error=UgiMorphologyDiversityComparisonError,
        label="baseline morphology program draw",
    )
    treatment_draw = read_json_object(
        inputs["treatment_program_draw"],
        error=UgiMorphologyDiversityComparisonError,
        label="treatment morphology program draw",
    )
    baseline_support_sha256 = baseline_draw.get("prior_audit", {}).get("joint_support_sha256")
    treatment_support_sha256 = treatment_draw.get("prior_audit", {}).get("joint_support_sha256")
    local_support = LocalChemistrySupport.from_mapping(
        read_json_object(
            inputs["role_morphology_policy"],
            error=UgiMorphologyDiversityComparisonError,
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
    support_audits = {
        "baseline": _semantic_support_audit(
            baseline_programs,
            baseline_targets,
            topology_policy=topology_policy,
            ester_policy=ester_policy,
            local_chemistry_support=local_support,
        ),
        "treatment": _semantic_support_audit(
            treatment_programs,
            treatment_targets,
            topology_policy=topology_policy,
            ester_policy=ester_policy,
            local_chemistry_support=local_support,
        ),
    }
    local_prior = UgiRoleChemistryPrior.from_training_data(
        assignments_path=inputs["ugi_assignments"],
        semantic_atoms_path=inputs["semantic_atoms"],
        semantic_bonds_path=inputs["semantic_bonds"],
        reaction_id="ugi_3cr_agile",
        expected_assignments_sha256=str(config["inputs"]["ugi_assignments"]["sha256"]),
        expected_semantic_atoms_sha256=str(config["inputs"]["semantic_atoms"]["sha256"]),
        expected_semantic_bonds_sha256=str(config["inputs"]["semantic_bonds"]["sha256"]),
    )
    semantic_policy = UgiMogSemanticGuidancePolicy.from_mapping(
        config["semantic_guidance"]
    ).bind_local_chemistry(local_prior)
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
        "program_count": count,
        "sample_steps": int(runtime["sample_steps"]),
        "terminal_decoder_seed": int(runtime["terminal_decoder_seed"]),
        "terminal_temperature": float(runtime.get("terminal_temperature", 1.0)),
    }
    arm_specs = {
        "baseline": {
            "programs": baseline_programs,
            "targets": baseline_targets,
            "draw": inputs["baseline_program_draw"],
            "method_id": "forge_seed0_local_chemistry_original_joint_program_law",
        },
        "treatment": {
            "programs": treatment_programs,
            "targets": treatment_targets,
            "draw": inputs["treatment_program_draw"],
            "method_id": "forge_seed0_local_chemistry_entropy_calibrated_joint_program_law",
        },
    }
    methods: dict[str, Any] = {}
    for label, arm in arm_specs.items():
        rows, sampling, sampling_path = _load_or_sample_current(
            repo=repo,
            output_dir=output_dir / label / "sampling",
            inputs=current_inputs,
            runtime=sampling_runtime,
            config_sha256=sha256_file(config_path),
            device=device,
            arm_id=str(config["base_checkpoint"]["arm_id"]),
            checkpoint_step=int(config["base_checkpoint"]["step"]),
            programs=arm["programs"],
            program_draw_sha256=sha256_file(arm["draw"]),
            ugi_topology_policy=topology_policy,
            local_chemistry_support=local_support,
            reaction_core_saturation_policy=core_policy,
            ugi_ester_chemotype_policy=ester_policy,
            ugi_all_role_semantic_targets=arm["targets"],
            ugi_all_role_semantic_target_sha256=sha256_file(arm["draw"]),
            ugi_mog_semantic_guidance_policy=semantic_policy,
        )
        assessment = _load_or_assess_current(
            rows=rows,
            sampling_path=sampling_path,
            output_dir=output_dir / label / "assessment",
            inputs=current_inputs,
            repo=repo,
            method_id=str(arm["method_id"]),
        )
        methods[label] = {
            "metrics": dict(assessment["assessment"]["metrics"]),
            "topology_diversity": summarize_topology_diversity(rows),
            "sampling": artifact_record(sampling_path),
            "sampling_summary": sampling,
            "assessment": artifact_record(output_dir / label / "assessment" / "result_index.json"),
            "program_draw": pin_record(arm["draw"], repo),
            "program_support_abstentions": _program_support_abstentions(sampling),
        }
    baseline = methods["baseline"]["metrics"]
    treatment = methods["treatment"]["metrics"]
    metric_deltas = _metric_deltas(treatment, baseline)
    baseline_topology = methods["baseline"]["topology_diversity"]
    treatment_topology = methods["treatment"]["topology_diversity"]
    topology_deltas = {
        key: float(treatment_topology[key]) - float(baseline_topology[key])
        for key in (
            "effective_complete_topology_count",
            "unique_complete_topology_signatures",
            "unique_complete_topology_signatures_per_attempt",
        )
    }
    preflight_gate = config["preflight_gate"]
    topology_ratio = _ratio(
        float(treatment_topology["effective_complete_topology_count"]),
        float(baseline_topology["effective_complete_topology_count"]),
    )
    preflight_checks = {
        "baseline_valid_fraction_sufficient": float(baseline["valid_fraction_per_attempt"])
        >= float(preflight_gate["minimum_valid_fraction"]),
        "baseline_exact_l1_sufficient": float(baseline["exact_l1_yield_per_attempt"])
        >= float(preflight_gate["minimum_exact_l1_yield"]),
        "treatment_valid_fraction_sufficient": float(treatment["valid_fraction_per_attempt"])
        >= float(preflight_gate["minimum_valid_fraction"]),
        "treatment_exact_l1_sufficient": float(treatment["exact_l1_yield_per_attempt"])
        >= float(preflight_gate["minimum_exact_l1_yield"]),
        "effective_topology_count_improved": topology_ratio is not None
        and topology_ratio >= float(preflight_gate["minimum_effective_topology_count_ratio"]),
        "program_support_abstentions_zero": all(
            method["program_support_abstentions"] == 0 for method in methods.values()
        ),
    }
    promotion_checks = _promotion_checks(
        baseline=baseline,
        treatment=treatment,
        baseline_topology=baseline_topology,
        treatment_topology=treatment_topology,
        gate=config["full_promotion_gate"],
    )
    structural_checks = {
        "attempt_denominator_matched": all(
            int(method["sampling_summary"]["samples"]) == count for method in methods.values()
        ),
        "candidate_selection_absent": True,
        "component_identity_conditioning_absent": True,
        "local_chemistry_policy_identical": True,
        "method_blind_assessment_shared": True,
        "no_repairs_or_retries": all(
            not method["sampling_summary"].get("repairs") for method in methods.values()
        ),
        "program_laws_different": sha256_file(inputs["baseline_program_draw"])
        != sha256_file(inputs["treatment_program_draw"]),
        "program_support_identical": isinstance(baseline_support_sha256, str)
        and baseline_support_sha256 == treatment_support_sha256,
        "semantic_target_support_complete": all(
            audit["all_pairs_supported"] for audit in support_audits.values()
        ),
        "training_route_or_oracle_calls_zero": True,
    }
    status = (
        "complete"
        if all(structural_checks.values())
        and (profile == "smoke" or all(preflight_checks.values()))
        else "fail"
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "profile": profile,
        "programs_per_method": count,
        "methods": methods,
        "treatment_minus_baseline": metric_deltas,
        "topology_diversity_deltas": topology_deltas,
        "preflight_gate": dict(preflight_gate),
        "preflight_checks": preflight_checks,
        "full_promotion_gate": dict(config["full_promotion_gate"]),
        "full_promotion_checks": promotion_checks,
        "promotion_decision": (
            "diagnostic_only"
            if profile == "smoke"
            else (
                "eligible_for_full_comparison"
                if profile == "h100_preflight" and status == "complete"
                else (
                    "eligible_for_blinded_visual_review"
                    if profile == "full" and status == "complete" and _passes(promotion_checks)
                    else "do_not_promote"
                )
            )
        ),
        "support_audits": support_audits,
        "program_law_audit": {
            "baseline_support_sha256": baseline_support_sha256,
            "treatment_support_sha256": treatment_support_sha256,
            "support_identical": baseline_support_sha256 == treatment_support_sha256,
            "treatment_temperature_selection": treatment_draw.get("temperature_selection"),
        },
        "structural_checks": structural_checks,
        "local_chemistry_reference": semantic_policy.local_chemistry_audit(),
        "implementation": pin_record(Path(__file__), repo),
        "topology_diversity_implementation": pin_record(
            repo / "forge/model/ugi_topology_diversity.py", repo
        ),
        "inputs": {
            "config": pin_record(config_path, repo),
            **{label: pin_record(path, repo) for label, path in sorted(inputs.items())},
        },
        "policy": dict(config["policy"]),
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    if status != "complete":
        raise UgiMorphologyDiversityComparisonError(
            f"morphology-diversity comparison failed: preflight={preflight_checks}, "
            f"structural={structural_checks}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("smoke", "h100_preflight", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_ugi_morphology_diversity_comparison(
        args.config.resolve(),
        args.repo.resolve(),
        args.output_dir.resolve(),
        profile=args.profile,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiMorphologyDiversityComparisonError",
    "run_ugi_morphology_diversity_comparison",
]
