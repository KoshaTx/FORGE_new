"""Matched terminal-only sweep of the Ugi role-conditioned soft chemistry prior."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
    INPUT_LABELS as BASE_INPUT_LABELS,
)
from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
    RECOVERY_INPUT_LABELS,
    SPECIALIST_RESULT_SCHEMA,
    _load_or_assess_current,
    _load_or_sample_current,
    _metric_deltas,
    _program_support_abstentions,
    _sample_supported_programs,
    _topology_policy,
)
from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
    _validate as validate_base_comparison,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.reaction_core_saturation import ReactionCoreSaturationPolicy
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_role_chemistry_prior import UgiRoleChemistryPrior

CONFIG_SCHEMA = "forge.ugi_role_chemistry_prior_comparison_config.v1"
RESULT_SCHEMA = "forge.ugi_role_chemistry_prior_comparison_result.v1"
INPUT_LABELS = {
    "base_comparison_config",
    "semantic_atoms",
    "semantic_bonds",
    *BASE_INPUT_LABELS,
    *RECOVERY_INPUT_LABELS,
}


class UgiRoleChemistryPriorComparisonError(ValueError):
    """The matched soft-prior comparison contract changed."""


def _runtime(config: Mapping[str, Any], *, profile: str, device: str) -> Mapping[str, Any]:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not isinstance(config.get("inputs"), Mapping)
        or set(config["inputs"]) != INPUT_LABELS
        or not isinstance(config.get("profiles"), Mapping)
        or not isinstance(config["profiles"].get(profile), Mapping)
    ):
        raise UgiRoleChemistryPriorComparisonError("unsupported soft-prior comparison config")
    runtime = config["profiles"][profile]
    strengths = runtime.get("strengths")
    if (
        runtime.get("device") != device
        or int(runtime.get("program_count", 0)) < 4
        or int(runtime.get("sample_steps", 0)) < 2
        or int(runtime.get("batch_size", 0)) < 1
        or int(runtime.get("layout_seed", -1)) < 0
        or int(runtime.get("flow_seed", -1)) < 0
        or not isinstance(strengths, list)
        or len(strengths) < 2
        or float(strengths[0]) != 0.0
        or any(float(value) < 0 for value in strengths)
        or sorted(set(float(value) for value in strengths))
        != [float(value) for value in strengths]
    ):
        raise UgiRoleChemistryPriorComparisonError("soft-prior runtime changed")
    if config.get("policy") != {
        "candidate_selection": False,
        "component_identity_conditioning": False,
        "heldout_training_access": False,
        "method_blind_assessment": True,
        "paired_program_order": True,
        "paired_random_stream": True,
        "repairs_or_retries": False,
        "route_or_oracle_calls": 0,
        "training_calls": 0,
    }:
        raise UgiRoleChemistryPriorComparisonError("soft-prior scientific policy changed")
    if config.get("selection_gate") != {
        "exact_l1_noninferiority_margin": 0.02,
        "local_support_noninferiority_margin": 0.02,
        "realism_c2st_required_reduction": 0.02,
        "mean_pairwise_ecfp4_distance_noninferiority_margin": 0.05,
        "unique_exact_l1_noninferiority_margin": 0.05,
        "validity_noninferiority_margin": 0.02,
    }:
        raise UgiRoleChemistryPriorComparisonError("soft-prior selection gate changed")
    return runtime


def _specialist_checkpoint(inputs: Mapping[str, Path]) -> Path:
    checkpoint = inputs["specialist_checkpoint"]
    result_path = inputs["specialist_result"]
    if result_path != checkpoint.with_name("result.json"):
        raise UgiRoleChemistryPriorComparisonError(
            "specialist result is not adjacent to its checkpoint"
        )
    result = read_json_object(
        result_path,
        error=UgiRoleChemistryPriorComparisonError,
        label="Ugi chemistry-specialist result",
    )
    if (
        result.get("schema_version") != SPECIALIST_RESULT_SCHEMA
        or result.get("status") != "pass"
        or result.get("target_program") != "ugi_3cr_agile"
        or result.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint)
    ):
        raise UgiRoleChemistryPriorComparisonError(
            "chemistry-specialist checkpoint is not admissible"
        )
    return checkpoint


def _passes_gate(
    deltas: Mapping[str, float | None], gate: Mapping[str, Any]
) -> bool:
    required = {
        "exact_l1_yield_per_attempt",
        "local_support_qualified_exact_l1_yield_per_attempt",
        "realism_c2st_auc",
        "mean_pairwise_ecfp4_distance",
        "unique_exact_l1_products_per_attempt",
        "valid_fraction_per_attempt",
    }
    if any(deltas.get(metric) is None for metric in required):
        return False
    return bool(
        float(deltas["exact_l1_yield_per_attempt"])
        >= -float(gate["exact_l1_noninferiority_margin"])
        and float(deltas["local_support_qualified_exact_l1_yield_per_attempt"])
        >= -float(gate["local_support_noninferiority_margin"])
        and float(deltas["valid_fraction_per_attempt"])
        >= -float(gate["validity_noninferiority_margin"])
        and float(deltas["realism_c2st_auc"])
        <= -float(gate["realism_c2st_required_reduction"])
        and float(deltas["mean_pairwise_ecfp4_distance"])
        >= -float(gate["mean_pairwise_ecfp4_distance_noninferiority_margin"])
        and float(deltas["unique_exact_l1_products_per_attempt"])
        >= -float(gate["unique_exact_l1_noninferiority_margin"])
    )


def run_ugi_role_chemistry_prior_comparison(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    """Evaluate nonzero prior strengths against a bit-identical lambda-zero control."""

    config = read_json_object(
        config_path,
        error=UgiRoleChemistryPriorComparisonError,
        label="Ugi role-chemistry prior comparison config",
    )
    runtime = _runtime(config, profile=profile, device=device)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise UgiRoleChemistryPriorComparisonError(
            f"comparison output directory is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_inputs = {
        label: resolve_pin(pin, repo, label=label)
        for label, pin in sorted(config["inputs"].items())
    }
    base_config = read_json_object(
        direct_inputs["base_comparison_config"],
        error=UgiRoleChemistryPriorComparisonError,
        label="base chemistry-specialist comparison config",
    )
    validate_base_comparison(base_config, profile=profile, device=device)
    pinned_base_inputs = {
        label: resolve_pin(pin, repo, label=label)
        for label, pin in sorted(base_config["inputs"].items())
    }
    required_base_inputs = BASE_INPUT_LABELS | RECOVERY_INPUT_LABELS
    if any(
        direct_inputs[label] != pinned_base_inputs[label] for label in required_base_inputs
    ):
        raise UgiRoleChemistryPriorComparisonError(
            "direct soft-prior inputs differ from the pinned base-comparison inputs"
        )
    base_inputs = {label: direct_inputs[label] for label in required_base_inputs}
    if not RECOVERY_INPUT_LABELS.issubset(base_inputs):
        raise UgiRoleChemistryPriorComparisonError(
            "soft-prior sweep requires the pinned completed chemistry specialist"
        )
    checkpoint = _specialist_checkpoint(base_inputs)

    local_support = LocalChemistrySupport.from_mapping(
        read_json_object(
            base_inputs["role_morphology_policy"],
            error=UgiRoleChemistryPriorComparisonError,
            label="role-local chemistry support",
        )
    )
    topology_policy = _topology_policy(base_inputs)
    core_policy = ReactionCoreSaturationPolicy.from_qualified_registry(
        base_inputs["qualified_reactions"],
        reaction_id="ugi_3cr_agile",
        expected_sha256=sha256_file(base_inputs["qualified_reactions"]),
    )
    ester_policy = UgiEsterChemotypePolicy.from_qualified_registry(
        base_inputs["qualified_reactions"],
        training_assignments_path=base_inputs["ugi_assignments"],
        reaction_id="ugi_3cr_agile",
        expected_sha256=sha256_file(base_inputs["qualified_reactions"]),
        expected_training_assignments_sha256=sha256_file(base_inputs["ugi_assignments"]),
    )
    prior = UgiRoleChemistryPrior.from_training_data(
        assignments_path=base_inputs["ugi_assignments"],
        semantic_atoms_path=direct_inputs["semantic_atoms"],
        semantic_bonds_path=direct_inputs["semantic_bonds"],
        expected_assignments_sha256=sha256_file(base_inputs["ugi_assignments"]),
        expected_semantic_atoms_sha256=sha256_file(direct_inputs["semantic_atoms"]),
        expected_semantic_bonds_sha256=sha256_file(direct_inputs["semantic_bonds"]),
    )
    programs, program_draw = _sample_supported_programs(
        base_inputs["production_cache"],
        ester_policy,
        count=int(runtime["program_count"]),
        seed=int(runtime["layout_seed"]),
    )
    program_draw_path = output_dir / "paired_program_draw.json"
    if program_draw_path.is_file():
        if read_json_object(
            program_draw_path,
            error=UgiRoleChemistryPriorComparisonError,
            label="persisted paired program draw",
        ) != program_draw:
            raise UgiRoleChemistryPriorComparisonError("persisted paired program draw changed")
    else:
        write_json(program_draw_path, program_draw)

    current_inputs = {
        "checkpoint_archive": base_inputs["base_checkpoint_archive"],
        "common_ugi_assessment_config": base_inputs["common_ugi_assessment_config"],
        "current_training_result": base_inputs["base_training_result"],
        "lipid_realism_config": base_inputs["lipid_realism_config"],
        "local_chemistry_config": base_inputs["local_chemistry_config"],
        "production_cache": base_inputs["production_cache"],
        "production_design": base_inputs["base_design"],
        "role_morphology_policy": base_inputs["role_morphology_policy"],
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
        "arm_id": str(base_config["base_checkpoint"]["arm_id"]),
        "checkpoint_step": int(base_config["base_checkpoint"]["step"]),
        "programs": programs,
        "program_draw_sha256": sha256_file(program_draw_path),
        "specialist_checkpoint": checkpoint,
        "ugi_topology_policy": topology_policy,
        "local_chemistry_support": local_support,
        "reaction_core_saturation_policy": core_policy,
        "ugi_ester_chemotype_policy": ester_policy,
        "ugi_role_chemistry_prior": prior,
    }
    methods: dict[str, Any] = {}
    reference_metrics: dict[str, Any] | None = None
    for strength in (float(value) for value in runtime["strengths"]):
        label = f"lambda_{strength:g}".replace(".", "p")
        rows, sampling, sampling_path = _load_or_sample_current(
            output_dir=output_dir / label / "sampling",
            ugi_role_chemistry_prior_strength=strength,
            **common_arguments,
        )
        assessment = _load_or_assess_current(
            rows=rows,
            sampling_path=sampling_path,
            output_dir=output_dir / label / "assessment",
            inputs=current_inputs,
            repo=repo,
            method_id=f"forge_ugi_chemistry_specialist_role_prior_{label}",
        )
        metrics = dict(assessment["assessment"]["metrics"])
        if strength == 0:
            reference_metrics = metrics
            deltas = {key: 0.0 if value is not None else None for key, value in metrics.items()}
        else:
            assert reference_metrics is not None
            deltas = _metric_deltas(metrics, reference_metrics)
        methods[label] = {
            "strength": strength,
            "metrics": metrics,
            "minus_lambda_zero": deltas,
            "sampling": artifact_record(sampling_path),
            "program_support_abstentions": _program_support_abstentions(sampling),
        }
    assert reference_metrics is not None
    passing = [
        value
        for value in methods.values()
        if value["strength"] > 0 and _passes_gate(value["minus_lambda_zero"], config["selection_gate"])
    ]
    selected = (
        min(
            passing,
            key=lambda value: (
                float(value["metrics"]["realism_c2st_auc"]),
                float(value["strength"]),
            ),
        )["strength"]
        if passing
        else None
    )
    if profile != "h100_preflight":
        selected = None
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "profile": profile,
        "programs_per_method": len(programs),
        "methods": methods,
        "selected_strength": selected,
        "selection_decision": (
            "diagnostic_only"
            if profile != "h100_preflight"
            else "promote_to_fresh_production"
            if selected is not None
            else "stop"
        ),
        "selection_gate": dict(config["selection_gate"]),
        "soft_prior": prior.to_mapping(),
        "lambda_zero_identity_contract": (
            "same checkpoint, programs, flow seed, topology, support masks and decoder; "
            "strength zero skips every additive logit term"
        ),
        "inputs": {
            "config": pin_record(config_path, repo),
            **{
                label: pin_record(path, repo) for label, path in sorted(direct_inputs.items())
            },
        },
        "policy": dict(config["policy"]),
        "candidate_selection": False,
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
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
    result = run_ugi_role_chemistry_prior_comparison(
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
