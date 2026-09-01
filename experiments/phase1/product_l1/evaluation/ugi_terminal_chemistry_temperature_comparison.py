"""Compare terminal Ugi chemistry interventions on measured Ugi structure support."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
    RECOVERY_INPUT_LABELS,
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
from forge.model.synthesis_program_sampling import (
    LEARNED_TOPOLOGY_THEN_CHEMISTRY_FACTORIZATION,
    UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
    UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
)
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy

CONFIG_SCHEMA = "forge.ugi_terminal_chemistry_temperature_comparison_config.v1"
RESULT_SCHEMA = "forge.ugi_terminal_chemistry_temperature_comparison_result.v1"
FLOW_CONFIG_SCHEMA = "forge.ugi_topology_conditioned_chemistry_flow_comparison_config.v1"
FLOW_RESULT_SCHEMA = "forge.ugi_topology_conditioned_chemistry_flow_comparison_result.v1"
FACTORIZATION_CONFIG_SCHEMA = "forge.ugi_learned_topology_then_chemistry_comparison_config.v1"
FACTORIZATION_RESULT_SCHEMA = "forge.ugi_learned_topology_then_chemistry_comparison_result.v1"
INPUT_LABELS = {"base_comparison_config", "ugi_matched_realism_config"}
FLOW_INPUT_LABELS = {
    *INPUT_LABELS,
    "specialist_checkpoint",
    "specialist_result",
}


class UgiTerminalChemistryTemperatureComparisonError(ValueError):
    """The terminal-chemistry calibration contract changed."""


def _runtime(config: Mapping[str, Any], *, profile: str, device: str) -> Mapping[str, Any]:
    if (
        config.get("schema_version")
        not in {CONFIG_SCHEMA, FLOW_CONFIG_SCHEMA, FACTORIZATION_CONFIG_SCHEMA}
        or not isinstance(config.get("inputs"), Mapping)
        or set(config["inputs"])
        != (
            FLOW_INPUT_LABELS
            if config.get("schema_version") in {FLOW_CONFIG_SCHEMA, FACTORIZATION_CONFIG_SCHEMA}
            else INPUT_LABELS
        )
        or not isinstance(config.get("profiles"), Mapping)
        or not isinstance(config["profiles"].get(profile), Mapping)
    ):
        raise UgiTerminalChemistryTemperatureComparisonError(
            "unsupported terminal-chemistry calibration config"
        )
    runtime = config["profiles"][profile]
    is_flow_comparison = config["schema_version"] in {
        FLOW_CONFIG_SCHEMA,
        FACTORIZATION_CONFIG_SCHEMA,
    }
    is_factorization_comparison = config["schema_version"] == FACTORIZATION_CONFIG_SCHEMA
    temperatures = runtime.get("temperatures")
    chemistry_flow_steps = runtime.get("chemistry_flow_steps")
    if (
        runtime.get("device") != device
        or int(runtime.get("program_count", 0)) < 4
        or int(runtime.get("sample_steps", 0)) < 2
        or int(runtime.get("batch_size", 0)) < 1
        or int(runtime.get("layout_seed", -1)) < 0
        or int(runtime.get("flow_seed", -1)) < 0
        or (
            not is_flow_comparison
            and (
                int(runtime.get("terminal_seed", -1)) < 0
                or not isinstance(temperatures, list)
                or len(temperatures) < 2
                or any(float(value) <= 0 for value in temperatures)
                or sorted(set(float(value) for value in temperatures))
                != [float(value) for value in temperatures]
            )
        )
        or (
            is_flow_comparison
            and (
                int(runtime.get("chemistry_flow_seed", -1)) < 0
                or not isinstance(chemistry_flow_steps, list)
                or len(chemistry_flow_steps) < 1
                or (is_factorization_comparison and len(chemistry_flow_steps) != 1)
                or any(int(value) < 2 for value in chemistry_flow_steps)
                or sorted(set(int(value) for value in chemistry_flow_steps))
                != [int(value) for value in chemistry_flow_steps]
            )
        )
    ):
        raise UgiTerminalChemistryTemperatureComparisonError(
            "terminal-chemistry calibration runtime changed"
        )
    expected_policy = {
        "candidate_selection": False,
        "component_identity_conditioning": False,
        "heldout_reference_access": False,
        "method_blind_assessment": True,
        "paired_program_order": True,
        "paired_flow_stream": True,
        "repairs_or_retries": False,
        "route_or_oracle_calls": 0,
        "training_calls": 0,
    }
    expected_policy[
        "paired_chemistry_source_stream" if is_flow_comparison else "paired_terminal_uniform_stream"
    ] = True
    if is_factorization_comparison:
        expected_policy["paired_topology_proposal_stream"] = True
    if config.get("policy") != expected_policy:
        raise UgiTerminalChemistryTemperatureComparisonError(
            "terminal-chemistry scientific policy changed"
        )
    if config.get("selection_gate") != {
        "exact_l1_noninferiority_margin": 0.02,
        "local_support_noninferiority_margin": 0.02,
        "realism_c2st_required_reduction": 0.02,
        "unique_exact_l1_required_increase": 0.10,
        "validity_noninferiority_margin": 0.02,
    }:
        raise UgiTerminalChemistryTemperatureComparisonError(
            "terminal-chemistry selection gate changed"
        )
    return runtime


def _passes_gate(deltas: Mapping[str, float | None], gate: Mapping[str, Any]) -> bool:
    required = {
        "exact_l1_yield_per_attempt",
        "local_support_qualified_exact_l1_yield_per_attempt",
        "realism_c2st_auc",
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
        and float(deltas["realism_c2st_auc"]) <= -float(gate["realism_c2st_required_reduction"])
        and float(deltas["unique_exact_l1_products_per_attempt"])
        >= float(gate["unique_exact_l1_required_increase"])
    )


def run_ugi_terminal_chemistry_temperature_comparison(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    """Compare one prespecified terminal-chemistry intervention on one paired draw."""

    config = read_json_object(
        config_path,
        error=UgiTerminalChemistryTemperatureComparisonError,
        label="terminal-chemistry calibration config",
    )
    runtime = _runtime(config, profile=profile, device=device)
    is_flow_comparison = config["schema_version"] in {
        FLOW_CONFIG_SCHEMA,
        FACTORIZATION_CONFIG_SCHEMA,
    }
    is_factorization_comparison = config["schema_version"] == FACTORIZATION_CONFIG_SCHEMA
    result_schema = (
        FACTORIZATION_RESULT_SCHEMA
        if is_factorization_comparison
        else FLOW_RESULT_SCHEMA if is_flow_comparison else RESULT_SCHEMA
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise UgiTerminalChemistryTemperatureComparisonError(
            f"calibration output directory is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_inputs = {
        label: resolve_pin(pin, repo, label=label)
        for label, pin in sorted(config["inputs"].items())
    }
    base_config = read_json_object(
        direct_inputs["base_comparison_config"],
        error=UgiTerminalChemistryTemperatureComparisonError,
        label="base chemistry-specialist comparison config",
    )
    validate_base_comparison(base_config, profile=profile, device=device)
    if not is_flow_comparison and not RECOVERY_INPUT_LABELS.issubset(base_config["inputs"]):
        raise UgiTerminalChemistryTemperatureComparisonError(
            "terminal-chemistry calibration requires the authenticated specialist checkpoint"
        )
    base_inputs = {
        label: resolve_pin(pin, repo, label=label)
        for label, pin in sorted(base_config["inputs"].items())
    }
    if is_flow_comparison:
        base_inputs["specialist_checkpoint"] = direct_inputs["specialist_checkpoint"]
        base_inputs["specialist_result"] = direct_inputs["specialist_result"]
    specialist_checkpoint = base_inputs["specialist_checkpoint"]
    specialist_result = read_json_object(
        base_inputs["specialist_result"],
        error=UgiTerminalChemistryTemperatureComparisonError,
        label="chemistry specialist result",
    )
    if (
        specialist_result.get("status") != "pass"
        or specialist_result.get("target_program") != "ugi_3cr_agile"
        or specialist_result.get("checkpoint", {}).get("sha256")
        != sha256_file(specialist_checkpoint)
    ):
        raise UgiTerminalChemistryTemperatureComparisonError(
            "chemistry-specialist checkpoint is not admissible"
        )
    realism_config = read_json_object(
        direct_inputs["ugi_matched_realism_config"],
        error=UgiTerminalChemistryTemperatureComparisonError,
        label="Ugi-matched realism config",
    )
    if realism_config.get("reference") != {
        "kind": "source_adjudicated_measured_ugi",
        "evaluation_fold": "calibration",
    }:
        raise UgiTerminalChemistryTemperatureComparisonError(
            "temperature calibration must not access the measured Ugi heldout reference"
        )

    local_support = LocalChemistrySupport.from_mapping(
        read_json_object(
            base_inputs["role_morphology_policy"],
            error=UgiTerminalChemistryTemperatureComparisonError,
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
    programs, program_draw = _sample_supported_programs(
        base_inputs["production_cache"],
        ester_policy,
        count=int(runtime["program_count"]),
        seed=int(runtime["layout_seed"]),
    )
    program_draw_path = output_dir / "paired_program_draw.json"
    if program_draw_path.is_file():
        if (
            read_json_object(
                program_draw_path,
                error=UgiTerminalChemistryTemperatureComparisonError,
                label="persisted paired program draw",
            )
            != program_draw
        ):
            raise UgiTerminalChemistryTemperatureComparisonError(
                "persisted paired program draw changed"
            )
    else:
        write_json(program_draw_path, program_draw)

    current_inputs = {
        "checkpoint_archive": base_inputs["base_checkpoint_archive"],
        "common_ugi_assessment_config": base_inputs["common_ugi_assessment_config"],
        "current_training_result": base_inputs["base_training_result"],
        "lipid_realism_config": direct_inputs["ugi_matched_realism_config"],
        "local_chemistry_config": base_inputs["local_chemistry_config"],
        "production_cache": base_inputs["production_cache"],
        "production_design": base_inputs["base_design"],
        "role_morphology_policy": base_inputs["role_morphology_policy"],
    }
    common_arguments = {
        "repo": repo,
        "inputs": current_inputs,
        "config_sha256": sha256_file(config_path),
        "device": device,
        "arm_id": str(base_config["base_checkpoint"]["arm_id"]),
        "checkpoint_step": int(base_config["base_checkpoint"]["step"]),
        "programs": programs,
        "program_draw_sha256": sha256_file(program_draw_path),
        "specialist_checkpoint": specialist_checkpoint,
        "ugi_topology_policy": topology_policy,
        "local_chemistry_support": local_support,
        "reaction_core_saturation_policy": core_policy,
        "ugi_ester_chemotype_policy": ester_policy,
    }

    methods: dict[str, Any] = {}
    reference_metrics: dict[str, Any] | None = None
    reference_label = "argmax"
    arms: list[dict[str, Any]] = []
    if is_factorization_comparison:
        chemistry_steps = int(runtime["chemistry_flow_steps"][0])
        reference_label = "constructive_topology"
        arms.extend(
            (
                {
                    "label": "constructive_topology",
                    "decode_policy": UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
                    "terminal_seed": None,
                    "temperature": 1.0,
                    "chemistry_flow_steps": chemistry_steps,
                    "chemistry_flow_seed": int(runtime["chemistry_flow_seed"]),
                    "sampling_factorization": "joint",
                },
                {
                    "label": "learned_topology",
                    "decode_policy": UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
                    "terminal_seed": None,
                    "temperature": 1.0,
                    "chemistry_flow_steps": chemistry_steps,
                    "chemistry_flow_seed": int(runtime["chemistry_flow_seed"]),
                    "sampling_factorization": (LEARNED_TOPOLOGY_THEN_CHEMISTRY_FACTORIZATION),
                },
            )
        )
    else:
        arms.append(
            {
                "label": "argmax",
                "decode_policy": UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
                "terminal_seed": None,
                "temperature": 1.0,
                "chemistry_flow_steps": 0,
                "chemistry_flow_seed": None,
                "sampling_factorization": "joint",
            }
        )
    if is_flow_comparison and not is_factorization_comparison:
        arms.extend(
            {
                "label": f"chemistry_flow_{int(value)}",
                "decode_policy": UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
                "terminal_seed": None,
                "temperature": 1.0,
                "chemistry_flow_steps": int(value),
                "chemistry_flow_seed": int(runtime["chemistry_flow_seed"]),
                "sampling_factorization": "joint",
            }
            for value in runtime["chemistry_flow_steps"]
        )
    elif not is_flow_comparison:
        arms.extend(
            {
                "label": f"temperature_{float(value):g}".replace(".", "p"),
                "decode_policy": UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
                "terminal_seed": int(runtime["terminal_seed"]),
                "temperature": float(value),
                "chemistry_flow_steps": 0,
                "chemistry_flow_seed": None,
                "sampling_factorization": "joint",
            }
            for value in runtime["temperatures"]
        )
    for arm in arms:
        label = str(arm["label"])
        sampling_runtime = {
            "batch_size": int(runtime["batch_size"]),
            "current_terminal_decode_policy": str(arm["decode_policy"]),
            "flow_seed": int(runtime["flow_seed"]),
            "program_count": int(runtime["program_count"]),
            "sample_steps": int(runtime["sample_steps"]),
            "sampling_factorization": str(arm["sampling_factorization"]),
            "terminal_decoder_seed": arm["terminal_seed"],
            "terminal_temperature": float(arm["temperature"]),
            "topology_conditioned_chemistry_steps": int(arm["chemistry_flow_steps"]),
            "topology_conditioned_chemistry_seed": arm["chemistry_flow_seed"],
        }
        rows, sampling, sampling_path = _load_or_sample_current(
            output_dir=output_dir / label / "sampling",
            runtime=sampling_runtime,
            **common_arguments,
        )
        assessment = _load_or_assess_current(
            rows=rows,
            sampling_path=sampling_path,
            output_dir=output_dir / label / "assessment",
            inputs=current_inputs,
            repo=repo,
            method_id=f"forge_ugi_contextual_specialist_{label}",
        )
        metrics = dict(assessment["assessment"]["metrics"])
        if label == reference_label:
            reference_metrics = metrics
            deltas = {key: 0.0 if value is not None else None for key, value in metrics.items()}
        else:
            assert reference_metrics is not None
            deltas = _metric_deltas(metrics, reference_metrics)
        methods[label] = {
            "terminal_decode_policy": arm["decode_policy"],
            "terminal_seed": arm["terminal_seed"],
            "temperature": arm["temperature"],
            "topology_conditioned_chemistry_steps": arm["chemistry_flow_steps"],
            "topology_conditioned_chemistry_seed": arm["chemistry_flow_seed"],
            "sampling_factorization": arm["sampling_factorization"],
            "metrics": metrics,
            "minus_reference": deltas,
            "sampling": artifact_record(sampling_path),
            "program_support_abstentions": _program_support_abstentions(sampling),
            "repairs": dict(sampling.get("repairs", {})),
        }
    assert reference_metrics is not None
    passing = [
        value
        for label, value in methods.items()
        if label != reference_label
        and not value["repairs"]
        and value["program_support_abstentions"] == 0
        and _passes_gate(value["minus_reference"], config["selection_gate"])
    ]
    selected = (
        min(
            passing,
            key=lambda value: (
                float(value["metrics"]["realism_c2st_auc"]),
                -float(value["metrics"]["unique_exact_l1_products_per_attempt"]),
                (
                    int(value["topology_conditioned_chemistry_steps"])
                    if is_flow_comparison
                    else float(value["temperature"])
                ),
            ),
        )
        if passing
        else None
    )
    if profile == "smoke":
        selected = None
    result = {
        "schema_version": result_schema,
        "status": "complete",
        "profile": profile,
        "programs_per_method": len(programs),
        "methods": methods,
        "reference_method": reference_label,
        "selected_sampling_factorization": (
            selected["sampling_factorization"]
            if is_factorization_comparison and selected is not None
            else None
        ),
        "selected_temperature": (
            None if is_flow_comparison or selected is None else selected["temperature"]
        ),
        "selected_topology_conditioned_chemistry_steps": (
            selected["topology_conditioned_chemistry_steps"]
            if is_flow_comparison and selected is not None
            else None
        ),
        "selection_decision": (
            "diagnostic_only"
            if profile == "smoke"
            else "freeze_for_one_heldout_confirmation" if selected is not None else "stop"
        ),
        "selection_gate": dict(config["selection_gate"]),
        "program_draw": artifact_record(program_draw_path),
        "specialist_checkpoint": artifact_record(specialist_checkpoint),
        "inputs": {
            "config": pin_record(config_path, repo),
            **{label: pin_record(path, repo) for label, path in sorted(direct_inputs.items())},
        },
        "policy": dict(config["policy"]),
        "candidate_selection": False,
        "calls": {"training": 0, "route": 0, "oracle": 0},
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    return result


def run_ugi_topology_conditioned_chemistry_flow_comparison(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    """Run the fixed-topology chemistry-flow specialization of the shared comparison."""

    config = read_json_object(
        config_path,
        error=UgiTerminalChemistryTemperatureComparisonError,
        label="topology-conditioned chemistry-flow comparison config",
    )
    if config.get("schema_version") != FLOW_CONFIG_SCHEMA:
        raise UgiTerminalChemistryTemperatureComparisonError(
            "topology-conditioned chemistry-flow runner received the wrong config schema"
        )
    return run_ugi_terminal_chemistry_temperature_comparison(
        config_path,
        repo,
        output_dir,
        profile=profile,
        device=device,
        resume=resume,
    )


def run_ugi_learned_topology_then_chemistry_comparison(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    """Compare learned parent/closure topology with the frozen constructive decoder."""

    config = read_json_object(
        config_path,
        error=UgiTerminalChemistryTemperatureComparisonError,
        label="learned topology-then-chemistry comparison config",
    )
    if config.get("schema_version") != FACTORIZATION_CONFIG_SCHEMA:
        raise UgiTerminalChemistryTemperatureComparisonError(
            "learned topology-then-chemistry runner received the wrong config schema"
        )
    return run_ugi_terminal_chemistry_temperature_comparison(
        config_path,
        repo,
        output_dir,
        profile=profile,
        device=device,
        resume=resume,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "h100_preflight", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    result = run_ugi_terminal_chemistry_temperature_comparison(
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
    "FACTORIZATION_CONFIG_SCHEMA",
    "FACTORIZATION_RESULT_SCHEMA",
    "FLOW_CONFIG_SCHEMA",
    "FLOW_RESULT_SCHEMA",
    "RESULT_SCHEMA",
    "UgiTerminalChemistryTemperatureComparisonError",
    "run_ugi_learned_topology_then_chemistry_comparison",
    "run_ugi_topology_conditioned_chemistry_flow_comparison",
    "run_ugi_terminal_chemistry_temperature_comparison",
]
