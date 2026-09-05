"""Apply a precommitted quantitative and blinded-visual Ugi development protocol."""

from __future__ import annotations

import argparse
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.phase1.multireaction.ugi_development_realism_comparison import (
    PANEL_CONFIG_SCHEMA,
    _validate_metric_panel,
    run_ugi_development_realism_comparison,
)
from experiments.phase1.multireaction.ugi_development_visual_review import (
    CONFIG_SCHEMA as VISUAL_CONFIG_SCHEMA,
)
from experiments.phase1.multireaction.ugi_development_visual_review import (
    SUPPORTED_CRITERIA,
    run_ugi_development_visual_review,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import read_json_object, write_json
from forge.model.ugi_mog_semantic_guidance import UgiMogSemanticGuidancePolicy

CONFIG_SCHEMA = "forge.ugi_development_adjudication_protocol.v1"
RESULT_SCHEMA = "forge.ugi_development_adjudication_protocol_result.v1"
COMPARISON_SCHEMA = "forge.ugi_all_role_semantic_program_comparison_result.v1"
EXPECTED_ATTEMPTS = 256

EXPECTED_GATE = {
    "effective_component_count_minimum_retained_ratio": 0.95,
    "exact_l1_absolute_minimum": 0.95,
    "exact_l1_noninferiority_margin": 0.01,
    "require_uniform_role_panel_improvement": True,
    "require_zero_program_support_abstentions": True,
    "unique_exact_l1_minimum_retained_ratio": 0.95,
    "validity_absolute_minimum": 0.95,
}
EXPECTED_VISUAL_RULE = {
    "maximum_treatment_pathology_pairs": 2,
    "minimum_assessable_pairs_per_criterion": 20,
    "minimum_net_treatment_preferences": 4,
    "require_treatment_pathologies_no_more_than_baseline": True,
    "require_treatment_preference_majority": True,
}
EXPECTED_POLICY = {
    "candidate_selection": False,
    "component_identity_conditioning": False,
    "include_invalid_or_failed_attempts_in_visual_review": True,
    "indices_selected_before_generation": True,
    "repair_or_retry": False,
    "route_or_oracle_calls": 0,
    "training_calls": 0,
}


class UgiDevelopmentAdjudicationProtocolError(ValueError):
    """The experiment or adjudication package violated its frozen contract."""


def _validate_protocol(value: Any) -> dict[str, Any]:
    expected = {
        "schema_version",
        "scientific_question",
        "inputs",
        "arms",
        "baseline_arm",
        "treatment_arm",
        "expected_attempts",
        "metric_panel",
        "promotion_gate",
        "visual_review",
        "policy",
        "nonclaims",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise UgiDevelopmentAdjudicationProtocolError("development protocol fields changed")
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise UgiDevelopmentAdjudicationProtocolError("development protocol schema changed")
    if not isinstance(value["inputs"], Mapping) or set(value["inputs"]) != {
        "comparison_config",
        "development_realism_config",
        "experiment_descriptor",
    }:
        raise UgiDevelopmentAdjudicationProtocolError("development protocol inputs changed")
    arms = value["arms"]
    baseline = value["baseline_arm"]
    treatment = value["treatment_arm"]
    if (
        not isinstance(arms, Mapping)
        or len(arms) != 2
        or baseline == treatment
        or {baseline, treatment} != set(arms)
    ):
        raise UgiDevelopmentAdjudicationProtocolError("development protocol arms changed")
    for arm_id, arm in arms.items():
        if not isinstance(arm, Mapping) or set(arm) != {
            "archive_member",
            "comparison_method_key",
            "method_id",
            "seed",
        }:
            raise UgiDevelopmentAdjudicationProtocolError(
                f"development protocol arm fields changed for {arm_id}"
            )
    if {arm["comparison_method_key"] for arm in arms.values()} != {
        "amine_semantic",
        "all_role_semantic",
    }:
        raise UgiDevelopmentAdjudicationProtocolError("comparison method keys changed")
    if value["expected_attempts"] != EXPECTED_ATTEMPTS:
        raise UgiDevelopmentAdjudicationProtocolError("preflight attempt count changed")
    _validate_metric_panel(value["metric_panel"])
    if value["promotion_gate"] != EXPECTED_GATE:
        raise UgiDevelopmentAdjudicationProtocolError("development promotion gate changed")
    visual = value["visual_review"]
    if not isinstance(visual, Mapping) or set(visual) != {
        "attempt_indices",
        "selection_seed",
        "blinding_seed",
        "criteria",
        "decision_rule",
    }:
        raise UgiDevelopmentAdjudicationProtocolError("visual-review protocol fields changed")
    indices = visual["attempt_indices"]
    if (
        not isinstance(indices, list)
        or len(indices) != 24
        or indices != sorted(set(indices))
        or indices
        != sorted(
            random.Random(int(visual["selection_seed"])).sample(
                range(EXPECTED_ATTEMPTS), len(indices)
            )
        )
    ):
        raise UgiDevelopmentAdjudicationProtocolError(
            "visual-review indices do not match the frozen output-blind draw"
        )
    criteria = visual["criteria"]
    if not isinstance(criteria, list) or tuple(criteria) not in SUPPORTED_CRITERIA:
        raise UgiDevelopmentAdjudicationProtocolError("visual-review criteria changed")
    if visual["decision_rule"] != EXPECTED_VISUAL_RULE:
        raise UgiDevelopmentAdjudicationProtocolError("visual-review gate changed")
    if value["policy"] != EXPECTED_POLICY:
        raise UgiDevelopmentAdjudicationProtocolError("development policy changed")
    return dict(value)


def _config_pin(path: Path, repo: Path) -> dict[str, str]:
    record = pin_record(path, repo)
    return {"path": str(record["path"]), "sha256": str(record["sha256"])}


def _ratio(numerator: float, denominator: float, *, label: str) -> float:
    if denominator <= 0:
        raise UgiDevelopmentAdjudicationProtocolError(f"baseline {label} is not positive")
    return numerator / denominator


def _validate_bound_inputs(
    protocol: Mapping[str, Any], inputs: Mapping[str, Path]
) -> UgiMogSemanticGuidancePolicy:
    comparison_config = read_json_object(
        inputs["comparison_config"],
        error=UgiDevelopmentAdjudicationProtocolError,
        label="atom-local comparison config",
    )
    baseline = comparison_config.get("baseline_semantic_guidance")
    treatment = comparison_config.get("semantic_guidance")
    if not isinstance(baseline, Mapping) or not isinstance(treatment, Mapping):
        raise UgiDevelopmentAdjudicationProtocolError(
            "comparison does not define paired semantic-guidance policies"
        )
    baseline_policy = UgiMogSemanticGuidancePolicy.from_mapping(baseline)
    treatment_policy = UgiMogSemanticGuidancePolicy.from_mapping(treatment)
    if (
        "local_chemistry_rank_weight" in baseline
        or "local_chemistry_bond_rank_weight" in baseline
        or baseline_policy.uses_local_chemistry
        or not treatment_policy.uses_local_chemistry
        or comparison_config.get("profiles", {}).get("h100_preflight", {}).get("program_count")
        != EXPECTED_ATTEMPTS
    ):
        raise UgiDevelopmentAdjudicationProtocolError(
            "comparison is not the frozen local-chemistry intervention"
        )
    descriptor = read_json_object(
        inputs["experiment_descriptor"],
        error=UgiDevelopmentAdjudicationProtocolError,
        label="atom-local experiment descriptor",
    )
    stages = descriptor.get("stages")
    if (
        not isinstance(descriptor.get("experiment_id"), str)
        or not descriptor["experiment_id"].startswith("phase1-ugi-")
        or not isinstance(stages, list)
        or len(stages) != 1
        or stages[0].get("id") != "preflight"
        or stages[0].get("config") != protocol["inputs"]["comparison_config"]
        or stages[0].get("resources", {}).get("gpu_type") != "H100!"
        or descriptor.get("metadata", {}).get("preflight_programs_per_method") != EXPECTED_ATTEMPTS
        or descriptor.get("metadata", {}).get("component_vocabulary") is not False
        or descriptor.get("metadata", {}).get("candidate_selection") is not False
        or descriptor.get("metadata", {}).get("training_calls") != 0
        or descriptor.get("metadata", {}).get("repair_or_retry_calls") != 0
        or descriptor.get("metadata", {}).get("full_run_requires_separate_authorization")
        is not True
    ):
        raise UgiDevelopmentAdjudicationProtocolError("experiment descriptor changed")
    return treatment_policy


def _validate_comparison(
    value: Any,
    protocol: Mapping[str, Any],
    *,
    expected_policy: UgiMogSemanticGuidancePolicy,
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != COMPARISON_SCHEMA
        or value.get("status") != "complete"
        or value.get("profile") != "h100_preflight"
        or value.get("programs_per_method") != EXPECTED_ATTEMPTS
        or set(value.get("methods", {})) != {"amine_semantic", "all_role_semantic"}
        or value.get("candidate_selection") is not False
        or value.get("calls") != {"training": 0, "route": 0, "oracle": 0}
    ):
        raise UgiDevelopmentAdjudicationProtocolError("preflight comparison is inadmissible")
    if not all(value.get("structural_checks", {}).values()) or not all(
        value.get("preflight_checks", {}).values()
    ):
        raise UgiDevelopmentAdjudicationProtocolError("preflight sampler gates failed")
    policy = value.get("policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("candidate_selection") is not False
        or policy.get("component_identity_conditioning") is not False
        or policy.get("repairs_or_retries") is not False
        or policy.get("training_calls") != 0
        or policy.get("route_or_oracle_calls") != 0
    ):
        raise UgiDevelopmentAdjudicationProtocolError("comparison policy changed")
    treatment_key = str(protocol["arms"][protocol["treatment_arm"]]["comparison_method_key"])
    baseline_key = str(protocol["arms"][protocol["baseline_arm"]]["comparison_method_key"])
    treatment = value["methods"][treatment_key]["sampling_summary"]
    baseline = value["methods"][baseline_key]["sampling_summary"]
    required = (
        treatment.get("ugi_all_role_semantic_joint_support_applied") is True
        and treatment.get("ugi_mog_semantic_guidance_applied") is True
        and treatment.get("ugi_local_atom_chemistry_mog_guidance_applied")
        is expected_policy.uses_local_chemistry_atoms
        and treatment.get("ugi_local_bond_chemistry_mog_guidance_applied")
        is expected_policy.uses_local_chemistry_bonds
        and treatment.get("repairs") == {}
        and baseline.get("ugi_all_role_semantic_joint_support_applied") is True
        and baseline.get("ugi_mog_semantic_guidance_applied") is True
        and baseline.get("ugi_local_atom_chemistry_mog_guidance_applied") is False
        and baseline.get("ugi_local_bond_chemistry_mog_guidance_applied") is False
        and baseline.get("repairs") == {}
    )
    if not required:
        raise UgiDevelopmentAdjudicationProtocolError(
            "comparison did not preserve the configured local-chemistry contrast"
        )
    expected_mapping = expected_policy.to_mapping()
    if (
        value.get("semantic_guidance") != expected_mapping
        or treatment.get("ugi_mog_semantic_guidance_policy") != expected_mapping
    ):
        raise UgiDevelopmentAdjudicationProtocolError(
            "comparison did not preserve the configured semantic-guidance policy"
        )
    expected_atom_total_variation_radius = (
        expected_policy.local_chemistry_atom_total_variation_radius
    )
    reported_radius = value.get("semantic_guidance", {}).get(
        "local_chemistry_atom_total_variation_radius"
    )
    sampled_radius = treatment.get("ugi_local_atom_chemistry_total_variation_radius")
    if reported_radius != expected_atom_total_variation_radius or sampled_radius != (
        expected_atom_total_variation_radius
    ):
        raise UgiDevelopmentAdjudicationProtocolError(
            "comparison did not preserve the configured atom-guidance trust region"
        )
    return dict(value)


def _panel_config(
    protocol: Mapping[str, Any],
    *,
    realism_config: Path,
    comparison_archive: Path,
    repo: Path,
) -> dict[str, Any]:
    arms = {
        arm_id: {
            "archive_member": arm["archive_member"],
            "method_id": arm["method_id"],
            "seed": arm["seed"],
        }
        for arm_id, arm in protocol["arms"].items()
    }
    return {
        "schema_version": PANEL_CONFIG_SCHEMA,
        "scientific_question": protocol["scientific_question"],
        "inputs": {
            "development_realism_config": _config_pin(realism_config, repo),
            "comparison_archive": _config_pin(comparison_archive, repo),
        },
        "arms": arms,
        "baseline_arm": protocol["baseline_arm"],
        "expected_attempts": protocol["expected_attempts"],
        "metric_panel": protocol["metric_panel"],
        "nonclaims": protocol["nonclaims"],
    }


def _visual_config(
    protocol: Mapping[str, Any],
    *,
    comparison_archive: Path,
    panel_result: Path,
    repo: Path,
) -> dict[str, Any]:
    visual = protocol["visual_review"]
    return {
        "schema_version": VISUAL_CONFIG_SCHEMA,
        "scientific_question": protocol["scientific_question"],
        "inputs": {
            "comparison_archive": _config_pin(comparison_archive, repo),
            "realism_panel_result": _config_pin(panel_result, repo),
        },
        "arms": {
            arm_id: {
                "archive_member": arm["archive_member"],
                "method_id": arm["method_id"],
                "seed": arm["seed"],
            }
            for arm_id, arm in protocol["arms"].items()
        },
        "baseline_arm": protocol["baseline_arm"],
        "treatment_arm": protocol["treatment_arm"],
        "expected_attempts": protocol["expected_attempts"],
        "attempt_indices": visual["attempt_indices"],
        "selection_seed": visual["selection_seed"],
        "blinding_seed": visual["blinding_seed"],
        "criteria": visual["criteria"],
        "decision_rule": visual["decision_rule"],
        "policy": {
            "candidate_selection": False,
            "indices_selected_by_output_blind_rng_before_review": True,
            "include_invalid_or_failed_attempts": True,
            "independent_left_right_blinding_per_pair": True,
            "repair_or_retry": False,
            "render_complete_molecule_only": True,
        },
        "nonclaims": protocol["nonclaims"],
    }


def run_ugi_development_adjudication_protocol(
    protocol_path: Path,
    repo: Path,
    comparison_result_path: Path,
    comparison_archive_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Adjudicate one paid preflight without changing its outputs or selecting molecules."""

    protocol = _validate_protocol(
        read_json_object(
            protocol_path,
            error=UgiDevelopmentAdjudicationProtocolError,
            label="Ugi development adjudication protocol",
        )
    )
    inputs = {
        label: resolve_pin(pin, repo, label=label) for label, pin in protocol["inputs"].items()
    }
    expected_policy = _validate_bound_inputs(protocol, inputs)
    comparison = _validate_comparison(
        read_json_object(
            comparison_result_path,
            error=UgiDevelopmentAdjudicationProtocolError,
            label="atom-local preflight comparison",
        ),
        protocol,
        expected_policy=expected_policy,
    )
    comparison_result_path = comparison_result_path.resolve()
    comparison_archive_path = comparison_archive_path.resolve()
    comparison_result_path.relative_to(repo.resolve())
    comparison_archive_path.relative_to(repo.resolve())
    if output_dir.exists() and any(output_dir.iterdir()):
        raise UgiDevelopmentAdjudicationProtocolError(
            f"development adjudication output is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_config_path = output_dir / "realism_panel_config.json"
    write_json(
        panel_config_path,
        _panel_config(
            protocol,
            realism_config=inputs["development_realism_config"],
            comparison_archive=comparison_archive_path,
            repo=repo,
        ),
    )
    panel_dir = output_dir / "realism_panel"
    panel = run_ugi_development_realism_comparison(panel_config_path, repo, panel_dir)

    baseline_arm = str(protocol["baseline_arm"])
    treatment_arm = str(protocol["treatment_arm"])
    baseline_key = str(protocol["arms"][baseline_arm]["comparison_method_key"])
    treatment_key = str(protocol["arms"][treatment_arm]["comparison_method_key"])
    baseline_metrics = comparison["methods"][baseline_key]["metrics"]
    treatment_metrics = comparison["methods"][treatment_key]["metrics"]
    gate = protocol["promotion_gate"]
    baseline_exact = float(baseline_metrics["exact_l1_yield_per_attempt"])
    treatment_exact = float(treatment_metrics["exact_l1_yield_per_attempt"])
    unique_ratio = _ratio(
        float(treatment_metrics["unique_exact_l1_products_per_attempt"]),
        float(baseline_metrics["unique_exact_l1_products_per_attempt"]),
        label="unique exact-L1 yield",
    )
    effective_ratio = _ratio(
        float(treatment_metrics["effective_component_count"]),
        float(baseline_metrics["effective_component_count"]),
        label="effective component count",
    )
    panel_result = panel["panel_comparison"]["comparisons"][treatment_arm]
    checks = {
        "effective_component_count_retained": effective_ratio
        >= float(gate["effective_component_count_minimum_retained_ratio"]),
        "exact_l1_absolute_minimum": treatment_exact >= float(gate["exact_l1_absolute_minimum"]),
        "exact_l1_noninferior": treatment_exact
        >= baseline_exact - float(gate["exact_l1_noninferiority_margin"]),
        "program_support_abstentions_zero": (
            comparison["methods"][baseline_key]["program_support_abstentions"] == 0
            and comparison["methods"][treatment_key]["program_support_abstentions"] == 0
        ),
        "role_panel_uniformly_improved": panel_result["classification"] == "uniformly_improved",
        "unique_exact_l1_retained": unique_ratio
        >= float(gate["unique_exact_l1_minimum_retained_ratio"]),
        "validity_absolute_minimum": float(treatment_metrics["valid_fraction_per_attempt"])
        >= float(gate["validity_absolute_minimum"]),
    }
    quantitative_pass = all(checks.values())
    visual: dict[str, Any] = {"status": "not_reached"}
    if quantitative_pass:
        visual_config_path = output_dir / "visual_review_config.json"
        write_json(
            visual_config_path,
            _visual_config(
                protocol,
                comparison_archive=comparison_archive_path,
                panel_result=panel_dir / "result.json",
                repo=repo,
            ),
        )
        visual_dir = output_dir / "blinded_visual_review"
        packet = run_ugi_development_visual_review(visual_config_path, repo, visual_dir)
        visual = {
            "status": "ready_for_blinded_review",
            "config": artifact_record(visual_config_path),
            "packet_result": artifact_record(visual_dir / "result.json"),
            "pairs": packet["pairs"],
            "blinding_key_status": "sealed_until_review_complete",
        }

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_question": protocol["scientific_question"],
        "inputs": {
            "protocol": pin_record(protocol_path, repo),
            "comparison_config": pin_record(inputs["comparison_config"], repo),
            "experiment_descriptor": pin_record(inputs["experiment_descriptor"], repo),
            "comparison_result": artifact_record(
                comparison_result_path, logical_path="preflight/result.json"
            ),
            "comparison_archive": artifact_record(
                comparison_archive_path, logical_path="preflight/comparison_details.tar"
            ),
        },
        "quantitative_checks": checks,
        "quantitative_decision": (
            "pass_seed0_quantitative_gate" if quantitative_pass else "do_not_promote"
        ),
        "unique_exact_l1_retained_ratio": unique_ratio,
        "effective_component_count_retained_ratio": effective_ratio,
        "realism_panel": artifact_record(panel_dir / "result.json"),
        "visual_review": visual,
        "promotion_decision": (
            "eligible_for_blinded_visual_review" if quantitative_pass else "do_not_promote"
        ),
        "calls": {
            "training": 0,
            "generation": 0,
            "filtering": 0,
            "repair": 0,
            "retry": 0,
            "route": 0,
            "oracle": 0,
            "candidate_selection": 0,
        },
        "nonclaims": protocol["nonclaims"],
    }
    write_json(output_dir / "adjudication.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--comparison-result", type=Path, required=True)
    parser.add_argument("--comparison-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    result = run_ugi_development_adjudication_protocol(
        args.protocol.resolve(),
        repo,
        args.comparison_result.resolve(),
        args.comparison_archive.resolve(),
        args.output_dir.resolve(),
    )
    print(result["promotion_decision"])


if __name__ == "__main__":
    main()
