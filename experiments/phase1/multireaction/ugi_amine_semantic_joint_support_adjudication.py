"""Adjudicate joint-support amine semantics with the frozen measured-Ugi role panel."""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.phase1.multireaction.lipid_realism_assessment import (
    run_lipid_realism_assessment,
)
from experiments.phase1.multireaction.ugi_development_realism_comparison import (
    _arm_summary,
    _panel_comparison,
    _validate_member,
    _validate_metric_panel,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import read_json_object, write_json

CONFIG_SCHEMA = "forge.ugi_amine_semantic_joint_support_adjudication_config.v1"
RESULT_SCHEMA = "forge.ugi_amine_semantic_joint_support_adjudication.v1"
COMPARISON_RESULT_SCHEMA = "forge.ugi_amine_semantic_program_comparison_result.v1"


class UgiAmineSemanticJointSupportAdjudicationError(ValueError):
    """A dependency or gate violates the frozen joint-support adjudication contract."""


def _validate_config(config: Any) -> dict[str, Any]:
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
        "nonclaims",
    }
    if not isinstance(config, Mapping) or set(config) != expected:
        raise UgiAmineSemanticJointSupportAdjudicationError(
            "joint-support adjudication config fields changed"
        )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise UgiAmineSemanticJointSupportAdjudicationError(
            "joint-support adjudication schema changed"
        )
    if not isinstance(config["inputs"], Mapping) or set(config["inputs"]) != {
        "development_realism_config"
    }:
        raise UgiAmineSemanticJointSupportAdjudicationError(
            "joint-support adjudication inputs changed"
        )
    if config["baseline_arm"] != "count_only" or config["treatment_arm"] != "amine_semantic":
        raise UgiAmineSemanticJointSupportAdjudicationError("adjudication arms changed")
    if config["expected_attempts"] != 3072:
        raise UgiAmineSemanticJointSupportAdjudicationError("attempt contract changed")
    if set(config["arms"]) != {"count_only", "amine_semantic"}:
        raise UgiAmineSemanticJointSupportAdjudicationError("adjudication arm set changed")
    for arm_id, arm in config["arms"].items():
        if not isinstance(arm, Mapping) or set(arm) != {"archive_member", "method_id", "seed"}:
            raise UgiAmineSemanticJointSupportAdjudicationError(
                f"adjudication arm fields changed for {arm_id}"
            )
    expected_gate = {
        "effective_component_count_minimum_retained_ratio": 0.8,
        "exact_l1_noninferiority_margin": 0.02,
        "local_support_noninferiority_margin": 0.02,
        "mean_pairwise_ecfp4_distance_noninferiority_margin": 0.05,
        "require_uniform_role_panel_improvement": True,
        "require_zero_program_support_abstentions": True,
        "unique_exact_l1_noninferiority_margin": 0.05,
        "validity_noninferiority_margin": 0.02,
    }
    if config["promotion_gate"] != expected_gate:
        raise UgiAmineSemanticJointSupportAdjudicationError("promotion gate changed")
    _validate_metric_panel(config["metric_panel"])
    return dict(config)


def _validate_comparison(result: Any, *, expected_attempts: int) -> dict[str, Any]:
    if (
        not isinstance(result, Mapping)
        or result.get("schema_version") != COMPARISON_RESULT_SCHEMA
        or result.get("status") != "complete"
        or result.get("profile") != "full"
        or result.get("programs_per_method") != expected_attempts
        or set(result.get("methods", {})) != {"count_only", "amine_semantic"}
    ):
        raise UgiAmineSemanticJointSupportAdjudicationError(
            "full comparison dependency is inadmissible"
        )
    structural = result.get("structural_checks")
    if not isinstance(structural, Mapping) or not structural or not all(
        value is True for value in structural.values()
    ):
        raise UgiAmineSemanticJointSupportAdjudicationError(
            "full comparison structural contract failed"
        )
    treatment_summary = result["methods"]["amine_semantic"].get("sampling_summary")
    if (
        not isinstance(treatment_summary, Mapping)
        or treatment_summary.get("ugi_amine_semantic_joint_support_applied") is not True
        or treatment_summary.get("repairs") != {}
    ):
        raise UgiAmineSemanticJointSupportAdjudicationError(
            "full comparison did not apply the joint-support no-repair decoder"
        )
    return dict(result)


def _run_role_panel(
    *,
    config: Mapping[str, Any],
    realism_config: Path,
    comparison_archive: Path,
    repo: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    expected_attempts = int(config["expected_attempts"])
    arm_results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="forge-ugi-joint-support-") as temporary:
        temporary_dir = Path(temporary)
        with tarfile.open(comparison_archive, mode="r:*") as archive:
            available = {member.name: member for member in archive.getmembers()}
            for arm_id, arm in sorted(config["arms"].items()):
                member_name = str(arm["archive_member"])
                member_path = _validate_member(member_name)
                member = available.get(member_name)
                if member is None or not member.isfile():
                    raise UgiAmineSemanticJointSupportAdjudicationError(
                        f"attempt ledger is absent from comparison archive: {member_name}"
                    )
                extracted_path = temporary_dir.joinpath(*member_path.parts)
                extracted_path.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise UgiAmineSemanticJointSupportAdjudicationError(
                        f"could not read attempt ledger: {member_name}"
                    )
                extracted_path.write_bytes(source.read())
                arm_results[arm_id] = run_lipid_realism_assessment(
                    realism_config,
                    repo,
                    extracted_path,
                    output_dir / arm_id,
                    method_id=str(arm["method_id"]),
                    seed=int(arm["seed"]),
                    expected_attempts=expected_attempts,
                )
    if not all(result["status"] == "pass" for result in arm_results.values()):
        raise UgiAmineSemanticJointSupportAdjudicationError(
            "one or more measured-Ugi role assessments failed"
        )
    summaries = {arm: _arm_summary(value) for arm, value in sorted(arm_results.items())}
    panel = _panel_comparison(
        summaries,
        baseline_arm=str(config["baseline_arm"]),
        panel=_validate_metric_panel(config["metric_panel"]),
    )
    return panel, summaries


def run_ugi_amine_semantic_joint_support_adjudication(
    config_path: Path,
    repo: Path,
    comparison_result_path: Path,
    comparison_archive_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Merge matched sampler gates with the frozen train-only measured-Ugi role panel."""

    config = _validate_config(
        read_json_object(
            config_path,
            error=UgiAmineSemanticJointSupportAdjudicationError,
            label="joint-support adjudication config",
        )
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise UgiAmineSemanticJointSupportAdjudicationError(
            f"adjudication output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    realism_config = resolve_pin(
        config["inputs"]["development_realism_config"],
        repo,
        label="development realism config",
    )
    comparison = _validate_comparison(
        read_json_object(
            comparison_result_path,
            error=UgiAmineSemanticJointSupportAdjudicationError,
            label="full amine-semantic comparison result",
        ),
        expected_attempts=int(config["expected_attempts"]),
    )
    panel, summaries = _run_role_panel(
        config=config,
        realism_config=realism_config,
        comparison_archive=comparison_archive_path,
        repo=repo,
        output_dir=output_dir,
    )
    baseline_id = str(config["baseline_arm"])
    treatment_id = str(config["treatment_arm"])
    baseline_metrics = comparison["methods"][baseline_id]["metrics"]
    treatment_metrics = comparison["methods"][treatment_id]["metrics"]
    delta = {
        key: float(treatment_metrics[key]) - float(baseline_metrics[key])
        for key in treatment_metrics
        if isinstance(treatment_metrics[key], (int, float))
        and not isinstance(treatment_metrics[key], bool)
        and isinstance(baseline_metrics.get(key), (int, float))
        and not isinstance(baseline_metrics.get(key), bool)
    }
    gate = config["promotion_gate"]
    retained_ratio = float(treatment_metrics["effective_component_count"]) / float(
        baseline_metrics["effective_component_count"]
    )
    treatment_summary = comparison["methods"][treatment_id]["sampling_summary"]
    control_summary = comparison["methods"][baseline_id]["sampling_summary"]
    panel_result = panel["comparisons"][treatment_id]
    checks = {
        "effective_component_count_preserved": retained_ratio
        >= float(gate["effective_component_count_minimum_retained_ratio"]),
        "exact_l1_preserved": delta["exact_l1_yield_per_attempt"]
        >= -float(gate["exact_l1_noninferiority_margin"]),
        "fixed_state_contract_preserved": treatment_summary["fixed_state_failures"] == 0
        and control_summary["fixed_state_failures"] == 0,
        "local_support_preserved": delta["local_support_qualified_exact_l1_yield_per_attempt"]
        >= -float(gate["local_support_noninferiority_margin"]),
        "no_repairs": treatment_summary["repairs"] == {} and control_summary["repairs"] == {},
        "pairwise_diversity_preserved": delta["mean_pairwise_ecfp4_distance"]
        >= -float(gate["mean_pairwise_ecfp4_distance_noninferiority_margin"]),
        "program_support_abstentions_zero": (
            comparison["methods"][treatment_id]["program_support_abstentions"] == 0
            if gate["require_zero_program_support_abstentions"]
            else True
        ),
        "role_panel_uniformly_improved": (
            panel_result["classification"] == "uniformly_improved"
            if gate["require_uniform_role_panel_improvement"]
            else True
        ),
        "unique_exact_l1_preserved": delta["unique_exact_l1_products_per_attempt"]
        >= -float(gate["unique_exact_l1_noninferiority_margin"]),
        "validity_preserved": delta["valid_fraction_per_attempt"]
        >= -float(gate["validity_noninferiority_margin"]),
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_question": config["scientific_question"],
        "expected_attempts_per_arm": int(config["expected_attempts"]),
        "inputs": {
            "config": pin_record(config_path, repo),
            "development_realism_config": pin_record(realism_config, repo),
            "comparison_result": artifact_record(
                comparison_result_path, logical_path="compare_full/result.json"
            ),
            "comparison_archive": artifact_record(
                comparison_archive_path, logical_path="compare_full/comparison_details.tar"
            ),
        },
        "arms": {
            arm: {
                "assessment": artifact_record(
                    output_dir / arm / "result.json", logical_path=f"{arm}/result.json"
                ),
                "summary": summaries[arm],
            }
            for arm in sorted(summaries)
        },
        "panel_comparison": panel,
        "sampler_metric_delta": delta,
        "effective_component_count_retained_ratio": retained_ratio,
        "promotion_gate": gate,
        "promotion_checks": checks,
        "promotion_decision": (
            "promote_seed0_development" if all(checks.values()) else "do_not_promote"
        ),
        "broad_r0_c2st": {
            "status": "secondary_compatibility_diagnostic",
            "baseline": baseline_metrics.get("realism_c2st_auc"),
            "treatment": treatment_metrics.get("realism_c2st_auc"),
            "treatment_minus_baseline": delta.get("realism_c2st_auc"),
            "used_for_promotion": False,
        },
        "calls": {
            "training": 0,
            "generation": 0,
            "repair": 0,
            "retry": 0,
            "route": 0,
            "oracle": 0,
            "candidate_selection": 0,
        },
        "nonclaims": config["nonclaims"],
    }
    write_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    parser.add_argument("--comparison-result", required=True, type=Path)
    parser.add_argument("--comparison-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_ugi_amine_semantic_joint_support_adjudication(
        args.config.resolve(),
        args.repo.resolve(),
        args.comparison_result.resolve(),
        args.comparison_archive.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiAmineSemanticJointSupportAdjudicationError",
    "run_ugi_amine_semantic_joint_support_adjudication",
]
