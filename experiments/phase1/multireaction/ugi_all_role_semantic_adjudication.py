"""Adjudicate all-role Ugi semantics against the frozen amine-semantic baseline."""

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

CONFIG_SCHEMA = "forge.ugi_all_role_semantic_adjudication_config.v1"
RESULT_SCHEMA = "forge.ugi_all_role_semantic_adjudication.v1"
COMPARISON_RESULT_SCHEMA = "forge.ugi_all_role_semantic_program_comparison_result.v1"
PREFLIGHT_ATTEMPTS = 256
FULL_ATTEMPTS = 3072


class UgiAllRoleSemanticAdjudicationError(ValueError):
    """A dependency or gate violates the frozen all-role realism contract."""


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
        "visual_review",
        "nonclaims",
    }
    if not isinstance(config, Mapping) or set(config) != expected:
        raise UgiAllRoleSemanticAdjudicationError("all-role adjudication fields changed")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise UgiAllRoleSemanticAdjudicationError("all-role adjudication schema changed")
    if not isinstance(config["inputs"], Mapping) or set(config["inputs"]) != {
        "development_realism_config"
    }:
        raise UgiAllRoleSemanticAdjudicationError("all-role adjudication inputs changed")
    if config["baseline_arm"] != "amine_semantic" or config["treatment_arm"] != (
        "all_role_semantic"
    ):
        raise UgiAllRoleSemanticAdjudicationError("all-role adjudication arms changed")
    if config["expected_attempts"] not in {PREFLIGHT_ATTEMPTS, FULL_ATTEMPTS}:
        raise UgiAllRoleSemanticAdjudicationError("all-role attempt contract changed")
    if set(config["arms"]) != {"amine_semantic", "all_role_semantic"}:
        raise UgiAllRoleSemanticAdjudicationError("all-role adjudication arm set changed")
    for arm_id, arm in config["arms"].items():
        if not isinstance(arm, Mapping) or set(arm) != {"archive_member", "method_id", "seed"}:
            raise UgiAllRoleSemanticAdjudicationError(
                f"all-role adjudication arm fields changed for {arm_id}"
            )
    expected_gate = {
        "effective_component_count_minimum_retained_ratio": 0.95,
        "exact_l1_absolute_minimum": 0.95,
        "exact_l1_noninferiority_margin": 0.01,
        "require_uniform_role_panel_improvement": True,
        "require_zero_program_support_abstentions": True,
        "unique_exact_l1_minimum_retained_ratio": 0.95,
        "validity_absolute_minimum": 0.95,
    }
    if config["promotion_gate"] != expected_gate:
        raise UgiAllRoleSemanticAdjudicationError("all-role promotion gate changed")
    if config["visual_review"] != {
        "blinded": True,
        "required_after_quantitative_gate": True,
        "criteria": [
            "tail morphology is at least as plausible as the frozen baseline",
            "head-tail balance is visibly improved",
            "no recurrent unsupported ring or heteroatom pathology is introduced",
        ],
    }:
        raise UgiAllRoleSemanticAdjudicationError("all-role visual-review contract changed")
    _validate_metric_panel(config["metric_panel"])
    return dict(config)


def _validate_comparison(result: Any, *, expected_attempts: int) -> dict[str, Any]:
    expected_profile = (
        "h100_preflight" if expected_attempts == PREFLIGHT_ATTEMPTS else "full"
    )
    if (
        not isinstance(result, Mapping)
        or result.get("schema_version") != COMPARISON_RESULT_SCHEMA
        or result.get("status") != "complete"
        or result.get("profile") != expected_profile
        or result.get("programs_per_method") != expected_attempts
        or set(result.get("methods", {})) != {"amine_semantic", "all_role_semantic"}
    ):
        raise UgiAllRoleSemanticAdjudicationError("all-role full comparison is inadmissible")
    structural = result.get("structural_checks")
    if not isinstance(structural, Mapping) or not structural or not all(
        value is True for value in structural.values()
    ):
        raise UgiAllRoleSemanticAdjudicationError("all-role structural contract failed")
    treatment = result["methods"]["all_role_semantic"].get("sampling_summary")
    baseline = result["methods"]["amine_semantic"].get("sampling_summary")
    if (
        not isinstance(treatment, Mapping)
        or not isinstance(baseline, Mapping)
        or treatment.get("ugi_all_role_semantic_joint_support_applied") is not True
        or baseline.get("ugi_amine_semantic_joint_support_applied") is not True
        or treatment.get("repairs") != {}
        or baseline.get("repairs") != {}
    ):
        raise UgiAllRoleSemanticAdjudicationError(
            "full comparison did not preserve the paired no-repair semantic decoders"
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
    with tempfile.TemporaryDirectory(prefix="forge-ugi-all-role-") as temporary:
        temporary_dir = Path(temporary)
        with tarfile.open(comparison_archive, mode="r:*") as archive:
            available = {member.name: member for member in archive.getmembers()}
            for arm_id, arm in sorted(config["arms"].items()):
                member_name = str(arm["archive_member"])
                member_path = _validate_member(member_name)
                member = available.get(member_name)
                if member is None or not member.isfile():
                    raise UgiAllRoleSemanticAdjudicationError(
                        f"attempt ledger is absent from comparison archive: {member_name}"
                    )
                extracted_path = temporary_dir.joinpath(*member_path.parts)
                extracted_path.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise UgiAllRoleSemanticAdjudicationError(
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
        raise UgiAllRoleSemanticAdjudicationError("one or more all-role assessments failed")
    summaries = {arm: _arm_summary(value) for arm, value in sorted(arm_results.items())}
    panel = _panel_comparison(
        summaries,
        baseline_arm=str(config["baseline_arm"]),
        panel=_validate_metric_panel(config["metric_panel"]),
    )
    return panel, summaries


def _positive_ratio(numerator: float, denominator: float, *, label: str) -> float:
    if denominator <= 0:
        raise UgiAllRoleSemanticAdjudicationError(f"baseline {label} is not positive")
    return numerator / denominator


def run_ugi_all_role_semantic_adjudication(
    config_path: Path,
    repo: Path,
    comparison_result_path: Path,
    comparison_archive_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Merge sampler preservation gates with the six-metric measured-Ugi role panel."""

    config = _validate_config(
        read_json_object(
            config_path,
            error=UgiAllRoleSemanticAdjudicationError,
            label="all-role semantic adjudication config",
        )
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise UgiAllRoleSemanticAdjudicationError(
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
            error=UgiAllRoleSemanticAdjudicationError,
            label="full all-role semantic comparison result",
        ),
        expected_attempts=int(config["expected_attempts"]),
    )
    comparison_stage = (
        "preflight"
        if int(config["expected_attempts"]) == PREFLIGHT_ATTEMPTS
        else "compare_full"
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
    exact_l1 = float(treatment_metrics["exact_l1_yield_per_attempt"])
    baseline_exact_l1 = float(baseline_metrics["exact_l1_yield_per_attempt"])
    unique_ratio = _positive_ratio(
        float(treatment_metrics["unique_exact_l1_products_per_attempt"]),
        float(baseline_metrics["unique_exact_l1_products_per_attempt"]),
        label="unique exact-L1 yield",
    )
    effective_ratio = _positive_ratio(
        float(treatment_metrics["effective_component_count"]),
        float(baseline_metrics["effective_component_count"]),
        label="effective component count",
    )
    gate = config["promotion_gate"]
    treatment_summary = comparison["methods"][treatment_id]["sampling_summary"]
    baseline_summary = comparison["methods"][baseline_id]["sampling_summary"]
    panel_result = panel["comparisons"][treatment_id]
    checks = {
        "effective_component_count_retained": effective_ratio
        >= float(gate["effective_component_count_minimum_retained_ratio"]),
        "exact_l1_absolute_minimum": exact_l1 >= float(gate["exact_l1_absolute_minimum"]),
        "exact_l1_noninferior": exact_l1
        >= baseline_exact_l1 - float(gate["exact_l1_noninferiority_margin"]),
        "fixed_state_contract_preserved": treatment_summary["fixed_state_failures"] == 0
        and baseline_summary["fixed_state_failures"] == 0,
        "no_repairs_or_retries": treatment_summary["repairs"] == {}
        and baseline_summary["repairs"] == {},
        "program_support_abstentions_zero": (
            comparison["methods"][treatment_id]["program_support_abstentions"] == 0
            and comparison["methods"][baseline_id]["program_support_abstentions"] == 0
            if gate["require_zero_program_support_abstentions"]
            else True
        ),
        "role_panel_uniformly_improved": (
            panel_result["classification"] == "uniformly_improved"
            if gate["require_uniform_role_panel_improvement"]
            else True
        ),
        "unique_exact_l1_retained": unique_ratio
        >= float(gate["unique_exact_l1_minimum_retained_ratio"]),
        "validity_absolute_minimum": float(treatment_metrics["valid_fraction_per_attempt"])
        >= float(gate["validity_absolute_minimum"]),
    }
    quantitative_pass = all(checks.values())
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_question": config["scientific_question"],
        "expected_attempts_per_arm": int(config["expected_attempts"]),
        "inputs": {
            "config": pin_record(config_path, repo),
            "development_realism_config": pin_record(realism_config, repo),
            "comparison_result": artifact_record(
                comparison_result_path, logical_path=f"{comparison_stage}/result.json"
            ),
            "comparison_archive": artifact_record(
                comparison_archive_path,
                logical_path=f"{comparison_stage}/comparison_details.tar",
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
        "unique_exact_l1_retained_ratio": unique_ratio,
        "effective_component_count_retained_ratio": effective_ratio,
        "promotion_gate": gate,
        "promotion_checks": checks,
        "quantitative_decision": (
            "pass_seed0_quantitative_gate" if quantitative_pass else "do_not_promote"
        ),
        "promotion_decision": (
            "eligible_for_blinded_visual_review" if quantitative_pass else "do_not_promote"
        ),
        "visual_review": {
            **dict(config["visual_review"]),
            "status": "pending" if quantitative_pass else "not_reached",
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
    run_ugi_all_role_semantic_adjudication(
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
    "UgiAllRoleSemanticAdjudicationError",
    "run_ugi_all_role_semantic_adjudication",
]
