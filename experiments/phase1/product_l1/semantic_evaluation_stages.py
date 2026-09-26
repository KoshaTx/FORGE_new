"""Semantic program comparison and adjudication stage adapters."""

from __future__ import annotations

import json
from typing import Any

from experiments._runtime.errors import StageError
from experiments._runtime.registry import stage
from experiments._runtime.stage import (
    ProducedArtifact,
    RunContext,
    StageResult,
    require_config_inputs,
)
from experiments.phase1.product_l1._stage_support import _copy, _deterministic_tar


def _require_successful_ugi_group_balanced_program_prior_preflight(
    config: dict[str, Any], preflight: dict[str, Any]
) -> None:
    """Require the real 256-program H100 execution gate before the full comparison."""

    expected = config.get("profiles", {}).get("h100_preflight", {}).get("program_count")
    checks = preflight.get("preflight_checks")
    if (
        preflight.get("schema_version")
        != "forge.ugi_group_balanced_program_prior_comparison_result.v1"
        or preflight.get("status") != "complete"
        or preflight.get("profile") != "h100_preflight"
        or preflight.get("programs_per_method") != expected
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise StageError(
            "group-balanced program-prior production comparison requires a successful "
            f"authenticated {expected}-program H100 preflight"
        )


def _ugi_group_balanced_program_prior_comparison_stage(
    context: RunContext,
    *,
    run_profile: str,
    gate_dependency_stage: str | None = None,
) -> StageResult:
    """Run the identity-free occurrence- versus group-balanced program-law comparison."""

    from experiments.phase1.product_l1.evaluation.ugi_group_balanced_program_prior_comparison import (
        run_ugi_group_balanced_program_prior_comparison,
    )

    config = context.config()
    require_config_inputs(context, config)
    if gate_dependency_stage is not None:
        preflight = json.loads(context.dependency(gate_dependency_stage, "result").path.read_text())
        _require_successful_ugi_group_balanced_program_prior_preflight(config, preflight)
    work = context.work_dir / "ugi_group_balanced_program_prior_comparison"
    result = run_ugi_group_balanced_program_prior_comparison(
        context.config_path,
        context.repo,
        work,
        profile=run_profile,
        device=context.resources.device,
        resume=context.resume,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [
        path for path in work.rglob("*") if path.is_file() and path.name != "result.json"
    ]
    _deterministic_tar(
        detail_files,
        context.output_path("comparison_details.tar"),
        base=work,
    )
    control = result["methods"]["occurrence_weighted"]["metrics"]
    treatment = result["methods"]["group_balanced"]["metrics"]
    delta = result["group_balanced_minus_occurrence_weighted"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_group_balanced_program_prior_comparison_result.v1",
            ),
            ProducedArtifact(
                "comparison_details",
                "comparison_details.tar",
                "forge.ugi_group_balanced_program_prior_comparison_archive.v1",
                rows=int(result["programs_per_method"]),
            ),
        ),
        metrics={
            "programs_per_method": int(result["programs_per_method"]),
            "occurrence_weighted_exact_l1": float(control["exact_l1_yield_per_attempt"]),
            "group_balanced_exact_l1": float(treatment["exact_l1_yield_per_attempt"]),
            "realism_c2st_delta": (
                None if delta["realism_c2st_auc"] is None else float(delta["realism_c2st_auc"])
            ),
        },
    )


@stage("evaluate.ugi.group-balanced-program-prior-comparison-preflight.v1")
def evaluate_ugi_group_balanced_program_prior_comparison_preflight_stage(
    context: RunContext,
) -> StageResult:
    """Run the paid execution-only H100 preflight for the new program law."""

    return _ugi_group_balanced_program_prior_comparison_stage(
        context,
        run_profile="h100_preflight",
    )


@stage("evaluate.ugi.group-balanced-program-prior-comparison.v1")
def evaluate_ugi_group_balanced_program_prior_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Run the full matched base-model realism comparison after preflight."""

    return _ugi_group_balanced_program_prior_comparison_stage(
        context,
        run_profile="full",
        gate_dependency_stage="preflight",
    )


def _require_successful_ugi_amine_semantic_program_preflight(
    config: dict[str, Any], preflight: dict[str, Any]
) -> None:
    """Require the real 256-program H100 semantic-support gate before production."""

    expected = config.get("profiles", {}).get("h100_preflight", {}).get("program_count")
    checks = preflight.get("preflight_checks")
    if (
        preflight.get("schema_version") != "forge.ugi_amine_semantic_program_comparison_result.v1"
        or preflight.get("status") != "complete"
        or preflight.get("profile") != "h100_preflight"
        or preflight.get("programs_per_method") != expected
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise StageError(
            "amine-semantic production comparison requires a successful authenticated "
            f"{expected}-program H100 preflight"
        )


def _ugi_amine_semantic_program_comparison_stage(
    context: RunContext,
    *,
    run_profile: str,
    gate_dependency_stage: str | None = None,
) -> StageResult:
    """Run the paired count-only versus identity-free amine-semantic comparison."""

    from experiments.phase1.product_l1.evaluation.ugi_amine_semantic_program_comparison import (
        run_ugi_amine_semantic_program_comparison,
    )

    config = context.config()
    require_config_inputs(context, config)
    if gate_dependency_stage is not None:
        preflight = json.loads(context.dependency(gate_dependency_stage, "result").path.read_text())
        _require_successful_ugi_amine_semantic_program_preflight(config, preflight)
    work = context.work_dir / "ugi_amine_semantic_program_comparison"
    result = run_ugi_amine_semantic_program_comparison(
        context.config_path,
        context.repo,
        work,
        profile=run_profile,
        device=context.resources.device,
        resume=context.resume,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [
        path for path in work.rglob("*") if path.is_file() and path.name != "result.json"
    ]
    _deterministic_tar(
        detail_files,
        context.output_path("comparison_details.tar"),
        base=work,
    )
    control = result["methods"]["count_only"]["metrics"]
    treatment = result["methods"]["amine_semantic"]["metrics"]
    delta = result["amine_semantic_minus_count_only"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_amine_semantic_program_comparison_result.v1",
            ),
            ProducedArtifact(
                "comparison_details",
                "comparison_details.tar",
                "forge.ugi_amine_semantic_program_comparison_archive.v1",
                rows=int(result["programs_per_method"]),
            ),
        ),
        metrics={
            "programs_per_method": int(result["programs_per_method"]),
            "count_only_exact_l1": float(control["exact_l1_yield_per_attempt"]),
            "amine_semantic_exact_l1": float(treatment["exact_l1_yield_per_attempt"]),
            "realism_c2st_delta": (
                None if delta["realism_c2st_auc"] is None else float(delta["realism_c2st_auc"])
            ),
        },
    )


@stage("evaluate.ugi.amine-semantic-program-comparison-preflight.v1")
def evaluate_ugi_amine_semantic_program_comparison_preflight_stage(
    context: RunContext,
) -> StageResult:
    """Run the paid execution-only H100 preflight for semantic head support."""

    return _ugi_amine_semantic_program_comparison_stage(
        context,
        run_profile="h100_preflight",
    )


@stage("evaluate.ugi.amine-semantic-program-comparison.v1")
def evaluate_ugi_amine_semantic_program_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Run the full paired semantic-program comparison after its strict preflight."""

    return _ugi_amine_semantic_program_comparison_stage(
        context,
        run_profile="full",
        gate_dependency_stage="preflight",
    )


@stage("evaluate.ugi.amine-semantic-joint-support-adjudication.v1")
def evaluate_ugi_amine_semantic_joint_support_adjudication_stage(
    context: RunContext,
) -> StageResult:
    """Apply the frozen measured-Ugi role panel to the full joint-support comparison."""

    from experiments.phase1.multireaction.ugi_amine_semantic_joint_support_adjudication import (
        run_ugi_amine_semantic_joint_support_adjudication,
    )

    config = context.config()
    require_config_inputs(context, config)
    comparison_result = context.dependency("compare_full", "result").path
    comparison_archive = context.dependency("compare_full", "comparison_details").path
    work = context.work_dir / "ugi_amine_semantic_joint_support_adjudication"
    result = run_ugi_amine_semantic_joint_support_adjudication(
        context.config_path,
        context.repo,
        comparison_result,
        comparison_archive,
        work,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [
        path for path in work.rglob("*") if path.is_file() and path.name != "result.json"
    ]
    _deterministic_tar(
        detail_files,
        context.output_path("adjudication_details.tar"),
        base=work,
    )
    treatment = result["panel_comparison"]["comparisons"]["amine_semantic"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_amine_semantic_joint_support_adjudication.v1",
            ),
            ProducedArtifact(
                "adjudication_details",
                "adjudication_details.tar",
                "forge.ugi_amine_semantic_joint_support_adjudication_archive.v1",
                rows=int(result["expected_attempts_per_arm"]),
            ),
        ),
        metrics={
            "promotion_decision": str(result["promotion_decision"]),
            "role_primary_metrics_improved": int(treatment["primary_metrics_improved"]),
            "role_primary_metrics_total": int(treatment["primary_metrics_total"]),
            "exact_l1_delta": float(result["sampler_metric_delta"]["exact_l1_yield_per_attempt"]),
        },
    )


def _require_successful_ugi_all_role_semantic_preflight(
    config: dict[str, Any], preflight: dict[str, Any]
) -> None:
    """Require the authenticated H100 all-role preservation gate before production."""

    expected = config.get("profiles", {}).get("h100_preflight", {}).get("program_count")
    checks = preflight.get("preflight_checks")
    if (
        preflight.get("schema_version")
        != "forge.ugi_all_role_semantic_program_comparison_result.v1"
        or preflight.get("status") != "complete"
        or preflight.get("profile") != "h100_preflight"
        or preflight.get("programs_per_method") != expected
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise StageError(
            "all-role semantic production comparison requires a successful authenticated "
            f"{expected}-program H100 preflight"
        )


def _require_successful_ugi_all_role_semantic_quantitative_preflight(
    adjudication: dict[str, Any], *, expected_attempts: int
) -> None:
    """Require every frozen realism, assembly, and diversity preflight gate."""

    checks = adjudication.get("promotion_checks")
    if (
        adjudication.get("schema_version") != "forge.ugi_all_role_semantic_adjudication.v1"
        or adjudication.get("status") != "complete"
        or adjudication.get("expected_attempts_per_arm") != expected_attempts
        or adjudication.get("quantitative_decision") != "pass_seed0_quantitative_gate"
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise StageError(
            "all-role semantic production comparison requires a successful authenticated "
            f"{expected_attempts}-program quantitative preflight"
        )


def _ugi_all_role_semantic_program_comparison_stage(
    context: RunContext,
    *,
    run_profile: str,
    gate_dependency_stage: str | None = None,
    quantitative_gate_dependency_stage: str | None = None,
) -> StageResult:
    """Run the paired frozen amine-semantic versus all-role semantic comparison."""

    from experiments.phase1.product_l1.evaluation.ugi_all_role_semantic_program_comparison import (
        run_ugi_all_role_semantic_program_comparison,
    )

    config = context.config()
    require_config_inputs(context, config)
    if gate_dependency_stage is not None:
        preflight = json.loads(context.dependency(gate_dependency_stage, "result").path.read_text())
        _require_successful_ugi_all_role_semantic_preflight(config, preflight)
    if quantitative_gate_dependency_stage is not None:
        adjudication = json.loads(
            context.dependency(quantitative_gate_dependency_stage, "result").path.read_text()
        )
        expected = int(config["profiles"]["h100_preflight"]["program_count"])
        _require_successful_ugi_all_role_semantic_quantitative_preflight(
            adjudication,
            expected_attempts=expected,
        )
    work = context.work_dir / "ugi_all_role_semantic_program_comparison"
    result = run_ugi_all_role_semantic_program_comparison(
        context.config_path,
        context.repo,
        work,
        profile=run_profile,
        device=context.resources.device,
        resume=context.resume,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [
        path for path in work.rglob("*") if path.is_file() and path.name != "result.json"
    ]
    _deterministic_tar(
        detail_files,
        context.output_path("comparison_details.tar"),
        base=work,
    )
    baseline = result["methods"]["amine_semantic"]["metrics"]
    treatment = result["methods"]["all_role_semantic"]["metrics"]
    delta = result["all_role_minus_amine_semantic"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_all_role_semantic_program_comparison_result.v1",
            ),
            ProducedArtifact(
                "comparison_details",
                "comparison_details.tar",
                "forge.ugi_all_role_semantic_program_comparison_archive.v1",
                rows=int(result["programs_per_method"]),
            ),
        ),
        metrics={
            "programs_per_method": int(result["programs_per_method"]),
            "amine_semantic_exact_l1": float(baseline["exact_l1_yield_per_attempt"]),
            "all_role_semantic_exact_l1": float(treatment["exact_l1_yield_per_attempt"]),
            "exact_l1_delta": float(delta["exact_l1_yield_per_attempt"]),
        },
    )


@stage("evaluate.ugi.all-role-semantic-program-comparison-preflight.v1")
def evaluate_ugi_all_role_semantic_program_comparison_preflight_stage(
    context: RunContext,
) -> StageResult:
    """Run the paid 256-program H100 all-role semantic preflight."""

    return _ugi_all_role_semantic_program_comparison_stage(
        context,
        run_profile="h100_preflight",
    )


@stage("evaluate.ugi.all-role-semantic-program-comparison.v1")
def evaluate_ugi_all_role_semantic_program_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Run the full all-role comparison after its exact-L1 preservation preflight."""

    return _ugi_all_role_semantic_program_comparison_stage(
        context,
        run_profile="full",
        gate_dependency_stage="preflight",
    )


@stage("evaluate.ugi.all-role-semantic-program-comparison-quantitative-gated.v1")
def evaluate_ugi_all_role_semantic_program_comparison_quantitative_gated_stage(
    context: RunContext,
) -> StageResult:
    """Run the full comparison only after the complete 256-program gate passes."""

    return _ugi_all_role_semantic_program_comparison_stage(
        context,
        run_profile="full",
        gate_dependency_stage="preflight",
        quantitative_gate_dependency_stage="adjudicate_preflight",
    )


def _require_successful_ugi_morphology_diversity_preflight(
    config: dict[str, Any], preflight: dict[str, Any]
) -> None:
    """Require the authenticated H100 morphology-diversity gate before production."""

    expected = config.get("profiles", {}).get("h100_preflight", {}).get("program_count")
    checks = preflight.get("preflight_checks")
    if (
        preflight.get("schema_version") != "forge.ugi_morphology_diversity_comparison_result.v1"
        or preflight.get("status") != "complete"
        or preflight.get("profile") != "h100_preflight"
        or preflight.get("programs_per_method") != expected
        or preflight.get("promotion_decision") != "eligible_for_full_comparison"
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise StageError(
            "morphology-diversity production comparison requires a successful authenticated "
            f"{expected}-program H100 preflight"
        )


def _ugi_morphology_diversity_comparison_stage(
    context: RunContext,
    *,
    run_profile: str,
    gate_dependency_stage: str | None = None,
) -> StageResult:
    """Compare the frozen and entropy-calibrated measured morphology laws."""

    from experiments.phase1.product_l1.evaluation.ugi_morphology_diversity_comparison import (
        run_ugi_morphology_diversity_comparison,
    )

    config = context.config()
    require_config_inputs(context, config)
    if gate_dependency_stage is not None:
        preflight = json.loads(context.dependency(gate_dependency_stage, "result").path.read_text())
        _require_successful_ugi_morphology_diversity_preflight(config, preflight)
    work = context.work_dir / "ugi_morphology_diversity_comparison"
    result = run_ugi_morphology_diversity_comparison(
        context.config_path,
        context.repo,
        work,
        profile=run_profile,
        device=context.resources.device,
        resume=context.resume,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [
        path for path in work.rglob("*") if path.is_file() and path.name != "result.json"
    ]
    _deterministic_tar(
        detail_files,
        context.output_path("comparison_details.tar"),
        base=work,
    )
    baseline = result["methods"]["baseline"]
    treatment = result["methods"]["treatment"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_morphology_diversity_comparison_result.v1",
            ),
            ProducedArtifact(
                "comparison_details",
                "comparison_details.tar",
                "forge.ugi_morphology_diversity_comparison_archive.v1",
                rows=int(result["programs_per_method"]),
            ),
        ),
        metrics={
            "programs_per_method": int(result["programs_per_method"]),
            "baseline_exact_l1": float(baseline["metrics"]["exact_l1_yield_per_attempt"]),
            "treatment_exact_l1": float(treatment["metrics"]["exact_l1_yield_per_attempt"]),
            "baseline_effective_topologies": float(
                baseline["topology_diversity"]["effective_complete_topology_count"]
            ),
            "treatment_effective_topologies": float(
                treatment["topology_diversity"]["effective_complete_topology_count"]
            ),
        },
    )


@stage("evaluate.ugi.morphology-diversity-comparison-preflight.v1")
def evaluate_ugi_morphology_diversity_comparison_preflight_stage(
    context: RunContext,
) -> StageResult:
    """Run the paid 256-program H100 morphology-diversity preflight."""

    return _ugi_morphology_diversity_comparison_stage(
        context,
        run_profile="h100_preflight",
    )


@stage("evaluate.ugi.morphology-diversity-comparison.v1")
def evaluate_ugi_morphology_diversity_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Run the full morphology comparison only after its H100 preflight passes."""

    return _ugi_morphology_diversity_comparison_stage(
        context,
        run_profile="full",
        gate_dependency_stage="preflight",
    )


def _ugi_all_role_semantic_adjudication_stage(
    context: RunContext,
    *,
    dependency_stage: str,
) -> StageResult:
    """Apply the complete frozen quantitative gate to one comparison stage."""

    from experiments.phase1.multireaction.ugi_all_role_semantic_adjudication import (
        run_ugi_all_role_semantic_adjudication,
    )

    config = context.config()
    require_config_inputs(context, config)
    work = context.work_dir / "ugi_all_role_semantic_adjudication"
    result = run_ugi_all_role_semantic_adjudication(
        context.config_path,
        context.repo,
        context.dependency(dependency_stage, "result").path,
        context.dependency(dependency_stage, "comparison_details").path,
        work,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [
        path for path in work.rglob("*") if path.is_file() and path.name != "result.json"
    ]
    _deterministic_tar(
        detail_files,
        context.output_path("adjudication_details.tar"),
        base=work,
    )
    treatment = result["panel_comparison"]["comparisons"]["all_role_semantic"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_all_role_semantic_adjudication.v1",
            ),
            ProducedArtifact(
                "adjudication_details",
                "adjudication_details.tar",
                "forge.ugi_all_role_semantic_adjudication_archive.v1",
                rows=int(result["expected_attempts_per_arm"]),
            ),
        ),
        metrics={
            "promotion_decision": str(result["promotion_decision"]),
            "role_primary_metrics_improved": int(treatment["primary_metrics_improved"]),
            "role_primary_metrics_total": int(treatment["primary_metrics_total"]),
            "exact_l1_delta": float(result["sampler_metric_delta"]["exact_l1_yield_per_attempt"]),
        },
    )


@stage("evaluate.ugi.all-role-semantic-adjudication.v1")
def evaluate_ugi_all_role_semantic_adjudication_stage(
    context: RunContext,
) -> StageResult:
    """Apply the six-metric measured-Ugi role panel to the full comparison."""

    return _ugi_all_role_semantic_adjudication_stage(
        context,
        dependency_stage="compare_full",
    )


@stage("evaluate.ugi.all-role-semantic-preflight-adjudication.v1")
def evaluate_ugi_all_role_semantic_preflight_adjudication_stage(
    context: RunContext,
) -> StageResult:
    """Apply the complete frozen quantitative gate to a 256-program preflight."""

    return _ugi_all_role_semantic_adjudication_stage(
        context,
        dependency_stage="preflight",
    )


@stage("evaluate.ugi.all-role-semantic-visual-review.v1")
def evaluate_ugi_all_role_semantic_visual_review_stage(
    context: RunContext,
) -> StageResult:
    """Prepare a fixed-index blinded morphology packet after quantitative adjudication."""

    from experiments.phase1.multireaction.ugi_all_role_semantic_visual_review import (
        run_ugi_all_role_semantic_visual_review,
    )

    config = context.config()
    require_config_inputs(context, config)
    work = context.work_dir / "ugi_all_role_semantic_visual_review"
    result = run_ugi_all_role_semantic_visual_review(
        context.config_path,
        context.repo,
        context.dependency("compare_full", "result").path,
        context.dependency("compare_full", "comparison_details").path,
        context.dependency("adjudicate", "result").path,
        work,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    _copy(work / "blinding_key.json", context.output_path("blinding_key.json"))
    _deterministic_tar(
        [
            work / "instructions.txt",
            work / "review_packet.html",
            work / "review_sheet.json",
        ],
        context.output_path("review_packet.tar"),
        base=work,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_all_role_semantic_visual_review.v1",
            ),
            ProducedArtifact(
                "review_packet",
                "review_packet.tar",
                "forge.ugi_all_role_semantic_blinded_review_packet.v1",
                rows=int(result["pairs"]),
            ),
            ProducedArtifact(
                "blinding_key",
                "blinding_key.json",
                "forge.ugi_all_role_semantic_blinding_key.v1",
                rows=int(result["pairs"]),
            ),
        ),
        metrics={
            "status": str(result["status"]),
            "pairs": int(result["pairs"]),
            "candidate_selection": bool(result["candidate_selection"]),
        },
    )
