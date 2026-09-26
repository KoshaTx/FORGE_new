"""Specialist comparison stage adapters and preflight gates."""

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


def _reaction_specialist_v0_comparison_stage(
    context: RunContext,
    *,
    topology_specialist: bool,
) -> StageResult:
    from experiments.phase1.product_l1.evaluation.reaction_specialist_ugi_v0_comparison import (
        run_reaction_specialist_ugi_v0_comparison,
    )

    config = context.config()
    require_config_inputs(context, config)
    if "specialize_ugi" in context.stage.needs:
        specialist_checkpoint = context.dependency("specialize_ugi", "checkpoint").path
    else:
        required_recovery = {"specialist_checkpoint", "specialist_result"}
        if not required_recovery.issubset(context.inputs):
            raise StageError(
                "standalone specialist comparison requires pinned checkpoint and result inputs"
            )
        specialist_checkpoint = context.input("specialist_checkpoint")
        if context.input("specialist_result") != specialist_checkpoint.with_name("result.json"):
            raise StageError("standalone specialist result must be adjacent to its checkpoint")
    work = context.work_dir / (
        "reaction_topology_specialist_ugi_v0_comparison"
        if topology_specialist
        else "reaction_specialist_ugi_v0_comparison"
    )
    result = run_reaction_specialist_ugi_v0_comparison(
        context.config_path,
        specialist_checkpoint,
        context.repo,
        work,
        profile=context.profile,
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
    specialist_id = (
        "forge_shared_plus_ugi_topology_specialist_exposure_matched_seed0"
        if topology_specialist
        else "forge_shared_plus_ugi_specialist_exposure_matched_seed0"
    )
    v0_id = "forge_v0_step_3000_program_matched"
    specialist = result["methods"][specialist_id]["metrics"]
    v0 = result["methods"][v0_id]["metrics"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                (
                    "forge.reaction_topology_specialist_ugi_v0_comparison_result.v2"
                    if topology_specialist
                    else "forge.reaction_specialist_ugi_v0_comparison_result.v1"
                ),
            ),
            ProducedArtifact(
                "comparison_details",
                "comparison_details.tar",
                (
                    "forge.reaction_topology_specialist_ugi_v0_comparison_archive.v2"
                    if topology_specialist
                    else "forge.reaction_specialist_ugi_v0_comparison_archive.v1"
                ),
                rows=int(result["programs_per_method"]),
            ),
        ),
        metrics={
            "programs_per_method": int(result["programs_per_method"]),
            "specialist_exact_l1": float(specialist["exact_l1_yield_per_attempt"]),
            "v0_exact_l1": float(v0["exact_l1_yield_per_attempt"]),
            "open_ended_exact_l1_delta": float(
                result["specialist_minus_v0"][
                    "unique_open_ended_whole_product_novel_exact_l1_products_per_attempt"
                ]
            ),
        },
        summary={
            "decision": str(result["decision"]),
            "training_calls": 0,
            "repairs_or_retries": False,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("evaluate.ugi.reaction-specialist-v0-comparison.v1")
def evaluate_ugi_reaction_specialist_v0_comparison_stage(context: RunContext) -> StageResult:
    """Compare the exposure-matched Ugi specialist with v0 on one ordered program draw."""

    return _reaction_specialist_v0_comparison_stage(
        context,
        topology_specialist=False,
    )


@stage("evaluate.ugi.reaction-topology-specialist-v0-comparison.v2")
def evaluate_ugi_reaction_topology_specialist_v0_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Compare exact program-coupled Ugi topology specialization with v0."""

    return _reaction_specialist_v0_comparison_stage(
        context,
        topology_specialist=True,
    )


def _require_successful_ugi_chemistry_preflight(
    config: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    """Reject production unless the real 64-program H100 preflight passed every gate."""

    expected_preflight = config.get("profiles", {}).get("h100_preflight", {})
    expected_program_count = expected_preflight.get("program_count")
    preflight_gates = preflight.get("preflight_gates")
    if (
        preflight.get("status") != "complete"
        or preflight.get("profile") != "h100_preflight"
        or preflight.get("programs_per_method") != expected_program_count
        or not isinstance(preflight_gates, dict)
        or not preflight_gates
        or not all(value is True for value in preflight_gates.values())
    ):
        raise StageError(
            "chemistry-specialist production evaluation requires a successful authenticated "
            f"{expected_program_count}-program h100_preflight result"
        )


def _ugi_chemistry_specialist_base_comparison_stage(
    context: RunContext,
    *,
    dependency_stage: str | None,
    result_schema: str,
    archive_schema: str,
    run_profile: str | None = None,
    gate_dependency_stage: str | None = None,
) -> StageResult:
    """Run one versioned chemistry-only Ugi comparison stage."""

    from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
        run_ugi_chemistry_specialist_comparison,
    )

    config = context.config()
    require_config_inputs(context, config)
    if gate_dependency_stage is not None:
        preflight_artifact = context.dependency(gate_dependency_stage, "result")
        preflight = json.loads(preflight_artifact.path.read_text())
        _require_successful_ugi_chemistry_preflight(config, preflight)
    if dependency_stage is None:
        required_recovery = {"specialist_checkpoint", "specialist_result"}
        if not required_recovery.issubset(context.inputs):
            raise StageError(
                "standalone chemistry-specialist comparison requires pinned checkpoint and "
                "result inputs"
            )
        specialist_checkpoint = context.input("specialist_checkpoint")
        if context.input("specialist_result") != specialist_checkpoint.with_name("result.json"):
            raise StageError(
                "standalone chemistry-specialist result must be adjacent to checkpoint"
            )
    else:
        specialist_checkpoint = context.dependency(dependency_stage, "checkpoint").path
    work = context.work_dir / "ugi_chemistry_specialist_base_comparison"
    result = run_ugi_chemistry_specialist_comparison(
        context.config_path,
        specialist_checkpoint,
        context.repo,
        work,
        profile=run_profile or context.profile,
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
    base = result["methods"]["base"]["metrics"]
    specialist_method_key = str(result.get("specialist_method_key", "chemistry_specialist"))
    specialist = result["methods"][specialist_method_key]["metrics"]
    delta_key = f"{specialist_method_key}_minus_base"
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                result_schema,
            ),
            ProducedArtifact(
                "comparison_details",
                "comparison_details.tar",
                archive_schema,
                rows=int(result["programs_per_method"]),
            ),
        ),
        metrics={
            "programs_per_method": int(result["programs_per_method"]),
            "base_exact_l1": float(base["exact_l1_yield_per_attempt"]),
            "specialist_exact_l1": float(specialist["exact_l1_yield_per_attempt"]),
            "exact_l1_delta": float(result[delta_key]["exact_l1_yield_per_attempt"]),
        },
        summary={
            "promotion_decision": str(result["promotion_decision"]),
            "training_calls": 0,
            "repairs_or_retries": False,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("evaluate.ugi.chemistry-specialist-base-comparison.v3")
def evaluate_ugi_chemistry_specialist_base_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Compare a chemistry-only Ugi delta with its frozen base on paired programs."""

    return _ugi_chemistry_specialist_base_comparison_stage(
        context,
        dependency_stage="specialize_ugi",
        result_schema="forge.ugi_chemistry_specialist_comparison_result.v3",
        archive_schema="forge.ugi_chemistry_specialist_comparison_archive.v3",
    )


@stage("evaluate.ugi.chemistry-specialist-base-comparison-preflight.v4")
def evaluate_ugi_chemistry_specialist_base_comparison_preflight_stage(
    context: RunContext,
) -> StageResult:
    """Gate the real sampler on measured-Ugi joint morphology support."""

    return _ugi_chemistry_specialist_base_comparison_stage(
        context,
        dependency_stage="preflight",
        result_schema="forge.ugi_chemistry_specialist_comparison_result.v4",
        archive_schema="forge.ugi_chemistry_specialist_comparison_archive.v4",
        run_profile="h100_preflight",
    )


@stage("evaluate.ugi.chemistry-specialist-base-comparison.v4")
def evaluate_ugi_chemistry_specialist_base_comparison_v4_stage(
    context: RunContext,
) -> StageResult:
    """Compare a chemistry-only Ugi delta on measured-Ugi joint programs."""

    return _ugi_chemistry_specialist_base_comparison_stage(
        context,
        dependency_stage="specialize_ugi",
        result_schema="forge.ugi_chemistry_specialist_comparison_result.v4",
        archive_schema="forge.ugi_chemistry_specialist_comparison_archive.v4",
    )


@stage("evaluate.ugi.chemistry-specialist-base-comparison-preflight-recovery.v4")
def evaluate_ugi_chemistry_specialist_base_comparison_preflight_recovery_stage(
    context: RunContext,
) -> StageResult:
    """Gate the real sampler using a pinned completed chemistry specialist."""

    return _ugi_chemistry_specialist_base_comparison_stage(
        context,
        dependency_stage=None,
        result_schema="forge.ugi_chemistry_specialist_comparison_result.v4",
        archive_schema="forge.ugi_chemistry_specialist_comparison_archive.v4",
        run_profile="h100_preflight",
    )


@stage("evaluate.ugi.chemistry-specialist-base-comparison-recovery.v4")
def evaluate_ugi_chemistry_specialist_base_comparison_recovery_stage(
    context: RunContext,
) -> StageResult:
    """Evaluate a pinned completed chemistry specialist on measured-Ugi programs."""

    return _ugi_chemistry_specialist_base_comparison_stage(
        context,
        dependency_stage=None,
        result_schema="forge.ugi_chemistry_specialist_comparison_result.v4",
        archive_schema="forge.ugi_chemistry_specialist_comparison_archive.v4",
        gate_dependency_stage="comparison_preflight",
    )


@stage("evaluate.ugi.joint-lipid-specialist-base-comparison-preflight.v5")
def evaluate_ugi_joint_lipid_specialist_base_comparison_preflight_stage(
    context: RunContext,
) -> StageResult:
    """Gate joint-prior Ugi adaptation on the paired measured-morphology sampler."""

    return _ugi_chemistry_specialist_base_comparison_stage(
        context,
        dependency_stage="preflight",
        result_schema="forge.ugi_joint_lipid_specialist_comparison_result.v5",
        archive_schema="forge.ugi_joint_lipid_specialist_comparison_archive.v5",
        run_profile="h100_preflight",
    )


@stage("evaluate.ugi.joint-lipid-specialist-base-comparison.v5")
def evaluate_ugi_joint_lipid_specialist_base_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Compare the full-output joint-prior Ugi adapter with its exact frozen base."""

    return _ugi_chemistry_specialist_base_comparison_stage(
        context,
        dependency_stage="specialize_ugi",
        result_schema="forge.ugi_joint_lipid_specialist_comparison_result.v5",
        archive_schema="forge.ugi_joint_lipid_specialist_comparison_archive.v5",
        gate_dependency_stage="comparison_preflight",
    )


@stage("evaluate.ugi.role-local-specialist-base-comparison-preflight.v6")
def evaluate_ugi_role_local_specialist_base_comparison_preflight_stage(
    context: RunContext,
) -> StageResult:
    """Gate role-local Ugi decoding on the paired measured-morphology sampler."""

    return _ugi_chemistry_specialist_base_comparison_stage(
        context,
        dependency_stage="preflight",
        result_schema="forge.ugi_role_local_specialist_comparison_result.v6",
        archive_schema="forge.ugi_role_local_specialist_comparison_archive.v6",
        run_profile="h100_preflight",
    )


@stage("evaluate.ugi.role-local-specialist-base-comparison.v6")
def evaluate_ugi_role_local_specialist_base_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Compare the role-local Ugi tree decoder with its exact frozen base."""

    return _ugi_chemistry_specialist_base_comparison_stage(
        context,
        dependency_stage="specialize_ugi",
        result_schema="forge.ugi_role_local_specialist_comparison_result.v6",
        archive_schema="forge.ugi_role_local_specialist_comparison_archive.v6",
        gate_dependency_stage="comparison_preflight",
    )


@stage("evaluate.ugi.measured-only-full-model-base-comparison-preflight.v7")
def evaluate_ugi_measured_only_full_model_base_comparison_preflight_stage(
    context: RunContext,
) -> StageResult:
    """Gate the measured-only full-model diagnostic on its paired sampler."""

    return _ugi_chemistry_specialist_base_comparison_stage(
        context,
        dependency_stage="preflight",
        result_schema="forge.ugi_measured_only_full_model_comparison_result.v7",
        archive_schema="forge.ugi_measured_only_full_model_comparison_archive.v7",
        run_profile="h100_preflight",
    )


@stage("evaluate.ugi.measured-only-full-model-base-comparison.v7")
def evaluate_ugi_measured_only_full_model_base_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Compare the measured-only full-model fine-tune with its exact frozen base."""

    return _ugi_chemistry_specialist_base_comparison_stage(
        context,
        dependency_stage="specialize_ugi",
        result_schema="forge.ugi_measured_only_full_model_comparison_result.v7",
        archive_schema="forge.ugi_measured_only_full_model_comparison_archive.v7",
        gate_dependency_stage="comparison_preflight",
    )


@stage("evaluate.ugi.structured-topology-specialist-base-comparison-preflight.v8")
def evaluate_ugi_structured_topology_specialist_base_comparison_preflight_stage(
    context: RunContext,
) -> StageResult:
    """Gate grammar-normalized Ugi topology scoring on paired measured morphologies."""

    return _ugi_chemistry_specialist_base_comparison_stage(
        context,
        dependency_stage="preflight",
        result_schema="forge.ugi_structured_topology_specialist_comparison_result.v8",
        archive_schema="forge.ugi_structured_topology_specialist_comparison_archive.v8",
        run_profile="h100_preflight",
    )


@stage("evaluate.ugi.structured-topology-specialist-base-comparison.v8")
def evaluate_ugi_structured_topology_specialist_base_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Compare the structured Ugi topology head with its exact frozen base."""

    return _ugi_chemistry_specialist_base_comparison_stage(
        context,
        dependency_stage="specialize_ugi",
        result_schema="forge.ugi_structured_topology_specialist_comparison_result.v8",
        archive_schema="forge.ugi_structured_topology_specialist_comparison_archive.v8",
        run_profile="h100_preflight",
    )
