"""Checkpoint and terminal chemistry evaluation stage adapters."""

from __future__ import annotations

import json

from experiments._runtime.errors import StageError
from experiments._runtime.registry import stage
from experiments._runtime.stage import (
    ProducedArtifact,
    RunContext,
    StageResult,
    require_config_inputs,
)
from experiments.phase1.product_l1._stage_support import _copy, _deterministic_tar


@stage("evaluate.ugi.tree-transformer-checkpoints.v1")
def evaluate_ugi_tree_transformer_checkpoints_stage(context: RunContext) -> StageResult:
    """Sample and assess every frozen checkpoint from one matched training arm."""

    from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_checkpoint_calibration import (  # noqa: E501
        STATIC_INPUT_LABELS,
        run_checkpoint_calibration,
    )

    config = context.config()
    configured = config.get("inputs")
    if not isinstance(configured, dict):
        raise StageError("checkpoint calibration config has no static inputs")
    for label in STATIC_INPUT_LABELS:
        pin = context.stage.inputs.get(label)
        if pin is None or configured.get(label) != pin.to_mapping():
            raise StageError(f"checkpoint calibration input changed: {label}")
        context.input(label)
    for label in ("training_result", "checkpoint_snapshots"):
        if label not in context.stage.inputs:
            raise StageError(f"checkpoint calibration lacks dynamic input: {label}")
    work = context.work_dir / "checkpoint_calibration"
    result = run_checkpoint_calibration(
        context.config_path,
        context.repo,
        work,
        training_result_path=context.input("training_result"),
        checkpoint_archive_path=context.input("checkpoint_snapshots"),
        profile=context.profile,
        resume=context.resume,
        device=context.resources.device,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [
        path
        for path in work.rglob("*")
        if path.is_file() and "checkpoints" not in path.relative_to(work).parts
    ]
    _deterministic_tar(
        detail_files,
        context.output_path("checkpoint_calibrations.tar"),
        base=work,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_tree_transformer_checkpoint_calibration.v1",
            ),
            ProducedArtifact(
                "checkpoint_calibrations",
                "checkpoint_calibrations.tar",
                "forge.ugi_tree_transformer_checkpoint_calibration_archive.v1",
                rows=len(result["checkpoint_steps"]),
            ),
        ),
        metrics={
            "checkpoints": len(result["checkpoint_steps"]),
            "programs_per_checkpoint": int(result["programs_per_checkpoint"]),
        },
        summary={
            "arm_id": result["arm_id"],
            "checkpoint_selection": "deferred_to_cross_arm_v0_adjudicator",
            "candidate_selection": False,
            "heldout_rows_used": False,
        },
    )


@stage("evaluate.ugi.selected-tree-production.v1")
def evaluate_selected_ugi_tree_production_stage(context: RunContext) -> StageResult:
    """Evaluate one fresh selected-tree seed on the frozen component-family stress draw."""

    from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_production import (
        STATIC_INPUT_LABELS,
        run_tree_transformer_production_evaluation,
    )

    config = context.config()
    configured = config.get("inputs")
    if not isinstance(configured, dict):
        raise StageError("selected-tree production evaluation has no static inputs")
    for label in STATIC_INPUT_LABELS:
        pin = context.stage.inputs.get(label)
        if pin is None or configured.get(label) != pin.to_mapping():
            raise StageError(f"selected-tree evaluation input changed: {label}")
        context.input(label)
    checkpoint = context.dependency("train", "checkpoint_latest")
    effective_config = context.dependency("train", "effective_config")
    training_result = context.dependency("train", "result")
    effective = json.loads(effective_config.path.read_text())
    training_seed = int(effective.get("seed", -1))
    work = context.work_dir / "selected_tree_production_evaluation"
    result = run_tree_transformer_production_evaluation(
        context.config_path,
        context.repo,
        work,
        checkpoint_path=checkpoint.path,
        effective_training_config_path=effective_config.path,
        training_result_path=training_result.path,
        training_seed=training_seed,
        profile=context.profile,
        resume=context.resume,
        device=context.resources.device,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [path for path in (work / "details").rglob("*") if path.is_file()]
    _deterministic_tar(
        detail_files,
        context.output_path("evaluation_details.tar"),
        base=work,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_tree_transformer_production_evaluation.v1",
            ),
            ProducedArtifact(
                "evaluation_details",
                "evaluation_details.tar",
                "forge.ugi_tree_transformer_production_evaluation_archive.v1",
                rows=int(result["programs"]),
            ),
        ),
        metrics={
            "programs": int(result["programs"]),
            "training_seed": training_seed,
            "exact_l1_yield_per_attempt": float(result["metrics"]["exact_l1_yield_per_attempt"]),
        },
        summary={
            "arm_id": result["arm_id"],
            "checkpoint_step": int(result["checkpoint_step"]),
            "candidate_selection": False,
            "heldout_rows_used": True,
            "heldout_selects_nothing": True,
        },
    )


@stage("evaluate.ugi.tree-relational-edge-constrained-resampling.v1")
def evaluate_ugi_tree_relational_edge_constrained_resampling_stage(
    context: RunContext,
) -> StageResult:
    """Resample one frozen tree-Transformer checkpoint with role-edge masks."""

    from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_constrained_resampling import (  # noqa: E501
        run_tree_transformer_constrained_resampling,
    )

    config = context.config()
    require_config_inputs(context, config)
    work = context.work_dir / "tree_relational_edge_constrained_resampling"
    result_path = work / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
    else:
        result = run_tree_transformer_constrained_resampling(
            context.config_path,
            context.repo,
            work,
            profile=context.profile,
            device=context.resources.device,
        )
    if (
        result.get("schema_version") != "forge.ugi_tree_transformer_constrained_resampling.v1"
        or result.get("status") != "complete"
        or int(result.get("attempts", -1)) != 3072
        or result.get("retraining") is not False
        or result.get("repairs_or_retries") is not False
    ):
        raise StageError("constrained resampling did not satisfy the frozen full-run contract")

    _copy(result_path, context.output_path("result.json"))
    detail_files = [path for path in work.rglob("*") if path.is_file() and path != result_path]
    _deterministic_tar(
        detail_files,
        context.output_path("diagnostic_details.tar"),
        base=work,
    )
    metrics = result["metrics"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_tree_transformer_constrained_resampling.v1",
                rows=int(result["attempts"]),
            ),
            ProducedArtifact(
                "diagnostic_details",
                "diagnostic_details.tar",
                "forge.ugi_tree_transformer_constrained_resampling_archive.v1",
                rows=int(result["attempts"]),
            ),
        ),
        metrics={
            "attempts": int(result["attempts"]),
            "valid_fraction_per_attempt": float(metrics["valid_fraction_per_attempt"]),
            "exact_l1_yield_per_attempt": float(metrics["exact_l1_yield_per_attempt"]),
            "local_support_qualified_exact_l1_yield_per_attempt": float(
                metrics["local_support_qualified_exact_l1_yield_per_attempt"]
            ),
            "local_unsupported_exact_l1": int(
                result["failure_counts"]["local_unsupported_exact_l1"]
            ),
        },
        summary={
            "all_gates_pass": bool(result["all_gates_pass"]),
            "decision": str(result["decision"]),
            "frozen_checkpoint_reused": True,
            "training_calls": 0,
            "repairs_or_retries": False,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("evaluate.ugi.v0-calibration-checkpoint.v1")
def evaluate_ugi_v0_calibration_checkpoint_stage(context: RunContext) -> StageResult:
    """Sample the frozen v0 checkpoint under the exact tree-Transformer calibration contract."""

    from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_checkpoint_calibration import (  # noqa: E501
        STATIC_INPUT_LABELS,
        run_reference_checkpoint_calibration,
    )

    config = context.config()
    configured = config.get("inputs")
    if not isinstance(configured, dict):
        raise StageError("v0 calibration config has no static inputs")
    for label in STATIC_INPUT_LABELS:
        pin = context.stage.inputs.get(label)
        if pin is None or configured.get(label) != pin.to_mapping():
            raise StageError(f"v0 calibration input changed: {label}")
        context.input(label)
    if "reference_checkpoint" not in context.stage.inputs:
        raise StageError("v0 calibration lacks its reference checkpoint")
    work = context.work_dir / "v0_checkpoint_calibration"
    result = run_reference_checkpoint_calibration(
        context.config_path,
        context.repo,
        work,
        checkpoint_path=context.input("reference_checkpoint"),
        profile=context.profile,
        resume=context.resume,
        device=context.resources.device,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [path for path in work.rglob("*") if path.is_file()]
    _deterministic_tar(
        detail_files,
        context.output_path("checkpoint_calibrations.tar"),
        base=work,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_reference_checkpoint_calibration.v1",
            ),
            ProducedArtifact(
                "checkpoint_calibrations",
                "checkpoint_calibrations.tar",
                "forge.ugi_tree_transformer_checkpoint_calibration_archive.v1",
                rows=1,
            ),
        ),
        metrics={
            "checkpoints": 1,
            "programs_per_checkpoint": int(result["programs_per_checkpoint"]),
        },
        summary={
            "arm_id": "v0_reference",
            "checkpoint_selection": False,
            "candidate_selection": False,
            "heldout_rows_used": False,
        },
    )


@stage("evaluate.ugi.v0-current-program-comparison.v1")
def evaluate_ugi_v0_current_program_comparison_stage(context: RunContext) -> StageResult:
    """Run the frozen v0 and mixed Transformer on one ordered Ugi program draw."""

    from experiments.phase1.product_l1.evaluation.ugi_v0_current_program_comparison import (
        run_ugi_v0_current_program_comparison,
    )

    config = context.config()
    require_config_inputs(context, config)
    work = context.work_dir / "ugi_v0_current_program_comparison"
    result = run_ugi_v0_current_program_comparison(
        context.config_path,
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
    current = result["methods"]["forge_mixed_transformer_program_matched_seed0"]["metrics"]
    reference = result["methods"]["forge_v0_step_3000_program_matched"]["metrics"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_v0_current_program_comparison.v1",
            ),
            ProducedArtifact(
                "comparison_details",
                "comparison_details.tar",
                "forge.ugi_v0_current_program_comparison_archive.v1",
                rows=int(result["programs_per_method"]),
            ),
        ),
        metrics={
            "programs_per_method": int(result["programs_per_method"]),
            "current_exact_l1": float(current["exact_l1_yield_per_attempt"]),
            "v0_exact_l1": float(reference["exact_l1_yield_per_attempt"]),
            "current_open_ended_exact_l1": float(
                current["unique_open_ended_whole_product_novel_exact_l1_products_per_attempt"]
            ),
            "v0_open_ended_exact_l1": float(
                reference["unique_open_ended_whole_product_novel_exact_l1_products_per_attempt"]
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


@stage("evaluate.ugi.transformer-morphology-projection-comparison.v1")
def evaluate_ugi_transformer_morphology_projection_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Compare both frozen mixed Transformers on one ordered full Ugi program draw."""

    from experiments.phase1.product_l1.evaluation.ugi_transformer_morphology_projection_comparison import (
        run_ugi_transformer_morphology_projection_comparison,
    )

    config = context.config()
    require_config_inputs(context, config)
    work = context.work_dir / "ugi_transformer_morphology_projection_comparison"
    result = run_ugi_transformer_morphology_projection_comparison(
        context.config_path,
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
    reduced_id = "forge_mixed_transformer_reduced_projection_seed0"
    full_id = "forge_mixed_transformer_full_role_morphology_seed0"
    reduced = result["methods"][reduced_id]["metrics"]
    full = result["methods"][full_id]["metrics"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_transformer_morphology_projection_comparison.v1",
            ),
            ProducedArtifact(
                "comparison_details",
                "comparison_details.tar",
                "forge.ugi_transformer_morphology_projection_comparison_archive.v1",
                rows=int(result["programs_per_method"]),
            ),
        ),
        metrics={
            "programs_per_method": int(result["programs_per_method"]),
            "reduced_exact_l1": float(reduced["exact_l1_yield_per_attempt"]),
            "full_exact_l1": float(full["exact_l1_yield_per_attempt"]),
            "exact_l1_delta": float(result["full_minus_reduced"]["exact_l1_yield_per_attempt"]),
        },
        summary={
            "decision": str(result["decision"]),
            "training_calls": 0,
            "repairs_or_retries": False,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("evaluate.ugi.role-chemistry-prior-comparison.v1")
def evaluate_ugi_role_chemistry_prior_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Calibrate identity-free train-fold local chemistry statistics at terminal decode."""

    from experiments.phase1.product_l1.evaluation.ugi_role_chemistry_prior_comparison import (
        run_ugi_role_chemistry_prior_comparison,
    )

    config = context.config()
    require_config_inputs(context, config)
    run_profile = "h100_preflight" if context.resources.device == "cuda" else "smoke"
    work = context.work_dir / "ugi_role_chemistry_prior_comparison"
    result = run_ugi_role_chemistry_prior_comparison(
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
    methods = list(result["methods"].values())
    reference = next(value for value in methods if float(value["strength"]) == 0.0)
    estimated = [value for value in methods if value["metrics"].get("realism_c2st_auc") is not None]
    best = min(estimated, key=lambda value: float(value["metrics"]["realism_c2st_auc"]))
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_role_chemistry_prior_comparison_result.v1",
            ),
            ProducedArtifact(
                "comparison_details",
                "comparison_details.tar",
                "forge.ugi_role_chemistry_prior_comparison_archive.v1",
                rows=int(result["programs_per_method"]) * len(methods),
            ),
        ),
        metrics={
            "programs_per_method": int(result["programs_per_method"]),
            "lambda_zero_exact_l1": float(reference["metrics"]["exact_l1_yield_per_attempt"]),
            "lambda_zero_realism_c2st": float(reference["metrics"]["realism_c2st_auc"]),
            "best_observed_realism_c2st": float(best["metrics"]["realism_c2st_auc"]),
        },
        summary={
            "selection_decision": str(result["selection_decision"]),
            "selected_strength": result["selected_strength"],
            "training_calls": 0,
            "repairs_or_retries": False,
            "component_identity_conditioning": False,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


def _ugi_terminal_chemistry_comparison_stage(
    context: RunContext,
    *,
    flow_comparison: bool,
    factorization_comparison: bool = False,
) -> StageResult:
    from experiments.phase1.product_l1.evaluation.ugi_terminal_chemistry_temperature_comparison import (
        run_ugi_learned_topology_then_chemistry_comparison,
        run_ugi_terminal_chemistry_temperature_comparison,
        run_ugi_topology_conditioned_chemistry_flow_comparison,
    )

    config = context.config()
    require_config_inputs(context, config)
    run_profile = "h100_preflight" if context.resources.device == "cuda" else "smoke"
    if factorization_comparison:
        stem = "ugi_learned_topology_then_chemistry_comparison"
        runner = run_ugi_learned_topology_then_chemistry_comparison
        result_schema = "forge.ugi_learned_topology_then_chemistry_comparison_result.v1"
        archive_schema = "forge.ugi_learned_topology_then_chemistry_comparison_archive.v1"
    elif flow_comparison:
        stem = "ugi_topology_conditioned_chemistry_flow_comparison"
        runner = run_ugi_topology_conditioned_chemistry_flow_comparison
        result_schema = "forge.ugi_topology_conditioned_chemistry_flow_comparison_result.v1"
        archive_schema = "forge.ugi_topology_conditioned_chemistry_flow_comparison_archive.v1"
    else:
        stem = "ugi_terminal_chemistry_temperature_comparison"
        runner = run_ugi_terminal_chemistry_temperature_comparison
        result_schema = "forge.ugi_terminal_chemistry_temperature_comparison_result.v1"
        archive_schema = "forge.ugi_terminal_chemistry_temperature_comparison_archive.v1"
    work = context.work_dir / stem
    result = runner(
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
    methods = list(result["methods"].values())
    reference = result["methods"][str(result["reference_method"])]
    estimated = [value for value in methods if value["metrics"].get("realism_c2st_auc") is not None]
    best = min(estimated, key=lambda value: float(value["metrics"]["realism_c2st_auc"]))
    reference_prefix = "constructive_topology" if factorization_comparison else "argmax"
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
                rows=int(result["programs_per_method"]) * len(methods),
            ),
        ),
        metrics={
            "programs_per_method": int(result["programs_per_method"]),
            f"{reference_prefix}_exact_l1": float(
                reference["metrics"]["exact_l1_yield_per_attempt"]
            ),
            f"{reference_prefix}_realism_c2st": float(reference["metrics"]["realism_c2st_auc"]),
            "best_observed_realism_c2st": float(best["metrics"]["realism_c2st_auc"]),
        },
        summary={
            "selection_decision": str(result["selection_decision"]),
            (
                "selected_topology_conditioned_chemistry_steps"
                if flow_comparison and not factorization_comparison
                else (
                    "selected_sampling_factorization"
                    if factorization_comparison
                    else "selected_temperature"
                )
            ): result[
                (
                    "selected_topology_conditioned_chemistry_steps"
                    if flow_comparison and not factorization_comparison
                    else (
                        "selected_sampling_factorization"
                        if factorization_comparison
                        else "selected_temperature"
                    )
                )
            ],
            "training_calls": 0,
            "repairs_or_retries": False,
            "component_identity_conditioning": False,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("evaluate.ugi.terminal-chemistry-temperature-comparison.v1")
def evaluate_ugi_terminal_chemistry_temperature_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Calibrate sequential masked chemistry draws against measured Ugi structures."""

    return _ugi_terminal_chemistry_comparison_stage(context, flow_comparison=False)


@stage("evaluate.ugi.topology-conditioned-chemistry-flow-comparison.v1")
def evaluate_ugi_topology_conditioned_chemistry_flow_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Compare one-pass chemistry with repeated denoising on one constructed Ugi topology."""

    return _ugi_terminal_chemistry_comparison_stage(context, flow_comparison=True)


@stage("evaluate.ugi.learned-topology-then-chemistry-comparison.v1")
def evaluate_ugi_learned_topology_then_chemistry_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Compare learned topology flow with the frozen constructive Ugi topology decoder."""

    return _ugi_terminal_chemistry_comparison_stage(
        context,
        flow_comparison=True,
        factorization_comparison=True,
    )
