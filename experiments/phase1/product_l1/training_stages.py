"""Product/L1 training stage adapters and frozen ablation configuration."""

from __future__ import annotations

import copy
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
from experiments.phase1.product_l1._stage_support import (
    _copy,
    _publish_training_outputs,
    _resume_training_work,
)
from forge.core.io import write_json

UGI_TREE_ABLATION_SCHEMA = "forge.ugi_tree_transformer_ablation_ladder.v1"


def _tree_ablation_effective_config(
    context: RunContext, design: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Resolve one matched ablation arm without duplicating full training configurations."""

    if design.get("schema_version") != UGI_TREE_ABLATION_SCHEMA:
        raise StageError("unsupported Ugi tree-Transformer ablation design")
    profile_orders = design.get("profile_arm_order")
    arms = design.get("arms")
    if not isinstance(profile_orders, dict) or not isinstance(arms, dict):
        raise StageError("Ugi tree-Transformer ablation design is incomplete")
    order = profile_orders.get(context.profile)
    if (
        not isinstance(order, list)
        or not order
        or any(not isinstance(value, str) or value not in arms for value in order)
        or len(set(order)) != len(order)
        or context.replicate >= len(order)
    ):
        raise StageError("profile replicate does not map to one declared ablation arm")
    arm_id = order[context.replicate]

    base_pin = design.get("base_config")
    declared_base = context.stage.inputs.get("base_config")
    if not isinstance(base_pin, dict) or declared_base is None:
        raise StageError("ablation design does not bind its base configuration")
    if base_pin != declared_base.to_mapping():
        raise StageError("ablation design and experiment disagree on the base configuration")
    base = json.loads(context.input("base_config").read_text())
    configured_inputs = base.get("inputs")
    chemistry_inputs = {
        label: pin.to_mapping()
        for label, pin in context.stage.inputs.items()
        if label != "base_config"
    }
    if configured_inputs != chemistry_inputs:
        raise StageError("ablation base config and experiment chemistry inputs differ")

    matched = design.get("matched_contract")
    full = base.get("full")
    duration = base.get("duration_contract")
    if (
        not isinstance(matched, dict)
        or not isinstance(full, dict)
        or not isinstance(duration, dict)
    ):
        raise StageError("ablation matched-duration contract is incomplete")
    if (
        int(full.get("steps", -1)) != int(matched.get("optimizer_steps", -2))
        or int(full.get("batch_size", -1)) != int(matched.get("batch_size", -2))
        or duration.get("maximum_weighted_training_draws") != matched.get("weighted_training_draws")
        or full.get("checkpoint_steps") != matched.get("checkpoint_steps")
        or base.get("seed") != matched.get("initialization_and_minibatch_seed")
        or base.get("promotion_contract", {}).get("heldout_selects_architecture_or_checkpoint")
        is not False
        or base.get("promotion_contract", {}).get(
            "development_training_authorized_after_implementation_review"
        )
        is not True
    ):
        raise StageError("ablation base config violates the matched development contract")

    arm = arms[arm_id]
    if not isinstance(arm, dict) or set(arm) != {
        "scientific_question",
        "model_overrides",
        "objective_overrides",
    }:
        raise StageError(f"ablation arm {arm_id!r} is malformed")
    effective = copy.deepcopy(base)
    effective["task"] = f"Ugi tree-Transformer matched development arm: {arm_id}"
    effective["experiment_arm"] = {
        "arm_id": arm_id,
        "design_sha256": context.stage.config.sha256,
        "replicate": context.replicate,
        "scientific_question": arm["scientific_question"],
    }
    effective["model"].update(dict(arm["model_overrides"]))
    effective["objective"] = copy.deepcopy(arm["objective_overrides"])
    profile_overrides = design.get("execution_profile_overrides", {}).get(context.profile, {})
    if not isinstance(profile_overrides, dict):
        raise StageError("ablation execution-profile override is malformed")
    effective[context.profile].update(copy.deepcopy(profile_overrides))
    return arm_id, effective


@stage("generate.ugi.joint-train.v1")
def train_ugi_joint_stage(context: RunContext) -> StageResult:
    """Run the frozen all-fold joint-flow contract with deterministic resume state."""

    from experiments.phase1.product_l1.training.ugi_joint_sparse_training import (
        train_ugi_joint_sparse,
    )

    context.dependency("cache", "receipt")
    config = context.config()
    require_config_inputs(context, config)
    runtime = config.get(context.profile)
    if not isinstance(runtime, dict) or runtime.get("device") != context.resources.device:
        raise StageError("joint training config and declared device differ")
    work = context.work_dir / "joint_training"
    resume_training = _resume_training_work(context, work)
    result = train_ugi_joint_sparse(
        context.config_path,
        context.repo,
        work,
        smoke=context.profile == "smoke",
        overwrite=False,
        resume=resume_training,
    )
    artifacts = _publish_training_outputs(
        context,
        work,
        result_schema="phase1_ugi_joint_sparse_training_result.v1",
        checkpoint_schema="phase1_ugi_joint_sparse_checkpoint.v1",
        progress_schema="phase1_ugi_joint_sparse_progress.v1",
    )
    return StageResult(
        artifacts=artifacts,
        metrics={
            "completed_steps": int(result["selection"]["completed_steps"]),
            "training_records": int(result["training_partition"]["training_records"]),
        },
        summary={
            "route_guidance": False,
            "oracle_guidance": False,
            "selection_mode": result["selection"]["mode"],
        },
    )


@stage("generate.ugi.joint-ablation-train.v1")
def train_ugi_joint_ablation_stage(context: RunContext) -> StageResult:
    """Materialize and train one replicate-indexed tree-Transformer ablation arm."""

    from experiments.phase1.product_l1.training.ugi_joint_sparse_training import (
        train_ugi_joint_sparse,
    )

    context.dependency("cache", "receipt")
    arm_id, effective = _tree_ablation_effective_config(context, context.config())
    runtime = effective.get(context.profile)
    if not isinstance(runtime, dict) or runtime.get("device") != context.resources.device:
        raise StageError("ablation training config and declared device differ")
    # Keep the generated config beside, rather than inside, the trainer output directory.  The
    # trainer deliberately refuses a nonempty fresh output directory; putting its own input config
    # there made a first run look like stale training state before step zero.
    work = context.work_dir / "joint_ablation_training"
    resume_training = _resume_training_work(context, work)
    effective_config_path = context.work_dir / "joint_ablation_effective_config.json"
    expected_effective = json.dumps(effective, indent=2, sort_keys=True) + "\n"
    if effective_config_path.is_file():
        if effective_config_path.read_text() != expected_effective:
            raise StageError("persisted effective ablation config changed")
    else:
        write_json(effective_config_path, effective)
    result = train_ugi_joint_sparse(
        effective_config_path,
        context.repo,
        work,
        smoke=context.profile == "smoke",
        overwrite=False,
        resume=resume_training,
    )
    artifacts = _publish_training_outputs(
        context,
        work,
        result_schema="phase1_ugi_joint_sparse_training_result.v1",
        checkpoint_schema="phase1_ugi_joint_sparse_checkpoint.v1",
        progress_schema="phase1_ugi_joint_sparse_progress.v1",
    )
    _copy(effective_config_path, context.output_path("effective_config.json"))
    return StageResult(
        artifacts=(
            *artifacts,
            ProducedArtifact(
                "effective_config",
                "effective_config.json",
                "phase1_ugi_joint_sparse_training_config.v1",
            ),
        ),
        metrics={
            "completed_steps": int(result["selection"]["completed_steps"]),
            "training_records": int(result["training_partition"]["training_records"]),
        },
        summary={
            "ablation_arm": arm_id,
            "candidate_selection": False,
            "heldout_selects_nothing": True,
            "route_guidance": False,
            "oracle_guidance": False,
        },
    )


@stage("generate.ugi.selected-tree-production-train.v1")
def train_selected_ugi_tree_production_stage(context: RunContext) -> StageResult:
    """Train one fresh seed of the calibration-selected Ugi tree Transformer."""

    from experiments.phase1.product_l1.training.ugi_joint_sparse_training import (
        train_ugi_joint_sparse,
    )
    from experiments.phase1.product_l1.training.ugi_tree_transformer_production import (
        build_production_training_config,
    )

    context.dependency("cache", "receipt")
    design = context.config()
    configured = design.get("inputs")
    if not isinstance(configured, dict):
        raise StageError("selected-tree production design has no inputs")
    for label, expected in configured.items():
        pin = context.stage.inputs.get(label)
        if pin is None or pin.to_mapping() != expected:
            raise StageError(f"selected-tree production input changed: {label}")
        context.input(label)
    base = json.loads(context.input("base_config").read_text())
    ablation = json.loads(context.input("ablation_design").read_text())
    adjudication = json.loads(context.input("calibration_adjudication").read_text())
    chemistry_labels = {
        "assignments",
        "semantic_products",
        "semantic_atoms",
        "atom_vocabulary",
        "prepared_cache",
    }
    if set(base.get("inputs", {})) != chemistry_labels or any(
        base["inputs"][label] != configured.get(label) for label in chemistry_labels
    ):
        raise StageError("selected-tree base config and production chemistry inputs differ")
    training_seed, effective = build_production_training_config(
        design,
        base,
        ablation,
        adjudication,
        profile=context.profile,
        replicate=context.replicate,
    )
    runtime = effective.get(context.profile)
    if not isinstance(runtime, dict) or runtime.get("device") != context.resources.device:
        raise StageError("selected-tree training config and declared device differ")
    work = context.work_dir / "selected_tree_production_training"
    resume_training = _resume_training_work(context, work)
    effective_config_path = context.work_dir / "selected_tree_effective_config.json"
    expected_effective = json.dumps(effective, indent=2, sort_keys=True) + "\n"
    if effective_config_path.is_file():
        if effective_config_path.read_text() != expected_effective:
            raise StageError("persisted selected-tree effective config changed")
    else:
        write_json(effective_config_path, effective)
    result = train_ugi_joint_sparse(
        effective_config_path,
        context.repo,
        work,
        smoke=context.profile == "smoke",
        overwrite=False,
        resume=resume_training,
    )
    artifacts = _publish_training_outputs(
        context,
        work,
        result_schema="phase1_ugi_joint_sparse_training_result.v1",
        checkpoint_schema="phase1_ugi_joint_sparse_checkpoint.v1",
        progress_schema="phase1_ugi_joint_sparse_progress.v1",
    )
    _copy(effective_config_path, context.output_path("effective_config.json"))
    return StageResult(
        artifacts=(
            *artifacts,
            ProducedArtifact(
                "effective_config",
                "effective_config.json",
                "phase1_ugi_joint_sparse_training_config.v1",
            ),
        ),
        metrics={
            "completed_steps": int(result["selection"]["completed_steps"]),
            "training_records": int(result["training_partition"]["training_records"]),
            "training_seed": training_seed,
        },
        summary={
            "arm_id": "tree_relations_and_routing",
            "fixed_final_step": int(runtime["steps"]),
            "heldout_selects_nothing": True,
            "route_guidance": False,
            "oracle_guidance": False,
        },
    )


@stage("generate.ugi.closure-train.v1")
def train_ugi_closure_stage(context: RunContext) -> StageResult:
    """Train the sparse closure scorer with resumable optimizer and RNG state."""

    from experiments.phase1.product_l1.training.ugi_closure_training import train_ugi_closure_scorer

    config = context.config()
    require_config_inputs(context, config)
    runtime = config.get(context.profile)
    if not isinstance(runtime, dict) or runtime.get("device") != context.resources.device:
        raise StageError("closure training config and declared device differ")
    work = context.work_dir / "closure_training"
    resume_training = _resume_training_work(context, work)
    result = train_ugi_closure_scorer(
        context.config_path,
        context.repo,
        work,
        smoke=context.profile == "smoke",
        overwrite=False,
        resume=resume_training,
    )
    artifacts = _publish_training_outputs(
        context,
        work,
        result_schema="phase1_ugi_sparse_closure_result.v2",
        checkpoint_schema="phase1_ugi_sparse_closure_checkpoint.v2",
        progress_schema="phase1_ugi_sparse_closure_progress.v1",
    )
    return StageResult(
        artifacts=artifacts,
        metrics={
            "best_step": int(result["selection"]["best_step"]),
            "completed_steps": int(result["selection"]["completed_steps"]),
        },
        summary={"selection_metric": result["selection"]["metric"]},
    )
