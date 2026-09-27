"""Compatible registration entry point and corpus adapters for product/L1 workflows."""

from __future__ import annotations

import gc

from experiments._runtime.errors import StageError
from experiments._runtime.registry import stage
from experiments._runtime.stage import (
    ProducedArtifact,
    RunContext,
    StageResult,
    require_config_inputs,
)
from experiments.phase1.product_l1._stage_support import _copy as _copy
from experiments.phase1.product_l1._stage_support import _deterministic_tar as _deterministic_tar
from experiments.phase1.product_l1._stage_support import (
    _publish_training_outputs as _publish_training_outputs,
)
from experiments.phase1.product_l1._stage_support import (
    _resume_training_work as _resume_training_work,
)
from experiments.phase1.product_l1.evaluation_stages import (
    _ugi_terminal_chemistry_comparison_stage as _ugi_terminal_chemistry_comparison_stage,
)
from experiments.phase1.product_l1.evaluation_stages import (
    evaluate_selected_ugi_tree_production_stage as evaluate_selected_ugi_tree_production_stage,
)
from experiments.phase1.product_l1.evaluation_stages import (
    evaluate_ugi_learned_topology_then_chemistry_comparison_stage as evaluate_ugi_learned_topology_then_chemistry_comparison_stage,
)
from experiments.phase1.product_l1.evaluation_stages import (
    evaluate_ugi_role_chemistry_prior_comparison_stage as evaluate_ugi_role_chemistry_prior_comparison_stage,
)
from experiments.phase1.product_l1.evaluation_stages import (
    evaluate_ugi_terminal_chemistry_temperature_comparison_stage as evaluate_ugi_terminal_chemistry_temperature_comparison_stage,
)
from experiments.phase1.product_l1.evaluation_stages import (
    evaluate_ugi_topology_conditioned_chemistry_flow_comparison_stage as evaluate_ugi_topology_conditioned_chemistry_flow_comparison_stage,
)
from experiments.phase1.product_l1.evaluation_stages import (
    evaluate_ugi_transformer_morphology_projection_comparison_stage as evaluate_ugi_transformer_morphology_projection_comparison_stage,
)
from experiments.phase1.product_l1.evaluation_stages import (
    evaluate_ugi_tree_relational_edge_constrained_resampling_stage as evaluate_ugi_tree_relational_edge_constrained_resampling_stage,
)
from experiments.phase1.product_l1.evaluation_stages import (
    evaluate_ugi_tree_transformer_checkpoints_stage as evaluate_ugi_tree_transformer_checkpoints_stage,
)
from experiments.phase1.product_l1.evaluation_stages import (
    evaluate_ugi_v0_calibration_checkpoint_stage as evaluate_ugi_v0_calibration_checkpoint_stage,
)
from experiments.phase1.product_l1.evaluation_stages import (
    evaluate_ugi_v0_current_program_comparison_stage as evaluate_ugi_v0_current_program_comparison_stage,
)
from experiments.phase1.product_l1.sampling_stages import sample_ugi_shards as sample_ugi_shards
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    _require_successful_ugi_all_role_semantic_preflight as _require_successful_ugi_all_role_semantic_preflight,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    _require_successful_ugi_all_role_semantic_quantitative_preflight as _require_successful_ugi_all_role_semantic_quantitative_preflight,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    _require_successful_ugi_amine_semantic_program_preflight as _require_successful_ugi_amine_semantic_program_preflight,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    _require_successful_ugi_group_balanced_program_prior_preflight as _require_successful_ugi_group_balanced_program_prior_preflight,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    _require_successful_ugi_morphology_diversity_preflight as _require_successful_ugi_morphology_diversity_preflight,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    _ugi_all_role_semantic_adjudication_stage as _ugi_all_role_semantic_adjudication_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    _ugi_all_role_semantic_program_comparison_stage as _ugi_all_role_semantic_program_comparison_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    _ugi_amine_semantic_program_comparison_stage as _ugi_amine_semantic_program_comparison_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    _ugi_group_balanced_program_prior_comparison_stage as _ugi_group_balanced_program_prior_comparison_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    _ugi_morphology_diversity_comparison_stage as _ugi_morphology_diversity_comparison_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    evaluate_ugi_all_role_semantic_adjudication_stage as evaluate_ugi_all_role_semantic_adjudication_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    evaluate_ugi_all_role_semantic_preflight_adjudication_stage as evaluate_ugi_all_role_semantic_preflight_adjudication_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    evaluate_ugi_all_role_semantic_program_comparison_preflight_stage as evaluate_ugi_all_role_semantic_program_comparison_preflight_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    evaluate_ugi_all_role_semantic_program_comparison_quantitative_gated_stage as evaluate_ugi_all_role_semantic_program_comparison_quantitative_gated_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    evaluate_ugi_all_role_semantic_program_comparison_stage as evaluate_ugi_all_role_semantic_program_comparison_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    evaluate_ugi_all_role_semantic_visual_review_stage as evaluate_ugi_all_role_semantic_visual_review_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    evaluate_ugi_amine_semantic_joint_support_adjudication_stage as evaluate_ugi_amine_semantic_joint_support_adjudication_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    evaluate_ugi_amine_semantic_program_comparison_preflight_stage as evaluate_ugi_amine_semantic_program_comparison_preflight_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    evaluate_ugi_amine_semantic_program_comparison_stage as evaluate_ugi_amine_semantic_program_comparison_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    evaluate_ugi_group_balanced_program_prior_comparison_preflight_stage as evaluate_ugi_group_balanced_program_prior_comparison_preflight_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    evaluate_ugi_group_balanced_program_prior_comparison_stage as evaluate_ugi_group_balanced_program_prior_comparison_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    evaluate_ugi_morphology_diversity_comparison_preflight_stage as evaluate_ugi_morphology_diversity_comparison_preflight_stage,
)
from experiments.phase1.product_l1.semantic_evaluation_stages import (
    evaluate_ugi_morphology_diversity_comparison_stage as evaluate_ugi_morphology_diversity_comparison_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    _reaction_specialist_v0_comparison_stage as _reaction_specialist_v0_comparison_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    _require_successful_ugi_chemistry_preflight as _require_successful_ugi_chemistry_preflight,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    _ugi_chemistry_specialist_base_comparison_stage as _ugi_chemistry_specialist_base_comparison_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_chemistry_specialist_base_comparison_preflight_recovery_stage as evaluate_ugi_chemistry_specialist_base_comparison_preflight_recovery_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_chemistry_specialist_base_comparison_preflight_stage as evaluate_ugi_chemistry_specialist_base_comparison_preflight_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_chemistry_specialist_base_comparison_recovery_stage as evaluate_ugi_chemistry_specialist_base_comparison_recovery_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_chemistry_specialist_base_comparison_stage as evaluate_ugi_chemistry_specialist_base_comparison_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_chemistry_specialist_base_comparison_v4_stage as evaluate_ugi_chemistry_specialist_base_comparison_v4_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_joint_lipid_specialist_base_comparison_preflight_stage as evaluate_ugi_joint_lipid_specialist_base_comparison_preflight_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_joint_lipid_specialist_base_comparison_stage as evaluate_ugi_joint_lipid_specialist_base_comparison_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_measured_only_full_model_base_comparison_preflight_stage as evaluate_ugi_measured_only_full_model_base_comparison_preflight_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_measured_only_full_model_base_comparison_stage as evaluate_ugi_measured_only_full_model_base_comparison_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_reaction_specialist_v0_comparison_stage as evaluate_ugi_reaction_specialist_v0_comparison_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_reaction_topology_specialist_v0_comparison_stage as evaluate_ugi_reaction_topology_specialist_v0_comparison_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_role_local_specialist_base_comparison_preflight_stage as evaluate_ugi_role_local_specialist_base_comparison_preflight_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_role_local_specialist_base_comparison_stage as evaluate_ugi_role_local_specialist_base_comparison_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_structured_topology_specialist_base_comparison_preflight_stage as evaluate_ugi_structured_topology_specialist_base_comparison_preflight_stage,
)
from experiments.phase1.product_l1.specialist_evaluation_stages import (
    evaluate_ugi_structured_topology_specialist_base_comparison_stage as evaluate_ugi_structured_topology_specialist_base_comparison_stage,
)
from experiments.phase1.product_l1.training_stages import (
    UGI_TREE_ABLATION_SCHEMA as UGI_TREE_ABLATION_SCHEMA,
)
from experiments.phase1.product_l1.training_stages import (
    _tree_ablation_effective_config as _tree_ablation_effective_config,
)
from experiments.phase1.product_l1.training_stages import (
    train_selected_ugi_tree_production_stage as train_selected_ugi_tree_production_stage,
)
from experiments.phase1.product_l1.training_stages import (
    train_ugi_closure_stage as train_ugi_closure_stage,
)
from experiments.phase1.product_l1.training_stages import (
    train_ugi_joint_ablation_stage as train_ugi_joint_ablation_stage,
)
from experiments.phase1.product_l1.training_stages import (
    train_ugi_joint_stage as train_ugi_joint_stage,
)
from forge.core.io import write_json


@stage("corpus.ugi.calibration-program-draw.v1")
def build_ugi_calibration_program_draw_stage(context: RunContext) -> StageResult:
    """Freeze one source- and calibration-role-balanced program draw."""

    from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_calibration import (
        build_calibration_program_draw,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = build_calibration_program_draw(
        context.config_path,
        context.repo,
        context.output_path("program_draw.json"),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "program_draw",
                "program_draw.json",
                "forge.ugi_tree_transformer_calibration_program_draw.v1",
                rows=len(result["samples"]),
            ),
        ),
        metrics={"programs": len(result["samples"])},
        summary={
            "candidate_selection": False,
            "evaluation_fold": "calibration",
            "heldout_rows_used": False,
            "paired_random_streams_required": True,
        },
    )


@stage("corpus.phase1.freeze.v1")
def freeze_phase1_corpus(context: RunContext) -> StageResult:
    """Build the exact Phase 1 corpus contract inside an isolated stage directory."""

    from forge.corpus import freeze_phase1_data_contract

    config = context.config()
    require_config_inputs(context, config)
    paths = {
        "ugi_assignments": context.output_path("ugi_l1_assignments.csv.gz"),
        "ugi_provenance": context.output_path("ugi_l1_constitutional_provenance.csv.gz"),
        "manifest": context.output_path("manifest.json"),
        "result": context.output_path("result.json"),
    }
    result = freeze_phase1_data_contract(
        context.config_path,
        context.repo,
        output_paths=paths,
        output_display_root=context.output_dir,
    )
    summary = result["summary"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "assignments",
                "ugi_l1_assignments.csv.gz",
                "phase1_ugi_l1_assignments.v1",
                rows=int(summary["ugi_l1_products"]),
            ),
            ProducedArtifact(
                "provenance",
                "ugi_l1_constitutional_provenance.csv.gz",
                "phase1_ugi_l1_constitutional_provenance.v1",
                rows=int(summary["ugi_source_rows"]),
            ),
            ProducedArtifact("manifest", "manifest.json", "phase1_product_l1_split_manifest.v3"),
            ProducedArtifact("result", "result.json", "phase1_product_l1_data_result.v3"),
        ),
        metrics={
            "r0_rows": int(summary["r0_rows"]),
            "r1_rows": int(summary["r1_rows"]),
            "ugi_l1_products": int(summary["ugi_l1_products"]),
        },
        summary={
            "guidance_enabled": False,
            "r1_sampling_weight": "realism_weight",
            "status": result["status"],
        },
    )


@stage("corpus.ugi.training-cache-verify.v1")
def verify_ugi_training_cache(context: RunContext) -> StageResult:
    """Validate the frozen tensor cache before any trainer can consume it."""

    from forge.corpus.training_cache import load_ugi_training_cache_payload

    config = context.config()
    configured_labels = set(config.get("inputs", {}))
    require_config_inputs(context, config, labels=configured_labels)
    cache_path = context.input("prepared_cache")
    payload = load_ugi_training_cache_payload(cache_path)
    cached_inputs = payload.get("inputs")
    if not isinstance(cached_inputs, dict):
        raise StageError("prepared training cache has no authenticated inputs")
    for label in sorted(configured_labels):
        if cached_inputs.get(label, {}).get("sha256") != config["inputs"][label]["sha256"]:
            raise StageError(f"prepared training cache input changed for {label!r}")
    records = payload.get("joint_records_by_fold")
    if not isinstance(records, dict):
        raise StageError("prepared training cache has no fold records")
    counts = {fold: len(values) for fold, values in records.items()}
    if counts != config.get("expected_fold_counts"):
        raise StageError(f"prepared training cache fold counts changed: {counts}")
    receipt = {
        "cache": context.stage.inputs["prepared_cache"].to_mapping(),
        "fold_counts": counts,
        "inputs": {label: config["inputs"][label] for label in sorted(configured_labels)},
        "schema_version": "forge.ugi_training_cache_receipt.v1",
        "status": "verified",
    }
    del payload, records
    gc.collect()
    write_json(context.output_path("receipt.json"), receipt)
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "receipt",
                "receipt.json",
                "forge.ugi_training_cache_receipt.v1",
                rows=sum(counts.values()),
            ),
        ),
        metrics={"records": sum(counts.values())},
        summary={"cache_sha256": context.stage.inputs["prepared_cache"].sha256},
    )


__all__ = [
    "freeze_phase1_corpus",
    "sample_ugi_shards",
    "train_ugi_closure_stage",
    "train_ugi_joint_stage",
    "verify_ugi_training_cache",
]
