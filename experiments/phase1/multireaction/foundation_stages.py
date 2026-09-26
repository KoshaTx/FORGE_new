"""Corpus and original multireaction training stage adapters.

These stable stage identifiers remain registered when ``stages`` is loaded. The
22-family COMPOSE trainer is registered separately in ``compose_lipid_training``.
"""

from __future__ import annotations

from experiments._runtime.registry import stage
from experiments._runtime.stage import (
    ProducedArtifact,
    RunContext,
    StageResult,
    require_config_inputs,
)


@stage("corpus.multireaction.lnpdb.v1")
def build_multireaction_corpus(context: RunContext) -> StageResult:
    """Build exact recursive programs and source/component-disjoint ledgers."""

    from forge.corpus.multireaction import build_multireaction_lnpdb_corpus

    config = context.config()
    require_config_inputs(context, config)
    outputs = {
        "atlas": context.output_path("reaction_program_atlas.csv.gz"),
        "steps": context.output_path("reaction_program_steps.csv.gz"),
        "semantic_atoms": context.output_path("semantic_atoms.csv.gz"),
        "provenance": context.output_path("source_provenance.csv.gz"),
        "splits": context.output_path("component_disjoint_splits.csv.gz"),
        "manifest": context.output_path("manifest.json"),
        "result": context.output_path("result.json"),
    }
    result = build_multireaction_lnpdb_corpus(
        context.config_path,
        context.repo,
        outputs=outputs,
    )
    summary = result["summary"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "atlas",
                "reaction_program_atlas.csv.gz",
                "forge.multireaction_program_atlas.v1",
                rows=int(summary["unique_source_products"]),
            ),
            ProducedArtifact(
                "steps",
                "reaction_program_steps.csv.gz",
                "forge.multireaction_program_steps.v1",
                rows=int(summary["exact_program_steps"]),
            ),
            ProducedArtifact(
                "semantic_atoms",
                "semantic_atoms.csv.gz",
                "forge.multireaction_semantic_atoms.v2",
                rows=int(summary["semantic_atom_rows"]),
            ),
            ProducedArtifact(
                "provenance",
                "source_provenance.csv.gz",
                "forge.multireaction_source_provenance.v1",
                rows=int(summary["source_rows"]),
            ),
            ProducedArtifact(
                "splits",
                "component_disjoint_splits.csv.gz",
                "forge.multireaction_component_disjoint_splits.v1",
                rows=int(summary["admitted_products"]),
            ),
            ProducedArtifact("manifest", "manifest.json", "forge.multireaction_lnpdb_manifest.v1"),
            ProducedArtifact("result", "result.json", "forge.multireaction_lnpdb_result.v1"),
        ),
        metrics={
            "admitted_products": int(summary["admitted_products"]),
            "abstained_products": int(summary["abstained_products"]),
            "exact_program_steps": int(summary["exact_program_steps"]),
            "semantic_origin_products": int(summary["semantic_origin_products"]),
        },
        summary={
            "biological_labels_used": False,
            "reductive_amination_substructure_rate_reported": False,
            "sampling_policy": summary["sampling_policy"],
        },
    )


@stage("corpus.multireaction.reaction-enumerated-expansion.v1")
def build_multireaction_reaction_enumerated_expansion(context: RunContext) -> StageResult:
    """Expand BL/LX from source-linked components with exact forward programs."""

    from forge.corpus.multireaction_expansion import build_multireaction_expansion

    config = context.config()
    require_config_inputs(context, config)
    outputs = {
        "components": context.output_path("component_registry.csv.gz"),
        "attempts": context.output_path("enumeration_attempts.csv.gz"),
        "atlas": context.output_path("reaction_program_atlas.csv.gz"),
        "steps": context.output_path("reaction_program_steps.csv.gz"),
        "semantic_atoms": context.output_path("semantic_atoms.csv.gz"),
        "provenance": context.output_path("source_provenance.csv.gz"),
        "splits": context.output_path("component_family_splits.csv.gz"),
        "manifest": context.output_path("manifest.json"),
        "result": context.output_path("result.json"),
    }
    result = build_multireaction_expansion(
        context.config_path,
        context.repo,
        outputs=outputs,
    )
    summary = result["summary"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "components",
                "component_registry.csv.gz",
                "forge.multireaction_expansion_components.v1",
                rows=int(summary["candidate_components"]),
            ),
            ProducedArtifact(
                "attempts",
                "enumeration_attempts.csv.gz",
                "forge.multireaction_expansion_attempts.v1",
                rows=int(summary["enumeration_attempts"]),
            ),
            ProducedArtifact(
                "atlas",
                "reaction_program_atlas.csv.gz",
                "forge.multireaction_program_atlas.v1",
                rows=int(summary["total_products"]),
            ),
            ProducedArtifact(
                "steps",
                "reaction_program_steps.csv.gz",
                "forge.multireaction_program_steps.v1",
                rows=int(summary["exact_program_steps"]),
            ),
            ProducedArtifact(
                "semantic_atoms",
                "semantic_atoms.csv.gz",
                "forge.multireaction_semantic_atoms.v2",
                rows=int(summary["semantic_atom_rows"]),
            ),
            ProducedArtifact(
                "provenance",
                "source_provenance.csv.gz",
                "forge.multireaction_source_provenance.v1",
            ),
            ProducedArtifact(
                "splits",
                "component_family_splits.csv.gz",
                "forge.multireaction_expanded_component_family_splits.v1",
                rows=int(summary["total_products"]),
            ),
            ProducedArtifact(
                "manifest",
                "manifest.json",
                "forge.multireaction_expansion_manifest.v1",
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.multireaction_expansion_result.v1",
            ),
        ),
        metrics={
            "source_products": int(summary["source_products_preserved"]),
            "computed_products": int(summary["computed_products_admitted"]),
            "training_products": int(summary["product_folds"]["train"]),
            "calibration_products": int(summary["product_folds"]["calibration"]),
            "heldout_products": int(summary["product_folds"]["heldout"]),
        },
        summary={
            "status": result["status"],
            "source_executed_evidence_rewritten": False,
            "biological_labels_used": False,
            "computed_products_are_synthesis_success": False,
            "reductive_amination_substructure_rate_reported": False,
        },
    )


@stage("corpus.multireaction.mixed-repeat-expansion.v1")
def build_multireaction_mixed_repeat_expansion(context: RunContext) -> StageResult:
    """Expand BL/LX with distinct source-linked repeat components in one exact program."""

    from forge.corpus.multireaction_mixed_expansion import (
        build_multireaction_mixed_expansion,
    )

    config = context.config()
    require_config_inputs(context, config)
    outputs = {
        "components": context.output_path("component_registry.csv.gz"),
        "attempts": context.output_path("mixed_enumeration_attempts.csv.gz"),
        "atlas": context.output_path("reaction_program_atlas.csv.gz"),
        "steps": context.output_path("reaction_program_steps.csv.gz"),
        "semantic_atoms": context.output_path("semantic_atoms.csv.gz"),
        "provenance": context.output_path("source_provenance.csv.gz"),
        "splits": context.output_path("component_family_splits.csv.gz"),
        "manifest": context.output_path("manifest.json"),
        "result": context.output_path("result.json"),
    }
    strict_model_support = "atom_vocabulary" in config.get("inputs", {})
    if strict_model_support:
        outputs["support_exclusions"] = context.output_path("model_support_exclusions.csv.gz")
    result = build_multireaction_mixed_expansion(
        context.config_path,
        context.repo,
        outputs=outputs,
    )
    summary = result["summary"]
    artifacts = [
        ProducedArtifact(
            "components",
            "component_registry.csv.gz",
            "forge.multireaction_expansion_components.v1",
        ),
        ProducedArtifact(
            "attempts",
            "mixed_enumeration_attempts.csv.gz",
            "forge.multireaction_mixed_expansion_attempts.v1",
            rows=int(summary["mixed_enumeration_attempts"]),
        ),
        ProducedArtifact(
            "atlas",
            "reaction_program_atlas.csv.gz",
            "forge.multireaction_program_atlas.v2",
            rows=int(summary["total_products"]),
        ),
        ProducedArtifact(
            "steps",
            "reaction_program_steps.csv.gz",
            "forge.multireaction_program_steps.v1",
            rows=int(summary["exact_program_steps"]),
        ),
        ProducedArtifact(
            "semantic_atoms",
            "semantic_atoms.csv.gz",
            "forge.multireaction_semantic_atoms.v2",
            rows=int(summary["semantic_atom_rows"]),
        ),
        ProducedArtifact(
            "provenance",
            "source_provenance.csv.gz",
            "forge.multireaction_source_provenance.v1",
        ),
        ProducedArtifact(
            "splits",
            "component_family_splits.csv.gz",
            "forge.multireaction_mixed_component_family_splits.v1",
            rows=int(summary["total_products"]),
        ),
        ProducedArtifact(
            "manifest",
            "manifest.json",
            "forge.multireaction_mixed_expansion_manifest.v1",
        ),
        ProducedArtifact(
            "result",
            "result.json",
            "forge.multireaction_mixed_expansion_result.v1",
        ),
    ]
    if strict_model_support:
        artifacts.insert(
            -2,
            ProducedArtifact(
                "support_exclusions",
                "model_support_exclusions.csv.gz",
                "forge.multireaction_model_support_exclusions.v1",
                rows=int(summary["homogeneous_model_support_exclusions"]),
            ),
        )
    return StageResult(
        artifacts=tuple(artifacts),
        metrics={
            "v1_products": int(summary["v1_products_preserved"]),
            "mixed_products": int(summary["mixed_products_admitted"]),
            "training_products": int(summary["product_folds"]["train"]),
            "calibration_products": int(summary["product_folds"]["calibration"]),
            "heldout_products": int(summary["product_folds"]["heldout"]),
        },
        summary={
            "status": result["status"],
            "source_executed_evidence_rewritten": False,
            "biological_labels_used": False,
            "computed_products_are_synthesis_success": False,
            "component_family_assignment_precedes_enumeration": True,
            "reductive_amination_substructure_rate_reported": False,
        },
    )


@stage("model.multireaction.training.v2")
@stage("model.multireaction.training_smoke.v1")
def train_multireaction(context: RunContext) -> StageResult:
    """Exercise the model/checkpoint/sampler seam under matched semantic controls."""

    from experiments.phase1.multireaction.training import run_multireaction_training

    config = context.config()
    require_config_inputs(context, config)
    result = run_multireaction_training(
        context.config_path,
        context.repo,
        context.output_path("."),
        work_dir=context.work_dir,
        resume=context.resume,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "checkpoint",
                "checkpoint.json",
                "forge.multireaction_sparse_flow_checkpoint.v2",
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.multireaction_training_result.v2",
            ),
        ),
        metrics={
            "training_arms": len(result["arms"]),
        },
        summary={
            "status": result["status"],
            "component_identifiers_used": False,
            "production_comparison": False,
        },
    )


@stage("model.multireaction.sampling.v2")
@stage("model.multireaction.sampling_smoke.v1")
def sample_multireaction(context: RunContext) -> StageResult:
    """Load the training dependency checkpoint and exercise program-conditioned sampling."""

    from experiments.phase1.multireaction.sampling import run_multireaction_sampling

    config = context.config()
    require_config_inputs(context, config)
    checkpoint = context.dependency("training", "checkpoint")
    result = run_multireaction_sampling(
        context.config_path,
        context.repo,
        checkpoint.path,
        context.output_path("."),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "samples",
                "samples.json",
                "forge.multireaction_samples.v2",
                rows=int(result["metrics"]["samples"]),
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.multireaction_sampling_result.v2",
            ),
        ),
        metrics={
            "samples": int(result["metrics"]["samples"]),
            "valid_samples": int(result["metrics"]["valid"]),
            "exact_l1_samples": int(result["metrics"]["exact_l1_program"]),
        },
        summary={
            "status": result["status"],
            "component_identifiers_used": False,
            "production_comparison": False,
        },
    )


@stage("model.multireaction.overfit-qualification.v1")
def qualify_multireaction_overfit(context: RunContext) -> StageResult:
    """Apply frozen overfit gates without hiding a negative result."""

    from experiments.phase1.multireaction.qualification import qualify_overfit_run

    config = context.config()
    require_config_inputs(context, config)
    result = qualify_overfit_run(
        context.config_path,
        context.repo,
        context.dependency("training", "result").path,
        context.dependency("sampling", "result").path,
        context.output_path("result.json"),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.multireaction_overfit_qualification.v1",
            ),
        ),
        metrics={
            "gate_pass": bool(result["gate_pass"]),
            "passed_gates": int(sum(result["gates"].values())),
            "total_gates": len(result["gates"]),
        },
        summary={
            "status": result["status"],
            "production_launch_authorized": False,
        },
    )


__all__ = [
    "build_multireaction_corpus",
    "build_multireaction_reaction_enumerated_expansion",
    "build_multireaction_mixed_repeat_expansion",
    "train_multireaction",
    "sample_multireaction",
    "qualify_multireaction_overfit",
]
