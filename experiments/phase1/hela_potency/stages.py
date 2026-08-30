"""Thin stage adapters for the consolidated potency-study data contract."""

from __future__ import annotations

from experiments._runtime.registry import stage
from experiments._runtime.stage import (
    ProducedArtifact,
    RunContext,
    StageResult,
    require_config_inputs,
)


@stage("potency.study-corpus.v2")
def build_potency_study_corpus_stage(context: RunContext) -> StageResult:
    """Materialize one row-preserving study corpus over the selected LNPDB records."""

    from forge.potency.study_data import build_potency_study_corpus

    config = context.config()
    require_config_inputs(context, config)
    result = build_potency_study_corpus(
        context.config_path,
        context.repo,
        ledger_path=context.output_path("observations.csv.gz"),
        result_path=context.output_path("result.json"),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "observations",
                "observations.csv.gz",
                "forge.potency_study_observations.v2",
                rows=int(result["artifact"]["rows"]),
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.potency_study_corpus_result.v2",
            ),
        ),
        metrics={
            "observations": int(result["artifact"]["rows"]),
            "studies": len(result["studies"]),
            "excluded_observations": int(result["source_accounting"]["excluded_observations"]),
        },
        summary={
            "status": result["status"],
            "row_source": "lnpdb",
            "cross_study_label_pooling": False,
            "biological_guidance": False,
        },
    )


@stage("potency.ugi-adapter-crossfit.v1")
def fit_ugi_potency_adapter_crossfit(context: RunContext) -> StageResult:
    """Fit only the bounded adapter and qualify its component-disjoint partial-state signal."""

    from experiments.phase1.hela_potency.potency_adapter import (
        RESULT_SCHEMA,
        run_potency_adapter_crossfit,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_potency_adapter_crossfit(
        context.config_path,
        context.repo,
        context.output_path("."),
        profile=context.profile,
        allocated_device=context.resources.device,
        progress_dir=context.work_dir / "folds",
        progress_commit=context.commit_progress,
        resume=context.resume,
    )
    artifacts = [
        ProducedArtifact("result", "result.json", RESULT_SCHEMA),
        ProducedArtifact(
            "adapter_bundle",
            "adapter_bundle.tar",
            "forge.ugi_hela_potency_adapter_bundle.v1",
        ),
    ]
    return StageResult(
        artifacts=tuple(artifacts),
        metrics={
            "observations": int(result["observations"]),
            "crossfit_folds": len(result["folds"]),
            "active_time_bins": len(result["active_time_bins"]),
        },
        summary={
            "status": result["status"],
            "base_generator_frozen": True,
            "guided_generation": False,
            "oracle_calls": 0,
            "synthesis_calls": 0,
        },
    )


@stage("potency.ugi-adapter-failure-attribution.v1")
def diagnose_ugi_potency_adapter_failure(context: RunContext) -> StageResult:
    """Attribute the frozen negative adapter result without fitting or sampling."""

    from experiments.phase1.hela_potency.potency_adapter_failure_attribution import (
        RESULT_SCHEMA,
        build_potency_adapter_failure_attribution,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = build_potency_adapter_failure_attribution(
        context.config_path,
        context.repo,
        context.output_path("result.json"),
    )
    support = result["attribution"]["effective_component_support"]
    return StageResult(
        artifacts=(ProducedArtifact("result", "result.json", RESULT_SCHEMA),),
        metrics={
            "nominal_products": int(result["dataset_geometry"]["rows"]),
            "independent_heads": int(support["heads_total"]),
            "independent_tail_pairs": int(support["aldehyde_isocyanide_pairs_total"]),
            "active_time_bins": len(result["held_out_signal"]["active_time_bins"]),
        },
        summary={
            "status": result["status"],
            "another_guidance_run_authorized": False,
            "gate_relaxed": False,
            "guided_generation": False,
        },
    )


@stage("potency.ugi-ordinal-adapter-crossfit.v2")
def fit_ugi_potency_ordinal_adapter_crossfit(context: RunContext) -> StageResult:
    """Fit paired real/shuffled ordinal adapters without running guided generation."""

    from experiments.phase1.hela_potency.potency_adapter_ordinal import (
        LEDGER_SCHEMA,
        RESULT_SCHEMA,
        run_potency_ordinal_crossfit,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_potency_ordinal_crossfit(
        context.config_path,
        context.repo,
        context.output_path("."),
        profile=context.profile,
        allocated_device=context.resources.device,
        progress_dir=context.work_dir / "folds",
        progress_commit=context.commit_progress,
        resume=context.resume,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact("result", "result.json", RESULT_SCHEMA),
            ProducedArtifact(
                "evaluation_scores",
                "evaluation_scores.csv.gz",
                LEDGER_SCHEMA,
                rows=sum(int(row["test_rows"]) for row in result["folds"])
                * len(config["validation"]["time_bins"]),
            ),
            ProducedArtifact(
                "adapter_bundle",
                "adapter_bundle.tar",
                "forge.ugi_hela_potency_ordinal_adapter_bundle.v2",
            ),
        ),
        metrics={
            "observations": int(result["observations"]),
            "crossfit_folds": len(result["folds"]),
            "active_time_bins": len(result["active_time_bins"]),
        },
        summary={
            "status": result["status"],
            "paired_real_shuffled_controls": True,
            "direct_percentile_monotonicity_tested": True,
            "guided_generation": False,
            "oracle_calls": 0,
            "synthesis_calls": 0,
        },
    )


@stage("potency.ugi-partial-state-value-crossfit.v1")
def fit_ugi_partial_state_value_crossfit(context: RunContext) -> StageResult:
    """Fit a direct graph-level value head without running guided generation."""

    from experiments.phase1.hela_potency.partial_state_value import (
        LEDGER_SCHEMA,
        RESULT_SCHEMA,
        run_partial_state_value_crossfit,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_partial_state_value_crossfit(
        context.config_path,
        context.repo,
        context.output_path("."),
        profile=context.profile,
        allocated_device=context.resources.device,
        progress_dir=context.work_dir / "folds",
        progress_commit=context.commit_progress,
        resume=context.resume,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact("result", "result.json", RESULT_SCHEMA),
            ProducedArtifact(
                "evaluation_scores",
                "evaluation_scores.csv.gz",
                LEDGER_SCHEMA,
                rows=sum(int(row["test_rows"]) for row in result["folds"])
                * len(config["validation"]["time_bins"]),
            ),
            ProducedArtifact(
                "value_head_bundle",
                "value_head_bundle.tar",
                "forge.ugi_hela_partial_state_value_bundle.v1",
            ),
        ),
        metrics={
            "observations": int(result["observations"]),
            "crossfit_folds": len(result["folds"]),
            "active_time_bins": len(result["active_time_bins"]),
        },
        summary={
            "status": result["status"],
            "base_generator_frozen": True,
            "direct_scalar_prediction": True,
            "paired_real_shuffled_controls": True,
            "guided_generation": False,
            "oracle_calls": 0,
            "synthesis_calls": 0,
        },
    )


@stage("potency.ugi-partial-state-attention-value-crossfit.v2")
def fit_ugi_partial_state_attention_value_crossfit(context: RunContext) -> StageResult:
    """Fit the node-preserving layerwise attention scorer without guided generation."""

    from experiments.phase1.hela_potency.partial_state_attention_value import (
        LEDGER_SCHEMA,
        RESULT_SCHEMA,
        run_partial_state_attention_value_crossfit,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_partial_state_attention_value_crossfit(
        context.config_path,
        context.repo,
        context.output_path("."),
        profile=context.profile,
        allocated_device=context.resources.device,
        progress_dir=context.work_dir / "folds",
        progress_commit=context.commit_progress,
        resume=context.resume,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact("result", "result.json", RESULT_SCHEMA),
            ProducedArtifact(
                "evaluation_scores",
                "evaluation_scores.csv.gz",
                LEDGER_SCHEMA,
                rows=sum(int(row["test_rows"]) for row in result["folds"])
                * len(result["feature_bank"]["times"]),
            ),
            ProducedArtifact(
                "value_head_bundle",
                "value_head_bundle.tar",
                "forge.ugi_hela_partial_state_attention_value_bundle.v2",
            ),
        ),
        metrics={
            "observations": int(result["observations"]),
            "crossfit_folds": len(result["folds"]),
            "active_time_bins": len(result["active_time_bins"]),
        },
        summary={
            "status": result["status"],
            "base_generator_frozen": True,
            "node_axis_preserved": True,
            "transformer_layers": int(result["feature_bank"]["layer_count"]),
            "predicted_clean_probabilities": True,
            "paired_real_shuffled_controls": True,
            "guided_generation": False,
            "oracle_calls": 0,
            "synthesis_calls": 0,
        },
    )


__all__ = [
    "build_potency_study_corpus_stage",
    "diagnose_ugi_potency_adapter_failure",
    "fit_ugi_potency_adapter_crossfit",
    "fit_ugi_potency_ordinal_adapter_crossfit",
    "fit_ugi_partial_state_attention_value_crossfit",
    "fit_ugi_partial_state_value_crossfit",
]
