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


__all__ = [
    "build_potency_study_corpus_stage",
    "fit_ugi_potency_adapter_crossfit",
]
