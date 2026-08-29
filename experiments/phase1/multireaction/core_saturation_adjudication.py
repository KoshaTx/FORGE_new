"""Adjudicate the final core-saturation model and separately retrained controls."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from experiments._runtime import verify_run_directory
from forge.core.hashing import pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json

from .final_production_adjudication import (
    _all_true,
    _arm_comparison,
    _arm_summary,
    _final_metrics,
    _ledger_counts,
)
from .mechanism_study import RESULT_SCHEMA as STUDY_RESULT_SCHEMA
from .production_adjudication import (
    CYCLIC_ARM,
    POSTHOC_ARM,
    UGI_PROGRAM,
    SynthesisProgramProductionAdjudicationError,
    _adjudicate_retention,
    _artifact_path,
    _load_production_run,
    _paired_descriptive_comparison,
)
from .production_evaluation import RESULT_SCHEMA as EVALUATION_RESULT_SCHEMA
from .production_training import RESULT_SCHEMA as TRAINING_RESULT_SCHEMA

RESULT_SCHEMA = "forge.shared_bias_core_saturation_adjudication.v1"
STUDY_IMPLEMENTATION = "model.transformer-mechanism-study.v1"
STUDY_NAME = "shared_bias_end_to_end_retraining"
FINAL_ARM = "shared_bias_program_role_source"
CONTROL_ARMS = ("ugi_only_conditioned", POSTHOC_ARM, CYCLIC_ARM)
PROGRAMS = (
    UGI_PROGRAM,
    "bl_2023_repeated_aza_michael",
    "lx_2024_repeated_reductive_amination",
)
EXPECTED_SEEDS = (20260825, 20260826, 20260827)
FINAL_EXPERIMENT_IDS = {
    0: "phase1-shared-bias-parallel-program-role-seed0-core-saturation-h100-v2",
    1: "phase1-shared-bias-parallel-program-role-seed1-core-saturation-h100-v2",
    2: "phase1-shared-bias-parallel-program-role-seed2-core-saturation-h100-v2",
}
CONTROL_EXPERIMENT_IDS = {
    "ugi_only_conditioned": "phase1-shared-bias-ugi-only-core-saturation-h100",
    POSTHOC_ARM: "phase1-shared-bias-shared-null-core-saturation-h100",
    CYCLIC_ARM: "phase1-shared-bias-cyclic-core-saturation-h100",
}


def _load_control_study_run(run_dir: Path, *, arm_id: str) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    verify_run_directory(run_dir)
    run = read_json_object(
        run_dir / "run.json",
        error=SynthesisProgramProductionAdjudicationError,
        label="core-saturation control run",
    )
    replicate = int(run.get("replicate", -1))
    if (
        run.get("experiment_id") != CONTROL_EXPERIMENT_IDS[arm_id]
        or run.get("profile") != "full"
        or run.get("status") != "complete"
        or set(run.get("stages", {})) != {"production"}
        or replicate not in {0, 1, 2}
    ):
        raise SynthesisProgramProductionAdjudicationError(
            f"{arm_id} requires one complete full-profile control run"
        )
    manifest = read_json_object(
        run_dir / "stages/production/manifest.json",
        error=SynthesisProgramProductionAdjudicationError,
        label=f"{arm_id} production manifest",
    )
    if manifest.get("implementation") != STUDY_IMPLEMENTATION:
        raise SynthesisProgramProductionAdjudicationError(
            f"{arm_id} used the wrong production implementation"
        )
    paths = {
        label: _artifact_path(run_dir, "production", manifest, label)
        for label in ("result", "training_result", "evaluation_result", "samples", "design")
    }
    study = read_json_object(
        paths["result"],
        error=SynthesisProgramProductionAdjudicationError,
        label=f"{arm_id} study result",
    )
    training = read_json_object(
        paths["training_result"],
        error=SynthesisProgramProductionAdjudicationError,
        label=f"{arm_id} training result",
    )
    evaluation = read_json_object(
        paths["evaluation_result"],
        error=SynthesisProgramProductionAdjudicationError,
        label=f"{arm_id} evaluation result",
    )
    seed = int(evaluation.get("seed", -1))
    if (
        study.get("schema_version") != STUDY_RESULT_SCHEMA
        or training.get("schema_version") != TRAINING_RESULT_SCHEMA
        or evaluation.get("schema_version") != EVALUATION_RESULT_SCHEMA
        or any(document.get("status") != "pass" for document in (study, training, evaluation))
        or any(document.get("profile") != "full" for document in (study, training, evaluation))
        or any(
            int(document.get("replicate", -1)) != replicate
            for document in (study, training, evaluation)
        )
        or any(int(document.get("seed", -1)) != seed for document in (study, training))
        or study.get("study") != STUDY_NAME
        or study.get("arms") != [arm_id]
        or set(training.get("arms", {})) != {arm_id}
        or set(evaluation.get("checkpoint_metrics", {})) != {arm_id}
        or not _all_true(training.get("gates"))
        or not _all_true(evaluation.get("gates"))
        or study.get("calls") != {"oracle": 0, "route": 0}
        or study.get("candidate_selection") is not False
        or evaluation.get("terminal_decode_policy")
        != "strict_reaction_core_saturation_argmax"
    ):
        raise SynthesisProgramProductionAdjudicationError(
            f"{arm_id} production identity or gates are inconsistent for replicate {replicate}"
        )
    for label, receipt_name in (
        ("training_result", "training"),
        ("evaluation_result", "evaluation"),
        ("samples", "samples"),
    ):
        receipt = study.get(receipt_name)
        if not isinstance(receipt, Mapping) or receipt.get("sha256") != str(sha256_file(paths[label])):
            raise SynthesisProgramProductionAdjudicationError(
                f"{arm_id} study result does not authenticate {label}"
            )
    inputs = manifest.get("external_inputs")
    design_pin = inputs.get("base_design") if isinstance(inputs, Mapping) else None
    if not isinstance(design_pin, Mapping):
        raise SynthesisProgramProductionAdjudicationError(f"{arm_id} omits its base-design pin")
    return {
        "run_dir": run_dir,
        "run": run,
        "manifest": manifest,
        "paths": paths,
        "study": study,
        "training": training,
        "evaluation": evaluation,
        "samples_path": paths["samples"],
        "design_pin": dict(design_pin),
        "effective_design_path": paths["design"],
    }


def _validate_group(
    items: Sequence[Mapping[str, Any]],
    *,
    arm_id: str,
    expected_programs: Sequence[str],
    allow_additional_programs: bool = False,
) -> None:
    if len(items) != 3:
        raise SynthesisProgramProductionAdjudicationError(f"{arm_id} requires three runs")
    replicates = [int(item["run"]["replicate"]) for item in items]
    seeds = [int(item["evaluation"]["seed"]) for item in items]
    if replicates != [0, 1, 2] or seeds != list(EXPECTED_SEEDS):
        raise SynthesisProgramProductionAdjudicationError(
            f"{arm_id} does not preserve paired replicates and training seeds"
        )
    for item in items:
        evaluation = item["evaluation"]
        metrics = _final_metrics(evaluation, arm_id, "9143")
        observed_programs = set(metrics)
        required_programs = set(expected_programs)
        if not required_programs.issubset(observed_programs) or (
            not allow_additional_programs and observed_programs != required_programs
        ):
            raise SynthesisProgramProductionAdjudicationError(
                f"{arm_id} final program set changed"
            )
        counts = _ledger_counts(
            item["samples_path"],
            arm_id=arm_id,
            programs=expected_programs,
            final_step=9143,
        )
        for program in expected_programs:
            row = metrics[program]
            if int(row.get("samples", -1)) != 3072 or counts[program]["attempts"] != 3072:
                raise SynthesisProgramProductionAdjudicationError(
                    f"{arm_id} changed the 3,072-attempt denominator for {program}"
                )
            observed = counts[program]["exact_l1_yield_per_attempt"]
            if not math.isclose(
                float(observed),
                float(row["exact_l1_yield_per_attempt"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise SynthesisProgramProductionAdjudicationError(
                    f"{arm_id} ledger and result disagree for {program}"
                )
            if "reductive_amination_substructure_hit_rate" in row:
                raise SynthesisProgramProductionAdjudicationError(
                    "forbidden reductive-amination substructure statistic was reported"
                )


def _evidence(items: Sequence[Mapping[str, Any]], repo: Path) -> list[dict[str, Any]]:
    output = []
    for item in items:
        evaluation_path = item.get("evaluation_path")
        if evaluation_path is None:
            paths = item.get("paths")
            evaluation_path = paths.get("evaluation_result") if isinstance(paths, Mapping) else None
        if not isinstance(evaluation_path, Path):
            raise SynthesisProgramProductionAdjudicationError(
                "adjudicated run has no authenticated evaluation result"
            )
        output.append(
            {
                "replicate": int(item["run"]["replicate"]),
                "seed": int(item["evaluation"]["seed"]),
                "run_id": str(item["run"]["run_id"]),
                "run_manifest": pin_record(item["run_dir"] / "run.json", repo),
                "evaluation_result": pin_record(evaluation_path, repo),
                "evaluation_samples": pin_record(item["samples_path"], repo),
                "effective_design": pin_record(item["effective_design_path"], repo),
            }
        )
    return output


def _matched_design_contract(design: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the prespecified arm intervention from an effective study design."""

    normalized = deepcopy(dict(design))
    training = normalized.get("training")
    if not isinstance(training, dict) or set(training.get("arms", {})) == set():
        raise SynthesisProgramProductionAdjudicationError(
            "effective study design has no training arm"
        )
    training.pop("arms")
    return normalized


def _catalogue_comparison(
    final_summary: Mapping[str, Any],
    catalogue_summary: Mapping[str, Any],
    frozen_comparison: Mapping[str, Any],
    design: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the final-model side of the frozen finite-catalogue comparison."""

    metadata = frozen_comparison.get("programs")
    if not isinstance(metadata, Mapping) or set(metadata) != set(PROGRAMS):
        raise SynthesisProgramProductionAdjudicationError(
            "frozen finite-catalogue comparison omits program metadata"
        )
    decision = design["ugi_retention"]["decision_rule"]
    metric_fields = {
        "raw_valid_fraction": ("raw_valid_fraction", "raw_valid_fraction"),
        "exact_l1_yield_per_attempt": (
            "exact_l1_yield_per_attempt",
            "exact_l1_yield_per_attempt",
        ),
        "unique_exact_l1_products_per_1000_attempts": (
            "unique_exact_l1_products_per_1000_attempts",
            "unique_exact_l1_products_per_1000_attempts",
        ),
        "unique_open_ended_exact_l1_products_per_1000_attempts": (
            "unique_open_ended_exact_l1_products_per_1000_attempts",
            "unique_open_ended_exact_l1_products_per_1000_attempts",
        ),
        "internal_diversity": ("internal_diversity", "internal_diversity"),
        "effective_component_count": (
            "effective_component_count",
            "effective_component_count",
        ),
        "component_novelty_fraction": (
            "component_novelty_fraction",
            "component_novelty_fraction",
        ),
        "whole_product_novelty_fraction": (
            "whole_lipid_novelty_fraction",
            "whole_product_novelty_fraction",
        ),
    }
    programs = {}
    for program in PROGRAMS:
        final_row = final_summary.get(program)
        catalogue_row = catalogue_summary.get(program)
        metadata_row = metadata.get(program)
        if not all(isinstance(row, Mapping) for row in (final_row, catalogue_row, metadata_row)):
            raise SynthesisProgramProductionAdjudicationError(
                f"finite-catalogue comparison is malformed for {program}"
            )
        comparisons = {}
        for metric, (final_name, catalogue_name) in metric_fields.items():
            final_metric = final_row.get(final_name)
            catalogue_metric = catalogue_row.get(catalogue_name)
            if not isinstance(final_metric, Mapping) or not isinstance(catalogue_metric, Mapping):
                raise SynthesisProgramProductionAdjudicationError(
                    f"finite-catalogue comparison omits {program}/{metric}"
                )
            comparisons[metric] = _paired_descriptive_comparison(
                final_metric.get("by_seed", []),
                catalogue_metric.get("by_seed", []),
                left_label="forge",
                right_label="catalogue",
                resamples=int(decision["resamples"]),
                seed=int(decision["seed"]),
                confidence_level=float(decision["confidence_level"]),
            )
        programs[program] = {
            "metrics": comparisons,
            "catalogue": dict(metadata_row["catalogue"]),
        }
    return {
        "forge_arm": FINAL_ARM,
        "catalogue_arm": "finite_component_catalogue_oracle",
        "primary_metric": "unique_open_ended_exact_l1_products_per_1000_attempts",
        "programs": programs,
        "interpretation": frozen_comparison.get("interpretation"),
        "decision_role": "descriptive_claim_matched_baseline_not_candidate_selection",
    }


def adjudicate_core_saturation_production(
    final_run_dirs: Sequence[Path],
    ugi_only_run_dirs: Sequence[Path],
    null_run_dirs: Sequence[Path],
    cyclic_run_dirs: Sequence[Path],
    catalogue_adjudication_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Aggregate the final graph Transformer and all separately retrained matched controls."""

    final = sorted(
        (_load_production_run(path) for path in final_run_dirs),
        key=lambda item: int(item["run"]["replicate"]),
    )
    for item in final:
        replicate = int(item["run"]["replicate"])
        if item["run"]["experiment_id"] != FINAL_EXPERIMENT_IDS[replicate]:
            raise SynthesisProgramProductionAdjudicationError(
                "final program-role run uses the wrong seed-specific experiment"
            )
        item["effective_design_path"] = resolve_pin(
            item["design_pin"], repo, label=f"final seed-{replicate} effective design"
        )
    groups = {
        "ugi_only_conditioned": sorted(
            (
                _load_control_study_run(path, arm_id="ugi_only_conditioned")
                for path in ugi_only_run_dirs
            ),
            key=lambda item: int(item["run"]["replicate"]),
        ),
        POSTHOC_ARM: sorted(
            (_load_control_study_run(path, arm_id=POSTHOC_ARM) for path in null_run_dirs),
            key=lambda item: int(item["run"]["replicate"]),
        ),
        CYCLIC_ARM: sorted(
            (_load_control_study_run(path, arm_id=CYCLIC_ARM) for path in cyclic_run_dirs),
            key=lambda item: int(item["run"]["replicate"]),
        ),
    }
    _validate_group(final, arm_id=FINAL_ARM, expected_programs=PROGRAMS)
    _validate_group(
        groups["ugi_only_conditioned"],
        arm_id="ugi_only_conditioned",
        expected_programs=(UGI_PROGRAM,),
        allow_additional_programs=True,
    )
    _validate_group(groups[POSTHOC_ARM], arm_id=POSTHOC_ARM, expected_programs=PROGRAMS)
    _validate_group(groups[CYCLIC_ARM], arm_id=CYCLIC_ARM, expected_programs=PROGRAMS)

    all_items = [*final, *(item for values in groups.values() for item in values)]
    designs = [
        read_json_object(
            item["effective_design_path"],
            error=SynthesisProgramProductionAdjudicationError,
            label="effective core-saturation study design",
        )
        for item in all_items
    ]
    matched_contracts = [_matched_design_contract(design) for design in designs]
    if any(contract != matched_contracts[0] for contract in matched_contracts[1:]):
        raise SynthesisProgramProductionAdjudicationError(
            "final and control runs differ outside the prespecified arm intervention"
        )
    design = designs[0]
    evaluations = {
        FINAL_ARM: [item["evaluation"] for item in final],
        **{arm_id: [item["evaluation"] for item in items] for arm_id, items in groups.items()},
    }
    summaries = {
        FINAL_ARM: _arm_summary(
            evaluations[FINAL_ARM], arm_id=FINAL_ARM, programs=PROGRAMS, final_step="9143"
        ),
        "ugi_only_conditioned": _arm_summary(
            evaluations["ugi_only_conditioned"],
            arm_id="ugi_only_conditioned",
            programs=(UGI_PROGRAM,),
            final_step="9143",
        ),
        POSTHOC_ARM: _arm_summary(
            evaluations[POSTHOC_ARM], arm_id=POSTHOC_ARM, programs=PROGRAMS, final_step="9143"
        ),
        CYCLIC_ARM: _arm_summary(
            evaluations[CYCLIC_ARM], arm_id=CYCLIC_ARM, programs=PROGRAMS, final_step="9143"
        ),
    }
    comparisons = {
        "final_vs_ugi_only": _arm_comparison(
            evaluations[FINAL_ARM],
            evaluations["ugi_only_conditioned"],
            left_arm=FINAL_ARM,
            right_arm="ugi_only_conditioned",
            programs=(UGI_PROGRAM,),
            final_step="9143",
            design=design,
        ),
        "final_vs_shared_null_posthoc": _arm_comparison(
            evaluations[FINAL_ARM],
            evaluations[POSTHOC_ARM],
            left_arm=FINAL_ARM,
            right_arm=POSTHOC_ARM,
            programs=PROGRAMS,
            final_step="9143",
            design=design,
        ),
        "final_vs_cyclic_program_id": _arm_comparison(
            evaluations[FINAL_ARM],
            evaluations[CYCLIC_ARM],
            left_arm=FINAL_ARM,
            right_arm=CYCLIC_ARM,
            programs=PROGRAMS,
            final_step="9143",
            design=design,
        ),
    }
    merged = []
    for final_evaluation, ugi_evaluation in zip(
        evaluations[FINAL_ARM], evaluations["ugi_only_conditioned"], strict=True
    ):
        merged.append(
            {
                "checkpoint_metrics": {
                    "shared_three_program_conditioned": final_evaluation["checkpoint_metrics"][FINAL_ARM],
                    "ugi_only_conditioned": ugi_evaluation["checkpoint_metrics"]["ugi_only_conditioned"],
                },
                "component_disjoint_metrics": {
                    "shared_three_program_conditioned": final_evaluation["component_disjoint_metrics"][FINAL_ARM],
                    "ugi_only_conditioned": ugi_evaluation["component_disjoint_metrics"]["ugi_only_conditioned"],
                },
            }
        )
    retention, retention_gates = _adjudicate_retention(merged, design)
    retention["challenger_arm"] = FINAL_ARM

    catalogue = read_json_object(
        catalogue_adjudication_path,
        error=SynthesisProgramProductionAdjudicationError,
        label="frozen finite-catalogue adjudication",
    )
    catalogue_summary = catalogue.get("finite_component_catalogue_arm_summary")
    frozen_catalogue_comparison = catalogue.get("finite_component_catalogue_comparison")
    if (
        catalogue.get("status") != "complete"
        or catalogue.get("candidate_selection") is not False
        or not isinstance(catalogue_summary, Mapping)
        or set(catalogue_summary) != set(PROGRAMS)
        or not isinstance(frozen_catalogue_comparison, Mapping)
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "frozen finite-catalogue adjudication is not admissible"
        )
    evidence = {
        FINAL_ARM: _evidence(final, repo),
        **{arm_id: _evidence(items, repo) for arm_id, items in groups.items()},
    }
    catalogue_comparison = _catalogue_comparison(
        summaries[FINAL_ARM],
        catalogue_summary,
        frozen_catalogue_comparison,
        design,
    )
    gates = {
        "three_replicates_per_arm": True,
        "paired_replicates_and_seeds": True,
        "matched_effective_design_contract": True,
        "final_checkpoint_only_reported": True,
        "attempt_budgets_match": True,
        "reaction_core_saturation_decoder": True,
        "candidate_selection_absent": True,
        "route_or_oracle_calls_zero": True,
        "reductive_amination_substructure_rate_absent": True,
        "coverage_and_precision_reported": True,
        **retention_gates,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete" if all(gates.values()) else "failed",
        "implementation": "model.shared-bias-core-saturation-adjudication.v1",
        "candidate_selection": False,
        "programs": list(PROGRAMS),
        "final_checkpoint_step": 9143,
        "attempts_per_program_per_seed": 3072,
        "final_arm": FINAL_ARM,
        "arm_summaries": summaries,
        "paired_seed_comparisons": comparisons,
        "ugi_retention": retention,
        "finite_component_catalogue_arm_summary": dict(catalogue_summary),
        "finite_component_catalogue_comparison": catalogue_comparison,
        "finite_component_catalogue_source": pin_record(catalogue_adjudication_path, repo),
        "evidence": evidence,
        "gates": gates,
        "nonclaims": [
            "Exact L1 replay is transform consistency, not synthesis-success probability.",
            "The three training seeds are descriptive independent units; molecule attempts are not retraining replicates.",
            "The finite catalogue is a separately frozen method under the same molecular inputs and attempt budget.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    if result["status"] != "complete":
        raise SynthesisProgramProductionAdjudicationError(
            f"core-saturation adjudication gates failed: {gates}"
        )
    return result


__all__ = [
    "FINAL_ARM",
    "RESULT_SCHEMA",
    "adjudicate_core_saturation_production",
]
