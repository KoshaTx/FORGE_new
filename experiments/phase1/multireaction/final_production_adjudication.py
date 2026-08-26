"""Adjudicate the promoted FORGE arm against frozen controls and catalogue runs.

The promoted BL-core-constrained arm was trained after the original four-arm Transformer study.
This module therefore keeps two ideas separate:

* Ugi retention and cross-method summaries are paired by independent training seed; and
* the older null and cyclic arms are descriptive controls, not one-factor causal interventions on
  the promoted model.  The separately frozen program-semantic intervention supplies that test.

All input run directories are independently verified before any number is emitted.  Historical
production aggregates are never rewritten.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments._runtime import verify_run_directory
from experiments._runtime.source import source_fingerprint
from forge.core.hashing import pin_record, resolve_pin, sha256_file
from forge.core.io import atomic_write, iter_jsonl, read_json_object, write_json

from .production_adjudication import (
    CONDITIONED_ARM,
    CYCLIC_ARM,
    POSTHOC_ARM,
    UGI_PROGRAM,
    SynthesisProgramProductionAdjudicationError,
    _adjudicate_retention,
    _artifact_path,
    _finite_catalogue_comparison,
    _load_catalogue_run,
    _load_production_run,
    _number,
    _optional_number,
    _paired_descriptive_comparison,
)

ADJUDICATION_SCHEMA = "forge.final_bl_core_production_adjudication.v1"
FINAL_EXPERIMENT_ID = "phase1-bl-core-constrained-production-h100"
FINAL_IMPLEMENTATION = "model.transformer-mechanism-study.v1"
FINAL_CONFIG_SCHEMA = "forge.transformer_mechanism_study_config.v1"
FINAL_RESULT_SCHEMA = "forge.transformer_mechanism_study_result.v1"
TRAINING_RESULT_SCHEMA = "forge.synthesis_program_production_training_result.v1"
EVALUATION_RESULT_SCHEMA = "forge.synthesis_program_production_evaluation_result.v1"
SAMPLES_SCHEMA = "forge.synthesis_program_production_samples.v1"
FINAL_ARM = "bl_core_constrained_repeat_aware"
CONTROL_ARMS = ("ugi_only_conditioned", POSTHOC_ARM, CYCLIC_ARM)
CHEMISTRY_INPUTS = (
    "multireaction_atlas",
    "multireaction_splits",
    "program_config",
    "qualified_reaction_families",
    "qualified_ugi_reactions",
    "ugi_assignments",
)
PAPER_METRICS = (
    "raw_valid_fraction",
    "connected_fraction",
    "exact_l1_decomposition_coverage",
    "exact_forward_replay_precision",
    "decomposition_abstention_fraction",
    "decomposition_ambiguity_fraction",
    "exact_l1_yield_per_attempt",
    "unique_exact_l1_products_per_1000_attempts",
    "unique_open_ended_exact_l1_products_per_1000_attempts",
    "whole_lipid_novelty_fraction",
    "component_novelty_fraction",
    "internal_diversity",
    "effective_component_count",
)
CATALOGUE_PAPER_METRICS = {
    "raw_valid_fraction": "valid_fraction",
    "connected_fraction": "connected_fraction",
    "exact_l1_decomposition_coverage": "retro_decomposition_coverage_among_valid",
    "exact_forward_replay_precision": "retro_transform_precision",
    "exact_l1_yield_per_attempt": "exact_l1_yield_per_attempt",
    "unique_exact_l1_products_per_1000_attempts": ("unique_exact_l1_products_per_1000_attempts"),
    "unique_open_ended_exact_l1_products_per_1000_attempts": (
        "unique_open_ended_exact_l1_products_per_1000_attempts"
    ),
    "whole_product_novelty_fraction": "whole_product_novel_to_train_fraction",
    "internal_diversity": "mean_pairwise_ecfp4_distance",
}


def seed_summary(values: Sequence[float | None]) -> dict[str, Any]:
    """Summarize independent seed cells without imputing undefined metrics."""

    cells = list(values)
    finite = [float(value) for value in cells if value is not None]
    complete = len(finite) == len(cells)
    return {
        "by_seed": cells,
        "mean": float(np.mean(finite)) if finite and complete else None,
        "sample_standard_deviation": (
            float(np.std(finite, ddof=1)) if complete and len(finite) >= 2 else None
        ),
        "defined_seed_count": len(finite),
        "expected_seed_count": len(cells),
        "all_seed_cells_defined": complete,
        "undefined_cells_imputed": False,
    }


def _all_true(mapping: object) -> bool:
    return (
        isinstance(mapping, Mapping)
        and bool(mapping)
        and all(value is True for value in mapping.values())
    )


def _pin_tuple(pin: object, *, label: str) -> tuple[str, str]:
    if not isinstance(pin, Mapping):
        raise SynthesisProgramProductionAdjudicationError(f"{label} is not a pin")
    path = pin.get("path")
    digest = pin.get("sha256")
    if not isinstance(path, str) or not path or not isinstance(digest, str) or len(digest) != 64:
        raise SynthesisProgramProductionAdjudicationError(f"{label} is malformed")
    return path, digest


def _load_final_run(run_dir: Path, repo: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    verify_run_directory(run_dir)
    run = read_json_object(
        run_dir / "run.json",
        error=SynthesisProgramProductionAdjudicationError,
        label="final production run manifest",
    )
    if (
        run.get("experiment_id") != FINAL_EXPERIMENT_ID
        or run.get("profile") != "full"
        or run.get("status") != "complete"
        or set(run.get("stages", {})) != {"study"}
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "final adjudication requires a complete full-profile promoted production run"
        )
    manifest = read_json_object(
        run_dir / "stages/study/manifest.json",
        error=SynthesisProgramProductionAdjudicationError,
        label="final production stage manifest",
    )
    if (
        manifest.get("implementation") != FINAL_IMPLEMENTATION
        or manifest.get("status") != "complete"
        or manifest.get("profile") != "full"
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "final production run used the wrong study implementation or profile"
        )

    paths = {
        label: _artifact_path(run_dir, "study", manifest, label)
        for label in ("result", "training_result", "evaluation_result", "samples")
    }
    study = read_json_object(
        paths["result"],
        error=SynthesisProgramProductionAdjudicationError,
        label="final study result",
    )
    training = read_json_object(
        paths["training_result"],
        error=SynthesisProgramProductionAdjudicationError,
        label="final training result",
    )
    evaluation = read_json_object(
        paths["evaluation_result"],
        error=SynthesisProgramProductionAdjudicationError,
        label="final evaluation result",
    )
    replicate = int(run.get("replicate", -1))
    seed = int(evaluation.get("seed", -1))
    if (
        study.get("schema_version") != FINAL_RESULT_SCHEMA
        or training.get("schema_version") != TRAINING_RESULT_SCHEMA
        or evaluation.get("schema_version") != EVALUATION_RESULT_SCHEMA
        or any(document.get("status") != "pass" for document in (study, training, evaluation))
        or any(document.get("profile") != "full" for document in (study, training, evaluation))
        or any(
            int(document.get("replicate", -1)) != replicate
            for document in (study, training, evaluation)
        )
        or any(int(document.get("seed", -1)) != seed for document in (study, training))
        or study.get("study") != "bl_core_constrained_production"
        or study.get("arms") != [FINAL_ARM]
        or set(training.get("arms", {})) != {FINAL_ARM}
        or set(evaluation.get("checkpoint_metrics", {})) != {FINAL_ARM}
        or set(evaluation.get("component_disjoint_metrics", {})) != {FINAL_ARM}
    ):
        raise SynthesisProgramProductionAdjudicationError(
            f"promoted production identity is inconsistent for replicate {replicate}"
        )
    if (
        not _all_true(training.get("gates"))
        or not _all_true(evaluation.get("gates"))
        or study.get("calls") != {"oracle": 0, "route": 0}
        or study.get("candidate_selection") is not False
        or evaluation.get("calls") != {"oracle": 0, "route": 0}
        or evaluation.get("selection", {}).get("candidate_selection") is not False
    ):
        raise SynthesisProgramProductionAdjudicationError(
            f"promoted production gates failed for replicate {replicate}"
        )
    referenced = {
        "training_result": training,
        "evaluation_result": evaluation,
        "samples": None,
    }
    for label, document in referenced.items():
        record = study.get(label.removesuffix("_result") if label != "samples" else label)
        if not isinstance(record, Mapping) or record.get("sha256") != str(
            sha256_file(paths[label])
        ):
            raise SynthesisProgramProductionAdjudicationError(
                f"final study does not bind its {label} artifact"
            )
        if document is not None and record.get("bytes") != paths[label].stat().st_size:
            raise SynthesisProgramProductionAdjudicationError(
                f"final study records the wrong {label} size"
            )

    config_pin = manifest.get("config")
    config_relative, config_sha256 = _pin_tuple(config_pin, label="final production config")
    config_path = resolve_pin(
        {"path": config_relative, "sha256": config_sha256},
        repo,
        label="final production config",
    )
    config = read_json_object(
        config_path,
        error=SynthesisProgramProductionAdjudicationError,
        label="final production config",
    )
    external_inputs = manifest.get("external_inputs")
    if (
        config.get("schema_version") != FINAL_CONFIG_SCHEMA
        or config.get("study") != "bl_core_constrained_production"
        or config.get("authorization", {}).get("authorized") is not True
        or config.get("promotion", {}).get("promoted_arm") != FINAL_ARM
        or config.get("checkpoint_selection") != "fixed_final_step_without_calibration_selection"
        or config.get("candidate_selection") is not False
        or config.get("route_calls") != 0
        or config.get("oracle_calls") != 0
        or not isinstance(external_inputs, Mapping)
        or dict(config.get("inputs", {}).get("base_design", {}))
        != dict(external_inputs.get("base_design", {}))
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "final production config does not preserve the promoted-arm contract"
        )
    return {
        "run_dir": run_dir,
        "run": run,
        "manifest": manifest,
        "study": study,
        "training": training,
        "evaluation": evaluation,
        "paths": paths,
        "config": config,
        "config_path": config_path,
        "design_pin": dict(external_inputs["base_design"]),
        "external_inputs": dict(external_inputs),
    }


def _chemistry_pins(
    item: Mapping[str, Any], *, label: str, repo: Path
) -> dict[str, dict[str, Any]]:
    inputs = item.get("external_inputs")
    if not isinstance(inputs, Mapping):
        raise SynthesisProgramProductionAdjudicationError(f"{label} has no external inputs")
    output: dict[str, dict[str, Any]] = {}
    for key in CHEMISTRY_INPUTS:
        pin = inputs.get(key)
        relative, digest = _pin_tuple(pin, label=f"{label}.{key}")
        resolved = resolve_pin({"path": relative, "sha256": digest}, repo, label=f"{label}.{key}")
        output[key] = pin_record(resolved, repo)
    return output


def _final_metrics(
    evaluation: Mapping[str, Any], arm_id: str, final_step: str
) -> Mapping[str, Any]:
    try:
        metrics = evaluation["checkpoint_metrics"][arm_id][final_step]["heldout"]
    except (KeyError, TypeError) as error:
        raise SynthesisProgramProductionAdjudicationError(
            f"evaluation omits {arm_id} final heldout metrics"
        ) from error
    if not isinstance(metrics, Mapping):
        raise SynthesisProgramProductionAdjudicationError(
            f"evaluation has malformed {arm_id} final heldout metrics"
        )
    return metrics


def _ledger_counts(
    path: Path,
    *,
    arm_id: str,
    programs: Sequence[str],
    final_step: int,
) -> dict[str, dict[str, int | float | None]]:
    expected_programs = set(programs)
    counts: dict[str, Counter[str]] = {program: Counter() for program in programs}
    header: Mapping[str, Any] | None = None
    for index, row in enumerate(iter_jsonl(path)):
        if index == 0 and isinstance(row, Mapping) and "schema_version" in row:
            header = row
            continue
        if not isinstance(row, Mapping):
            raise SynthesisProgramProductionAdjudicationError("sample ledger contains a non-object")
        if (
            row.get("arm_id") != arm_id
            or row.get("evaluation_split") != "heldout"
            or int(row.get("checkpoint_step", -1)) != final_step
        ):
            continue
        program = str(row.get("program_id"))
        if program not in expected_programs:
            raise SynthesisProgramProductionAdjudicationError(
                f"sample ledger contains unexpected heldout program {program!r}"
            )
        counter = counts[program]
        counter["attempts"] += 1
        if row.get("valid") is True:
            counter["valid"] += 1
        if row.get("valid") is True and row.get("exact_l1_program") is True:
            counter["exact_l1"] += 1
            if int(row.get("forward_verified_trace_count", 0)) > 0:
                counter["forward_verified"] += 1
    if not isinstance(header, Mapping) or header.get("schema_version") != SAMPLES_SCHEMA:
        raise SynthesisProgramProductionAdjudicationError(
            "sample ledger has no supported schema header"
        )
    output: dict[str, dict[str, int | float | None]] = {}
    for program, counter in counts.items():
        attempts = counter["attempts"]
        valid = counter["valid"]
        exact_l1 = counter["exact_l1"]
        output[program] = {
            "attempts": attempts,
            "valid": valid,
            "exact_l1": exact_l1,
            "forward_verified": counter["forward_verified"],
            "raw_valid_fraction": valid / attempts if attempts else None,
            "exact_l1_yield_per_attempt": exact_l1 / attempts if attempts else None,
            "exact_l1_coverage_among_valid": exact_l1 / valid if valid else None,
            "forward_replay_precision_among_exact_l1": (
                counter["forward_verified"] / exact_l1 if exact_l1 else None
            ),
        }
    return output


def _arm_summary(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    arm_id: str,
    programs: Sequence[str],
    final_step: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for program in programs:
        rows = [
            _final_metrics(evaluation, arm_id, final_step)[program] for evaluation in evaluations
        ]
        result[program] = {
            metric: seed_summary(
                [
                    _optional_number(row.get(metric), label=f"{arm_id}.{program}.{metric}")
                    for row in rows
                ]
            )
            for metric in PAPER_METRICS
        }
    return result


def _arm_comparison(
    left_evaluations: Sequence[Mapping[str, Any]],
    right_evaluations: Sequence[Mapping[str, Any]],
    *,
    left_arm: str,
    right_arm: str,
    programs: Sequence[str],
    final_step: str,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    decision = design["ugi_retention"]["decision_rule"]
    output: dict[str, Any] = {}
    for program in programs:
        left_rows = [
            _final_metrics(evaluation, left_arm, final_step)[program]
            for evaluation in left_evaluations
        ]
        right_rows = [
            _final_metrics(evaluation, right_arm, final_step)[program]
            for evaluation in right_evaluations
        ]
        output[program] = {
            metric: _paired_descriptive_comparison(
                [
                    _optional_number(row.get(metric), label=f"{left_arm}.{program}.{metric}")
                    for row in left_rows
                ],
                [
                    _optional_number(row.get(metric), label=f"{right_arm}.{program}.{metric}")
                    for row in right_rows
                ],
                left_label="final_forge",
                right_label=right_arm,
                resamples=int(decision["resamples"]),
                seed=int(decision["seed"]),
                confidence_level=float(decision["confidence_level"]),
            )
            for metric in PAPER_METRICS
        }
    return output


def _catalogue_summary(
    catalogues: Sequence[Mapping[str, Any]], programs: Sequence[str]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for program in programs:
        metric_rows = [item["result"]["metrics"]["per_program"][program] for item in catalogues]
        sampled_rows = [item["result"]["sampled_component_metrics"][program] for item in catalogues]
        summaries = {
            label: seed_summary(
                [
                    _optional_number(row.get(field), label=f"catalogue.{program}.{label}")
                    for row in metric_rows
                ]
            )
            for label, field in CATALOGUE_PAPER_METRICS.items()
        }
        for label in ("component_novelty_fraction", "effective_component_count"):
            summaries[label] = seed_summary(
                [
                    _optional_number(row.get(label), label=f"catalogue.{program}.{label}")
                    for row in sampled_rows
                ]
            )
        output[program] = summaries
    return output


def _snapshot_evidence(
    *,
    output_path: Path,
    repo: Path,
    group: str,
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in items:
        replicate = int(item["run"]["replicate"])
        destination = output_path.parent / "evidence" / group / f"replicate_{replicate}"
        if group == "final_forge":
            sources = {
                "run_manifest": item["run_dir"] / "run.json",
                "stage_manifest": item["run_dir"] / "stages/study/manifest.json",
                "study_result": item["paths"]["result"],
                "training_result": item["paths"]["training_result"],
                "evaluation_result": item["paths"]["evaluation_result"],
                "evaluation_samples": item["paths"]["samples"],
            }
            seed = int(item["evaluation"]["seed"])
        elif group == "controls":
            sources = {
                "run_manifest": item["run_dir"] / "run.json",
                "training_stage_manifest": item["run_dir"] / "stages/training/manifest.json",
                "evaluation_stage_manifest": item["run_dir"] / "stages/evaluation/manifest.json",
                "training_result": item["training_path"],
                "evaluation_result": item["evaluation_path"],
                "evaluation_samples": item["samples_path"],
            }
            seed = int(item["evaluation"]["seed"])
        else:
            sources = {
                "run_manifest": item["run_dir"] / "run.json",
                "catalogue_stage_manifest": item["run_dir"] / "stages/catalogue/manifest.json",
                "catalogue_result": item["result_path"],
                "catalogue_samples": item["samples_path"],
            }
            seed = int(item["result"]["seed"])
        pins: dict[str, dict[str, Any]] = {}
        for label, source in sources.items():
            suffix = ".jsonl.gz" if label.endswith("samples") else ".json"
            target = destination / f"{label}{suffix}"
            atomic_write(target, source.read_bytes())
            pins[label] = pin_record(target, repo)
        evidence.append(
            {
                "replicate": replicate,
                "seed": seed,
                "run_id": item["run"]["run_id"],
                **pins,
            }
        )
    return evidence


def adjudicate_final_production_runs(
    final_run_dirs: Sequence[Path],
    control_run_dirs: Sequence[Path],
    catalogue_run_dirs: Sequence[Path],
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Verify and aggregate the three promoted, control and catalogue replicates."""

    repo = repo.resolve()
    output_path = output_path.resolve()
    try:
        output_path.relative_to(repo)
    except ValueError as error:
        raise SynthesisProgramProductionAdjudicationError(
            "final adjudication output must stay inside the repository"
        ) from error
    final = sorted(
        (_load_final_run(path, repo) for path in final_run_dirs),
        key=lambda item: int(item["run"]["replicate"]),
    )
    controls = sorted(
        (_load_production_run(path) for path in control_run_dirs),
        key=lambda item: int(item["run"]["replicate"]),
    )
    catalogues = sorted(
        (_load_catalogue_run(path) for path in catalogue_run_dirs),
        key=lambda item: int(item["run"]["replicate"]),
    )
    if any(len(group) != 3 for group in (final, controls, catalogues)):
        raise SynthesisProgramProductionAdjudicationError(
            "final adjudication requires exactly three final, control and catalogue runs"
        )
    expected_replicates = [0, 1, 2]
    replicate_sets = [
        [int(item["run"]["replicate"]) for item in group] for group in (final, controls, catalogues)
    ]
    if any(values != expected_replicates for values in replicate_sets):
        raise SynthesisProgramProductionAdjudicationError(
            "final, control and catalogue replicate ids must each be [0, 1, 2]"
        )

    design_pins = {
        _pin_tuple(item["design_pin"], label="production design") for item in final + controls
    }
    if len(design_pins) != 1:
        raise SynthesisProgramProductionAdjudicationError(
            "promoted and control runs do not share the frozen Transformer base design"
        )
    design_relative, design_sha256 = next(iter(design_pins))
    design_path = resolve_pin(
        {"path": design_relative, "sha256": design_sha256},
        repo,
        label="Transformer production design",
    )
    design = read_json_object(
        design_path,
        error=SynthesisProgramProductionAdjudicationError,
        label="Transformer production design",
    )
    expected_seeds = [int(value) for value in design["training"]["replicate_seeds"]]
    seed_sets = [
        [int(item["evaluation"]["seed"]) for item in final],
        [int(item["evaluation"]["seed"]) for item in controls],
        [int(item["result"]["seed"]) for item in catalogues],
    ]
    if any(values != expected_seeds for values in seed_sets):
        raise SynthesisProgramProductionAdjudicationError(
            "final, control and catalogue seeds differ from the frozen design"
        )
    source_hashes = {
        str(item["run"].get("source_sha256")) for item in final + controls + catalogues
    }
    if len(source_hashes) != 1:
        raise SynthesisProgramProductionAdjudicationError(
            "final, control and catalogue runs do not share one executable source"
        )
    for group in (final, controls, catalogues):
        if len({str(item["run"].get("spec_sha256")) for item in group}) != 1:
            raise SynthesisProgramProductionAdjudicationError(
                "replicates within one run group do not share one experiment specification"
            )

    chemistry_groups = (
        [
            _chemistry_pins(item, label=f"final replicate {index}", repo=repo)
            for index, item in enumerate(final)
        ]
        + [
            _chemistry_pins(
                {"external_inputs": item["manifests"]["evaluation"]["external_inputs"]},
                label=f"control replicate {index}",
                repo=repo,
            )
            for index, item in enumerate(controls)
        ]
        + [
            _chemistry_pins(
                {"external_inputs": item["manifest"]["external_inputs"]},
                label=f"catalogue replicate {index}",
                repo=repo,
            )
            for index, item in enumerate(catalogues)
        ]
    )
    if any(pins != chemistry_groups[0] for pins in chemistry_groups[1:]):
        raise SynthesisProgramProductionAdjudicationError(
            "final, control and catalogue runs do not share the same molecular inputs"
        )

    programs = tuple(str(value) for value in design["programs"])
    final_step = int(design["training"]["checkpoint_steps"][-1])
    attempts = int(
        design["evaluation"]["native_sampling"][
            "heldout_samples_per_supported_program_at_final_checkpoint_per_seed"
        ]
    )
    final_ledgers = [
        _ledger_counts(
            item["paths"]["samples"], arm_id=FINAL_ARM, programs=programs, final_step=final_step
        )
        for item in final
    ]
    null_ledgers = [
        _ledger_counts(
            item["samples_path"], arm_id=POSTHOC_ARM, programs=programs, final_step=final_step
        )
        for item in controls
    ]
    for index, (final_item, control_item, catalogue_item) in enumerate(
        zip(final, controls, catalogues, strict=True)
    ):
        final_metrics = _final_metrics(final_item["evaluation"], FINAL_ARM, str(final_step))
        control_metrics = _final_metrics(control_item["evaluation"], POSTHOC_ARM, str(final_step))
        catalogue_metrics = catalogue_item["result"]["metrics"]["per_program"]
        for program in programs:
            if (
                final_ledgers[index][program]["attempts"] != attempts
                or null_ledgers[index][program]["attempts"] != attempts
                or int(final_metrics[program]["samples"]) != attempts
                or int(control_metrics[program]["samples"]) != attempts
                or int(catalogue_metrics[program]["samples"]) != attempts
            ):
                raise SynthesisProgramProductionAdjudicationError(
                    f"replicate {index} does not preserve {attempts} attempts for {program}"
                )
            forbidden = "reductive_amination_substructure_hit_rate"
            if (
                forbidden in final_metrics[program]
                or forbidden in control_metrics[program]
                or forbidden in catalogue_metrics[program]
            ):
                raise SynthesisProgramProductionAdjudicationError(
                    "the forbidden reductive-amination substructure statistic was reported"
                )
            for ledger, metrics, label in (
                (final_ledgers[index][program], final_metrics[program], "final"),
                (null_ledgers[index][program], control_metrics[program], "null"),
            ):
                if not math.isclose(
                    _number(
                        ledger["exact_l1_yield_per_attempt"],
                        label=f"{label} ledger exact-L1 yield",
                    ),
                    _number(
                        metrics["exact_l1_yield_per_attempt"],
                        label=f"{label} result exact-L1 yield",
                    ),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise SynthesisProgramProductionAdjudicationError(
                        f"{label} ledger and result disagree for replicate {index}, {program}"
                    )

    final_evaluations = [item["evaluation"] for item in final]
    control_evaluations = [item["evaluation"] for item in controls]
    merged_evaluations: list[dict[str, Any]] = []
    for final_evaluation, control_evaluation in zip(
        final_evaluations, control_evaluations, strict=True
    ):
        merged_evaluations.append(
            {
                "checkpoint_metrics": {
                    "ugi_only_conditioned": control_evaluation["checkpoint_metrics"][
                        "ugi_only_conditioned"
                    ],
                    CONDITIONED_ARM: final_evaluation["checkpoint_metrics"][FINAL_ARM],
                },
                "component_disjoint_metrics": {
                    "ugi_only_conditioned": control_evaluation["component_disjoint_metrics"][
                        "ugi_only_conditioned"
                    ],
                    CONDITIONED_ARM: final_evaluation["component_disjoint_metrics"][FINAL_ARM],
                },
            }
        )
    retention, retention_gates = _adjudicate_retention(merged_evaluations, design)
    retention["challenger_arm"] = FINAL_ARM

    aliased_final = [
        {
            "evaluation": {
                "checkpoint_metrics": {
                    CONDITIONED_ARM: item["evaluation"]["checkpoint_metrics"][FINAL_ARM]
                }
            }
        }
        for item in final
    ]
    catalogue_comparison = _finite_catalogue_comparison(aliased_final, catalogues, design)
    catalogue_comparison["forge_arm"] = FINAL_ARM

    summaries = {
        FINAL_ARM: _arm_summary(
            final_evaluations,
            arm_id=FINAL_ARM,
            programs=programs,
            final_step=str(final_step),
        ),
        POSTHOC_ARM: _arm_summary(
            control_evaluations,
            arm_id=POSTHOC_ARM,
            programs=programs,
            final_step=str(final_step),
        ),
        CYCLIC_ARM: _arm_summary(
            control_evaluations,
            arm_id=CYCLIC_ARM,
            programs=programs,
            final_step=str(final_step),
        ),
        "ugi_only_conditioned": _arm_summary(
            control_evaluations,
            arm_id="ugi_only_conditioned",
            programs=(UGI_PROGRAM,),
            final_step=str(final_step),
        ),
    }
    comparisons = {
        "final_vs_ugi_only": _arm_comparison(
            final_evaluations,
            control_evaluations,
            left_arm=FINAL_ARM,
            right_arm="ugi_only_conditioned",
            programs=(UGI_PROGRAM,),
            final_step=str(final_step),
            design=design,
        ),
        "final_vs_shared_null_posthoc": _arm_comparison(
            final_evaluations,
            control_evaluations,
            left_arm=FINAL_ARM,
            right_arm=POSTHOC_ARM,
            programs=programs,
            final_step=str(final_step),
            design=design,
        ),
        "final_vs_cyclic_program_id": _arm_comparison(
            final_evaluations,
            control_evaluations,
            left_arm=FINAL_ARM,
            right_arm=CYCLIC_ARM,
            programs=programs,
            final_step=str(final_step),
            design=design,
        ),
    }
    evidence = {
        "final_forge": _snapshot_evidence(
            output_path=output_path, repo=repo, group="final_forge", items=final
        ),
        "controls": _snapshot_evidence(
            output_path=output_path, repo=repo, group="controls", items=controls
        ),
        "catalogue": _snapshot_evidence(
            output_path=output_path, repo=repo, group="catalogue", items=catalogues
        ),
    }
    all_retention_gates = all(retention_gates.values())
    gates = {
        "three_final_replicates": True,
        "three_control_replicates": True,
        "three_catalogue_replicates": True,
        "paired_replicates_and_seeds": True,
        "one_executable_source": True,
        "one_transformer_base_design": True,
        "common_molecular_inputs": True,
        "attempt_budgets_match": True,
        "coverage_and_precision_reported": True,
        "candidate_selection_absent": True,
        "route_or_oracle_calls_zero": True,
        "repairs_or_retries_absent": True,
        "reductive_amination_substructure_rate_absent": True,
        "held_reaction_family_not_a_hard_gate": (
            design["evaluation"]["held_reaction_family"]["hard_gate"] is False
        ),
        **retention_gates,
    }
    result = {
        "schema_version": ADJUDICATION_SCHEMA,
        "status": "complete",
        "scientific_decision": {
            "final_forge_ugi_retention_noninferior": all_retention_gates,
            "negative_result_is_valid": not all_retention_gates,
            "control_comparisons_are_descriptive": True,
            "program_semantic_causal_claim_deferred_to_intervention": True,
        },
        "gates": gates,
        "attempts_per_program_per_seed": attempts,
        "final_checkpoint_step": final_step,
        "programs": list(programs),
        "arm_summaries": summaries,
        "finite_component_catalogue_arm_summary": _catalogue_summary(catalogues, programs),
        "paired_seed_comparisons": comparisons,
        "attempt_ledger_audit": {
            "final_forge_by_seed": final_ledgers,
            "shared_null_posthoc_by_seed": null_ledgers,
            "filter": "valid_and_exact_l1_program",
            "same_attempt_count_not_paired_molecule_rows": True,
        },
        "ugi_retention": retention,
        "finite_component_catalogue_comparison": catalogue_comparison,
        "design": pin_record(design_path, repo),
        "final_config": pin_record(final[0]["config_path"], repo),
        "common_molecular_inputs": chemistry_groups[0],
        "input_source_sha256": next(iter(source_hashes)),
        "input_spec_sha256_by_group": {
            "final_forge": final[0]["run"]["spec_sha256"],
            "controls": controls[0]["run"]["spec_sha256"],
            "catalogue": catalogues[0]["run"]["spec_sha256"],
        },
        "adjudicator_source_sha256": source_fingerprint(repo),
        "implementation": pin_record(Path(__file__), repo),
        "evidence": evidence,
        "calls": {"route": 0, "oracle": 0},
        "candidate_selection": False,
        "nonclaims": [
            "Exact L1 replay is transform consistency, not synthesis-success probability.",
            "The shared-null and cyclic arms predate the promoted BL-core constraint and are descriptive controls, not one-factor interventions on the final model.",
            "Program-semantic causality requires the separately frozen paired inference intervention.",
            "BL and LX exact replay do not inherit Ugi prospective validation.",
            "Held-reaction-family generalization is secondary and not a hard gate.",
        ],
    }
    write_json(output_path, result)
    return result


__all__ = [
    "ADJUDICATION_SCHEMA",
    "FINAL_ARM",
    "adjudicate_final_production_runs",
    "seed_summary",
]
