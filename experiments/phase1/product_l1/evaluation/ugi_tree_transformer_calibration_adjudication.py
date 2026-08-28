"""Adjudicate the paired tree-Transformer checkpoint calibration against frozen v0.

The calibration fold may select one development architecture and checkpoint.  It never selects a
molecule, and held-component identities never enter the decision.  Every comparison is reconstructed
from the method-blind per-attempt ledgers rather than trusting a manually transcribed summary.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments._runtime import verify_run_directory
from experiments._runtime.source import source_fingerprint
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json
from forge.model.common_ugi_benchmark import load_ugi_identity_references
from forge.model.reaction_program_evaluation import effective_count

CONFIG_SCHEMA = "forge.ugi_tree_transformer_calibration_adjudication_config.v1"
RESULT_SCHEMA = "forge.ugi_tree_transformer_calibration_adjudication.v1"
TRANSFORMER_RESULT_SCHEMA = "forge.ugi_tree_transformer_checkpoint_calibration.v1"
REFERENCE_RESULT_SCHEMA = "forge.ugi_reference_checkpoint_calibration.v1"
STAGE_ID = "calibration"
ROLES = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")


class UgiTreeTransformerCalibrationAdjudicationError(ValueError):
    """The calibration evidence or frozen selection contract changed."""


@dataclass(frozen=True)
class CheckpointEvidence:
    """Aligned per-attempt evidence for one model checkpoint."""

    arm_id: str
    checkpoint_step: int
    method_id: str
    valid: np.ndarray
    exact_l1: np.ndarray
    local_supported: np.ndarray
    role_supported: np.ndarray
    tail_supported: np.ndarray
    component_eligible: np.ndarray
    component_novel: np.ndarray
    primary_smiles: tuple[str | None, ...]
    generated_components: tuple[str, ...]
    metrics: dict[str, float]


@dataclass(frozen=True)
class RunEvidence:
    """Authenticated result and detail archive from one completed calibration run."""

    run_dir: Path
    run: dict[str, Any]
    result_path: Path
    result: dict[str, Any]
    archive_path: Path


def _artifact_path(
    run_dir: Path,
    manifest: dict[str, Any],
    label: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    record = artifacts.get(label) if isinstance(artifacts, dict) else None
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise UgiTreeTransformerCalibrationAdjudicationError(
            f"calibration run has no {label!r} artifact"
        )
    path = run_dir / "stages" / STAGE_ID / str(record["path"])
    if sha256_file(path) != record.get("sha256"):
        raise UgiTreeTransformerCalibrationAdjudicationError(
            f"calibration artifact digest changed: {label}"
        )
    return path


def _load_run(run_dir: Path) -> RunEvidence:
    verify_run_directory(run_dir)
    run = read_json_object(
        run_dir / "run.json",
        error=UgiTreeTransformerCalibrationAdjudicationError,
        label="calibration run",
    )
    manifest = read_json_object(
        run_dir / "stages" / STAGE_ID / "manifest.json",
        error=UgiTreeTransformerCalibrationAdjudicationError,
        label="calibration manifest",
    )
    if (
        run.get("status") != "complete"
        or run.get("profile") != "full"
        or int(run.get("replicate", -1)) != 0
        or run.get("stages", {}).get(STAGE_ID, {}).get("status") != "complete"
        or manifest.get("backend") != "modal"
    ):
        raise UgiTreeTransformerCalibrationAdjudicationError(
            "calibration evidence is not one completed full-profile Modal run"
        )
    result_path = _artifact_path(run_dir, manifest, "result")
    archive_path = _artifact_path(run_dir, manifest, "checkpoint_calibrations")
    result = read_json_object(
        result_path,
        error=UgiTreeTransformerCalibrationAdjudicationError,
        label="calibration result",
    )
    return RunEvidence(run_dir, run, result_path, result, archive_path)


def _jsonl_member(archive: tarfile.TarFile, name: str, *, rows: int) -> list[dict[str, Any]]:
    try:
        member = archive.getmember(name)
    except KeyError as error:
        raise UgiTreeTransformerCalibrationAdjudicationError(
            f"calibration archive lacks {name!r}"
        ) from error
    if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
        raise UgiTreeTransformerCalibrationAdjudicationError(
            f"calibration archive member is unsafe: {name!r}"
        )
    handle = archive.extractfile(member)
    if handle is None:
        raise UgiTreeTransformerCalibrationAdjudicationError(
            f"calibration archive member is unreadable: {name!r}"
        )
    try:
        payload = gzip.decompress(handle.read()).decode("utf-8")
        values = [json.loads(line) for line in io.StringIO(payload)]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise UgiTreeTransformerCalibrationAdjudicationError(
            f"calibration archive member is malformed: {name!r}"
        ) from error
    if not values or values[0].get("rows") != rows or len(values) != rows + 1:
        raise UgiTreeTransformerCalibrationAdjudicationError(
            f"calibration archive row count changed: {name!r}"
        )
    return values[1:]


def _json_member(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    try:
        member = archive.getmember(name)
    except KeyError as error:
        raise UgiTreeTransformerCalibrationAdjudicationError(
            f"calibration archive lacks {name!r}"
        ) from error
    if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
        raise UgiTreeTransformerCalibrationAdjudicationError(
            f"calibration archive member is unsafe: {name!r}"
        )
    handle = archive.extractfile(member)
    try:
        value = json.loads(handle.read()) if handle is not None else None
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise UgiTreeTransformerCalibrationAdjudicationError(
            f"calibration archive member is malformed: {name!r}"
        ) from error
    if not isinstance(value, dict):
        raise UgiTreeTransformerCalibrationAdjudicationError(
            f"calibration archive member is not an object: {name!r}"
        )
    return value


def _trace_components(row: dict[str, Any]) -> tuple[str, ...] | None:
    if row.get("exact_l1_program") is not True or int(row.get("exact_l1_trace_count", -1)) != 1:
        return None
    traces = row.get("exact_l1_traces")
    if not isinstance(traces, list) or len(traces) != 1:
        raise UgiTreeTransformerCalibrationAdjudicationError(
            "unique exact-L1 row has no unique trace"
        )
    components = traces[0].get("components_by_role")
    if not isinstance(components, dict) or set(components) != set(ROLES):
        raise UgiTreeTransformerCalibrationAdjudicationError(
            "unique exact-L1 trace roles changed"
        )
    values: list[str] = []
    for role in ROLES:
        value = components[role]
        if not isinstance(value, str) or not value:
            raise UgiTreeTransformerCalibrationAdjudicationError(
                f"unique exact-L1 {role} component changed"
            )
        values.append(value)
    return tuple(values)


def _point_metrics(
    *,
    valid: np.ndarray,
    exact_l1: np.ndarray,
    local_supported: np.ndarray,
    role_supported: np.ndarray,
    tail_supported: np.ndarray,
    component_eligible: np.ndarray,
    component_novel: np.ndarray,
    primary_smiles: tuple[str | None, ...],
    generated_components: tuple[str, ...],
) -> dict[str, float]:
    component_denominator = int(component_eligible.sum())
    if component_denominator == 0:
        raise UgiTreeTransformerCalibrationAdjudicationError(
            "checkpoint has no uniquely decomposed exact-L1 products"
        )
    count = len(valid)
    return {
        "valid_fraction_per_attempt": float(valid.mean()),
        "exact_l1_yield_per_attempt": float(exact_l1.mean()),
        "local_support_qualified_exact_l1_yield_per_attempt": float(
            local_supported.mean()
        ),
        "role_supported_exact_l1_yield_per_attempt": float(role_supported.mean()),
        "tail_supported_exact_l1_yield_per_attempt": float(tail_supported.mean()),
        "component_novelty_fraction": float(
            component_novel.sum() / component_denominator
        ),
        "effective_component_count": float(effective_count(generated_components) or 0.0),
        "unique_open_ended_whole_product_novel_exact_l1_products_per_attempt": (
            len({value for value in primary_smiles if value is not None}) / count
        ),
    }


def _checkpoint_evidence(
    archive_path: Path,
    *,
    arm_id: str,
    step: int,
    programs: int,
    training_products: set[str],
    training_components: dict[str, set[str]],
    reported_metrics: dict[str, Any],
) -> CheckpointEvidence:
    prefix = f"step_{step:04d}/assessment"
    with tarfile.open(archive_path, "r") as archive:
        common = _jsonl_member(
            archive, f"{prefix}/common/assessed_attempts.jsonl.gz", rows=programs
        )
        local = _jsonl_member(
            archive, f"{prefix}/local_chemistry/assessed_attempts.jsonl.gz", rows=programs
        )
        morphology = _jsonl_member(
            archive, f"{prefix}/role_morphology_attempts.jsonl.gz", rows=programs
        )
        common_result = _json_member(archive, f"{prefix}/common/result.json")

    valid = np.zeros(programs, dtype=np.bool_)
    exact_l1 = np.zeros(programs, dtype=np.bool_)
    local_supported = np.zeros(programs, dtype=np.bool_)
    role_supported = np.zeros(programs, dtype=np.bool_)
    tail_supported = np.zeros(programs, dtype=np.bool_)
    component_eligible = np.zeros(programs, dtype=np.bool_)
    component_novel = np.zeros(programs, dtype=np.bool_)
    primary_smiles: list[str | None] = [None] * programs
    generated_components: list[str] = []
    method_id: str | None = None

    for index, (common_row, local_row, morphology_row) in enumerate(
        zip(common, local, morphology, strict=True)
    ):
        identities = {
            (row.get("method_id"), row.get("seed"), row.get("attempt_index"))
            for row in (common_row, local_row, morphology_row)
        }
        if len(identities) != 1 or next(iter(identities))[2] != index:
            raise UgiTreeTransformerCalibrationAdjudicationError(
                f"paired attempt identity changed at row {index}"
            )
        observed_method = common_row.get("method_id")
        if not isinstance(observed_method, str) or not observed_method:
            raise UgiTreeTransformerCalibrationAdjudicationError(
                f"checkpoint method identity is malformed at row {index}"
            )
        method_id = method_id or observed_method
        if observed_method != method_id:
            raise UgiTreeTransformerCalibrationAdjudicationError(
                "checkpoint contains multiple method identities"
            )

        valid[index] = common_row.get("valid") is True
        exact_l1[index] = common_row.get("exact_l1_program") is True
        local_supported[index] = local_row.get("local_support_qualified_exact_l1") is True
        traces = morphology_row.get("trace_assessments")
        if exact_l1[index] and (not isinstance(traces, list) or not traces):
            raise UgiTreeTransformerCalibrationAdjudicationError(
                f"exact-L1 morphology trace is missing at row {index}"
            )
        trace_rows = traces if isinstance(traces, list) else []
        role_supported[index] = exact_l1[index] and any(
            row.get("all_roles_within_observed_hard_bounds") is True
            and row.get("all_role_ring_signatures_supported") is True
            for row in trace_rows
        )
        tail_supported[index] = exact_l1[index] and any(
            row.get("tails_within_observed_hard_bounds") is True
            and row.get("tail_ring_signatures_supported") is True
            for row in trace_rows
        )

        components = _trace_components(common_row)
        if components is None:
            continue
        component_eligible[index] = True
        generated_components.extend(components)
        novel = any(
            component not in training_components[role]
            for role, component in zip(ROLES, components, strict=True)
        )
        component_novel[index] = novel
        canonical = common_row.get("canonical_smiles")
        if novel and isinstance(canonical, str) and canonical not in training_products:
            primary_smiles[index] = canonical

    if method_id is None:
        raise UgiTreeTransformerCalibrationAdjudicationError("checkpoint has no attempts")
    metrics = _point_metrics(
        valid=valid,
        exact_l1=exact_l1,
        local_supported=local_supported,
        role_supported=role_supported,
        tail_supported=tail_supported,
        component_eligible=component_eligible,
        component_novel=component_novel,
        primary_smiles=tuple(primary_smiles),
        generated_components=tuple(generated_components),
    )
    common_metrics = common_result.get("common_assessment", {}).get("metrics", {})
    primary_per_1000 = common_metrics.get(
        "unique_open_ended_whole_product_novel_exact_l1_products_per_1000_attempts"
    )
    if not isinstance(primary_per_1000, (int, float)) or not math.isclose(
        metrics[
            "unique_open_ended_whole_product_novel_exact_l1_products_per_attempt"
        ],
        float(primary_per_1000) / 1000.0,
        abs_tol=1e-12,
    ):
        raise UgiTreeTransformerCalibrationAdjudicationError(
            "reconstructed primary metric differs from the common assessor"
        )
    for metric in (
        "valid_fraction_per_attempt",
        "exact_l1_yield_per_attempt",
        "local_support_qualified_exact_l1_yield_per_attempt",
        "role_supported_exact_l1_yield_per_attempt",
        "tail_supported_exact_l1_yield_per_attempt",
        "component_novelty_fraction",
        "effective_component_count",
    ):
        observed = reported_metrics.get(metric)
        if not isinstance(observed, (int, float)) or not math.isclose(
            metrics[metric], float(observed), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise UgiTreeTransformerCalibrationAdjudicationError(
                f"reconstructed {metric} differs from the reported checkpoint metric"
            )
    return CheckpointEvidence(
        arm_id=arm_id,
        checkpoint_step=step,
        method_id=method_id,
        valid=valid,
        exact_l1=exact_l1,
        local_supported=local_supported,
        role_supported=role_supported,
        tail_supported=tail_supported,
        component_eligible=component_eligible,
        component_novel=component_novel,
        primary_smiles=tuple(primary_smiles),
        generated_components=tuple(generated_components),
        metrics=metrics,
    )


def _interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def _mean_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    indices: np.ndarray,
    confidence: float,
) -> dict[str, float]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    estimates = difference[indices].mean(axis=1)
    lower, upper = _interval(estimates, confidence)
    return {
        "candidate": float(candidate.mean()),
        "reference": float(reference.mean()),
        "difference": float(difference.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
    }


def _conditional_difference(
    candidate_num: np.ndarray,
    candidate_den: np.ndarray,
    reference_num: np.ndarray,
    reference_den: np.ndarray,
    indices: np.ndarray,
    confidence: float,
) -> dict[str, float]:
    candidate_denominator = candidate_den[indices].sum(axis=1)
    reference_denominator = reference_den[indices].sum(axis=1)
    if np.any(candidate_denominator == 0) or np.any(reference_denominator == 0):
        raise UgiTreeTransformerCalibrationAdjudicationError(
            "bootstrap produced an empty component-novelty denominator"
        )
    candidate_estimates = candidate_num[indices].sum(axis=1) / candidate_denominator
    reference_estimates = reference_num[indices].sum(axis=1) / reference_denominator
    estimates = candidate_estimates - reference_estimates
    lower, upper = _interval(estimates, confidence)
    candidate_point = float(candidate_num.sum() / candidate_den.sum())
    reference_point = float(reference_num.sum() / reference_den.sum())
    return {
        "candidate": candidate_point,
        "reference": reference_point,
        "difference": candidate_point - reference_point,
        "ci_lower": lower,
        "ci_upper": upper,
    }


def _unique_rates(values: tuple[str | None, ...], indices: np.ndarray) -> np.ndarray:
    identifiers = {value: index for index, value in enumerate(sorted(set(values) - {None}))}
    encoded = np.asarray(
        [identifiers[value] if value is not None else -1 for value in values], dtype=np.int32
    )
    rates = np.empty(len(indices), dtype=np.float64)
    denominator = encoded.size
    for row_index, sample in enumerate(indices):
        observed = encoded[sample]
        rates[row_index] = np.unique(observed[observed >= 0]).size / denominator
    return rates


def _primary_difference(
    candidate: tuple[str | None, ...],
    reference: tuple[str | None, ...],
    indices: np.ndarray,
    confidence: float,
) -> dict[str, float]:
    candidate_estimates = _unique_rates(candidate, indices)
    reference_estimates = _unique_rates(reference, indices)
    estimates = candidate_estimates - reference_estimates
    lower, upper = _interval(estimates, confidence)
    candidate_point = len({value for value in candidate if value is not None}) / len(candidate)
    reference_point = len({value for value in reference if value is not None}) / len(reference)
    return {
        "candidate": candidate_point,
        "reference": reference_point,
        "difference": candidate_point - reference_point,
        "ci_lower": lower,
        "ci_upper": upper,
    }


def _candidate_comparison(
    candidate: CheckpointEvidence,
    reference: CheckpointEvidence,
    *,
    indices: np.ndarray,
    confidence: float,
) -> dict[str, dict[str, float]]:
    comparisons = {
        "valid_fraction_per_attempt": _mean_difference(
            candidate.valid, reference.valid, indices, confidence
        ),
        "exact_l1_yield_per_attempt": _mean_difference(
            candidate.exact_l1, reference.exact_l1, indices, confidence
        ),
        "local_support_qualified_exact_l1_yield_per_attempt": _mean_difference(
            candidate.local_supported, reference.local_supported, indices, confidence
        ),
        "role_supported_exact_l1_yield_per_attempt": _mean_difference(
            candidate.role_supported, reference.role_supported, indices, confidence
        ),
        "tail_supported_exact_l1_yield_per_attempt": _mean_difference(
            candidate.tail_supported, reference.tail_supported, indices, confidence
        ),
        "component_novelty_fraction": _conditional_difference(
            candidate.component_novel,
            candidate.component_eligible,
            reference.component_novel,
            reference.component_eligible,
            indices,
            confidence,
        ),
        "unique_open_ended_whole_product_novel_exact_l1_products_per_attempt": (
            _primary_difference(
                candidate.primary_smiles,
                reference.primary_smiles,
                indices,
                confidence,
            )
        ),
    }
    candidate_effective = candidate.metrics["effective_component_count"]
    reference_effective = reference.metrics["effective_component_count"]
    comparisons["effective_component_count"] = {
        "candidate": candidate_effective,
        "reference": reference_effective,
        "ratio": candidate_effective / reference_effective if reference_effective else math.nan,
    }
    return comparisons


def _gate_candidate(
    comparisons: dict[str, dict[str, float]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    margin = float(gates["absolute_noninferiority_margin"])
    noninferiority = {
        metric: values["ci_lower"] >= -margin
        for metric in gates["noninferiority_metrics"]
        if (values := comparisons[metric])
    }
    primary = comparisons[str(gates["primary_metric"])]
    primary_point = primary["difference"] >= float(
        gates["minimum_primary_absolute_improvement"]
    )
    primary_interval = (
        primary["ci_lower"] > 0.0
        if gates["primary_bootstrap_interval_must_exclude_zero"] is True
        else True
    )
    effective_count = comparisons["effective_component_count"]["ratio"] >= float(
        gates["minimum_effective_component_count_ratio"]
    )
    checks = {
        **{f"{metric}_noninferior": value for metric, value in noninferiority.items()},
        "effective_component_count_ratio": effective_count,
        "primary_absolute_improvement": primary_point,
        "primary_interval_excludes_zero": primary_interval,
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}


def _selection_key(candidate: dict[str, Any], primary_metric: str) -> tuple[Any, ...]:
    metrics = candidate["metrics"]
    return (
        -float(metrics[primary_metric]),
        -float(metrics["tail_supported_exact_l1_yield_per_attempt"]),
        -float(metrics["role_supported_exact_l1_yield_per_attempt"]),
        -float(metrics["local_support_qualified_exact_l1_yield_per_attempt"]),
        -float(metrics["exact_l1_yield_per_attempt"]),
        -float(metrics["valid_fraction_per_attempt"]),
        int(candidate["checkpoint_step"]),
        str(candidate["arm_id"]),
    )


def adjudicate_tree_transformer_calibration(
    run_dirs: list[Path],
    config_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Verify five run directories and apply the pre-output model/checkpoint selection rule."""

    repo = repo.resolve()
    output_path = output_path.resolve()
    try:
        output_path.relative_to(repo)
    except ValueError as error:
        raise UgiTreeTransformerCalibrationAdjudicationError(
            "calibration adjudication output must stay inside the repository"
        ) from error
    if output_path.exists():
        raise UgiTreeTransformerCalibrationAdjudicationError(
            f"calibration adjudication output already exists: {output_path}"
        )
    config = read_json_object(
        config_path,
        error=UgiTreeTransformerCalibrationAdjudicationError,
        label="tree-Transformer calibration adjudication config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("status") != (
        "frozen_before_full_calibration_outputs_exist"
    ):
        raise UgiTreeTransformerCalibrationAdjudicationError(
            "calibration adjudication config is not pre-output frozen"
        )
    expected = config.get("expected_evidence")
    uncertainty = config.get("uncertainty")
    gates = config.get("gates")
    reporting = config.get("reporting")
    if not all(isinstance(value, dict) for value in (expected, uncertainty, gates, reporting)):
        raise UgiTreeTransformerCalibrationAdjudicationError(
            "calibration adjudication policy is malformed"
        )
    if reporting != {
        "model_and_checkpoint_selection_only": True,
        "candidate_selection": False,
        "heldout_selects_nothing": True,
        "route_calls": 0,
        "oracle_calls": 0,
        "negative_results_are_complete_results": True,
    }:
        raise UgiTreeTransformerCalibrationAdjudicationError(
            "calibration reporting guardrails changed"
        )

    transformer_experiments = expected["transformer_experiments"]
    reference_experiment = str(expected["reference_experiment"])
    expected_experiments = set(transformer_experiments) | {reference_experiment}
    if len(run_dirs) != len(expected_experiments):
        raise UgiTreeTransformerCalibrationAdjudicationError(
            "calibration adjudication requires four Transformer runs and one v0 run"
        )
    runs = [_load_run(path.resolve()) for path in run_dirs]
    by_experiment = {str(value.run["experiment_id"]): value for value in runs}
    if set(by_experiment) != expected_experiments or len(by_experiment) != len(runs):
        raise UgiTreeTransformerCalibrationAdjudicationError(
            "calibration experiment set changed or contains duplicates"
        )

    assignments = resolve_pin(config["inputs"]["ugi_assignments"], repo, label="ugi_assignments")
    program_draw = resolve_pin(config["inputs"]["program_draw"], repo, label="program_draw")
    expected_program_draw_sha = sha256_file(program_draw)
    training_products, training_components, _ = load_ugi_identity_references(
        assignments, roles=ROLES
    )
    programs = int(expected["programs_per_checkpoint"])
    steps = [int(value) for value in expected["checkpoint_steps"]]

    candidates: list[CheckpointEvidence] = []
    for experiment_id, arm_id in transformer_experiments.items():
        evidence = by_experiment[experiment_id]
        result = evidence.result
        result_program_draw = result.get("inputs", {}).get("program_draw")
        if (
            result.get("schema_version") != TRANSFORMER_RESULT_SCHEMA
            or result.get("status") != "complete"
            or result.get("arm_id") != arm_id
            or result.get("checkpoint_steps") != steps
            or int(result.get("programs_per_checkpoint", -1)) != programs
            or result.get("sampling_device") != expected["sampling_device"]
            or result.get("paired_program_order") is not True
            or result.get("paired_random_streams") is not True
            or result.get("heldout_rows_used") is not False
            or not isinstance(result_program_draw, dict)
            or result_program_draw.get("sha256") != expected_program_draw_sha
        ):
            raise UgiTreeTransformerCalibrationAdjudicationError(
                f"Transformer calibration contract changed: {experiment_id}"
            )
        for step in steps:
            reported = result.get("checkpoint_assessments", {}).get(str(step), {}).get("metrics")
            if not isinstance(reported, dict):
                raise UgiTreeTransformerCalibrationAdjudicationError(
                    f"Transformer checkpoint summary is missing: {experiment_id}:{step}"
                )
            candidates.append(
                _checkpoint_evidence(
                    evidence.archive_path,
                    arm_id=str(arm_id),
                    step=step,
                    programs=programs,
                    training_products=training_products,
                    training_components=training_components,
                    reported_metrics=reported,
                )
            )

    reference_run = by_experiment[reference_experiment]
    reference_result = reference_run.result
    reference_program_draw = reference_result.get("inputs", {}).get("program_draw")
    reference_step = int(expected["reference_checkpoint_step"])
    if (
        reference_result.get("schema_version") != REFERENCE_RESULT_SCHEMA
        or reference_result.get("status") != "complete"
        or reference_result.get("arm_id") != expected["reference_arm"]
        or reference_result.get("checkpoint_steps") != [reference_step]
        or int(reference_result.get("programs_per_checkpoint", -1)) != programs
        or reference_result.get("sampling_device") != expected["sampling_device"]
        or reference_result.get("paired_program_order") is not True
        or reference_result.get("paired_random_streams") is not True
        or reference_result.get("heldout_rows_used") is not False
        or not isinstance(reference_program_draw, dict)
        or reference_program_draw.get("sha256") != expected_program_draw_sha
    ):
        raise UgiTreeTransformerCalibrationAdjudicationError(
            "frozen v0 calibration contract changed"
        )
    reference = _checkpoint_evidence(
        reference_run.archive_path,
        arm_id=str(expected["reference_arm"]),
        step=reference_step,
        programs=programs,
        training_products=training_products,
        training_components=training_components,
        reported_metrics=reference_result["checkpoint_assessments"][str(reference_step)][
            "metrics"
        ],
    )

    resamples = int(uncertainty["resamples"])
    confidence = float(uncertainty["confidence"])
    if (
        uncertainty.get("method")
        != "paired_nonparametric_bootstrap_over_fixed_program_indices"
        or resamples < 1000
        or not 0.5 < confidence < 1.0
    ):
        raise UgiTreeTransformerCalibrationAdjudicationError(
            "calibration bootstrap contract changed"
        )
    rng = np.random.default_rng(int(uncertainty["seed"]))
    indices = rng.integers(0, programs, size=(resamples, programs), dtype=np.int32)

    candidate_results = []
    for candidate in candidates:
        comparisons = _candidate_comparison(
            candidate, reference, indices=indices, confidence=confidence
        )
        gate = _gate_candidate(comparisons, gates)
        candidate_results.append(
            {
                "candidate_id": f"{candidate.arm_id}:step_{candidate.checkpoint_step:04d}",
                "arm_id": candidate.arm_id,
                "checkpoint_step": candidate.checkpoint_step,
                "method_id": candidate.method_id,
                "metrics": candidate.metrics,
                "versus_v0": comparisons,
                "gate": gate,
            }
        )

    primary_metric = str(gates["primary_metric"])
    eligible = [value for value in candidate_results if value["gate"]["status"] == "pass"]
    eligible.sort(key=lambda value: _selection_key(value, primary_metric))
    selected = eligible[0] if eligible else None
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "decision": "promote_transformer" if selected else "no_transformer_promotion",
        "selected_model": (
            {
                "candidate_id": selected["candidate_id"],
                "arm_id": selected["arm_id"],
                "checkpoint_step": selected["checkpoint_step"],
            }
            if selected
            else None
        ),
        "reference": {
            "arm_id": reference.arm_id,
            "checkpoint_step": reference.checkpoint_step,
            "method_id": reference.method_id,
            "metrics": reference.metrics,
        },
        "eligible_candidates": [value["candidate_id"] for value in eligible],
        "candidates": candidate_results,
        "uncertainty": dict(uncertainty),
        "gates": dict(gates),
        "inputs": {
            "config": pin_record(config_path, repo),
            "program_draw": pin_record(program_draw, repo),
            "ugi_assignments": pin_record(assignments, repo),
            "runs": {
                experiment_id: {
                    "run": artifact_record(evidence.run_dir / "run.json"),
                    "result": artifact_record(evidence.result_path),
                    "checkpoint_calibrations": artifact_record(evidence.archive_path),
                }
                for experiment_id, evidence in sorted(by_experiment.items())
            },
        },
        "adjudicator_source_sha256": source_fingerprint(repo),
        "model_and_checkpoint_selection_only": True,
        "candidate_selection": False,
        "heldout_rows_used": False,
        "route_calls": 0,
        "oracle_calls": 0,
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_path, result)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "CheckpointEvidence",
    "UgiTreeTransformerCalibrationAdjudicationError",
    "adjudicate_tree_transformer_calibration",
]
