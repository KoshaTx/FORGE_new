"""Audit frozen HeLa oracle reliability under morphology-proposal reweighting."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import stable_json as _stable_json
from forge.design.ugi_restartable_terminal_support_adapter import (
    canonical_morphology_program_bytes,
)
from forge.design.ugi_training_cache import load_ugi_training_cache

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_enriched_oracle_audit_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_enriched_oracle_audit.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_morphology_enriched_oracle_audit_ledger.v1"
EXPECTED_SCOPE = {
    "read_only": True,
    "existing_frozen_artifacts_only": True,
    "oracle_calls": 0,
    "potency_predictions_generated": False,
    "generator_trajectories_advanced": False,
    "candidate_selection": False,
    "guidance_authorized": False,
    "route_calls": 0,
    "synthesis_calls": 0,
    "proposal_calls": 0,
    "sealed_holdout_access": False,
}
EXPECTED_INPUTS = {
    "applicability_result",
    "complete_proposal_ledger",
    "complete_proposal_result",
    "curated_agile",
    "heldout_oof_ledger",
    "runner",
    "selective_risk_result",
    "source",
    "tests",
    "training_cache",
}
VIEWS = ("product", "amine", "aldehyde", "isocyanide")
DISTANCE_KINDS = ("fingerprint", "descriptor")


class UgiMorphologyEnrichedOracleAuditError(RuntimeError):
    """Raised when the frozen oracle-shift audit contract changes."""


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiMorphologyEnrichedOracleAuditError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiMorphologyEnrichedOracleAuditError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiMorphologyEnrichedOracleAuditError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiMorphologyEnrichedOracleAuditError(f"pin escapes repository: {label}") from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiMorphologyEnrichedOracleAuditError(f"pin changed: {label}")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiMorphologyEnrichedOracleAuditError(f"invalid CSV: {path}") from error


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError
                output.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise UgiMorphologyEnrichedOracleAuditError(f"invalid JSONL: {path}") from error
    return output


def _csv_gzip_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row[field] for field in fields} for row in rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as handle:
        handle.write(text.getvalue().encode())
    return output.getvalue()


def _finite(value: Any, *, label: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as error:
        raise UgiMorphologyEnrichedOracleAuditError(f"{label} is not numeric") from error
    if not math.isfinite(output):
        raise UgiMorphologyEnrichedOracleAuditError(f"{label} is not finite")
    return output


def effective_sample_size(weights: np.ndarray) -> float:
    """Return the scale-invariant importance-weight effective sample size."""

    if (
        weights.ndim != 1
        or not len(weights)
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise UgiMorphologyEnrichedOracleAuditError("invalid importance weights")
    return float(weights.sum() ** 2 / np.sum(weights**2))


def _weighted_correlation(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float | None:
    total = float(weights.sum())
    left_mean = float(np.dot(weights, left) / total)
    right_mean = float(np.dot(weights, right) / total)
    left_delta = left - left_mean
    right_delta = right - right_mean
    denominator = math.sqrt(
        float(np.dot(weights, left_delta**2)) * float(np.dot(weights, right_delta**2))
    )
    if denominator <= 0.0:
        return None
    return float(np.dot(weights, left_delta * right_delta) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def weighted_prediction_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
    covered90: np.ndarray,
) -> dict[str, float | int | None]:
    """Return decision-relevant prediction metrics under positive weights."""

    if (
        observed.ndim != 1
        or predicted.shape != observed.shape
        or weights.shape != observed.shape
        or covered90.shape != observed.shape
        or not len(observed)
        or np.any(~np.isfinite(observed))
        or np.any(~np.isfinite(predicted))
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise UgiMorphologyEnrichedOracleAuditError("invalid weighted metric arrays")
    total = float(weights.sum())
    residual = observed - predicted
    true_mean = float(np.dot(weights, observed) / total)
    denominator = float(np.dot(weights, (observed - true_mean) ** 2))
    ranked_observed = _average_ranks(observed)
    ranked_predicted = _average_ranks(predicted)
    spearman = _weighted_correlation(ranked_predicted, ranked_observed, weights)
    threshold = _weighted_quantile(predicted, weights, 0.9)
    top = predicted >= threshold
    top_mean = float(np.dot(weights[top], observed[top]) / weights[top].sum())
    return {
        "records": len(observed),
        "effective_sample_size": effective_sample_size(weights),
        "effective_sample_fraction": effective_sample_size(weights) / len(weights),
        "mae": float(np.dot(weights, np.abs(residual)) / total),
        "rmse": float(math.sqrt(np.dot(weights, residual**2) / total)),
        "r2": 1.0 - float(np.dot(weights, residual**2)) / denominator if denominator > 0 else None,
        "weighted_midrank_spearman": spearman,
        "conformal90_coverage": float(np.dot(weights, covered90.astype(float)) / total),
        "observed_mean": true_mean,
        "predicted_top_decile_threshold": threshold,
        "observed_mean_in_predicted_top_decile": top_mean,
        "predicted_top_decile_observed_gain": top_mean - true_mean,
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise UgiMorphologyEnrichedOracleAuditError("invalid weighted quantile")
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    target = probability * float(cumulative[-1])
    return float(
        values[order[min(int(np.searchsorted(cumulative, target, side="left")), len(order) - 1)]]
    )


def _program_by_product(training_cache: Path) -> dict[str, str]:
    corpus, records_by_fold = load_ugi_training_cache(training_cache)
    output: dict[str, str] = {}
    for fold in sorted(corpus.assignments_by_fold):
        assignments = corpus.assignments_by_fold[fold]
        records = records_by_fold[fold]
        for assignment, record in zip(assignments, records, strict=True):
            product = str(assignment["canonical_product_smiles"])
            digest = hashlib.sha256(canonical_morphology_program_bytes(record.program)).hexdigest()
            previous = output.get(product)
            if previous is not None and previous != digest:
                raise UgiMorphologyEnrichedOracleAuditError(
                    "one canonical product has conflicting morphology programs"
                )
            output[product] = digest
    return output


def _normalized_radius(row: Mapping[str, str], thresholds: Mapping[str, Any]) -> float:
    ratios = []
    for view in VIEWS:
        for kind in DISTANCE_KINDS:
            denominator = _finite(
                thresholds[view][kind]["interpolative_max"],
                label=f"{view} {kind} threshold",
            )
            distance = _finite(row[f"{view}_{kind}_distance"], label=f"{view} {kind} distance")
            if denominator <= 0.0 or distance < 0.0:
                raise UgiMorphologyEnrichedOracleAuditError("invalid normalized-radius input")
            ratios.append(distance / denominator)
    return float(max(ratios))


def _scheme_summary(rows: Sequence[Mapping[str, Any]], *, enriched: bool) -> dict[str, Any]:
    selected = [row for row in rows if bool(row["inside_fixed_radius"])]
    if not selected:
        raise UgiMorphologyEnrichedOracleAuditError("one scheme has no supported rows")
    weights = np.asarray(
        [float(row["proposal_over_broad_weight"]) if enriched else 1.0 for row in selected]
    )
    all_weights = np.asarray(
        [float(row["proposal_over_broad_weight"]) if enriched else 1.0 for row in rows]
    )
    return {
        "retained_records": len(selected),
        "weighted_support_coverage": float(weights.sum() / all_weights.sum()),
        "metrics": weighted_prediction_metrics(
            np.asarray([float(row["y_true"]) for row in selected]),
            np.asarray([float(row["y_pred"]) for row in selected]),
            weights,
            np.asarray([bool(row["covered90"]) for row in selected]),
        ),
    }


def _equal_scheme_summary(
    rows: Sequence[Mapping[str, Any]], schemes: Sequence[str], *, enriched: bool
) -> dict[str, Any]:
    per_scheme = {
        scheme: _scheme_summary([row for row in rows if row["scheme"] == scheme], enriched=enriched)
        for scheme in schemes
    }
    metrics = [record["metrics"] for record in per_scheme.values()]
    return {
        "per_scheme": per_scheme,
        "equal_scheme_mae": float(np.mean([float(record["mae"]) for record in metrics])),
        "equal_scheme_rmse": float(np.mean([float(record["rmse"]) for record in metrics])),
        "equal_scheme_conformal90_coverage": float(
            np.mean([float(record["conformal90_coverage"]) for record in metrics])
        ),
        "equal_scheme_weighted_midrank_spearman": float(
            np.mean([float(record["weighted_midrank_spearman"]) for record in metrics])
        ),
        "equal_scheme_top_decile_observed_gain": float(
            np.mean([float(record["predicted_top_decile_observed_gain"]) for record in metrics])
        ),
        "minimum_effective_sample_fraction": float(
            min(float(record["effective_sample_fraction"]) for record in metrics)
        ),
        "minimum_weighted_support_coverage": float(
            min(float(record["weighted_support_coverage"]) for record in per_scheme.values())
        ),
    }


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], schemes: Sequence[str], *, replicates: int, seed: int
) -> dict[str, Any]:
    labels = sorted({str(row["label"]) for row in rows})
    indexed = {(str(row["scheme"]), str(row["label"])): row for row in rows}
    if len(indexed) != len(labels) * len(schemes):
        raise UgiMorphologyEnrichedOracleAuditError("scheme-label OOF lattice is incomplete")
    random = np.random.default_rng(seed)
    mae_delta = np.empty(replicates, dtype=float)
    rmse_delta = np.empty(replicates, dtype=float)
    coverage = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled = random.choice(labels, size=len(labels), replace=True)
        broad_mae = []
        broad_rmse = []
        enriched_mae = []
        enriched_rmse = []
        enriched_coverage = []
        for scheme in schemes:
            selected = [indexed[(scheme, str(label))] for label in sampled]
            supported = [row for row in selected if bool(row["inside_fixed_radius"])]
            if not supported:
                raise UgiMorphologyEnrichedOracleAuditError(
                    "bootstrap replicate has no supported rows"
                )
            observed = np.asarray([float(row["y_true"]) for row in supported])
            predicted = np.asarray([float(row["y_pred"]) for row in supported])
            covered = np.asarray([bool(row["covered90"]) for row in supported])
            broad_metrics = weighted_prediction_metrics(
                observed, predicted, np.ones(len(supported)), covered
            )
            weighted_metrics = weighted_prediction_metrics(
                observed,
                predicted,
                np.asarray([float(row["proposal_over_broad_weight"]) for row in supported]),
                covered,
            )
            broad_mae.append(float(broad_metrics["mae"]))
            broad_rmse.append(float(broad_metrics["rmse"]))
            enriched_mae.append(float(weighted_metrics["mae"]))
            enriched_rmse.append(float(weighted_metrics["rmse"]))
            enriched_coverage.append(float(weighted_metrics["conformal90_coverage"]))
        broad_mae_mean = float(np.mean(broad_mae))
        broad_rmse_mean = float(np.mean(broad_rmse))
        mae_delta[replicate] = float(np.mean(enriched_mae)) / broad_mae_mean - 1.0
        rmse_delta[replicate] = float(np.mean(enriched_rmse)) / broad_rmse_mean - 1.0
        coverage[replicate] = float(np.mean(enriched_coverage))
    return {
        "unit": "exact_product_label",
        "replicates": replicates,
        "mae_relative_change_ci95": [
            float(np.quantile(mae_delta, 0.025)),
            float(np.quantile(mae_delta, 0.975)),
        ],
        "rmse_relative_change_ci95": [
            float(np.quantile(rmse_delta, 0.025)),
            float(np.quantile(rmse_delta, 0.975)),
        ],
        "enriched_conformal90_coverage_ci95": [
            float(np.quantile(coverage, 0.025)),
            float(np.quantile(coverage, 0.975)),
        ],
    }


def build_morphology_enriched_oracle_audit(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Build a structured-OOF morphology-shift audit without new oracle calls."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="morphology-enriched oracle config")
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("scope") != EXPECTED_SCOPE
    ):
        raise UgiMorphologyEnrichedOracleAuditError("oracle-shift config changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != EXPECTED_INPUTS:
        raise UgiMorphologyEnrichedOracleAuditError("oracle-shift input pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in raw_inputs.items()}
    applicability = _load_json(paths["applicability_result"], label="applicability result")
    selective = _load_json(paths["selective_risk_result"], label="selective-risk result")
    proposal_result = _load_json(paths["complete_proposal_result"], label="proposal result")
    if (
        selective.get("continuous_selective_risk", {})
        .get("fixed_normalized_radius_primary", {})
        .get("acceptance", {})
        .get("all_criteria_pass")
        is not True
        or proposal_result.get("decision", {}).get("complete_support_frozen") is not True
        or proposal_result.get("decision", {}).get("partial_state_smc_authorized") is not False
    ):
        raise UgiMorphologyEnrichedOracleAuditError("required prior decisions changed")
    ledger_meta = proposal_result.get("artifacts", {}).get("proposal_ledger.jsonl.gz", {})
    if ledger_meta.get("sha256") != sha256_file(paths["complete_proposal_ledger"]):
        raise UgiMorphologyEnrichedOracleAuditError("proposal ledger identity changed")
    proposal_rows = _read_jsonl_gzip(paths["complete_proposal_ledger"])
    proposal_by_program = {str(row["program_sha256"]): row for row in proposal_rows}
    if len(proposal_by_program) != 57190:
        raise UgiMorphologyEnrichedOracleAuditError("complete morphology support changed")
    program_by_product = _program_by_product(paths["training_cache"])
    curated_rows = _read_csv(paths["curated_agile"])
    curated = {str(row["label"]): row for row in curated_rows}
    if len(curated) != 1100:
        raise UgiMorphologyEnrichedOracleAuditError("curated AGILE corpus changed")
    schemes = tuple(str(value) for value in config["analysis"]["primary_schemes"])
    oof = [row for row in _read_csv(paths["heldout_oof_ledger"]) if row["scheme"] in schemes]
    if len(oof) != len(curated) * len(schemes):
        raise UgiMorphologyEnrichedOracleAuditError("structured OOF lattice changed")
    thresholds = applicability.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise UgiMorphologyEnrichedOracleAuditError("applicability thresholds are missing")
    active_radius = float(config["analysis"]["active_radius"])
    eligible_labels: set[str] = set()
    excluded_labels: set[str] = set()
    label_programs: dict[str, str] = {}
    for label, source in curated.items():
        product = str(source["model_smiles"])
        program_sha256 = program_by_product.get(product)
        if program_sha256 is None:
            raise UgiMorphologyEnrichedOracleAuditError(
                "one measured product is absent from the frozen morphology cache"
            )
        label_programs[label] = program_sha256
        if program_sha256 in proposal_by_program:
            eligible_labels.add(label)
        else:
            excluded_labels.add(label)
    support_overlap = len(eligible_labels) / len(curated)
    augmented: list[dict[str, Any]] = []
    for row in oof:
        label = str(row["label"])
        source = curated.get(label)
        if source is None:
            raise UgiMorphologyEnrichedOracleAuditError("OOF label missing from curated corpus")
        if label not in eligible_labels:
            continue
        program_sha256 = label_programs[label]
        proposal = proposal_by_program[program_sha256]
        radius = _normalized_radius(row, thresholds)
        augmented.append(
            {
                **row,
                "program_sha256": program_sha256,
                "support_score": float(proposal["support_score"]),
                "proposal_over_broad_weight": float(proposal["proposal_probability"])
                / float(proposal["broad_prior_probability"]),
                "max_normalized_radius": radius,
                "inside_fixed_radius": radius <= active_radius,
                "covered90": str(row["covered90"]).lower() == "true",
            }
        )
    augmented.sort(key=lambda row: (str(row["scheme"]), str(row["label"])))
    broad = _equal_scheme_summary(augmented, schemes, enriched=False)
    enriched = _equal_scheme_summary(augmented, schemes, enriched=True)
    shifts = {
        "mae_relative_change": enriched["equal_scheme_mae"] / broad["equal_scheme_mae"] - 1.0,
        "rmse_relative_change": enriched["equal_scheme_rmse"] / broad["equal_scheme_rmse"] - 1.0,
        "conformal90_coverage_change": enriched["equal_scheme_conformal90_coverage"]
        - broad["equal_scheme_conformal90_coverage"],
        "weighted_midrank_spearman_change": enriched["equal_scheme_weighted_midrank_spearman"]
        - broad["equal_scheme_weighted_midrank_spearman"],
    }
    bootstrap_policy = config["analysis"]["clustered_bootstrap"]
    bootstrap = _cluster_bootstrap(
        augmented,
        schemes,
        replicates=int(bootstrap_policy["replicates"]),
        seed=int(bootstrap_policy["seed"]),
    )
    gates = config["analysis"]["gates"]
    checks = {
        "measured_product_support_overlap": support_overlap
        >= float(gates["minimum_measured_product_support_overlap"]),
        "mae_shift": shifts["mae_relative_change"]
        <= float(gates["maximum_equal_scheme_mae_relative_increase"])
        and bootstrap["mae_relative_change_ci95"][1]
        <= float(gates["maximum_bootstrap_upper_mae_relative_increase"]),
        "rmse_shift": shifts["rmse_relative_change"]
        <= float(gates["maximum_equal_scheme_rmse_relative_increase"])
        and bootstrap["rmse_relative_change_ci95"][1]
        <= float(gates["maximum_bootstrap_upper_rmse_relative_increase"]),
        "conformal_coverage": enriched["equal_scheme_conformal90_coverage"]
        >= float(gates["minimum_equal_scheme_conformal90_coverage"])
        and bootstrap["enriched_conformal90_coverage_ci95"][0]
        >= float(gates["minimum_bootstrap_lower_conformal90_coverage"]),
        "effective_sample_size": enriched["minimum_effective_sample_fraction"]
        >= float(gates["minimum_per_scheme_effective_sample_fraction"]),
        "supported_mass": enriched["minimum_weighted_support_coverage"]
        >= float(gates["minimum_per_scheme_weighted_support_coverage"]),
        "per_scheme_ranking": all(
            float(record["metrics"]["weighted_midrank_spearman"])
            >= float(gates["minimum_per_scheme_weighted_midrank_spearman"])
            for record in enriched["per_scheme"].values()
        ),
        "per_scheme_fit": all(
            record["metrics"]["r2"] is not None
            and float(record["metrics"]["r2"]) >= float(gates["minimum_per_scheme_r2"])
            for record in enriched["per_scheme"].values()
        ),
        "top_decile_enrichment": enriched["equal_scheme_top_decile_observed_gain"]
        >= float(gates["minimum_equal_scheme_top_decile_observed_gain"]),
    }
    passed = all(checks.values())
    fields = (
        "scheme",
        "fold",
        "label",
        "program_sha256",
        "support_score",
        "proposal_over_broad_weight",
        "max_normalized_radius",
        "inside_fixed_radius",
        "y_true",
        "y_pred",
        "absolute_error",
        "conformal_q90",
        "covered90",
    )
    ledger = _csv_gzip_bytes(augmented, fields)
    inputs = {
        label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
        for label, path in sorted(paths.items())
    }
    inputs["config"] = {
        "path": str(config_path.relative_to(repo)),
        "sha256": sha256_file(config_path),
    }
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_morphology_enriched_structured_oof_oracle_audit",
        "scope": dict(EXPECTED_SCOPE),
        "inputs": inputs,
        "definition": {
            "endpoint": "expt_Hela",
            "oracle_predictions": "persisted_structured_out_of_fold_only",
            "chemical_support": "fixed_multiview_normalized_radius_at_most_one",
            "shift_weight": "complete_morphology_proposal_probability_divided_by_broad_prior_probability",
            "weight_interpretation": "covariate_shift_sensitivity_not_exact_transport_of_generated_chemistry",
            "exact_novelty_used_as_gate": False,
            "morphology_score_uses_potency_or_oracle_error": False,
        },
        "counts": {
            "exact_products": len(curated),
            "exact_products_inside_generator_morphology_support": len(eligible_labels),
            "exact_products_outside_generator_morphology_support": len(excluded_labels),
            "measured_product_support_overlap": support_overlap,
            "unique_excluded_morphology_programs": len(
                {label_programs[label] for label in excluded_labels}
            ),
            "structured_schemes": len(schemes),
            "oof_rows": len(augmented),
            "measured_products_mapped_to_complete_support": len(
                {str(row["label"]) for row in augmented}
            ),
        },
        "broad_supported_distribution": broad,
        "morphology_enriched_supported_distribution": enriched,
        "shift": shifts,
        "clustered_bootstrap": bootstrap,
        "gates": {"criteria": dict(gates), "checks": checks, "all_pass": passed},
        "artifacts": {
            "reweighted_oof.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rows": len(augmented),
                "sha256": sha256_bytes(ledger),
                "logical_sha256": _logical_sha256(augmented),
            }
        },
        "decision": {
            "morphology_enriched_oracle_stress_test_passed": passed,
            "matched_bounded_potency_diagnostic_design_may_proceed": passed,
            "potency_guidance_authorized": False,
            "partial_state_smc_authorized": False,
            "prospective_biological_reliability_claimed": False,
            "next_gate": (
                "freeze matched morphology-proposal potency-versus-posthoc diagnostic"
                if passed
                else "do not tilt potency; retain terminal abstention and posthoc ranking"
            ),
        },
        "nonclaims": [
            "Morphology states absent from the generator training-fold support are excluded rather than added from held-out data.",
            "Importance reweighting does not create biological labels for generated molecules.",
            "This audit cannot establish prospective potency or calibration under adaptive generation.",
            "Passing this audit does not authorize partial-state SMC or prospective candidate lock.",
        ],
    }
    result = {**content, "result_sha256": _logical_sha256(content)}
    return result, ledger


__all__ = [
    "UgiMorphologyEnrichedOracleAuditError",
    "build_morphology_enriched_oracle_audit",
    "effective_sample_size",
    "weighted_prediction_metrics",
]
