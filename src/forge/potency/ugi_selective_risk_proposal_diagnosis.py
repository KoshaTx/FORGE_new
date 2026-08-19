"""Diagnose Ugi applicability attrition and continuous selective-risk signal.

This module is deliberately nonselecting.  It reads frozen terminal ledgers and
persisted out-of-fold oracle errors, but it does not call an oracle, advance a
generator, score potency, alter the version-3 applicability policy, or authorize
guidance.  Its two purposes are:

1. separate chemical-distance attrition from the exact-role evidence policy;
2. test whether continuous multiview distances rank held-out oracle error better
   than the frozen categorical distribution bins.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.hashing import sha256_json as _sha256_payload

CONFIG_SCHEMA_VERSION = "phase1_ugi_selective_risk_proposal_diagnosis_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_selective_risk_proposal_diagnosis.v1"
RISK_LEDGER_SCHEMA_VERSION = "phase1_ugi_selective_risk_oof_ledger.v1"

VIEWS = ("product", "amine", "aldehyde", "isocyanide")
DISTANCE_KINDS = ("fingerprint", "descriptor")
MODELS = (
    "constant_q90",
    "categorical_bin_q90",
    "max_radius_ratio_q90",
    "monotone_multiview_q90",
)
SUPPORTED_PATTERNS = {
    ("amine",): "amine_only",
    ("aldehyde", "isocyanide"): "aldehyde_isocyanide_pair",
}

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
    "broad_census_ledger",
    "broad_census_result",
    "heldout_oof_ledger",
    "lambda_zero_ledger",
    "lambda_zero_result",
    "runner",
    "source",
    "tests",
}

RISK_LEDGER_FIELDS = (
    "scheme",
    "fold",
    "label",
    "risk_cv_fold",
    "distribution_bin",
    "absolute_error",
    *(f"{view}_{kind}_radius_ratio" for view in VIEWS for kind in DISTANCE_KINDS),
    "max_radius_ratio",
    *(f"{model}_risk" for model in MODELS),
)


class UgiSelectiveRiskDiagnosisError(RuntimeError):
    """Raised when a frozen input or the nonselecting contract is violated."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiSelectiveRiskDiagnosisError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiSelectiveRiskDiagnosisError(f"{label} must contain one JSON object")
    return value


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiSelectiveRiskDiagnosisError(f"invalid {label}: {path}") from error
    if not rows:
        raise UgiSelectiveRiskDiagnosisError(f"{label} is empty")
    return rows


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiSelectiveRiskDiagnosisError(f"{label} pin is malformed")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiSelectiveRiskDiagnosisError(f"{label} path escapes repository") from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiSelectiveRiskDiagnosisError(f"{label} hash changed")
    return path


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0", ""}:
        return False
    raise UgiSelectiveRiskDiagnosisError(f"invalid Boolean value: {value!r}")


def _parse_roles(value: Any) -> tuple[str, ...]:
    try:
        roles = tuple(str(item) for item in json.loads(str(value)))
    except (TypeError, json.JSONDecodeError) as error:
        raise UgiSelectiveRiskDiagnosisError("invalid exact-unseen-role record") from error
    if any(role not in VIEWS[1:] for role in roles) or len(set(roles)) != len(roles):
        raise UgiSelectiveRiskDiagnosisError("invalid exact-unseen-role membership")
    return roles


def _is_measured(row: Mapping[str, Any], *, source: str) -> bool:
    if source == "broad_census":
        return row.get("exact_identity_provenance") == "exact_measured_combination"
    if source == "lambda_zero_seed":
        return _parse_bool(row.get("exact_measured_combination", False))
    raise UgiSelectiveRiskDiagnosisError(f"unknown gate source: {source}")


def _valid_exact_l1(row: Mapping[str, Any], *, source: str) -> bool:
    if source == "broad_census":
        return True
    return (
        _parse_bool(row.get("terminal_present", False))
        and _parse_bool(row.get("terminal_valid", False))
        and _parse_bool(row.get("exact_l1", False))
    )


def _all_interpolative(row: Mapping[str, Any], views: Sequence[str] = VIEWS) -> bool:
    return all(row.get(f"{view}_distribution_bin") == "interpolative" for view in views)


def _pattern_supported(row: Mapping[str, Any]) -> bool:
    return _parse_roles(row.get("exact_unseen_roles_json", "[]")) in SUPPORTED_PATTERNS


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _count_and_fraction(numerator: int, denominator: int) -> dict[str, float | int | None]:
    return {
        "count": int(numerator),
        "denominator": int(denominator),
        "fraction": _safe_fraction(numerator, denominator),
    }


def gate_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate validity, multiview chemistry and role-evidence attrition."""

    if source not in {"broad_census", "lambda_zero_seed"}:
        raise UgiSelectiveRiskDiagnosisError("unsupported gate-diagnostic source")
    attempts = len(rows)
    valid = [row for row in rows if _valid_exact_l1(row, source=source)]
    measured = [row for row in valid if _is_measured(row, source=source)]
    novel = [row for row in valid if not _is_measured(row, source=source)]
    all_view_supported = [row for row in novel if _all_interpolative(row)]
    pattern_supported = [row for row in novel if _pattern_supported(row)]
    active = [row for row in all_view_supported if _pattern_supported(row)]

    pattern_counts: Counter[str] = Counter()
    for row in valid:
        roles = _parse_roles(row.get("exact_unseen_roles_json", "[]"))
        label = "+".join(roles) if roles else "none_exact_measured"
        if _all_interpolative(row):
            pattern_counts[label] += 1

    view_acceptance = {
        view: _count_and_fraction(
            sum(row.get(f"{view}_distribution_bin") == "interpolative" for row in novel),
            len(novel),
        )
        for view in VIEWS
    }
    marginal_product = math.prod(
        float(record["fraction"] or 0.0) for record in view_acceptance.values()
    )
    leave_one_view_out: dict[str, Any] = {}
    for omitted in VIEWS:
        retained_views = tuple(view for view in VIEWS if view != omitted)
        chemistry_rows = [row for row in novel if _all_interpolative(row, retained_views)]
        pattern_rows = [row for row in chemistry_rows if _pattern_supported(row)]
        leave_one_view_out[omitted] = {
            "chemistry_supported": _count_and_fraction(len(chemistry_rows), len(novel)),
            "increment_over_all_four": len(chemistry_rows) - len(all_view_supported),
            "active_if_omitted": _count_and_fraction(len(pattern_rows), len(novel)),
            "active_increment_over_all_four": len(pattern_rows) - len(active),
        }

    allow_boundary = [
        row
        for row in novel
        if all(
            row.get(f"{view}_distribution_bin") in {"interpolative", "boundary"} for view in VIEWS
        )
    ]
    modality_acceptance: dict[str, Any] | None = None
    if thresholds is not None:
        modality_acceptance = {}
        for view in VIEWS:
            fingerprint_threshold = float(thresholds[view]["fingerprint"]["interpolative_max"])
            descriptor_threshold = float(thresholds[view]["descriptor"]["interpolative_max"])
            fingerprint_count = 0
            descriptor_count = 0
            both_count = 0
            for row in novel:
                fingerprint_pass = (
                    float(row[f"{view}_fingerprint_distance"]) <= fingerprint_threshold
                )
                descriptor_pass = float(row[f"{view}_descriptor_distance"]) <= descriptor_threshold
                fingerprint_count += fingerprint_pass
                descriptor_count += descriptor_pass
                both_count += fingerprint_pass and descriptor_pass
            modality_acceptance[view] = {
                "fingerprint": _count_and_fraction(fingerprint_count, len(novel)),
                "descriptor": _count_and_fraction(descriptor_count, len(novel)),
                "both": _count_and_fraction(both_count, len(novel)),
                "fingerprint_only": fingerprint_count - both_count,
                "descriptor_only": descriptor_count - both_count,
            }
    morphology_fields = (
        ("branch_class",)
        if source == "broad_census"
        else ("bounded_neighborhood", "branch_class", "program_index")
    )
    morphology: dict[str, Any] = {}
    for field in morphology_fields:
        grouped: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"attempts": 0, "valid_exact_l1": 0, "active": 0}
        )
        for row in rows:
            key = str(row.get(field, ""))
            grouped[key]["attempts"] += 1
            if _valid_exact_l1(row, source=source):
                grouped[key]["valid_exact_l1"] += 1
            if (
                _valid_exact_l1(row, source=source)
                and not _is_measured(row, source=source)
                and _all_interpolative(row)
                and _pattern_supported(row)
            ):
                grouped[key]["active"] += 1
        morphology[field] = {
            "groups": len(grouped),
            "groups_with_active_terminal": sum(item["active"] > 0 for item in grouped.values()),
            "counts": dict(sorted(grouped.items())),
        }

    return {
        "source": source,
        "attempts_or_postvalid_records": attempts,
        "input_denominator_note": (
            "postvalid_exact_l1_only"
            if source == "broad_census"
            else "all_scheduled_terminal_attempts"
        ),
        "valid_exact_l1": _count_and_fraction(len(valid), attempts),
        "exact_measured_neutral": _count_and_fraction(len(measured), len(valid)),
        "novel_valid_exact_l1": _count_and_fraction(len(novel), len(valid)),
        "novel_all_four_views_interpolative": _count_and_fraction(
            len(all_view_supported), len(novel)
        ),
        "novel_supported_role_pattern_irrespective_of_chemistry": _count_and_fraction(
            len(pattern_supported), len(novel)
        ),
        "active_intersection": {
            "among_attempts_or_records": _count_and_fraction(len(active), attempts),
            "among_valid_exact_l1": _count_and_fraction(len(active), len(valid)),
            "among_novel_valid_exact_l1": _count_and_fraction(len(active), len(novel)),
            "among_chemically_supported_novel": _count_and_fraction(
                len(active), len(all_view_supported)
            ),
        },
        "chemically_supported_but_role_policy_abstains": _count_and_fraction(
            len(all_view_supported) - len(active), len(all_view_supported)
        ),
        "all_four_interpolative_patterns": dict(sorted(pattern_counts.items())),
        "view_acceptance_among_novel": view_acceptance,
        "independence_diagnostic": {
            "product_of_marginal_view_acceptance": marginal_product,
            "observed_joint_acceptance": _safe_fraction(len(all_view_supported), len(novel)),
            "observed_to_independence_ratio": (
                _safe_fraction(len(all_view_supported), len(novel)) / marginal_product
                if marginal_product
                else None
            ),
            "interpretation": "descriptive_only_views_are_not_independent",
        },
        "leave_one_view_out": leave_one_view_out,
        "distance_modality_acceptance_among_novel": modality_acceptance,
        "all_views_inside_boundary_radius": _count_and_fraction(len(allow_boundary), len(novel)),
        "morphology_strata": morphology,
    }


def _finite_quantile(values: Sequence[float], probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 0:
        raise UgiSelectiveRiskDiagnosisError("cannot estimate a quantile from no records")
    index = min(
        ordered.size - 1,
        max(0, int(math.ceil((ordered.size + 1) * probability) - 1)),
    )
    return float(ordered[index])


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


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    ranked_left = _average_ranks(left)
    ranked_right = _average_ranks(right)
    value = float(np.corrcoef(ranked_left, ranked_right)[0, 1])
    return value if math.isfinite(value) else None


def _balanced_label_folds(labels: Sequence[str], folds: int, *, salt: str) -> dict[str, int]:
    unique = sorted(
        set(labels),
        key=lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest(),
    )
    if len(unique) < folds:
        raise UgiSelectiveRiskDiagnosisError("too few unique labels for grouped risk CV")
    return {label: index % folds for index, label in enumerate(unique)}


def _normalized_features(
    rows: Sequence[Mapping[str, str]], thresholds: Mapping[str, Any]
) -> tuple[np.ndarray, list[str]]:
    names: list[str] = []
    denominators: list[float] = []
    for view in VIEWS:
        for kind in DISTANCE_KINDS:
            names.append(f"{view}_{kind}_radius_ratio")
            threshold = float(thresholds[view][kind]["interpolative_max"])
            if not math.isfinite(threshold) or threshold <= 0:
                raise UgiSelectiveRiskDiagnosisError("invalid interpolative threshold")
            denominators.append(threshold)
    features = []
    for row in rows:
        values = []
        for view in VIEWS:
            for kind in DISTANCE_KINDS:
                value = float(row[f"{view}_{kind}_distance"])
                if not math.isfinite(value) or value < 0:
                    raise UgiSelectiveRiskDiagnosisError("invalid held-out distance")
                values.append(value)
        features.append([value / denominator for value, denominator in zip(values, denominators)])
    return np.asarray(features, dtype=float), names


def _conformal_adjustment(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    coverage: float,
) -> float:
    return _finite_quantile(observed - predicted, coverage)


def _coverage_metrics(
    observed: np.ndarray,
    predicted_risk: np.ndarray,
    coverage_fractions: Sequence[float],
) -> dict[str, Any]:
    order = np.argsort(predicted_risk, kind="mergesort")
    curves: dict[str, Any] = {}
    for fraction in coverage_fractions:
        selected = order[: max(1, int(math.ceil(len(order) * fraction)))]
        curves[f"{fraction:.6g}"] = {
            "records": int(len(selected)),
            "mean_absolute_error": float(np.mean(observed[selected])),
            "q90_absolute_error": float(np.quantile(observed[selected], 0.9)),
            "mean_predicted_risk": float(np.mean(predicted_risk[selected])),
        }
    return {
        "records": int(len(observed)),
        "empirical_q90_coverage": float(np.mean(observed <= predicted_risk)),
        "mean_predicted_risk": float(np.mean(predicted_risk)),
        "risk_error_spearman": _spearman(predicted_risk, observed),
        "selective_curve": curves,
    }


def _prediction_metrics(rows: Sequence[Mapping[str, str]]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "records": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "spearman": None,
        }
    observed = np.asarray([float(row["y_true"]) for row in rows], dtype=float)
    predicted = np.asarray([float(row["y_pred"]) for row in rows], dtype=float)
    residual = observed - predicted
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    return {
        "records": len(rows),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else None,
        "spearman": _spearman(predicted, observed),
    }


def _confidence_interval(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise UgiSelectiveRiskDiagnosisError("invalid clustered-bootstrap statistic")
    return {
        "lower_95": float(np.quantile(array, 0.025)),
        "median": float(np.quantile(array, 0.5)),
        "upper_95": float(np.quantile(array, 0.975)),
    }


def fixed_radius_selective_risk(
    rows: Sequence[Mapping[str, str]],
    thresholds: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the fixed worst normalized radius without fitting a selector."""

    features, feature_names = _normalized_features(rows, thresholds)
    radius = np.max(features, axis=1)
    primary_schemes = tuple(str(value) for value in policy["primary_schemes"])
    stress_schemes = tuple(str(value) for value in policy["external_stress_schemes"])
    observed_schemes = {str(row["scheme"]) for row in rows}
    if set(primary_schemes) | set(stress_schemes) != observed_schemes:
        raise UgiSelectiveRiskDiagnosisError("fixed-radius scheme partition changed")

    augmented = [dict(row, max_radius_ratio=float(radius[index])) for index, row in enumerate(rows)]
    primary = [row for row in augmented if row["scheme"] in primary_schemes]
    stress = [row for row in augmented if row["scheme"] in stress_schemes]
    if len({row["label"] for row in primary}) * len(primary_schemes) != len(primary):
        raise UgiSelectiveRiskDiagnosisError("primary schemes do not share one row per label")

    thresholds_to_report = tuple(float(value) for value in policy["fixed_radius_thresholds"])
    threshold_curve: dict[str, Any] = {}
    for radius_threshold in thresholds_to_report:
        retained = [row for row in primary if float(row["max_radius_ratio"]) <= radius_threshold]
        per_scheme = {
            scheme: {
                "coverage": _safe_fraction(
                    sum(row["scheme"] == scheme for row in retained),
                    sum(row["scheme"] == scheme for row in primary),
                ),
                "metrics": _prediction_metrics(
                    [row for row in retained if row["scheme"] == scheme]
                ),
            }
            for scheme in primary_schemes
        }
        scheme_mae = [record["metrics"]["mae"] for record in per_scheme.values()]
        scheme_rmse = [record["metrics"]["rmse"] for record in per_scheme.values()]
        threshold_curve[f"{radius_threshold:.6g}"] = {
            "records": len(retained),
            "coverage": _safe_fraction(len(retained), len(primary)),
            "pooled_metrics": _prediction_metrics(retained),
            "equal_scheme_mae": (
                float(np.mean(scheme_mae))
                if all(value is not None for value in scheme_mae)
                else None
            ),
            "equal_scheme_rmse": (
                float(np.mean(scheme_rmse))
                if all(value is not None for value in scheme_rmse)
                else None
            ),
            "per_scheme": per_scheme,
        }

    active_key = f"{float(policy['fixed_radius_active_threshold']):.6g}"
    if active_key not in threshold_curve:
        raise UgiSelectiveRiskDiagnosisError("active radius missing from threshold curve")
    active = threshold_curve[active_key]
    full_metrics = {
        scheme: _prediction_metrics([row for row in primary if row["scheme"] == scheme])
        for scheme in primary_schemes
    }
    full_equal_mae = float(np.mean([record["mae"] for record in full_metrics.values()]))
    full_equal_rmse = float(np.mean([record["rmse"] for record in full_metrics.values()]))
    mae_reduction = (full_equal_mae - active["equal_scheme_mae"]) / full_equal_mae
    rmse_reduction = (full_equal_rmse - active["equal_scheme_rmse"]) / full_equal_rmse
    per_scheme_relative_mae_reduction = {
        scheme: (
            float(full_metrics[scheme]["mae"])
            - float(active["per_scheme"][scheme]["metrics"]["mae"])
        )
        / float(full_metrics[scheme]["mae"])
        for scheme in primary_schemes
    }

    primary_radius = np.asarray([float(row["max_radius_ratio"]) for row in primary])
    primary_error = np.asarray([float(row["absolute_error"]) for row in primary])
    risk_error_rho = _spearman(primary_radius, primary_error)

    labels = sorted({str(row["label"]) for row in primary})
    by_scheme_label = {(str(row["scheme"]), str(row["label"])): row for row in primary}
    bootstrap = policy["clustered_bootstrap"]
    random = np.random.default_rng(int(bootstrap["seed"]))
    replicates = int(bootstrap["replicates"])
    bootstrap_mae_reduction: list[float] = []
    bootstrap_rmse_reduction: list[float] = []
    bootstrap_coverage: list[float] = []
    bootstrap_rho: list[float] = []
    active_threshold = float(policy["fixed_radius_active_threshold"])
    for _ in range(replicates):
        sampled_labels = random.choice(labels, size=len(labels), replace=True)
        full_mae: list[float] = []
        retained_mae: list[float] = []
        full_rmse: list[float] = []
        retained_rmse: list[float] = []
        scheme_coverage: list[float] = []
        sampled_radius: list[float] = []
        sampled_error: list[float] = []
        for scheme in primary_schemes:
            scheme_rows = [by_scheme_label[(scheme, str(label))] for label in sampled_labels]
            errors = np.asarray([float(row["absolute_error"]) for row in scheme_rows])
            radii = np.asarray([float(row["max_radius_ratio"]) for row in scheme_rows])
            keep = radii <= active_threshold
            if not np.any(keep):
                raise UgiSelectiveRiskDiagnosisError("bootstrap replicate has empty scheme support")
            full_mae.append(float(np.mean(errors)))
            retained_mae.append(float(np.mean(errors[keep])))
            full_rmse.append(float(np.sqrt(np.mean(errors**2))))
            retained_rmse.append(float(np.sqrt(np.mean(errors[keep] ** 2))))
            scheme_coverage.append(float(np.mean(keep)))
            sampled_radius.extend(radii.tolist())
            sampled_error.extend(errors.tolist())
        mean_full_mae = float(np.mean(full_mae))
        mean_full_rmse = float(np.mean(full_rmse))
        bootstrap_mae_reduction.append(
            (mean_full_mae - float(np.mean(retained_mae))) / mean_full_mae
        )
        bootstrap_rmse_reduction.append(
            (mean_full_rmse - float(np.mean(retained_rmse))) / mean_full_rmse
        )
        bootstrap_coverage.append(float(np.mean(scheme_coverage)))
        sampled_rho = _spearman(
            np.asarray(sampled_radius, dtype=float), np.asarray(sampled_error, dtype=float)
        )
        if sampled_rho is None:
            raise UgiSelectiveRiskDiagnosisError("bootstrap risk correlation is undefined")
        bootstrap_rho.append(sampled_rho)

    active_rows_by_scheme = {
        scheme: [
            row
            for row in primary
            if row["scheme"] == scheme and float(row["max_radius_ratio"]) <= active_threshold
        ]
        for scheme in primary_schemes
    }
    positive_fold_fractions: dict[str, float | None] = {}
    for scheme, scheme_rows in active_rows_by_scheme.items():
        fold_metrics = [
            _prediction_metrics([row for row in scheme_rows if row["fold"] == fold])
            for fold in sorted({row["fold"] for row in primary if row["scheme"] == scheme})
        ]
        eligible = [metric for metric in fold_metrics if metric["r2"] is not None]
        positive_fold_fractions[scheme] = (
            sum(float(metric["r2"]) > 0.0 for metric in eligible) / len(eligible)
            if eligible
            else None
        )

    confidence = {
        "equal_scheme_mae_relative_reduction": _confidence_interval(bootstrap_mae_reduction),
        "equal_scheme_rmse_relative_reduction": _confidence_interval(bootstrap_rmse_reduction),
        "equal_scheme_retained_coverage": _confidence_interval(bootstrap_coverage),
        "risk_error_spearman": _confidence_interval(bootstrap_rho),
    }
    criteria = policy["fixed_radius_acceptance"]
    single_role_schemes = tuple(str(value) for value in criteria["single_role_schemes"])
    criterion_results = {
        "mae_reduction": mae_reduction >= float(criteria["minimum_mae_relative_reduction"])
        and confidence["equal_scheme_mae_relative_reduction"]["lower_95"] > 0.0,
        "rmse_reduction": rmse_reduction >= float(criteria["minimum_rmse_relative_reduction"])
        and confidence["equal_scheme_rmse_relative_reduction"]["lower_95"] > 0.0,
        "positive_risk_error_correlation": risk_error_rho is not None
        and confidence["risk_error_spearman"]["lower_95"] > 0.0,
        "overall_coverage": float(active["coverage"])
        >= float(criteria["minimum_overall_coverage"]),
        "single_role_coverage": all(
            float(active["per_scheme"][scheme]["coverage"])
            >= float(criteria["minimum_single_role_coverage"])
            for scheme in single_role_schemes
        ),
        "per_scheme_prediction_quality": all(
            record["metrics"]["r2"] is not None
            and float(record["metrics"]["r2"]) > float(criteria["minimum_per_scheme_r2"])
            and record["metrics"]["spearman"] is not None
            and float(record["metrics"]["spearman"])
            > float(criteria["minimum_per_scheme_spearman"])
            for record in active["per_scheme"].values()
        ),
        "positive_fold_fraction": all(
            value is not None and value >= float(criteria["minimum_positive_fold_fraction"])
            for value in positive_fold_fractions.values()
        ),
    }

    stress_result: dict[str, Any] = {}
    for scheme in stress_schemes:
        scheme_rows = [row for row in stress if row["scheme"] == scheme]
        retained = [
            row for row in scheme_rows if float(row["max_radius_ratio"]) <= active_threshold
        ]
        stress_result[scheme] = {
            "all": _prediction_metrics(scheme_rows),
            "inside_fixed_radius": _prediction_metrics(retained),
            "coverage": _safe_fraction(len(retained), len(scheme_rows)),
        }

    return {
        "definition": {
            "score": "maximum_distance_divided_by_frozen_interpolative_radius",
            "features": feature_names,
            "target_free": True,
            "active_diagnostic_threshold": active_threshold,
            "threshold_selected_from_errors": False,
            "exact_novelty_used": False,
        },
        "primary_schemes": list(primary_schemes),
        "external_stress_schemes": list(stress_schemes),
        "threshold_curve": threshold_curve,
        "active_threshold_summary": {
            "full_equal_scheme_mae": full_equal_mae,
            "inside_equal_scheme_mae": active["equal_scheme_mae"],
            "mae_relative_reduction": mae_reduction,
            "full_per_scheme_metrics": full_metrics,
            "per_scheme_mae_relative_reduction": per_scheme_relative_mae_reduction,
            "full_equal_scheme_rmse": full_equal_rmse,
            "inside_equal_scheme_rmse": active["equal_scheme_rmse"],
            "rmse_relative_reduction": rmse_reduction,
            "risk_error_spearman": risk_error_rho,
            "positive_r2_fold_fraction_by_scheme": positive_fold_fractions,
            "clustered_bootstrap": {
                "unit": "exact_product_label",
                "replicates": replicates,
                "confidence_intervals": confidence,
            },
        },
        "acceptance": {
            "criteria_frozen_in_config": dict(criteria),
            "criterion_results": criterion_results,
            "all_criteria_pass": all(criterion_results.values()),
            "interpretation": "exploratory_diagnostic_screen_not_independent_confirmation",
        },
        "external_stress": stress_result,
    }


def cross_validated_selective_risk(
    rows: Sequence[Mapping[str, str]],
    thresholds: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit nested label-grouped q90 risk models to persisted OOF errors."""

    labels = [str(row["label"]) for row in rows]
    schemes = [str(row["scheme"]) for row in rows]
    bins = [str(row["distribution_bin"]) for row in rows]
    errors = np.asarray([float(row["absolute_error"]) for row in rows], dtype=float)
    if not np.all(np.isfinite(errors)) or np.any(errors < 0):
        raise UgiSelectiveRiskDiagnosisError("invalid persisted absolute error")
    features, feature_names = _normalized_features(rows, thresholds)

    folds = int(policy["outer_group_folds"])
    coverage = float(policy["risk_coverage"])
    calibration_stride = int(policy["calibration_label_stride"])
    seed = int(policy["random_seed"])
    coverage_fractions = tuple(float(value) for value in policy["coverage_fractions"])
    label_fold = _balanced_label_folds(labels, folds, salt=str(policy["fold_salt"]))

    predictions = {model: np.empty(len(rows), dtype=float) for model in MODELS}
    assigned_folds = np.asarray([label_fold[label] for label in labels], dtype=int)
    fold_records: list[dict[str, Any]] = []
    bin_array = np.asarray(bins, dtype=object)

    for fold in range(folds):
        test = assigned_folds == fold
        remaining = ~test
        remaining_labels = sorted(
            {labels[index] for index in np.flatnonzero(remaining)},
            key=lambda value: hashlib.sha256(
                f"{policy['calibration_salt']}:{fold}:{value}".encode()
            ).hexdigest(),
        )
        calibration_labels = set(remaining_labels[::calibration_stride])
        calibration = np.asarray(
            [remaining[index] and labels[index] in calibration_labels for index in range(len(rows))]
        )
        fit = remaining & ~calibration
        if not np.any(fit) or not np.any(calibration) or not np.any(test):
            raise UgiSelectiveRiskDiagnosisError("empty nested risk-CV partition")

        raw_calibration: dict[str, np.ndarray] = {}
        raw_test: dict[str, np.ndarray] = {}
        constant = _finite_quantile(errors[fit], coverage)
        raw_calibration["constant_q90"] = np.full(int(np.sum(calibration)), constant)
        raw_test["constant_q90"] = np.full(int(np.sum(test)), constant)

        calibration_indices = np.flatnonzero(calibration)
        test_indices = np.flatnonzero(test)
        categorical_calibration = np.empty(len(calibration_indices), dtype=float)
        categorical_test = np.empty(len(test_indices), dtype=float)
        for distribution_bin in ("interpolative", "boundary", "extrapolative"):
            bin_fit = fit & (bin_array == distribution_bin)
            if not np.any(bin_fit):
                raise UgiSelectiveRiskDiagnosisError("categorical risk bin absent in fit data")
            value = _finite_quantile(errors[bin_fit], coverage)
            categorical_calibration[
                [bin_array[index] == distribution_bin for index in calibration_indices]
            ] = value
            categorical_test[[bin_array[index] == distribution_bin for index in test_indices]] = (
                value
            )
        raw_calibration["categorical_bin_q90"] = categorical_calibration
        raw_test["categorical_bin_q90"] = categorical_test

        model_parameters = policy["monotone_hgb"]
        model_inputs = {
            "max_radius_ratio_q90": np.log1p(np.max(features, axis=1))[:, None],
            "monotone_multiview_q90": np.log1p(features),
        }
        for model_name, model_features in model_inputs.items():
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=coverage,
                learning_rate=float(model_parameters["learning_rate"]),
                max_iter=int(model_parameters["max_iter"]),
                max_leaf_nodes=int(model_parameters["max_leaf_nodes"]),
                max_depth=int(model_parameters["max_depth"]),
                min_samples_leaf=int(model_parameters["min_samples_leaf"]),
                l2_regularization=float(model_parameters["l2_regularization"]),
                monotonic_cst=[1] * model_features.shape[1],
                early_stopping=False,
                random_state=seed,
            )
            model.fit(model_features[fit], errors[fit])
            raw_calibration[model_name] = model.predict(model_features[calibration])
            raw_test[model_name] = model.predict(model_features[test])

        fold_detail: dict[str, Any] = {
            "fold": fold,
            "fit_labels": len({labels[index] for index in np.flatnonzero(fit)}),
            "calibration_labels": len(calibration_labels),
            "test_labels": len({labels[index] for index in test_indices}),
            "fit_rows": int(np.sum(fit)),
            "calibration_rows": int(np.sum(calibration)),
            "test_rows": int(np.sum(test)),
            "models": {},
        }
        for model_name in MODELS:
            adjustment = _conformal_adjustment(
                errors[calibration], raw_calibration[model_name], coverage=coverage
            )
            corrected = np.maximum(0.0, raw_test[model_name] + adjustment)
            predictions[model_name][test] = corrected
            fold_detail["models"][model_name] = {
                "calibration_adjustment": adjustment,
                "test_coverage": float(np.mean(errors[test] <= corrected)),
            }
        fold_records.append(fold_detail)

    overall = {
        model: _coverage_metrics(errors, predictions[model], coverage_fractions) for model in MODELS
    }
    by_scheme: dict[str, Any] = {}
    for scheme in sorted(set(schemes)):
        selected = np.asarray([value == scheme for value in schemes])
        by_scheme[scheme] = {
            model: _coverage_metrics(
                errors[selected], predictions[model][selected], coverage_fractions
            )
            for model in MODELS
        }

    categorical = overall["categorical_bin_q90"]
    multiview = overall["monotone_multiview_q90"]
    curve_improvements = {}
    for fraction in coverage_fractions:
        key = f"{fraction:.6g}"
        curve_improvements[key] = (
            categorical["selective_curve"][key]["mean_absolute_error"]
            - multiview["selective_curve"][key]["mean_absolute_error"]
        )
    per_scheme_improvements: dict[str, Any] = {}
    for scheme, metrics in by_scheme.items():
        per_scheme_improvements[scheme] = {
            f"{fraction:.6g}": (
                metrics["categorical_bin_q90"]["selective_curve"][f"{fraction:.6g}"][
                    "mean_absolute_error"
                ]
                - metrics["monotone_multiview_q90"]["selective_curve"][f"{fraction:.6g}"][
                    "mean_absolute_error"
                ]
            )
            for fraction in coverage_fractions
        }

    screen = policy["screen"]
    checked_fractions = tuple(f"{float(value):.6g}" for value in screen["coverage_fractions"])
    overall_improves = all(
        curve_improvements[fraction] >= float(screen["minimum_mae_improvement"])
        for fraction in checked_fractions
    )
    coverage_passes = multiview["empirical_q90_coverage"] >= coverage - float(
        screen["maximum_coverage_shortfall"]
    )
    rho_gain = (multiview["risk_error_spearman"] or 0.0) - (
        categorical["risk_error_spearman"] or 0.0
    )
    rho_passes = rho_gain >= float(screen["minimum_spearman_gain"])
    schemes_improved = sum(
        all(per_scheme_improvements[scheme][fraction] >= 0.0 for fraction in checked_fractions)
        for scheme in per_scheme_improvements
    )
    scheme_fraction = schemes_improved / len(per_scheme_improvements)
    improvement_consistency_passes = scheme_fraction >= float(
        screen["minimum_scheme_improvement_fraction"]
    )
    global_selection_scheme_coverage: dict[str, Any] = {}
    global_order = np.argsort(predictions["monotone_multiview_q90"], kind="mergesort")
    for fraction in (float(value) for value in screen["coverage_fractions"]):
        selected = global_order[: max(1, int(math.ceil(len(global_order) * fraction)))]
        global_selection_scheme_coverage[f"{fraction:.6g}"] = {
            scheme: sum(schemes[index] == scheme for index in selected)
            / sum(value == scheme for value in schemes)
            for scheme in sorted(set(schemes))
        }
    support_balance_passes = all(
        coverage >= float(screen["minimum_global_selection_per_scheme_coverage"])
        for record in global_selection_scheme_coverage.values()
        for coverage in record.values()
    )
    scheme_passes = improvement_consistency_passes and support_balance_passes

    ledger_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        output: dict[str, Any] = {
            "scheme": row["scheme"],
            "fold": row["fold"],
            "label": row["label"],
            "risk_cv_fold": int(assigned_folds[index]),
            "distribution_bin": row["distribution_bin"],
            "absolute_error": float(row["absolute_error"]),
        }
        for feature_index, feature_name in enumerate(feature_names):
            output[feature_name] = float(features[index, feature_index])
        output["max_radius_ratio"] = float(np.max(features[index]))
        for model_name in MODELS:
            output[f"{model_name}_risk"] = float(predictions[model_name][index])
        ledger_rows.append(output)

    return (
        {
            "contract": {
                "target": "persisted_out_of_fold_absolute_error",
                "outer_grouping": "exact_product_label",
                "outer_folds": folds,
                "same_label_never_crosses_risk_cv_folds": True,
                "nested_calibration": True,
                "risk_coverage": coverage,
                "features": feature_names,
                "models": list(MODELS),
                "no_generated_candidate_selected": True,
                "no_policy_threshold_derived": True,
            },
            "folds": fold_records,
            "overall": overall,
            "by_scheme": by_scheme,
            "comparison_to_categorical": {
                "mean_absolute_error_improvement": curve_improvements,
                "risk_error_spearman_gain": rho_gain,
                "per_scheme_mean_absolute_error_improvement": per_scheme_improvements,
                "schemes_improved_at_all_screen_fractions": schemes_improved,
                "scheme_improvement_fraction": scheme_fraction,
                "global_selection_scheme_coverage": global_selection_scheme_coverage,
            },
            "screen": {
                "criteria_frozen_in_config": dict(screen),
                "overall_selective_mae_pass": overall_improves,
                "overall_q90_coverage_pass": coverage_passes,
                "risk_error_spearman_gain_pass": rho_passes,
                "per_scheme_improvement_consistency_pass": improvement_consistency_passes,
                "global_selection_support_balance_pass": support_balance_passes,
                "cross_scheme_consistency_pass": scheme_passes,
                "all_screen_criteria_pass": (
                    overall_improves and coverage_passes and rho_passes and scheme_passes
                ),
                "interpretation": "diagnostic_signal_only_not_policy_authorization",
            },
        },
        ledger_rows,
    )


def _risk_ledger_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    text_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(text_buffer, fieldnames=RISK_LEDGER_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in RISK_LEDGER_FIELDS})
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as compressed:
        compressed.write(text_buffer.getvalue().encode())
    return output.getvalue()


def build_selective_risk_proposal_diagnosis(
    repo: Path,
    config_path: Path,
) -> tuple[dict[str, Any], bytes]:
    """Build a deterministic, write-once nonselecting diagnosis artifact."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="diagnosis config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiSelectiveRiskDiagnosisError("unsupported diagnosis config schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiSelectiveRiskDiagnosisError("diagnosis scope changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != EXPECTED_INPUTS:
        raise UgiSelectiveRiskDiagnosisError("diagnosis input pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in inputs.items()}

    applicability = _load_json(paths["applicability_result"], label="applicability result")
    broad_result = _load_json(paths["broad_census_result"], label="broad census result")
    pilot_result = _load_json(paths["lambda_zero_result"], label="lambda-zero result")
    thresholds = applicability.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(VIEWS):
        raise UgiSelectiveRiskDiagnosisError("applicability thresholds changed")
    if broad_result.get("scope", {}).get("potency_predictions_consumed") is not False:
        raise UgiSelectiveRiskDiagnosisError("broad census is not potency-free")
    if pilot_result.get("scope", {}).get("oracle_calls") != 0:
        raise UgiSelectiveRiskDiagnosisError("lambda-zero result called an oracle")

    broad_rows = _read_csv(paths["broad_census_ledger"], label="broad census ledger")
    pilot_rows = _read_csv(paths["lambda_zero_ledger"], label="lambda-zero ledger")
    heldout_rows = _read_csv(paths["heldout_oof_ledger"], label="held-out OOF ledger")
    broad_gate = gate_diagnostics(broad_rows, source="broad_census", thresholds=thresholds)
    pilot_gate = gate_diagnostics(pilot_rows, source="lambda_zero_seed", thresholds=thresholds)
    fixed_radius = fixed_radius_selective_risk(heldout_rows, thresholds, config["risk_policy"])
    primary_schemes = set(config["risk_policy"]["primary_schemes"])
    primary_heldout_rows = [row for row in heldout_rows if row["scheme"] in primary_schemes]
    risk, risk_rows = cross_validated_selective_risk(
        primary_heldout_rows, thresholds, config["risk_policy"]
    )
    risk_ledger = _risk_ledger_bytes(risk_rows)

    broad_chemistry = broad_gate["novel_all_four_views_interpolative"]["count"]
    broad_active = broad_gate["active_intersection"]["among_chemically_supported_novel"]["count"]
    pilot_programs = pilot_gate["morphology_strata"]["program_index"]
    screen_passed = bool(fixed_radius["acceptance"]["all_criteria_pass"])
    scheme_consistent = bool(risk["screen"]["cross_scheme_consistency_pass"])

    def relative(path: Path) -> str:
        return str(path.relative_to(repo))

    recorded_inputs = {
        label: {"path": relative(path), "sha256": sha256_file(path)}
        for label, path in sorted(paths.items())
    }
    recorded_inputs["config"] = {
        "path": relative(config_path),
        "sha256": sha256_file(config_path),
    }
    content: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed_nonselecting_diagnosis",
        "scope": dict(EXPECTED_SCOPE),
        "config": config,
        "inputs": recorded_inputs,
        "gate_attribution": {
            "broad_frozen_pool": broad_gate,
            "lambda_zero_seed": pilot_gate,
            "interpretation": {
                "exact_novelty_is_not_chemical_ood": True,
                "role_policy_is_separate_from_multiview_chemical_support": True,
                "broad_pool_is_postvalid_and_cannot_estimate_generator_invalidity": True,
                "pilot_seed_can_estimate_invalidity_but_not_guidance_efficacy": True,
                "product_view_is_the_largest_leave_one_view_bottleneck": True,
                "dropping_product_view_is_not_authorized": True,
            },
        },
        "continuous_selective_risk": {
            "fixed_normalized_radius_primary": fixed_radius,
            "learned_monotone_challenger": risk,
            "primary_method": "fixed_normalized_radius",
            "challenger_can_replace_primary": False,
        },
        "diagnosis": {
            "broad_chemically_supported_novel": broad_chemistry,
            "broad_active_under_current_role_policy": broad_active,
            "broad_chemically_supported_novel_rejected_by_role_policy": (
                broad_chemistry - broad_active
            ),
            "fixed_radius_exploratory_screen_passed_all_criteria": screen_passed,
            "learned_risk_challenger_cross_scheme_consistent": scheme_consistent,
            "pilot_programs_observed": pilot_programs["groups"],
            "pilot_programs_with_active_terminal": pilot_programs["groups_with_active_terminal"],
            "morphology_failure_established": False,
            "reason_morphology_not_established": (
                "one seed contains too few morphology programs and active terminals to "
                "separate morphology proposal quality from terminal chemistry"
            ),
            "current_v3_policy_replaced": False,
            "nonzero_guidance_authorized": False,
            "dynamic_morphology_proposal_authorized": False,
            "partial_state_smc_authorized": False,
            "next_gate": (
                "freeze and independently validate a role-scheme-aware continuous selective-risk "
                "policy, then run a dynamic frozen-prior terminal census with saved partial-state "
                "rollouts before deciding between rejection, dynamic morphology proposals, and "
                "delayed SMC"
            ),
        },
        "nonclaims": [
            "This audit does not replace or relax the version-3 applicability policy.",
            "The exploratory screen is not an independent preregistered confirmation.",
            "This audit does not authorize potency scoring or nonzero guidance.",
            "This audit does not establish that morphology sampling caused low eligibility.",
            "This audit does not establish that the continuous risk model is calibrated for "
            "all-three-new products or prospective generated molecules.",
            "This audit does not select, rank or lock any molecule.",
        ],
        "artifacts": {
            "selective_risk_oof.csv.gz": {
                "schema_version": RISK_LEDGER_SCHEMA_VERSION,
                "records": len(risk_rows),
                "contains_biological_target_values": False,
                "contains_persisted_oof_absolute_errors": True,
                "contains_generated_candidate_scores": False,
                "sha256": sha256_bytes(risk_ledger),
            }
        },
    }
    return {**content, "result_sha256": _sha256_payload(content)}, risk_ledger
