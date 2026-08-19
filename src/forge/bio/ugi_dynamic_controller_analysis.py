"""Leakage-free controller choice from the frozen Ugi terminal census.

The analysis is intentionally nonselecting.  It labels complete terminal
products with the already frozen, target-free multiview structural radius, then
tests whether that terminal label can be predicted from (M0) the morphology
program or (M1) a saved categorical partial state.  Potency, routes, synthesis,
terminal structures and trajectory identifiers are never model features.
"""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from forge.bio import ugi_distributional_applicability as applicability_v1
from forge.bio import ugi_distributional_applicability_v2 as applicability_v2
from forge.data.r1_prime_audit import sha256_bytes, sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi_dynamic_controller_analysis_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_dynamic_controller_analysis.v1"
TERMINAL_LEDGER_SCHEMA_VERSION = "phase1_ugi_dynamic_controller_terminal_support.v1"
OOF_LEDGER_SCHEMA_VERSION = "phase1_ugi_dynamic_controller_oof.v1"

ROLES = ("amine", "aldehyde", "isocyanide")
ROLE_COMPONENT_KEYS = {
    "amine": "amine_head",
    "aldehyde": "oxoester_aldehyde_body_tail",
    "isocyanide": "isocyanide_tail",
}
VIEWS = ("product", *ROLES)
EXPECTED_SCOPE = {
    "read_only": True,
    "frozen_generator_only": True,
    "target_free_terminal_support_only": True,
    "potency_predictions_consumed": False,
    "oracle_calls": 0,
    "route_calls": 0,
    "synthesis_calls": 0,
    "proposal_calls": 0,
    "generator_trajectories_advanced": False,
    "candidate_selection": False,
    "sealed_holdout_access": False,
    "smc_execution": False,
}
EXPECTED_INPUTS = {
    "applicability_result",
    "census_result",
    "census_validation_result",
    "curated_agile",
    "partial_state_manifest",
    "runner",
    "source",
    "terminal_ledger",
    "tests",
}


class UgiDynamicControllerAnalysisError(RuntimeError):
    """Raised when the frozen controller-analysis contract changes."""


@dataclass(frozen=True)
class Observation:
    observation_id: str
    group: str
    features: tuple[float, ...]
    successes: int
    trials: int
    terminal_indices: tuple[int, ...]
    checkpoint: int | None


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiDynamicControllerAnalysisError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiDynamicControllerAnalysisError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiDynamicControllerAnalysisError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiDynamicControllerAnalysisError(f"pin escapes repository: {label}") from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiDynamicControllerAnalysisError(f"pin changed: {label}")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiDynamicControllerAnalysisError(f"invalid CSV ledger: {path}") from error
    if not rows:
        raise UgiDynamicControllerAnalysisError(f"empty CSV ledger: {path}")
    return rows


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt") as handle:
            for line_number, line in enumerate(handle, 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("row is not an object")
                output.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise UgiDynamicControllerAnalysisError(
            f"invalid JSONL ledger at line {len(output) + 1}: {path}"
        ) from error
    if not output:
        raise UgiDynamicControllerAnalysisError("terminal ledger is empty")
    return output


def _csv_gz_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as handle:
        handle.write(text.getvalue().encode())
    return output.getvalue()


def decode_program_b64(value: str) -> dict[str, tuple[int, int, int]]:
    """Decode and strictly validate one canonical morphology-program payload."""

    try:
        outer = json.loads(base64.b64decode(value, validate=True))
        program = outer["program"]
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise UgiDynamicControllerAnalysisError("invalid encoded morphology program") from error
    expected = {"node_counts", "junction_budgets", "cycle_ranks", "attachment_counts"}
    if not isinstance(program, Mapping) or set(program) != expected:
        raise UgiDynamicControllerAnalysisError("morphology-program fields changed")
    output: dict[str, tuple[int, int, int]] = {}
    for key in sorted(expected):
        values = tuple(int(item) for item in program[key])
        if len(values) != 3 or any(item < 0 for item in values):
            raise UgiDynamicControllerAnalysisError("invalid morphology-program values")
        output[key] = values
    if any(value < 1 for value in output["node_counts"]):
        raise UgiDynamicControllerAnalysisError("empty component exterior")
    return output


def morphology_features(program: Mapping[str, Sequence[int]]) -> tuple[tuple[str, ...], np.ndarray]:
    """Return the frozen 12-coordinate M0 feature vector."""

    names: list[str] = []
    values: list[float] = []
    for field in ("node_counts", "junction_budgets", "cycle_ranks", "attachment_counts"):
        raw = tuple(int(value) for value in program[field])
        if len(raw) != 3:
            raise UgiDynamicControllerAnalysisError("M0 program is not role-complete")
        for role, value in zip(ROLES, raw, strict=True):
            names.append(f"{role}_{field}")
            values.append(float(value))
    return tuple(names), np.asarray(values, dtype=np.float64)


def _normalized_histogram(values: np.ndarray, classes: int) -> np.ndarray:
    if values.ndim != 1 or np.any(values < 0) or np.any(values >= classes):
        raise UgiDynamicControllerAnalysisError("categorical checkpoint channel is invalid")
    if not len(values):
        return np.zeros(classes, dtype=np.float64)
    return np.bincount(values.astype(np.int64), minlength=classes).astype(np.float64) / len(values)


def partial_state_features(
    program: Mapping[str, Sequence[int]],
    channels: Mapping[str, torch.Tensor],
    *,
    checkpoint: int,
    sample_steps: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return M0 plus leakage-free categorical checkpoint summaries."""

    m0_names, m0 = morphology_features(program)
    required = {
        "decoration_anchors",
        "decoration_atoms",
        "decoration_bonds",
        "nodes",
        "offspring",
        "parent_bonds",
    }
    if set(channels) != required or not 0 <= checkpoint <= sample_steps:
        raise UgiDynamicControllerAnalysisError("checkpoint channel contract changed")
    arrays: dict[str, np.ndarray] = {}
    for key, value in channels.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 1:
            raise UgiDynamicControllerAnalysisError("checkpoint tensor shape changed")
        arrays[key] = value.detach().cpu().numpy()[0].astype(np.int64, copy=False)

    node_counts = tuple(int(value) for value in program["node_counts"])
    total_nodes = sum(node_counts)
    if any(len(arrays[key]) != total_nodes for key in ("nodes", "offspring", "parent_bonds")):
        raise UgiDynamicControllerAnalysisError("checkpoint node width changed")

    names = list(m0_names)
    features = m0.tolist()
    for step in (2, 4, 6):
        names.append(f"checkpoint_is_{step}")
        features.append(float(checkpoint == step))

    start = 0
    for role, count in zip(ROLES, node_counts, strict=True):
        stop = start + count
        role_offspring = arrays["offspring"][start:stop]
        role_nodes = arrays["nodes"][start:stop]
        role_bonds = arrays["parent_bonds"][start:stop]
        for label, vector in (
            ("offspring", _normalized_histogram(role_offspring, 4)),
            ("node_state", _normalized_histogram(role_nodes, 14)),
            ("parent_bond", _normalized_histogram(role_bonds, 4)),
        ):
            for category, value in enumerate(vector):
                names.append(f"{role}_{label}_{category}_fraction")
                features.append(float(value))
        names.extend(
            (
                f"{role}_realized_junction_fraction",
                f"{role}_path_token_fraction",
                f"{role}_leaf_token_fraction",
            )
        )
        features.extend(
            (
                float(np.mean(role_offspring >= 2)),
                float(np.mean(role_offspring == 1)),
                float(np.mean(role_offspring == 0)),
            )
        )
        start = stop

    anchors = arrays["decoration_anchors"]
    active = anchors > 0
    names.append("active_decoration_fraction")
    features.append(float(np.mean(active)) if len(active) else 0.0)
    for label, values, classes in (
        ("active_decoration_atom", arrays["decoration_atoms"][active], 14),
        ("active_decoration_bond", arrays["decoration_bonds"][active], 4),
    ):
        for category, value in enumerate(_normalized_histogram(values, classes)):
            names.append(f"{label}_{category}_fraction")
            features.append(float(value))
    vector = np.asarray(features, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise UgiDynamicControllerAnalysisError("nonfinite checkpoint feature")
    return tuple(names), vector


def _sigmoid(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value, dtype=np.float64)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


@dataclass
class BinomialRidge:
    l2: float
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    coefficients_: np.ndarray | None = None

    def fit(self, features: np.ndarray, successes: np.ndarray, trials: np.ndarray) -> BinomialRidge:
        if features.ndim != 2 or len(features) != len(successes) or len(features) != len(trials):
            raise UgiDynamicControllerAnalysisError("invalid binomial training arrays")
        if np.any(trials <= 0) or np.any(successes < 0) or np.any(successes > trials):
            raise UgiDynamicControllerAnalysisError("invalid binomial outcomes")
        weights = trials.astype(np.float64)
        self.mean_ = np.average(features, axis=0, weights=weights)
        variance = np.average((features - self.mean_) ** 2, axis=0, weights=weights)
        self.scale_ = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        standardized = (features - self.mean_) / self.scale_
        design = np.column_stack((np.ones(len(features)), standardized))
        fraction = successes / trials
        prevalence = float(np.clip(successes.sum() / trials.sum(), 1e-6, 1.0 - 1e-6))
        beta = np.zeros(design.shape[1], dtype=np.float64)
        beta[0] = math.log(prevalence / (1.0 - prevalence))
        penalty = np.full(design.shape[1], float(self.l2), dtype=np.float64)
        penalty[0] = 0.0
        for _ in range(100):
            probabilities = np.clip(_sigmoid(design @ beta), 1e-8, 1.0 - 1e-8)
            gradient = design.T @ (weights * (probabilities - fraction)) + penalty * beta
            curvature = weights * probabilities * (1.0 - probabilities)
            hessian = design.T @ (curvature[:, None] * design) + np.diag(penalty + 1e-9)
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError as error:
                raise UgiDynamicControllerAnalysisError("binomial ridge solve failed") from error
            beta -= step
            if float(np.max(np.abs(step))) < 1e-8:
                break
        self.coefficients_ = beta
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coefficients_ is None:
            raise UgiDynamicControllerAnalysisError("binomial model is not fitted")
        design = np.column_stack((np.ones(len(features)), (features - self.mean_) / self.scale_))
        return np.clip(_sigmoid(design @ self.coefficients_), 1e-8, 1.0 - 1e-8)


def stable_group_folds(groups: Sequence[str], *, folds: int, salt: str) -> np.ndarray:
    if folds < 2:
        raise UgiDynamicControllerAnalysisError("at least two folds are required")
    assigned: dict[str, int] = {}
    for group in sorted(set(groups)):
        digest = hashlib.sha256(f"{salt}|{group}".encode()).digest()
        assigned[group] = int.from_bytes(digest[:8], "big") % folds
    output = np.asarray([assigned[group] for group in groups], dtype=np.int64)
    if len(set(output.tolist())) != folds:
        raise UgiDynamicControllerAnalysisError("group hash did not populate every fold")
    return output


def _fit_oof(
    observations: Sequence[Observation],
    *,
    folds: int,
    salt: str,
    l2: float,
    hgb: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    features = np.asarray([record.features for record in observations], dtype=np.float64)
    successes = np.asarray([record.successes for record in observations], dtype=np.float64)
    trials = np.asarray([record.trials for record in observations], dtype=np.float64)
    fractions = successes / trials
    groups = [record.group for record in observations]
    assigned = stable_group_folds(groups, folds=folds, salt=salt)
    logistic = np.zeros(len(observations), dtype=np.float64)
    challenger = np.zeros(len(observations), dtype=np.float64)
    intercept = np.zeros(len(observations), dtype=np.float64)
    for fold in range(folds):
        train = assigned != fold
        test = ~train
        if (
            not np.any(test)
            or successes[train].sum() <= 0
            or successes[train].sum() >= trials[train].sum()
        ):
            raise UgiDynamicControllerAnalysisError("degenerate program-disjoint fold")
        prevalence = float(successes[train].sum() / trials[train].sum())
        intercept[test] = prevalence
        logistic[test] = (
            BinomialRidge(l2=l2)
            .fit(features[train], successes[train], trials[train])
            .predict(features[test])
        )
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(hgb["learning_rate"]),
            max_iter=int(hgb["max_iter"]),
            max_leaf_nodes=int(hgb["max_leaf_nodes"]),
            max_depth=int(hgb["max_depth"]),
            min_samples_leaf=int(hgb["min_samples_leaf"]),
            l2_regularization=float(hgb["l2_regularization"]),
            random_state=int(hgb["random_state"]),
        )
        model.fit(features[train], fractions[train], sample_weight=trials[train])
        challenger[test] = np.clip(model.predict(features[test]), 1e-8, 1.0 - 1e-8)
    return {
        "fold": assigned,
        "intercept": intercept,
        "logistic": logistic,
        "hgb": challenger,
    }


def _expanded_outcomes(
    successes: np.ndarray, trials: np.ndarray, predictions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    scores: list[float] = []
    for success, trial, prediction in zip(successes, trials, predictions, strict=True):
        labels.extend([1] * int(success))
        labels.extend([0] * int(trial - success))
        scores.extend([float(prediction)] * int(trial))
    return np.asarray(labels, dtype=np.int8), np.asarray(scores, dtype=np.float64)


def prediction_metrics(
    observations: Sequence[Observation], predictions: np.ndarray
) -> dict[str, Any]:
    successes = np.asarray([record.successes for record in observations], dtype=np.float64)
    trials = np.asarray([record.trials for record in observations], dtype=np.float64)
    fraction = successes / trials
    probability = np.clip(np.asarray(predictions, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    brier_terms = fraction * (1.0 - probability) ** 2 + (1.0 - fraction) * probability**2
    log_terms = -(fraction * np.log(probability) + (1.0 - fraction) * np.log1p(-probability))
    labels, scores = _expanded_outcomes(successes, trials, probability)
    order = np.argsort(probability, kind="mergesort")
    bins = np.array_split(order, min(10, len(order)))
    ece = 0.0
    for indices in bins:
        weight = float(trials[indices].sum())
        observed = float(successes[indices].sum() / weight)
        expected = float(np.average(probability[indices], weights=trials[indices]))
        ece += weight * abs(observed - expected)
    ece /= float(trials.sum())
    logits = np.log(probability / (1.0 - probability))[:, None]
    calibration = BinomialRidge(l2=1e-8).fit(logits, successes, trials)
    slope = (
        float(calibration.coefficients_[1]) if calibration.coefficients_ is not None else math.nan
    )
    return {
        "observations": len(observations),
        "trials": int(trials.sum()),
        "successes": int(successes.sum()),
        "prevalence": float(successes.sum() / trials.sum()),
        "brier": float(np.average(brier_terms, weights=trials)),
        "log_loss": float(np.average(log_terms, weights=trials)),
        "ece_10_equal_observation_bins": ece,
        "calibration_slope": slope,
        "average_precision": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
    }


def _rank_select(predictions: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 < fraction < 1.0:
        raise UgiDynamicControllerAnalysisError("allocation fraction is invalid")
    count = max(1, int(math.ceil(len(predictions) * fraction)))
    order = np.argsort(-predictions, kind="mergesort")
    selected = np.zeros(len(predictions), dtype=bool)
    selected[order[:count]] = True
    return selected


def allocation_metrics(
    observations: Sequence[Observation],
    predictions: np.ndarray,
    terminal_rows: Sequence[Mapping[str, Any]],
    *,
    fraction: float,
) -> tuple[dict[str, Any], np.ndarray]:
    selected = _rank_select(predictions, fraction)
    successes = np.asarray([record.successes for record in observations], dtype=np.float64)
    trials = np.asarray([record.trials for record in observations], dtype=np.float64)
    baseline_rate = float(successes.sum() / trials.sum())
    selected_rate = float(successes[selected].sum() / trials[selected].sum())
    selected_terminals = [
        index
        for keep, record in zip(selected, observations, strict=True)
        if keep
        for index in record.terminal_indices
    ]
    supported_smiles = {
        str(terminal_rows[index]["smiles"])
        for index in selected_terminals
        if terminal_rows[index]["support"] and terminal_rows[index]["smiles"]
    }
    all_supported_smiles = {
        str(row["smiles"]) for row in terminal_rows if row["support"] and row["smiles"]
    }
    selected_diverse_yield = len(supported_smiles) / len(selected_terminals)
    baseline_diverse_yield = len(all_supported_smiles) / len(terminal_rows)
    return (
        {
            "allocation_fraction": float(selected.mean()),
            "selected_observations": int(selected.sum()),
            "selected_trials": int(trials[selected].sum()),
            "selected_successes": int(successes[selected].sum()),
            "baseline_support_rate": baseline_rate,
            "selected_support_rate": selected_rate,
            "support_rate_relative_improvement": selected_rate / baseline_rate - 1.0,
            "baseline_unique_supported_smiles_per_attempt": baseline_diverse_yield,
            "selected_unique_supported_smiles_per_attempt": selected_diverse_yield,
            "diverse_yield_relative_improvement": (
                selected_diverse_yield / baseline_diverse_yield - 1.0
                if baseline_diverse_yield > 0
                else 0.0
            ),
            "selected_unique_program_groups": len(
                {record.group for keep, record in zip(selected, observations, strict=True) if keep}
            ),
            "selected_program_group_fraction": len(
                {record.group for keep, record in zip(selected, observations, strict=True) if keep}
            )
            / len({record.group for record in observations}),
        },
        selected,
    )


def _cluster_bootstrap_difference(
    observations: Sequence[Observation],
    first: np.ndarray,
    second: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    groups = sorted({record.group for record in observations})
    index_by_group = {group: position for position, group in enumerate(groups)}
    numerator = np.zeros(len(groups), dtype=np.float64)
    denominator = np.zeros(len(groups), dtype=np.float64)
    for index, record in enumerate(observations):
        group_index = index_by_group[record.group]
        y = record.successes / record.trials
        first_loss = y * (1.0 - first[index]) ** 2 + (1.0 - y) * first[index] ** 2
        second_loss = y * (1.0 - second[index]) ** 2 + (1.0 - y) * second[index] ** 2
        numerator[group_index] += record.trials * (first_loss - second_loss)
        denominator[group_index] += record.trials
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        values[replicate] = float(np.dot(counts, numerator) / np.dot(counts, denominator))
    return {
        "point_difference": float(numerator.sum() / denominator.sum()),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "replicates": replicates,
        "unit": "program_sha256",
    }


def controller_decision(
    m0: Mapping[str, Any],
    m1: Mapping[str, Any],
    *,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply frozen, complexity-penalizing go/no-go rules."""

    m0_checks = {
        "brier_relative_reduction": m0["brier_relative_reduction"]
        >= float(gates["m0_minimum_brier_relative_reduction"]),
        "average_precision_lift": m0["average_precision_lift"]
        >= float(gates["m0_minimum_average_precision_lift"]),
        "auroc": m0["metrics"]["auroc"] >= float(gates["minimum_auroc"]),
        "calibration_slope": float(gates["minimum_calibration_slope"])
        <= m0["metrics"]["calibration_slope"]
        <= float(gates["maximum_calibration_slope"]),
        "program_coverage": m0["allocation"]["selected_program_group_fraction"]
        >= float(gates["minimum_program_coverage"]),
        "support_yield": m0["allocation"]["support_rate_relative_improvement"]
        >= float(gates["m0_minimum_support_yield_relative_improvement"]),
        "diverse_yield": m0["allocation"]["diverse_yield_relative_improvement"]
        >= float(gates["minimum_diverse_yield_relative_improvement"]),
        "clustered_brier_ci": m0["brier_difference_bootstrap"]["ci95_lower"] > 0.0,
    }
    m1_checks = {
        "brier_relative_reduction_over_m0": m1["brier_relative_reduction_over_m0"]
        >= float(gates["m1_minimum_brier_relative_reduction_over_m0"]),
        "average_precision_lift_over_m0": m1["average_precision_lift_over_m0"]
        >= float(gates["m1_minimum_average_precision_lift_over_m0"]),
        "auroc": m1["metrics"]["auroc"] >= float(gates["minimum_auroc"]),
        "calibration_slope": float(gates["minimum_calibration_slope"])
        <= m1["metrics"]["calibration_slope"]
        <= float(gates["maximum_calibration_slope"]),
        "support_yield_over_m0": m1["support_yield_relative_improvement_over_m0"]
        >= float(gates["m1_minimum_support_yield_relative_improvement_over_m0"]),
        "diverse_yield": m1["allocation"]["diverse_yield_relative_improvement"]
        >= float(gates["minimum_diverse_yield_relative_improvement"]),
        "clustered_brier_ci": m1["brier_difference_bootstrap"]["ci95_lower"] > 0.0,
    }
    m0_pass = all(m0_checks.values())
    m1_pass = all(m1_checks.values())
    if m1_pass and (
        m0["allocation"]["selected_support_rate"]
        < (1.0 - float(gates["simplicity_tolerance"])) * m1["allocation"]["selected_support_rate"]
    ):
        selected = "delayed_partial_state_controller_experiment"
    elif m0_pass:
        selected = "dynamic_morphology_proposal_experiment"
    else:
        selected = "plain_terminal_screening"
    return {
        "m0_checks": m0_checks,
        "m0_pass": m0_pass,
        "m1_checks": m1_checks,
        "m1_pass": m1_pass,
        "selected_next_controller": selected,
        "smc_execution_authorized": False,
        "potency_guidance_authorized": False,
        "interpretation": "controller_experiment_choice_only",
    }


def _terminal_support_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    references: Mapping[str, applicability_v2.CountChemicalReference],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        terminal = row.get("native_terminal")
        if not isinstance(terminal, Mapping):
            raise UgiDynamicControllerAnalysisError("native terminal payload is missing")
        valid = bool(terminal.get("valid")) and bool(
            terminal.get("l1_forward_verification", {}).get("exact_product_reconstructed")
        )
        radius = math.inf
        smiles = ""
        exact_measured = False
        exact_new_roles: tuple[str, ...] = ()
        if valid:
            smiles = str(terminal["smiles"])
            components = terminal.get("component_smiles_by_role")
            if not isinstance(components, Mapping) or set(components) != set(
                ROLE_COMPONENT_KEYS.values()
            ):
                raise UgiDynamicControllerAnalysisError("terminal component payload changed")
            query = {
                "product_smiles": smiles,
                "amine_smiles": str(components[ROLE_COMPONENT_KEYS["amine"]]),
                "aldehyde_smiles": str(components[ROLE_COMPONENT_KEYS["aldehyde"]]),
                "isocyanide_smiles": str(components[ROLE_COMPONENT_KEYS["isocyanide"]]),
            }
            distances = applicability_v1._distance_fields(references, query, generated=True)
            ratios = []
            for view in VIEWS:
                ratios.extend(
                    (
                        distances[view].fingerprint
                        / float(thresholds[view]["fingerprint"]["interpolative_max"]),
                        distances[view].descriptor
                        / float(thresholds[view]["descriptor"]["interpolative_max"]),
                    )
                )
            radius = float(max(ratios))
            exact_measured = (
                applicability_v1._canonical(smiles) in references["product"].canonical_set
            )
            exact_new_roles = tuple(
                role
                for role in ROLES
                if applicability_v1._canonical(query[f"{role}_smiles"])
                not in references[role].canonical_set
            )
        support = bool(valid and radius <= 1.0)
        output.append(
            {
                "terminal_index": index,
                "shard_index": int(row["shard_index"]),
                "selection_rank": int(row["selection_rank"]),
                "population_index": int(row["population_index"]),
                "particle_index": int(row["particle_index"]),
                "state_replicate": int(row["state_replicate"]),
                "checkpoint": int(row["checkpoint_index"]),
                "rollout_index": int(row["rollout_index"]),
                "program_sha256": str(row["program_sha256"]),
                "valid_exact_l1": valid,
                "radius": radius if math.isfinite(radius) else "",
                "support": support,
                "exact_measured_product": exact_measured,
                "exact_new_roles_json": json.dumps(exact_new_roles, separators=(",", ":")),
                "smiles": smiles,
            }
        )
    return output


def _build_observations(
    repo: Path,
    *,
    census_root: Path,
    manifest_path: Path,
    terminal_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], list[Observation], tuple[str, ...], list[Observation]]:
    by_state: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    by_program_row: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(terminal_rows):
        by_state[
            (
                int(row["shard_index"]),
                int(row["particle_index"]),
                int(row["checkpoint"]),
            )
        ].append(index)
        by_program_row[int(row["selection_rank"])].append(index)
    if set(map(len, by_state.values())) != {4} or set(map(len, by_program_row.values())) != {24}:
        raise UgiDynamicControllerAnalysisError("terminal replication lattice changed")

    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, list) or not manifest:
        raise UgiDynamicControllerAnalysisError("partial-state manifest changed")
    m1_names: tuple[str, ...] | None = None
    m1: list[Observation] = []
    program_payloads: dict[int, tuple[dict[str, tuple[int, int, int]], str]] = {}
    for artifact in manifest:
        path = (census_root / str(artifact["path"])).resolve()
        try:
            path.relative_to(census_root)
        except ValueError as error:
            raise UgiDynamicControllerAnalysisError("partial state escapes census root") from error
        if path.is_symlink() or not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise UgiDynamicControllerAnalysisError("partial-state artifact pin changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint = int(payload["step"])
        shard_index = int(artifact["shard_index"])
        for particle in payload["particles"]:
            particle_index = int(particle["global_particle_index"])
            key = (shard_index, particle_index, checkpoint)
            if key not in by_state:
                raise UgiDynamicControllerAnalysisError("partial state lacks terminal outcomes")
            program = decode_program_b64(str(particle["program_b64"]))
            names, features = partial_state_features(
                program,
                particle["channels"],
                checkpoint=checkpoint,
                sample_steps=int(particle["sample_steps"]),
            )
            if m1_names is None:
                m1_names = names
            elif names != m1_names:
                raise UgiDynamicControllerAnalysisError("M1 feature schema changed")
            terminal_indices = tuple(sorted(by_state[key]))
            group = str(terminal_rows[terminal_indices[0]]["program_sha256"])
            if group != str(particle["program_sha256"]):
                raise UgiDynamicControllerAnalysisError("partial-state program identity changed")
            selection_rank = int(terminal_rows[terminal_indices[0]]["selection_rank"])
            program_payloads.setdefault(selection_rank, (program, group))
            if program_payloads[selection_rank] != (program, group):
                raise UgiDynamicControllerAnalysisError("program row changed across states")
            successes = sum(bool(terminal_rows[index]["support"]) for index in terminal_indices)
            m1.append(
                Observation(
                    observation_id=(
                        f"shard-{shard_index:02d}-particle-{particle_index:04d}"
                        f"-step-{checkpoint:02d}"
                    ),
                    group=group,
                    features=tuple(features.tolist()),
                    successes=successes,
                    trials=4,
                    terminal_indices=terminal_indices,
                    checkpoint=checkpoint,
                )
            )
    if len(m1) != 6144 or m1_names is None:
        raise UgiDynamicControllerAnalysisError("M1 observation count changed")

    m0_names: tuple[str, ...] | None = None
    m0: list[Observation] = []
    for selection_rank, terminal_indices_list in sorted(by_program_row.items()):
        if selection_rank not in program_payloads:
            raise UgiDynamicControllerAnalysisError("program row has no checkpoint payload")
        program, group = program_payloads[selection_rank]
        names, features = morphology_features(program)
        if m0_names is None:
            m0_names = names
        elif names != m0_names:
            raise UgiDynamicControllerAnalysisError("M0 feature schema changed")
        terminal_indices = tuple(sorted(terminal_indices_list))
        successes = sum(bool(terminal_rows[index]["support"]) for index in terminal_indices)
        m0.append(
            Observation(
                observation_id=f"selection-rank-{selection_rank:04d}",
                group=group,
                features=tuple(features.tolist()),
                successes=successes,
                trials=24,
                terminal_indices=terminal_indices,
                checkpoint=None,
            )
        )
    if len(m0) != 1024 or m0_names is None:
        raise UgiDynamicControllerAnalysisError("M0 observation count changed")
    return m0_names, m0, m1_names, sorted(m1, key=lambda record: record.observation_id)


def build_dynamic_controller_analysis(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes, bytes]:
    """Build the deterministic, nonselecting controller-choice analysis."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="controller config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiDynamicControllerAnalysisError("unsupported controller config schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiDynamicControllerAnalysisError("controller scope changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != EXPECTED_INPUTS:
        raise UgiDynamicControllerAnalysisError("controller input pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in raw_inputs.items()}

    census = _load_json(paths["census_result"], label="census result")
    validation = _load_json(paths["census_validation_result"], label="census validation")
    applicability = _load_json(paths["applicability_result"], label="applicability result")
    if validation.get("aggregate_result_sha256") != census.get("result_sha256"):
        raise UgiDynamicControllerAnalysisError("independent census validation does not match")
    if census.get("counts", {}).get("terminal_attempts") != 24576:
        raise UgiDynamicControllerAnalysisError("census is incomplete")
    thresholds = applicability.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(VIEWS):
        raise UgiDynamicControllerAnalysisError("frozen applicability thresholds changed")

    curated = _read_csv(paths["curated_agile"])
    reference_rows = [{**row, "product_smiles": row["model_smiles"]} for row in curated]
    references = applicability_v2._references(reference_rows)
    raw_terminals = _read_jsonl_gz(paths["terminal_ledger"])
    terminal_rows = _terminal_support_rows(
        raw_terminals, references=references, thresholds=thresholds
    )
    census_root = paths["census_result"].parent
    m0_names, m0, m1_names, m1 = _build_observations(
        repo,
        census_root=census_root,
        manifest_path=paths["partial_state_manifest"],
        terminal_rows=terminal_rows,
    )

    analysis = config["analysis"]
    oof_m0 = _fit_oof(
        m0,
        folds=int(analysis["outer_folds"]),
        salt=str(analysis["fold_salt"]),
        l2=float(analysis["logistic_l2"]),
        hgb=analysis["hgb"],
    )
    oof_m1 = _fit_oof(
        m1,
        folds=int(analysis["outer_folds"]),
        salt=str(analysis["fold_salt"]),
        l2=float(analysis["logistic_l2"]),
        hgb=analysis["hgb"],
    )
    m0_metrics = prediction_metrics(m0, oof_m0["logistic"])
    m0_intercept = prediction_metrics(m0, oof_m0["intercept"])
    m0_hgb = prediction_metrics(m0, oof_m0["hgb"])
    m0_allocation, m0_selected = allocation_metrics(
        m0,
        oof_m0["logistic"],
        terminal_rows,
        fraction=float(analysis["allocation_fraction"]),
    )
    m0_prediction_by_group: dict[str, list[float]] = defaultdict(list)
    for record, prediction in zip(m0, oof_m0["logistic"], strict=True):
        m0_prediction_by_group[record.group].append(float(prediction))
    m0_on_m1 = np.asarray(
        [np.mean(m0_prediction_by_group[record.group]) for record in m1], dtype=np.float64
    )
    m1_metrics = prediction_metrics(m1, oof_m1["logistic"])
    m1_m0 = prediction_metrics(m1, m0_on_m1)
    m1_hgb = prediction_metrics(m1, oof_m1["hgb"])
    m1_allocation, m1_selected = allocation_metrics(
        m1,
        oof_m1["logistic"],
        terminal_rows,
        fraction=float(analysis["allocation_fraction"]),
    )
    m0_allocation_on_m1, _ = allocation_metrics(
        m1,
        m0_on_m1,
        terminal_rows,
        fraction=float(analysis["allocation_fraction"]),
    )
    bootstrap = analysis["clustered_bootstrap"]
    m0_bootstrap = _cluster_bootstrap_difference(
        m0,
        oof_m0["intercept"],
        oof_m0["logistic"],
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
    )
    m1_bootstrap = _cluster_bootstrap_difference(
        m1,
        m0_on_m1,
        oof_m1["logistic"],
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]) + 1,
    )

    m0_record = {
        "feature_count": len(m0_names),
        "feature_names": list(m0_names),
        "metrics": m0_metrics,
        "intercept_metrics": m0_intercept,
        "hgb_challenger_metrics": m0_hgb,
        "brier_relative_reduction": 1.0 - m0_metrics["brier"] / m0_intercept["brier"],
        "average_precision_lift": m0_metrics["average_precision"] / m0_metrics["prevalence"],
        "allocation": m0_allocation,
        "brier_difference_bootstrap": m0_bootstrap,
    }
    m1_record = {
        "feature_count": len(m1_names),
        "feature_names": list(m1_names),
        "metrics": m1_metrics,
        "m0_on_m1_metrics": m1_m0,
        "hgb_challenger_metrics": m1_hgb,
        "brier_relative_reduction_over_m0": 1.0 - m1_metrics["brier"] / m1_m0["brier"],
        "average_precision_lift_over_m0": m1_metrics["average_precision"]
        / m1_m0["average_precision"]
        - 1.0,
        "allocation": m1_allocation,
        "m0_allocation_on_m1": m0_allocation_on_m1,
        "support_yield_relative_improvement_over_m0": (
            m1_allocation["selected_support_rate"] / m0_allocation_on_m1["selected_support_rate"]
            - 1.0
        ),
        "brier_difference_bootstrap": m1_bootstrap,
        "by_checkpoint": {
            str(checkpoint): prediction_metrics(
                [record for record in m1 if record.checkpoint == checkpoint],
                oof_m1["logistic"][np.asarray([record.checkpoint == checkpoint for record in m1])],
            )
            for checkpoint in (2, 4, 6)
        },
    }
    decision = controller_decision(m0_record, m1_record, gates=analysis["gates"])

    terminal_fields = (
        "terminal_index",
        "shard_index",
        "selection_rank",
        "population_index",
        "particle_index",
        "state_replicate",
        "checkpoint",
        "rollout_index",
        "program_sha256",
        "valid_exact_l1",
        "radius",
        "support",
        "exact_measured_product",
        "exact_new_roles_json",
        "smiles",
    )
    terminal_bytes = _csv_gz_bytes(terminal_rows, terminal_fields)
    oof_rows: list[dict[str, Any]] = []
    for level, observations, predictions, selected in (
        ("M0", m0, oof_m0, m0_selected),
        ("M1", m1, oof_m1, m1_selected),
    ):
        for index, record in enumerate(observations):
            oof_rows.append(
                {
                    "level": level,
                    "observation_id": record.observation_id,
                    "program_sha256": record.group,
                    "checkpoint": "" if record.checkpoint is None else record.checkpoint,
                    "successes": record.successes,
                    "trials": record.trials,
                    "fold": int(predictions["fold"][index]),
                    "intercept_probability": float(predictions["intercept"][index]),
                    "logistic_probability": float(predictions["logistic"][index]),
                    "hgb_probability": float(predictions["hgb"][index]),
                    "selected_top_fraction": bool(selected[index]),
                }
            )
    oof_fields = (
        "level",
        "observation_id",
        "program_sha256",
        "checkpoint",
        "successes",
        "trials",
        "fold",
        "intercept_probability",
        "logistic_probability",
        "hgb_probability",
        "selected_top_fraction",
    )
    oof_bytes = _csv_gz_bytes(oof_rows, oof_fields)

    recorded_inputs = {
        label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
        for label, path in sorted(paths.items())
    }
    recorded_inputs["config"] = {
        "path": str(config_path.relative_to(repo)),
        "sha256": sha256_file(config_path),
    }
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_dynamic_controller_analysis",
        "scope": dict(EXPECTED_SCOPE),
        "inputs": recorded_inputs,
        "support_definition": {
            "terminal_valid_exact_l1": True,
            "fixed_multiview_normalized_radius_maximum": 1.0,
            "potency_used": False,
            "role_novelty_used_as_gate": False,
        },
        "census": {
            "terminal_attempts": len(terminal_rows),
            "valid_exact_l1": sum(bool(row["valid_exact_l1"]) for row in terminal_rows),
            "supported": sum(bool(row["support"]) for row in terminal_rows),
            "exact_measured_supported": sum(
                bool(row["support"] and row["exact_measured_product"]) for row in terminal_rows
            ),
            "all_three_roles_new_supported": sum(
                bool(
                    row["support"]
                    and row["exact_new_roles_json"] == '["amine","aldehyde","isocyanide"]'
                )
                for row in terminal_rows
            ),
            "unique_program_rows": len(m0),
            "unique_program_sha256": len({record.group for record in m0}),
            "partial_state_observations": len(m1),
        },
        "analysis_contract": {
            "outer_group": "program_sha256",
            "outer_folds": int(analysis["outer_folds"]),
            "m0_outcome": "binomial_k_of_24",
            "m1_outcome": "binomial_k_of_4",
            "m0_features": "frozen_12_coordinate_morphology_program_only",
            "m1_features": "M0_plus_saved_categorical_checkpoint_summaries",
            "terminal_information_in_features": False,
            "identifiers_or_rng_in_features": False,
            "hgb_is_challenger_only": True,
        },
        "m0_morphology": m0_record,
        "m1_partial_state": m1_record,
        "decision": decision,
        "artifacts": {
            "terminal_support.csv.gz": {
                "schema_version": TERMINAL_LEDGER_SCHEMA_VERSION,
                "records": len(terminal_rows),
                "sha256": sha256_bytes(terminal_bytes),
            },
            "oof_predictions.csv.gz": {
                "schema_version": OOF_LEDGER_SCHEMA_VERSION,
                "records": len(oof_rows),
                "sha256": sha256_bytes(oof_bytes),
            },
        },
        "nonclaims": [
            "This analysis does not score or optimize potency.",
            "This analysis does not authorize SMC execution.",
            "This analysis does not call routes, synthesis or proposal engines.",
            "The fixed structural radius is not a calibrated probability of biological success.",
            "A selected controller experiment still requires an unused-program confirmation.",
        ],
    }
    result["result_sha256"] = _logical_sha256(result)
    return result, terminal_bytes, oof_bytes


__all__ = [
    "BinomialRidge",
    "Observation",
    "UgiDynamicControllerAnalysisError",
    "build_dynamic_controller_analysis",
    "controller_decision",
    "decode_program_b64",
    "morphology_features",
    "partial_state_features",
    "prediction_metrics",
    "stable_group_folds",
]
