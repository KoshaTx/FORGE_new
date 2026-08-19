"""Run the leak-aware classical lane of the M0-07 AGILE oracle matrix.

This module evaluates count-based Morgan fingerprints, structure-computable
RDKit descriptors, and their concatenation with fixed classical regressors.
All preprocessing is fitted on training rows only. Calibration rows are used
only after model fitting to construct split-conformal intervals.

The lane is an M0 audit. It does not freeze a biological oracle.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import tempfile
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from forge.bio.agile_reconciliation import sha256_file

CONFIG_SCHEMA_VERSION = "m0_07_oracle_classical_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_oracle_classical.v1"
STAGES = ("train", "calibration", "test")
SERIALIZED_FLOAT_SIGNIFICANT_DIGITS = 12

METRIC_FIELDS = (
    "representation",
    "model",
    "endpoint",
    "scheme",
    "fold",
    "fit_seed",
    "train_rows",
    "calibration_rows",
    "test_rows",
    "calibration_r2",
    "calibration_rmse",
    "calibration_mae",
    "calibration_pearson_r",
    "calibration_spearman_rho",
    "test_r2",
    "test_rmse",
    "test_mae",
    "test_pearson_r",
    "test_spearman_rho",
    "conformal_q80",
    "test_coverage80",
    "test_mean_interval_width80",
    "conformal_q90",
    "test_coverage90",
    "test_mean_interval_width90",
    "conformal_q95",
    "test_coverage95",
    "test_mean_interval_width95",
    "fit_warning_count",
    "fit_warning_categories",
)

PREDICTION_FIELDS = (
    "representation",
    "model",
    "endpoint",
    "scheme",
    "fold",
    "label",
    "y_true",
    "y_pred",
    "absolute_error",
)

APPLICABILITY_FIELDS = (
    "source_row_index",
    "canonical_model_smiles",
    "nearest_curated_label",
    "max_binary_morgan_tanimoto",
)


class OracleClassicalError(ValueError):
    """Raised when the classical oracle lane violates its frozen contract."""


@dataclass(frozen=True)
class FeatureBundle:
    """In-memory deterministic feature representations."""

    labels: tuple[str, ...]
    model_smiles: tuple[str, ...]
    matrices: Mapping[str, np.ndarray]
    feature_names: Mapping[str, tuple[str, ...]]
    descriptor_failures: Mapping[str, int]
    descriptor_nonfinite: Mapping[str, int]


@dataclass(frozen=True)
class Partition:
    """One frozen train, calibration, and test partition."""

    scheme: str
    fold: int
    train_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise OracleClassicalError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OracleClassicalError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OracleClassicalError(f"{label} must be a JSON object")
    return payload


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _verify_hash(path: Path, expected: Any, label: str, repo_root: Path) -> dict[str, Any]:
    if not isinstance(expected, str) or len(expected) != 64:
        raise OracleClassicalError(f"{label} expected sha256 must contain 64 characters")
    if not path.is_file():
        raise OracleClassicalError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise OracleClassicalError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return {
        "path": _portable(path, repo_root),
        "sha256": observed,
        "bytes": path.stat().st_size,
    }


def load_classical_config(path: Path) -> dict[str, Any]:
    """Load and validate the frozen classical-lane configuration."""

    config = _load_json(path, "M0-07 classical oracle config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise OracleClassicalError(
            f"unsupported classical config schema: {config.get('schema_version')!r}"
        )
    if config.get("randomness", {}).get("seed") != 1729:
        raise OracleClassicalError("classical oracle config must freeze seed 1729")
    if (
        config.get("randomness", {}).get("numeric_serialization_significant_digits")
        != SERIALIZED_FLOAT_SIGNIFICANT_DIGITS
    ):
        raise OracleClassicalError(
            "classical oracle config must serialize floats at 12 significant digits"
        )
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise OracleClassicalError("classical oracle policy must be an object")
    required_policy = {
        "fit_preprocessors_on_training_rows_only": True,
        "calibration_rows_used_only_after_model_fit": True,
        "random_split_role": "reproduction_diagnostic_only",
        "virtual_candidate_role": "unlabeled_applicability_only",
        "oracle_model_frozen": False,
    }
    for field, expected in required_policy.items():
        if policy.get(field) != expected:
            raise OracleClassicalError(
                f"policy {field!r} must be {expected!r}, found {policy.get(field)!r}"
            )
    if config.get("evaluation", {}).get("model_selection_uses_random_split") is not False:
        raise OracleClassicalError("random split must not be used for model selection")
    required_models = {"ridge", "random_forest", "xgboost", "mlp"}
    if set(config.get("models", {})) != required_models:
        raise OracleClassicalError(f"classical model set must be {sorted(required_models)}")
    required_representations = {
        "morgan_count",
        "rdkit_expert",
        "morgan_plus_rdkit_expert",
    }
    if set(config.get("representations", {})) != required_representations:
        raise OracleClassicalError(
            f"classical representations must be {sorted(required_representations)}"
        )
    coverages = config.get("evaluation", {}).get("conformal_coverages")
    if coverages != [0.8, 0.9, 0.95]:
        raise OracleClassicalError("conformal coverages must be [0.8, 0.9, 0.95]")
    return config


def _read_csv(path: Path, required_fields: Sequence[str], label: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise OracleClassicalError(f"cannot read {label}: {path}: {exc}") from exc
    missing = set(required_fields) - set(frame.columns)
    if missing:
        raise OracleClassicalError(f"{label} is missing fields: {sorted(missing)}")
    return frame


def _parse_molecule(smiles: str, label: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise OracleClassicalError(f"{label} contains invalid SMILES: {smiles!r}")
    return molecule


def _canonical_model_smiles(smiles: str, label: str) -> str:
    molecule = _parse_molecule(smiles, label)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _array_sha256(matrix: np.ndarray, names: Sequence[str]) -> str:
    canonical = np.asarray(matrix, dtype="<f8", order="C")
    canonical = np.where(np.isfinite(canonical), canonical, np.nan)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(canonical.shape)).encode())
    digest.update(b"\x00")
    digest.update("\x1f".join(names).encode())
    digest.update(b"\x00")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def build_feature_bundle(
    frame: pd.DataFrame,
    representation_config: Mapping[str, Any],
) -> FeatureBundle:
    """Compute deterministic Morgan-count and RDKit descriptor matrices."""

    labels = tuple(str(value) for value in frame["label"])
    if len(labels) != len(set(labels)):
        raise OracleClassicalError("curated oracle labels are not unique")
    model_smiles = tuple(str(value) for value in frame["model_smiles"])
    molecules = [
        _parse_molecule(smiles, f"curated record {label}")
        for label, smiles in zip(labels, model_smiles, strict=True)
    ]

    morgan_config = representation_config["morgan_count"]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(morgan_config["radius"]),
        fpSize=int(morgan_config["size"]),
        includeChirality=bool(morgan_config["include_chirality"]),
    )
    morgan = np.vstack(
        [generator.GetCountFingerprintAsNumPy(molecule) for molecule in molecules]
    ).astype(np.float64, copy=False)
    morgan_names = tuple(f"morgan_count_{index}" for index in range(morgan.shape[1]))

    descriptor_entries = sorted(Descriptors.descList, key=lambda item: item[0])
    descriptor_names = tuple(name for name, _ in descriptor_entries)
    descriptor_matrix = np.empty((len(molecules), len(descriptor_entries)), dtype=np.float64)
    descriptor_failures: Counter[str] = Counter()
    for row_index, molecule in enumerate(molecules):
        for column_index, (name, function) in enumerate(descriptor_entries):
            try:
                value = float(function(molecule))
            except Exception:  # RDKit descriptor plugins do not share one exception type.
                value = math.nan
                descriptor_failures[name] += 1
            descriptor_matrix[row_index, column_index] = value if math.isfinite(value) else math.nan
    if np.any(np.all(~np.isfinite(descriptor_matrix), axis=1)):
        bad_rows = np.flatnonzero(np.all(~np.isfinite(descriptor_matrix), axis=1))
        raise OracleClassicalError(
            f"all RDKit descriptors failed for curated rows: {bad_rows[:10].tolist()}"
        )
    descriptor_nonfinite = {
        descriptor_names[index]: int(np.sum(~np.isfinite(descriptor_matrix[:, index])))
        for index in range(descriptor_matrix.shape[1])
        if np.any(~np.isfinite(descriptor_matrix[:, index]))
    }
    combined = np.hstack((morgan, descriptor_matrix))
    combined_names = morgan_names + tuple(f"rdkit_{name}" for name in descriptor_names)
    matrices = {
        "morgan_count": morgan,
        "rdkit_expert": descriptor_matrix,
        "morgan_plus_rdkit_expert": combined,
    }
    names = {
        "morgan_count": morgan_names,
        "rdkit_expert": descriptor_names,
        "morgan_plus_rdkit_expert": combined_names,
    }
    return FeatureBundle(
        labels=labels,
        model_smiles=model_smiles,
        matrices=matrices,
        feature_names=names,
        descriptor_failures=dict(sorted(descriptor_failures.items())),
        descriptor_nonfinite=descriptor_nonfinite,
    )


def load_partitions(
    assignments: pd.DataFrame,
    labels: Sequence[str],
) -> list[Partition]:
    """Map frozen label assignments to feature-row indices and revalidate partitions."""

    label_to_index = {label: index for index, label in enumerate(labels)}
    if len(label_to_index) != len(labels):
        raise OracleClassicalError("feature labels are not unique")
    assignment_labels = set(str(value) for value in assignments["label"])
    if assignment_labels != set(labels):
        raise OracleClassicalError("split assignment labels do not match the curated oracle data")
    partitions: list[Partition] = []
    grouped = assignments.groupby(["scheme", "fold"], sort=True)
    for (scheme_value, fold_value), rows in grouped:
        scheme = str(scheme_value)
        fold = int(fold_value)
        if len(rows) != len(labels) or rows["label"].nunique() != len(labels):
            raise OracleClassicalError(
                f"{scheme} fold {fold} does not assign every label exactly once"
            )
        stages = set(str(value) for value in rows["stage"])
        if stages != set(STAGES):
            raise OracleClassicalError(
                f"{scheme} fold {fold} stages are {sorted(stages)}, expected {list(STAGES)}"
            )

        def indices_for(stage: str) -> np.ndarray:
            stage_labels = sorted(str(value) for value in rows.loc[rows["stage"] == stage, "label"])
            return np.asarray([label_to_index[label] for label in stage_labels], dtype=np.int64)

        train = indices_for("train")
        calibration = indices_for("calibration")
        test = indices_for("test")
        flattened = np.concatenate((train, calibration, test))
        if len(flattened) != len(labels) or len(set(flattened.tolist())) != len(labels):
            raise OracleClassicalError(f"{scheme} fold {fold} partition overlaps or omits rows")
        partitions.append(
            Partition(
                scheme=scheme,
                fold=fold,
                train_indices=train,
                calibration_indices=calibration,
                test_indices=test,
            )
        )
    return partitions


def derive_fit_seed(
    global_seed: int,
    representation: str,
    model: str,
    endpoint: str,
    scheme: str,
    fold: int,
) -> int:
    """Derive a stable positive 31-bit seed for one fit."""

    key = "\x1f".join((str(global_seed), representation, model, endpoint, scheme, str(fold)))
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return value % (2**31 - 1) or 1


def build_estimator(
    model_name: str,
    model_config: Mapping[str, Any],
    *,
    seed: int,
) -> RegressorMixin:
    """Construct one fixed-hyperparameter, train-only preprocessing pipeline."""

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if model_name == "ridge":
        model = Ridge(alpha=float(model_config["alpha"]))
        return Pipeline(
            [
                ("impute", imputer),
                ("scale", StandardScaler()),
                ("model", model),
            ]
        )
    if model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=int(model_config["n_estimators"]),
            max_features=model_config["max_features"],
            min_samples_leaf=int(model_config["min_samples_leaf"]),
            random_state=seed,
            n_jobs=int(model_config["n_jobs"]),
        )
        return Pipeline([("impute", imputer), ("model", model)])
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise OracleClassicalError(
                "xgboost is required for M0-07; install the oracle optional dependencies"
            ) from exc
        model = XGBRegressor(
            n_estimators=int(model_config["n_estimators"]),
            max_depth=int(model_config["max_depth"]),
            learning_rate=float(model_config["learning_rate"]),
            subsample=float(model_config["subsample"]),
            colsample_bytree=float(model_config["colsample_bytree"]),
            reg_lambda=float(model_config["reg_lambda"]),
            tree_method=str(model_config["tree_method"]),
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=int(model_config["n_jobs"]),
            verbosity=0,
        )
        return Pipeline([("impute", imputer), ("model", model)])
    if model_name == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=tuple(int(value) for value in model_config["hidden_layer_sizes"]),
            activation=str(model_config["activation"]),
            solver=str(model_config["solver"]),
            alpha=float(model_config["alpha"]),
            learning_rate_init=float(model_config["learning_rate_init"]),
            batch_size=int(model_config["batch_size"]),
            max_iter=int(model_config["max_iter"]),
            early_stopping=bool(model_config["early_stopping"]),
            validation_fraction=float(model_config["validation_fraction"]),
            n_iter_no_change=int(model_config["n_iter_no_change"]),
            random_state=seed,
        )
        pipeline = Pipeline(
            [
                ("impute", imputer),
                ("scale", StandardScaler()),
                ("model", model),
            ]
        )
        return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
    raise OracleClassicalError(f"unsupported classical model: {model_name}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate the frozen regression metric set on original endpoint units."""

    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if truth.shape != prediction.shape or truth.size < 2:
        raise OracleClassicalError("regression metrics require aligned vectors of length >= 2")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise OracleClassicalError("regression metrics received nonfinite values")
    truth_sd = float(np.std(truth))
    prediction_sd = float(np.std(prediction))
    pearson = (
        float(np.corrcoef(truth, prediction)[0, 1])
        if truth_sd > 0.0 and prediction_sd > 0.0
        else math.nan
    )
    spearman_result = spearmanr(truth, prediction)
    spearman = float(spearman_result.statistic)
    return {
        "r2": float(r2_score(truth, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(truth, prediction))),
        "mae": float(mean_absolute_error(truth, prediction)),
        "pearson_r": pearson,
        "spearman_rho": spearman,
    }


def conformal_radius(residuals: np.ndarray, coverage: float) -> float:
    """Return the finite-sample split-conformal absolute-residual radius."""

    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise OracleClassicalError("conformal calibration residuals must be finite and nonempty")
    if not 0.0 < coverage < 1.0:
        raise OracleClassicalError(f"invalid conformal coverage: {coverage}")
    rank = min(values.size, math.ceil((values.size + 1) * coverage))
    return float(np.partition(values, rank - 1)[rank - 1])


def evaluate_partition(
    features: np.ndarray,
    targets: np.ndarray,
    labels: Sequence[str],
    partition: Partition,
    *,
    representation: str,
    model_name: str,
    endpoint: str,
    model_config: Mapping[str, Any],
    global_seed: int,
    coverages: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit and evaluate one representation/model/endpoint/partition tuple."""

    fit_seed = derive_fit_seed(
        global_seed,
        representation,
        model_name,
        endpoint,
        partition.scheme,
        partition.fold,
    )
    estimator = build_estimator(model_name, model_config, seed=fit_seed)
    train_x = features[partition.train_indices]
    train_y = targets[partition.train_indices]
    calibration_x = features[partition.calibration_indices]
    calibration_y = targets[partition.calibration_indices]
    test_x = features[partition.test_indices]
    test_y = targets[partition.test_indices]
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always")
        estimator.fit(train_x, train_y)
    calibration_prediction = np.asarray(estimator.predict(calibration_x), dtype=np.float64)
    test_prediction = np.asarray(estimator.predict(test_x), dtype=np.float64)
    if not np.all(np.isfinite(calibration_prediction)) or not np.all(np.isfinite(test_prediction)):
        raise OracleClassicalError(
            f"{representation}/{model_name}/{endpoint}/{partition.scheme}/"
            f"{partition.fold} produced nonfinite predictions"
        )
    calibration_metrics = regression_metrics(calibration_y, calibration_prediction)
    metrics = regression_metrics(test_y, test_prediction)
    radii = {
        int(round(coverage * 100)): conformal_radius(
            np.abs(calibration_y - calibration_prediction), coverage
        )
        for coverage in coverages
    }
    metric_row: dict[str, Any] = {
        "representation": representation,
        "model": model_name,
        "endpoint": endpoint,
        "scheme": partition.scheme,
        "fold": partition.fold,
        "fit_seed": fit_seed,
        "train_rows": len(partition.train_indices),
        "calibration_rows": len(partition.calibration_indices),
        "test_rows": len(partition.test_indices),
        "calibration_r2": calibration_metrics["r2"],
        "calibration_rmse": calibration_metrics["rmse"],
        "calibration_mae": calibration_metrics["mae"],
        "calibration_pearson_r": calibration_metrics["pearson_r"],
        "calibration_spearman_rho": calibration_metrics["spearman_rho"],
        "test_r2": metrics["r2"],
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "test_pearson_r": metrics["pearson_r"],
        "test_spearman_rho": metrics["spearman_rho"],
        "fit_warning_count": len(fit_warnings),
        "fit_warning_categories": "|".join(
            sorted({warning.category.__name__ for warning in fit_warnings})
        ),
    }
    for percent, radius in sorted(radii.items()):
        coverage_value = float(
            np.mean((test_y >= test_prediction - radius) & (test_y <= test_prediction + radius))
        )
        metric_row[f"conformal_q{percent}"] = radius
        metric_row[f"test_coverage{percent}"] = coverage_value
        metric_row[f"test_mean_interval_width{percent}"] = 2.0 * radius

    prediction_rows: list[dict[str, Any]] = []
    for local_index, row_index in enumerate(partition.test_indices):
        prediction = float(test_prediction[local_index])
        truth = float(test_y[local_index])
        row: dict[str, Any] = {
            "representation": representation,
            "model": model_name,
            "endpoint": endpoint,
            "scheme": partition.scheme,
            "fold": partition.fold,
            "label": labels[row_index],
            "y_true": truth,
            "y_pred": prediction,
            "absolute_error": abs(truth - prediction),
        }
        prediction_rows.append(row)
    return metric_row, prediction_rows


def _finite_mean(values: Sequence[Any], label: str) -> float:
    numeric = np.asarray([float(value) for value in values], dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        raise OracleClassicalError(f"cannot aggregate nonfinite values for {label}")
    return float(np.mean(finite))


def aggregate_selection_metrics(
    metric_rows: Sequence[Mapping[str, Any]],
    selection_schemes: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Aggregate folds, then equally weight every eligible evaluation scheme."""

    eligible = set(selection_schemes)
    by_scheme: defaultdict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        scheme = str(row["scheme"])
        if scheme in eligible:
            key = (
                str(row["endpoint"]),
                str(row["representation"]),
                str(row["model"]),
                scheme,
            )
            by_scheme[key].append(row)
    scheme_rows: list[dict[str, Any]] = []
    for (endpoint, representation, model, scheme), rows in sorted(by_scheme.items()):
        scheme_rows.append(
            {
                "endpoint": endpoint,
                "representation": representation,
                "model": model,
                "scheme": scheme,
                "folds": len(rows),
                "test_r2": _finite_mean([row["test_r2"] for row in rows], "test_r2"),
                "test_rmse": _finite_mean([row["test_rmse"] for row in rows], "test_rmse"),
                "test_mae": _finite_mean([row["test_mae"] for row in rows], "test_mae"),
                "test_pearson_r": _finite_mean(
                    [row["test_pearson_r"] for row in rows], "test_pearson_r"
                ),
                "test_spearman_rho": _finite_mean(
                    [row["test_spearman_rho"] for row in rows], "test_spearman_rho"
                ),
                "absolute_90pct_coverage_gap": _finite_mean(
                    [abs(float(row["test_coverage90"]) - 0.9) for row in rows],
                    "absolute_90pct_coverage_gap",
                ),
                "mean_interval_width90": _finite_mean(
                    [row["test_mean_interval_width90"] for row in rows],
                    "mean_interval_width90",
                ),
            }
        )
    by_candidate: defaultdict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scheme_rows:
        by_candidate[(str(row["endpoint"]), str(row["representation"]), str(row["model"]))].append(
            row
        )
    aggregate_rows: list[dict[str, Any]] = []
    for (endpoint, representation, model), rows in sorted(by_candidate.items()):
        observed_schemes = {str(row["scheme"]) for row in rows}
        if observed_schemes != eligible:
            raise OracleClassicalError(
                f"{endpoint}/{representation}/{model} has selection schemes "
                f"{sorted(observed_schemes)}, expected {sorted(eligible)}"
            )
        aggregate_rows.append(
            {
                "endpoint": endpoint,
                "representation": representation,
                "model": model,
                "eligible_schemes": len(rows),
                "mean_test_r2": _finite_mean([row["test_r2"] for row in rows], "mean_test_r2"),
                "mean_test_rmse": _finite_mean(
                    [row["test_rmse"] for row in rows], "mean_test_rmse"
                ),
                "mean_test_mae": _finite_mean([row["test_mae"] for row in rows], "mean_test_mae"),
                "mean_test_pearson_r": _finite_mean(
                    [row["test_pearson_r"] for row in rows], "mean_test_pearson_r"
                ),
                "mean_test_spearman_rho": _finite_mean(
                    [row["test_spearman_rho"] for row in rows],
                    "mean_test_spearman_rho",
                ),
                "mean_absolute_90pct_coverage_gap": _finite_mean(
                    [row["absolute_90pct_coverage_gap"] for row in rows],
                    "mean_absolute_90pct_coverage_gap",
                ),
                "mean_interval_width90": _finite_mean(
                    [row["mean_interval_width90"] for row in rows],
                    "mean_interval_width90",
                ),
            }
        )
    leaders: dict[str, dict[str, Any]] = {}
    endpoints = sorted({str(row["endpoint"]) for row in aggregate_rows})
    for endpoint in endpoints:
        candidates = [row for row in aggregate_rows if row["endpoint"] == endpoint]
        leader = min(
            candidates,
            key=lambda row: (
                -float(row["mean_test_r2"]),
                float(row["mean_test_rmse"]),
                float(row["mean_absolute_90pct_coverage_gap"]),
                str(row["representation"]),
                str(row["model"]),
            ),
        )
        leaders[endpoint] = dict(leader)
    return aggregate_rows, leaders


def build_applicability_rows(
    curated_labels: Sequence[str],
    curated_smiles: Sequence[str],
    virtual_frame: pd.DataFrame,
    applicability_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Calculate maximum structural proximity for every unlabeled virtual candidate."""

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(applicability_config["radius"]),
        fpSize=int(applicability_config["size"]),
        includeChirality=bool(applicability_config["include_chirality"]),
    )
    curated_fingerprints = [
        generator.GetFingerprint(_parse_molecule(smiles, f"curated {label}"))
        for label, smiles in zip(curated_labels, curated_smiles, strict=True)
    ]
    rows: list[dict[str, Any]] = []
    constitutional_smiles: list[str] = []
    for source_row in virtual_frame.itertuples(index=False):
        source_index = int(source_row.source_row_index)
        model_smiles = _canonical_model_smiles(
            str(source_row.canonical_isomeric_smiles),
            f"virtual candidate row {source_index}",
        )
        constitutional_smiles.append(model_smiles)
        fingerprint = generator.GetFingerprint(
            _parse_molecule(model_smiles, f"virtual candidate row {source_index}")
        )
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, curated_fingerprints)
        nearest_index = int(np.argmax(similarities))
        rows.append(
            {
                "source_row_index": source_index,
                "canonical_model_smiles": model_smiles,
                "nearest_curated_label": curated_labels[nearest_index],
                "max_binary_morgan_tanimoto": float(similarities[nearest_index]),
            }
        )
    values = np.asarray([row["max_binary_morgan_tanimoto"] for row in rows], dtype=np.float64)
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0000000001]
    counts, _ = np.histogram(values, bins=bins)
    summary = {
        "records": len(rows),
        "unique_constitutional_model_graphs": len(set(constitutional_smiles)),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
        "histogram": [
            {
                "lower_inclusive": bins[index],
                "upper_bound": min(bins[index + 1], 1.0),
                "upper_inclusive": index == len(counts) - 1,
                "rows": int(counts[index]),
            }
            for index in range(len(counts))
        ],
        "interpretation": str(applicability_config["interpretation"]),
    }
    return rows, summary


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, f".{SERIALIZED_FLOAT_SIGNIFICANT_DIGITS}g")
    return value


def _canonicalize_result_numbers(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OracleClassicalError("result JSON cannot contain nonfinite numbers")
        return float(format(value, f".{SERIALIZED_FLOAT_SIGNIFICANT_DIGITS}g"))
    if isinstance(value, dict):
        return {key: _canonicalize_result_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize_result_numbers(item) for item in value]
    return value


def _render_csv_gzip(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
    sort_key: Any,
) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=sort_key):
        writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})
    return gzip.compress(text.getvalue().encode(), compresslevel=9, mtime=0)


def _artifact_record(payload: bytes) -> dict[str, Any]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def run_classical_oracle_matrix(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Run and atomically persist the complete M0-07 classical oracle lane."""

    config = load_classical_config(config_path)
    config_record = {
        "path": _portable(config_path, repo_root),
        "sha256": sha256_file(config_path),
        "bytes": config_path.stat().st_size,
    }
    inputs: dict[str, dict[str, Any]] = {}
    input_paths: dict[str, Path] = {}
    for name, specification in config["inputs"].items():
        if not isinstance(specification, dict) or not isinstance(specification.get("path"), str):
            raise OracleClassicalError(f"input {name!r} is malformed")
        path = repo_root / specification["path"]
        inputs[name] = _verify_hash(path, specification.get("sha256"), f"input {name}", repo_root)
        input_paths[name] = path

    curated = _read_csv(
        input_paths["curated_oracle_data"],
        ("label", "model_smiles", "expt_Hela", "expt_Raw"),
        "curated oracle data",
    )
    virtual = _read_csv(
        input_paths["agile_virtual_candidate_library"],
        ("source_row_index", "canonical_isomeric_smiles"),
        "AGILE virtual candidate library",
    )
    assignments = _read_csv(
        input_paths["oracle_split_assignments"],
        ("scheme", "fold", "label", "stage", "group_id"),
        "oracle split assignments",
    )
    split_manifest = _load_json(input_paths["oracle_split_manifest"], "oracle split manifest")
    expected = config["expected"]
    observed_counts = {
        "curated_records": len(curated),
        "virtual_candidate_records": len(virtual),
        "assignment_rows": len(assignments),
        "evaluation_schemes": int(assignments["scheme"].nunique()),
    }
    expected_counts = {key: int(expected[key]) for key in observed_counts}
    if observed_counts != expected_counts:
        raise OracleClassicalError(
            f"classical input counts differ from frozen expectations: "
            f"{observed_counts} versus {expected_counts}"
        )
    selection_schemes = split_manifest.get("evaluation_contract", {}).get(
        "selection_eligible_schemes"
    )
    if not isinstance(selection_schemes, list) or not all(
        isinstance(value, str) for value in selection_schemes
    ):
        raise OracleClassicalError("split manifest lacks selection-eligible schemes")
    if "lantern_random" in selection_schemes:
        raise OracleClassicalError("split manifest incorrectly selects on the random split")

    bundle = build_feature_bundle(curated, config["representations"])
    partitions = load_partitions(assignments, bundle.labels)
    if len(partitions) != 32:
        raise OracleClassicalError(f"expected 32 evaluation partitions, found {len(partitions)}")
    endpoints = [str(value) for value in config["evaluation"]["endpoints"]]
    endpoint_values: dict[str, np.ndarray] = {}
    for endpoint in endpoints:
        try:
            values = curated[endpoint].to_numpy(dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise OracleClassicalError(f"endpoint {endpoint!r} is unavailable or invalid") from exc
        if not np.all(np.isfinite(values)):
            raise OracleClassicalError(f"endpoint {endpoint!r} contains nonfinite values")
        endpoint_values[endpoint] = values

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    global_seed = int(config["randomness"]["seed"])
    coverages = [float(value) for value in config["evaluation"]["conformal_coverages"]]
    for representation in sorted(bundle.matrices):
        features = bundle.matrices[representation]
        for model_name in sorted(config["models"]):
            model_config = config["models"][model_name]
            for endpoint in endpoints:
                for partition in partitions:
                    metric_row, fitted_predictions = evaluate_partition(
                        features,
                        endpoint_values[endpoint],
                        bundle.labels,
                        partition,
                        representation=representation,
                        model_name=model_name,
                        endpoint=endpoint,
                        model_config=model_config,
                        global_seed=global_seed,
                        coverages=coverages,
                    )
                    metric_rows.append(metric_row)
                    prediction_rows.extend(fitted_predictions)
    expected_fits = len(bundle.matrices) * len(config["models"]) * len(endpoints) * len(partitions)
    if len(metric_rows) != expected_fits:
        raise OracleClassicalError(
            f"classical matrix produced {len(metric_rows)} fits, expected {expected_fits}"
        )
    aggregate_rows, lane_leaders = aggregate_selection_metrics(metric_rows, selection_schemes)
    applicability_rows, applicability_summary = build_applicability_rows(
        bundle.labels,
        bundle.model_smiles,
        virtual,
        config["applicability"],
    )

    metric_payload = _render_csv_gzip(
        metric_rows,
        METRIC_FIELDS,
        sort_key=lambda row: (
            str(row["endpoint"]),
            str(row["representation"]),
            str(row["model"]),
            str(row["scheme"]),
            int(row["fold"]),
        ),
    )
    prediction_payload = _render_csv_gzip(
        prediction_rows,
        PREDICTION_FIELDS,
        sort_key=lambda row: (
            str(row["endpoint"]),
            str(row["representation"]),
            str(row["model"]),
            str(row["scheme"]),
            int(row["fold"]),
            str(row["label"]),
        ),
    )
    applicability_payload = _render_csv_gzip(
        applicability_rows,
        APPLICABILITY_FIELDS,
        sort_key=lambda row: int(row["source_row_index"]),
    )
    artifacts = {
        "oracle_classical_metrics.csv.gz": _artifact_record(metric_payload),
        "oracle_classical_predictions.csv.gz": _artifact_record(prediction_payload),
        "oracle_classical_applicability.csv.gz": _artifact_record(applicability_payload),
    }
    feature_manifest = {
        representation: {
            "rows": int(matrix.shape[0]),
            "columns": int(matrix.shape[1]),
            "sha256": _array_sha256(matrix, bundle.feature_names[representation]),
            "feature_names": list(bundle.feature_names[representation]),
        }
        for representation, matrix in sorted(bundle.matrices.items())
    }
    warning_total = sum(int(row["fit_warning_count"]) for row in metric_rows)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-07",
        "status": "classical_lane_complete_oracle_not_frozen",
        "generated_utc": config["generated_utc"],
        "configuration": config_record,
        "inputs": inputs,
        "randomness": config["randomness"],
        "summary": {
            **observed_counts,
            "partitions": len(partitions),
            "representations": len(bundle.matrices),
            "models": len(config["models"]),
            "endpoints": len(endpoints),
            "model_fits": len(metric_rows),
            "test_prediction_rows": len(prediction_rows),
            "fit_warnings": warning_total,
        },
        "features": feature_manifest,
        "descriptor_audit": {
            "descriptor_count": len(bundle.feature_names["rdkit_expert"]),
            "calculation_failure_counts": bundle.descriptor_failures,
            "nonfinite_value_counts": bundle.descriptor_nonfinite,
            "imputation_policy": "median fit on each training partition only",
        },
        "evaluation_contract": {
            "selection_eligible_schemes": selection_schemes,
            "random_split_role": "reproduction diagnostic only",
            "preprocessing_fit_scope": "training rows only",
            "calibration_use": "post-fit split-conformal absolute residuals only",
            "selection_weighting": config["evaluation"]["selection_weighting"],
            "metrics": config["evaluation"]["metrics"],
            "conformal_coverages": coverages,
        },
        "selection_aggregate": aggregate_rows,
        "classical_lane_leaders": lane_leaders,
        "applicability": applicability_summary,
        "lantern_source_audit": {
            "pinned_commit": "11240f29ef92323649ae60d177b21df77e2d428b",
            "finding": (
                "LANTERN pipeline/preprocess.py fits MinMaxScaler to all features and labels "
                "before applying its persisted split. FORGE does not reproduce that leakage in "
                "selection-eligible evaluation."
            ),
            "forge_random_diagnostic": (
                "uses LANTERN's exact persisted random assignments with train-only preprocessing"
            ),
        },
        "policy": config["policy"],
        "decision": {
            "oracle_model_frozen": False,
            "classical_lane_complete": True,
            "remaining_required_lanes": [
                "molecular_graph",
                "region_aware_graph",
                "lipid_pretrained_encoder",
            ],
            "reason": (
                "The predeclared full representation matrix and calibration comparison are not "
                "complete."
            ),
        },
        "artifacts": artifacts,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
    }
    try:
        import xgboost

        result["software"]["xgboost"] = xgboost.__version__
    except ImportError:
        result["software"]["xgboost"] = None
    result = _canonicalize_result_numbers(result)
    result_payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.classical.", dir=output_dir.parent))
    try:
        (staging / "oracle_classical_metrics.csv.gz").write_bytes(metric_payload)
        (staging / "oracle_classical_predictions.csv.gz").write_bytes(prediction_payload)
        (staging / "oracle_classical_applicability.csv.gz").write_bytes(applicability_payload)
        (staging / "oracle_classical_result.json").write_bytes(result_payload)
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(staging.iterdir()):
            os.replace(path, output_dir / path.name)
    finally:
        staging.rmdir()
    return result
