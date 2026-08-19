"""Reproduce the released LANTERN AGILE checkpoint as an audit-only diagnostic.

The upstream pipeline fits one MinMaxScaler to every feature and the HeLa
label before applying its persisted split. This module reproduces that behavior
exactly and labels it as test-information leakage. Nothing produced here is
eligible for FORGE oracle selection.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import platform
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from forge.potency.agile_reconciliation import sha256_file

CONFIG_SCHEMA_VERSION = "m0_07_lantern_reproduction_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_lantern_reproduction.v1"
SERIALIZED_FLOAT_SIGNIFICANT_DIGITS = 12
PREDICTION_FIELDS = ("stage", "source_row_index", "smiles", "y_true", "y_pred", "absolute_error")


class LanternReproductionError(ValueError):
    """Raised when the LANTERN reproduction violates its frozen contract."""


class _NumpyArrayUnpickler(pickle.Unpickler):
    """Load only the NumPy globals present in the pinned feature dictionaries."""

    _ALLOWED = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"prohibited pickle global: {module}.{name}")
        return super().find_class(module, name)


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise LanternReproductionError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LanternReproductionError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LanternReproductionError(f"{label} must be a JSON object")
    return payload


def load_lantern_config(path: Path) -> dict[str, Any]:
    """Load and validate the source-faithful diagnostic configuration."""

    config = _load_json(path, "LANTERN reproduction config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise LanternReproductionError(
            f"unsupported LANTERN config schema: {config.get('schema_version')!r}"
        )
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise LanternReproductionError("LANTERN policy must be an object")
    required_policy = {
        "role": "released_checkpoint_reproduction_diagnostic_only",
        "selection_eligible": False,
        "calibration_claim_allowed": False,
        "untrusted_pickle_globals_allowlisted": True,
        "oracle_model_frozen": False,
    }
    for field, expected in required_policy.items():
        if policy.get(field) != expected:
            raise LanternReproductionError(
                f"policy {field!r} must be {expected!r}, found {policy.get(field)!r}"
            )
    preprocessing = config.get("preprocessing", {})
    if preprocessing.get("fit_scope") != "all_1100_features_and_hela_labels_before_split":
        raise LanternReproductionError("the reproduction must declare full-data scaling")
    if preprocessing.get("leakage_status") != "test_information_leakage":
        raise LanternReproductionError("the reproduction must label full-data scaling as leakage")
    randomness = config.get("randomness", {})
    if randomness.get("training_performed") is not False:
        raise LanternReproductionError("released-checkpoint reproduction must not train")
    if (
        randomness.get("numeric_serialization_significant_digits")
        != SERIALIZED_FLOAT_SIGNIFICANT_DIGITS
    ):
        raise LanternReproductionError("LANTERN outputs must use 12 significant digits")
    return config


def _verify_input(
    path: Path,
    expected_sha256: Any,
    label: str,
    repo_root: Path,
) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise LanternReproductionError(f"{label} expected SHA256 must contain 64 characters")
    if not path.is_file():
        raise LanternReproductionError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise LanternReproductionError(
            f"{label} hash mismatch: expected {expected_sha256}, observed {observed}"
        )
    return {
        "path": _portable(path, repo_root),
        "sha256": observed,
        "bytes": path.stat().st_size,
    }


def load_numpy_feature_dictionary(
    path: Path,
    *,
    expected_rows: int,
    expected_columns: int,
    label: str,
) -> dict[str, np.ndarray]:
    """Load a hash-verified NumPy feature dictionary with an allowlisted unpickler."""

    try:
        with path.open("rb") as handle:
            payload = _NumpyArrayUnpickler(handle).load()
    except (OSError, pickle.UnpicklingError, ValueError, TypeError) as exc:
        raise LanternReproductionError(f"cannot load {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict) or len(payload) != expected_rows:
        raise LanternReproductionError(
            f"{label} must contain {expected_rows} records, found "
            f"{len(payload) if isinstance(payload, dict) else type(payload).__name__}"
        )
    validated: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise LanternReproductionError(f"{label} contains a non-string molecular key")
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (expected_columns,):
            raise LanternReproductionError(
                f"{label} feature {key!r} has shape {array.shape}, expected {(expected_columns,)}"
            )
        if not np.all(np.isfinite(array)):
            raise LanternReproductionError(f"{label} feature {key!r} contains nonfinite values")
        validated[key] = array
    return validated


def _load_split(path: Path, expected_rows: int) -> dict[str, np.ndarray]:
    try:
        raw = np.load(path, allow_pickle=True)
    except (OSError, ValueError) as exc:
        raise LanternReproductionError(f"cannot load LANTERN split: {path}: {exc}") from exc
    if raw.shape != (3,) or raw.dtype != object:
        raise LanternReproductionError(
            f"LANTERN split must be an object array with shape (3,), found {raw.shape}/{raw.dtype}"
        )
    names = ("train", "validation", "test")
    stages: dict[str, np.ndarray] = {}
    for name, values in zip(names, raw, strict=True):
        indices = np.asarray(values, dtype=np.int64)
        if indices.ndim != 1 or len(indices) != len(set(indices.tolist())):
            raise LanternReproductionError(f"LANTERN {name} indices are malformed")
        if np.any(indices < 0) or np.any(indices >= expected_rows):
            raise LanternReproductionError(f"LANTERN {name} index is outside the data range")
        stages[name] = indices
    flattened = np.concatenate(tuple(stages.values()))
    if len(flattened) != expected_rows or set(flattened.tolist()) != set(range(expected_rows)):
        raise LanternReproductionError("LANTERN split does not cover every source row exactly once")
    return stages


def _build_network(torch: Any, input_count: int, hidden_dimensions: Sequence[int]) -> Any:
    dimensions = [input_count, *[int(value) for value in hidden_dimensions], 1]
    layers: list[Any] = []
    for index, (source, target) in enumerate(zip(dimensions[:-1], dimensions[1:], strict=True)):
        layers.append(torch.nn.Linear(source, target))
        if index < len(dimensions) - 2:
            layers.append(torch.nn.ReLU())

    class FeedforwardRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(*layers)

        def forward(self, features: Any) -> Any:
            return self.network(features)

    return FeedforwardRegressor()


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise LanternReproductionError(
            "truth and prediction must be aligned one-dimensional arrays"
        )
    pearson = float(pearsonr(truth, prediction).statistic) if len(truth) > 1 else math.nan
    spearman = float(spearmanr(truth, prediction).statistic) if len(truth) > 1 else math.nan
    return {
        "rows": int(len(truth)),
        "r2": float(r2_score(truth, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(truth, prediction))),
        "mae": float(mean_absolute_error(truth, prediction)),
        "pearson_r": pearson,
        "spearman_rho": spearman,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, f".{SERIALIZED_FLOAT_SIGNIFICANT_DIGITS}g")
    return value


def _render_prediction_gzip(rows: Sequence[Mapping[str, Any]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=PREDICTION_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda item: (str(item["stage"]), int(item["source_row_index"]))):
        writer.writerow({field: _csv_value(row[field]) for field in PREDICTION_FIELDS})
    return gzip.compress(text.getvalue().encode(), compresslevel=9, mtime=0)


def _canonicalize_numbers(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LanternReproductionError("result JSON cannot contain nonfinite values")
        return float(format(value, f".{SERIALIZED_FLOAT_SIGNIFICANT_DIGITS}g"))
    if isinstance(value, dict):
        return {key: _canonicalize_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize_numbers(item) for item in value]
    return value


def run_lantern_reproduction(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Evaluate the released LANTERN checkpoint under its source preprocessing."""

    try:
        import torch
    except ImportError as exc:
        raise LanternReproductionError(
            "PyTorch is required; install the project oracle optional dependency"
        ) from exc

    config = load_lantern_config(config_path)
    input_paths: dict[str, Path] = {}
    input_records: dict[str, dict[str, Any]] = {}
    for name, specification in config["inputs"].items():
        path = repo_root / str(specification["path"])
        input_paths[name] = path
        input_records[name] = _verify_input(
            path,
            specification.get("sha256"),
            f"input {name}",
            repo_root,
        )

    try:
        frame = pd.read_csv(input_paths["lantern_curated_hela"])
    except (OSError, ValueError) as exc:
        raise LanternReproductionError(f"cannot read LANTERN AGILE data: {exc}") from exc
    if list(frame.columns) != ["SMILES", "Target"]:
        raise LanternReproductionError(
            f"LANTERN AGILE columns differ from the pinned source: {list(frame.columns)}"
        )
    expected = config["expected"]
    rows = int(expected["records"])
    if len(frame) != rows or frame["SMILES"].nunique() != rows:
        raise LanternReproductionError("LANTERN AGILE data must contain 1,100 unique structures")
    labels = frame["Target"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(labels)):
        raise LanternReproductionError("LANTERN HeLa labels contain nonfinite values")

    circular = load_numpy_feature_dictionary(
        input_paths["lantern_circular_features"],
        expected_rows=rows,
        expected_columns=int(expected["circular_feature_count"]),
        label="LANTERN circular features",
    )
    expert = load_numpy_feature_dictionary(
        input_paths["lantern_expert_features"],
        expected_rows=rows,
        expected_columns=int(expected["expert_feature_count"]),
        label="LANTERN expert features",
    )
    smiles = [str(value) for value in frame["SMILES"]]
    missing = [value for value in smiles if value not in circular or value not in expert]
    if missing:
        raise LanternReproductionError(f"LANTERN features are missing structures: {missing[:5]}")
    features = np.vstack(
        [np.concatenate((circular[value], expert[value])) for value in smiles]
    ).astype(np.float64, copy=False)
    if features.shape != (rows, int(expected["combined_feature_count"])):
        raise LanternReproductionError(
            f"combined LANTERN features have shape {features.shape}, "
            f"expected {(rows, int(expected['combined_feature_count']))}"
        )

    combined = np.hstack((features, labels[:, None]))
    scaler = MinMaxScaler(feature_range=tuple(config["preprocessing"]["feature_range"]))
    scaled = scaler.fit_transform(combined)
    scaled_features = scaled[:, :-1]
    split = _load_split(input_paths["lantern_random_split"], rows)
    observed_stage_counts = {name: len(indices) for name, indices in split.items()}
    expected_stage_counts = {
        "train": int(expected["train_records"]),
        "validation": int(expected["validation_records"]),
        "test": int(expected["test_records"]),
    }
    if observed_stage_counts != expected_stage_counts:
        raise LanternReproductionError(
            f"LANTERN split counts differ: {observed_stage_counts} versus {expected_stage_counts}"
        )

    torch.set_num_threads(int(config["randomness"]["torch_threads"]))
    torch.use_deterministic_algorithms(bool(config["randomness"]["deterministic_algorithms"]))
    network = _build_network(
        torch,
        input_count=features.shape[1],
        hidden_dimensions=config["model"]["hidden_dimensions"],
    )
    try:
        state = torch.load(
            input_paths["lantern_released_mlp"],
            map_location="cpu",
            weights_only=True,
        )
        network.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LanternReproductionError(f"cannot load released LANTERN checkpoint: {exc}") from exc
    network.eval()
    with torch.inference_mode():
        scaled_predictions = (
            network(torch.as_tensor(scaled_features, dtype=torch.float32)).detach().cpu().numpy()
        )
    inverse_input = np.hstack((scaled_features, scaled_predictions))
    predictions = scaler.inverse_transform(inverse_input)[:, -1]
    if not np.all(np.isfinite(predictions)):
        raise LanternReproductionError("released LANTERN checkpoint produced nonfinite predictions")

    prediction_rows: list[dict[str, Any]] = []
    stage_metrics: dict[str, dict[str, float]] = {}
    for stage, indices in split.items():
        stage_metrics[stage] = _metrics(labels[indices], predictions[indices])
        for source_index in indices:
            truth = float(labels[source_index])
            prediction = float(predictions[source_index])
            prediction_rows.append(
                {
                    "stage": stage,
                    "source_row_index": int(source_index),
                    "smiles": smiles[source_index],
                    "y_true": truth,
                    "y_pred": prediction,
                    "absolute_error": abs(truth - prediction),
                }
            )

    prediction_payload = _render_prediction_gzip(prediction_rows)
    prediction_record = {
        "path": "results/m0_07/lantern_reproduction_predictions.csv.gz",
        "sha256": hashlib.sha256(prediction_payload).hexdigest(),
        "bytes": len(prediction_payload),
    }
    feature_hash = hashlib.sha256(
        np.asarray(features, dtype="<f8", order="C").tobytes()
    ).hexdigest()
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-07",
        "status": "released_checkpoint_reproduced_diagnostic_only",
        "generated_utc": config["generated_utc"],
        "configuration": {
            "path": _portable(config_path, repo_root),
            "sha256": sha256_file(config_path),
            "bytes": config_path.stat().st_size,
        },
        "inputs": input_records,
        "source": config["source"],
        "reproduction_contract": {
            "feature_dimensions": {
                "circular": int(expected["circular_feature_count"]),
                "expert": int(expected["expert_feature_count"]),
                "combined": features.shape[1],
            },
            "feature_matrix_sha256": feature_hash,
            "architecture": config["model"],
            "preprocessing": config["preprocessing"],
            "split_counts": observed_stage_counts,
            "checkpoint_training_repeated": False,
        },
        "metrics": stage_metrics,
        "artifact": prediction_record,
        "policy": {
            **config["policy"],
            "interpretation": (
                "The released checkpoint score reproduces LANTERN's committed implementation. "
                "Because the source scaler is fitted to all features and labels before splitting, "
                "the score contains test-information leakage and cannot select the FORGE oracle."
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "randomness": config["randomness"],
    }
    result = _canonicalize_numbers(result)
    result_payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lantern.", dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        (temporary_dir / "lantern_reproduction_predictions.csv.gz").write_bytes(prediction_payload)
        (temporary_dir / "lantern_reproduction_result.json").write_bytes(result_payload)
        for name in (
            "lantern_reproduction_predictions.csv.gz",
            "lantern_reproduction_result.json",
        ):
            os.replace(temporary_dir / name, output_dir / name)
    return result
