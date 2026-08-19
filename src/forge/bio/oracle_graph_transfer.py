"""Leakage-safe transfer evaluation for the label-free R0 D-MPNN encoder.

The transfer lane compares a frozen-encoder linear probe with end-to-end
fine-tuning. Epoch selection and target scaling use only the training stage of
each frozen oracle partition. Calibration and test targets are accessed only
after the refitted model has been hashed.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import multiprocessing
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forge.bio.oracle_classical import (
    aggregate_selection_metrics,
    conformal_radius,
    regression_metrics,
)
from forge.bio.oracle_graph import (
    DMPNNEncoder,
    GraphFeatureVocabulary,
    OracleGraphRecord,
    WholeGraphRegressor,
    collate_oracle_records,
    load_oracle_graph_records,
)
from forge.bio.oracle_graph_jobs import build_graph_job_rows
from forge.bio.oracle_graph_pretraining import RESULT_SCHEMA_VERSION as PRETRAINING_SCHEMA_VERSION
from forge.bio.oracle_graph_profile import _budget_batches, _model_digest
from forge.core.hashing import sha256_file as _sha256_file
from forge.core.io import atomic_write as _atomic_write
from forge.core.io import pretty_json_bytes as _stable_json

CONFIG_SCHEMA_VERSION = "m0_07_oracle_graph_transfer_config.v1"
FIT_SCHEMA_VERSION = "m0_07_oracle_graph_transfer_fit.v1"
RESULT_SCHEMA_VERSION = "m0_07_oracle_graph_transfer_matrix.v1"
TRANSFER_VARIANTS = (
    "r0_pretrained_frozen_linear",
    "r0_pretrained_finetuned_dmpnn",
)
ENDPOINTS = ("expt_Hela", "expt_Raw")
REQUIRED_SELECTION_SCHEMES = frozenset(
    {
        "lantern_scaffold_balanced",
        "held_head_5fold",
        "held_aldehyde_5fold",
        "held_isocyanide_5fold",
        "held_head_aldehyde_pair_5fold",
        "held_head_isocyanide_pair_5fold",
        "held_aldehyde_isocyanide_pair_5fold",
    }
)
REQUIRED_DATA_POLICY = {
    "biological_labels_used": False,
    "component_annotations_used": False,
    "provenance_fields_used": False,
    "virtual_candidates_used": False,
    "stereochemistry_used": False,
    "graph_truncation_allowed": False,
}
METRIC_FIELDS = (
    "representation",
    "model",
    "endpoint",
    "scheme",
    "fold",
    "train_rows",
    "calibration_rows",
    "test_rows",
    "seed_count",
    "mean_selected_epochs",
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
)
PREDICTION_FIELDS = (
    "representation",
    "model",
    "endpoint",
    "scheme",
    "fold",
    "label",
    "y_true",
    "ensemble_y_pred",
    "ensemble_standard_deviation",
    "absolute_error",
)


class OracleGraphTransferError(ValueError):
    """Raised when pretrained-oracle transfer violates its frozen contract."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise OracleGraphTransferError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OracleGraphTransferError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OracleGraphTransferError(f"{label} must contain a JSON object")
    return payload


def _verify_input(
    repo_root: Path,
    specification: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    relative = specification.get("path")
    expected = specification.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise OracleGraphTransferError(f"{label} input specification is incomplete")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise OracleGraphTransferError(f"{label} path must be repository-relative")
    path = (repo_root / relative_path).resolve()
    if not path.is_relative_to(repo_root.resolve()) or not path.is_file():
        raise OracleGraphTransferError(f"{label} not found inside repository: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise OracleGraphTransferError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return {"path": relative, "sha256": observed, "bytes": path.stat().st_size}


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    try:
        with opener(path, "rt", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except FileNotFoundError as exc:
        raise OracleGraphTransferError(f"input not found: {path}") from exc


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(item) for item in value]
    return value


def _state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _feature_vocabulary_dict(
    vocabulary: GraphFeatureVocabulary,
) -> dict[str, list[str]]:
    return vocabulary.to_dict()


def load_full_pretraining_checkpoint(
    path: Path,
    *,
    vocabulary: GraphFeatureVocabulary,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and validate a label-free full-pretraining checkpoint on CPU."""

    if expected_sha256 is not None and _sha256_file(path) != expected_sha256:
        raise OracleGraphTransferError("pretraining checkpoint hash changed")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (FileNotFoundError, RuntimeError, TypeError) as exc:
        raise OracleGraphTransferError(f"could not load pretraining checkpoint: {path}") from exc
    if not isinstance(payload, dict):
        raise OracleGraphTransferError("pretraining checkpoint must contain a mapping")
    if payload.get("schema_version") != PRETRAINING_SCHEMA_VERSION or payload.get("mode") != "full":
        raise OracleGraphTransferError(
            "only the completed full pretraining checkpoint is authorized downstream"
        )
    if payload.get("data_policy") != REQUIRED_DATA_POLICY:
        raise OracleGraphTransferError("pretraining checkpoint data policy changed")
    if payload.get("feature_vocabulary") != _feature_vocabulary_dict(vocabulary):
        raise OracleGraphTransferError("pretraining checkpoint feature vocabulary changed")
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict) or architecture.get("encoder") != "sparse_dmpnn":
        raise OracleGraphTransferError("pretraining checkpoint is not the sparse D-MPNN")
    for field in ("hidden_dim", "depth"):
        if not isinstance(architecture.get(field), int) or int(architecture[field]) < 1:
            raise OracleGraphTransferError(f"pretraining architecture lacks {field}")
    dropout = architecture.get("dropout")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise OracleGraphTransferError("pretraining architecture has invalid dropout")
    state_dict = payload.get("encoder_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise OracleGraphTransferError("pretraining checkpoint lacks encoder weights")
    encoder = DMPNNEncoder(
        vocabulary.atom_feature_dim,
        vocabulary.bond_feature_dim,
        int(architecture["hidden_dim"]),
        int(architecture["depth"]),
        float(architecture["dropout"]),
    )
    try:
        encoder.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise OracleGraphTransferError("pretraining encoder weights are incompatible") from exc
    if any(not torch.isfinite(value).all() for value in encoder.state_dict().values()):
        raise OracleGraphTransferError("pretraining encoder contains nonfinite weights")
    return payload


def validate_pretraining_result(
    result: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
) -> None:
    """Require an explicitly downstream-authorized, label-free full run."""

    checkpoint = result.get("checkpoint")
    decision = result.get("decision")
    if (
        result.get("schema_version") != PRETRAINING_SCHEMA_VERSION
        or result.get("status") != "completed_label_free_r0_pretraining"
        or result.get("mode") != "full"
        or result.get("data_policy") != REQUIRED_DATA_POLICY
        or not isinstance(checkpoint, dict)
        or checkpoint.get("sha256") != checkpoint_sha256
        or checkpoint.get("downstream_use_authorized") is not True
        or not isinstance(decision, dict)
        or decision.get("full_checkpoint_selected_without_agile_labels") is not True
    ):
        raise OracleGraphTransferError(
            "pretraining result does not authorize downstream encoder transfer"
        )


def _fit_seed(repeat_seed: int, row: Mapping[str, Any]) -> int:
    key = "\x1f".join(
        (
            "pretrained-transfer",
            str(repeat_seed),
            str(row["architecture"]),
            str(row["endpoint"]),
            str(row["scheme"]),
            str(row["fold"]),
        )
    )
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**31 - 1) or 1


def _inner_split(
    labels: Sequence[str],
    *,
    scheme: str,
    fold: int,
    fraction: float,
) -> tuple[list[int], list[int]]:
    if not 0.0 < fraction < 0.5:
        raise OracleGraphTransferError("inner-validation fraction is invalid")
    order = sorted(
        range(len(labels)),
        key=lambda index: hashlib.sha256(
            f"pretrained-inner\x1f{scheme}\x1f{fold}\x1f{labels[index]}".encode()
        ).hexdigest(),
    )
    count = max(1, round(len(labels) * fraction))
    validation = set(order[:count])
    fit = [index for index in range(len(labels)) if index not in validation]
    if not fit:
        raise OracleGraphTransferError("inner-validation split exhausted training rows")
    return fit, sorted(validation)


def _target_scaler(
    records: Sequence[OracleGraphRecord],
    indices: Sequence[int],
    endpoint_index: int,
) -> tuple[float, float]:
    values = np.asarray(
        [float(records[index].targets[endpoint_index]) for index in indices],
        dtype=np.float64,
    )
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0.0:
        raise OracleGraphTransferError("training-derived target scaler is invalid")
    return mean, scale


def build_transfer_model(
    variant: str,
    *,
    vocabulary: GraphFeatureVocabulary,
    checkpoint: Mapping[str, Any],
    seed: int,
) -> WholeGraphRegressor:
    """Initialize a whole-product regressor from the fixed R0 encoder."""

    if variant not in TRANSFER_VARIANTS:
        raise OracleGraphTransferError(f"unsupported transfer variant: {variant}")
    architecture = checkpoint["architecture"]
    torch.manual_seed(seed)
    encoder = DMPNNEncoder(
        vocabulary.atom_feature_dim,
        vocabulary.bond_feature_dim,
        int(architecture["hidden_dim"]),
        int(architecture["depth"]),
        float(architecture["dropout"]),
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    model = WholeGraphRegressor(encoder, int(architecture["hidden_dim"]), outputs=1)
    if variant == "r0_pretrained_frozen_linear":
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    return model


def _optimizer(
    model: WholeGraphRegressor,
    *,
    variant: str,
    model_config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    head_parameters = [
        parameter for parameter in model.head.parameters() if parameter.requires_grad
    ]
    groups: list[dict[str, Any]] = [
        {
            "params": head_parameters,
            "lr": float(model_config["head_learning_rate"]),
        }
    ]
    if variant == "r0_pretrained_finetuned_dmpnn":
        groups.append(
            {
                "params": [
                    parameter for parameter in model.encoder.parameters() if parameter.requires_grad
                ],
                "lr": float(model_config["encoder_learning_rate"]),
            }
        )
    return torch.optim.AdamW(
        groups,
        weight_decay=float(model_config["weight_decay"]),
    )


def _train_epoch(
    model: WholeGraphRegressor,
    records: Sequence[OracleGraphRecord],
    indices: Sequence[int],
    *,
    endpoint_index: int,
    mean: float,
    scale: float,
    optimizer: torch.optim.Optimizer,
    frozen_encoder: bool,
    model_config: Mapping[str, Any],
    order_seed: int,
) -> float:
    generator = torch.Generator().manual_seed(order_seed)
    order = torch.tensor(list(indices), dtype=torch.long)[
        torch.randperm(len(indices), generator=generator)
    ].tolist()
    batches = _budget_batches(
        records,
        order,
        role_aware=False,
        maximum_graphs=int(model_config["maximum_graphs_per_batch"]),
        maximum_atoms=int(model_config["maximum_atoms_per_batch"]),
        maximum_directed_edges=int(model_config["maximum_directed_edges_per_batch"]),
    )
    model.train()
    if frozen_encoder:
        model.encoder.eval()
    total = 0.0
    count = 0
    for batch_indices in batches:
        batch, _ = collate_oracle_records([records[index] for index in batch_indices])
        target = (batch.targets[:, endpoint_index] - mean) / scale
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch.inputs).squeeze(-1)
        loss = torch.mean((prediction - target) ** 2)
        if not torch.isfinite(loss):
            raise OracleGraphTransferError("nonfinite transfer training loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise OracleGraphTransferError("nonfinite transfer training gradient")
        optimizer.step()
        total += float(loss.detach()) * len(batch_indices)
        count += len(batch_indices)
    return total / count


def _validation_loss(
    model: WholeGraphRegressor,
    records: Sequence[OracleGraphRecord],
    indices: Sequence[int],
    *,
    endpoint_index: int,
    mean: float,
    scale: float,
    model_config: Mapping[str, Any],
) -> float:
    batches = _budget_batches(
        records,
        list(indices),
        role_aware=False,
        maximum_graphs=int(model_config["maximum_graphs_per_batch"]),
        maximum_atoms=int(model_config["maximum_atoms_per_batch"]),
        maximum_directed_edges=int(model_config["maximum_directed_edges_per_batch"]),
    )
    model.eval()
    squared_error = 0.0
    count = 0
    with torch.no_grad():
        for batch_indices in batches:
            batch, _ = collate_oracle_records([records[index] for index in batch_indices])
            target = (batch.targets[:, endpoint_index] - mean) / scale
            prediction = model(batch.inputs).squeeze(-1)
            squared_error += float(torch.sum((prediction - target) ** 2))
            count += len(batch_indices)
    return squared_error / count


def _predict(
    model: WholeGraphRegressor,
    records: Sequence[OracleGraphRecord],
    indices: Sequence[int],
    *,
    mean: float,
    scale: float,
    model_config: Mapping[str, Any],
) -> list[float]:
    batches = _budget_batches(
        records,
        list(indices),
        role_aware=False,
        maximum_graphs=int(model_config["maximum_graphs_per_batch"]),
        maximum_atoms=int(model_config["maximum_atoms_per_batch"]),
        maximum_directed_edges=int(model_config["maximum_directed_edges_per_batch"]),
    )
    model.eval()
    output: list[float] = []
    with torch.no_grad():
        for batch_indices in batches:
            batch, _ = collate_oracle_records([records[index] for index in batch_indices])
            prediction = model(batch.inputs).squeeze(-1)
            output.extend(float(value) for value in (prediction * scale + mean))
    return output


def _stage_indices(
    row: Mapping[str, Any],
    *,
    assignments: Sequence[Mapping[str, str]],
    label_to_index: Mapping[str, int],
) -> dict[str, list[int]]:
    stages: dict[str, list[str]] = defaultdict(list)
    for assignment in assignments:
        if str(assignment["scheme"]) == str(row["scheme"]) and int(assignment["fold"]) == int(
            row["fold"]
        ):
            stages[str(assignment["stage"])].append(str(assignment["label"]))
    if set(stages) != {"train", "calibration", "test"}:
        raise OracleGraphTransferError("transfer split partition is incomplete")
    flattened = [label for stage in stages.values() for label in stage]
    if len(flattened) != len(set(flattened)):
        raise OracleGraphTransferError("transfer split stages overlap")
    try:
        return {
            stage: [label_to_index[label] for label in sorted(labels)]
            for stage, labels in stages.items()
        }
    except KeyError as exc:
        raise OracleGraphTransferError(f"split label is absent from curated data: {exc}") from exc


def fit_transfer_job(
    row: Mapping[str, Any],
    *,
    records: Sequence[OracleGraphRecord],
    label_to_index: Mapping[str, int],
    assignments: Sequence[Mapping[str, str]],
    vocabulary: GraphFeatureVocabulary,
    checkpoint: Mapping[str, Any],
    checkpoint_sha256: str,
    config: Mapping[str, Any],
    config_sha256: str,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Fit one transfer seed before observing calibration or test targets."""

    stage_indices = _stage_indices(
        row,
        assignments=assignments,
        label_to_index=label_to_index,
    )
    endpoint = str(row["endpoint"])
    if endpoint not in ENDPOINTS:
        raise OracleGraphTransferError(f"unsupported transfer endpoint: {endpoint}")
    endpoint_index = ENDPOINTS.index(endpoint)
    variant = str(row["architecture"])
    if variant not in TRANSFER_VARIANTS:
        raise OracleGraphTransferError(f"unsupported transfer variant: {variant}")
    fit_seed = _fit_seed(int(row["seed"]), row)
    train_indices = stage_indices["train"]
    train_labels = [records[index].label for index in train_indices]
    local_fit, local_validation = _inner_split(
        train_labels,
        scheme=str(row["scheme"]),
        fold=int(row["fold"]),
        fraction=float(config["epoch_selection"]["inner_validation_fraction"]),
    )
    inner_fit = [train_indices[index] for index in local_fit]
    inner_validation = [train_indices[index] for index in local_validation]
    selection_mean, selection_scale = _target_scaler(
        records,
        inner_fit,
        endpoint_index,
    )
    frozen_encoder = variant == "r0_pretrained_frozen_linear"
    model = build_transfer_model(
        variant,
        vocabulary=vocabulary,
        checkpoint=checkpoint,
        seed=fit_seed,
    )
    optimizer = _optimizer(model, variant=variant, model_config=config["model"])
    source_encoder_digest = _state_dict_digest(checkpoint["encoder_state_dict"])
    best_epoch = 1
    best_validation = math.inf
    stale = 0
    trace: list[dict[str, Any]] = []
    for epoch in range(1, int(config["epoch_selection"]["maximum_epochs"]) + 1):
        fit_loss = _train_epoch(
            model,
            records,
            inner_fit,
            endpoint_index=endpoint_index,
            mean=selection_mean,
            scale=selection_scale,
            optimizer=optimizer,
            frozen_encoder=frozen_encoder,
            model_config=config["model"],
            order_seed=fit_seed + epoch,
        )
        validation_loss = _validation_loss(
            model,
            records,
            inner_validation,
            endpoint_index=endpoint_index,
            mean=selection_mean,
            scale=selection_scale,
            model_config=config["model"],
        )
        trace.append(
            {
                "epoch": epoch,
                "inner_fit_loss": fit_loss,
                "inner_validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation - float(config["epoch_selection"]["minimum_delta"]):
            best_epoch = epoch
            best_validation = validation_loss
            stale = 0
        else:
            stale += 1
        if stale >= int(config["epoch_selection"]["patience"]):
            break

    refit_mean, refit_scale = _target_scaler(records, train_indices, endpoint_index)
    model = build_transfer_model(
        variant,
        vocabulary=vocabulary,
        checkpoint=checkpoint,
        seed=fit_seed,
    )
    optimizer = _optimizer(model, variant=variant, model_config=config["model"])
    refit_losses = []
    for epoch in range(1, best_epoch + 1):
        refit_losses.append(
            _train_epoch(
                model,
                records,
                train_indices,
                endpoint_index=endpoint_index,
                mean=refit_mean,
                scale=refit_scale,
                optimizer=optimizer,
                frozen_encoder=frozen_encoder,
                model_config=config["model"],
                order_seed=fit_seed + 100_000 + epoch,
            )
        )
    model_digest = _model_digest(model)
    encoder_digest = _model_digest(model.encoder)
    if frozen_encoder and encoder_digest != source_encoder_digest:
        raise OracleGraphTransferError("frozen encoder changed during linear-probe fitting")

    calibration_indices = stage_indices["calibration"]
    test_indices = stage_indices["test"]
    calibration_prediction = _predict(
        model,
        records,
        calibration_indices,
        mean=refit_mean,
        scale=refit_scale,
        model_config=config["model"],
    )
    test_prediction = _predict(
        model,
        records,
        test_indices,
        mean=refit_mean,
        scale=refit_scale,
        model_config=config["model"],
    )
    return {
        "schema_version": FIT_SCHEMA_VERSION,
        "status": "completed",
        "job": dict(row),
        "fit_seed": fit_seed,
        "config_sha256": config_sha256,
        "input_hashes": dict(sorted(input_hashes.items())),
        "pretrained_encoder": {
            "checkpoint_sha256": checkpoint_sha256,
            "source_encoder_state_sha256": source_encoder_digest,
            "frozen": frozen_encoder,
        },
        "epoch_selection": {
            "fit_rows": len(inner_fit),
            "validation_rows": len(inner_validation),
            "target_mean": selection_mean,
            "target_scale": selection_scale,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation,
            "epochs_evaluated": len(trace),
            "trace": trace,
            "calibration_or_test_targets_accessed": False,
        },
        "refit": {
            "train_rows": len(train_indices),
            "target_mean": refit_mean,
            "target_scale": refit_scale,
            "epoch_count": best_epoch,
            "losses": refit_losses,
            "model_sha256_before_calibration_or_test_access": model_digest,
            "encoder_sha256_before_calibration_or_test_access": encoder_digest,
        },
        "calibration": {
            "labels": [records[index].label for index in calibration_indices],
            "truth": [
                float(records[index].targets[endpoint_index]) for index in calibration_indices
            ],
            "prediction": calibration_prediction,
        },
        "test": {
            "labels": [records[index].label for index in test_indices],
            "truth": [float(records[index].targets[endpoint_index]) for index in test_indices],
            "prediction": test_prediction,
        },
        "boundary": {
            "pretraining_used_agile_labels": False,
            "calibration_or_test_used_for_scaling": False,
            "calibration_or_test_used_for_epoch_selection": False,
            "calibration_or_test_used_for_weight_updates": False,
            "calibration_and_test_accessed_after_model_hash": True,
        },
    }


def _validate_config_sections(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION or config.get("seed") != 1729:
        raise OracleGraphTransferError("unsupported transfer config schema or seed")
    matrix = config.get("matrix")
    model = config.get("model")
    selection = config.get("epoch_selection")
    evaluation = config.get("evaluation")
    execution = config.get("execution")
    if not all(
        isinstance(value, dict) for value in (matrix, model, selection, evaluation, execution)
    ):
        raise OracleGraphTransferError("transfer config sections are incomplete")
    if tuple(sorted(matrix.get("variants", ()))) != tuple(sorted(TRANSFER_VARIANTS)):
        raise OracleGraphTransferError("transfer variant matrix changed")
    if tuple(sorted(matrix.get("endpoints", ()))) != tuple(sorted(ENDPOINTS)):
        raise OracleGraphTransferError("transfer endpoint matrix changed")
    seeds = matrix.get("repeat_seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
        raise OracleGraphTransferError("transfer matrix requires three unique seeds")
    if (
        selection.get("calibration_used") is not False
        or selection.get("test_used") is not False
        or selection.get("target_scaling") != "inner_fit_only_then_all_train_on_refit"
    ):
        raise OracleGraphTransferError("transfer epoch-selection leakage boundary changed")
    if (
        evaluation.get("calibrate_after_three_seed_ensemble") is not True
        or evaluation.get("random_split_role") != "diagnostic_only"
        or execution.get("workers") != 4
        or execution.get("torch_threads_per_worker") != 2
        or execution.get("resume_policy") != "validate_completed_output_then_skip"
        or execution.get("device") != "cpu"
    ):
        raise OracleGraphTransferError("transfer execution or evaluation policy changed")
    for field in (
        "head_learning_rate",
        "encoder_learning_rate",
        "weight_decay",
        "maximum_graphs_per_batch",
        "maximum_atoms_per_batch",
        "maximum_directed_edges_per_batch",
    ):
        value = model.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
            raise OracleGraphTransferError(f"invalid transfer model field: {field}")


def _load_worker_inputs(
    config_path: Path,
    repo_root: Path,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, dict[str, Any]],
    list[OracleGraphRecord],
    list[dict[str, str]],
    dict[str, Any],
    GraphFeatureVocabulary,
]:
    config = _load_json(config_path, "graph transfer config")
    _validate_config_sections(config)
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise OracleGraphTransferError("transfer config lacks inputs")
    verified = {
        name: _verify_input(repo_root, specification, name)
        for name, specification in sorted(inputs.items())
        if isinstance(specification, dict)
    }
    required = {
        "curated_oracle_data",
        "graph_corpus_result",
        "oracle_split_assignments",
        "oracle_split_manifest",
        "pretraining_checkpoint",
        "pretraining_result",
    }
    if set(verified) != required:
        raise OracleGraphTransferError("transfer config inputs are incomplete")
    vocabulary = GraphFeatureVocabulary.from_corpus_result(
        repo_root / verified["graph_corpus_result"]["path"]
    )
    checkpoint = load_full_pretraining_checkpoint(
        repo_root / verified["pretraining_checkpoint"]["path"],
        vocabulary=vocabulary,
        expected_sha256=verified["pretraining_checkpoint"]["sha256"],
    )
    pretraining_result = _load_json(
        repo_root / verified["pretraining_result"]["path"],
        "pretraining result",
    )
    validate_pretraining_result(
        pretraining_result,
        checkpoint_sha256=verified["pretraining_checkpoint"]["sha256"],
    )
    if pretraining_result.get("feature_vocabulary") != vocabulary.to_dict():
        raise OracleGraphTransferError("pretraining result vocabulary changed")
    records = load_oracle_graph_records(
        repo_root / verified["curated_oracle_data"]["path"],
        vocabulary,
    )
    if len(records) != int(config["expected"]["curated_records"]):
        raise OracleGraphTransferError("curated graph record count changed")
    assignments = _read_csv(repo_root / verified["oracle_split_assignments"]["path"])
    return (
        config,
        _sha256_file(config_path),
        verified,
        records,
        assignments,
        checkpoint,
        vocabulary,
    )


def _job_rows(
    config: Mapping[str, Any],
    assignments: Sequence[Mapping[str, str]],
    split_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    contract = split_manifest.get("evaluation_contract")
    if not isinstance(contract, dict):
        raise OracleGraphTransferError("split manifest lacks the evaluation contract")
    selection_schemes = contract.get("selection_eligible_schemes")
    if (
        not isinstance(selection_schemes, list)
        or set(selection_schemes) != REQUIRED_SELECTION_SCHEMES
        or contract.get("random_diagnostic_scheme") != "lantern_random"
    ):
        raise OracleGraphTransferError("held-component selection contract changed")
    rows = build_graph_job_rows(
        assignments,
        architectures=[str(value) for value in config["matrix"]["variants"]],
        endpoints=[str(value) for value in config["matrix"]["endpoints"]],
        seeds=[int(value) for value in config["matrix"]["repeat_seeds"]],
        selection_schemes=set(selection_schemes),
        fit_output_root=str(config["execution"]["fit_output_root"]),
    )
    if len(rows) != int(config["expected"]["jobs"]):
        raise OracleGraphTransferError("transfer job count changed")
    return rows


def _fit_output_path(repo_root: Path, row: Mapping[str, Any]) -> Path:
    relative = Path(str(row["output_relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise OracleGraphTransferError("transfer fit output escaped the repository")
    return repo_root / relative


def validate_completed_transfer_fit(
    path: Path,
    *,
    row: Mapping[str, Any],
    config_sha256: str,
    input_hashes: Mapping[str, str],
) -> bool:
    """Validate a resumable output or report that it does not yet exist."""

    if not path.exists():
        return False
    result = _load_json(path, "completed transfer fit")
    if (
        result.get("schema_version") != FIT_SCHEMA_VERSION
        or result.get("status") != "completed"
        or result.get("job", {}).get("job_id") != row["job_id"]
        or result.get("config_sha256") != config_sha256
        or result.get("input_hashes") != dict(sorted(input_hashes.items()))
        or result.get("boundary", {}).get("calibration_and_test_accessed_after_model_hash")
        is not True
    ):
        raise OracleGraphTransferError(f"completed transfer fit is stale or corrupt: {path}")
    return True


def _run_fit_shard(
    config_path_text: str,
    repo_root_text: str,
    shard_index: int,
    shard_count: int,
) -> dict[str, int]:
    repo_root = Path(repo_root_text)
    config_path = Path(config_path_text)
    (
        config,
        config_sha256,
        verified,
        records,
        assignments,
        checkpoint,
        vocabulary,
    ) = _load_worker_inputs(config_path, repo_root)
    torch.set_num_threads(int(config["execution"]["torch_threads_per_worker"]))
    try:
        torch.set_num_interop_threads(int(config["execution"]["torch_interop_threads_per_worker"]))
    except RuntimeError:
        if torch.get_num_interop_threads() != int(
            config["execution"]["torch_interop_threads_per_worker"]
        ):
            raise
    torch.use_deterministic_algorithms(True)
    split_manifest = _load_json(
        repo_root / verified["oracle_split_manifest"]["path"],
        "oracle split manifest",
    )
    rows = _job_rows(config, assignments, split_manifest)
    selected = [
        row for row in rows if int(str(row["job_id"])[:16], 16) % shard_count == shard_index
    ]
    label_to_index = {record.label: index for index, record in enumerate(records)}
    input_hashes = {name: metadata["sha256"] for name, metadata in sorted(verified.items())}
    completed = 0
    skipped = 0
    for row in selected:
        path = _fit_output_path(repo_root, row)
        if validate_completed_transfer_fit(
            path,
            row=row,
            config_sha256=config_sha256,
            input_hashes=input_hashes,
        ):
            skipped += 1
            continue
        result = fit_transfer_job(
            row,
            records=records,
            label_to_index=label_to_index,
            assignments=assignments,
            vocabulary=vocabulary,
            checkpoint=checkpoint,
            checkpoint_sha256=verified["pretraining_checkpoint"]["sha256"],
            config=config,
            config_sha256=config_sha256,
            input_hashes=input_hashes,
        )
        _atomic_write(path, _stable_json(_finite_or_none(result)))
        completed += 1
    return {"selected": len(selected), "completed": completed, "skipped": skipped}


def run_transfer_fit_workers(
    config_path: Path,
    repo_root: Path,
) -> list[dict[str, int]]:
    """Run all missing transfer fits in four deterministic CPU shards."""

    config = _load_json(config_path, "graph transfer config")
    _validate_config_sections(config)
    workers = int(config["execution"]["workers"])
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [
            executor.submit(
                _run_fit_shard,
                str(config_path),
                str(repo_root),
                shard,
                workers,
            )
            for shard in range(workers)
        ]
        return [future.result() for future in futures]


def ensemble_transfer_fits(
    fits: Sequence[Mapping[str, Any]],
    *,
    coverages: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Combine seed predictions and calibrate only the completed ensemble."""

    if len(fits) != 3:
        raise OracleGraphTransferError("transfer ensemble requires exactly three seeds")
    ordered = sorted(fits, key=lambda fit: int(fit["job"]["seed"]))
    reference = ordered[0]
    identity = tuple(
        reference["job"][field] for field in ("architecture", "endpoint", "scheme", "fold")
    )
    for fit in ordered:
        if (
            tuple(fit["job"][field] for field in ("architecture", "endpoint", "scheme", "fold"))
            != identity
        ):
            raise OracleGraphTransferError("transfer ensemble mixes different partitions")
        for stage in ("calibration", "test"):
            if (
                fit[stage]["labels"] != reference[stage]["labels"]
                or fit[stage]["truth"] != reference[stage]["truth"]
            ):
                raise OracleGraphTransferError("transfer ensemble rows differ across seeds")
    calibration_truth = np.asarray(reference["calibration"]["truth"], dtype=np.float64)
    calibration_predictions = np.asarray(
        [fit["calibration"]["prediction"] for fit in ordered],
        dtype=np.float64,
    )
    calibration_prediction = np.mean(calibration_predictions, axis=0)
    test_truth = np.asarray(reference["test"]["truth"], dtype=np.float64)
    test_predictions = np.asarray(
        [fit["test"]["prediction"] for fit in ordered],
        dtype=np.float64,
    )
    test_prediction = np.mean(test_predictions, axis=0)
    test_standard_deviation = np.std(test_predictions, axis=0)
    calibration_metrics = regression_metrics(
        calibration_truth,
        calibration_prediction,
    )
    metrics = regression_metrics(test_truth, test_prediction)
    variant, endpoint, scheme, fold = identity
    metric_row: dict[str, Any] = {
        "representation": variant,
        "model": "pretrained_dmpnn_3seed_ensemble",
        "endpoint": endpoint,
        "scheme": scheme,
        "fold": int(fold),
        "train_rows": int(reference["job"]["train_rows"]),
        "calibration_rows": len(calibration_truth),
        "test_rows": len(test_truth),
        "seed_count": len(ordered),
        "mean_selected_epochs": float(
            np.mean([fit["epoch_selection"]["best_epoch"] for fit in ordered])
        ),
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
    }
    residuals = np.abs(calibration_truth - calibration_prediction)
    for coverage in coverages:
        percent = int(round(100 * float(coverage)))
        radius = conformal_radius(residuals, float(coverage))
        empirical = float(
            np.mean(
                (test_truth >= test_prediction - radius) & (test_truth <= test_prediction + radius)
            )
        )
        metric_row[f"conformal_q{percent}"] = radius
        metric_row[f"test_coverage{percent}"] = empirical
        metric_row[f"test_mean_interval_width{percent}"] = 2.0 * radius
    prediction_rows = [
        {
            "representation": variant,
            "model": "pretrained_dmpnn_3seed_ensemble",
            "endpoint": endpoint,
            "scheme": scheme,
            "fold": int(fold),
            "label": label,
            "y_true": float(test_truth[index]),
            "ensemble_y_pred": float(test_prediction[index]),
            "ensemble_standard_deviation": float(test_standard_deviation[index]),
            "absolute_error": float(abs(test_truth[index] - test_prediction[index])),
        }
        for index, label in enumerate(reference["test"]["labels"])
    ]
    return metric_row, prediction_rows


def _csv_gzip(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    sort_key: Any,
) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=sort_key):
        writer.writerow(
            {
                field: (
                    ""
                    if value is None
                    else format(value, ".12g")
                    if isinstance(value, float)
                    else value
                )
                for field in fields
                for value in [row.get(field, "")]
            }
        )
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as archive:
        archive.write(text.getvalue().encode())
    return output.getvalue()


def aggregate_transfer_fits(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Aggregate resumable fits into calibrated three-seed transfer ensembles."""

    (
        config,
        config_sha256,
        verified,
        _records,
        assignments,
        _checkpoint,
        _vocabulary,
    ) = _load_worker_inputs(config_path, repo_root)
    split_manifest = _load_json(
        repo_root / verified["oracle_split_manifest"]["path"],
        "oracle split manifest",
    )
    rows = _job_rows(config, assignments, split_manifest)
    input_hashes = {name: metadata["sha256"] for name, metadata in sorted(verified.items())}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fit_sources: list[dict[str, str]] = []
    for row in rows:
        path = _fit_output_path(repo_root, row)
        if not validate_completed_transfer_fit(
            path,
            row=row,
            config_sha256=config_sha256,
            input_hashes=input_hashes,
        ):
            raise OracleGraphTransferError(f"transfer fit is missing: {path}")
        grouped[str(row["ensemble_id"])].append(_load_json(path, "completed transfer fit"))
        fit_sources.append(
            {
                "job_id": str(row["job_id"]),
                "path": str(row["output_relative_path"]),
                "sha256": _sha256_file(path),
            }
        )
    if len(grouped) != int(config["expected"]["ensembles"]):
        raise OracleGraphTransferError("transfer ensemble count changed")
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for fits in grouped.values():
        metric, predictions = ensemble_transfer_fits(
            fits,
            coverages=[float(value) for value in config["evaluation"]["conformal_coverages"]],
        )
        metric_rows.append(metric)
        prediction_rows.extend(predictions)
    selection_schemes = split_manifest["evaluation_contract"]["selection_eligible_schemes"]
    aggregate_rows, endpoint_leaders = aggregate_selection_metrics(
        metric_rows,
        selection_schemes,
    )
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        by_variant[str(row["representation"])].append(row)
    endpoint_weights = {
        str(endpoint): float(weight)
        for endpoint, weight in config["matrix"]["endpoint_weights"].items()
    }
    combined = []
    for variant, endpoint_rows in sorted(by_variant.items()):
        if {str(row["endpoint"]) for row in endpoint_rows} != set(endpoint_weights):
            raise OracleGraphTransferError(f"{variant} lacks one endpoint aggregate")
        combined.append(
            {
                "variant": variant,
                "weighted_mean_test_r2": sum(
                    endpoint_weights[str(row["endpoint"])] * float(row["mean_test_r2"])
                    for row in endpoint_rows
                ),
                "weighted_mean_test_rmse": sum(
                    endpoint_weights[str(row["endpoint"])] * float(row["mean_test_rmse"])
                    for row in endpoint_rows
                ),
                "endpoint_rows": endpoint_rows,
            }
        )
    leader = min(
        combined,
        key=lambda row: (
            -float(row["weighted_mean_test_r2"]),
            float(row["weighted_mean_test_rmse"]),
            str(row["variant"]),
        ),
    )
    metrics_name = "oracle_graph_transfer_metrics.csv.gz"
    predictions_name = "oracle_graph_transfer_predictions.csv.gz"
    result_name = "oracle_graph_transfer_result.json"
    metrics_payload = _csv_gzip(
        _finite_or_none(metric_rows),
        METRIC_FIELDS,
        sort_key=lambda row: (
            row["endpoint"],
            row["representation"],
            row["scheme"],
            row["fold"],
        ),
    )
    predictions_payload = _csv_gzip(
        _finite_or_none(prediction_rows),
        PREDICTION_FIELDS,
        sort_key=lambda row: (
            row["endpoint"],
            row["representation"],
            row["scheme"],
            row["fold"],
            row["label"],
        ),
    )
    _atomic_write(output_dir / metrics_name, metrics_payload)
    _atomic_write(output_dir / predictions_name, predictions_payload)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed_pretrained_oracle_transfer_pending_scientific_freeze",
        "inputs": {
            "config": {
                "path": str(config_path.resolve().relative_to(repo_root.resolve())),
                "sha256": config_sha256,
                "bytes": config_path.stat().st_size,
            },
            **verified,
        },
        "artifacts": {
            "metrics": {
                "path": metrics_name,
                "sha256": hashlib.sha256(metrics_payload).hexdigest(),
                "bytes": len(metrics_payload),
            },
            "predictions": {
                "path": predictions_name,
                "sha256": hashlib.sha256(predictions_payload).hexdigest(),
                "bytes": len(predictions_payload),
            },
        },
        "fit_sources": {
            "count": len(fit_sources),
            "entries": sorted(fit_sources, key=lambda row: row["job_id"]),
        },
        "summary": {
            "fits": len(rows),
            "ensembles": len(grouped),
            "variants": list(TRANSFER_VARIANTS),
            "endpoints": list(ENDPOINTS),
            "selection_schemes": selection_schemes,
            "endpoint_leaders": endpoint_leaders,
            "combined_rows": combined,
            "combined_transfer_leader": leader,
        },
        "boundary": {
            "encoder_pretraining_is_label_free": True,
            "epoch_selection_is_train_only": True,
            "target_scaling_is_train_only": True,
            "calibration_occurs_after_three_seed_ensemble": True,
            "random_split_is_diagnostic_only": True,
        },
        "decision": {
            "oracle_model_frozen": False,
            "reason": "transfer comparison must be adjudicated with the full representation matrix",
        },
    }
    result = _finite_or_none(result)
    _atomic_write(output_dir / result_name, _stable_json(result))
    return result


def run_oracle_graph_transfer(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Run missing transfer fits and aggregate the completed seed ensembles."""

    workers = run_transfer_fit_workers(config_path, repo_root)
    result = aggregate_transfer_fits(config_path, output_dir, repo_root)
    result["worker_summary"] = workers
    return result
