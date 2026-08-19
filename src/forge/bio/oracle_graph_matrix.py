"""Leakage-safe supervised graph oracle fitting and ensemble evaluation."""

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
from torch import nn

from forge.bio.oracle_classical import (
    aggregate_selection_metrics,
    conformal_radius,
    regression_metrics,
)
from forge.bio.oracle_graph import (
    GraphFeatureVocabulary,
    OracleGraphRecord,
    collate_oracle_records,
    load_oracle_graph_records,
)
from forge.bio.oracle_graph_jobs import build_graph_job_rows
from forge.bio.oracle_graph_profile import _budget_batches, _model, _model_digest
from forge.core.hashing import sha256_file as _sha256_file
from forge.core.io import atomic_write as _atomic_write
from forge.core.io import pretty_json_bytes as _stable_json

CONFIG_SCHEMA_VERSION = "m0_07_oracle_graph_matrix_config.v1"
FIT_SCHEMA_VERSION = "m0_07_oracle_graph_fit.v1"
RESULT_SCHEMA_VERSION = "m0_07_oracle_graph_matrix.v1"
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


class OracleGraphMatrixError(ValueError):
    """Raised when supervised graph training violates its frozen contract."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise OracleGraphMatrixError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OracleGraphMatrixError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise OracleGraphMatrixError(f"{label} must contain an object")
    return value


def _verify_input(
    repo_root: Path,
    specification: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    relative = specification.get("path")
    expected = specification.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise OracleGraphMatrixError(f"{label} input specification is incomplete")
    path = repo_root / relative
    if not path.is_file():
        raise OracleGraphMatrixError(f"{label} not found: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise OracleGraphMatrixError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return {"path": relative, "sha256": observed, "bytes": path.stat().st_size}


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    try:
        with opener(path, "rt", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except FileNotFoundError as exc:
        raise OracleGraphMatrixError(f"input not found: {path}") from exc


def _fit_seed(repeat_seed: int, row: Mapping[str, Any]) -> int:
    key = "\x1f".join(
        (
            str(repeat_seed),
            str(row["architecture"]),
            str(row["endpoint"]),
            str(row["scheme"]),
            str(row["fold"]),
        )
    )
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return value % (2**31 - 1) or 1


def _inner_split(
    labels: Sequence[str],
    *,
    scheme: str,
    fold: int,
    fraction: float,
) -> tuple[list[int], list[int]]:
    if not 0.0 < fraction < 0.5:
        raise OracleGraphMatrixError("inner-validation fraction is invalid")
    ordered = sorted(
        range(len(labels)),
        key=lambda index: hashlib.sha256(
            f"inner-validation\x1f{scheme}\x1f{fold}\x1f{labels[index]}".encode()
        ).hexdigest(),
    )
    count = max(1, round(len(labels) * fraction))
    validation = set(ordered[:count])
    fit = [index for index in range(len(labels)) if index not in validation]
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
        raise OracleGraphMatrixError("target scaler is invalid")
    return mean, scale


def _record_size(record: OracleGraphRecord, role_aware: bool) -> tuple[int, int]:
    graphs = (
        (record.product, record.amine, record.aldehyde, record.isocyanide)
        if role_aware
        else (record.product,)
    )
    return (
        sum(graph.num_nodes for graph in graphs),
        sum(graph.num_directed_edges for graph in graphs),
    )


def _train_epoch(
    model: nn.Module,
    records: Sequence[OracleGraphRecord],
    indices: Sequence[int],
    *,
    endpoint_index: int,
    mean: float,
    scale: float,
    optimizer: torch.optim.Optimizer,
    role_aware: bool,
    model_config: Mapping[str, Any],
    order_seed: int,
) -> float:
    generator = torch.Generator().manual_seed(order_seed)
    shuffled = torch.tensor(list(indices), dtype=torch.long)[
        torch.randperm(len(indices), generator=generator)
    ].tolist()
    batches = _budget_batches(
        records,
        shuffled,
        role_aware=role_aware,
        maximum_graphs=int(model_config["maximum_graphs_per_batch"]),
        maximum_atoms=int(model_config["maximum_atoms_per_batch"]),
        maximum_directed_edges=int(model_config["maximum_directed_edges_per_batch"]),
    )
    model.train()
    weighted_loss = 0.0
    examples = 0
    for batch_indices in batches:
        supervised, _ = collate_oracle_records([records[index] for index in batch_indices])
        target = (supervised.targets[:, endpoint_index] - mean) / scale
        optimizer.zero_grad(set_to_none=True)
        prediction = model(supervised.inputs).squeeze(-1)
        loss = torch.mean((prediction - target) ** 2)
        if not torch.isfinite(loss):
            raise OracleGraphMatrixError("nonfinite graph training loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise OracleGraphMatrixError("nonfinite graph training gradient")
        optimizer.step()
        weighted_loss += float(loss.detach()) * len(batch_indices)
        examples += len(batch_indices)
    return weighted_loss / examples


def _validation_loss(
    model: nn.Module,
    records: Sequence[OracleGraphRecord],
    indices: Sequence[int],
    *,
    endpoint_index: int,
    mean: float,
    scale: float,
    role_aware: bool,
    model_config: Mapping[str, Any],
) -> float:
    batches = _budget_batches(
        records,
        list(indices),
        role_aware=role_aware,
        maximum_graphs=int(model_config["maximum_graphs_per_batch"]),
        maximum_atoms=int(model_config["maximum_atoms_per_batch"]),
        maximum_directed_edges=int(model_config["maximum_directed_edges_per_batch"]),
    )
    model.eval()
    squared_error = 0.0
    examples = 0
    with torch.no_grad():
        for batch_indices in batches:
            supervised, _ = collate_oracle_records([records[index] for index in batch_indices])
            target = (supervised.targets[:, endpoint_index] - mean) / scale
            prediction = model(supervised.inputs).squeeze(-1)
            squared_error += float(torch.sum((prediction - target) ** 2))
            examples += len(batch_indices)
    return squared_error / examples


def _predict(
    model: nn.Module,
    records: Sequence[OracleGraphRecord],
    indices: Sequence[int],
    *,
    mean: float,
    scale: float,
    role_aware: bool,
    model_config: Mapping[str, Any],
) -> list[float]:
    batches = _budget_batches(
        records,
        list(indices),
        role_aware=role_aware,
        maximum_graphs=int(model_config["maximum_graphs_per_batch"]),
        maximum_atoms=int(model_config["maximum_atoms_per_batch"]),
        maximum_directed_edges=int(model_config["maximum_directed_edges_per_batch"]),
    )
    model.eval()
    output = []
    with torch.no_grad():
        for batch_indices in batches:
            supervised, _ = collate_oracle_records([records[index] for index in batch_indices])
            prediction = model(supervised.inputs).squeeze(-1)
            output.extend((prediction * scale + mean).tolist())
    return [float(value) for value in output]


def _new_model(
    architecture: str,
    vocabulary: GraphFeatureVocabulary,
    model_config: Mapping[str, Any],
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    return _model(
        architecture,
        vocabulary=vocabulary,
        hidden_dim=int(model_config["hidden_dim"]),
        depth=int(model_config["depth"]),
        dropout=float(model_config["dropout"]),
    )


def fit_graph_job(
    row: Mapping[str, Any],
    *,
    records: Sequence[OracleGraphRecord],
    label_to_index: Mapping[str, int],
    assignments: Sequence[Mapping[str, str]],
    vocabulary: GraphFeatureVocabulary,
    config: Mapping[str, Any],
    config_sha256: str,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Fit one seed without using calibration or test targets for selection."""

    scheme = str(row["scheme"])
    fold = int(row["fold"])
    stage_labels: dict[str, list[str]] = defaultdict(list)
    for assignment in assignments:
        if str(assignment["scheme"]) == scheme and int(assignment["fold"]) == fold:
            stage_labels[str(assignment["stage"])].append(str(assignment["label"]))
    if set(stage_labels) != {"train", "calibration", "test"}:
        raise OracleGraphMatrixError(f"{scheme} fold {fold} stages are incomplete")
    stage_indices = {
        stage: [label_to_index[label] for label in sorted(labels)]
        for stage, labels in stage_labels.items()
    }
    endpoint = str(row["endpoint"])
    endpoint_index = {"expt_Hela": 0, "expt_Raw": 1}[endpoint]
    architecture = str(row["architecture"])
    role_aware = architecture == "ugi_component_role_aware_dmpnn"
    repeat_seed = int(row["seed"])
    fit_seed = _fit_seed(repeat_seed, row)
    train_indices = stage_indices["train"]
    train_labels = [records[index].label for index in train_indices]
    inner_fit_local, inner_validation_local = _inner_split(
        train_labels,
        scheme=scheme,
        fold=fold,
        fraction=float(config["epoch_selection"]["inner_validation_fraction"]),
    )
    inner_fit = [train_indices[index] for index in inner_fit_local]
    inner_validation = [train_indices[index] for index in inner_validation_local]
    selection_mean, selection_scale = _target_scaler(
        records,
        inner_fit,
        endpoint_index,
    )
    model = _new_model(architecture, vocabulary, config["model"], fit_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    best_epoch = 1
    best_validation = math.inf
    stale_epochs = 0
    selection_trace = []
    for epoch in range(1, int(config["epoch_selection"]["maximum_epochs"]) + 1):
        train_loss = _train_epoch(
            model,
            records,
            inner_fit,
            endpoint_index=endpoint_index,
            mean=selection_mean,
            scale=selection_scale,
            optimizer=optimizer,
            role_aware=role_aware,
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
            role_aware=role_aware,
            model_config=config["model"],
        )
        selection_trace.append(
            {
                "epoch": epoch,
                "inner_fit_loss": train_loss,
                "inner_validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation - float(config["epoch_selection"]["minimum_delta"]):
            best_validation = validation_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(config["epoch_selection"]["patience"]):
            break

    refit_mean, refit_scale = _target_scaler(records, train_indices, endpoint_index)
    model = _new_model(architecture, vocabulary, config["model"], fit_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
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
                role_aware=role_aware,
                model_config=config["model"],
                order_seed=fit_seed + 100_000 + epoch,
            )
        )
    checkpoint_sha256 = _model_digest(model)

    calibration_indices = stage_indices["calibration"]
    test_indices = stage_indices["test"]
    calibration_prediction = _predict(
        model,
        records,
        calibration_indices,
        mean=refit_mean,
        scale=refit_scale,
        role_aware=role_aware,
        model_config=config["model"],
    )
    test_prediction = _predict(
        model,
        records,
        test_indices,
        mean=refit_mean,
        scale=refit_scale,
        role_aware=role_aware,
        model_config=config["model"],
    )
    result = {
        "schema_version": FIT_SCHEMA_VERSION,
        "status": "completed",
        "job": dict(row),
        "fit_seed": fit_seed,
        "config_sha256": config_sha256,
        "input_hashes": dict(sorted(input_hashes.items())),
        "epoch_selection": {
            "fit_rows": len(inner_fit),
            "validation_rows": len(inner_validation),
            "target_mean": selection_mean,
            "target_scale": selection_scale,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation,
            "epochs_evaluated": len(selection_trace),
            "trace": selection_trace,
            "calibration_or_test_targets_accessed": False,
        },
        "refit": {
            "train_rows": len(train_indices),
            "target_mean": refit_mean,
            "target_scale": refit_scale,
            "epoch_count": best_epoch,
            "losses": refit_losses,
            "checkpoint_sha256_before_calibration_or_test_access": (checkpoint_sha256),
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
            "calibration_or_test_used_for_scaling": False,
            "calibration_or_test_used_for_epoch_selection": False,
            "calibration_or_test_used_for_weight_updates": False,
            "calibration_and_test_accessed_after_checkpoint_hash": True,
        },
    }
    return result


def _load_worker_inputs(
    config_path: Path,
    repo_root: Path,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, dict[str, Any]],
    list[OracleGraphRecord],
    list[dict[str, str]],
    GraphFeatureVocabulary,
]:
    config = _load_json(config_path, "graph matrix config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise OracleGraphMatrixError("unsupported graph-matrix config schema")
    if config.get("seed") != 1729:
        raise OracleGraphMatrixError("graph-matrix seed must remain 1729")
    matrix = config.get("matrix")
    model_config = config.get("model")
    epoch_selection = config.get("epoch_selection")
    evaluation = config.get("evaluation")
    execution = config.get("execution")
    if not all(
        isinstance(value, dict)
        for value in (
            matrix,
            model_config,
            epoch_selection,
            evaluation,
            execution,
        )
    ):
        raise OracleGraphMatrixError("graph matrix config sections are incomplete")
    if (
        epoch_selection.get("calibration_used") is not False
        or epoch_selection.get("test_used") is not False
        or epoch_selection.get("target_scaling") != "inner_fit_only_then_all_train_on_refit"
    ):
        raise OracleGraphMatrixError("epoch-selection leakage boundary changed")
    if (
        evaluation.get("calibrate_after_three_seed_ensemble") is not True
        or evaluation.get("random_split_role") != "diagnostic_only"
        or execution.get("workers") != 4
        or execution.get("torch_threads_per_worker") != 2
        or execution.get("resume_policy") != "validate_completed_output_then_skip"
    ):
        raise OracleGraphMatrixError("graph evaluation or execution policy changed")
    if set(matrix.get("architectures", ())) != {
        "whole_graph_dmpnn",
        "whole_graph_edge_gin",
        "ugi_component_role_aware_dmpnn",
    }:
        raise OracleGraphMatrixError("graph architecture set changed")
    if set(matrix.get("endpoints", ())) != {"expt_Hela", "expt_Raw"}:
        raise OracleGraphMatrixError("graph endpoint set changed")
    if matrix.get("select_one_architecture_across_endpoints") is not True:
        raise OracleGraphMatrixError("graph matrix must select one architecture")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise OracleGraphMatrixError("graph matrix config lacks inputs")
    verified = {
        name: _verify_input(repo_root, specification, name)
        for name, specification in sorted(inputs.items())
        if isinstance(specification, dict)
    }
    required = {
        "curated_oracle_data",
        "graph_corpus_result",
        "graph_profile_result",
        "oracle_split_assignments",
        "oracle_split_manifest",
    }
    if set(verified) != required:
        raise OracleGraphMatrixError("graph matrix inputs are incomplete")
    vocabulary = GraphFeatureVocabulary.from_corpus_result(
        repo_root / verified["graph_corpus_result"]["path"]
    )
    graph_profile = _load_json(
        repo_root / verified["graph_profile_result"]["path"],
        "graph profile result",
    )
    if (
        graph_profile.get("status") != "passed_train_only_graph_runtime_gate"
        or graph_profile.get("decision", {}).get("full_supervised_graph_matrix_authorized")
        is not True
    ):
        raise OracleGraphMatrixError("graph profile did not authorize the matrix")
    records = load_oracle_graph_records(
        repo_root / verified["curated_oracle_data"]["path"],
        vocabulary,
    )
    if len(records) != int(config["expected"]["curated_records"]):
        raise OracleGraphMatrixError("curated graph record count changed")
    assignments = _read_csv(repo_root / verified["oracle_split_assignments"]["path"])
    return (
        config,
        _sha256_file(config_path),
        verified,
        records,
        assignments,
        vocabulary,
    )


def _job_rows(
    config: Mapping[str, Any],
    assignments: Sequence[Mapping[str, str]],
    split_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selection_schemes = split_manifest.get("evaluation_contract", {}).get(
        "selection_eligible_schemes"
    )
    if not isinstance(selection_schemes, list) or "lantern_random" in selection_schemes:
        raise OracleGraphMatrixError("selection split contract is invalid")
    matrix = config["matrix"]
    rows = build_graph_job_rows(
        assignments,
        architectures=[str(value) for value in matrix["architectures"]],
        endpoints=[str(value) for value in matrix["endpoints"]],
        seeds=[int(value) for value in matrix["repeat_seeds"]],
        selection_schemes={str(value) for value in selection_schemes},
        fit_output_root=str(config["execution"]["fit_output_root"]),
    )
    if len(rows) != int(config["expected"]["jobs"]):
        raise OracleGraphMatrixError("graph job count changed")
    return rows


def _fit_output_path(repo_root: Path, row: Mapping[str, Any]) -> Path:
    return repo_root / str(row["output_relative_path"])


def _validate_completed_fit(
    path: Path,
    *,
    row: Mapping[str, Any],
    config_sha256: str,
    input_hashes: Mapping[str, str],
) -> bool:
    if not path.exists():
        return False
    value = _load_json(path, "completed graph fit")
    if (
        value.get("schema_version") != FIT_SCHEMA_VERSION
        or value.get("status") != "completed"
        or value.get("job", {}).get("job_id") != row["job_id"]
        or value.get("config_sha256") != config_sha256
        or value.get("input_hashes") != dict(sorted(input_hashes.items()))
    ):
        raise OracleGraphMatrixError(f"completed graph fit is stale or corrupt: {path}")
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
    jobs = _job_rows(config, assignments, split_manifest)
    selected = [
        row for row in jobs if int(str(row["job_id"])[:16], 16) % shard_count == shard_index
    ]
    label_to_index = {record.label: index for index, record in enumerate(records)}
    input_hashes = {name: record["sha256"] for name, record in sorted(verified.items())}
    completed = 0
    skipped = 0
    for row in selected:
        output_path = _fit_output_path(repo_root, row)
        if _validate_completed_fit(
            output_path,
            row=row,
            config_sha256=config_sha256,
            input_hashes=input_hashes,
        ):
            skipped += 1
            continue
        result = fit_graph_job(
            row,
            records=records,
            label_to_index=label_to_index,
            assignments=assignments,
            vocabulary=vocabulary,
            config=config,
            config_sha256=config_sha256,
            input_hashes=input_hashes,
        )
        _atomic_write(output_path, _stable_json(result))
        completed += 1
    return {"selected": len(selected), "completed": completed, "skipped": skipped}


def run_graph_fit_workers(
    config_path: Path,
    repo_root: Path,
) -> list[dict[str, int]]:
    """Run every missing graph fit across the frozen number of processes."""

    config = _load_json(config_path, "graph matrix config")
    workers = int(config["execution"]["workers"])
    if workers != 4:
        raise OracleGraphMatrixError("graph matrix must use four bounded workers")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [
            executor.submit(
                _run_fit_shard,
                str(config_path),
                str(repo_root),
                shard_index,
                workers,
            )
            for shard_index in range(workers)
        ]
        return [future.result() for future in futures]


def _metric_row(
    *,
    architecture: str,
    endpoint: str,
    scheme: str,
    fold: int,
    train_rows: int,
    calibration_truth: np.ndarray,
    calibration_prediction: np.ndarray,
    test_truth: np.ndarray,
    test_prediction: np.ndarray,
    selected_epochs: Sequence[int],
    coverages: Sequence[float],
) -> dict[str, Any]:
    calibration_metrics = regression_metrics(
        calibration_truth,
        calibration_prediction,
    )
    metrics = regression_metrics(test_truth, test_prediction)
    row: dict[str, Any] = {
        "representation": architecture,
        "model": "neural_3seed_ensemble",
        "endpoint": endpoint,
        "scheme": scheme,
        "fold": fold,
        "train_rows": train_rows,
        "calibration_rows": len(calibration_truth),
        "test_rows": len(test_truth),
        "seed_count": len(selected_epochs),
        "mean_selected_epochs": float(np.mean(selected_epochs)),
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
        percent = int(round(100 * coverage))
        radius = conformal_radius(residuals, coverage)
        empirical = float(
            np.mean(
                (test_truth >= test_prediction - radius) & (test_truth <= test_prediction + radius)
            )
        )
        row[f"conformal_q{percent}"] = radius
        row[f"test_coverage{percent}"] = empirical
        row[f"test_mean_interval_width{percent}"] = 2.0 * radius
    return row


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
                field: (format(value, ".12g") if isinstance(value, float) else value)
                for field in fields
                for value in [row.get(field, "")]
            }
        )
    return gzip.compress(text.getvalue().encode(), compresslevel=9, mtime=0)


def aggregate_graph_fits(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Aggregate seed predictions, calibrate ensembles, and rank architectures."""

    (
        config,
        config_sha256,
        verified,
        _records,
        assignments,
        _vocabulary,
    ) = _load_worker_inputs(config_path, repo_root)
    split_manifest = _load_json(
        repo_root / verified["oracle_split_manifest"]["path"],
        "oracle split manifest",
    )
    jobs = _job_rows(config, assignments, split_manifest)
    input_hashes = {name: record["sha256"] for name, record in sorted(verified.items())}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fit_sources: list[dict[str, str]] = []
    for row in jobs:
        path = _fit_output_path(repo_root, row)
        if not _validate_completed_fit(
            path,
            row=row,
            config_sha256=config_sha256,
            input_hashes=input_hashes,
        ):
            raise OracleGraphMatrixError(f"graph fit is missing: {path}")
        grouped[str(row["ensemble_id"])].append(_load_json(path, "completed graph fit"))
        fit_sources.append(
            {
                "job_id": str(row["job_id"]),
                "path": str(row["output_relative_path"]),
                "sha256": _sha256_file(path),
            }
        )
    if len(grouped) != int(config["expected"]["ensembles"]):
        raise OracleGraphMatrixError("graph ensemble count changed")
    metric_rows = []
    prediction_rows = []
    expected_seed_count = len(config["matrix"]["repeat_seeds"])
    for fits in grouped.values():
        if len(fits) != expected_seed_count:
            raise OracleGraphMatrixError("graph ensemble lacks a frozen seed")
        fits.sort(key=lambda value: int(value["job"]["seed"]))
        reference = fits[0]
        architecture = str(reference["job"]["architecture"])
        endpoint = str(reference["job"]["endpoint"])
        scheme = str(reference["job"]["scheme"])
        fold = int(reference["job"]["fold"])
        for stage in ("calibration", "test"):
            labels = reference[stage]["labels"]
            truth = reference[stage]["truth"]
            for fit in fits[1:]:
                if fit[stage]["labels"] != labels or fit[stage]["truth"] != truth:
                    raise OracleGraphMatrixError(
                        f"{architecture}/{endpoint}/{scheme}/{fold} "
                        f"{stage} rows differ across seeds"
                    )
        calibration_truth = np.asarray(reference["calibration"]["truth"], dtype=np.float64)
        calibration_predictions = np.asarray(
            [fit["calibration"]["prediction"] for fit in fits],
            dtype=np.float64,
        )
        calibration_prediction = np.mean(calibration_predictions, axis=0)
        test_truth = np.asarray(reference["test"]["truth"], dtype=np.float64)
        test_predictions = np.asarray(
            [fit["test"]["prediction"] for fit in fits],
            dtype=np.float64,
        )
        test_prediction = np.mean(test_predictions, axis=0)
        test_standard_deviation = np.std(test_predictions, axis=0)
        metric_rows.append(
            _metric_row(
                architecture=architecture,
                endpoint=endpoint,
                scheme=scheme,
                fold=fold,
                train_rows=int(reference["job"]["train_rows"]),
                calibration_truth=calibration_truth,
                calibration_prediction=calibration_prediction,
                test_truth=test_truth,
                test_prediction=test_prediction,
                selected_epochs=[int(fit["epoch_selection"]["best_epoch"]) for fit in fits],
                coverages=[float(value) for value in config["evaluation"]["conformal_coverages"]],
            )
        )
        for index, label in enumerate(reference["test"]["labels"]):
            prediction_rows.append(
                {
                    "representation": architecture,
                    "model": "neural_3seed_ensemble",
                    "endpoint": endpoint,
                    "scheme": scheme,
                    "fold": fold,
                    "label": label,
                    "y_true": float(test_truth[index]),
                    "ensemble_y_pred": float(test_prediction[index]),
                    "ensemble_standard_deviation": float(test_standard_deviation[index]),
                    "absolute_error": float(abs(test_truth[index] - test_prediction[index])),
                }
            )
    selection_schemes = split_manifest["evaluation_contract"]["selection_eligible_schemes"]
    aggregate_rows, endpoint_leaders = aggregate_selection_metrics(
        metric_rows,
        selection_schemes,
    )
    by_architecture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        by_architecture[str(row["representation"])].append(row)
    endpoint_weights = {
        str(key): float(value) for key, value in config["matrix"]["endpoint_weights"].items()
    }
    combined_rows = []
    for architecture, rows in sorted(by_architecture.items()):
        if {str(row["endpoint"]) for row in rows} != set(endpoint_weights):
            raise OracleGraphMatrixError(f"{architecture} lacks one endpoint aggregate")
        combined_rows.append(
            {
                "architecture": architecture,
                "weighted_mean_test_r2": sum(
                    endpoint_weights[str(row["endpoint"])] * float(row["mean_test_r2"])
                    for row in rows
                ),
                "weighted_mean_test_rmse": sum(
                    endpoint_weights[str(row["endpoint"])] * float(row["mean_test_rmse"])
                    for row in rows
                ),
                "endpoint_rows": rows,
            }
        )
    combined_leader = min(
        combined_rows,
        key=lambda row: (
            -float(row["weighted_mean_test_r2"]),
            float(row["weighted_mean_test_rmse"]),
            str(row["architecture"]),
        ),
    )
    metric_payload = _csv_gzip(
        metric_rows,
        METRIC_FIELDS,
        sort_key=lambda row: (
            row["endpoint"],
            row["representation"],
            row["scheme"],
            row["fold"],
        ),
    )
    prediction_payload = _csv_gzip(
        prediction_rows,
        PREDICTION_FIELDS,
        sort_key=lambda row: (
            row["endpoint"],
            row["representation"],
            row["scheme"],
            row["fold"],
            row["label"],
        ),
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed_graph_oracle_matrix_pending_scientific_freeze",
        "inputs": {
            "config": {
                "path": str(config_path.resolve().relative_to(repo_root.resolve())),
                "sha256": config_sha256,
                "bytes": config_path.stat().st_size,
            },
            **verified,
        },
        "summary": {
            "fits": len(jobs),
            "ensembles": len(grouped),
            "metric_rows": len(metric_rows),
            "prediction_rows": len(prediction_rows),
        },
        "fit_sources": {
            "count": len(fit_sources),
            "entries": sorted(fit_sources, key=lambda row: row["job_id"]),
        },
        "evaluation": config["evaluation"],
        "endpoint_leaders": endpoint_leaders,
        "combined_architecture_rows": combined_rows,
        "combined_architecture_leader": combined_leader,
        "decision": {
            "oracle_model_frozen": False,
            "requires_scientific_review_and_pretraining_comparison": True,
            "random_split_used_for_selection": False,
            "calibration_applied_after_seed_ensemble": True,
        },
        "artifacts": {
            "oracle_graph_matrix_metrics.csv.gz": {
                "sha256": hashlib.sha256(metric_payload).hexdigest(),
                "bytes": len(metric_payload),
            },
            "oracle_graph_matrix_predictions.csv.gz": {
                "sha256": hashlib.sha256(prediction_payload).hexdigest(),
                "bytes": len(prediction_payload),
            },
        },
    }
    result_payload = _stable_json(result)
    _atomic_write(output_dir / "oracle_graph_matrix_metrics.csv.gz", metric_payload)
    _atomic_write(
        output_dir / "oracle_graph_matrix_predictions.csv.gz",
        prediction_payload,
    )
    _atomic_write(output_dir / "oracle_graph_matrix_result.json", result_payload)
    return result


def run_oracle_graph_matrix(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Run missing fits in parallel and aggregate the complete graph matrix."""

    worker_summary = run_graph_fit_workers(config_path, repo_root)
    result = aggregate_graph_fits(config_path, output_dir, repo_root)
    result["worker_summary"] = worker_summary
    _atomic_write(output_dir / "oracle_graph_matrix_result.json", _stable_json(result))
    return result
