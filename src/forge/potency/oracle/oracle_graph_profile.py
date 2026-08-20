"""Train-only runtime and tensorization gate for M0-07 graph oracle models."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from forge.core.hashing import sha256_file as _sha256_file
from forge.potency.oracle.oracle_graph import (
    DMPNNEncoder,
    EdgeGINEncoder,
    GraphFeatureVocabulary,
    GraphModelInput,
    OracleGraphRecord,
    UgiRoleAwareRegressor,
    WholeGraphRegressor,
    batch_graphs,
    collate_oracle_records,
    load_oracle_graph_records,
    permute_graph_tensor,
    tensorize_smiles,
)

CONFIG_SCHEMA_VERSION = "m0_07_oracle_graph_profile_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_oracle_graph_profile.v1"


class OracleGraphProfileError(ValueError):
    """Raised when the graph profiling contract is violated."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise OracleGraphProfileError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OracleGraphProfileError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise OracleGraphProfileError(f"{label} must contain an object")
    return value


def _verify_input(
    repo_root: Path,
    specification: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    relative = specification.get("path")
    expected = specification.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise OracleGraphProfileError(f"{label} input specification is incomplete")
    path = repo_root / relative
    if not path.is_file():
        raise OracleGraphProfileError(f"{label} not found: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise OracleGraphProfileError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return {"path": relative, "sha256": observed, "bytes": path.stat().st_size}


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    try:
        with opener(path, "rt", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except FileNotFoundError as exc:
        raise OracleGraphProfileError(f"input not found: {path}") from exc


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    commit = run("rev-parse", "HEAD").strip()
    status = run("status", "--porcelain=v1")
    diff = run("diff", "--binary", "HEAD")
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def _set_torch_threads(intraop: int, interop: int) -> None:
    torch.set_num_threads(intraop)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError as exc:
        if torch.get_num_interop_threads() != interop:
            raise OracleGraphProfileError(
                "PyTorch inter-op threads were initialized before the profile"
            ) from exc


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _load_profile_labels(
    path: Path,
    *,
    scheme: str,
    fold: int,
    stage: str,
) -> set[str]:
    rows = _read_csv(path)
    labels = {
        row["label"]
        for row in rows
        if row["scheme"] == scheme and int(row["fold"]) == fold and row["stage"] == stage
    }
    if not labels:
        raise OracleGraphProfileError("profile partition selects no labels")
    return labels


def _tensorization_audit(
    path: Path,
    *,
    smiles_fields: Sequence[str],
    label_field: str,
    vocabulary: GraphFeatureVocabulary,
) -> dict[str, Any]:
    rows = _read_csv(path)
    if rows:
        missing = (set(smiles_fields) | {label_field}) - set(rows[0])
        if missing:
            raise OracleGraphProfileError(f"{path} lacks tensorization fields: {sorted(missing)}")
    graphs = 0
    atoms = 0
    directed_edges = 0
    maximum_atoms = 0
    started = time.perf_counter()
    for row in rows:
        for field in smiles_fields:
            graph = tensorize_smiles(
                row[field],
                vocabulary,
                label=f"{path.name} {row[label_field]} {field}",
            )
            graphs += 1
            atoms += graph.num_nodes
            directed_edges += graph.num_directed_edges
            maximum_atoms = max(maximum_atoms, graph.num_nodes)
    wall_seconds = time.perf_counter() - started
    return {
        "records": len(rows),
        "graphs": graphs,
        "atoms": atoms,
        "directed_edges": directed_edges,
        "maximum_atoms": maximum_atoms,
        "wall_seconds": wall_seconds,
        "graphs_per_second": graphs / wall_seconds,
    }


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


def _budget_batches(
    records: Sequence[OracleGraphRecord],
    order: Sequence[int],
    *,
    role_aware: bool,
    maximum_graphs: int,
    maximum_atoms: int,
    maximum_directed_edges: int,
) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    atoms = 0
    directed_edges = 0
    for index in order:
        record_atoms, record_edges = _record_size(records[index], role_aware)
        if record_atoms > maximum_atoms or record_edges > maximum_directed_edges:
            raise OracleGraphProfileError(
                "one admitted graph record exceeds the frozen batch budget"
            )
        would_overflow = current and (
            len(current) >= maximum_graphs
            or atoms + record_atoms > maximum_atoms
            or directed_edges + record_edges > maximum_directed_edges
        )
        if would_overflow:
            batches.append(current)
            current = []
            atoms = 0
            directed_edges = 0
        current.append(index)
        atoms += record_atoms
        directed_edges += record_edges
    if current:
        batches.append(current)
    return batches


def _model(
    architecture: str,
    *,
    vocabulary: GraphFeatureVocabulary,
    hidden_dim: int,
    depth: int,
    dropout: float,
) -> nn.Module:
    if architecture in {"whole_graph_dmpnn", "ugi_component_role_aware_dmpnn"}:
        encoder: nn.Module = DMPNNEncoder(
            vocabulary.atom_feature_dim,
            vocabulary.bond_feature_dim,
            hidden_dim,
            depth,
            dropout,
        )
    elif architecture == "whole_graph_edge_gin":
        encoder = EdgeGINEncoder(
            vocabulary.atom_feature_dim,
            vocabulary.bond_feature_dim,
            hidden_dim,
            depth,
            dropout,
        )
    else:
        raise OracleGraphProfileError(f"unsupported profile architecture: {architecture}")
    if architecture == "ugi_component_role_aware_dmpnn":
        return UgiRoleAwareRegressor(encoder, hidden_dim, outputs=1)
    return WholeGraphRegressor(encoder, hidden_dim, outputs=1)


def _permuted_inputs(
    record: OracleGraphRecord,
    *,
    seed: int,
) -> GraphModelInput:
    generator = torch.Generator().manual_seed(seed)
    graphs = []
    for graph in (record.product, record.amine, record.aldehyde, record.isocyanide):
        node_order = torch.randperm(graph.num_nodes, generator=generator)
        edge_order = torch.randperm(graph.num_directed_edges, generator=generator)
        graphs.append(permute_graph_tensor(graph, node_order, edge_order))
    return GraphModelInput(
        product=batch_graphs([graphs[0]]),
        amine=batch_graphs([graphs[1]]),
        aldehyde=batch_graphs([graphs[2]]),
        isocyanide=batch_graphs([graphs[3]]),
    )


def _train_epoch(
    model: nn.Module,
    records: Sequence[OracleGraphRecord],
    *,
    order: Sequence[int],
    role_aware: bool,
    target_mean: float,
    target_scale: float,
    optimizer: torch.optim.Optimizer,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    model.train()
    batches = _budget_batches(
        records,
        order,
        role_aware=role_aware,
        maximum_graphs=int(training["maximum_graphs_per_batch"]),
        maximum_atoms=int(training["maximum_atoms_per_batch"]),
        maximum_directed_edges=int(training["maximum_directed_edges_per_batch"]),
    )
    losses = []
    examples = 0
    atoms = 0
    directed_edges = 0
    for indices in batches:
        supervised, _ = collate_oracle_records([records[index] for index in indices])
        targets = (supervised.targets[:, 0] - target_mean) / target_scale
        optimizer.zero_grad(set_to_none=True)
        prediction = model(supervised.inputs).squeeze(-1)
        loss = torch.mean((prediction - targets) ** 2)
        if not torch.isfinite(loss):
            raise OracleGraphProfileError("nonfinite train-only profile loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise OracleGraphProfileError("nonfinite train-only profile gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
        examples += len(indices)
        record_graphs = [
            (
                (
                    record.product,
                    record.amine,
                    record.aldehyde,
                    record.isocyanide,
                )
                if role_aware
                else (record.product,)
            )
            for record in (records[index] for index in indices)
        ]
        atoms += sum(graph.num_nodes for graphs in record_graphs for graph in graphs)
        directed_edges += sum(
            graph.num_directed_edges for graphs in record_graphs for graph in graphs
        )
    return {
        "mean_loss": sum(losses) / len(losses),
        "steps": len(batches),
        "examples": examples,
        "atoms": atoms,
        "directed_edges": directed_edges,
    }


def _profile_architecture(
    architecture: str,
    *,
    records: Sequence[OracleGraphRecord],
    vocabulary: GraphFeatureVocabulary,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    training = config["training"]
    seed = int(config["seed"])
    role_aware = architecture == "ugi_component_role_aware_dmpnn"
    targets = torch.tensor(
        [float(record.targets[0]) for record in records],
        dtype=torch.float64,
    )
    target_mean = float(targets.mean())
    target_scale = float(targets.std(unbiased=False))
    if not math.isfinite(target_scale) or target_scale <= 0:
        raise OracleGraphProfileError("train-only target scale is not positive")
    repeats = []
    tolerance = float(config["acceptance"]["permutation_invariance_absolute_tolerance"])
    for repeat in range(int(training["repeats"])):
        repeat_seed = seed + 10_000 * repeat
        _seed_everything(repeat_seed)
        model = _model(
            architecture,
            vocabulary=vocabulary,
            hidden_dim=int(training["hidden_dim"]),
            depth=int(training["depth"]),
            dropout=float(training["dropout"]),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        for epoch in range(int(training["warmup_epochs"])):
            generator = torch.Generator().manual_seed(repeat_seed + epoch)
            order = torch.randperm(len(records), generator=generator).tolist()
            _train_epoch(
                model,
                records,
                order=order,
                role_aware=role_aware,
                target_mean=target_mean,
                target_scale=target_scale,
                optimizer=optimizer,
                training=training,
            )
        epoch_results = []
        for epoch in range(int(training["timed_epochs"])):
            generator = torch.Generator().manual_seed(
                repeat_seed + int(training["warmup_epochs"]) + epoch
            )
            order = torch.randperm(len(records), generator=generator).tolist()
            started = time.perf_counter()
            result = _train_epoch(
                model,
                records,
                order=order,
                role_aware=role_aware,
                target_mean=target_mean,
                target_scale=target_scale,
                optimizer=optimizer,
                training=training,
            )
            wall_seconds = time.perf_counter() - started
            epoch_results.append(
                {
                    **result,
                    "wall_seconds": wall_seconds,
                    "examples_per_second": result["examples"] / wall_seconds,
                    "atoms_per_second": result["atoms"] / wall_seconds,
                    "directed_edges_per_second": (result["directed_edges"] / wall_seconds),
                }
            )
        model.eval()
        original, _ = collate_oracle_records([records[0]])
        permuted = _permuted_inputs(records[0], seed=repeat_seed + 999)
        with torch.no_grad():
            original_prediction = model(original.inputs)
            permuted_prediction = model(permuted)
        maximum_difference = float(torch.max(torch.abs(original_prediction - permuted_prediction)))
        if maximum_difference > tolerance:
            raise OracleGraphProfileError(
                f"{architecture} permutation difference {maximum_difference} exceeds {tolerance}"
            )
        repeats.append(
            {
                "repeat": repeat,
                "seed": repeat_seed,
                "epochs": epoch_results,
                "checkpoint_sha256": _model_digest(model),
                "permutation_maximum_absolute_difference": maximum_difference,
                "peak_process_rss_bytes_after": _peak_rss_bytes(),
            }
        )
    all_epoch_times = [epoch["wall_seconds"] for repeat in repeats for epoch in repeat["epochs"]]
    all_examples_per_second = [
        epoch["examples_per_second"] for repeat in repeats for epoch in repeat["epochs"]
    ]
    maximum_wall = float(config["acceptance"]["maximum_profile_wall_seconds_per_architecture"])
    if max(all_epoch_times) > maximum_wall:
        raise OracleGraphProfileError(f"{architecture} exceeded the profile wall-time gate")
    return {
        "architecture": architecture,
        "role_aware": role_aware,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "target_scaler": {
            "fit_records": len(records),
            "mean": target_mean,
            "population_standard_deviation": target_scale,
        },
        "repeats": repeats,
        "epoch_wall_seconds": {
            "minimum": min(all_epoch_times),
            "median": float(np.median(all_epoch_times)),
            "maximum": max(all_epoch_times),
        },
        "examples_per_second": {
            "minimum": min(all_examples_per_second),
            "median": float(np.median(all_examples_per_second)),
            "maximum": max(all_examples_per_second),
        },
    }


def run_oracle_graph_profile(
    config_path: Path,
    output_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Run the predeclared train-only graph implementation profile."""

    config = _load_json(config_path, "graph profile config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise OracleGraphProfileError("unsupported graph-profile config schema")
    if config.get("seed") != 1729:
        raise OracleGraphProfileError("graph-profile seed must remain 1729")
    partition = config.get("partition")
    if (
        not isinstance(partition, dict)
        or partition.get("stage") != "train"
        or partition.get("calibration_or_test_targets_used") is not False
    ):
        raise OracleGraphProfileError("graph profile must remain train-only")
    training = config.get("training")
    if (
        not isinstance(training, dict)
        or training.get("early_stopping") is not False
        or training.get("calibration_used") is not False
        or training.get("profile_endpoint") != "expt_Hela"
    ):
        raise OracleGraphProfileError("graph training-profile boundary changed")
    expected_architectures = {
        "whole_graph_dmpnn",
        "whole_graph_edge_gin",
        "ugi_component_role_aware_dmpnn",
    }
    if set(training.get("architectures", ())) != expected_architectures:
        raise OracleGraphProfileError("graph profile architecture set changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise OracleGraphProfileError("graph profile lacks inputs")
    verified = {
        name: _verify_input(repo_root, specification, name)
        for name, specification in sorted(inputs.items())
        if isinstance(specification, dict)
    }
    if set(verified) != {
        "curated_oracle_data",
        "graph_corpus_result",
        "r0_pretraining_ledger",
        "split_assignments",
        "virtual_candidate_library",
    }:
        raise OracleGraphProfileError("graph profile inputs are incomplete")
    _set_torch_threads(
        int(training["torch_threads"]),
        int(training["torch_interop_threads"]),
    )
    _seed_everything(int(config["seed"]))
    vocabulary = GraphFeatureVocabulary.from_corpus_result(
        repo_root / verified["graph_corpus_result"]["path"]
    )
    profile_labels = _load_profile_labels(
        repo_root / verified["split_assignments"]["path"],
        scheme=str(partition["scheme"]),
        fold=int(partition["fold"]),
        stage=str(partition["stage"]),
    )
    expected = config.get("expected")
    if not isinstance(expected, dict):
        raise OracleGraphProfileError("graph profile lacks expected counts")
    if len(profile_labels) != expected["profile_train_records"]:
        raise OracleGraphProfileError("profile training-label count changed")
    records = load_oracle_graph_records(
        repo_root / verified["curated_oracle_data"]["path"],
        vocabulary,
        allowed_labels=profile_labels,
    )
    records.sort(key=lambda record: record.label)
    tensorization = {
        "curated_product_and_roles": _tensorization_audit(
            repo_root / verified["curated_oracle_data"]["path"],
            smiles_fields=("model_smiles", "A_smiles", "B_smiles", "C_smiles"),
            label_field="label",
            vocabulary=vocabulary,
        ),
        "virtual_applicability_products": _tensorization_audit(
            repo_root / verified["virtual_candidate_library"]["path"],
            smiles_fields=("canonical_isomeric_smiles",),
            label_field="source_row_index",
            vocabulary=vocabulary,
        ),
        "r0_pretraining_products": _tensorization_audit(
            repo_root / verified["r0_pretraining_ledger"]["path"],
            smiles_fields=("constitutional_smiles",),
            label_field="graph_id",
            vocabulary=vocabulary,
        ),
    }
    if (
        tensorization["curated_product_and_roles"]["records"] != expected["curated_records"]
        or tensorization["virtual_applicability_products"]["records"]
        != expected["virtual_candidate_records"]
        or tensorization["r0_pretraining_products"]["records"] != expected["r0_pretraining_records"]
    ):
        raise OracleGraphProfileError("graph tensorization record count changed")
    architecture_profiles = [
        _profile_architecture(
            architecture,
            records=records,
            vocabulary=vocabulary,
            config=config,
        )
        for architecture in training["architectures"]
    ]
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "passed_train_only_graph_runtime_gate",
        "generated_utc": datetime.now(UTC).isoformat(),
        "seed": config["seed"],
        "inputs": {
            "config": {
                "path": str(config_path.resolve().relative_to(repo_root.resolve())),
                "sha256": _sha256_file(config_path),
                "bytes": config_path.stat().st_size,
            },
            **verified,
        },
        "repository": _git_state(repo_root),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "device": "cpu",
            "num_workers": 0,
            "mixed_precision": False,
            "compilation": False,
        },
        "partition": {
            **partition,
            "selected_records": len(records),
            "calibration_or_test_targets_accessed": False,
        },
        "feature_vocabulary": vocabulary.to_dict(),
        "feature_dimensions": {
            "atoms": vocabulary.atom_feature_dim,
            "bonds": vocabulary.bond_feature_dim,
        },
        "tensorization": tensorization,
        "architecture_profiles": architecture_profiles,
        "decision": {
            "full_supervised_graph_matrix_authorized": True,
            "oracle_model_frozen": False,
            "scientific_model_selected_by_runtime": False,
            "calibration_or_test_labels_used": False,
        },
    }
    _atomic_write_json(output_path, result)
    return result
