"""Deterministic training loop for the Ugi-first Phase 1 morphology flow."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from forge.bio.ugi_semantic_annotations import ROLE_NAMES
from forge.core.io import write_json as _atomic_json
from forge.product.defog_feasibility import sha256_file
from forge.product.ugi_morphology_corpus import (
    balanced_product_weights,
    component_marginal_errors,
    component_program_ledger,
    expanded_component_census,
    family_balanced_component_weights,
    load_expanded_ugi_morphology_corpus,
    load_ugi_morphology_corpus,
    sample_family_balanced_records,
)
from forge.product.ugi_morphology_flow import (
    UgiMorphologyFlow,
    UgiMorphologySample,
    collate_ugi_morphology_records,
    noise_ugi_morphology_batch,
    sample_ugi_morphologies,
    ugi_morphology_flow_loss,
    ugi_morphology_statistics,
)
from forge.product.ugi_morphology_program import (
    component_offspring_marginals,
    component_weighted_offspring_marginals,
    component_weighted_program_pool,
    sample_component_weighted_programs,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None


class UgiMorphologyTrainingError(RuntimeError):
    """Raised when production morphology training violates its frozen contract."""


_IMPLEMENTATION_PATHS = {
    "corpus": Path(__file__).with_name("ugi_morphology_corpus.py").resolve(),
    "flow": Path(__file__).with_name("ugi_morphology_flow.py").resolve(),
    "program": Path(__file__).with_name("ugi_morphology_program.py").resolve(),
    "training": Path(__file__).resolve(),
}


def _atomic_checkpoint(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _resolve_input(repo: Path, record: dict[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = repo / path
    if not path.is_file():
        raise UgiMorphologyTrainingError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != str(record["sha256"]):
        raise UgiMorphologyTrainingError(
            f"{label} hash mismatch: expected {record['sha256']}, observed {observed}"
        )
    return path, {"path": str(path), "sha256": observed}


def _set_determinism(seed: int, device: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _move(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


def _validation_loss(
    model: Any,
    records: tuple[Any, ...],
    source_tensors: Any,
    *,
    maximum_children: int,
    batch_size: int,
    batches: int,
    seed: int,
    device: Any,
    label_smoothing: float,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    totals = []
    role_values: dict[str, list[float]] = {}
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            indices = rng.choice(
                len(records),
                size=min(batch_size, len(records)),
                replace=False,
            )
            local = tuple(records[int(index)] for index in indices)
            batch = _move(
                collate_ugi_morphology_records(
                    local,
                    maximum_nodes=max(record.program.node_count for record in local),
                    maximum_children=maximum_children,
                ),
                device,
            )
            t = torch.rand(len(local), generator=generator, device=device).clamp(0.02, 0.98)
            noisy = noise_ugi_morphology_batch(batch, source_tensors, t, generator)
            predictions = model(
                noisy["offspring"],
                batch["role_states"],
                batch["within_role_positions"],
                batch["programs"],
                t,
                batch["node_mask"],
            )
            _, metrics = ugi_morphology_flow_loss(
                predictions,
                batch,
                label_smoothing=label_smoothing,
            )
            totals.append(metrics["total"])
            for key, value in metrics.items():
                if key != "total":
                    role_values.setdefault(key, []).append(value)
    return {
        "total": float(np.mean(totals)),
        **{key: float(np.mean(values)) for key, values in role_values.items()},
    }


def _reference_samples(records: tuple[Any, ...]) -> list[UgiMorphologySample]:
    return [
        UgiMorphologySample(
            program=record.program,
            offspring=tuple(component.offspring.copy() for component in record.components),
        )
        for record in records
    ]


def train_ugi_morphology(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    smoke: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Train with strict component folds and serial morphology sampling."""

    if torch is None:
        raise UgiMorphologyTrainingError("Ugi morphology training requires torch")
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UgiMorphologyTrainingError(f"invalid training config: {exc}") from exc
    schema_version = config.get("schema_version")
    if schema_version not in {
        "phase1_ugi_morphology_training_config.v1",
        "phase1_ugi_morphology_training_config.v2",
    }:
        raise UgiMorphologyTrainingError("unsupported Ugi morphology training config")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise UgiMorphologyTrainingError(f"output directory is nonempty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {}
    resolved = {}
    for label, record in config["inputs"].items():
        resolved[label], inputs[label] = _resolve_input(repo, record, label)

    expanded_mode = schema_version == "phase1_ugi_morphology_training_config.v2"
    product_weights: np.ndarray | None = None
    marginal_errors: dict[str, float] | None = None
    uniform_marginal_errors: dict[str, float] | None = None
    product_weight_ratio: float | None = None
    if expanded_mode:
        corpus = load_expanded_ugi_morphology_corpus(
            resolved["component_exemplar_ledger"],
            resolved["semantic_products"],
            resolved["semantic_atoms"],
            resolved["atom_vocabulary"],
        )
        train_components = tuple(
            component
            for key, component in corpus.unique_components.items()
            if corpus.component_metadata[key]["family_fold"] == "train"
        )
        calibration_records = sample_family_balanced_records(
            corpus,
            fold="calibration",
            count=int(config["sampling"]["calibration_pool_size"]),
            rng=np.random.default_rng(int(config["seed"]) + 400),
            id_prefix="fixed-calibration",
        )
        data_summary: dict[str, Any] = {
            **expanded_component_census(corpus),
            "product_rows_materialized": 0,
            "training_sampling": "uniform_family_then_uniform_component_per_role",
            "validation_sampling": "frozen_family_balanced_component_combinations",
        }
    else:
        corpus = load_ugi_morphology_corpus(
            resolved["assignments"],
            resolved["semantic_products"],
            resolved["semantic_atoms"],
            resolved["atom_vocabulary"],
        )
        train_records = corpus.records_by_fold["train"]
        calibration_records = corpus.records_by_fold["calibration"]
        train_assignments = corpus.assignments_by_fold["train"]
        product_weights = balanced_product_weights(train_assignments)
        marginal_errors = component_marginal_errors(train_assignments, product_weights)
        uniform_weights = np.full(len(train_assignments), 1.0 / len(train_assignments))
        uniform_marginal_errors = component_marginal_errors(train_assignments, uniform_weights)
        if any(
            marginal_errors[role] > uniform_marginal_errors[role] + 1e-12
            for role in marginal_errors
        ):
            raise UgiMorphologyTrainingError(
                "regularized product sampler increased component imbalance: "
                f"weighted={marginal_errors}, uniform={uniform_marginal_errors}"
            )
        product_weight_ratio = float(product_weights.max() / product_weights.min())
        if product_weight_ratio > 12.0:
            raise UgiMorphologyTrainingError(
                f"regularized product sampler has excessive weight ratio: {product_weight_ratio}"
            )
        train_components = tuple(corpus.unique_components.values())
        data_summary = {
            "products_by_fold": {
                fold: len(records) for fold, records in corpus.records_by_fold.items()
            },
            "unique_components": len(corpus.unique_components),
            "component_marginal_maximum_errors": marginal_errors,
            "uniform_row_component_marginal_errors": uniform_marginal_errors,
            "maximum_to_minimum_product_weight_ratio": product_weight_ratio,
            "training_sampling": "iterative_raking_with_50_percent_uniform_row_regularization",
        }

    mode = "smoke" if smoke else "full"
    runtime = dict(config[mode])
    model_config = dict(config["model"])
    if smoke:
        model_config.update(runtime.pop("model_overrides"))
    device = torch.device(str(runtime["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise UgiMorphologyTrainingError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise UgiMorphologyTrainingError("MPS requested but unavailable")
    seed = int(config["seed"])
    _set_determinism(seed, device)
    maximum_children = int(model_config["maximum_children"])
    source_marginals = (
        component_offspring_marginals(
            train_components,
            maximum_children=maximum_children,
            probability_floor=float(model_config["source_probability_floor"]),
            component_weights=family_balanced_component_weights(corpus, fold="train"),
        )
        if expanded_mode
        else component_weighted_offspring_marginals(
            train_records,
            maximum_children=maximum_children,
            probability_floor=float(model_config["source_probability_floor"]),
        )
    )
    source_tensors = torch.as_tensor(
        source_marginals,
        dtype=torch.float32,
        device=device,
    )
    model = UgiMorphologyFlow(
        maximum_children=maximum_children,
        maximum_component_atoms=int(model_config["maximum_component_atoms"]),
        maximum_total_atoms=int(model_config["maximum_total_atoms"]),
        maximum_junction_budget=int(model_config["maximum_junction_budget"]),
        maximum_cycle_rank=int(model_config["maximum_cycle_rank"]),
        maximum_attachment_count=(
            int(model_config["maximum_attachment_count"])
            if "maximum_attachment_count" in model_config
            else None
        ),
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["layers"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(runtime["learning_rate"]),
        weight_decay=float(runtime["weight_decay"]),
    )
    torch_generator = torch.Generator(device=device).manual_seed(seed + 1)
    numpy_generator = np.random.default_rng(seed + 2)
    program_rng = np.random.default_rng(seed + 3)
    program_pools = None if expanded_mode else component_weighted_program_pool(train_records)
    branch_run_policy = config.get("sampling", {}).get("maximum_adjacent_branch_run_by_role")
    maximum_adjacent_branch_runs = (
        tuple(int(branch_run_policy[role]) for role in ROLE_NAMES)
        if branch_run_policy is not None
        else None
    )
    label_smoothing = float(runtime.get("label_smoothing", 0.0))
    if not 0.0 <= label_smoothing < 1.0:
        raise UgiMorphologyTrainingError("label smoothing must be in [0, 1)")
    early_stopping = runtime.get("early_stopping", {})
    early_stopping_patience = int(early_stopping.get("patience", 0))
    early_stopping_min_delta = float(early_stopping.get("min_delta", 0.0))
    early_stopping_minimum_steps = int(early_stopping.get("minimum_steps", 0))
    if (
        early_stopping_patience < 0
        or early_stopping_min_delta < 0
        or early_stopping_minimum_steps < 0
    ):
        raise UgiMorphologyTrainingError("invalid early-stopping policy")
    losses = []
    evaluations = []
    progress_path = output_dir / "progress.json"
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    best_checkpoint_path = output_dir / "checkpoint_best.pt"
    best_calibration_loss = float("inf")
    best_step = 0
    evaluations_without_improvement = 0
    stop_reason: str | None = None

    def evaluate(step: int) -> dict[str, Any]:
        local_reference = calibration_records[
            : min(int(runtime["eval_samples"]), len(calibration_records))
        ]
        calibration_programs = tuple(record.program for record in local_reference)
        generated, sampling = sample_ugi_morphologies(
            model,
            calibration_programs,
            source_marginals,
            sample_steps=int(runtime["sample_steps"]),
            batch_size=int(runtime["batch_size"]),
            seed=seed + 10_000 + step,
            device=str(device),
            maximum_adjacent_branch_runs=maximum_adjacent_branch_runs,
        )
        if expanded_mode:
            exploratory_programs = tuple(
                record.program
                for record in sample_family_balanced_records(
                    corpus,
                    fold="train",
                    count=len(local_reference),
                    rng=program_rng,
                    id_prefix=f"exploratory-{step:08d}",
                )
            )
        else:
            exploratory_programs = sample_component_weighted_programs(
                program_pools,
                count=len(local_reference),
                rng=program_rng,
            )
        exploratory, exploratory_sampling = sample_ugi_morphologies(
            model,
            exploratory_programs,
            source_marginals,
            sample_steps=int(runtime["sample_steps"]),
            batch_size=int(runtime["batch_size"]),
            seed=seed + 30_000 + step,
            device=str(device),
            maximum_adjacent_branch_runs=maximum_adjacent_branch_runs,
        )
        evaluation = {
            "step": step,
            "calibration_loss": _validation_loss(
                model,
                calibration_records,
                source_tensors,
                maximum_children=maximum_children,
                batch_size=int(runtime["batch_size"]),
                batches=int(runtime["validation_batches"]),
                seed=seed + 20_000,
                device=device,
                label_smoothing=label_smoothing,
            ),
            "sampling": sampling,
            "reference": ugi_morphology_statistics(_reference_samples(local_reference)),
            "generated": ugi_morphology_statistics(generated),
            "calibration_programs": [program.__dict__ for program in calibration_programs],
            "offspring": [[values.tolist() for values in sample.offspring] for sample in generated],
            "exploratory": {
                "sampling": exploratory_sampling,
                "statistics": ugi_morphology_statistics(exploratory),
                "programs": [program.__dict__ for program in exploratory_programs],
                "offspring": [
                    [values.tolist() for values in sample.offspring] for sample in exploratory
                ],
            },
        }
        evaluations.append(evaluation)
        return evaluation

    def checkpoint_package(step: int) -> dict[str, Any]:
        return {
            "schema_version": "phase1_ugi_morphology_checkpoint.v2",
            "step": step,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "source_marginals": source_marginals,
            "training_state": {
                "losses": losses,
                "evaluations": evaluations,
                "best_calibration_loss": best_calibration_loss,
                "best_step": best_step,
                "evaluations_without_improvement": evaluations_without_improvement,
                "torch_generator_state": torch_generator.get_state(),
                "torch_global_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_states": (
                    [value.cpu() for value in torch.cuda.get_rng_state_all()]
                    if device.type == "cuda"
                    else []
                ),
                "numpy_generator_state": numpy_generator.bit_generator.state,
                "program_generator_state": program_rng.bit_generator.state,
                "python_random_state": random.getstate(),
            },
        }

    def write_progress(step: int, *, status: str) -> None:
        _atomic_json(
            progress_path,
            {
                "schema_version": "phase1_ugi_morphology_progress.v2",
                "status": status,
                "latest_loss": losses[-1] if losses else None,
                "evaluations": evaluations,
                "selection": {
                    "fixed_validation_corruption_seed": seed + 20_000,
                    "label_smoothing": label_smoothing,
                    "best_calibration_loss": best_calibration_loss,
                    "best_step": best_step,
                    "evaluations_without_improvement": evaluations_without_improvement,
                    "early_stopping_patience": early_stopping_patience,
                    "early_stopping_min_delta": early_stopping_min_delta,
                    "early_stopping_minimum_steps": early_stopping_minimum_steps,
                    "stop_reason": stop_reason,
                },
                "step": step,
            },
        )

    initial_evaluation = evaluate(0)
    best_calibration_loss = float(initial_evaluation["calibration_loss"]["total"])
    initial_package = checkpoint_package(0)
    _atomic_checkpoint(checkpoint_path, initial_package)
    _atomic_checkpoint(best_checkpoint_path, initial_package)
    write_progress(0, status="running")
    model.train()
    completed_step = 0
    for step in range(1, int(runtime["steps"]) + 1):
        completed_step = step
        if expanded_mode:
            local = sample_family_balanced_records(
                corpus,
                fold="train",
                count=int(runtime["batch_size"]),
                rng=numpy_generator,
                id_prefix=f"train-{step:08d}",
            )
        else:
            indices = numpy_generator.choice(
                len(train_records),
                size=int(runtime["batch_size"]),
                replace=True,
                p=product_weights,
            )
            local = tuple(train_records[int(index)] for index in indices)
        batch = _move(
            collate_ugi_morphology_records(
                local,
                maximum_nodes=max(record.program.node_count for record in local),
                maximum_children=maximum_children,
            ),
            device,
        )
        t = torch.rand(len(local), generator=torch_generator, device=device).clamp(0.02, 0.98)
        noisy = noise_ugi_morphology_batch(batch, source_tensors, t, torch_generator)
        predictions = model(
            noisy["offspring"],
            batch["role_states"],
            batch["within_role_positions"],
            batch["programs"],
            t,
            batch["node_mask"],
        )
        loss, components = ugi_morphology_flow_loss(
            predictions,
            batch,
            label_smoothing=label_smoothing,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(runtime["gradient_clip_norm"]),
        )
        optimizer.step()
        losses.append({"step": step, **components, "gradient_norm": float(gradient_norm.detach())})
        if step % int(runtime["eval_every"]) == 0 or step == int(runtime["steps"]):
            model.eval()
            evaluation = evaluate(step)
            calibration_loss = float(evaluation["calibration_loss"]["total"])
            improved = calibration_loss < best_calibration_loss - early_stopping_min_delta
            if improved:
                best_calibration_loss = calibration_loss
                best_step = step
                evaluations_without_improvement = 0
            elif step >= early_stopping_minimum_steps:
                evaluations_without_improvement += 1
            package = checkpoint_package(step)
            _atomic_checkpoint(checkpoint_path, package)
            if improved:
                _atomic_checkpoint(best_checkpoint_path, package)
            should_stop = (
                early_stopping_patience > 0
                and step >= early_stopping_minimum_steps
                and evaluations_without_improvement >= early_stopping_patience
            )
            if should_stop:
                stop_reason = "calibration_early_stopping"
            terminal = should_stop or step == int(runtime["steps"])
            write_progress(step, status="complete" if terminal else "running")
            model.train()
            if should_stop:
                break

    ledger_path = output_dir / "component_program_ledger.json"
    _atomic_json(ledger_path, component_program_ledger(corpus))
    result = {
        "schema_version": "phase1_ugi_morphology_training.v2",
        "status": "complete",
        "mode": mode,
        "scope": (
            "Ugi-core-conditioned atom-resolution topology flow over three generated precursor "
            "exteriors; chemistry, closure endpoints, L2 routes, oracles, and guidance deferred"
        ),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "inputs": inputs,
        "implementation": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in sorted(_IMPLEMENTATION_PATHS.items())
        },
        "data": {
            **data_summary,
            "component_ids_in_model_state": False,
            "clean_graph_distances_in_topology_model": False,
        },
        "optimization": {
            "initial": losses[0],
            "final": losses[-1],
            "requested_steps": int(runtime["steps"]),
            "completed_steps": completed_step,
            "total_loss_reduction_fraction": (
                (losses[0]["total"] - losses[-1]["total"]) / losses[0]["total"]
            ),
        },
        "selection": {
            "fixed_validation_corruption_seed": seed + 20_000,
            "label_smoothing": label_smoothing,
            "best_calibration_loss": best_calibration_loss,
            "best_step": best_step,
            "stop_reason": stop_reason or "maximum_steps",
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "early_stopping_minimum_steps": early_stopping_minimum_steps,
        },
        "evaluations": evaluations,
        "checkpoint": {
            "path": str(best_checkpoint_path),
            "sha256": sha256_file(best_checkpoint_path),
        },
        "checkpoint_latest": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "component_program_ledger": {
            "path": str(ledger_path),
            "sha256": sha256_file(ledger_path),
        },
    }
    _atomic_json(output_dir / "result.json", result)
    return result
