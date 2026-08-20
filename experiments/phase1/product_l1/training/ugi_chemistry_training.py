"""Deterministic bounded training loop for Ugi chemistry realization."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from experiments.phase1.product_l1.training.ugi_training_cache import load_ugi_training_cache
from forge.core.io import write_json as _atomic_json
from forge.corpus.ugi_chemistry_corpus import (
    UgiChemistryRecord,
    load_expanded_ugi_chemistry_corpus,
    load_ugi_chemistry_corpus,
)
from forge.corpus.ugi_morphology_corpus import (
    balanced_product_weights,
    source_stratified_family_weights,
)
from forge.model.defog_feasibility import sha256_file
from forge.model.ugi_chemistry_flow import (
    UgiChemistryFlow,
    UgiChemistryFlowError,
    UgiChemistrySample,
    chemistry_sample_statistics,
    chemistry_sample_to_molecule,
    chemistry_source_marginals,
    collate_ugi_chemistry_records,
    noise_ugi_chemistry_batch,
    sample_ugi_chemistry,
    ugi_chemistry_flow_loss,
)
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None


class UgiChemistryTrainingError(RuntimeError):
    """Raised when chemistry training violates its frozen contract."""


_IMPLEMENTATION_PATHS = {
    "corpus": Path(__file__).with_name("ugi_chemistry_corpus.py").resolve(),
    "flow": Path(__file__).with_name("ugi_chemistry_flow.py").resolve(),
    "interface": Path(__file__).with_name("ugi_chemistry_interface.py").resolve(),
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


def _resolve_input(repo: Path, record: dict[str, Any], label: str) -> tuple[Path, dict[str, str]]:
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = repo / path
    if not path.is_file():
        raise UgiChemistryTrainingError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != str(record["sha256"]):
        raise UgiChemistryTrainingError(
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


def _coverage_subset_indices(
    assignments: tuple[Any, ...],
    *,
    limit: int,
) -> np.ndarray:
    """Select a deterministic bounded set that covers precursor identities early."""

    if limit < 1:
        raise UgiChemistryTrainingError("bounded chemistry subset must be nonempty")
    unseen = {(role, row[f"{role}_smiles"]) for row in assignments for role in ROLE_NAMES}
    selected = []
    selected_set = set()
    for index, row in enumerate(assignments):
        local = {(role, row[f"{role}_smiles"]) for role in ROLE_NAMES}
        if local & unseen:
            selected.append(index)
            selected_set.add(index)
            unseen.difference_update(local)
        if len(selected) >= limit:
            break
    if len(selected) < min(limit, len(assignments)):
        for index in range(len(assignments)):
            if index not in selected_set:
                selected.append(index)
            if len(selected) >= limit:
                break
    return np.asarray(selected, dtype=np.int64)


def _source_stratified_coverage_subset_indices(
    assignments: tuple[Any, ...],
    *,
    limit: int,
) -> np.ndarray:
    """Bound a smoke corpus without silently dropping a provenance stratum."""

    sources = sorted({str(row.get("source_stratum") or "unspecified") for row in assignments})
    if len(sources) == 1:
        return _coverage_subset_indices(assignments, limit=limit)
    if limit < len(sources):
        raise UgiChemistryTrainingError("bounded subset is smaller than source count")
    base, remainder = divmod(min(limit, len(assignments)), len(sources))
    selected: list[int] = []
    for source_index, source in enumerate(sources):
        global_indices = [
            index
            for index, row in enumerate(assignments)
            if str(row.get("source_stratum") or "unspecified") == source
        ]
        quota = min(len(global_indices), base + int(source_index < remainder))
        local_assignments = tuple(assignments[index] for index in global_indices)
        local_indices = _coverage_subset_indices(local_assignments, limit=quota)
        selected.extend(global_indices[int(index)] for index in local_indices)
    return np.asarray(sorted(selected), dtype=np.int64)


def _training_weights(
    assignments: tuple[Any, ...],
    sampling: dict[str, Any],
) -> np.ndarray:
    mode = sampling.get("mode", "role_component_raked")
    if mode == "role_component_raked":
        return balanced_product_weights(assignments)
    if mode == "source_stratified_role_family_raked":
        source_mass = sampling.get("source_mass")
        return source_stratified_family_weights(
            assignments,
            source_mass=(
                {str(key): float(value) for key, value in source_mass.items()}
                if source_mass is not None
                else None
            ),
            uniform_row_mixture=float(sampling.get("uniform_row_mixture", 0.5)),
        )
    raise UgiChemistryTrainingError(f"unsupported training sampling mode: {mode}")


def _loss_on_records(
    model: Any,
    records: tuple[UgiChemistryRecord, ...],
    sources: dict[str, Any],
    *,
    batch_size: int,
    batches: int,
    seed: int,
    device: Any,
    maximum_decorations: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    metrics: dict[str, list[float]] = {}
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            indices = rng.choice(
                len(records),
                size=min(batch_size, len(records)),
                replace=False,
            )
            local = tuple(records[int(index)] for index in indices)
            clean = _move(
                collate_ugi_chemistry_records(
                    local,
                    maximum_nodes=max(record.condition.node_count for record in local),
                    maximum_closures=max(record.condition.closure_count for record in local),
                    maximum_decorations=maximum_decorations,
                ),
                device,
            )
            t = torch.rand(len(local), generator=generator, device=device).clamp(0.02, 0.98)
            noisy = noise_ugi_chemistry_batch(clean, sources, t, generator)
            predictions = model(
                nodes=noisy["nodes"],
                parent_bonds=noisy["parent_bonds"],
                closure_bonds=noisy["closure_bonds"],
                t=t,
                topology=clean,
            )
            _, local_metrics = ugi_chemistry_flow_loss(predictions, clean)
            for key, value in local_metrics.items():
                metrics.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def _records_by_source(
    records: tuple[UgiChemistryRecord, ...],
    assignments: tuple[Any, ...],
) -> dict[str, tuple[UgiChemistryRecord, ...]]:
    grouped: dict[str, list[UgiChemistryRecord]] = {}
    for record, assignment in zip(records, assignments, strict=True):
        source = str(assignment.get("source_stratum") or "unspecified")
        grouped.setdefault(source, []).append(record)
    return {source: tuple(values) for source, values in sorted(grouped.items())}


def _macro_average_metrics(values: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = set.intersection(*(set(metrics) for metrics in values.values()))
    return {
        key: float(np.mean([metrics[key] for metrics in values.values()])) for key in sorted(keys)
    }


def _target_smiles(record: UgiChemistryRecord, vocabulary: tuple[Any, ...]) -> str:
    target = record.target
    sample = UgiChemistrySample(
        atom_states=target.atom_states.copy(),
        parent_bond_states=target.parent_bond_states.copy(),
        closure_bond_states=target.closure_bond_states.copy(),
        decoration_anchor=0,
        # Always use the typed sparse-decoration representation for the
        # target.  The legacy singular path means specifically O=; it is not
        # correct for singleton F-, O-, or S- decorations in the expanded
        # corpus.
        decoration_anchors=target.decorations.anchor_indices.copy() + 1,
        decoration_atom_states=target.decorations.atom_states.copy(),
        decoration_bond_states=target.decorations.bond_states.copy(),
    )
    molecule = chemistry_sample_to_molecule(record.condition, sample, vocabulary)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _sample_metrics(
    records: tuple[UgiChemistryRecord, ...],
    samples: list[UgiChemistrySample],
    vocabulary: tuple[Any, ...],
) -> dict[str, Any]:
    statistics = chemistry_sample_statistics(
        tuple(record.condition for record in records),
        samples,
        vocabulary,
    )
    exact = 0
    generated_smiles = []
    for record, sample in zip(records, samples, strict=True):
        try:
            molecule = chemistry_sample_to_molecule(record.condition, sample, vocabulary)
            observed = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        except (ValueError, RuntimeError):
            observed = None
        generated_smiles.append(observed)
        exact += int(observed == _target_smiles(record, vocabulary))
    statistics["exact_target_matches"] = exact
    statistics["exact_target_fraction"] = exact / len(records)
    statistics["generated_smiles_aligned"] = generated_smiles
    return statistics


def train_ugi_chemistry(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    smoke: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Train the exact-topology chemistry gate before end-to-end composition."""

    if torch is None:
        raise UgiChemistryTrainingError("Ugi chemistry training requires torch")
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UgiChemistryTrainingError(f"invalid chemistry config: {exc}") from exc
    schema_version = config.get("schema_version")
    if schema_version not in {
        "phase1_ugi_chemistry_training_config.v1",
        "phase1_ugi_chemistry_training_config.v2",
    }:
        raise UgiChemistryTrainingError("unsupported Ugi chemistry training config")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise UgiChemistryTrainingError(f"output directory is nonempty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {}
    resolved = {}
    for label, record in config["inputs"].items():
        resolved[label], inputs[label] = _resolve_input(repo, record, label)
    if "prepared_cache" in resolved:
        corpus, _ = load_ugi_training_cache(resolved["prepared_cache"])
        observed_fold_counts = {
            fold: len(records) for fold, records in corpus.records_by_fold.items()
        }
        if observed_fold_counts != config["expected_fold_counts"]:
            raise UgiChemistryTrainingError(
                f"prepared chemistry fold counts changed: {observed_fold_counts}"
            )
        checkpoint_schema = "phase1_ugi_chemistry_checkpoint.v2"
        result_schema = "phase1_ugi_chemistry_training_result.v2"
    elif schema_version == "phase1_ugi_chemistry_training_config.v2":
        corpus = load_expanded_ugi_chemistry_corpus(
            resolved["assignments"],
            resolved["semantic_products"],
            resolved["semantic_atoms"],
            resolved["atom_vocabulary"],
        )
        observed_fold_counts = {
            fold: len(records) for fold, records in corpus.records_by_fold.items()
        }
        if observed_fold_counts != config["expected_fold_counts"]:
            raise UgiChemistryTrainingError(
                f"expanded chemistry fold counts changed: {observed_fold_counts}"
            )
        checkpoint_schema = "phase1_ugi_chemistry_checkpoint.v2"
        result_schema = "phase1_ugi_chemistry_training_result.v2"
    else:
        corpus = load_ugi_chemistry_corpus(
            resolved["assignments"],
            resolved["semantic_products"],
            resolved["semantic_atoms"],
            resolved["atom_vocabulary"],
        )
        checkpoint_schema = "phase1_ugi_chemistry_checkpoint.v1"
        result_schema = "phase1_ugi_chemistry_training_result.v1"
    mode = "smoke" if smoke else "full"
    runtime = dict(config[mode])
    model_config = dict(config["model"])
    if smoke:
        model_config.update(runtime.pop("model_overrides"))
    device = torch.device(str(runtime["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise UgiChemistryTrainingError("CUDA requested but unavailable")
    seed = int(config["seed"])
    _set_determinism(seed, device)

    train_all = corpus.records_by_fold["train"]
    train_assignments_all = corpus.assignments_by_fold["train"]
    calibration_all = corpus.records_by_fold["calibration"]
    calibration_assignments_all = corpus.assignments_by_fold["calibration"]
    if smoke:
        indices = _source_stratified_coverage_subset_indices(
            train_assignments_all,
            limit=int(runtime["bounded_train_records"]),
        )
        train_records = tuple(train_all[int(index)] for index in indices)
        train_assignments = tuple(train_assignments_all[int(index)] for index in indices)
    else:
        train_records = train_all
        train_assignments = train_assignments_all
    calibration_indices = _coverage_subset_indices(
        calibration_assignments_all,
        limit=min(int(runtime["eval_samples"]), len(calibration_all)),
    )
    calibration_samples = tuple(calibration_all[int(index)] for index in calibration_indices)
    calibration_by_source = _records_by_source(
        calibration_all,
        calibration_assignments_all,
    )
    sampling = dict(config.get("sampling", {}))
    product_weights = _training_weights(train_assignments, sampling)
    source_arrays = chemistry_source_marginals(
        train_records,
        atom_classes=len(corpus.atom_vocabulary),
        bond_classes=int(model_config["bond_classes"]),
        probability_floor=float(model_config["source_probability_floor"]),
        record_weights=product_weights,
        maximum_decorations=int(model_config.get("maximum_decorations", 1)),
    )
    sources = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in source_arrays.items()
    }
    model = UgiChemistryFlow(
        atom_classes=len(corpus.atom_vocabulary),
        bond_classes=int(model_config["bond_classes"]),
        maximum_nodes=int(model_config["maximum_nodes"]),
        maximum_distance=int(model_config["maximum_distance"]),
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["layers"]),
        dropout=float(model_config["dropout"]),
        maximum_decorations=int(model_config.get("maximum_decorations", 1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(runtime["learning_rate"]),
        weight_decay=float(runtime["weight_decay"]),
    )
    torch_generator = torch.Generator(device=device).manual_seed(seed + 1)
    numpy_generator = np.random.default_rng(seed + 2)
    losses = []
    evaluations = []
    early_stopping = runtime.get("early_stopping", {})
    early_stopping_patience = int(early_stopping.get("patience", 0))
    early_stopping_min_delta = float(early_stopping.get("min_delta", 0.0))
    early_stopping_minimum_steps = int(early_stopping.get("minimum_steps", 0))
    if (
        early_stopping_patience < 0
        or early_stopping_min_delta < 0
        or early_stopping_minimum_steps < 0
    ):
        raise UgiChemistryTrainingError("invalid chemistry early-stopping policy")
    latest_checkpoint_path = output_dir / "checkpoint_latest.pt"
    best_checkpoint_path = output_dir / "checkpoint_best.pt"
    best_calibration_loss = float("inf")
    best_step = 0
    evaluations_without_improvement = 0
    stop_reason = ""

    def evaluate(step: int) -> dict[str, Any]:
        try:
            samples, sampling = sample_ugi_chemistry(
                model,
                tuple(record.condition for record in calibration_samples),
                source_arrays,
                atom_classes=len(corpus.atom_vocabulary),
                sample_steps=int(runtime["sample_steps"]),
                batch_size=int(runtime["batch_size"]),
                # Fixed records and RNG make serial checkpoint comparisons causal.
                seed=seed + 10_000,
                device=str(device),
                atom_vocabulary=corpus.atom_vocabulary,
            )
            generated = _sample_metrics(
                calibration_samples,
                samples,
                corpus.atom_vocabulary,
            )
        except UgiChemistryFlowError as exc:
            # A decoder support failure is a failed generation metric, not a
            # reason to discard the loss trajectory or abort checkpointing.
            sampling = {
                "samples": len(calibration_samples),
                "terminal_decoder": (
                    "valence_constrained_categorical_argmax_without_structural_repair"
                ),
                "terminal_constraint_failure": str(exc),
            }
            generated = {
                "samples": len(calibration_samples),
                "valid_molecules": 0,
                "valid_fraction": 0.0,
                "unique_valid_molecules": 0,
                "terminal_constraint_failure": str(exc),
            }
        calibration_loss_by_source = {
            source: _loss_on_records(
                model,
                records,
                sources,
                batch_size=int(runtime["batch_size"]),
                batches=int(runtime["validation_batches"]),
                seed=seed + 30_000 + source_index,
                device=device,
                maximum_decorations=int(model_config.get("maximum_decorations", 1)),
            )
            for source_index, (source, records) in enumerate(calibration_by_source.items())
        }
        evaluation = {
            "step": step,
            "training_subset_loss": _loss_on_records(
                model,
                train_records,
                sources,
                batch_size=int(runtime["batch_size"]),
                batches=int(runtime["validation_batches"]),
                seed=seed + 20_000,
                device=device,
                maximum_decorations=int(model_config.get("maximum_decorations", 1)),
            ),
            "calibration_loss": _macro_average_metrics(calibration_loss_by_source),
            "calibration_loss_by_source": calibration_loss_by_source,
            "sampling": sampling,
            "generated": generated,
            "calibration_product_ids": [record.product_id for record in calibration_samples],
        }
        evaluations.append(evaluation)
        return evaluation

    def checkpoint_package(step: int) -> dict[str, Any]:
        return {
            "schema_version": checkpoint_schema,
            "step": step,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "source_marginals": source_arrays,
            "inputs": inputs,
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
                "python_random_state": random.getstate(),
            },
        }

    def write_progress(step: int, status: str) -> None:
        _atomic_json(
            output_dir / "progress.json",
            {
                "schema_version": f"{result_schema}.progress",
                "status": status,
                "step": step,
                "losses": losses,
                "evaluations": evaluations,
                "selection": {
                    "fixed_calibration_corruption_seed": seed + 30_000,
                    "fixed_calibration_sampling_seed": seed + 10_000,
                    "best_calibration_loss": best_calibration_loss,
                    "best_step": best_step,
                    "evaluations_without_improvement": evaluations_without_improvement,
                    "early_stopping_patience": early_stopping_patience,
                    "early_stopping_min_delta": early_stopping_min_delta,
                    "early_stopping_minimum_steps": early_stopping_minimum_steps,
                    "stop_reason": stop_reason,
                },
            },
        )

    initial_evaluation = evaluate(0)
    best_calibration_loss = float(initial_evaluation["calibration_loss"]["total"])
    initial_package = checkpoint_package(0)
    _atomic_checkpoint(latest_checkpoint_path, initial_package)
    _atomic_checkpoint(best_checkpoint_path, initial_package)
    write_progress(0, "running")
    model.train()
    completed_step = 0
    for step in range(1, int(runtime["steps"]) + 1):
        completed_step = step
        indices = numpy_generator.choice(
            len(train_records),
            size=int(runtime["batch_size"]),
            replace=True,
            p=product_weights,
        )
        local = tuple(train_records[int(index)] for index in indices)
        clean = _move(
            collate_ugi_chemistry_records(
                local,
                maximum_nodes=max(record.condition.node_count for record in local),
                maximum_closures=max(record.condition.closure_count for record in local),
                maximum_decorations=int(model_config.get("maximum_decorations", 1)),
            ),
            device,
        )
        t = torch.rand(len(local), generator=torch_generator, device=device).clamp(0.02, 0.98)
        noisy = noise_ugi_chemistry_batch(clean, sources, t, torch_generator)
        predictions = model(
            nodes=noisy["nodes"],
            parent_bonds=noisy["parent_bonds"],
            closure_bonds=noisy["closure_bonds"],
            t=t,
            topology=clean,
        )
        loss, components = ugi_chemistry_flow_loss(predictions, clean)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(runtime["gradient_clip_norm"]),
        )
        optimizer.step()
        losses.append(
            {
                "step": step,
                **components,
                "gradient_norm": float(gradient_norm.detach()),
            }
        )
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
            _atomic_checkpoint(latest_checkpoint_path, package)
            if improved:
                _atomic_checkpoint(best_checkpoint_path, package)
            should_stop = (
                early_stopping_patience > 0
                and step >= early_stopping_minimum_steps
                and evaluations_without_improvement >= early_stopping_patience
            )
            if should_stop:
                stop_reason = "calibration_early_stopping"
            write_progress(
                step,
                "complete" if should_stop or step == int(runtime["steps"]) else "running",
            )
            model.train()
            if should_stop:
                break

    result = {
        "schema_version": result_schema,
        "status": "complete",
        "mode": mode,
        "seed": seed,
        "inputs": inputs,
        "implementation": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in sorted(_IMPLEMENTATION_PATHS.items())
        },
        "corpus": {
            "strict_fold_counts": {
                fold: len(records) for fold, records in corpus.records_by_fold.items()
            },
            "training_records_used": len(train_records),
            "calibration_records": len(calibration_all),
            "fixed_calibration_sampling_records": len(calibration_samples),
            "atom_classes": len(corpus.atom_vocabulary),
            "fixed_core_schema": {
                "atom_states": list(corpus.core_schema.atom_state_by_core_position),
                "bonds": [list(row) for row in corpus.core_schema.bond_state_by_core_positions],
            },
        },
        "model": model_config,
        "runtime": runtime,
        "sampling": {
            **sampling,
            "observed_source_mass": {
                source: float(
                    product_weights[
                        np.asarray(
                            [
                                index
                                for index, row in enumerate(train_assignments)
                                if str(row.get("source_stratum") or "unspecified") == source
                            ],
                            dtype=np.int64,
                        )
                    ].sum()
                )
                for source in sorted(
                    {str(row.get("source_stratum") or "unspecified") for row in train_assignments}
                )
            },
        },
        "source_marginals": {key: value.tolist() for key, value in source_arrays.items()},
        "losses": losses,
        "evaluations": evaluations,
        "selection": {
            "fixed_calibration_corruption_seed": seed + 30_000,
            "fixed_calibration_sampling_seed": seed + 10_000,
            "best_calibration_loss": best_calibration_loss,
            "best_step": best_step,
            "completed_steps": completed_step,
            "stop_reason": stop_reason or "maximum_steps",
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "early_stopping_minimum_steps": early_stopping_minimum_steps,
        },
        "checkpoint": {
            "path": str(best_checkpoint_path),
            "sha256": sha256_file(best_checkpoint_path),
        },
        "checkpoint_latest": {
            "path": str(latest_checkpoint_path),
            "sha256": sha256_file(latest_checkpoint_path),
        },
        "boundary": {
            "topology": "exact_condition_for_bounded_chemistry_gate",
            "terminal_decoder": "categorical_argmax_without_structural_repair",
            "closure_endpoints": "conditioned_not_generated_in_this_gate",
            "route_oracle_or_guidance": False,
        },
    }
    _atomic_json(output_dir / "result.json", result)
    return result
