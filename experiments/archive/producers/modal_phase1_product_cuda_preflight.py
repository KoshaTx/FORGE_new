#!/usr/bin/env python3
"""Run the bounded Phase 1 sparse whole-lipid CUDA preflight on Modal."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
REMOTE_SOURCE_ROOT = Path("/root")
DEFAULT_CONFIG = REPO_ROOT / "configs/model/phase1_product_cuda_preflight.json"
DEFAULT_OUTPUT = REPO_ROOT / "results/phase1/product_cuda_preflight.json"
GPU_CLASS = "L4"
REMOTE_SOURCE_FILES = (
    "forge/__init__.py",
    "forge/data/__init__.py",
    "forge/data/r0_splits.py",
    "forge/product/__init__.py",
    "forge/product/defog_feasibility.py",
    "forge/product/lipid_context.py",
    "forge/product/phase1_flow.py",
    "forge/product/sparse_topology_feasibility.py",
)

app = modal.App("forge-phase1-product-cuda-preflight")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.11.0",
    "numpy==2.4.2",
    "rdkit==2025.9.6",
)
for relative_source in REMOTE_SOURCE_FILES:
    image = image.add_local_file(
        SOURCE_ROOT / relative_source,
        remote_path=str(REMOTE_SOURCE_ROOT / relative_source),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _resolve_declared_input(
    config: dict[str, Any],
    key: str,
    label: str,
    config_path: Path,
) -> Path:
    record = config.get(key)
    if not isinstance(record, dict) or not {"path", "sha256"}.issubset(record):
        raise ValueError(f"{key} must declare path and sha256")
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = REPO_ROOT / path
    observed = _sha256_file(path)
    if observed != str(record["sha256"]):
        raise ValueError(
            f"{label} SHA-256 mismatch: "
            f"expected {record['sha256']}, observed {observed}; update {config_path}"
        )
    return path


@app.function(
    image=image,
    gpu=GPU_CLASS,
    cpu=2.0,
    memory=8192,
    timeout=1200,
)
def run_cuda_preflight(payload: dict[str, Any]) -> dict[str, Any]:
    """Exercise production tensorization and training twice on one CUDA device."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(
        payload["preflight"]["runtime"]["cublas_workspace_config"]
    )
    sys.path.insert(0, str(REMOTE_SOURCE_ROOT))

    import numpy as np
    import rdkit
    import torch

    from forge.model.defog_feasibility import AtomState, _model_state_sha256, _parameter_count
    from forge.model.phase1_flow import (
        SparseWholeLipidFlow,
        _checkpoint_package,
        _compact_training_record,
        _degree_continuation_log_prior,
        _initialize_closure_ring_size_prior,
        _initialize_parent_distance_prior,
        _load_resume_checkpoint,
        _model_architecture_kwargs,
        _region_atom_marginal,
        _region_marginal,
        _save_checkpoint_atomic,
        _train_one_step,
    )
    from forge.model.sparse_topology_feasibility import (
        sparse_constitutional_roundtrip_exact,
        sparse_roundtrip_exact,
        tensorize_sparse_row,
    )

    preflight = payload["preflight"]
    product = payload["product"]
    if str(preflight["required_gpu_class"]) != GPU_CLASS:
        raise RuntimeError("preflight GPU class disagrees with the Modal function declaration")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the requested GPU container")
    if str(product["execution"]["precision"]) != "float32":
        raise RuntimeError("Phase 1 product precision is no longer float32")
    if product["execution"]["mixed_precision"] is not False:
        raise RuntimeError("Phase 1 product mixed precision must remain disabled")
    if product["execution"]["deterministic_algorithms"] is not True:
        raise RuntimeError("Phase 1 product deterministic-algorithm gate was weakened")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    vocabulary = tuple(
        AtomState(
            str(record["symbol"]),
            int(record["formal_charge"]),
            bool(record["aromatic"]),
            int(record.get("explicit_hydrogens", 0)),
        )
        for record in payload["atom_vocabulary"]
    )
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}
    records = []
    record_audit = []
    for sample in preflight["samples"]:
        sparse = tensorize_sparse_row(
            {
                "r0_structure_id": str(sample["structure_id"]),
                "canonical_isomeric_smiles": str(sample["canonical_isomeric_smiles"]),
            },
            atom_to_index,
            preserve_aromaticity=bool(product["model"].get("preserve_aromaticity", False)),
            root_strategy=str(product["model"].get("root_strategy", "canonical")),
            region_scheme=str(product["model"].get("region_scheme", "none")),
        )
        exact_sparse = sparse_roundtrip_exact(sparse)
        exact_constitution = sparse_constitutional_roundtrip_exact(sparse, vocabulary)
        if not exact_sparse or not exact_constitution:
            raise RuntimeError(f"preflight tensorization failed for {sparse.structure_id}")
        records.append(_compact_training_record(sparse))
        record_audit.append(
            {
                "structure_id": sparse.structure_id,
                "heavy_atoms": sparse.node_count,
                "closures": sparse.closure_count,
                "exact_sparse_roundtrip": exact_sparse,
                "exact_constitutional_roundtrip": exact_constitution,
            }
        )
    total_closures = sum(record.closure_count for record in records)
    minimum_closures = int(preflight["acceptance"]["minimum_total_closures"])
    if total_closures < minimum_closures:
        raise RuntimeError(
            f"preflight has {total_closures} closures, below the required {minimum_closures}"
        )

    model_config = product["model"]
    training_config = product["training"]
    # Positive-smoothed marginals exercise every frozen node and bond state.
    node_p0 = torch.full(
        (len(vocabulary),),
        1.0 / len(vocabulary),
        dtype=torch.float32,
        device=device,
    )
    bond_classes = int(model_config.get("bond_classes", 3))
    bond_p0 = torch.full(
        (bond_classes,),
        1.0 / bond_classes,
        dtype=torch.float32,
        device=device,
    )
    region_classes = int(model_config.get("region_classes", 0))
    region_marginal = _region_marginal(tuple(records), region_classes)
    region_atom_marginal = _region_atom_marginal(
        tuple(records),
        len(vocabulary),
        region_classes,
    )
    degree_continuation_log_prior = _degree_continuation_log_prior(
        tuple(records),
        region_classes=region_classes,
        maximum_degree=int(model_config.get("degree_prior_maximum_degree", 0)),
        probability_floor=float(model_config.get("degree_continuation_probability_floor", 0.01)),
    )
    region_p0 = (
        torch.tensor(region_marginal, dtype=torch.float32, device=device)
        if region_marginal is not None
        else None
    )
    region_atom_p0 = (
        torch.tensor(region_atom_marginal, dtype=torch.float32, device=device)
        if region_atom_marginal is not None
        else None
    )

    def initialize(seed: int) -> tuple[Any, Any, Any, Any]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        model = SparseWholeLipidFlow(
            node_classes=len(vocabulary),
            hidden_dim=int(model_config["hidden_dim"]),
            layers=int(model_config["layers"]),
            maximum_closures=int(model_config["maximum_closure_slots"]),
            maximum_heavy_atoms=int(model_config["maximum_heavy_atoms"]),
            dropout=float(model_config["dropout"]),
            **_model_architecture_kwargs(model_config),
        ).to(device)
        _initialize_parent_distance_prior(model, tuple(records), model_config)
        _initialize_closure_ring_size_prior(model, tuple(records), model_config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
        generator = torch.Generator(device=device).manual_seed(seed + 2)
        rng = np.random.default_rng(seed + 3)
        return model, optimizer, generator, rng

    def train_steps(
        model: Any,
        optimizer: Any,
        generator: Any,
        count: int,
        history: list[dict[str, float]],
    ) -> None:
        for _ in range(count):
            metrics = _train_one_step(
                model,
                tuple(records),
                optimizer,
                device=device,
                maximum_closures=int(model_config["maximum_closure_slots"]),
                node_marginal=node_p0,
                bond_marginal=bond_p0,
                region_marginal=region_p0,
                region_atom_marginal=region_atom_p0,
                generator=generator,
                gradient_clip_norm=float(training_config["gradient_clip_norm"]),
            )
            if not all(np.isfinite(value) for value in metrics.values()):
                raise RuntimeError(f"non-finite CUDA training metric: {metrics}")
            history.append(metrics)

    def summarize(model: Any, history: list[dict[str, float]]) -> dict[str, Any]:
        torch.cuda.synchronize(device)
        return {
            "loss_history": history,
            "model_state_sha256": _model_state_sha256(model),
            "parameters": _parameter_count(model),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }

    def run_once() -> dict[str, Any]:
        model, optimizer, generator, _ = initialize(int(preflight["seed"]))
        torch.cuda.reset_peak_memory_stats(device)
        history: list[dict[str, float]] = []
        train_steps(
            model,
            optimizer,
            generator,
            int(preflight["steps"]),
            history,
        )
        return summarize(model, history)

    def run_with_resume() -> dict[str, Any]:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        seed = int(preflight["seed"])
        split_step = max(1, int(preflight["steps"]) // 2)
        model, optimizer, generator, rng = initialize(seed)
        torch.cuda.reset_peak_memory_stats(device)
        history: list[dict[str, float]] = []
        train_steps(model, optimizer, generator, split_step, history)
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            checkpoint = output_dir / "checkpoint.pt"
            package = _checkpoint_package(
                config_sha256="a" * 64,
                data_manifest_sha256="b" * 64,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                atom_vocabulary=vocabulary,
                node_marginal=node_p0.detach().cpu().numpy(),
                bond_marginal=bond_p0.detach().cpu().numpy(),
                step=split_step,
                generator=generator,
                rng=rng,
                losses=history,
                source_totals={"r0": split_step * len(records)},
                bucket_totals={"preflight": split_step * len(records)},
                examples_seen=split_step * len(records),
                validation_history=(),
                best_validation_step=0,
                best_validation_loss=float("inf"),
                elapsed_wall_seconds=0.0,
                region_marginal=region_marginal,
                region_atom_marginal=region_atom_marginal,
                degree_continuation_log_prior=degree_continuation_log_prior,
            )
            _save_checkpoint_atomic(checkpoint, package)

            resumed_model, resumed_optimizer, resumed_generator, resumed_rng = initialize(
                seed + 997
            )
            restored = _load_resume_checkpoint(
                checkpoint,
                output_dir=output_dir,
                config_sha256="a" * 64,
                data_manifest_sha256="b" * 64,
                model=resumed_model,
                optimizer=resumed_optimizer,
                device=device,
                generator=resumed_generator,
                rng=resumed_rng,
            )
            resumed_history = [
                {str(key): float(value) for key, value in row.items()}
                for row in restored["training_state"]["losses"]
            ]
            train_steps(
                resumed_model,
                resumed_optimizer,
                resumed_generator,
                int(preflight["steps"]) - split_step,
                resumed_history,
            )
            summary = summarize(resumed_model, resumed_history)
            summary["checkpoint_step"] = split_step
            summary["checkpoint_model_state_sha256"] = package["model_state_sha256"]
            return summary

    repetitions = [run_once() for _ in range(int(preflight["repetitions"]))]
    reference = repetitions[0]
    exact_losses = all(
        repetition["loss_history"] == reference["loss_history"] for repetition in repetitions[1:]
    )
    exact_states = all(
        repetition["model_state_sha256"] == reference["model_state_sha256"]
        for repetition in repetitions[1:]
    )
    if preflight["acceptance"]["require_exact_repeat_loss_history"] and not exact_losses:
        raise RuntimeError("deterministic CUDA repetitions produced different loss histories")
    if preflight["acceptance"]["require_exact_repeat_model_state"] and not exact_states:
        raise RuntimeError("deterministic CUDA repetitions produced different model states")
    resumed = run_with_resume()
    exact_resume_losses = resumed["loss_history"] == reference["loss_history"]
    exact_resume_state = resumed["model_state_sha256"] == reference["model_state_sha256"]
    if preflight["acceptance"]["require_exact_resume_loss_history"] and not exact_resume_losses:
        raise RuntimeError("CUDA checkpoint resume changed the training loss history")
    if preflight["acceptance"]["require_exact_resume_model_state"] and not exact_resume_state:
        raise RuntimeError("CUDA checkpoint resume changed the final model state")

    properties = torch.cuda.get_device_properties(device)
    return {
        "status": "pass",
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(properties.total_memory),
            "requested_class": GPU_CLASS,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rdkit": rdkit.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": int(torch.backends.cudnn.version()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "precision": "float32",
            "mixed_precision": False,
        },
        "samples": record_audit,
        "total_closures": total_closures,
        "repetitions": repetitions,
        "resumed_run": resumed,
        "determinism": {
            "exact_loss_history": exact_losses,
            "exact_model_state": exact_states,
            "exact_resume_loss_history": exact_resume_losses,
            "exact_resume_model_state": exact_resume_state,
        },
    }


@app.local_entrypoint()
def main(
    config_path: str = str(DEFAULT_CONFIG),
    output_path: str = str(DEFAULT_OUTPUT),
) -> None:
    config_file = Path(config_path).resolve()
    output_file = Path(output_path).resolve()
    preflight = _load_object(config_file, "CUDA preflight config")
    if preflight.get("schema_version") != "phase1_product_cuda_preflight_config.v1":
        raise ValueError("unsupported CUDA preflight config schema")
    product_path = _resolve_declared_input(
        preflight,
        "product_config",
        "Phase 1 product config",
        config_file,
    )
    vocabulary_path = _resolve_declared_input(
        preflight,
        "atom_vocabulary_source",
        "atom-vocabulary source",
        config_file,
    )
    product = _load_object(product_path, "Phase 1 product config")
    vocabulary_source = _load_object(vocabulary_path, "atom-vocabulary source")
    vocabulary = sorted(
        vocabulary_source["atom_vocabulary"],
        key=lambda record: int(record["index"]),
    )
    if [int(record["index"]) for record in vocabulary] != list(range(len(vocabulary))):
        raise ValueError("atom-vocabulary source indices must be contiguous from zero")

    remote = run_cuda_preflight.remote(
        {
            "preflight": preflight,
            "product": product,
            "atom_vocabulary": vocabulary,
        }
    )
    result = {
        "schema_version": "phase1_product_cuda_preflight_result.v1",
        "task": preflight["task"],
        "status": remote["status"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "preflight_config": {
                "path": str(config_file.relative_to(REPO_ROOT)),
                "sha256": _sha256_file(config_file),
            },
            "product_config": {
                "path": str(product_path.relative_to(REPO_ROOT)),
                "sha256": _sha256_file(product_path),
            },
            "atom_vocabulary_source": {
                "path": str(vocabulary_path.relative_to(REPO_ROOT)),
                "sha256": _sha256_file(vocabulary_path),
            },
            "mounted_sources": {
                f"src/{relative}": _sha256_file(SOURCE_ROOT / relative)
                for relative in REMOTE_SOURCE_FILES
            },
            "runner": {
                "path": "scripts/modal_phase1_product_cuda_preflight.py",
                "sha256": _sha256_file(Path(__file__)),
            },
        },
        "acceptance": preflight["acceptance"],
        "result": remote,
        "decision": {
            "cuda_training_path_authorized": True,
            "scope": (
                "kernel, device-RNG, forward, backward, optimizer, checkpoint-resume, "
                "and determinism preflight only"
            ),
            "does_not_establish": [
                "full-run convergence",
                "molecular generation quality",
                "biological activity",
                "synthesis route closure",
            ],
        },
    }
    _atomic_json(output_file, result)
    print(json.dumps(result, indent=2, sort_keys=True))
