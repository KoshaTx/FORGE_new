"""Deterministic COMPOSE training with update-boundary restart checkpoints."""

from __future__ import annotations

import fcntl
import json
import math
import os
import platform
import random
import shutil
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments._runtime.source import source_fingerprint
from forge.core.hashing import resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json
from forge.corpus.compose_lipid_training_data import (
    ComposeLipidTrainingData,
    require_training_admission,
)
from forge.model.compose_lipid_training import compose_lipid_training_step, validate_objective
from forge.model.synthesis_program_training import build_synthesis_program_flow
from forge.model.training_restart import (
    TrainingRestartError,
    atomic_torch_save,
    capture_training_random_state,
    restore_training_random_state,
)

CONFIG_SCHEMA = "forge.compose_lipid_training_config.v1"
CHECKPOINT_SCHEMA = "forge.compose_lipid_training_checkpoint.v1"
RESULT_SCHEMA = "forge.compose_lipid_training_run.v1"
DATA_INPUTS = {"population", "verification", "measure", "admission"}
RUN_SOURCES = {
    "experiments/phase1/multireaction/compose_lipid_run.py",
    "forge/model/training_restart.py",
    "experiments/phase1/multireaction/compose_lipid_training.py",
}
PARALLEL_GPUS = {"six_gpu_shards_v1": 6, "adaptive_eight_gpu_shards_v1": 8}


def read_admitted_config(repo: Path, path: Path) -> dict[str, Any]:
    """Check admission before allocating a model or contacting a compute service."""
    config = read_json_object(path, error=TrainingRestartError)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise TrainingRestartError("Unsupported COMPOSE training config")
    inputs = config.get("inputs", {})
    if set(inputs) not in (
        DATA_INPUTS | {"noise_marginals"},
        DATA_INPUTS | {"noise_marginals", "mapped_cache"},
        DATA_INPUTS | {"noise_marginals", "prepared_tensors"},
        DATA_INPUTS | {"noise_marginals", "mapped_cache", "prepared_tensors"},
    ):
        raise TrainingRestartError("Training inputs must bind data, admission and noise marginals")
    require_training_admission(repo, **{name: inputs[name] for name in DATA_INPUTS})
    admission = json.loads(resolve_pin(inputs["admission"], repo, label="admission").read_text())
    validation = json.loads(
        resolve_pin(admission["inputs"]["validation"], repo, label="validation").read_text()
    )
    snapshot = json.loads(
        resolve_pin(
            validation["inputs"]["source-snapshot.json"], repo, label="source snapshot"
        ).read_text()
    )
    if not RUN_SOURCES <= set(snapshot):
        raise TrainingRestartError("Training validation predates the restartable trainer")
    runtime = config["runtime"]
    prepared = runtime.get("prepared_loading")
    if ("prepared_tensors" in inputs) != (prepared in ("synchronous", "prefetch")) or (
        prepared is not None and prepared not in ("synchronous", "prefetch")
    ):
        raise TrainingRestartError(
            "Prepared loading requires a pinned tensor cache and explicit policy"
        )
    if prepared and (
        runtime.get("node_padding") not in ("batch", "family")
        or runtime.get("repeat_supervision") != "exact_fragment"
        or runtime.get("core_conditioning") != "qualified_core"
    ):
        raise TrainingRestartError("Prepared tensor conditioning/padding policy is unsupported")
    if prepared == "prefetch":
        if runtime.get("family_schedule") != "balanced_cycles":
            raise TrainingRestartError("Prepared prefetch requires balanced cycles")
        for key in ("prefetch_depth", "prefetch_workers"):
            if type(runtime.get(key)) is not int or runtime[key] < 1:
                raise TrainingRestartError(f"runtime.{key} must be a positive integer")
    if prepared:
        from forge.corpus.compose_lipid_tensor_cache import SCHEMA as TENSOR_CACHE_SCHEMA

        cached = read_json_object(
            resolve_pin(inputs["prepared_tensors"], repo, label="prepared tensor manifest"),
            error=TrainingRestartError,
        )
        expected_inputs = {
            k: inputs[k]
            for k in ("population", "verification", "measure", "mapped_cache")
            if k in inputs
        }
        expected_policy = dict(
            maximum_nodes=config["model"]["maximum_heavy_atoms"],
            maximum_closures=config["model"]["maximum_closures"],
            node_padding="batch",
            repeat_supervision="exact_fragment",
            core_conditioning="qualified_core",
        )
        if (
            cached.get("schema_version") != TENSOR_CACHE_SCHEMA
            or cached.get("complete") is not True
            or cached.get("inputs") != expected_inputs
            or cached.get("policy") != expected_policy
            or type(cached.get("records")) is not int
            or cached["records"] < 1
            or not cached.get("shards")
        ):
            raise TrainingRestartError("Prepared tensor manifest identity or completeness changed")
    family_count = runtime.get("families_per_batch")
    if runtime.get("family_schedule", "independent") not in ("independent", "balanced_cycles"):
        raise TrainingRestartError("Unknown family schedule")
    if runtime.get("family_schedule") == "balanced_cycles" and family_count is None:
        raise TrainingRestartError("Balanced cycles require family-block sampling")
    if family_count is not None and (
        type(family_count) is not int or family_count < 1 or runtime["batch_size"] % family_count
    ):
        raise TrainingRestartError("Batch size must divide into families_per_batch")
    if runtime.get("core_conditioning", "adapter") not in ("adapter", "qualified_core"):
        raise TrainingRestartError("Unknown core conditioning")
    objective = validate_objective(config.get("objective"))
    parallel = runtime.get("parallel_backend")
    relation_backend = runtime.get("relation_embedding_backend")
    if relation_backend is not None:
        if relation_backend != "ordered_fp32_v1":
            raise TrainingRestartError("Unknown relation_embedding_backend")
        if parallel not in PARALLEL_GPUS or config["model"].get("attention_heads") != 8:
            raise TrainingRestartError(
                "Ordered relation embedding requires a qualified parallel backend/eight heads"
            )
    if parallel == "adaptive_eight_gpu_shards_v1" and relation_backend != "ordered_fp32_v1":
        raise TrainingRestartError(
            "Adaptive eight-GPU shards require ordered FP32 relation embedding"
        )
    if parallel is not None:
        if parallel not in PARALLEL_GPUS:
            raise TrainingRestartError("Unknown parallel_backend")
        if (
            prepared != "prefetch"
            or family_count != 3
            or runtime["batch_size"] % (12 if PARALLEL_GPUS[parallel] == 8 else 6)
            or runtime.get("node_padding") != "family"
            or objective.get("gradient_balancing") != "pcgrad"
            or objective.get("pcgrad_backend") != "sequential"
        ):
            raise TrainingRestartError(
                "Parallel shards require prepared prefetch, three divisible family blocks, "
                "family padding and sequential PCGrad"
            )
    if objective and config["model"].get("architecture") != "reaction_program_graph_transformer":
        raise TrainingRestartError("Restored objectives require the transformer")
    if (
        objective.get("offspring_weight", 0) or objective.get("junction_consistency_weight", 0)
    ) and config["model"].get("maximum_children", 0) < 1:
        raise TrainingRestartError("Branching supervision requires explicit child-count support")
    if runtime.get("node_padding", "model") not in ("model", "batch", "family"):
        raise TrainingRestartError("runtime.node_padding must be 'model', 'batch' or 'family'")
    if (
        runtime.get("node_padding") == "family"
        and objective.get("gradient_balancing", "pooled") == "pooled"
    ):
        raise TrainingRestartError("Family padding requires family-balanced gradients")
    repeat_mode = runtime.get("repeat_supervision", "serialization")
    if repeat_mode not in ("serialization", "exact_fragment") or (
        repeat_mode == "exact_fragment"
        and config["model"].get("architecture") != "reaction_program_graph_transformer"
    ):
        raise TrainingRestartError("Invalid runtime.repeat_supervision for this architecture")
    if (
        runtime.get("precision") != "float32"
        or runtime.get("deterministic") is not True
        or runtime.get("workers") != 0
    ):
        raise TrainingRestartError("This runner requires deterministic float32 with workers=0")
    if type(runtime.get("fill_uninitialized_memory", True)) is not bool:
        raise TrainingRestartError("runtime.fill_uninitialized_memory must be a boolean")
    for key in (
        "optimizer_steps",
        "checkpoint_interval",
        "batch_size",
        "cpu_threads",
        "maximum_cached_shards",
    ):
        if type(runtime.get(key)) is not int or runtime[key] < 1:
            raise TrainingRestartError(f"runtime.{key} must be a positive integer")
    if config["model"].get("architecture") not in {
        "sparse_mpnn",
        "reaction_program_graph_transformer",
    }:
        raise TrainingRestartError("Unsupported COMPOSE architecture")
    optimizer = config["optimizer"]
    if set(optimizer) != {"lr", "weight_decay", "betas", "eps"}:
        raise TrainingRestartError("AdamW requires explicit lr, weight_decay, betas and eps")
    for key in ("lr", "eps", "weight_decay"):
        value = optimizer[key]
        if not math.isfinite(value) or value < 0 or (key != "weight_decay" and value == 0):
            raise TrainingRestartError(f"Invalid optimizer.{key}")
    if len(optimizer["betas"]) != 2 or any(not 0 <= b < 1 for b in optimizer["betas"]):
        raise TrainingRestartError("Invalid optimizer.betas")
    clip = config["gradient_clip_norm"]
    weights = config["semantic_weights"]
    if not math.isfinite(clip) or clip <= 0:
        raise TrainingRestartError("gradient_clip_norm must be positive and finite")
    if (
        set(weights) != {"role_weight", "core_weight", "repeat_consistency_weight"}
        or any(not math.isfinite(v) or v < 0 for v in weights.values())
        or (config["model"]["architecture"] == "sparse_mpnn" and any(weights.values()))
    ):
        raise TrainingRestartError("Invalid semantic loss weights for this architecture")
    return config


def training_marginals(
    repo: Path, config: dict, data: ComposeLipidTrainingData, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    document = json.loads(
        resolve_pin(
            config["inputs"]["noise_marginals"], repo, label="training noise marginals"
        ).read_text()
    )
    if document.get("schema_version") != "forge.compose_lipid_noise_marginals.v1" or document.get(
        "inputs"
    ) != {key: data.identity[key] for key in ("population", "measure")}:
        raise TrainingRestartError("Noise marginals belong to a different population or measure")
    values = []
    for name, size in (
        ("node", len(data.atom_vocabulary)),
        ("bond", config["model"]["bond_classes"]),
    ):
        value = torch.tensor(document[name], dtype=torch.float32, device=device)
        if (
            value.shape
            not in (
                (size,),
                (len(data.vocabulary.program_states), len(data.vocabulary.role_states), size),
            )
            or not torch.isfinite(value).all()
            or (value < 0).any()
            or not torch.allclose(value.sum(-1), torch.ones_like(value.sum(-1)), rtol=0, atol=1e-6)
        ):
            raise TrainingRestartError(f"Invalid {name} noise marginal")
        values.append(value)
    return tuple(values)


def _runtime_identity(
    device: torch.device, threads: int, fill_uninitialized_memory: bool = True
) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.platform(),
        "torch_build": torch.__config__.show(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "cpu_threads": threads,
        "gpu_names": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if device.type == "cuda"
            else []
        ),
        "precision": "float32",
        "deterministic": True,
        "fill_uninitialized_memory": fill_uninitialized_memory,
    }


@contextmanager
def _deterministic_memory_fill(enabled: bool) -> Iterator[None]:
    """Scope allocator diagnostics without changing deterministic kernel selection."""
    previous = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = enabled
    try:
        yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous


def run_training(
    repo: Path,
    config_path: Path,
    work_dir: Path,
    output_dir: Path,
    *,
    device: str,
    seed: int,
    resume: bool,
    commit_progress: Callable[[], None],
) -> dict[str, Any]:
    """Run or explicitly resume; failed updates never become admitted metrics.

    Checkpoints are trusted, locally produced artifacts. Verify their digest before
    loading their pickle payload. A commit failure propagates without automatic retry.
    """
    config = read_admitted_config(repo, config_path)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise TrainingRestartError("seed must be a uint32 integer")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise TrainingRestartError("Only CPU and CUDA are supported")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise TrainingRestartError("CUDA requested but unavailable")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise TrainingRestartError("Set CUBLAS_WORKSPACE_CONFIG before CUDA initialization")
    runtime = config["runtime"]
    parallel_kind = runtime.get("parallel_backend")
    parallel = parallel_kind in PARALLEL_GPUS
    if parallel and (
        target.type != "cuda"
        or target.index not in (None, 0)
        or torch.cuda.device_count() != PARALLEL_GPUS[parallel_kind]
    ):
        count = "eight" if PARALLEL_GPUS[parallel_kind] == 8 else "six"
        raise TrainingRestartError(f"{parallel_kind} requires exactly {count} CUDA devices")
    fill_memory = runtime.get("fill_uninitialized_memory", True)
    identity = {
        "config_sha256": str(sha256_file(config_path)),
        "inputs": config["inputs"],
        "seed": seed,
        "source_sha256": source_fingerprint(Path(__file__).resolve().parents[3]),
        "runtime": _runtime_identity(target, runtime["cpu_threads"], fill_memory),
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    with (
        (work_dir / ".writer.lock").open("a") as lock,
        _deterministic_memory_fill(fill_memory),
        ExitStack() as resources,
    ):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TrainingRestartError("Another trainer owns this checkpoint directory") from error
        latest = work_dir / "latest.json"
        if latest.exists() != resume:
            raise TrainingRestartError(
                "Explicit resume requires an existing checkpoint; fresh runs forbid one"
            )
        pointer = read_json_object(latest, error=TrainingRestartError) if resume else None
        checkpoint = None
        if pointer is not None:
            if pointer.get("identity") != identity:
                raise TrainingRestartError("Restart input, source, config, seed or runtime changed")
            checkpoint = resolve_pin(pointer["checkpoint"], work_dir, label="restart checkpoint")
        with ComposeLipidTrainingData(
            repo,
            **{name: config["inputs"][name] for name in DATA_INPUTS},
            maximum_cached_shards=runtime["maximum_cached_shards"],
            mapped_cache=config["inputs"].get("mapped_cache"),
        ) as data:
            if config["model"]["maximum_heavy_atoms"] < data.maximum_heavy_atoms:
                raise TrainingRestartError("Model support would exclude large molecules")
            node, bond = training_marginals(repo, config, data, target)
            tensor_cache = None
            if "prepared_tensors" in config["inputs"]:
                from forge.corpus.compose_lipid_tensor_cache import PreparedTensorCache

                tensor_cache = PreparedTensorCache(
                    repo,
                    config["inputs"]["prepared_tensors"],
                    inputs={
                        k: config["inputs"][k]
                        for k in ("population", "verification", "measure", "mapped_cache")
                        if k in config["inputs"]
                    },
                    policy=dict(
                        maximum_nodes=config["model"]["maximum_heavy_atoms"],
                        maximum_closures=config["model"]["maximum_closures"],
                        node_padding="batch",
                        repeat_supervision="exact_fragment",
                        core_conditioning="qualified_core",
                    ),
                )
                if tensor_cache.metadata["records"] != len(data):
                    raise TrainingRestartError(
                        "Prepared tensor cache has incomplete cohort coverage"
                    )
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.set_default_dtype(torch.float32)
            torch.set_num_threads(runtime["cpu_threads"])
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            rng = np.random.default_rng(seed)
            generator = torch.Generator(device=target).manual_seed(seed)
            model = build_synthesis_program_flow(
                vocabulary=data.vocabulary,
                node_classes=len(data.atom_vocabulary),
                model_config=config["model"],
                device=target,
            ).float()
            optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
            completed, metrics = 0, {}
            family_names = [
                row[0]
                for row in data._database.execute(
                    "SELECT DISTINCT family FROM weights ORDER BY family"
                )
            ]
            cycle_schedule = runtime.get("family_schedule") == "balanced_cycles"
            exposure = {name: 0 for name in family_names}
            if checkpoint is not None:
                state = torch.load(checkpoint, map_location=target, weights_only=False)
                if (
                    state.get("schema_version") != CHECKPOINT_SCHEMA
                    or state.get("identity") != identity
                ):
                    raise TrainingRestartError("Checkpoint identity changed")
                completed = state["completed_steps"]
                if (
                    type(completed) is not int
                    or not 0 <= completed <= runtime["optimizer_steps"]
                    or state["examples_seen"] != completed * runtime["batch_size"]
                    or pointer["completed_steps"] != completed
                ):
                    raise TrainingRestartError("Checkpoint counters changed")
                model.load_state_dict(state["model"], strict=True)
                optimizer.load_state_dict(state["optimizer"])
                restore_training_random_state(state["random"], rng, generator, device=target)
                metrics = state["last_metrics"]
                if cycle_schedule:
                    from forge.model.family_exposure import expected_exposure

                    exposure = state["family_presentations"]
                    expected = expected_exposure(
                        families=len(family_names),
                        per_batch=runtime["families_per_batch"],
                        batch_size=runtime["batch_size"],
                        steps=completed,
                        seed=seed,
                    )
                    if exposure != dict(zip(family_names, expected.tolist(), strict=True)):
                        raise TrainingRestartError("Family exposure checkpoint counters changed")

            def save() -> None:
                path = work_dir / f"checkpoint_{completed:09d}.pt"
                random_state = capture_training_random_state(rng, generator, device=target)
                atomic_torch_save(
                    path,
                    {
                        "schema_version": CHECKPOINT_SCHEMA,
                        "identity": identity,
                        "completed_steps": completed,
                        "examples_seen": completed * runtime["batch_size"],
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "random": random_state,
                        "last_metrics": metrics,
                        **({"family_presentations": exposure} if cycle_schedule else {}),
                    },
                )
                write_json(
                    latest,
                    {
                        "identity": identity,
                        "completed_steps": completed,
                        "checkpoint": {"path": path.name, "sha256": str(sha256_file(path))},
                        **(
                            dict(
                                family_presentations=exposure,
                                last_metrics=metrics,
                                examples_seen=completed * runtime["batch_size"],
                            )
                            if cycle_schedule
                            else {}
                        ),
                    },
                )
                commit_progress()
                # Transport SDK randomness must not become training randomness.
                restore_training_random_state(random_state, rng, generator, device=target)
                # Retain the previous generation until the next generation is durable.
                for old in sorted(work_dir.glob("checkpoint_*.pt"))[:-2]:
                    old.unlink()

            if not resume:
                save()
            parallel_trainer = None
            if parallel:
                from forge.model.compose_lipid_workers import PersistentTrainer

                random_state = capture_training_random_state(rng, generator, device=target)
                parallel_trainer = PersistentTrainer(
                    model,
                    optimizer,
                    config,
                    node,
                    bond,
                    seed=seed,
                    vocabulary=data.vocabulary,
                    node_classes=len(data.atom_vocabulary),
                    mode=(
                        "adaptive_parallel"
                        if parallel_kind == "adaptive_eight_gpu_shards_v1"
                        else "split_parallel"
                    ),
                )
                resources.callback(parallel_trainer.close)
                restore_training_random_state(random_state, rng, generator, device=target)
            batches = None
            if runtime.get("prepared_loading") == "prefetch":
                from forge.model.compose_lipid_prefetch import CudaBatches, PreparedBatches

                batches = resources.enter_context(
                    PreparedBatches(
                        data,
                        tensor_cache,
                        sampler_state=rng.bit_generator.state,
                        seed=seed,
                        start=completed,
                        stop=runtime["optimizer_steps"],
                        families=len(family_names),
                        families_per_batch=runtime["families_per_batch"],
                        batch_size=runtime["batch_size"],
                        depth=runtime["prefetch_depth"],
                        workers=runtime["prefetch_workers"],
                        pin_memory=target.type == "cuda" and not parallel,
                    )
                )
                if target.type == "cuda" and not parallel:
                    batches = CudaBatches(batches, target)
                    resources.callback(batches.close)
            for step in range(completed + 1, runtime["optimizer_steps"] + 1):
                family_selection = None
                if cycle_schedule:
                    from forge.model.family_exposure import balanced_families

                    family_selection = balanced_families(
                        families=len(family_names),
                        per_batch=runtime["families_per_batch"],
                        step=step - 1,
                        seed=seed,
                    )
                ticket = next(batches) if batches is not None else None
                if ticket is not None and ticket.step != step - 1:
                    raise TrainingRestartError("Prepared batch sequence changed")
                if parallel_trainer is not None:
                    metrics = parallel_trainer.step(ticket.tensors, step - 1)
                else:
                    metrics = compose_lipid_training_step(
                        data,
                        model,
                        optimizer,
                        batch_size=runtime["batch_size"],
                        maximum_nodes=config["model"]["maximum_heavy_atoms"],
                        maximum_closures=config["model"]["maximum_closures"],
                        node_padding=runtime.get("node_padding", "model"),
                        repeat_supervision=runtime.get("repeat_supervision", "serialization"),
                        architecture=config["model"]["architecture"],
                        device=target,
                        rng=rng,
                        generator=generator,
                        node_marginal=node,
                        bond_marginal=bond,
                        semantic_weights=config["semantic_weights"],
                        core_conditioning=runtime.get("core_conditioning", "adapter"),
                        families_per_batch=runtime.get("families_per_batch"),
                        family_selection=family_selection,
                        objective=config.get("objective"),
                        gradient_clip_norm=config["gradient_clip_norm"],
                        tensor_cache=tensor_cache,
                        prepared_batch=ticket.tensors if ticket is not None else None,
                    )
                if ticket is not None:
                    rng.bit_generator.state = ticket.sampler_state
                completed = step
                if family_selection is not None:
                    for family in family_selection:
                        exposure[family_names[family]] += (
                            runtime["batch_size"] // runtime["families_per_batch"]
                        )
                if step % runtime["checkpoint_interval"] == 0 or step == runtime["optimizer_steps"]:
                    save()
            pointer = read_json_object(latest, error=TrainingRestartError)
            final = resolve_pin(pointer["checkpoint"], work_dir, label="final checkpoint")
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(final, output_dir / "checkpoint.pt")
            result = {
                "schema_version": RESULT_SCHEMA,
                "status": "complete",
                "identity": identity,
                "completed_steps": completed,
                "examples_seen": completed * runtime["batch_size"],
                "last_metrics": metrics,
                "checkpoint_sha256": str(sha256_file(final)),
                **({"family_presentations": exposure} if cycle_schedule else {}),
            }
            write_json(output_dir / "result.json", result)
            commit_progress()
            return result
