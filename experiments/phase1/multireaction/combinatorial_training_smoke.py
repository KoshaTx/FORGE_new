"""Deterministic local optimizer-step gate for the twelve-library shared cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.combinatorial_program_cache import RESULT_SCHEMA as CACHE_RESULT_SCHEMA
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.synthesis_program_training import (
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
    synthesis_program_fixed_state_exact,
    synthesis_program_forward_loss,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - training dependency
    torch = None  # type: ignore[assignment]

CONFIG_SCHEMA = "forge.combinatorial_training_smoke_config.v1"
RESULT_SCHEMA = "forge.combinatorial_training_smoke.v1"


class CombinatorialTrainingSmokeError(ValueError):
    """The local shared-model smoke contract failed."""


def _tensor_digest(values: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        if torch is None or not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _one_step(
    *,
    cache: SynthesisProgramProductionCache,
    records: tuple,
    model_config: dict[str, Any],
    optimization: dict[str, Any],
    node_marginal: np.ndarray,
    bond_marginal: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    assert torch is not None
    torch.manual_seed(seed)
    model = build_synthesis_program_flow(
        vocabulary=cache.vocabulary,
        node_classes=len(cache.atom_vocabulary),
        model_config=model_config,
        device=torch.device("cpu"),
    )
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    clean = collate_synthesis_program_training_batch(
        records,
        maximum_closures=int(model_config["maximum_closures"]),
        conditioning="program",
        vocabulary=cache.vocabulary,
    )
    clean_digest = _tensor_digest(clean)
    model_before = _tensor_digest(dict(model.state_dict()))
    batch_size = len(records)
    t = torch.full((batch_size,), float(optimization["flow_time"]), dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    loss, metrics, noisy = synthesis_program_forward_loss(
        model,
        clean,
        torch.as_tensor(node_marginal, dtype=torch.float32),
        torch.as_tensor(bond_marginal, dtype=torch.float32),
        t,
        generator,
    )
    if not bool(torch.isfinite(loss)):
        raise CombinatorialTrainingSmokeError("shared smoke loss is nonfinite")
    fixed_exact = synthesis_program_fixed_state_exact(noisy, clean)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite_gradients = bool(gradients) and all(
        bool(torch.isfinite(value).all()) for value in gradients
    )
    nonzero_gradients = sum(int(bool(torch.count_nonzero(value))) for value in gradients)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(optimization["gradient_clip_norm"])
    )
    optimizer.step()
    model_after = _tensor_digest(dict(model.state_dict()))
    metric_values = {
        key: float(value.detach()) if torch.is_tensor(value) else float(value)
        for key, value in sorted(metrics.items())
    }
    return {
        "loss": float(loss.detach()),
        "metrics": metric_values,
        "gradient_norm": float(gradient_norm.detach()),
        "finite_gradients": finite_gradients,
        "nonzero_gradient_tensors": nonzero_gradients,
        "fixed_noisy_states_exact": fixed_exact,
        "clean_batch_digest_before": clean_digest,
        "clean_batch_digest_after": _tensor_digest(clean),
        "model_digest_before": model_before,
        "model_digest_after": model_after,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def run_combinatorial_training_smoke(
    repo_root: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Run two identical local one-step replicas and require byte-level state determinism."""

    if torch is None:
        raise CombinatorialTrainingSmokeError("PyTorch is required for the training smoke")
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise CombinatorialTrainingSmokeError(
            "config/output must be inside the repository and output must be fresh"
        )
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise CombinatorialTrainingSmokeError("unsupported training-smoke config")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CombinatorialTrainingSmokeError("seed must be a nonnegative integer")
    if config.get("record_selection") != "lexicographically_first_train_record_per_program":
        raise CombinatorialTrainingSmokeError("training-smoke record selection changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, dict) or set(raw_inputs) != {
        "cache_result",
        "cache",
        "cache_config",
    }:
        raise CombinatorialTrainingSmokeError("training-smoke inputs changed")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in sorted(raw_inputs.items())}
    cache_result = json.loads(paths["cache_result"].read_text())
    if (
        cache_result.get("schema_version") != CACHE_RESULT_SCHEMA
        or cache_result.get("status") != "pass"
        or cache_result.get("config") != raw_inputs["cache_config"]
        or cache_result.get("artifacts", {}).get("cache.npz")
        != {
            **raw_inputs["cache"],
            "bytes": paths["cache"].stat().st_size,
        }
    ):
        raise CombinatorialTrainingSmokeError(
            "packed cache result is not authenticated and passing"
        )
    model_config = dict(config["model"])
    optimization = dict(config["optimization"])
    if (
        model_config.get("architecture") != "reaction_program_sparse_whole_lipid_flow"
        or int(model_config["maximum_heavy_atoms"])
        != int(cache_result["support"]["maximum_heavy_atoms"])
        or int(model_config["maximum_closures"]) != int(cache_result["support"]["maximum_closures"])
        or int(optimization["optimizer_steps_per_replica"]) != 1
        or int(optimization["determinism_replicas"]) != 2
        or float(optimization["learning_rate"]) <= 0
        or float(optimization["weight_decay"]) < 0
        or not 0 < float(optimization["flow_time"]) < 1
        or float(optimization["gradient_clip_norm"]) <= 0
        or float(optimization["source_probability_floor"]) <= 0
        or int(optimization["cpu_threads"]) != 1
    ):
        raise CombinatorialTrainingSmokeError("model or optimizer smoke contract changed")
    config_hash = str(sha256_file(config_path))
    source_root = Path(__file__).resolve().parents[3]
    source_names = (
        "experiments/phase1/multireaction/combinatorial_training_smoke.py",
        "forge/corpus/combinatorial_program_cache.py",
        "forge/corpus/synthesis_program_production_cache.py",
        "forge/model/synthesis_program_training.py",
        "forge/model/reaction_program_flow.py",
    )
    sources = {name: str(sha256_file(source_root / name)) for name in source_names}
    start = time.monotonic()
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with SynthesisProgramProductionCache(paths["cache"]) as cache:
            programs = tuple(cache.vocabulary.program_states[1:])
            if len(programs) != int(config["expected_programs"]):
                raise CombinatorialTrainingSmokeError("cache program count changed")
            selected = tuple(
                int(cache.indices(program_id=program, fold="train")[0]) for program in programs
            )
            records = cache.records(selected)
            masses = {program: 1.0 / len(programs) for program in programs}
            measure = cache.training_measure(masses)
            node_marginal, bond_marginal = cache.source_marginals(
                measure,
                node_classes=len(cache.atom_vocabulary),
                bond_classes=int(model_config["bond_classes"]),
                probability_floor=float(optimization["source_probability_floor"]),
            )
            replicas = [
                _one_step(
                    cache=cache,
                    records=records,
                    model_config=model_config,
                    optimization=optimization,
                    node_marginal=node_marginal,
                    bond_marginal=bond_marginal,
                    seed=seed,
                )
                for _ in range(2)
            ]
            selected_records = [
                {
                    "program_id": cache.program_id(index),
                    "record_id": cache.record_id(index),
                    "heavy_atoms": cache.record(index).node_count,
                    "closures": cache.record(index).graph.closure_count,
                }
                for index in selected
            ]
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
        torch.set_num_threads(previous_threads)
    comparison_fields = (
        "loss",
        "metrics",
        "gradient_norm",
        "finite_gradients",
        "nonzero_gradient_tensors",
        "fixed_noisy_states_exact",
        "clean_batch_digest_before",
        "clean_batch_digest_after",
        "model_digest_before",
        "model_digest_after",
        "parameter_count",
    )
    deterministic = all(replicas[0][key] == replicas[1][key] for key in comparison_fields)
    primary = replicas[0]
    gates = {
        "all_programs_exercised_once": [row["program_id"] for row in selected_records]
        == list(programs),
        "full_declared_support_preserved": max(row["heavy_atoms"] for row in selected_records)
        <= int(model_config["maximum_heavy_atoms"])
        and max(row["closures"] for row in selected_records)
        <= int(model_config["maximum_closures"]),
        "loss_and_gradient_norm_finite": math.isfinite(primary["loss"])
        and math.isfinite(primary["gradient_norm"]),
        "finite_nonzero_gradients": primary["finite_gradients"]
        and primary["nonzero_gradient_tensors"] > 0,
        "optimizer_updated_model_state": primary["model_digest_before"]
        != primary["model_digest_after"],
        "fixed_states_preserved_during_noising": primary["fixed_noisy_states_exact"],
        "clean_batch_not_mutated": primary["clean_batch_digest_before"]
        == primary["clean_batch_digest_after"],
        "identical_replica_results": deterministic,
        "generation_calls_zero": True,
    }
    if not all(gates.values()):
        raise CombinatorialTrainingSmokeError(f"training smoke failed: {gates}")
    for name, pin in raw_inputs.items():
        resolve_pin(pin, repo, label=name)
    if str(sha256_file(config_path)) != config_hash or any(
        str(sha256_file(source_root / name)) != digest for name, digest in sources.items()
    ):
        raise CombinatorialTrainingSmokeError("code or configuration changed during training smoke")
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - start,
        "config": {"path": str(config_path.relative_to(repo)), "sha256": config_hash},
        "inputs": raw_inputs,
        "sources": sources,
        "environment": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "deterministic_algorithms": True,
            "cpu_threads": 1,
        },
        "selected_records": selected_records,
        "source_marginals": {
            "node": node_marginal.tolist(),
            "bond": bond_marginal.tolist(),
        },
        "replicas": replicas,
        "gates": gates,
        "optimizer_steps_executed": 2,
        "optimizer_steps_per_replica": 1,
        "generator_sampling_calls": 0,
        "remote_compute": False,
        "nonclaims": [
            "A finite local optimizer step establishes plumbing only; it does not establish convergence, generation quality or improved realism.",
            "No molecule was sampled and no checkpoint was promoted.",
            "The smoke does not change structural, reconstruction, diversity or novelty gates.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".combinatorial-smoke-", dir=output.parent) as temp:
        work = Path(temp)
        (work / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.rename(work, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_combinatorial_training_smoke(args.repo_root, args.config, args.output_dir)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
