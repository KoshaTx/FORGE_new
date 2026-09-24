"""Deterministic twelve-family learning gate for the shared sparse flow."""

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
from forge.model.reaction_program_flow import synthesis_program_flow_loss
from forge.model.synthesis_program_training import (
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
    synthesis_program_fixed_state_exact,
    synthesis_program_forward,
    synthesis_program_reconstruction_metrics,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - training dependency
    torch = None  # type: ignore[assignment]

CONFIG_SCHEMA = "forge.combinatorial_overfit_gate_config.v1"
RESULT_SCHEMA = "forge.combinatorial_overfit_gate.v1"


class CombinatorialOverfitGateError(ValueError):
    """The bounded shared-model learning contract is invalid."""


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


def _slice_batch(values: dict[str, Any], index: int, batch_size: int) -> dict[str, Any]:
    return {
        key: (
            value[index : index + 1]
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size
            else value
        )
        for key, value in values.items()
    }


def _select_records(
    cache: SynthesisProgramProductionCache,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    selected: list[int] = []
    descriptions: list[dict[str, Any]] = []
    offsets = cache.arrays["node_offsets"]
    for program_id in cache.vocabulary.program_states[1:]:
        eligible = cache.indices(program_id=program_id, fold="train")
        if eligible.size == 0:
            raise CombinatorialOverfitGateError(f"program has no training record: {program_id}")
        sizes = offsets[eligible + 1] - offsets[eligible]
        median_size = int(np.sort(sizes)[(len(sizes) - 1) // 2])
        candidates = eligible[sizes == median_size]
        index = min((int(value) for value in candidates), key=cache.record_id)
        record = cache.record(index)
        selected.append(index)
        descriptions.append(
            {
                "program_id": program_id,
                "record_id": cache.record_id(index),
                "selection": "lower_median_heavy_atom_count_then_record_id",
                "heavy_atoms": record.node_count,
                "closures": record.graph.closure_count,
            }
        )
    return np.asarray(selected, dtype=np.int64), descriptions


def _predict_fixed_corruption(
    model: Any,
    clean: dict[str, Any],
    node_marginal: Any,
    bond_marginal: Any,
    *,
    flow_time: float,
    corruption_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    batch_size = int(clean["nodes"].shape[0])
    times = torch.full((batch_size,), flow_time, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(corruption_seed)
    return synthesis_program_forward(
        model,
        clean,
        node_marginal,
        bond_marginal,
        times,
        generator,
    )


def _equal_family_losses(
    predictions: dict[str, Any], clean: dict[str, Any], programs: tuple[str, ...]
) -> tuple[Any, dict[str, Any]]:
    batch_size = len(programs)
    losses: dict[str, Any] = {}
    for index, program_id in enumerate(programs):
        loss, _ = synthesis_program_flow_loss(
            _slice_batch(predictions, index, batch_size),
            _slice_batch(clean, index, batch_size),
            materialize_metrics=False,
        )
        losses[program_id] = loss
    return torch.stack(list(losses.values())).mean(), losses


def _evaluate(
    model: Any,
    clean: dict[str, Any],
    node_marginal: Any,
    bond_marginal: Any,
    programs: tuple[str, ...],
    *,
    flow_time: float,
    corruption_seed: int,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        predictions, noisy = _predict_fixed_corruption(
            model,
            clean,
            node_marginal,
            bond_marginal,
            flow_time=flow_time,
            corruption_seed=corruption_seed,
        )
        total, losses = _equal_family_losses(predictions, clean, programs)
        per_program: dict[str, Any] = {}
        for index, program_id in enumerate(programs):
            per_program[program_id] = {
                "loss": float(losses[program_id]),
                "reconstruction": synthesis_program_reconstruction_metrics(
                    _slice_batch(predictions, index, len(programs)),
                    _slice_batch(clean, index, len(programs)),
                ),
            }
    return {
        "total_equal_family_loss": float(total),
        "fixed_noisy_states_exact": synthesis_program_fixed_state_exact(noisy, clean),
        "per_program": per_program,
    }


def _validate_contract(config: dict[str, Any], cache_result: dict[str, Any]) -> None:
    model = config.get("model", {})
    optimization = config.get("optimization", {})
    acceptance = config.get("acceptance", {})
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("record_selection") != "lower_median_heavy_atom_count_then_record_id"
        or set(model)
        != {
            "architecture",
            "hidden_dim",
            "layers",
            "maximum_closures",
            "maximum_heavy_atoms",
            "dropout",
            "bond_classes",
        }
        or set(optimization)
        != {
            "steps",
            "learning_rate",
            "weight_decay",
            "flow_time",
            "gradient_clip_norm",
            "source_probability_floor",
            "cpu_threads",
            "corruption_policy",
            "family_loss_policy",
            "report_steps",
        }
        or set(acceptance)
        != {
            "maximum_final_to_initial_loss_ratio",
            "minimum_improved_programs",
            "minimum_exact_tensor_records",
        }
        or model.get("architecture") != "reaction_program_sparse_whole_lipid_flow"
        or int(model.get("maximum_heavy_atoms", -1))
        != int(cache_result["support"]["maximum_heavy_atoms"])
        or int(model.get("maximum_closures", -1))
        != int(cache_result["support"]["maximum_closures"])
        or int(optimization.get("steps", 0)) < 10
        or float(optimization.get("learning_rate", 0.0)) <= 0.0
        or float(optimization.get("weight_decay", -1.0)) < 0.0
        or float(optimization.get("gradient_clip_norm", 0.0)) <= 0.0
        or not 0.0 < float(optimization.get("flow_time", 0.0)) < 1.0
        or float(optimization.get("source_probability_floor", 0.0)) <= 0.0
        or int(optimization.get("cpu_threads", 0)) < 1
        or optimization.get("corruption_policy") != "one_fixed_draw_reused_for_learning_gate"
        or optimization.get("family_loss_policy") != "equal_program_mass"
        or not 0.0 < float(acceptance.get("maximum_final_to_initial_loss_ratio", 0.0)) < 1.0
        or int(acceptance.get("minimum_improved_programs", 0)) < 1
        or int(acceptance.get("minimum_exact_tensor_records", -1)) < 0
        or config.get("candidate_selection") is not False
        or config.get("generation_calls") != 0
        or config.get("remote_compute") is not False
    ):
        raise CombinatorialOverfitGateError(
            "overfit-gate model, optimization or acceptance contract changed"
        )


def run_combinatorial_overfit_gate(
    repo_root: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Fit one fixed corrupted training record per family under a prespecified budget."""

    if torch is None:
        raise CombinatorialOverfitGateError("PyTorch is required for the overfit gate")
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise CombinatorialOverfitGateError(
            "config/output must be inside the repository and output must be fresh"
        )
    config = json.loads(config_path.read_text())
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, dict) or set(raw_inputs) != {
        "cache_result",
        "cache",
        "cache_config",
        "training_smoke_result",
    }:
        raise CombinatorialOverfitGateError("overfit-gate inputs changed")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in sorted(raw_inputs.items())}
    cache_result = json.loads(paths["cache_result"].read_text())
    smoke_result = json.loads(paths["training_smoke_result"].read_text())
    if (
        cache_result.get("schema_version") != CACHE_RESULT_SCHEMA
        or cache_result.get("status") != "pass"
        or cache_result.get("config") != raw_inputs["cache_config"]
        or cache_result.get("artifacts", {}).get("cache.npz")
        != {**raw_inputs["cache"], "bytes": paths["cache"].stat().st_size}
        or smoke_result.get("status") != "pass"
        or not all(smoke_result.get("gates", {}).values())
        or smoke_result.get("inputs", {}).get("cache") != raw_inputs["cache"]
    ):
        raise CombinatorialOverfitGateError(
            "prerequisite cache or training smoke is not authenticated and passing"
        )
    _validate_contract(config, cache_result)
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CombinatorialOverfitGateError("seed must be a nonnegative integer")
    expected_programs = int(config["expected_programs"])
    acceptance = config["acceptance"]
    if (
        int(acceptance["minimum_improved_programs"]) > expected_programs
        or int(acceptance["minimum_exact_tensor_records"]) > expected_programs
    ):
        raise CombinatorialOverfitGateError("acceptance count exceeds the declared program count")
    config_hash = str(sha256_file(config_path))
    source_root = Path(__file__).resolve().parents[3]
    source_names = (
        "experiments/phase1/multireaction/combinatorial_overfit_gate.py",
        "forge/corpus/combinatorial_program_cache.py",
        "forge/corpus/synthesis_program_production_cache.py",
        "forge/model/synthesis_program_training.py",
        "forge/model/reaction_program_flow.py",
    )
    sources = {name: str(sha256_file(source_root / name)) for name in source_names}
    optimization = config["optimization"]
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    start = time.monotonic()
    try:
        torch.manual_seed(seed)
        torch.set_num_threads(int(optimization["cpu_threads"]))
        torch.use_deterministic_algorithms(True)
        with SynthesisProgramProductionCache(paths["cache"]) as cache:
            programs = tuple(cache.vocabulary.program_states[1:])
            if len(programs) != expected_programs:
                raise CombinatorialOverfitGateError("cache program count changed")
            selected, selected_records = _select_records(cache)
            records = cache.records(selected)
            clean = collate_synthesis_program_training_batch(
                records,
                maximum_closures=int(config["model"]["maximum_closures"]),
                conditioning="program",
                vocabulary=cache.vocabulary,
            )
            clean_digest = _tensor_digest(clean)
            measure = cache.training_measure({program: 1.0 / len(programs) for program in programs})
            node_marginal, bond_marginal = cache.source_marginals(
                measure,
                node_classes=len(cache.atom_vocabulary),
                bond_classes=int(config["model"]["bond_classes"]),
                probability_floor=float(optimization["source_probability_floor"]),
            )
            node_p0 = torch.as_tensor(node_marginal, dtype=torch.float32)
            bond_p0 = torch.as_tensor(bond_marginal, dtype=torch.float32)
            model = build_synthesis_program_flow(
                vocabulary=cache.vocabulary,
                node_classes=len(cache.atom_vocabulary),
                model_config=config["model"],
                device=torch.device("cpu"),
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(optimization["learning_rate"]),
                weight_decay=float(optimization["weight_decay"]),
            )
            corruption_seed = seed + 1
            initial = _evaluate(
                model,
                clean,
                node_p0,
                bond_p0,
                programs,
                flow_time=float(optimization["flow_time"]),
                corruption_seed=corruption_seed,
            )
            trajectory = [
                {"step": 0, "total_equal_family_loss": initial["total_equal_family_loss"]}
            ]
            raw_report_steps = optimization.get("report_steps")
            if (
                not isinstance(raw_report_steps, list)
                or not raw_report_steps
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in raw_report_steps
                )
                or raw_report_steps != sorted(set(raw_report_steps))
                or raw_report_steps[0] < 1
                or raw_report_steps[-1] != int(optimization["steps"])
            ):
                raise CombinatorialOverfitGateError("report steps must end at the training budget")
            report_steps = set(raw_report_steps)
            all_finite = True
            fixed_exact = bool(initial["fixed_noisy_states_exact"])
            maximum_gradient_norm = 0.0
            completed_steps = 0
            model.train()
            for step in range(1, int(optimization["steps"]) + 1):
                optimizer.zero_grad(set_to_none=True)
                predictions, noisy = _predict_fixed_corruption(
                    model,
                    clean,
                    node_p0,
                    bond_p0,
                    flow_time=float(optimization["flow_time"]),
                    corruption_seed=corruption_seed,
                )
                loss, _ = _equal_family_losses(predictions, clean, programs)
                if not bool(torch.isfinite(loss)):
                    all_finite = False
                    break
                fixed_exact = fixed_exact and synthesis_program_fixed_state_exact(noisy, clean)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(optimization["gradient_clip_norm"])
                )
                maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
                all_finite = all_finite and math.isfinite(float(gradient_norm))
                optimizer.step()
                completed_steps = step
                if step in report_steps:
                    trajectory.append(
                        {"step": step, "total_equal_family_loss": float(loss.detach())}
                    )
            final = _evaluate(
                model,
                clean,
                node_p0,
                bond_p0,
                programs,
                flow_time=float(optimization["flow_time"]),
                corruption_seed=corruption_seed,
            )
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
        torch.set_num_threads(previous_threads)
    initial_loss = float(initial["total_equal_family_loss"])
    final_loss = float(final["total_equal_family_loss"])
    improved = [
        program
        for program in programs
        if final["per_program"][program]["loss"] < initial["per_program"][program]["loss"]
    ]
    exact = sum(
        int(final["per_program"][program]["reconstruction"]["exact_tensor_records"])
        for program in programs
    )
    per_program = {
        program: {
            "initial_loss": initial["per_program"][program]["loss"],
            "final_loss": final["per_program"][program]["loss"],
            "loss_reduction_fraction": 1.0
            - final["per_program"][program]["loss"] / initial["per_program"][program]["loss"],
            "initial_reconstruction": initial["per_program"][program]["reconstruction"],
            "final_reconstruction": final["per_program"][program]["reconstruction"],
        }
        for program in programs
    }
    gates = {
        "all_programs_represented_once": len(selected_records) == expected_programs
        and [row["program_id"] for row in selected_records] == list(programs),
        "support_bounds_preserved": max(row["heavy_atoms"] for row in selected_records)
        <= int(config["model"]["maximum_heavy_atoms"])
        and max(row["closures"] for row in selected_records)
        <= int(config["model"]["maximum_closures"]),
        "all_losses_and_gradients_finite": all_finite
        and math.isfinite(initial_loss)
        and math.isfinite(final_loss)
        and math.isfinite(maximum_gradient_norm),
        "fixed_states_never_noised": fixed_exact and bool(final["fixed_noisy_states_exact"]),
        "clean_batch_not_mutated": clean_digest == _tensor_digest(clean),
        "total_loss_reduction": final_loss / initial_loss
        <= float(acceptance["maximum_final_to_initial_loss_ratio"]),
        "required_programs_improved": len(improved) >= int(acceptance["minimum_improved_programs"]),
        "required_exact_tensor_reconstruction": exact
        >= int(acceptance["minimum_exact_tensor_records"]),
        "generation_calls_zero": True,
    }
    for name, pin in raw_inputs.items():
        resolve_pin(pin, repo, label=name)
    if str(sha256_file(config_path)) != config_hash or any(
        str(sha256_file(source_root / name)) != digest for name, digest in sources.items()
    ):
        raise CombinatorialOverfitGateError("code or configuration changed during overfit gate")
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - start,
        "scientific_question": config["scientific_question"],
        "hypothesis": config["hypothesis"],
        "alternative_explanation": config["alternative_explanation"],
        "config": {"path": str(config_path.relative_to(repo)), "sha256": config_hash},
        "inputs": raw_inputs,
        "sources": sources,
        "environment": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "deterministic_algorithms": True,
            "cpu_threads": int(optimization["cpu_threads"]),
        },
        "selected_records": selected_records,
        "model": {**config["model"], "parameter_count": parameter_count},
        "training": {
            "steps": int(optimization["steps"]),
            "examples_seen": int(optimization["steps"]) * expected_programs,
            "corruption_policy": optimization["corruption_policy"],
            "family_loss_policy": optimization["family_loss_policy"],
            "initial_total_loss": initial_loss,
            "final_total_loss": final_loss,
            "final_to_initial_loss_ratio": final_loss / initial_loss,
            "maximum_gradient_norm_before_clipping": maximum_gradient_norm,
            "trajectory": trajectory,
        },
        "per_program": per_program,
        "programs_with_improved_loss": improved,
        "exact_tensor_records": exact,
        "gates": gates,
        "optimizer_steps_executed": completed_steps,
        "generator_sampling_calls": 0,
        "checkpoint_retained": False,
        "remote_compute": False,
        "decision": (
            "shared_sparse_learning_gate_passed"
            if all(gates.values())
            else "shared_sparse_learning_gate_failed"
        ),
        "nonclaims": [
            "Fitting one fixed corruption per family is a learning-capability diagnostic, not evidence of generalization or generation quality.",
            "No molecule was sampled and no diagnostic checkpoint was retained or promoted.",
            "This gate does not establish improved realism, synthesis success or production readiness.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".combinatorial-overfit-", dir=output.parent) as temp:
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
    result = run_combinatorial_overfit_gate(args.repo_root, args.config, args.output_dir)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
