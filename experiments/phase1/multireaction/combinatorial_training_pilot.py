"""Bounded balanced-training pilot with exhaustive per-family calibration evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.combinatorial_program_cache import RESULT_SCHEMA as CACHE_RESULT_SCHEMA
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.defog_feasibility import _model_state_sha256
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

CONFIG_SCHEMA = "forge.combinatorial_training_pilot_config.v1"
RESULT_SCHEMA = "forge.combinatorial_training_pilot.v1"


class CombinatorialTrainingPilotError(ValueError):
    """The bounded twelve-family pilot violates its frozen contract."""


def _slice_batch(values: dict[str, Any], index: int, batch_size: int) -> dict[str, Any]:
    return {
        key: (
            value[index : index + 1]
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size
            else value
        )
        for key, value in values.items()
    }


def _record_losses(predictions: dict[str, Any], clean: dict[str, Any]) -> list[Any]:
    batch_size = int(clean["nodes"].shape[0])
    losses = []
    for index in range(batch_size):
        loss, _ = synthesis_program_flow_loss(
            _slice_batch(predictions, index, batch_size),
            _slice_batch(clean, index, batch_size),
            materialize_metrics=False,
        )
        losses.append(loss)
    return losses


def _validate_contract(config: dict[str, Any], cache_result: dict[str, Any]) -> None:
    model = config.get("model", {})
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    acceptance = config.get("acceptance", {})
    if (
        config.get("schema_version") != CONFIG_SCHEMA
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
        or model.get("architecture") != "reaction_program_sparse_whole_lipid_flow"
        or int(model.get("maximum_heavy_atoms", -1))
        != int(cache_result["support"]["maximum_heavy_atoms"])
        or int(model.get("maximum_closures", -1))
        != int(cache_result["support"]["maximum_closures"])
        or set(training)
        != {
            "steps",
            "records_per_program_per_step",
            "sampling_policy",
            "family_loss_policy",
            "flow_time_policy",
            "minimum_flow_time",
            "maximum_flow_time",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "source_probability_floor",
            "loss_window_steps",
            "report_steps",
            "cpu_threads",
        }
        or int(training.get("steps", 0)) < 10
        or int(training.get("records_per_program_per_step", 0)) != 1
        or training.get("sampling_policy")
        != "realism_weight_within_family_equal_mass_between_families"
        or training.get("family_loss_policy") != "equal_program_mass"
        or training.get("flow_time_policy") != "uniform_bounded_per_record"
        or not 0.0
        < float(training.get("minimum_flow_time", 0.0))
        < float(training.get("maximum_flow_time", 0.0))
        < 1.0
        or float(training.get("learning_rate", 0.0)) <= 0.0
        or float(training.get("weight_decay", -1.0)) < 0.0
        or float(training.get("gradient_clip_norm", 0.0)) <= 0.0
        or float(training.get("source_probability_floor", 0.0)) <= 0.0
        or not 1 <= int(training.get("loss_window_steps", 0)) <= int(training.get("steps", 0)) // 2
        or int(training.get("cpu_threads", 0)) < 1
        or set(evaluation)
        != {
            "fold",
            "batch_size",
            "flow_time",
            "corruption_policy",
            "maximum_records_per_program",
        }
        or evaluation.get("fold") not in {"train", "calibration"}
        or int(evaluation.get("batch_size", 0)) < 1
        or not 0.0 < float(evaluation.get("flow_time", 0.0)) < 1.0
        or evaluation.get("corruption_policy") != "fixed_paired_draws_before_and_after_training"
        or (
            evaluation.get("maximum_records_per_program") is not None
            and int(evaluation["maximum_records_per_program"]) < 1
        )
        or set(acceptance)
        != {
            "maximum_final_to_initial_training_loss_ratio",
            "maximum_final_to_initial_calibration_loss_ratio",
            "minimum_improved_calibration_programs",
            "require_ugi_exact_reconstruction_not_regressed",
        }
        or not 0.0
        < float(acceptance.get("maximum_final_to_initial_training_loss_ratio", 0.0))
        < 1.0
        or not 0.0
        < float(acceptance.get("maximum_final_to_initial_calibration_loss_ratio", 0.0))
        < 1.0
        or int(acceptance.get("minimum_improved_calibration_programs", 0)) < 1
        or acceptance.get("require_ugi_exact_reconstruction_not_regressed") is not True
        or config.get("candidate_selection") is not False
        or config.get("generation_calls") != 0
        or config.get("heldout_structure_access") is not False
        or config.get("remote_compute") is not False
    ):
        raise CombinatorialTrainingPilotError(
            "training-pilot model, optimization, evaluation or acceptance contract changed"
        )
    report_steps = training.get("report_steps")
    if (
        not isinstance(report_steps, list)
        or not report_steps
        or any(isinstance(value, bool) or not isinstance(value, int) for value in report_steps)
        or report_steps != sorted(set(report_steps))
        or report_steps[0] < 1
        or report_steps[-1] != int(training["steps"])
    ):
        raise CombinatorialTrainingPilotError("report steps must end at the pilot budget")


def _evaluation_indices(
    cache: SynthesisProgramProductionCache,
    program_id: str,
    *,
    fold: str,
    maximum_records: int | None,
) -> np.ndarray:
    indices = cache.indices(program_id=program_id, fold=fold)
    if indices.size == 0:
        raise CombinatorialTrainingPilotError(
            f"program has no records in evaluation fold {fold}: {program_id}"
        )
    if maximum_records is None or len(indices) <= maximum_records:
        return indices
    ordered = sorted((int(index) for index in indices), key=cache.record_id)
    positions = np.linspace(0, len(ordered) - 1, num=maximum_records, dtype=np.int64)
    return np.asarray([ordered[int(position)] for position in positions], dtype=np.int64)


def _evaluate(
    model: Any,
    cache: SynthesisProgramProductionCache,
    node_marginal: Any,
    bond_marginal: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    maximum_records = evaluation["maximum_records_per_program"]
    maximum_records = None if maximum_records is None else int(maximum_records)
    output: dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        for program_state, program_id in enumerate(cache.vocabulary.program_states[1:], start=1):
            indices = _evaluation_indices(
                cache,
                program_id,
                fold=str(evaluation["fold"]),
                maximum_records=maximum_records,
            )
            losses: list[float] = []
            field_correct: defaultdict[str, int] = defaultdict(int)
            field_total: defaultdict[str, int] = defaultdict(int)
            exact_records = 0
            fixed_exact = True
            batch_size = int(evaluation["batch_size"])
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                records = cache.records(selected)
                clean = collate_synthesis_program_training_batch(
                    records,
                    maximum_closures=int(config["model"]["maximum_closures"]),
                    conditioning="program",
                    vocabulary=cache.vocabulary,
                )
                times = torch.full(
                    (len(records),), float(evaluation["flow_time"]), dtype=torch.float32
                )
                generator = torch.Generator(device="cpu").manual_seed(
                    int(config["seed"]) + 100_000 + program_state * 10_000 + start
                )
                predictions, noisy = synthesis_program_forward(
                    model,
                    clean,
                    node_marginal,
                    bond_marginal,
                    times,
                    generator,
                )
                losses.extend(float(value) for value in _record_losses(predictions, clean))
                reconstruction = synthesis_program_reconstruction_metrics(predictions, clean)
                exact_records += int(reconstruction["exact_tensor_records"])
                for field, count in reconstruction["field_correct"].items():
                    field_correct[field] += int(count)
                for field, count in reconstruction["field_total"].items():
                    field_total[field] += int(count)
                fixed_exact = fixed_exact and synthesis_program_fixed_state_exact(noisy, clean)
                fixed_exact = fixed_exact and bool(reconstruction["fixed_states_exact"])
            output[program_id] = {
                "records": len(indices),
                "mean_loss": float(np.mean(losses)),
                "exact_tensor_records": exact_records,
                "exact_tensor_fraction": exact_records / len(indices),
                "field_accuracy": {
                    field: field_correct[field] / max(field_total[field], 1)
                    for field in sorted(field_total)
                },
                "fixed_states_exact": fixed_exact,
            }
    return output


def run_combinatorial_training_pilot(
    repo_root: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Run a local balanced pilot and paired exhaustive calibration evaluation."""

    if torch is None:
        raise CombinatorialTrainingPilotError("PyTorch is required for the training pilot")
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise CombinatorialTrainingPilotError(
            "config/output must be inside the repository and output must be fresh"
        )
    config = json.loads(config_path.read_text())
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, dict) or set(raw_inputs) != {
        "cache_result",
        "cache",
        "cache_config",
        "overfit_gate_result",
    }:
        raise CombinatorialTrainingPilotError("training-pilot inputs changed")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in sorted(raw_inputs.items())}
    cache_result = json.loads(paths["cache_result"].read_text())
    overfit = json.loads(paths["overfit_gate_result"].read_text())
    if (
        cache_result.get("schema_version") != CACHE_RESULT_SCHEMA
        or cache_result.get("status") != "pass"
        or cache_result.get("config") != raw_inputs["cache_config"]
        or cache_result.get("artifacts", {}).get("cache.npz")
        != {**raw_inputs["cache"], "bytes": paths["cache"].stat().st_size}
        or overfit.get("schema_version") != "forge.combinatorial_overfit_gate.v1"
        or overfit.get("status") != "pass"
        or not all(overfit.get("gates", {}).values())
        or overfit.get("inputs", {}).get("cache") != raw_inputs["cache"]
    ):
        raise CombinatorialTrainingPilotError(
            "prerequisite cache or overfit gate is not authenticated and passing"
        )
    _validate_contract(config, cache_result)
    seed = config.get("seed")
    expected_programs = config.get("expected_programs")
    expected_evaluation_records = config.get("expected_evaluation_records")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(expected_programs, bool)
        or not isinstance(expected_programs, int)
        or expected_programs < 1
        or isinstance(expected_evaluation_records, bool)
        or not isinstance(expected_evaluation_records, int)
        or expected_evaluation_records < expected_programs
        or int(config["acceptance"]["minimum_improved_calibration_programs"]) > expected_programs
    ):
        raise CombinatorialTrainingPilotError("seed or expected population contract is invalid")
    config_hash = str(sha256_file(config_path))
    source_root = Path(__file__).resolve().parents[3]
    source_names = (
        "experiments/phase1/multireaction/combinatorial_training_pilot.py",
        "forge/corpus/combinatorial_program_cache.py",
        "forge/corpus/synthesis_program_production_cache.py",
        "forge/model/synthesis_program_training.py",
        "forge/model/reaction_program_flow.py",
    )
    sources = {name: str(sha256_file(source_root / name)) for name in source_names}
    training = config["training"]
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    start = time.monotonic()
    try:
        torch.manual_seed(seed)
        torch.set_num_threads(int(training["cpu_threads"]))
        torch.use_deterministic_algorithms(True)
        sampling_rng = np.random.default_rng(seed + 1)
        noise_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
        with SynthesisProgramProductionCache(paths["cache"]) as cache:
            programs = tuple(cache.vocabulary.program_states[1:])
            if len(programs) != expected_programs:
                raise CombinatorialTrainingPilotError("cache program count changed")
            program_mass = {program: 1.0 / len(programs) for program in programs}
            measure = cache.training_measure(program_mass)
            train_indices: dict[str, np.ndarray] = {}
            train_probabilities: dict[str, np.ndarray] = {}
            for program in programs:
                indices = cache.indices(program_id=program, fold="train")
                probabilities = measure[indices]
                if indices.size == 0 or float(probabilities.sum()) <= 0.0:
                    raise CombinatorialTrainingPilotError(
                        f"program has no positive training measure: {program}"
                    )
                train_indices[program] = indices
                train_probabilities[program] = probabilities / probabilities.sum()
            node_marginal, bond_marginal = cache.source_marginals(
                measure,
                node_classes=len(cache.atom_vocabulary),
                bond_classes=int(config["model"]["bond_classes"]),
                probability_floor=float(training["source_probability_floor"]),
            )
            node_p0 = torch.as_tensor(node_marginal, dtype=torch.float32)
            bond_p0 = torch.as_tensor(bond_marginal, dtype=torch.float32)
            model = build_synthesis_program_flow(
                vocabulary=cache.vocabulary,
                node_classes=len(cache.atom_vocabulary),
                model_config=config["model"],
                device=torch.device("cpu"),
            )
            initial_model_sha256 = _model_state_sha256(model)
            initial_calibration = _evaluate(model, cache, node_p0, bond_p0, config)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
            )
            losses: list[float] = []
            trajectory: list[dict[str, Any]] = []
            report_steps = set(int(value) for value in training["report_steps"])
            sampling_digest = hashlib.sha256()
            unique_records: defaultdict[str, set[str]] = defaultdict(set)
            fixed_exact = True
            all_finite = True
            maximum_gradient_norm = 0.0
            completed_steps = 0
            model.train()
            for step in range(1, int(training["steps"]) + 1):
                selected = np.asarray(
                    [
                        sampling_rng.choice(train_indices[program], p=train_probabilities[program])
                        for program in programs
                    ],
                    dtype=np.int64,
                )
                records = cache.records(selected)
                for program, index in zip(programs, selected, strict=True):
                    record_id = cache.record_id(int(index))
                    sampling_digest.update(f"{step}\t{program}\t{record_id}\n".encode())
                    unique_records[program].add(record_id)
                clean = collate_synthesis_program_training_batch(
                    records,
                    maximum_closures=int(config["model"]["maximum_closures"]),
                    conditioning="program",
                    vocabulary=cache.vocabulary,
                )
                times = torch.empty(len(programs), dtype=torch.float32).uniform_(
                    float(training["minimum_flow_time"]),
                    float(training["maximum_flow_time"]),
                    generator=noise_generator,
                )
                optimizer.zero_grad(set_to_none=True)
                predictions, noisy = synthesis_program_forward(
                    model,
                    clean,
                    node_p0,
                    bond_p0,
                    times,
                    noise_generator,
                )
                family_losses = _record_losses(predictions, clean)
                loss = torch.stack(family_losses).mean()
                if not bool(torch.isfinite(loss)):
                    all_finite = False
                    break
                fixed_exact = fixed_exact and synthesis_program_fixed_state_exact(noisy, clean)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_norm"])
                )
                maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
                all_finite = all_finite and math.isfinite(float(gradient_norm))
                optimizer.step()
                value = float(loss.detach())
                losses.append(value)
                completed_steps = step
                if step in report_steps:
                    trajectory.append({"step": step, "equal_family_loss": value})
            final_calibration = _evaluate(model, cache, node_p0, bond_p0, config)
            final_model_sha256 = _model_state_sha256(model)
            fold_counts = cache.fold_counts()
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
        torch.set_num_threads(previous_threads)
    window = int(training["loss_window_steps"])
    if len(losses) < 2 * window:
        raise CombinatorialTrainingPilotError("pilot ended before both loss windows completed")
    initial_training_loss = float(np.mean(losses[:window]))
    final_training_loss = float(np.mean(losses[-window:]))
    calibration: dict[str, Any] = {}
    improved: list[str] = []
    for program in programs:
        before = initial_calibration[program]
        after = final_calibration[program]
        if after["mean_loss"] < before["mean_loss"]:
            improved.append(program)
        calibration[program] = {
            "records": after["records"],
            "initial_mean_loss": before["mean_loss"],
            "final_mean_loss": after["mean_loss"],
            "final_to_initial_loss_ratio": after["mean_loss"] / before["mean_loss"],
            "initial_exact_tensor_records": before["exact_tensor_records"],
            "final_exact_tensor_records": after["exact_tensor_records"],
            "initial_field_accuracy": before["field_accuracy"],
            "final_field_accuracy": after["field_accuracy"],
            "fixed_states_exact": before["fixed_states_exact"] and after["fixed_states_exact"],
        }
    initial_equal_family_calibration = float(
        np.mean([initial_calibration[program]["mean_loss"] for program in programs])
    )
    final_equal_family_calibration = float(
        np.mean([final_calibration[program]["mean_loss"] for program in programs])
    )
    evaluation_records = sum(row["records"] for row in calibration.values())
    ugi = calibration.get("ugi_3cr_agile")
    ugi_not_regressed = ugi is None or (
        ugi["final_exact_tensor_records"] >= ugi["initial_exact_tensor_records"]
    )
    acceptance = config["acceptance"]
    gates = {
        "all_programs_train_sampled_each_step": completed_steps == int(training["steps"])
        and all(len(unique_records[program]) > 0 for program in programs),
        "realism_weight_and_equal_family_sampling_used": True,
        "all_losses_and_gradients_finite": all_finite
        and all(math.isfinite(value) for value in losses),
        "fixed_states_preserved": fixed_exact
        and all(row["fixed_states_exact"] for row in calibration.values()),
        "training_loss_reduction": final_training_loss / initial_training_loss
        <= float(acceptance["maximum_final_to_initial_training_loss_ratio"]),
        "calibration_loss_reduction": final_equal_family_calibration
        / initial_equal_family_calibration
        <= float(acceptance["maximum_final_to_initial_calibration_loss_ratio"]),
        "required_calibration_programs_improved": len(improved)
        >= int(acceptance["minimum_improved_calibration_programs"]),
        "ugi_exact_reconstruction_not_regressed": ugi_not_regressed,
        "all_expected_evaluation_records_used": evaluation_records == expected_evaluation_records,
        "heldout_structure_access_zero": True,
        "generation_calls_zero": True,
        "checkpoint_not_retained_or_promoted": True,
    }
    for name, pin in raw_inputs.items():
        resolve_pin(pin, repo, label=name)
    if str(sha256_file(config_path)) != config_hash or any(
        str(sha256_file(source_root / name)) != digest for name, digest in sources.items()
    ):
        raise CombinatorialTrainingPilotError("code or configuration changed during pilot")
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
            "cpu_threads": int(training["cpu_threads"]),
        },
        "population": {
            "train_fold_counts": {
                program: int(fold_counts[program]["train"]) for program in programs
            },
            "evaluation_fold": config["evaluation"]["fold"],
            "evaluation_records": evaluation_records,
            "programs": list(programs),
        },
        "model": {
            **config["model"],
            "parameter_count": parameter_count,
            "initial_state_sha256": initial_model_sha256,
            "final_state_sha256": final_model_sha256,
        },
        "training": {
            "steps": int(training["steps"]),
            "completed_steps": completed_steps,
            "examples_seen": completed_steps * len(programs),
            "initial_window_mean_loss": initial_training_loss,
            "final_window_mean_loss": final_training_loss,
            "final_to_initial_loss_ratio": final_training_loss / initial_training_loss,
            "maximum_gradient_norm_before_clipping": maximum_gradient_norm,
            "trajectory": trajectory,
            "sampling_ledger_sha256": sampling_digest.hexdigest(),
            "unique_training_records_by_program": {
                program: len(unique_records[program]) for program in programs
            },
        },
        "calibration": {
            "initial_equal_family_mean_loss": initial_equal_family_calibration,
            "final_equal_family_mean_loss": final_equal_family_calibration,
            "final_to_initial_loss_ratio": final_equal_family_calibration
            / initial_equal_family_calibration,
            "programs_with_improved_loss": improved,
            "by_program": calibration,
        },
        "gates": gates,
        "generator_sampling_calls": 0,
        "heldout_structure_access": False,
        "checkpoint_retained": False,
        "remote_compute": False,
        "decision": (
            "limited_balanced_training_pilot_passed"
            if all(gates.values())
            else "limited_balanced_training_pilot_failed"
        ),
        "nonclaims": [
            "Calibration denoising improvement is not unconditional generation quality or improved lipid realism.",
            "The pilot uses one training seed and does not estimate across-training uncertainty.",
            "No heldout structure, molecule generation, retained checkpoint, candidate selection, synthesis call or remote compute was used.",
            "Generation-level validity, exact Ugi reconstruction, diversity and novelty remain unmeasured by this pilot.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".combinatorial-pilot-", dir=output.parent) as temp:
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
    result = run_combinatorial_training_pilot(args.repo_root, args.config, args.output_dir)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
