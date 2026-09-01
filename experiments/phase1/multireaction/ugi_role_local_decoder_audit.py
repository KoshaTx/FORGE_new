"""Attribute Ugi reconstruction errors to topology versus topology-conditioned chemistry."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from experiments.phase1.multireaction.reaction_specialization import (
    ReactionProgramSpecializationError,
    _load_base_package,
    _set_determinism,
)
from forge.core.hashing import pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.synthesis_program_training import (
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
    move_tensors,
    synthesis_program_forward,
    synthesis_program_topology_conditioned_forward,
)

try:
    import torch
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]


CONFIG_SCHEMA = "forge.ugi_role_local_decoder_attribution_config.v1"
RESULT_SCHEMA = "forge.ugi_role_local_decoder_attribution.v1"


class UgiRoleLocalDecoderAuditError(ValueError):
    """The teacher-forced attribution contract or authenticated input is invalid."""


def _empty_metric() -> dict[str, float | int]:
    return {"correct": 0, "total": 0, "cross_entropy_sum": 0.0}


def _accumulate_metric(
    metric: dict[str, float | int],
    logits: Any,
    targets: Any,
    mask: Any,
) -> None:
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    if int(selected_targets.numel()) == 0:
        return
    metric["correct"] = int(metric["correct"]) + int(
        (selected_logits.argmax(dim=-1) == selected_targets).sum().item()
    )
    metric["total"] = int(metric["total"]) + int(selected_targets.numel())
    metric["cross_entropy_sum"] = float(metric["cross_entropy_sum"]) + float(
        functional.cross_entropy(selected_logits, selected_targets, reduction="sum").item()
    )


def _finalize(metrics: Mapping[str, Mapping[str, float | int]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, metric in sorted(metrics.items()):
        total = int(metric["total"])
        result[name] = {
            "correct": int(metric["correct"]),
            "total": total,
            "accuracy": None if total == 0 else int(metric["correct"]) / total,
            "cross_entropy": (None if total == 0 else float(metric["cross_entropy_sum"]) / total),
        }
    return result


def _new_role_metrics(role_names: tuple[str, ...]) -> dict[str, dict[str, float | int]]:
    return {
        f"{role}:{field}": _empty_metric()
        for role in role_names
        for field in ("nodes", "parents", "parent_bonds")
    }


def _accumulate_role_metrics(
    metrics: dict[str, dict[str, float | int]],
    predictions: Mapping[str, Any],
    clean: Mapping[str, Any],
    role_names: tuple[str, ...],
    *,
    include_topology: bool,
) -> None:
    exterior = clean["node_mask"] & (clean["core_position_states"] == 1)
    for role_index, role_name in enumerate(role_names, start=1):
        role_mask = exterior & (clean["role_states"] == role_index)
        _accumulate_metric(
            metrics[f"{role_name}:nodes"],
            predictions["nodes"],
            clean["nodes"],
            role_mask & clean["atom_variable_mask"],
        )
        _accumulate_metric(
            metrics[f"{role_name}:parent_bonds"],
            predictions["parent_bonds"],
            clean["parent_bonds"],
            role_mask & clean["parent_bond_variable_mask"],
        )
        if include_topology:
            _accumulate_metric(
                metrics[f"{role_name}:parents"],
                predictions["parents"],
                clean["parents"],
                role_mask & clean["parent_variable_mask"],
            )


def run_ugi_role_local_decoder_audit(
    config_path: Path,
    repo: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one deterministic train-fold-only teacher-forcing audit on the frozen base model."""

    if torch is None:
        raise UgiRoleLocalDecoderAuditError("teacher-forced attribution requires torch")
    config = read_json_object(
        config_path,
        error=UgiRoleLocalDecoderAuditError,
        label="role-local decoder attribution config",
    )
    if not isinstance(config, Mapping) or set(config) != {
        "schema_version",
        "scientific_question",
        "target_program",
        "seed",
        "records",
        "batch_size",
        "flow_times",
        "inputs",
        "base_checkpoint",
        "nonclaims",
    }:
        raise UgiRoleLocalDecoderAuditError("attribution config fields changed")
    if config["schema_version"] != CONFIG_SCHEMA or config["target_program"] != "ugi_3cr_agile":
        raise UgiRoleLocalDecoderAuditError("attribution scope changed")
    records = int(config["records"])
    batch_size = int(config["batch_size"])
    flow_times = tuple(float(value) for value in config["flow_times"])
    if (
        records < 1
        or batch_size < 1
        or batch_size > records
        or not flow_times
        or any(not 0.0 < value < 1.0 for value in flow_times)
    ):
        raise UgiRoleLocalDecoderAuditError("attribution sampling geometry is invalid")
    raw_inputs = config["inputs"]
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != {
        "base_checkpoint_archive",
        "base_training_result",
        "base_design",
        "production_cache",
    }:
        raise UgiRoleLocalDecoderAuditError("attribution input pins changed")
    paths = {label: resolve_pin(pin, repo, label=label) for label, pin in raw_inputs.items()}
    if output_dir.exists() and any(output_dir.iterdir()):
        raise UgiRoleLocalDecoderAuditError(f"output directory is not empty: {output_dir}")

    device = torch.device("cpu")
    seed = int(config["seed"])
    _set_determinism(seed, 2, device)
    training = read_json_object(
        paths["base_training_result"],
        error=UgiRoleLocalDecoderAuditError,
        label="base training result",
    )
    base = config["base_checkpoint"]
    if not isinstance(base, Mapping):
        raise UgiRoleLocalDecoderAuditError("base checkpoint contract is missing")
    try:
        package, member_name = _load_base_package(
            archive_path=paths["base_checkpoint_archive"],
            training=training,
            arm_id=str(base["arm_id"]),
            step=int(base["step"]),
            expected_member_sha256=str(base["member_sha256"]),
            expected_model_state_sha256=str(base["model_state_sha256"]),
            expected_design_sha256=str(sha256_file(paths["base_design"])),
            expected_cache_sha256=str(sha256_file(paths["production_cache"])),
            device=device,
        )
    except ReactionProgramSpecializationError as error:
        raise UgiRoleLocalDecoderAuditError(str(error)) from error

    cache = SynthesisProgramProductionCache(paths["production_cache"])
    try:
        target_program = str(config["target_program"])
        target_state = cache.vocabulary.program_to_index[target_program]
        program_mass = {
            program: float(program == target_program)
            for program in cache.vocabulary.program_states[1:]
        }
        measure = cache.training_measure(program_mass)
        support = np.flatnonzero(measure > 0)
        if len(support) < records:
            raise UgiRoleLocalDecoderAuditError("attribution support is smaller than its sample")
        probabilities = measure[support]
        probabilities /= probabilities.sum()
        rng = np.random.default_rng(seed)
        selected = rng.choice(support, size=records, replace=False, p=probabilities)
        model = build_synthesis_program_flow(
            vocabulary=cache.vocabulary,
            node_classes=len(cache.atom_vocabulary),
            model_config=package["model_config"],
            device=device,
        )
        model.load_state_dict(package["model_state"], strict=True)
        model.eval()
        node_p0 = torch.as_tensor(package["node_marginal"], dtype=torch.float32)
        bond_p0 = torch.as_tensor(package["bond_marginal"], dtype=torch.float32)
        role_names = tuple(cache.vocabulary.role_states[1:])
        time_results: dict[str, Any] = {}
        with torch.no_grad():
            for time_index, flow_time in enumerate(flow_times):
                joint_metrics = _new_role_metrics(role_names)
                teacher_metrics = _new_role_metrics(role_names)
                generator = torch.Generator(device=device).manual_seed(seed + 1000 + time_index)
                for start in range(0, records, batch_size):
                    batch_indices = selected[start : start + batch_size]
                    clean = move_tensors(
                        collate_synthesis_program_training_batch(
                            cache.records(batch_indices),
                            maximum_closures=int(package["model_config"]["maximum_closures"]),
                            conditioning="program",
                            vocabulary=cache.vocabulary,
                        ),
                        device,
                    )
                    if torch.any(clean["program_states"] != target_state):
                        raise UgiRoleLocalDecoderAuditError(
                            "attribution sample escaped the Ugi train-fold support"
                        )
                    t = torch.full((len(batch_indices),), flow_time, dtype=torch.float32)
                    joint, noisy = synthesis_program_forward(
                        model,
                        clean,
                        node_p0,
                        bond_p0,
                        t,
                        generator,
                    )
                    teacher = synthesis_program_topology_conditioned_forward(
                        model,
                        clean,
                        noisy,
                        t,
                    )
                    _accumulate_role_metrics(
                        joint_metrics,
                        joint,
                        clean,
                        role_names,
                        include_topology=True,
                    )
                    _accumulate_role_metrics(
                        teacher_metrics,
                        teacher,
                        clean,
                        role_names,
                        include_topology=False,
                    )
                time_results[f"{flow_time:.6f}"] = {
                    "joint_noisy_state": _finalize(joint_metrics),
                    "target_topology_noisy_chemistry": _finalize(teacher_metrics),
                }
    finally:
        cache.close()

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass",
        "scientific_question": config["scientific_question"],
        "target_program": config["target_program"],
        "seed": seed,
        "records": records,
        "batch_size": batch_size,
        "flow_times": list(flow_times),
        "config": pin_record(config_path, repo),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(paths.items())},
        "base_checkpoint": {
            "member_name": member_name,
            "member_sha256": str(base["member_sha256"]),
            "model_state_sha256": str(base["model_state_sha256"]),
        },
        "training_rows_only": True,
        "selected_cache_row_indices_sha256": hashlib.sha256(
            np.asarray(selected, dtype="<i8").tobytes()
        ).hexdigest(),
        "results_by_flow_time": time_results,
        "training_or_generation_calls": 0,
        "candidate_selection": False,
        "heldout_product_structures_accessed": False,
        "nonclaims": config["nonclaims"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "result.json", result)
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    run_ugi_role_local_decoder_audit(
        args.config.resolve(),
        args.repo.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiRoleLocalDecoderAuditError",
    "run_ugi_role_local_decoder_audit",
]
