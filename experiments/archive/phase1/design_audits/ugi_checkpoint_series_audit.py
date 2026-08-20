"""Audit molecular samples from serial Ugi joint-flow checkpoints."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from forge.core.io import read_json_object
from forge.core.io import write_json as _atomic_json
from forge.model.defog_feasibility import sha256_file


class UgiCheckpointSeriesAuditError(RuntimeError):
    """Raised when serial checkpoint evidence is incomplete or inconsistent."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return read_json_object(path, error=UgiCheckpointSeriesAuditError, label=label)


def _resolve(reference: str, repo: Path) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else repo / path


def _checkpoint_step(sample: dict[str, Any], sample_path: Path) -> tuple[int, Path]:
    try:
        checkpoint_reference = str(sample["checkpoints"]["joint"])
    except (KeyError, TypeError) as error:
        raise UgiCheckpointSeriesAuditError(
            f"sample result lacks a joint checkpoint: {sample_path}"
        ) from error
    checkpoint_path = Path(checkpoint_reference)
    stem = checkpoint_path.stem
    marker = "checkpoint_step_"
    if marker not in stem:
        raise UgiCheckpointSeriesAuditError(
            f"sample result does not reference a serial checkpoint: {sample_path}"
        )
    try:
        step = int(stem.rsplit(marker, 1)[1])
    except ValueError as error:
        raise UgiCheckpointSeriesAuditError(
            f"invalid serial checkpoint step in {sample_path}"
        ) from error
    return step, checkpoint_path


def audit_ugi_checkpoint_series(
    repo: Path,
    *,
    training_result_path: Path,
    program_probe_path: Path,
    sample_result_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Validate and summarize a complete serial molecular-sampling series."""

    training = _load_json(training_result_path, "training result")
    probe = _load_json(program_probe_path, "program probe")
    if training.get("status") != "complete":
        raise UgiCheckpointSeriesAuditError("training result is not complete")
    if probe.get("schema_version") != "phase1_ugi_program_probe.v1" or not probe.get("samples"):
        raise UgiCheckpointSeriesAuditError("program probe is empty or unsupported")

    snapshot_hashes = {
        int(row["step"]): str(row["sha256"]) for row in training.get("checkpoint_snapshots", ())
    }
    if not snapshot_hashes:
        raise UgiCheckpointSeriesAuditError("training result has no serial checkpoints")

    rows = []
    sample_inputs = []
    observed_steps: set[int] = set()
    reference_signature: str | None = None
    for sample_result_path in sample_result_paths:
        sample = _load_json(sample_result_path, "sample result")
        if sample.get("status") != "complete":
            raise UgiCheckpointSeriesAuditError(
                f"sample result is not complete: {sample_result_path}"
            )
        step, checkpoint_reference = _checkpoint_step(sample, sample_result_path)
        if step in observed_steps:
            raise UgiCheckpointSeriesAuditError(f"duplicate sampled checkpoint step: {step}")
        observed_steps.add(step)
        if step not in snapshot_hashes:
            raise UgiCheckpointSeriesAuditError(f"sampled step is not frozen: {step}")

        checkpoint_path = _resolve(str(checkpoint_reference), repo)
        observed_checkpoint_hash = sha256_file(checkpoint_path)
        if observed_checkpoint_hash != snapshot_hashes[step]:
            raise UgiCheckpointSeriesAuditError(f"checkpoint hash mismatch at step {step}")
        matched_probe = _resolve(str(sample.get("matched_staged_result", "")), repo)
        if matched_probe.resolve() != program_probe_path.resolve():
            raise UgiCheckpointSeriesAuditError(
                f"step {step} was not sampled on the declared fixed probe"
            )

        reference = sample.get("reference_comparison")
        statistics = sample.get("statistics")
        if not isinstance(reference, dict) or not isinstance(statistics, dict):
            raise UgiCheckpointSeriesAuditError(
                f"sample result lacks molecular metrics: {sample_result_path}"
            )
        frozen = reference["all_frozen_ugi"]
        generated = reference["generated_descriptors"]
        signature = json.dumps(frozen["reference_descriptors"], sort_keys=True)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise UgiCheckpointSeriesAuditError("reference descriptor summaries changed")

        valid = int(statistics["valid_molecules"])
        unique = int(statistics["unique_valid_molecules"])
        rows.append(
            {
                "step": step,
                "valid_fraction": float(statistics["valid_fraction"]),
                "unique_valid_molecules": unique,
                "unique_fraction_among_valid": float(unique / valid) if valid else 0.0,
                "aromatic_valid_fraction": float(statistics["aromatic_valid_fraction"]),
                "mean_decorations": float(statistics["mean_decorations"]),
                "exact_frozen_product_matches": int(frozen["exact_matches"]),
                "median_nearest_frozen_product_morgan_tanimoto": float(
                    frozen["nearest_morgan_tanimoto"]["median"]
                ),
                "generated_median_heteroatoms": float(generated["heteroatoms"]["median"]),
                "generated_median_logp": float(generated["logp"]["median"]),
                "generated_median_rings": float(generated["rings"]["median"]),
                "generated_median_rotatable_bonds": float(generated["rotatable_bonds"]["median"]),
                "failure_types": statistics["failure_types"],
                "sample_result": {
                    "path": str(sample_result_path),
                    "sha256": sha256_file(sample_result_path),
                },
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": observed_checkpoint_hash,
                },
            }
        )
        sample_inputs.append(str(sample_result_path))

    expected_steps = set(snapshot_hashes)
    if observed_steps != expected_steps:
        missing = sorted(expected_steps - observed_steps)
        extra = sorted(observed_steps - expected_steps)
        raise UgiCheckpointSeriesAuditError(
            f"serial sample set is incomplete; missing={missing}, extra={extra}"
        )
    rows.sort(key=lambda row: int(row["step"]))

    first_reference = _load_json(sample_result_paths[0], "sample result")["reference_comparison"][
        "all_frozen_ugi"
    ]
    result = {
        "schema_version": "phase1_ugi_joint_checkpoint_series_audit.v1",
        "status": "complete",
        "inputs": {
            "training_result": {
                "path": str(training_result_path),
                "sha256": sha256_file(training_result_path),
            },
            "program_probe": {
                "path": str(program_probe_path),
                "sha256": sha256_file(program_probe_path),
            },
            "sample_results": sample_inputs,
        },
        "training_selection": training["selection"],
        "reference": {
            "molecules": int(first_reference["reference_molecules"]),
            "descriptor_medians": {
                key: float(value["median"])
                for key, value in first_reference["reference_descriptors"].items()
            },
        },
        "checkpoints": rows,
        "decision": {
            "production_checkpoint_frozen": False,
            "reason": (
                "Serial sampling exposes a realism-versus-reproduction tradeoff; "
                "held-component and generated-component novelty evaluation remains required."
            ),
            "calibration_best_step": int(training["selection"]["best_step"]),
        },
        "scope_boundary": {
            "product_and_l1_only": True,
            "route_or_oracle_guidance": False,
        },
    }
    _atomic_json(output_path, result)
    return result
