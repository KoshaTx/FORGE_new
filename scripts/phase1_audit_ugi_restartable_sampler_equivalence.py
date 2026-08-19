#!/usr/bin/env python3
"""Audit restartable sampling against the frozen monolithic schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from forge.product.defog_feasibility import sha256_file
from forge.product.ugi_end_to_end_sampling import _atomic_json, _load_checkpoint
from forge.product.ugi_joint_end_to_end_sampling import _load_matched_programs
from forge.product.ugi_joint_sparse_flow import (
    UgiJointSparseFlow,
    UgiJointSparseTerminal,
    _legacy_sample_ugi_joint_sparse_terminals,
    sample_ugi_joint_sparse_terminals,
)
from forge.product.ugi_selected_generator_implementation import (
    build_selected_generator_implementation_qualification,
)


class RestartableSamplerAuditError(RuntimeError):
    """Raised when the frozen zero-guidance equivalence contract fails."""


def _terminal_sha256(terminals: list[UgiJointSparseTerminal]) -> str:
    digest = hashlib.sha256()
    for terminal in terminals:
        digest.update(
            json.dumps(terminal.program.__dict__, sort_keys=True, separators=(",", ":")).encode()
        )
        for offspring in terminal.offspring:
            digest.update(np.ascontiguousarray(offspring).tobytes())
        for values in (
            terminal.atom_logits,
            terminal.parent_bond_logits,
            terminal.decoration_anchor_logits,
            terminal.decoration_atom_logits,
            terminal.decoration_bond_logits,
            terminal.hidden,
        ):
            contiguous = np.ascontiguousarray(values)
            digest.update(str(contiguous.dtype).encode())
            digest.update(json.dumps(contiguous.shape).encode())
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _assert_equal(
    expected: list[UgiJointSparseTerminal],
    observed: list[UgiJointSparseTerminal],
) -> None:
    if len(expected) != len(observed):
        raise RestartableSamplerAuditError("terminal counts differ")
    for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
        if left.program != right.program:
            raise RestartableSamplerAuditError(f"program differs at terminal {index}")
        arrays = (*left.offspring, left.atom_logits, left.parent_bond_logits)
        comparisons = (*right.offspring, right.atom_logits, right.parent_bond_logits)
        arrays += (
            left.decoration_anchor_logits,
            left.decoration_atom_logits,
            left.decoration_bond_logits,
            left.hidden,
        )
        comparisons += (
            right.decoration_anchor_logits,
            right.decoration_atom_logits,
            right.decoration_bond_logits,
            right.hidden,
        )
        if any(
            not np.array_equal(expected_array, observed_array)
            for expected_array, observed_array in zip(arrays, comparisons, strict=True)
        ):
            raise RestartableSamplerAuditError(f"terminal tensor differs at index {index}")


def build_audit(repo: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "phase1_ugi_restartable_sampler_equivalence_config.v1":
        raise RestartableSamplerAuditError("unsupported audit config")
    checkpoint_path = repo / config["joint_checkpoint"]["path"]
    program_path = repo / config["program_draw"]["path"]
    for spec, path in (
        (config["joint_checkpoint"], checkpoint_path),
        (config["program_draw"], program_path),
    ):
        if sha256_file(path) != spec["sha256"]:
            raise RestartableSamplerAuditError(f"input hash mismatch: {path}")
    implementation = build_selected_generator_implementation_qualification(repo)
    if implementation.implementation_sha256 != config.get("generator_implementation_sha256"):
        raise RestartableSamplerAuditError(
            "productive generator implementation differs from the frozen config"
        )
    checkpoint = _load_checkpoint(checkpoint_path, "phase1_ugi_joint_sparse_checkpoint.v1")
    programs, _ = _load_matched_programs(program_path)
    programs = programs[: int(config["program_count"])]
    architecture = dict(checkpoint["model_config"])
    architecture.pop("source_probability_floor")
    atom_classes = len(checkpoint["source_marginals"]["atoms"][0])
    model = UgiJointSparseFlow(atom_classes=atom_classes, **architecture)
    model.load_state_dict(checkpoint["model_state"])
    sources = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in checkpoint["source_marginals"].items()
    }
    comparisons = []
    for batch_size in config["batch_sizes"]:
        arguments = {
            "sample_steps": int(config["sample_steps"]),
            "batch_size": int(batch_size),
            "seed": int(config["seed"]),
            "device": str(config["device"]),
        }
        expected, expected_metadata = _legacy_sample_ugi_joint_sparse_terminals(
            model, programs, sources, **arguments
        )
        observed, observed_metadata = sample_ugi_joint_sparse_terminals(
            model, programs, sources, **arguments
        )
        if expected_metadata != observed_metadata:
            raise RestartableSamplerAuditError("sampler metadata differs")
        _assert_equal(expected, observed)
        expected_hash = _terminal_sha256(expected)
        observed_hash = _terminal_sha256(observed)
        if expected_hash != observed_hash:
            raise RestartableSamplerAuditError("terminal hashes differ")
        comparisons.append(
            {
                "batch_size": batch_size,
                "terminals": len(observed),
                "bitwise_equal": True,
                "terminal_sha256": observed_hash,
                "metadata": observed_metadata,
            }
        )
    return {
        "schema_version": "phase1_ugi_restartable_sampler_equivalence.v1",
        "status": "complete",
        "scope": config["scope"],
        "production_synthesis_guidance": False,
        "biological_guidance": False,
        "inputs": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "joint_checkpoint": config["joint_checkpoint"],
            "program_draw": config["program_draw"],
            "generator_implementation": implementation.to_dict(),
        },
        "program_count": len(programs),
        "sample_steps": int(config["sample_steps"]),
        "seed": int(config["seed"]),
        "comparisons": comparisons,
        "decision": "restartable_zero_guidance_schedule_is_bitwise_equivalent",
        "nonclaims": [
            "This audit does not qualify a synthesis value.",
            "This audit does not authorize production synthesis guidance.",
            "This audit does not evaluate biological guidance.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_restartable_sampler_equivalence_v1.json"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else repo / args.config
    config = json.loads(config_path.read_text())
    output_dir = repo / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_audit(repo, config_path)
    _atomic_json(output_dir / "result.json", result)


if __name__ == "__main__":
    main()
