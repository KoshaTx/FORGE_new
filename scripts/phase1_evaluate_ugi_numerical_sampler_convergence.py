#!/usr/bin/env python3
"""Evaluate numerical convergence of the categorical Ugi joint-flow sampler."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from forge.bio.ugi_semantic_annotations import ROLE_NAMES
from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_terminal_decoder_challenger import (
    TAIL_ROLES,
    summarize_terminal_decoder_arm,
)

REPO = Path(__file__).resolve().parents[1]
SAMPLE_STEPS = (8, 16, 32, 64)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _normalized(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0:
        raise ValueError("distribution has no mass")
    return values.astype(np.float64, copy=False) / total


def _distribution_distance(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    p = _normalized(left)
    q = _normalized(right)
    midpoint = 0.5 * (p + q)

    def kl_divergence(first: np.ndarray, second: np.ndarray) -> float:
        active = first > 0
        return float(np.sum(first[active] * np.log(first[active] / second[active])))

    return {
        "total_variation": float(0.5 * np.abs(p - q).sum()),
        "jensen_shannon_nats": float(
            0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)
        ),
    }


def _generated_role_vector(summary: Mapping[str, Any], role: str) -> np.ndarray:
    buckets = summary["by_role_distance"][role]
    return np.asarray(
        [
            state
            for distance in sorted(buckets, key=int)
            for state in buckets[distance]["atom_state_counts"]
        ],
        dtype=np.float64,
    )


def _reference_role_vector(reference: Mapping[str, Any], role: str) -> np.ndarray:
    buckets = reference["role_distance_atom_reference"]["by_role_distance"][role]
    return np.asarray(
        [
            state
            for distance in sorted(buckets, key=int)
            for state in buckets[distance]["atom_state_mass"]
        ],
        dtype=np.float64,
    )


def _atom_state_distances(
    summary: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    return {
        role: _distribution_distance(
            _generated_role_vector(summary, role),
            _reference_role_vector(reference, role),
        )
        for role in ROLE_NAMES
    }


def _successive_atom_distances(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    return {
        role: _distribution_distance(
            _generated_role_vector(left, role),
            _generated_role_vector(right, role),
        )
        for role in ROLE_NAMES
    }


def _tail_feature_error(arm: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    by_role: dict[str, Any] = {}
    absolute_errors = []
    for role in TAIL_ROLES:
        observed = arm["tail_chemotypes_exact_l1_eligible_only"][role][
            "feature_occurrence_fractions"
        ]
        target = reference["role_component_reference"][role]["feature_occurrence_fractions"]
        errors = {name: abs(float(observed[name]) - float(target[name])) for name in target}
        absolute_errors.extend(errors.values())
        by_role[role] = {
            "observed": observed,
            "target": target,
            "absolute_error": errors,
            "mean_absolute_error": float(np.mean(tuple(errors.values()))),
        }
    return {
        "by_role": by_role,
        "mean_absolute_error_across_tail_role_features": float(np.mean(absolute_errors)),
    }


def _assert_matched_programs(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    left_rows = left.get("samples")
    right_rows = right.get("samples")
    if not isinstance(left_rows, list) or not isinstance(right_rows, list):
        raise ValueError("numerical arms lack sample rows")
    if len(left_rows) != len(right_rows):
        raise ValueError("numerical arms have different sample counts")
    for index, (left_row, right_row) in enumerate(zip(left_rows, right_rows, strict=True)):
        for field in ("product_id", "program", "source_stratum", "branch_class"):
            if left_row.get(field) != right_row.get(field):
                raise ValueError(f"matched program row {index} differs at {field}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_numerical_sampler_convergence_v1.json",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO / "results/phase1/ugi_numerical_sampler_convergence_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(REPO / "results/phase1/ugi_numerical_sampler_convergence_v1/evaluation.json"),
    )
    args = parser.parse_args()

    config = _load(args.config)
    if config.get("schema_version") != "phase1_ugi_numerical_sampler_convergence_config.v1":
        raise ValueError("unexpected numerical-convergence config schema")
    if tuple(config["design"]["sample_steps"]) != SAMPLE_STEPS:
        raise ValueError("numerical grids changed after the audit was frozen")
    verified_inputs: dict[str, Any] = {}
    for name, specification in config["inputs"].items():
        path = REPO / specification["path"]
        digest = sha256_file(path)
        if digest != specification["sha256"]:
            raise ValueError(f"frozen input changed: {name}")
        verified_inputs[name] = {"path": specification["path"], "sha256": digest}
    reference = _load(REPO / config["inputs"]["balanced_training_reference"]["path"])
    raw = {
        steps: _load(args.input_root / f"steps_{steps:02d}" / "result.json")
        for steps in SAMPLE_STEPS
    }
    for steps in SAMPLE_STEPS[1:]:
        _assert_matched_programs(raw[SAMPLE_STEPS[0]], raw[steps])

    arms: dict[str, Any] = {}
    for steps, result in raw.items():
        arm = summarize_terminal_decoder_arm(result)
        endpoint_summary = result["statistics"]["flow_endpoint_atom_states"]
        terminal_summary = result["statistics"]["terminal_atom_states"]
        arms[str(steps)] = {
            **arm,
            "sample_steps": steps,
            "runtime_seconds": float(result["sampling"]["total_seconds"]),
            "flow_endpoint_states_available": int(endpoint_summary["available_terminal_states"]),
            "corrected_terminal_states_available": int(terminal_summary["valid_terminal_samples"]),
            "flow_endpoint_atom_distance_from_training": _atom_state_distances(
                endpoint_summary, reference
            ),
            "corrected_terminal_atom_distance_from_training": _atom_state_distances(
                terminal_summary, reference
            ),
            "tail_feature_error_from_training": _tail_feature_error(arm, reference),
        }

    successive: dict[str, Any] = {}
    for left_steps, right_steps in zip(SAMPLE_STEPS[:-1], SAMPLE_STEPS[1:], strict=True):
        left = raw[left_steps]
        right = raw[right_steps]
        left_arm = arms[str(left_steps)]
        right_arm = arms[str(right_steps)]
        successive[f"{left_steps}_to_{right_steps}"] = {
            "flow_endpoint_atom_distribution": _successive_atom_distances(
                left["statistics"]["flow_endpoint_atom_states"],
                right["statistics"]["flow_endpoint_atom_states"],
            ),
            "corrected_terminal_atom_distribution": _successive_atom_distances(
                left["statistics"]["terminal_atom_states"],
                right["statistics"]["terminal_atom_states"],
            ),
            "valid_fraction_delta": right_arm["valid_fraction"] - left_arm["valid_fraction"],
            "exact_l1_fraction_of_valid_delta": right_arm["exact_l1_fraction_of_valid"]
            - left_arm["exact_l1_fraction_of_valid"],
            "tail_feature_mean_absolute_error_delta": right_arm["tail_feature_error_from_training"][
                "mean_absolute_error_across_tail_role_features"
            ]
            - left_arm["tail_feature_error_from_training"][
                "mean_absolute_error_across_tail_role_features"
            ],
            "runtime_multiplier": right_arm["runtime_seconds"] / left_arm["runtime_seconds"],
        }

    output = {
        "schema_version": "phase1_ugi_numerical_sampler_convergence_evaluation.v1",
        "status": "complete_selection_visible_numerical_sweep",
        "config": {
            "path": str(args.config.relative_to(REPO)),
            "sha256": sha256_file(args.config),
        },
        "inputs": verified_inputs,
        "matched_program_rows_verified": True,
        "arms": arms,
        "successive_grid_comparisons": successive,
        "interpretation_policy": {
            "raw_endpoint_is_pre_terminal_feasibility_correction": True,
            "terminal_distribution_is_stochastic_feasibility_corrected": True,
            "atom_or_motif_quota_fitted": False,
            "terminal_temperature_fitted": False,
            "production_policy_changed": False,
            "independent_confirmation_required_after_step_selection": True,
        },
    }
    if not all(
        math.isfinite(float(value)) for arm in arms.values() for value in [arm["valid_fraction"]]
    ):
        raise ValueError("numerical audit produced a non-finite statistic")
    _atomic_json(args.output, output)
    print(json.dumps({"output": str(args.output), "successive": successive}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
