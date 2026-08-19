#!/usr/bin/env python3
"""Evaluate the matched atom/decorations stochastic-channel diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_terminal_decoder_challenger import (
    TAIL_ROLES,
    _assert_matched_rows,
    summarize_terminal_decoder_arm,
)

REPO = Path(__file__).resolve().parents[1]
MODES = (
    "bond_stochastic",
    "atom_bond_stochastic",
    "decoration_bond_stochastic",
    "stochastic",
)


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


def _arm_delta(baseline: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    tail_deltas: dict[str, Any] = {}
    for role in TAIL_ROLES:
        left = baseline["tail_chemotypes_exact_l1_eligible_only"][role]
        right = challenger["tail_chemotypes_exact_l1_eligible_only"][role]
        tail_deltas[role] = {
            "effective_exact_component_count": right["effective_exact_component_count"]
            - left["effective_exact_component_count"],
            "effective_occurrence_weighted_chemotype_count": right[
                "effective_occurrence_weighted_chemotype_count"
            ]
            - left["effective_occurrence_weighted_chemotype_count"],
            "feature_occurrence_fractions": {
                name: right["feature_occurrence_fractions"][name]
                - left["feature_occurrence_fractions"][name]
                for name in sorted(left["feature_occurrence_fractions"])
            },
        }
    return {
        "valid_fraction": challenger["valid_fraction"] - baseline["valid_fraction"],
        "exact_l1_fraction_of_valid": challenger["exact_l1_fraction_of_valid"]
        - baseline["exact_l1_fraction_of_valid"],
        "exact_l1_eligible_molecules": challenger["exact_l1_eligible_molecules"]
        - baseline["exact_l1_eligible_molecules"],
        "semantic_failures_among_valid": {
            name: challenger["semantic_failures_among_valid"][name]
            - baseline["semantic_failures_among_valid"][name]
            for name in baseline["semantic_failures_among_valid"]
        },
        "tail_chemotype_deltas": tail_deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_stochastic_channel_isolation_v1.json",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO / "results/phase1/ugi_stochastic_channel_isolation_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/ugi_stochastic_channel_isolation_v1/evaluation.json",
    )
    args = parser.parse_args()

    config = _load(args.config)
    if config.get("schema_version") != "phase1_ugi_stochastic_channel_isolation_config.v1":
        raise ValueError("unexpected channel-isolation config schema")
    if tuple(config["design"]["matched_terminal_decoder_modes"]) != MODES:
        raise ValueError("decoder arms changed after the diagnostic was frozen")
    verified_inputs: dict[str, Any] = {}
    for name, specification in config["inputs"].items():
        path = REPO / specification["path"]
        digest = sha256_file(path)
        if digest != specification["sha256"]:
            raise ValueError(f"frozen input changed: {name}")
        verified_inputs[name] = {"path": specification["path"], "sha256": digest}

    raw = {mode: _load(args.input_root / mode / "result.json") for mode in MODES}
    baseline_raw = raw["bond_stochastic"]
    for mode in MODES[1:]:
        _assert_matched_rows(baseline_raw, raw[mode])
    summaries = {mode: summarize_terminal_decoder_arm(raw[mode]) for mode in MODES}
    baseline = summaries["bond_stochastic"]
    comparisons = {mode: _arm_delta(baseline, summaries[mode]) for mode in MODES[1:]}
    output = {
        "schema_version": "phase1_ugi_stochastic_channel_isolation_evaluation.v1",
        "status": "complete_selection_visible_channel_isolation",
        "config": {
            "path": str(args.config.relative_to(REPO)),
            "sha256": sha256_file(args.config),
        },
        "inputs": verified_inputs,
        "draws": {
            mode: {
                "path": str((args.input_root / mode / "result.json").relative_to(REPO)),
                "sha256": sha256_file(args.input_root / mode / "result.json"),
            }
            for mode in MODES
        },
        "matched_topology_rows_verified": True,
        "arms": summaries,
        "comparisons_to_bond_stochastic": comparisons,
        "interpretation_policy": {
            "atom_effect": "atom_bond_stochastic minus bond_stochastic",
            "decoration_effect": "decoration_bond_stochastic minus bond_stochastic",
            "full_effect": "stochastic minus bond_stochastic",
            "production_policy_changed": False,
            "calibration_selected": False,
            "independent_confirmation_required": True,
        },
    }
    _atomic_json(args.output, output)
    print(json.dumps({"output": str(args.output), "comparisons": comparisons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
