#!/usr/bin/env python3
"""Combine integrity, chemistry, branch geometry, and diversity checkpoint evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.corpus.r1_prime_audit import sha256_file
from experiments.archive.phase1.design_audits.ugi_decoration_checkpoint_screen import (
    MINIMUM_EFFECTIVE_COMPONENT_COUNT,
    MINIMUM_UNIQUE_COMPONENTS,
    MINIMUM_UNIQUE_VALID_FRACTION,
    MINIMUM_VALID_FRACTION,
    evaluate_checkpoint_arm,
    pareto_frontier,
)

REPO = Path(__file__).resolve().parents[3]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO / "results/phase1/ugi_decoration_coupling_checkpoint_screen_v1",
    )
    parser.add_argument(
        "--chemistry-audit",
        type=Path,
        default=(
            REPO
            / "results/phase1/ugi_decoration_coupling_checkpoint_screen_v1"
            / "program_matched_tail_chemistry_all_v1.json"
        ),
    )
    parser.add_argument(
        "--branch-audit",
        type=Path,
        default=(
            REPO
            / "results/phase1/ugi_decoration_coupling_checkpoint_screen_v1"
            / "branch_arm_geometry_all_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO
            / "results/phase1/ugi_decoration_coupling_checkpoint_screen_v1"
            / "checkpoint_evaluation_v1.json"
        ),
    )
    args = parser.parse_args()
    chemistry = _load(args.chemistry_audit)
    branch = _load(args.branch_audit)
    if (
        chemistry.get("heldout_fold_used") is not False
        or branch.get("heldout_fold_used") is not False
    ):
        raise ValueError("checkpoint selection audits must exclude heldout reference chemistry")
    arm_names = sorted(set(chemistry["arms"]).intersection(branch["arms"]))
    arms: dict[str, Any] = {}
    inputs: dict[str, Any] = {
        "chemistry_audit": {
            "path": str(args.chemistry_audit),
            "sha256": sha256_file(args.chemistry_audit),
        },
        "branch_audit": {
            "path": str(args.branch_audit),
            "sha256": sha256_file(args.branch_audit),
        },
    }
    for name in arm_names:
        path = args.input_root / name / "result.json"
        sample = _load(path)
        arms[name] = evaluate_checkpoint_arm(
            sample=sample,
            chemistry_arm=chemistry["arms"][name],
            branch_arm=branch["arms"][name],
        )
        inputs[name] = {"path": str(path), "sha256": sha256_file(path)}
    frontier = pareto_frontier(arms)
    output = {
        "schema_version": "phase1_ugi_decoration_checkpoint_evaluation.v1",
        "status": "complete_selection_screen_not_final_promotion",
        "heldout_reference_used": False,
        "hard_gates": {
            "minimum_valid_fraction": MINIMUM_VALID_FRACTION,
            "minimum_unique_valid_fraction": MINIMUM_UNIQUE_VALID_FRACTION,
            "minimum_unique_components": MINIMUM_UNIQUE_COMPONENTS,
            "minimum_effective_component_count": MINIMUM_EFFECTIVE_COMPONENT_COUNT,
            "exact_l1_fraction_of_valid": 1.0,
            "adjacent_tail_branch_component_fraction": 0.0,
        },
        "schedule_sensitive_diversity_diagnostics": {
            "top_1_and_top_5_component_fractions_are_not_hard_gates": True,
            "rationale": (
                "the checkpoint block has a different morphology allocation from the earlier "
                "architecture-selection draw; concentration is compared with the program-balanced "
                "training reference instead of transferring an unconditional ceiling"
            ),
        },
        "pareto_definition": {
            "eligibility": "passes every frozen integrity and collapse gate",
            "maximize": ["valid_fraction"],
            "minimize": [
                "program_balanced_local_chemistry_mae",
                "program_matched_branch_geometry_jsd_bits",
            ],
            "branch_geometry_use": (
                "the joint categorical geometry signature is compared by Jensen-Shannon "
                "divergence, avoiding a hand-weighted combination of branch counts, depths, and "
                "arm lengths; it must be rechecked in independent confirmation"
            ),
        },
        "inputs": inputs,
        "arms": arms,
        "descriptive_pareto_frontier": frontier,
        "decision": {
            "production_checkpoint_promoted": None,
            "independent_multiseed_confirmation_required": True,
        },
    }
    _atomic_json(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "descriptive_pareto_frontier": frontier,
                "arms": {
                    name: {
                        "passes_all_hard_gates": arm["passes_all_hard_gates"],
                        **arm["pareto_axes"],
                    }
                    for name, arm in arms.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
