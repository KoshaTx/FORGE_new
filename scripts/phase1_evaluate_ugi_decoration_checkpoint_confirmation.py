#!/usr/bin/env python3
"""Apply the frozen final-duration rule to independent checkpoint confirmation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any

from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_decoration_checkpoint_screen import (
    evaluate_checkpoint_arm,
    select_confirmed_training_duration,
)
from forge.product.ugi_joint_end_to_end_sampling import (
    joint_sampling_result_matches_request,
)

REPO = Path(__file__).resolve().parents[1]


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


def _verify_policy_inputs(policy: Mapping[str, Any]) -> None:
    specifications = [
        *policy["screen_evidence"].values(),
        *policy["finalists"].values(),
        *policy["implementation"].values(),
        policy["confirmation_design"]["program_schedule"],
        policy["reference_policy"]["training_cache"],
    ]
    for specification in specifications:
        if not isinstance(specification, Mapping) or "path" not in specification:
            continue
        path = REPO / str(specification["path"])
        if sha256_file(path) != specification["sha256"]:
            raise ValueError(f"frozen confirmation input changed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=Path,
        default=(
            REPO / "configs/model/phase1_ugi_decoration_checkpoint_confirmation_policy_v1.json"
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO / "results/phase1/ugi_decoration_coupling_checkpoint_confirmation_v1",
    )
    parser.add_argument(
        "--chemistry-audit",
        type=Path,
        default=(
            REPO
            / "results/phase1/ugi_decoration_coupling_checkpoint_confirmation_v1"
            / "program_matched_tail_chemistry_v1.json"
        ),
    )
    parser.add_argument(
        "--branch-audit",
        type=Path,
        default=(
            REPO
            / "results/phase1/ugi_decoration_coupling_checkpoint_confirmation_v1"
            / "branch_arm_geometry_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO
            / "results/phase1/ugi_decoration_coupling_checkpoint_confirmation_v1"
            / "evaluation_v1.json"
        ),
    )
    args = parser.parse_args()
    policy = _load(args.policy)
    if policy.get("status") != "frozen_before_independent_confirmation_sampling":
        raise ValueError("confirmation policy is not frozen")
    _verify_policy_inputs(policy)
    if sha256_file(Path(__file__)) != policy["implementation"]["confirmation_evaluator"]["sha256"]:
        raise ValueError("confirmation evaluator changed after policy freeze")
    chemistry = _load(args.chemistry_audit)
    branch = _load(args.branch_audit)
    if (
        chemistry.get("heldout_fold_used") is not False
        or branch.get("heldout_fold_used") is not False
    ):
        raise ValueError("confirmation selection cannot use heldout reference chemistry")

    replicates = {
        int(row["replicate"]): row for row in policy["confirmation_design"]["paired_replicates"]
    }
    by_step: dict[str, Any] = {}
    inputs: dict[str, Any] = {
        "policy": {"path": str(args.policy), "sha256": sha256_file(args.policy)},
        "chemistry_audit": {
            "path": str(args.chemistry_audit),
            "sha256": sha256_file(args.chemistry_audit),
        },
        "branch_audit": {
            "path": str(args.branch_audit),
            "sha256": sha256_file(args.branch_audit),
        },
    }
    for raw_step, checkpoint in policy["finalists"].items():
        step = int(raw_step)
        replicate_results = {}
        for replicate, seeds in sorted(replicates.items()):
            name = f"step_{step}_replicate_{replicate}"
            path = args.input_root / f"step_{step}" / f"replicate_{replicate}" / "result.json"
            sample = _load(path)
            if not (
                str(sample.get("matched_staged_result", "")).endswith(
                    policy["confirmation_design"]["program_schedule"]["path"]
                )
                and joint_sampling_result_matches_request(
                    sample,
                    seed=int(seeds["flow_seed"]),
                    program_offset=int(
                        policy["confirmation_design"]["program_schedule"]["program_offset"]
                    ),
                    program_limit=int(
                        policy["confirmation_design"]["program_schedule"]["program_limit"]
                    ),
                    terminal_decoder_mode="stochastic",
                    terminal_decoder_seed=int(seeds["terminal_seed"]),
                    terminal_temperature=float(
                        policy["confirmation_design"]["terminal_temperature"]
                    ),
                    checkpoint_filename=Path(checkpoint["path"]).name,
                )
            ):
                raise ValueError(f"confirmation result disagrees with policy: {name}")
            replicate_results[str(replicate)] = evaluate_checkpoint_arm(
                sample=sample,
                chemistry_arm=chemistry["arms"][name],
                branch_arm=branch["arms"][name],
            )
            inputs[name] = {"path": str(path), "sha256": sha256_file(path)}
        by_step[str(step)] = {
            "replicates": replicate_results,
            "passes_all_hard_gates": all(
                result["passes_all_hard_gates"] for result in replicate_results.values()
            ),
            "mean_axes": {
                "valid_fraction": mean(
                    result["valid_fraction"] for result in replicate_results.values()
                ),
                "program_balanced_local_chemistry_mae": mean(
                    result["pareto_axes"]["minimize_program_balanced_local_chemistry_mae"]
                    for result in replicate_results.values()
                ),
                "program_matched_branch_geometry_jsd_bits": mean(
                    result["pareto_axes"]["minimize_program_matched_branch_geometry_jsd_bits"]
                    for result in replicate_results.values()
                ),
            },
        }

    margins = policy["selection_margins"]
    decision = select_confirmed_training_duration(
        by_step,
        chemistry_noninferiority_margin=float(margins["chemistry_noninferiority_absolute_mae"]),
        branch_jsd_noninferiority_margin_bits=float(
            margins["branch_geometry_noninferiority_jsd_bits"]
        ),
        validity_tie_margin=float(margins["validity_tie_absolute_fraction"]),
    )
    output = {
        "schema_version": "phase1_ugi_decoration_checkpoint_confirmation_evaluation.v1",
        "status": (
            "complete_training_duration_selected"
            if decision["selected_checkpoint_step"] is not None
            else "complete_no_training_duration_selected"
        ),
        "heldout_reference_used": False,
        "inputs": inputs,
        "by_step": by_step,
        "decision": {
            **decision,
            "selected_object": (
                "fixed training duration for a from-scratch all-fold production refit; the "
                "development checkpoint itself is not the production generator"
            ),
        },
    }
    _atomic_json(args.output, output)
    print(json.dumps({"output": str(args.output), **output["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
