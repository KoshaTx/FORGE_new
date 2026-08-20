#!/usr/bin/env python3
"""Freeze the diagnostic fixed-budget rollout/controller plumbing audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from forge.model.defog_feasibility import sha256_file
from experiments.phase1.product_l1.sampling.ugi_end_to_end_sampling import _atomic_json
from experiments.phase1.synthesis_guidance.guidance.ugi_synthesis_guidance import (
    LockedRolloutTerminal,
    RolloutBudgetLimits,
    TerminalValueEvaluation,
    run_fixed_budget_terminal_rollouts,
    select_smc_ancestry,
)


class FixedBudgetRolloutAuditError(RuntimeError):
    """Raised when the diagnostic rollout plumbing violates its contract."""


def _verified_input(repo: Path, specification: dict[str, Any]) -> dict[str, str]:
    path = repo / specification["path"]
    observed = sha256_file(path)
    if observed != specification["sha256"]:
        raise FixedBudgetRolloutAuditError(f"input hash mismatch: {path}")
    return {"path": str(path), "sha256": observed}


def _serialized_records(batch: Any) -> list[dict[str, Any]]:
    return [asdict(record) for record in batch.records]


def build_audit(repo: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "phase1_ugi_fixed_budget_rollout_plumbing_config.v1":
        raise FixedBudgetRolloutAuditError("unsupported audit config")
    restartable = _verified_input(repo, config["restartable_sampler_result"])
    structured_value = _verified_input(repo, config["structured_value_result"])
    evaluation_calls: list[int] = []

    def complete(particle: int, rollout: int, seed: int) -> LockedRolloutTerminal:
        index = particle * int(config["rollouts_per_particle"]) + rollout
        return LockedRolloutTerminal(
            terminal_id=f"diagnostic-{index}-{seed}",
            terminal_locked=True,
            terminal_valid=index != 1,
            exact_l1=index not in {1, 2},
            payload={"index": index},
        )

    def evaluate(
        terminal: LockedRolloutTerminal,
        remaining_planner: int,
        remaining_verifier: int,
    ) -> TerminalValueEvaluation:
        if not terminal.terminal_locked or not terminal.terminal_valid or not terminal.exact_l1:
            raise FixedBudgetRolloutAuditError("evaluator received an inadmissible terminal")
        if remaining_planner < 1 or remaining_verifier < 1:
            raise FixedBudgetRolloutAuditError("evaluator received insufficient budget")
        index = int(terminal.payload["index"])
        evaluation_calls.append(index)
        return TerminalValueEvaluation(
            value=float(index),
            value_policy_id="fake-diagnostic-v1",
            logical_planner_calls=1,
            verifier_calls=1,
        )

    budget = RolloutBudgetLimits(**config["budget_limits"])
    arguments = {
        "particle_count": int(config["particle_count"]),
        "rollouts_per_particle": int(config["rollouts_per_particle"]),
        "base_seed": int(config["seed"]),
        "arm": "diagnostic-guided",
        "program_index": int(config["program_index"]),
        "checkpoint_index": int(config["checkpoint_index"]),
        "budget_limits": budget,
        "complete_terminal": complete,
        "evaluate_terminal": evaluate,
    }
    observed = run_fixed_budget_terminal_rollouts(**arguments)
    first_evaluation_calls = tuple(evaluation_calls)
    evaluation_calls.clear()
    repeated = run_fixed_budget_terminal_rollouts(**arguments)
    if _serialized_records(observed) != _serialized_records(repeated):
        raise FixedBudgetRolloutAuditError("keyed rollout records are not deterministic")
    if tuple(evaluation_calls) != first_evaluation_calls:
        raise FixedBudgetRolloutAuditError("terminal evaluation order is not deterministic")
    dispositions = [record.disposition.value for record in observed.records]
    if dispositions != config["expected_dispositions"]:
        raise FixedBudgetRolloutAuditError("terminal dispositions differ from frozen expectation")
    if first_evaluation_calls != (0, 3):
        raise FixedBudgetRolloutAuditError("partial or inadmissible states reached the evaluator")

    zero = select_smc_ancestry(
        np.zeros(2),
        np.asarray([100.0, -100.0]),
        np.zeros(2),
        current_beta=1.0,
        previous_beta=0.0,
        guidance_strength=0.0,
        base_seed=int(config["seed"]),
        arm="diagnostic-guided",
        program_index=int(config["program_index"]),
        checkpoint_index=int(config["checkpoint_index"]),
        rollout_index=0,
    )
    analytic = select_smc_ancestry(
        np.zeros(2),
        np.asarray([0.0, np.log(3.0)]),
        np.zeros(2),
        current_beta=1.0,
        previous_beta=0.0,
        guidance_strength=1.0,
        base_seed=int(config["seed"]),
        arm="diagnostic-guided",
        program_index=int(config["program_index"]),
        checkpoint_index=int(config["checkpoint_index"]),
        rollout_index=0,
    )
    if not np.array_equal(zero.ancestors, np.arange(2)) or zero.resampled:
        raise FixedBudgetRolloutAuditError("zero guidance did not retain identity ancestry")
    if not np.allclose(analytic.probabilities, [0.25, 0.75]):
        raise FixedBudgetRolloutAuditError("analytic fake-value ancestry law changed")

    script_path = Path(__file__).resolve()
    guidance_path = repo / "src/forge/product/ugi_synthesis_guidance.py"
    sampling_path = repo / "src/forge/product/ugi_joint_sparse_sampling.py"
    return {
        "schema_version": "phase1_ugi_fixed_budget_rollout_plumbing.v1",
        "status": "complete",
        "scope": config["scope"],
        "production_synthesis_guidance": False,
        "biological_guidance": False,
        "inputs": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "restartable_sampler_result": restartable,
            "structured_value_result": structured_value,
            "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
            "guidance_source": {
                "path": str(guidance_path),
                "sha256": sha256_file(guidance_path),
            },
            "sampling_source": {
                "path": str(sampling_path),
                "sha256": sha256_file(sampling_path),
            },
        },
        "rollout_contract": {
            "particle_count": observed.particle_count,
            "rollouts_per_particle": observed.rollouts_per_particle,
            "requested_rollouts": len(observed.records),
            "terminal_completions": observed.terminal_completions,
            "logical_planner_calls": observed.logical_planner_calls,
            "verifier_calls": observed.verifier_calls,
            "budget_exhausted": observed.budget_exhausted,
            "dispositions": dispositions,
            "route_evaluated_indices": list(first_evaluation_calls),
            "deterministic_repetition": True,
        },
        "controller_contract": {
            "zero_guidance_identity": True,
            "zero_guidance_consumed_resampling_seed": zero.keyed_seed is not None,
            "analytic_two_particle_probabilities": analytic.probabilities.tolist(),
            "keyed_resampling_seed": analytic.keyed_seed,
        },
        "decision": "diagnostic_fixed_budget_terminal_rollout_plumbing_qualified",
        "remaining_blockers": [
            "No production synthesis scalar policy is frozen.",
            "No real route value is connected to the controller.",
            "Matched guided and post-hoc arm orchestration is not implemented.",
            "Current exact route support remains too sparse for production guidance.",
        ],
        "nonclaims": [
            "Fake diagnostic values are not synthesis-success probabilities.",
            "This audit does not authorize production synthesis guidance.",
            "This audit does not evaluate biological guidance.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_fixed_budget_rollout_plumbing_v1.json"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else repo / args.config
    config = json.loads(config_path.read_text())
    output_dir = repo / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "result.json", build_audit(repo, config_path))


if __name__ == "__main__":
    main()
