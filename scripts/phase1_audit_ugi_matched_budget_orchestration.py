#!/usr/bin/env python3
"""Freeze the synthetic matched-budget guided/post-hoc orchestration audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from forge.design.flow.defog_feasibility import sha256_file
from forge.design.sampling.ugi_end_to_end_sampling import _atomic_json
from forge.design.schedule.ugi_matched_budget_orchestration import (
    DiagnosticRouteAssessment,
    LockedMatchedTerminal,
    MatchedBudgetLimits,
    MatchedGenerationRequest,
    MatchedScheduleEntry,
    RouteComputeUsage,
    run_diagnostic_matched_budget_arms,
)


class MatchedBudgetAuditError(RuntimeError):
    """Raised when the frozen synthetic orchestration audit changes."""


def _verified_input(repo: Path, specification: dict[str, str]) -> dict[str, str]:
    path = repo / specification["path"]
    observed = sha256_file(path)
    if observed != specification["sha256"]:
        raise MatchedBudgetAuditError(f"input hash mismatch: {path}")
    return {"path": specification["path"], "sha256": observed}


def _route_usage(payload: dict[str, Any]) -> RouteComputeUsage:
    return RouteComputeUsage(
        logical_planner_calls=int(payload["logical_planner_calls"]),
        physical_planner_calls=int(payload["physical_planner_calls"]),
        logical_verifier_calls=int(payload["logical_verifier_calls"]),
        physical_verifier_calls=int(payload["physical_verifier_calls"]),
    )


def _schedule(config: dict[str, Any]) -> tuple[MatchedScheduleEntry, ...]:
    return tuple(
        MatchedScheduleEntry(
            unit_id=entry["unit_id"],
            morphology_program=entry["morphology_program_utf8"].encode(),
            program_index=int(entry["program_index"]),
            particle_index=int(entry["particle_index"]),
            checkpoint_index=int(entry["checkpoint_index"]),
            generator_checkpoint_sha256=config["generator_checkpoint_sha256"],
            closure_checkpoint_sha256=config["closure_checkpoint_sha256"],
            rollout_index=int(entry["rollout_index"]),
            productive_generation_calls=int(entry["productive_generation_calls"]),
            route_reservation=_route_usage(entry["route_reservation"]),
        )
        for entry in config["schedule"]
    )


def build_audit(repo: Path, config_path: Path) -> dict[str, Any]:
    """Run the deterministic fake-value contract and return its owned result."""

    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "phase1_ugi_matched_budget_orchestration_config.v1":
        raise MatchedBudgetAuditError("unsupported audit config")
    inputs = {
        name: _verified_input(repo, specification)
        for name, specification in sorted(config["inputs"].items())
    }
    events: list[str] = []
    cache_by_clone: dict[str, set[str]] = {}

    def generate(request: MatchedGenerationRequest) -> LockedMatchedTerminal:
        events.append(f"lock:{request.arm.value}:{request.entry.unit_id}")
        return LockedMatchedTerminal(
            unit_id=request.entry.unit_id,
            morphology_program_sha256=request.entry.morphology_program_sha256,
            checkpoint_index=request.entry.checkpoint_index,
            generator_checkpoint_sha256=request.entry.generator_checkpoint_sha256,
            closure_checkpoint_sha256=request.entry.closure_checkpoint_sha256,
            terminal_id=f"{request.arm.value}:{request.entry.unit_id}",
            terminal_locked=True,
            terminal_valid=True,
            exact_l1=True,
            terminal_bytes=(
                f"fake-terminal:{request.entry.unit_id}:{request.productive_seed}".encode()
            ),
            generation_trace_bytes=(
                f"fake-trace:{request.entry.unit_id}:{request.productive_seed}".encode()
            ),
        )

    def assess(terminal, context):
        events.append(f"assess:{context.arm.value}:{terminal.unit_id}")
        cache = cache_by_clone.setdefault(context.cache_clone_id, set())
        cache_hit = "fake-shared-route-key" in cache
        cache.add("fake-shared-route-key")
        return DiagnosticRouteAssessment(
            value_policy_id="fake-structured-value-v1",
            diagnostic_fields=(
                ("closure_state", "fake-complete"),
                ("evidence_state", "fake-diagnostic"),
            ),
            usage=RouteComputeUsage(
                logical_planner_calls=1,
                physical_planner_calls=0 if cache_hit else 1,
                logical_verifier_calls=1,
                physical_verifier_calls=1,
            ),
        )

    budget_payload = config["budget_limits"]
    run = run_diagnostic_matched_budget_arms(
        _schedule(config),
        budget_limits=MatchedBudgetLimits(
            productive_generation_calls=int(budget_payload["productive_generation_calls"]),
            terminal_completions=int(budget_payload["terminal_completions"]),
            final_candidates=int(budget_payload["final_candidates"]),
            route=_route_usage(budget_payload["route"]),
        ),
        base_seed=int(config["seed"]),
        zero_guidance=config["zero_guidance"],
        cache_snapshot_sha256=config["cache_snapshot_sha256"],
        generate_terminal=generate,
        assess_terminal=assess,
    )
    if run.shared_censored_unit_ids != tuple(config["expected_shared_censored_unit_ids"]):
        raise MatchedBudgetAuditError("shared censored suffix changed")
    if not run.zero_guidance_bitwise_equivalent:
        raise MatchedBudgetAuditError("zero-guidance byte equivalence failed")
    if run.guided_cache_clone_id == run.post_hoc_cache_clone_id:
        raise MatchedBudgetAuditError("matched arms reused one cache namespace")
    post_hoc_locks = [
        index for index, event in enumerate(events) if event.startswith("lock:post_hoc:")
    ]
    post_hoc_assessments = [
        index for index, event in enumerate(events) if event.startswith("assess:post_hoc:")
    ]
    if post_hoc_assessments and max(post_hoc_locks) >= min(post_hoc_assessments):
        raise MatchedBudgetAuditError("post-hoc planner ran before terminal manifest lock")
    if run.guided_ledger.route.physical_planner_calls != 1:
        raise MatchedBudgetAuditError("guided cache-clone physical-call accounting changed")
    if run.post_hoc_ledger.route.physical_planner_calls != 1:
        raise MatchedBudgetAuditError("post-hoc inherited a warm guided cache")

    return {
        "schema_version": "phase1_ugi_matched_budget_orchestration.v1",
        "status": "complete",
        "scope": config["scope"],
        "seed": int(config["seed"]),
        "inputs": {
            "config": {
                "path": str(config_path.relative_to(repo)),
                "sha256": sha256_file(config_path),
            },
            **inputs,
        },
        "matched_contract": asdict(run),
        "event_contract": {
            "events": events,
            "all_post_hoc_terminals_locked_before_route_assessment": True,
            "cache_snapshot_sha256": run.cache_snapshot_sha256,
            "guided_cache_clone_id": run.guided_cache_clone_id,
            "post_hoc_cache_clone_id": run.post_hoc_cache_clone_id,
            "warm_cache_inheritance": False,
        },
        "decision": "diagnostic_matched_budget_orchestration_contract_qualified",
        "production_synthesis_guidance": False,
        "biological_guidance": False,
        "candidate_selection_performed": False,
        "scalar_synthesis_value_defined": False,
        "real_route_planner_executed": False,
        "sealed_holdout_accessed": False,
        "remaining_blockers": [
            "No production synthesis scalar policy is frozen.",
            "No real route value or planner/cache implementation is connected to this contract.",
            "Current exact route support remains insufficient for production guidance.",
        ],
        "nonclaims": [
            "Fake structured fields are not synthesis-success probabilities.",
            "This audit does not run synthesis guidance or candidate selection.",
            "This audit does not evaluate biological guidance.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_matched_budget_orchestration_v1.json"),
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
