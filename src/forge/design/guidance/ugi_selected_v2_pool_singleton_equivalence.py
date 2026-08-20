"""Pooled-versus-singleton restartable equivalence for selected Ugi v2.

The selected guidance lane stores 64 independently seeded particles in one
controller state.  This audit proves that each pooled particle has the same
categorical and RNG state as a separately initialized singleton at steps
0, 2, 4, 6 and 8.  Checkpoint completions are compared with the corresponding
singleton continuation, and productive finals are also compared with the
direct selected generator callback.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.data.r1_prime_audit import sha256_file
from forge.design.flow.ugi_restartable_terminal_support_adapter import (
    native_completion_record_from_locked_terminal,
)
from forge.design.flow.ugi_selected_guidance_adapter_v2 import (
    build_selected_model_restartable_guidance_lane_v2,
)
from forge.design.guidance.ugi_synthesis_guidance import keyed_random_seed
from forge.design.sampling.ugi_selected_restartable_generator import SAMPLE_STEPS
from forge.design.sampling.ugi_selected_restartable_generator_v2 import (
    GENERATOR_CHECKPOINT_SHA256,
    MAXIMUM_ADJACENT_BRANCH_RUNS,
    TERMINAL_DECODER_ID,
)
from forge.design.schedule.ugi_nonzero_guidance_runner import (
    GuidanceTerminalCompletionReceipt,
    ParticleGroupDesign,
    load_grouped_smc_schedule_qualification,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_selected_v2_pool_singleton_equivalence_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_selected_v2_pool_singleton_equivalence.v1"


class UgiSelectedV2PoolSingletonEquivalenceError(RuntimeError):
    """Raised when pooled and singleton restartable execution differ."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiSelectedV2PoolSingletonEquivalenceError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiSelectedV2PoolSingletonEquivalenceError(f"{label} must be a JSON object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise UgiSelectedV2PoolSingletonEquivalenceError(f"{label} pin is malformed")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiSelectedV2PoolSingletonEquivalenceError(
            f"{label} path escapes repository"
        ) from error
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiSelectedV2PoolSingletonEquivalenceError(f"{label} hash changed")
    return path


def _normalized_completion(
    receipt: GuidanceTerminalCompletionReceipt,
) -> dict[str, Any]:
    if receipt.terminal is None:
        detail = receipt.error_detail or ""
        return {
            "status": "completion_error",
            "error_detail": detail,
            "product_transition_calls": receipt.product_transition_calls,
        }
    terminal = receipt.terminal
    candidate = native_completion_record_from_locked_terminal(terminal)
    return {
        "status": "terminal",
        "terminal_valid": terminal.terminal_valid,
        "exact_l1": terminal.exact_l1,
        "terminal_bytes_sha256": hashlib.sha256(terminal.terminal_bytes).hexdigest(),
        "candidate_record_sha256": _sha256_payload(candidate),
        "product_transition_calls": receipt.product_transition_calls,
    }


def _exact_completion(receipt: GuidanceTerminalCompletionReceipt) -> dict[str, Any]:
    normalized = _normalized_completion(receipt)
    if receipt.terminal is None:
        return normalized
    terminal = receipt.terminal
    return {
        **normalized,
        "unit_id": terminal.unit_id,
        "terminal_id": terminal.terminal_id,
        "morphology_program_sha256": terminal.morphology_program_sha256,
        "checkpoint_index": terminal.checkpoint_index,
        "generator_checkpoint_sha256": terminal.generator_checkpoint_sha256,
        "closure_checkpoint_sha256": terminal.closure_checkpoint_sha256,
        "generation_trace_sha256": hashlib.sha256(terminal.generation_trace_bytes).hexdigest(),
    }


def _particle_state_record(particle: Any) -> dict[str, Any]:
    return {
        "program_sha256": particle.program_sha256,
        "categorical_state_sha256": particle.categorical_state_sha256,
        "rng_state_sha256": particle.rng_state_sha256,
        "step": particle.trajectory.step,
    }


def build_selected_v2_pool_singleton_equivalence(
    repo: Path,
    config_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute the selected-v2 pooled/singleton/direct equivalence contract."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load(config_path, label="pool-singleton equivalence config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiSelectedV2PoolSingletonEquivalenceError("unsupported pool-singleton config schema")
    if config.get("scope") != {
        "guidance": False,
        "routing": False,
        "biology": False,
        "candidate_selection": False,
        "holdout_access": False,
    }:
        raise UgiSelectedV2PoolSingletonEquivalenceError("equivalence scope changed")
    design_value = config.get("design")
    if design_value != {
        "assignment_seed": 20260821,
        "morphology_program_count": 16,
        "particles_per_program": 4,
        "sample_steps": 8,
        "state_steps": [0, 2, 4, 6, 8],
        "completion_checkpoints": [2, 4, 6],
    }:
        raise UgiSelectedV2PoolSingletonEquivalenceError("equivalence design changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise UgiSelectedV2PoolSingletonEquivalenceError("equivalence inputs are missing")
    paths = {label: _pin(repo, value, label=label) for label, value in inputs.items()}

    qualification = load_grouped_smc_schedule_qualification(paths["grouped_schedule"])
    assignment = qualification.by_seed()[design_value["assignment_seed"]]
    design = ParticleGroupDesign(
        morphology_program_count=design_value["morphology_program_count"],
        particles_per_program=design_value["particles_per_program"],
    )
    expanded_programs = design.expanded_programs(assignment.programs)
    lane = build_selected_model_restartable_guidance_lane_v2(repo)
    pooled_receipt = lane.initialize(
        expanded_programs,
        seed=assignment.seed,
        particle_seeds=assignment.stochastic_particle_seeds,
        device="cpu",
    )
    pooled_state = pooled_receipt.state
    singleton_states = []
    for program, particle_seed in zip(
        expanded_programs,
        assignment.stochastic_particle_seeds,
        strict=True,
    ):
        singleton_states.append(
            lane.initialize(
                (program,),
                seed=assignment.seed,
                particle_seeds=(particle_seed,),
                device="cpu",
            ).state
        )

    rows: list[dict[str, Any]] = []

    def compare_states(step: int) -> None:
        for particle_index, singleton in enumerate(singleton_states):
            pooled_record = _particle_state_record(pooled_state.particles[particle_index])
            singleton_record = _particle_state_record(singleton.particles[0])
            if pooled_record != singleton_record:
                raise UgiSelectedV2PoolSingletonEquivalenceError(
                    f"pooled/singleton state differs at step {step}, particle {particle_index}"
                )
            rows.append(
                {
                    "comparison": "state",
                    "step": step,
                    "particle_index": particle_index,
                    "pooled": pooled_record,
                    "singleton": singleton_record,
                    "equal": True,
                }
            )

    compare_states(0)
    previous = 0
    completion_checkpoints = set(design_value["completion_checkpoints"])
    for step in design_value["state_steps"][1:]:
        pooled_advanced = lane.advance(pooled_state, target_step=step)
        expected_pooled_calls = design.particle_count * (step - previous)
        if pooled_advanced.product_transition_calls != expected_pooled_calls:
            raise UgiSelectedV2PoolSingletonEquivalenceError("pooled transition accounting changed")
        pooled_state = pooled_advanced.state
        advanced_singletons = []
        for singleton in singleton_states:
            advanced = lane.advance(singleton, target_step=step)
            if advanced.product_transition_calls != step - previous:
                raise UgiSelectedV2PoolSingletonEquivalenceError(
                    "singleton transition accounting changed"
                )
            advanced_singletons.append(advanced.state)
        singleton_states = advanced_singletons
        compare_states(step)

        if step in completion_checkpoints:
            for particle_index, singleton in enumerate(singleton_states):
                program_index = particle_index // design.particles_per_program
                local_index = particle_index % design.particles_per_program
                seed = keyed_random_seed(
                    assignment.seed,
                    arm="matched_checkpoint_completion",
                    program_index=program_index,
                    particle_index=local_index,
                    checkpoint_index=step,
                    rollout_index=0,
                )
                pooled_before = pooled_state.state_sha256
                singleton_before = singleton.state_sha256
                pooled_completion = lane.complete_terminal(
                    pooled_state,
                    particle_index=particle_index,
                    seed=seed,
                    checkpoint_index=step,
                )
                singleton_completion = lane.complete_terminal(
                    singleton,
                    particle_index=0,
                    seed=seed,
                    checkpoint_index=step,
                )
                pooled_value = _normalized_completion(pooled_completion)
                singleton_value = _normalized_completion(singleton_completion)
                if pooled_value != singleton_value:
                    raise UgiSelectedV2PoolSingletonEquivalenceError(
                        f"checkpoint completion differs at step {step}, particle {particle_index}"
                    )
                if (
                    pooled_before != pooled_state.state_sha256
                    or singleton_before != singleton.state_sha256
                ):
                    raise UgiSelectedV2PoolSingletonEquivalenceError(
                        "checkpoint completion mutated its source state"
                    )
                rows.append(
                    {
                        "comparison": "checkpoint_completion",
                        "step": step,
                        "particle_index": particle_index,
                        "program_index": program_index,
                        "seed": seed,
                        "pooled": pooled_value,
                        "singleton": singleton_value,
                        "source_states_unchanged": True,
                        "equal": True,
                    }
                )
        previous = step

    for particle_index, (particle_seed, singleton) in enumerate(
        zip(assignment.stochastic_particle_seeds, singleton_states, strict=True)
    ):
        invocation_seed = particle_seed + 1
        pooled_completion = lane.complete_terminal(
            pooled_state,
            particle_index=particle_index,
            seed=invocation_seed,
            checkpoint_index=SAMPLE_STEPS,
        )
        singleton_completion = lane.complete_terminal(
            singleton,
            particle_index=0,
            seed=invocation_seed,
            checkpoint_index=SAMPLE_STEPS,
        )
        pooled_value = _normalized_completion(pooled_completion)
        singleton_value = _normalized_completion(singleton_completion)
        if pooled_value != singleton_value:
            raise UgiSelectedV2PoolSingletonEquivalenceError(
                f"productive pooled/singleton completion differs at particle {particle_index}"
            )
        request = lane.completion_generation_request(
            pooled_state,
            particle_index=particle_index,
            invocation_seed=invocation_seed,
            checkpoint_index=SAMPLE_STEPS,
        )
        direct_completion = GuidanceTerminalCompletionReceipt(
            terminal=lane.callback(request),
            product_transition_calls=0,
        )
        pooled_exact = _exact_completion(pooled_completion)
        direct_exact = _exact_completion(direct_completion)
        if pooled_exact != direct_exact:
            raise UgiSelectedV2PoolSingletonEquivalenceError(
                f"productive direct callback differs at particle {particle_index}"
            )
        rows.append(
            {
                "comparison": "productive_completion",
                "step": SAMPLE_STEPS,
                "particle_index": particle_index,
                "program_index": particle_index // design.particles_per_program,
                "seed": particle_seed,
                "pooled": pooled_value,
                "singleton": singleton_value,
                "direct": direct_exact,
                "pooled_exact": pooled_exact,
                "equal": True,
            }
        )

    result_content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "selected_v2_pool_singleton_and_direct_callback_equivalent",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "selected_generator": {
            "checkpoint_sha256": GENERATOR_CHECKPOINT_SHA256,
            "terminal_decoder_id": TERMINAL_DECODER_ID,
            "maximum_adjacent_branch_runs": list(MAXIMUM_ADJACENT_BRANCH_RUNS),
            "base_adapter_identity_sha256": lane.adapter_identity_sha256,
        },
        "execution": {
            "assignment_seed": assignment.seed,
            "assignment_sha256": assignment.assignment_sha256,
            "particles": design.particle_count,
            "state_comparisons": design.particle_count * len(design_value["state_steps"]),
            "checkpoint_completion_comparisons": design.particle_count
            * len(design_value["completion_checkpoints"]),
            "productive_singleton_comparisons": design.particle_count,
            "productive_direct_callback_comparisons": design.particle_count,
            "source_state_mutations": 0,
            "all_equal": True,
            "comparison_rows_sha256": _sha256_payload(rows),
        },
        "scope": {
            **config["scope"],
            "production_execution": False,
        },
    }
    return (
        {**result_content, "result_sha256": _sha256_payload(result_content)},
        rows,
    )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UgiSelectedV2PoolSingletonEquivalenceError",
    "build_selected_v2_pool_singleton_equivalence",
]
