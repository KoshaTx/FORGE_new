"""Full grouped equivalence audit for the selected v2 Ugi generator.

This audit compares all 64 restartable particles from one frozen grouped
assignment with an independent low-level completion at checkpoints 2, 4 and 6,
and with the direct selected step-2000 callback at the productive terminal.
It performs no route assessment, guidance, selection, biology or holdout work.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_joint_end_to_end_sampling import complete_ugi_joint_terminals
from forge.product.ugi_joint_sparse_sampling import (
    advance_ugi_joint_sparse_state,
    finalize_ugi_joint_sparse_state,
)
from forge.product.ugi_nonzero_guidance_runner import (
    GuidanceTerminalCompletionReceipt,
    ParticleGroupDesign,
    load_grouped_smc_schedule_qualification,
)
from forge.product.ugi_selected_guidance_adapter import _replace_generator_state
from forge.product.ugi_selected_guidance_adapter_v2 import (
    SelectedModelRestartableGuidanceLaneV2,
    build_selected_model_restartable_guidance_lane_v2,
)
from forge.product.ugi_selected_restartable_generator import SAMPLE_STEPS
from forge.product.ugi_selected_restartable_generator_v2 import (
    GENERATOR_CHECKPOINT_SHA256,
    MAXIMUM_ADJACENT_BRANCH_RUNS,
    TERMINAL_DECODER_ID,
    TERMINAL_TEMPERATURE,
)
from forge.product.ugi_synthesis_guidance import keyed_random_seed

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


CONFIG_SCHEMA_VERSION = "phase1_ugi_selected_v2_full_equivalence_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_selected_v2_full_equivalence.v1"


class UgiSelectedV2FullEquivalenceError(RuntimeError):
    """Raised when the restartable selected-v2 lane differs from its reference."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiSelectedV2FullEquivalenceError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiSelectedV2FullEquivalenceError(f"{label} must be a JSON object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise UgiSelectedV2FullEquivalenceError(f"{label} pin is malformed")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiSelectedV2FullEquivalenceError(f"{label} path escapes repository") from error
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiSelectedV2FullEquivalenceError(f"{label} hash changed")
    return path


def _terminal_record(receipt: GuidanceTerminalCompletionReceipt) -> dict[str, Any]:
    if receipt.terminal is None:
        return {
            "status": "completion_error",
            "error_detail": receipt.error_detail,
            "product_transition_calls": receipt.product_transition_calls,
        }
    terminal = receipt.terminal
    return {
        "status": "terminal",
        "terminal_id": terminal.terminal_id,
        "terminal_valid": terminal.terminal_valid,
        "exact_l1": terminal.exact_l1,
        "terminal_bytes_sha256": hashlib.sha256(terminal.terminal_bytes).hexdigest(),
        "generation_trace_sha256": hashlib.sha256(terminal.generation_trace_bytes).hexdigest(),
        "product_transition_calls": receipt.product_transition_calls,
    }


def _reference_checkpoint_completion(
    lane: SelectedModelRestartableGuidanceLaneV2,
    state: Any,
    *,
    particle_index: int,
    seed: int,
    checkpoint: int,
) -> GuidanceTerminalCompletionReceipt:
    """Complete one checkpoint through the low-level sparse and chemistry APIs."""

    if torch is None:
        raise UgiSelectedV2FullEquivalenceError("selected-v2 equivalence requires torch")
    current = lane._require_current_state(state)
    particle = current.particles[particle_index]
    transition_calls = SAMPLE_STEPS - current.step
    trajectory = particle.trajectory.clone()
    try:
        if transition_calls:
            rollout_state = torch.Generator(device=trajectory.device).manual_seed(seed).get_state()
            trajectory = _replace_generator_state(trajectory, rollout_state)
            trajectory = advance_ugi_joint_sparse_state(
                lane.callback.model,
                trajectory,
                target_step=SAMPLE_STEPS,
            )
        finalization = finalize_ugi_joint_sparse_state(
            lane.callback.model,
            trajectory,
            tree_generator_state=torch.Generator().manual_seed(seed + 1).get_state(),
            allowed_ring_sizes=lane.callback.allowed_ring_sizes,
            maximum_heavy_degree=lane.callback.maximum_heavy_degree,
            maximum_adjacent_branch_runs=MAXIMUM_ADJACENT_BRANCH_RUNS,
        )
        if len(finalization.terminals) != 1:
            raise UgiSelectedV2FullEquivalenceError(
                "reference checkpoint did not finalize exactly one sparse terminal"
            )
        completion = complete_ugi_joint_terminals(
            lane.callback.model,
            lane.callback.closure_model,
            finalization.terminals,
            lane.callback.corpus,
            program_metadata=({},),
            closure_generator_state=torch.Generator().manual_seed(seed + 1).get_state(),
            allowed_ring_sizes=lane.callback.allowed_ring_sizes,
            maximum_heavy_degree=lane.callback.maximum_heavy_degree,
            l1_reaction=lane.callback.reaction,
            terminal_decoder_mode="bond_stochastic",
            terminal_generator_state=torch.Generator().manual_seed(seed + 2).get_state(),
            terminal_temperature=TERMINAL_TEMPERATURE,
        )
        if len(completion.rows) != 1:
            raise UgiSelectedV2FullEquivalenceError(
                "reference checkpoint did not produce exactly one completion row"
            )
        request = lane.completion_generation_request(
            current,
            particle_index=particle_index,
            invocation_seed=seed,
            checkpoint_index=checkpoint,
        )
        terminal = lane._lock_completion_row(completion.rows[0], request)
    except Exception as error:
        return GuidanceTerminalCompletionReceipt(
            terminal=None,
            product_transition_calls=transition_calls,
            error_detail=f"{type(error).__name__}: {error}",
        )
    return GuidanceTerminalCompletionReceipt(
        terminal=terminal,
        product_transition_calls=transition_calls,
    )


def _require_equal(
    observed: GuidanceTerminalCompletionReceipt,
    reference: GuidanceTerminalCompletionReceipt,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    left = _terminal_record(observed)
    right = _terminal_record(reference)
    if left != right:
        raise UgiSelectedV2FullEquivalenceError(f"selected-v2 equivalence failed: {label}")
    return left, right


def build_selected_v2_full_equivalence(
    repo: Path,
    config_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute the frozen 64-particle selected-v2 equivalence audit."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load(config_path, label="selected-v2 equivalence config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiSelectedV2FullEquivalenceError("unsupported equivalence config schema")
    if config.get("scope") != {
        "guidance": False,
        "routing": False,
        "biology": False,
        "candidate_selection": False,
        "holdout_access": False,
    }:
        raise UgiSelectedV2FullEquivalenceError("equivalence scope changed")
    design = config.get("design")
    if design != {
        "assignment_seed": 20260821,
        "morphology_program_count": 16,
        "particles_per_program": 4,
        "sample_steps": 8,
        "checkpoints": [2, 4, 6],
    }:
        raise UgiSelectedV2FullEquivalenceError("equivalence design changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise UgiSelectedV2FullEquivalenceError("equivalence inputs are missing")
    paths = {label: _pin(repo, value, label=label) for label, value in inputs.items()}

    qualification = load_grouped_smc_schedule_qualification(paths["grouped_schedule"])
    assignment = qualification.by_seed()[design["assignment_seed"]]
    group_design = ParticleGroupDesign(
        morphology_program_count=design["morphology_program_count"],
        particles_per_program=design["particles_per_program"],
    )
    programs = group_design.expanded_programs(assignment.programs)
    lane = build_selected_model_restartable_guidance_lane_v2(repo)
    initialized = lane.initialize(
        programs,
        seed=assignment.seed,
        particle_seeds=assignment.stochastic_particle_seeds,
        device="cpu",
    )
    state = initialized.state
    rows: list[dict[str, Any]] = []
    previous = 0
    for checkpoint in design["checkpoints"]:
        advanced = lane.advance(state, target_step=checkpoint)
        expected_calls = group_design.particle_count * (checkpoint - previous)
        if advanced.product_transition_calls != expected_calls:
            raise UgiSelectedV2FullEquivalenceError(
                "checkpoint transition count differs from frozen schedule"
            )
        state = advanced.state
        for particle_index in range(group_design.particle_count):
            program_index = particle_index // group_design.particles_per_program
            local_index = particle_index % group_design.particles_per_program
            seed = keyed_random_seed(
                assignment.seed,
                arm="matched_checkpoint_completion",
                program_index=program_index,
                particle_index=local_index,
                checkpoint_index=checkpoint,
                rollout_index=0,
            )
            observed = lane.complete_terminal(
                state,
                particle_index=particle_index,
                seed=seed,
                checkpoint_index=checkpoint,
            )
            reference = _reference_checkpoint_completion(
                lane,
                state,
                particle_index=particle_index,
                seed=seed,
                checkpoint=checkpoint,
            )
            left, right = _require_equal(
                observed,
                reference,
                label=f"checkpoint={checkpoint},particle={particle_index}",
            )
            rows.append(
                {
                    "stage": "checkpoint",
                    "checkpoint": checkpoint,
                    "particle_index": particle_index,
                    "program_index": program_index,
                    "seed": seed,
                    "observed": left,
                    "reference": right,
                    "equal": True,
                }
            )
        previous = checkpoint

    advanced = lane.advance(state, target_step=SAMPLE_STEPS)
    if advanced.product_transition_calls != group_design.particle_count * (SAMPLE_STEPS - previous):
        raise UgiSelectedV2FullEquivalenceError(
            "productive transition count differs from frozen schedule"
        )
    state = advanced.state
    for particle_index, particle_seed in enumerate(assignment.stochastic_particle_seeds):
        invocation_seed = particle_seed + 1
        observed = lane.complete_terminal(
            state,
            particle_index=particle_index,
            seed=invocation_seed,
            checkpoint_index=SAMPLE_STEPS,
        )
        request = lane.completion_generation_request(
            state,
            particle_index=particle_index,
            invocation_seed=invocation_seed,
            checkpoint_index=SAMPLE_STEPS,
        )
        direct = GuidanceTerminalCompletionReceipt(
            terminal=lane.callback(request),
            product_transition_calls=0,
        )
        left, right = _require_equal(
            observed,
            direct,
            label=f"productive,particle={particle_index}",
        )
        rows.append(
            {
                "stage": "productive",
                "checkpoint": SAMPLE_STEPS,
                "particle_index": particle_index,
                "program_index": particle_index // group_design.particles_per_program,
                "seed": particle_seed,
                "observed": left,
                "reference": right,
                "equal": True,
            }
        )

    result_content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "selected_v2_grouped_restartable_and_direct_callback_equivalent",
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
            "adapter_identity_sha256": lane.adapter_identity_sha256,
        },
        "execution": {
            "assignment_seed": assignment.seed,
            "assignment_sha256": assignment.assignment_sha256,
            "particles": group_design.particle_count,
            "checkpoint_comparisons": group_design.particle_count * len(design["checkpoints"]),
            "productive_direct_callback_comparisons": group_design.particle_count,
            "all_equal": True,
            "comparison_rows_sha256": _sha256_payload(rows),
        },
        "scope": config["scope"],
    }
    return (
        {**result_content, "result_sha256": _sha256_payload(result_content)},
        rows,
    )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UgiSelectedV2FullEquivalenceError",
    "build_selected_v2_full_equivalence",
]
