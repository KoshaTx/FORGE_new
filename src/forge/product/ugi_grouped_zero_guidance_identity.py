"""Execute a nonselecting grouped lambda-zero identity qualification.

This module authenticates the selected Ugi generator, the current restartable
equivalence receipt, and the frozen grouped-SMC schedule before exercising one
real 16-program by 4-particle assignment.  Two independently initialized but
identical selected-model pools are advanced through the frozen checkpoints.
Every checkpoint completion is paired bit for bit, the treatment copy receives
identity ancestry only, and every final particle is also compared with the
frozen one-particle selected-generator callback.

The qualification deliberately performs no route evaluation, synthesis-value
calculation, candidate selection, biological scoring, or holdout access.  It is
an execution identity gate, not a nonzero-guidance experiment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_nonzero_guidance_runner import (
    GuidanceTerminalCompletionReceipt,
    ParticleGroupDesign,
    load_grouped_smc_schedule_qualification,
)
from forge.product.ugi_restartable_terminal_support_adapter import (
    native_completion_record_from_locked_terminal,
)
from forge.product.ugi_selected_guidance_adapter import (
    SelectedGuidanceState,
    build_selected_model_restartable_guidance_lane,
)
from forge.product.ugi_selected_restartable_generator import SAMPLE_STEPS
from forge.product.ugi_synthesis_guidance import keyed_random_seed

CONFIG_SCHEMA_VERSION = "phase1_ugi_grouped_zero_guidance_identity_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_grouped_zero_guidance_identity.v1"

_EXPECTED_POLICY = {
    "ancestry": "identity_only",
    "biological_guidance": False,
    "candidate_selection": False,
    "device": "cpu",
    "direct_selected_callback_reference": True,
    "nonzero_guidance": False,
    "particles_per_program": 4,
    "program_count": 16,
    "route_evaluation": False,
    "sealed_holdout_access": False,
    "synthesis_value_evaluation": False,
}

_EXPECTED_INPUTS = {
    "adapter_source",
    "adapter_tests",
    "builder_runner",
    "builder_source",
    "builder_tests",
    "grouped_schedule",
    "nonzero_runner_config",
    "nonzero_runner_plan",
    "nonzero_runner_source",
    "nonzero_runner_tests",
    "restartable_equivalence_v2",
    "selected_generator_implementation",
    "selected_restartable_generator",
}


class UgiGroupedZeroGuidanceIdentityError(RuntimeError):
    """Raised when the grouped lambda-zero identity contract fails."""


def _stable_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise UgiGroupedZeroGuidanceIdentityError(
            "identity record is not canonically serializable"
        ) from error


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiGroupedZeroGuidanceIdentityError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiGroupedZeroGuidanceIdentityError(f"{label} must contain one JSON object")
    return value


def _pin(repo: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise UgiGroupedZeroGuidanceIdentityError(f"{label} pin is malformed")
    path = (repo / str(value.get("path"))).resolve()
    try:
        relative = path.relative_to(repo)
    except ValueError as error:
        raise UgiGroupedZeroGuidanceIdentityError(f"{label} escapes repository") from error
    lowered = "/".join(relative.parts).lower()
    if "holdout" in lowered or "sealed" in lowered:
        raise UgiGroupedZeroGuidanceIdentityError(f"{label} is forbidden")
    if sha256_file(path) != value.get("sha256"):
        raise UgiGroupedZeroGuidanceIdentityError(f"{label} hash changed")
    return path


def _validate_restartable_v2(value: dict[str, Any]) -> None:
    comparisons = value.get("comparisons")
    try:
        valid = (
            value["schema_version"] == "phase1_ugi_restartable_sampler_equivalence.v1"
            and value["status"] == "complete"
            and value["decision"] == "restartable_zero_guidance_schedule_is_bitwise_equivalent"
            and value["sample_steps"] == SAMPLE_STEPS
            and value["production_synthesis_guidance"] is False
            and value["biological_guidance"] is False
            and isinstance(comparisons, list)
            and [row["batch_size"] for row in comparisons] == [1, 4, 12]
            and all(
                row["bitwise_equal"] is True
                and row["terminals"] == 12
                and row["metadata"]["sample_steps"] == SAMPLE_STEPS
                and row["metadata"]["terminal_tree_repairs"] == 0
                for row in comparisons
            )
        )
    except (KeyError, TypeError) as error:
        raise UgiGroupedZeroGuidanceIdentityError("restartable v2 receipt is malformed") from error
    if not valid:
        raise UgiGroupedZeroGuidanceIdentityError(
            "restartable v2 receipt does not preserve the old batch partitions"
        )


def _validate_blocked_runner_plan(value: dict[str, Any]) -> None:
    if (
        value.get("schema_version") != "forge.ugi_nonzero_guidance_runner_plan.v1"
        or value.get("status") != "device_agnostic_runner_prepared_nonzero_execution_blocked"
    ):
        raise UgiGroupedZeroGuidanceIdentityError(
            "nonzero runner plan is not the expected blocked plan"
        )
    scope = value.get("scope")
    authorization = value.get("authorization")
    if not isinstance(scope, Mapping) or not isinstance(authorization, Mapping):
        raise UgiGroupedZeroGuidanceIdentityError("runner plan scope is malformed")
    forbidden_true = (
        "production_nonzero_execution",
        "biological_guidance",
        "sealed_holdout_access",
        "prospective_candidate_lock",
    )
    if any(scope.get(key) is True for key in forbidden_true):
        raise UgiGroupedZeroGuidanceIdentityError(
            "runner plan unexpectedly authorizes forbidden execution scope"
        )
    if authorization.get("nonzero_guidance_authorized") is not False:
        raise UgiGroupedZeroGuidanceIdentityError(
            "runner plan does not remain fail-closed for nonzero guidance"
        )


def _terminal_record(receipt: GuidanceTerminalCompletionReceipt) -> dict[str, Any]:
    terminal = receipt.terminal
    if terminal is None:
        return {
            "completion_error": receipt.error_detail,
            "completion_error_sha256": _sha256_payload(receipt.error_detail),
            "exact_l1": False,
            "terminal_present": False,
            "terminal_valid": False,
        }
    native = native_completion_record_from_locked_terminal(terminal)
    return {
        "exact_l1": terminal.exact_l1,
        "generation_trace_sha256": terminal.generation_trace_sha256,
        "native_completion_sha256": _sha256_payload(native),
        "terminal_id": terminal.terminal_id,
        "terminal_present": True,
        "terminal_sha256": terminal.terminal_sha256,
        "terminal_valid": terminal.terminal_valid,
        "unit_id_sha256": hashlib.sha256(terminal.unit_id.encode()).hexdigest(),
    }


def _assert_completion_identity(
    left: GuidanceTerminalCompletionReceipt,
    right: GuidanceTerminalCompletionReceipt,
    *,
    expected_transition_calls: int,
    label: str,
) -> dict[str, Any]:
    if left.product_transition_calls != expected_transition_calls or (
        right.product_transition_calls != expected_transition_calls
    ):
        raise UgiGroupedZeroGuidanceIdentityError(
            f"{label}: completion transition count differs from the frozen schedule"
        )
    if left.terminal_completions != 1 or right.terminal_completions != 1:
        raise UgiGroupedZeroGuidanceIdentityError(f"{label}: completion accounting changed")
    if left.error_detail != right.error_detail:
        raise UgiGroupedZeroGuidanceIdentityError(f"{label}: completion errors differ")
    if (left.terminal is None) != (right.terminal is None):
        raise UgiGroupedZeroGuidanceIdentityError(f"{label}: terminal presence differs")
    if left.terminal is not None:
        if left.terminal != right.terminal:
            raise UgiGroupedZeroGuidanceIdentityError(f"{label}: locked terminals differ")
        if left.terminal.terminal_bytes != right.terminal.terminal_bytes:
            raise UgiGroupedZeroGuidanceIdentityError(f"{label}: terminal bytes differ")
        if left.terminal.generation_trace_bytes != right.terminal.generation_trace_bytes:
            raise UgiGroupedZeroGuidanceIdentityError(f"{label}: trace bytes differ")
        if native_completion_record_from_locked_terminal(
            left.terminal
        ) != native_completion_record_from_locked_terminal(right.terminal):
            raise UgiGroupedZeroGuidanceIdentityError(f"{label}: native completion records differ")
    return _terminal_record(left)


def _expanded_programs(programs: tuple[bytes, ...]) -> tuple[bytes, ...]:
    return ParticleGroupDesign(
        morphology_program_count=16,
        particles_per_program=4,
    ).expanded_programs(programs)


def _productive_final_invocation_seed(particle_seed: int) -> int:
    if isinstance(particle_seed, bool) or not isinstance(particle_seed, int) or particle_seed < 0:
        raise UgiGroupedZeroGuidanceIdentityError(
            "productive final particle seed must be a nonnegative integer"
        )
    return particle_seed + 1


def build_grouped_zero_guidance_identity(
    repo: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Run one authenticated real selected-model grouped lambda-zero identity gate."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load(config_path, label="grouped lambda-zero identity config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiGroupedZeroGuidanceIdentityError("unsupported identity config schema")
    if config.get("policy") != _EXPECTED_POLICY:
        raise UgiGroupedZeroGuidanceIdentityError("identity policy changed")
    if config.get("checkpoints") != [2, 4, 6] or config.get("sample_steps") != SAMPLE_STEPS:
        raise UgiGroupedZeroGuidanceIdentityError("identity flow schedule changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != _EXPECTED_INPUTS:
        raise UgiGroupedZeroGuidanceIdentityError("identity input set changed")
    paths = {label: _pin(repo, record, label=label) for label, record in inputs.items()}

    restartable = _load(
        paths["restartable_equivalence_v2"],
        label="restartable v2 equivalence receipt",
    )
    _validate_restartable_v2(restartable)
    runner_plan = _load(paths["nonzero_runner_plan"], label="blocked runner plan")
    _validate_blocked_runner_plan(runner_plan)

    schedule = load_grouped_smc_schedule_qualification(paths["grouped_schedule"])
    assignment_seed = config.get("assignment_seed")
    if assignment_seed != 20260821:
        raise UgiGroupedZeroGuidanceIdentityError(
            "only the first frozen calibration assignment is authorized"
        )
    assignments = schedule.by_seed()
    if assignment_seed not in assignments:
        raise UgiGroupedZeroGuidanceIdentityError(
            "requested calibration assignment is absent from the frozen schedule"
        )
    assignment = assignments[assignment_seed]
    if len(assignment.programs) != 16 or len(assignment.stochastic_particle_seeds) != 64:
        raise UgiGroupedZeroGuidanceIdentityError(
            "frozen calibration assignment is not 16 programs by 4 particles"
        )
    programs = _expanded_programs(assignment.programs)
    lane = build_selected_model_restartable_guidance_lane(repo)

    left_initial = lane.initialize(
        programs,
        seed=assignment.seed,
        particle_seeds=assignment.stochastic_particle_seeds,
        device="cpu",
    )
    right_initial = lane.initialize(
        programs,
        seed=assignment.seed,
        particle_seeds=assignment.stochastic_particle_seeds,
        device="cpu",
    )
    if not isinstance(left_initial.state, SelectedGuidanceState) or not isinstance(
        right_initial.state, SelectedGuidanceState
    ):
        raise UgiGroupedZeroGuidanceIdentityError("selected lane returned an untyped state")
    if left_initial.state.state_sha256 != right_initial.state.state_sha256:
        raise UgiGroupedZeroGuidanceIdentityError(
            "identically seeded selected-model pools differ at initialization"
        )
    if (
        left_initial.consumed_particle_seed_manifest_sha256
        != right_initial.consumed_particle_seed_manifest_sha256
    ):
        raise UgiGroupedZeroGuidanceIdentityError("particle seed manifests differ")
    expected_particle_seed_manifest_sha256 = _sha256_payload(assignment.stochastic_particle_seeds)
    if (
        left_initial.consumed_particle_seed_manifest_sha256
        != expected_particle_seed_manifest_sha256
    ):
        raise UgiGroupedZeroGuidanceIdentityError(
            "selected lane did not consume the exact frozen particle-seed manifest"
        )
    for particle_index, (program_bytes, particle_seed) in enumerate(
        zip(programs, assignment.stochastic_particle_seeds, strict=True)
    ):
        for label, state in (("left", left_initial.state), ("right", right_initial.state)):
            particle = state.particles[particle_index]
            if (
                particle.global_particle_index != particle_index
                or particle.program_index != particle_index // 4
                or particle.program_bytes != program_bytes
                or particle.original_particle_seed != particle_seed
            ):
                raise UgiGroupedZeroGuidanceIdentityError(
                    f"{label} particle {particle_index} differs from the frozen assignment"
                )

    left_state = left_initial.state
    right_state = right_initial.state
    checkpoint_records: list[dict[str, Any]] = []
    checkpoint_completion_calls = 0
    transition_calls = 0
    previous_step = 0
    for checkpoint in config["checkpoints"]:
        left_advance = lane.advance(left_state, target_step=checkpoint)
        right_advance = lane.advance(right_state, target_step=checkpoint)
        expected_advance_calls = 64 * (checkpoint - previous_step)
        if (
            left_advance.product_transition_calls != expected_advance_calls
            or right_advance.product_transition_calls != expected_advance_calls
        ):
            raise UgiGroupedZeroGuidanceIdentityError(
                f"checkpoint {checkpoint}: advance accounting changed"
            )
        left_state = left_advance.state
        right_state = right_advance.state
        transition_calls += 2 * expected_advance_calls
        if left_state.state_sha256 != right_state.state_sha256:
            raise UgiGroupedZeroGuidanceIdentityError(
                f"checkpoint {checkpoint}: selected-model pools differ before completion"
            )
        state_sha256_before = left_state.state_sha256
        terminal_records = []
        expected_completion_calls = SAMPLE_STEPS - checkpoint
        for particle_index in range(64):
            program_index = particle_index // 4
            within_program_particle_index = particle_index % 4
            rollout_seed = keyed_random_seed(
                assignment.seed,
                arm="matched_checkpoint_completion",
                program_index=program_index,
                particle_index=within_program_particle_index,
                checkpoint_index=checkpoint,
                rollout_index=0,
            )
            left_completion = lane.complete_terminal(
                left_state,
                particle_index=particle_index,
                seed=rollout_seed,
                checkpoint_index=checkpoint,
            )
            right_completion = lane.complete_terminal(
                right_state,
                particle_index=particle_index,
                seed=rollout_seed,
                checkpoint_index=checkpoint,
            )
            record = _assert_completion_identity(
                left_completion,
                right_completion,
                expected_transition_calls=expected_completion_calls,
                label=f"checkpoint {checkpoint} particle {particle_index}",
            )
            terminal_records.append(
                {
                    "particle_index": particle_index,
                    "program_index": program_index,
                    "within_program_particle_index": within_program_particle_index,
                    "rollout_seed": rollout_seed,
                    **record,
                }
            )
            checkpoint_completion_calls += 2
            transition_calls += 2 * expected_completion_calls
        if (
            left_state.state_sha256 != state_sha256_before
            or right_state.state_sha256 != state_sha256_before
        ):
            raise UgiGroupedZeroGuidanceIdentityError(
                f"checkpoint {checkpoint}: terminal completion mutated a source state"
            )
        identity = tuple(range(64))
        ancestry = lane.apply_ancestry(left_state, identity)
        if ancestry.product_transition_calls != 0:
            raise UgiGroupedZeroGuidanceIdentityError(
                f"checkpoint {checkpoint}: identity ancestry consumed transitions"
            )
        if ancestry.state.state_sha256 != left_state.state_sha256:
            raise UgiGroupedZeroGuidanceIdentityError(
                f"checkpoint {checkpoint}: identity ancestry changed state identity"
            )
        left_state = ancestry.state
        if left_state.state_sha256 != right_state.state_sha256:
            raise UgiGroupedZeroGuidanceIdentityError(
                f"checkpoint {checkpoint}: identity ancestry broke paired equivalence"
            )
        checkpoint_records.append(
            {
                "checkpoint": checkpoint,
                "paired_completion_count": 64,
                "completion_transition_calls_per_arm": 64 * expected_completion_calls,
                "state_sha256": left_state.state_sha256,
                "categorical_state_sha256": left_state.categorical_state_sha256,
                "lineage_sha256": left_state.lineage_sha256,
                "identity_ancestry_sha256_preserved": True,
                "terminal_present_count": sum(
                    item["terminal_present"] for item in terminal_records
                ),
                "terminal_valid_count": sum(item["terminal_valid"] for item in terminal_records),
                "exact_l1_count": sum(item["exact_l1"] for item in terminal_records),
                "terminal_manifest_sha256": _sha256_payload(terminal_records),
                "terminals": terminal_records,
            }
        )
        previous_step = checkpoint

    left_final_advance = lane.advance(left_state, target_step=SAMPLE_STEPS)
    right_final_advance = lane.advance(right_state, target_step=SAMPLE_STEPS)
    expected_final_advance_calls = 64 * (SAMPLE_STEPS - previous_step)
    if (
        left_final_advance.product_transition_calls != expected_final_advance_calls
        or right_final_advance.product_transition_calls != expected_final_advance_calls
    ):
        raise UgiGroupedZeroGuidanceIdentityError("final advance accounting changed")
    transition_calls += 2 * expected_final_advance_calls
    left_state = left_final_advance.state
    right_state = right_final_advance.state
    if left_state.state_sha256 != right_state.state_sha256:
        raise UgiGroupedZeroGuidanceIdentityError("final selected-model pool states differ")
    final_state_sha256 = left_state.state_sha256

    final_records = []
    for particle_index in range(64):
        invocation_seed = _productive_final_invocation_seed(
            assignment.stochastic_particle_seeds[particle_index]
        )
        left_completion = lane.complete_terminal(
            left_state,
            particle_index=particle_index,
            seed=invocation_seed,
            checkpoint_index=SAMPLE_STEPS,
        )
        right_completion = lane.complete_terminal(
            right_state,
            particle_index=particle_index,
            seed=invocation_seed,
            checkpoint_index=SAMPLE_STEPS,
        )
        record = _assert_completion_identity(
            left_completion,
            right_completion,
            expected_transition_calls=0,
            label=f"productive final particle {particle_index}",
        )
        if left_completion.terminal is None:
            raise UgiGroupedZeroGuidanceIdentityError(
                f"productive final particle {particle_index}: selected completion failed"
            )
        request = lane.completion_generation_request(
            left_state,
            particle_index=particle_index,
            invocation_seed=invocation_seed,
            checkpoint_index=SAMPLE_STEPS,
        )
        reference = lane.selected_lane.callback(request)
        if left_completion.terminal != reference:
            raise UgiGroupedZeroGuidanceIdentityError(
                f"productive final particle {particle_index}: direct callback differs"
            )
        if (
            left_completion.terminal.terminal_bytes != reference.terminal_bytes
            or left_completion.terminal.generation_trace_bytes != reference.generation_trace_bytes
            or native_completion_record_from_locked_terminal(left_completion.terminal)
            != native_completion_record_from_locked_terminal(reference)
        ):
            raise UgiGroupedZeroGuidanceIdentityError(
                f"productive final particle {particle_index}: callback bytes differ"
            )
        final_records.append(
            {
                "particle_index": particle_index,
                "program_index": particle_index // 4,
                "within_program_particle_index": particle_index % 4,
                "invocation_seed": invocation_seed,
                "productive_seed": assignment.stochastic_particle_seeds[particle_index],
                "direct_selected_callback_bitwise_equal": True,
                **record,
            }
        )
    if (
        left_state.state_sha256 != final_state_sha256
        or right_state.state_sha256 != final_state_sha256
    ):
        raise UgiGroupedZeroGuidanceIdentityError(
            "productive final completion mutated a source state"
        )

    selected_bindings = lane.selected_lane.bindings
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "selected_model_grouped_lambda_zero_bitwise_qualified",
        "decision": "grouped_lambda_zero_identity_qualified_nonselecting",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {
                "path": str(path.relative_to(repo)),
                "sha256": sha256_file(path),
            }
            for label, path in sorted(paths.items())
        },
        "runtime_versions": restartable["inputs"]["generator_implementation"]["runtime_versions"],
        "selected_generator": {
            "adapter_identity_sha256": lane.adapter_identity_sha256,
            "bindings_sha256": selected_bindings.canonical_sha256,
            "generator_implementation_sha256": (selected_bindings.generator_implementation_sha256),
            "restartable_equivalence_receipt_sha256": (
                selected_bindings.restartable_equivalence_receipt_sha256
            ),
            "sample_steps": SAMPLE_STEPS,
        },
        "assignment": {
            "seed": assignment.seed,
            "schedule_sha256": assignment.schedule_sha256,
            "assignment_sha256": assignment.assignment_sha256,
            "program_count": len(assignment.programs),
            "particles_per_program": 4,
            "particle_count": len(assignment.stochastic_particle_seeds),
            "program_sha256s": list(assignment.program_sha256s),
            "particle_seed_manifest_sha256": expected_particle_seed_manifest_sha256,
        },
        "old_partition_requalification": {
            "batch_sizes": [row["batch_size"] for row in restartable["comparisons"]],
            "terminal_sha256s": [row["terminal_sha256"] for row in restartable["comparisons"]],
            "all_bitwise_equal": True,
        },
        "grouped_execution": {
            "initial_state_sha256": left_initial.state.state_sha256,
            "checkpoint_records": checkpoint_records,
            "final_state_sha256": final_state_sha256,
            "final_records": final_records,
            "final_terminal_manifest_sha256": _sha256_payload(final_records),
            "paired_checkpoint_comparisons": 3 * 64,
            "checkpoint_completion_receipts": checkpoint_completion_calls,
            "paired_productive_final_completions": 128,
            "direct_selected_callback_comparisons": 64,
            "product_transition_calls_across_paired_lanes": transition_calls,
            "all_checkpoint_pairs_bitwise_equal": True,
            "all_identity_ancestry_digest_preserving": True,
            "all_productive_final_pairs_bitwise_equal": True,
            "all_direct_selected_callback_references_bitwise_equal": True,
            "source_states_never_mutated_by_completion": True,
        },
        "scope": {
            "ancestry": "identity_only",
            "biological_guidance": False,
            "candidate_selection": False,
            "nonzero_guidance_authorized": False,
            "private_holdout_accessed": False,
            "production_execution": False,
            "route_evaluation": False,
            "synthesis_value_evaluation": False,
        },
        "nonclaims": [
            "This qualification does not execute nonzero synthesis guidance.",
            "This qualification does not evaluate routes or synthesis values.",
            "This qualification does not select candidates or use biology.",
            "This qualification does not authorize production execution.",
        ],
    }
    return {**result, "result_sha256": _sha256_payload(result)}


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UgiGroupedZeroGuidanceIdentityError",
    "build_grouped_zero_guidance_identity",
]
