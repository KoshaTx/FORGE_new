"""Matched selected-v3 HeLa potency-only diagnostic orchestration.

The orchestration is generic over the eight already-frozen 16x4 schedules. It
first executes a global lambda-zero identity gate for every seed and only then
permits the nonzero diagnostic.  The post-hoc arm receives the same generator
schedule and terminal-completion attempts, but its potency assessments occur
only after the productive pool is locked and never alter ancestry.

This source owns no execution authorization and performs no synthesis,
retrosynthesis, proposal-model, candidate-selection or prospective-lock work.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from experiments.phase1.hela_potency.evaluation import (
    HeLaBatchPredictor,
    HeLaPotencyDiagnosticPolicy,
    HeLaPotencyEvaluation,
)
from experiments.phase1.synthesis_guidance.guidance.ugi_synthesis_guidance import (
    keyed_random_seed,
    select_smc_ancestry,
)
from experiments.phase1.synthesis_guidance.schedule.ugi_nonzero_guidance_runner import (
    FrozenSeedProgramAssignment,
    GroupedSMCScheduleQualification,
    GuidanceSchedule,
    GuidanceStateReceipt,
    GuidanceTerminalCompletionReceipt,
    ParticleGroupDesign,
    RestartableGuidanceLane,
)
from forge.core.hashing import sha256_json as _sha256_payload
from forge.synthesis.matched import LockedMatchedTerminal

RESULT_SCHEMA_VERSION = "phase1_ugi_hela_potency_guidance_diagnostic.v1"
EXPECTED_SEEDS = tuple(range(20260821, 20260829))
CALIBRATION_SEEDS = (20260821, 20260822, 20260823)
EVALUATION_SEEDS = (20260824, 20260825, 20260826, 20260827, 20260828)


class UgiHeLaPotencyGuidanceSeamError(RuntimeError):
    """Raised when matched potency orchestration violates its frozen design."""


def _terminal_identity(terminal: LockedMatchedTerminal | None) -> dict[str, Any]:
    if terminal is None:
        return {"terminal": None}
    return {
        "terminal_id": terminal.terminal_id,
        "terminal_sha256": terminal.terminal_sha256,
        "generation_trace_sha256": terminal.generation_trace_sha256,
        "terminal_valid": terminal.terminal_valid,
        "exact_l1": terminal.exact_l1,
    }


def _lock_manifest(terminals: Sequence[LockedMatchedTerminal | None]) -> str:
    digest = hashlib.sha256()
    for terminal in terminals:
        if terminal is None:
            digest.update((0).to_bytes(8, "big"))
            continue
        for value in (
            terminal.terminal_id.encode(),
            terminal.terminal_bytes,
            terminal.generation_trace_bytes,
        ):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def _accept_state(
    receipt: GuidanceStateReceipt,
    *,
    expected_transition_calls: int,
    particle_seed_manifest_sha256: str | None = None,
) -> Any:
    if not isinstance(receipt, GuidanceStateReceipt):
        raise UgiHeLaPotencyGuidanceSeamError("lane returned an untyped state receipt")
    if receipt.product_transition_calls != expected_transition_calls:
        raise UgiHeLaPotencyGuidanceSeamError("state transition use differs from schedule")
    if receipt.consumed_particle_seed_manifest_sha256 != particle_seed_manifest_sha256:
        raise UgiHeLaPotencyGuidanceSeamError("lane consumed a different particle seed manifest")
    return receipt.state


@dataclass(frozen=True)
class TerminalAttemptBatch:
    """All scheduled completions for one checkpoint or productive endpoint."""

    checkpoint: int
    terminals: tuple[LockedMatchedTerminal | None, ...]
    error_details: tuple[str | None, ...]
    product_transition_calls: int
    terminal_completions: int
    manifest_sha256: str


def _complete_batch(
    lane: RestartableGuidanceLane,
    state: Any,
    *,
    assignment: FrozenSeedProgramAssignment,
    design: ParticleGroupDesign,
    schedule: GuidanceSchedule,
    checkpoint: int,
    productive: bool,
) -> TerminalAttemptBatch:
    terminals: list[LockedMatchedTerminal | None] = []
    errors: list[str | None] = []
    transitions = 0
    for particle_index in range(design.particle_count):
        if productive:
            seed = assignment.stochastic_particle_seeds[particle_index] + 1
        else:
            program_index = particle_index // design.particles_per_program
            local_index = particle_index % design.particles_per_program
            seed = keyed_random_seed(
                assignment.seed,
                arm="matched_checkpoint_completion",
                program_index=program_index,
                particle_index=local_index,
                checkpoint_index=checkpoint,
                rollout_index=0,
            )
        completion = lane.complete_terminal(
            state,
            particle_index=particle_index,
            seed=seed,
            checkpoint_index=checkpoint,
        )
        if not isinstance(completion, GuidanceTerminalCompletionReceipt):
            raise UgiHeLaPotencyGuidanceSeamError("lane returned an untyped completion receipt")
        expected = 0 if productive else schedule.sample_steps - checkpoint
        if completion.product_transition_calls != expected:
            raise UgiHeLaPotencyGuidanceSeamError(
                "terminal completion transition use differs from schedule"
            )
        transitions += completion.product_transition_calls
        terminals.append(completion.terminal)
        errors.append(completion.error_detail)
    rows = [
        {
            "particle_index": index,
            **_terminal_identity(terminal),
            "error_detail": errors[index],
        }
        for index, terminal in enumerate(terminals)
    ]
    return TerminalAttemptBatch(
        checkpoint=checkpoint,
        terminals=tuple(terminals),
        error_details=tuple(errors),
        product_transition_calls=transitions,
        terminal_completions=len(terminals),
        manifest_sha256=_sha256_payload(rows),
    )


def _evaluation_batch(
    attempts: TerminalAttemptBatch,
    *,
    policy: HeLaPotencyDiagnosticPolicy,
    predictor: HeLaBatchPredictor,
) -> tuple[HeLaPotencyEvaluation | None, ...]:
    present = [terminal for terminal in attempts.terminals if terminal is not None]
    assessed = iter(policy.evaluate_batch(present, predictor)) if present else iter(())
    output: list[HeLaPotencyEvaluation | None] = []
    for terminal in attempts.terminals:
        output.append(None if terminal is None else next(assessed))
    try:
        next(assessed)
    except StopIteration:
        pass
    else:  # pragma: no cover - impossible for a contract-conforming evaluator.
        raise UgiHeLaPotencyGuidanceSeamError("potency evaluations are misaligned")
    return tuple(output)


def _potential(value: HeLaPotencyEvaluation | None) -> float:
    return 0.0 if value is None else value.potential


@dataclass(frozen=True)
class LambdaZeroSeedGate:
    seed: int
    assignment_sha256: str
    checkpoint_manifests: tuple[tuple[int, str], ...]
    productive_manifest_sha256: str
    product_transition_calls_per_arm: int
    terminal_completions_per_arm: int
    bitwise_identity_passed: bool
    historical_hard_reference_passed: bool | None
    receipt_sha256: str


def run_lambda_zero_identity_gate(
    assignment: FrozenSeedProgramAssignment,
    *,
    lane: RestartableGuidanceLane,
    design: ParticleGroupDesign,
    schedule: GuidanceSchedule,
    device: str,
    historical_productive_manifest: str | None = None,
) -> LambdaZeroSeedGate:
    """Require exact two-arm identity under the full zero-potential schedule."""

    expanded = design.expanded_programs(assignment.programs)
    seed_manifest = _sha256_payload(assignment.stochastic_particle_seeds)
    left = _accept_state(
        lane.initialize(
            expanded,
            seed=assignment.seed,
            particle_seeds=assignment.stochastic_particle_seeds,
            device=device,
        ),
        expected_transition_calls=0,
        particle_seed_manifest_sha256=seed_manifest,
    )
    right = _accept_state(
        lane.initialize(
            expanded,
            seed=assignment.seed,
            particle_seeds=assignment.stochastic_particle_seeds,
            device=device,
        ),
        expected_transition_calls=0,
        particle_seed_manifest_sha256=seed_manifest,
    )
    transitions = 0
    completions = 0
    previous = 0
    checkpoint_manifests = []
    for checkpoint in schedule.checkpoints:
        expected = design.particle_count * (checkpoint - previous)
        left = _accept_state(
            lane.advance(left, target_step=checkpoint), expected_transition_calls=expected
        )
        right = _accept_state(
            lane.advance(right, target_step=checkpoint), expected_transition_calls=expected
        )
        transitions += expected
        left_batch = _complete_batch(
            lane,
            left,
            assignment=assignment,
            design=design,
            schedule=schedule,
            checkpoint=checkpoint,
            productive=False,
        )
        right_batch = _complete_batch(
            lane,
            right,
            assignment=assignment,
            design=design,
            schedule=schedule,
            checkpoint=checkpoint,
            productive=False,
        )
        if left_batch != right_batch:
            raise UgiHeLaPotencyGuidanceSeamError(
                f"lambda-zero checkpoint identity failed for seed {assignment.seed}"
            )
        transitions += left_batch.product_transition_calls
        completions += left_batch.terminal_completions
        checkpoint_manifests.append((checkpoint, left_batch.manifest_sha256))
        identity = tuple(range(design.particle_count))
        left = _accept_state(lane.apply_ancestry(left, identity), expected_transition_calls=0)
        previous = checkpoint
    expected = design.particle_count * (schedule.sample_steps - previous)
    left = _accept_state(
        lane.advance(left, target_step=schedule.sample_steps), expected_transition_calls=expected
    )
    right = _accept_state(
        lane.advance(right, target_step=schedule.sample_steps), expected_transition_calls=expected
    )
    transitions += expected
    left_final = _complete_batch(
        lane,
        left,
        assignment=assignment,
        design=design,
        schedule=schedule,
        checkpoint=schedule.sample_steps,
        productive=True,
    )
    right_final = _complete_batch(
        lane,
        right,
        assignment=assignment,
        design=design,
        schedule=schedule,
        checkpoint=schedule.sample_steps,
        productive=True,
    )
    if left_final != right_final:
        raise UgiHeLaPotencyGuidanceSeamError(
            f"lambda-zero productive identity failed for seed {assignment.seed}"
        )
    completions += left_final.terminal_completions
    productive_manifest = _lock_manifest(left_final.terminals)
    historical_passed = None
    if historical_productive_manifest is not None:
        historical_passed = productive_manifest == historical_productive_manifest
        if not historical_passed:
            raise UgiHeLaPotencyGuidanceSeamError(
                "seed 20260821 differs from the historical selected-v3 hard reference"
            )
    expected_transitions = (
        design.particle_count * schedule.sample_steps
        + design.particle_count
        * sum(schedule.sample_steps - checkpoint for checkpoint in schedule.checkpoints)
    )
    expected_completions = design.particle_count * (len(schedule.checkpoints) + 1)
    if transitions != expected_transitions or completions != expected_completions:
        raise UgiHeLaPotencyGuidanceSeamError("lambda-zero realized compute changed")
    content = {
        "seed": assignment.seed,
        "assignment_sha256": assignment.assignment_sha256,
        "checkpoint_manifests": checkpoint_manifests,
        "productive_manifest_sha256": productive_manifest,
        "product_transition_calls_per_arm": transitions,
        "terminal_completions_per_arm": completions,
        "bitwise_identity_passed": True,
        "historical_hard_reference_passed": historical_passed,
    }
    return LambdaZeroSeedGate(**content, receipt_sha256=_sha256_payload(content))


@dataclass(frozen=True)
class PotencyCheckpointRecord:
    checkpoint: int
    beta: float
    attempt_manifest_sha256: str
    potentials: tuple[float, ...]
    active_count: int
    tempered_before: tuple[float, ...]
    tempered_targets: tuple[float, ...]
    incremental_log_weights: tuple[float, ...]
    ancestry_probabilities: tuple[float, ...]
    local_ancestors_by_program: tuple[tuple[int, ...], ...]
    global_ancestors: tuple[int, ...]
    nonuniform_program_count: int
    replay_verified: bool


@dataclass(frozen=True)
class PotencySeedResult:
    seed: int
    assignment_sha256: str
    guided_checkpoint_records: tuple[PotencyCheckpointRecord, ...]
    delayed_posthoc_checkpoint_manifest_sha256s: tuple[tuple[int, str], ...]
    guided_terminal_manifest_sha256: str
    posthoc_terminal_manifest_sha256: str
    guided_terminal_utilities: tuple[float, ...]
    posthoc_terminal_utilities: tuple[float, ...]
    guided_mean_terminal_utility: float
    posthoc_mean_terminal_utility: float
    paired_terminal_utility_delta: float
    guided_eligible_fraction: float
    posthoc_eligible_fraction: float
    product_transition_calls_per_arm: int
    terminal_completions_per_arm: int
    scheduled_potency_attempts_per_arm: int
    synthesis_calls: int
    proposal_calls: int
    receipt_sha256: str


def _potency_checkpoint(
    *,
    evaluations: Sequence[HeLaPotencyEvaluation | None],
    attempts: TerminalAttemptBatch,
    design: ParticleGroupDesign,
    checkpoint: int,
    beta: float,
    guidance_strength: float,
    seed: int,
    tempered_before: np.ndarray,
) -> tuple[PotencyCheckpointRecord, np.ndarray, np.ndarray]:
    potentials = np.asarray([_potential(value) for value in evaluations], dtype=np.float64)
    if potentials.shape != (design.particle_count,):
        raise UgiHeLaPotencyGuidanceSeamError("potency vector differs from particle design")
    targets = tempered_before.copy()
    active = np.asarray(
        [value is not None and value.action == "potency_guidance" for value in evaluations],
        dtype=bool,
    )
    targets[active] = beta * potentials[active]
    increments = targets - tempered_before
    global_ancestors = np.arange(design.particle_count, dtype=np.int64)
    local_rows = []
    probability_rows = []
    nonuniform = 0
    carried = targets.copy()
    for program_index in range(design.morphology_program_count):
        group = design.group_slice(program_index)
        result = select_smc_ancestry(
            np.zeros(design.particles_per_program, dtype=np.float64),
            increments[group],
            np.zeros(design.particles_per_program, dtype=np.float64),
            current_beta=1.0,
            previous_beta=0.0,
            guidance_strength=guidance_strength,
            base_seed=seed,
            arm="guided",
            program_index=program_index,
            checkpoint_index=checkpoint,
            rollout_index=0,
        )
        replay = select_smc_ancestry(
            np.zeros(design.particles_per_program, dtype=np.float64),
            increments[group],
            np.zeros(design.particles_per_program, dtype=np.float64),
            current_beta=1.0,
            previous_beta=0.0,
            guidance_strength=guidance_strength,
            base_seed=seed,
            arm="guided",
            program_index=program_index,
            checkpoint_index=checkpoint,
            rollout_index=0,
        )
        if not (
            np.array_equal(result.ancestors, replay.ancestors)
            and np.array_equal(result.probabilities, replay.probabilities)
        ):
            raise UgiHeLaPotencyGuidanceSeamError("ancestry replay was not deterministic")
        local = result.ancestors.astype(np.int64)
        global_indices = np.arange(group.start, group.stop, dtype=np.int64)
        global_ancestors[group] = global_indices[local]
        carried[group] = targets[group][local]
        local_rows.append(tuple(int(value) for value in local))
        probability_rows.extend(float(value) for value in result.probabilities)
        uniform = np.full(design.particles_per_program, 1.0 / design.particles_per_program)
        nonuniform += int(not np.array_equal(result.probabilities, uniform))
    record = PotencyCheckpointRecord(
        checkpoint=checkpoint,
        beta=beta,
        attempt_manifest_sha256=attempts.manifest_sha256,
        potentials=tuple(float(value) for value in potentials),
        active_count=int(active.sum()),
        tempered_before=tuple(float(value) for value in tempered_before),
        tempered_targets=tuple(float(value) for value in targets),
        incremental_log_weights=tuple(float(value) for value in increments),
        ancestry_probabilities=tuple(probability_rows),
        local_ancestors_by_program=tuple(local_rows),
        global_ancestors=tuple(int(value) for value in global_ancestors),
        nonuniform_program_count=nonuniform,
        replay_verified=True,
    )
    return record, carried, global_ancestors


def run_potency_seed(
    assignment: FrozenSeedProgramAssignment,
    *,
    lane: RestartableGuidanceLane,
    policy: HeLaPotencyDiagnosticPolicy,
    predictor: HeLaBatchPredictor,
    design: ParticleGroupDesign,
    schedule: GuidanceSchedule,
    guidance_strength: float,
    device: str,
) -> PotencySeedResult:
    """Run one prespecified seed after the global lambda-zero gate has passed."""

    if guidance_strength != 0.25:
        raise UgiHeLaPotencyGuidanceSeamError("diagnostic guidance strength must equal 0.25")
    expanded = design.expanded_programs(assignment.programs)
    seed_manifest = _sha256_payload(assignment.stochastic_particle_seeds)
    guided = _accept_state(
        lane.initialize(
            expanded,
            seed=assignment.seed,
            particle_seeds=assignment.stochastic_particle_seeds,
            device=device,
        ),
        expected_transition_calls=0,
        particle_seed_manifest_sha256=seed_manifest,
    )
    posthoc = _accept_state(
        lane.initialize(
            expanded,
            seed=assignment.seed,
            particle_seeds=assignment.stochastic_particle_seeds,
            device=device,
        ),
        expected_transition_calls=0,
        particle_seed_manifest_sha256=seed_manifest,
    )
    posthoc_snapshots: dict[int, Any] = {}
    checkpoint_records = []
    guided_transitions = 0
    posthoc_transitions = 0
    guided_completions = 0
    posthoc_completions = 0
    tempered = np.zeros(design.particle_count, dtype=np.float64)
    previous = 0
    for position, checkpoint in enumerate(schedule.checkpoints):
        expected = design.particle_count * (checkpoint - previous)
        guided = _accept_state(
            lane.advance(guided, target_step=checkpoint), expected_transition_calls=expected
        )
        posthoc = _accept_state(
            lane.advance(posthoc, target_step=checkpoint), expected_transition_calls=expected
        )
        guided_transitions += expected
        posthoc_transitions += expected
        posthoc_snapshots[checkpoint] = _accept_state(
            lane.snapshot(posthoc), expected_transition_calls=0
        )
        attempts = _complete_batch(
            lane,
            guided,
            assignment=assignment,
            design=design,
            schedule=schedule,
            checkpoint=checkpoint,
            productive=False,
        )
        evaluations = _evaluation_batch(attempts, policy=policy, predictor=predictor)
        guided_transitions += attempts.product_transition_calls
        guided_completions += attempts.terminal_completions
        record, tempered, ancestors = _potency_checkpoint(
            evaluations=evaluations,
            attempts=attempts,
            design=design,
            checkpoint=checkpoint,
            beta=schedule.checkpoint_betas[position],
            guidance_strength=guidance_strength,
            seed=assignment.seed,
            tempered_before=tempered,
        )
        checkpoint_records.append(record)
        guided = _accept_state(
            lane.apply_ancestry(guided, tuple(int(value) for value in ancestors)),
            expected_transition_calls=0,
        )
        previous = checkpoint
    expected = design.particle_count * (schedule.sample_steps - previous)
    guided = _accept_state(
        lane.advance(guided, target_step=schedule.sample_steps),
        expected_transition_calls=expected,
    )
    posthoc = _accept_state(
        lane.advance(posthoc, target_step=schedule.sample_steps),
        expected_transition_calls=expected,
    )
    guided_transitions += expected
    posthoc_transitions += expected
    guided_final = _complete_batch(
        lane,
        guided,
        assignment=assignment,
        design=design,
        schedule=schedule,
        checkpoint=schedule.sample_steps,
        productive=True,
    )
    posthoc_final = _complete_batch(
        lane,
        posthoc,
        assignment=assignment,
        design=design,
        schedule=schedule,
        checkpoint=schedule.sample_steps,
        productive=True,
    )
    guided_completions += guided_final.terminal_completions
    posthoc_completions += posthoc_final.terminal_completions
    guided_final_evaluations = _evaluation_batch(guided_final, policy=policy, predictor=predictor)
    posthoc_final_evaluations = _evaluation_batch(posthoc_final, policy=policy, predictor=predictor)
    delayed_manifests = []
    for checkpoint in schedule.checkpoints:
        attempts = _complete_batch(
            lane,
            posthoc_snapshots[checkpoint],
            assignment=assignment,
            design=design,
            schedule=schedule,
            checkpoint=checkpoint,
            productive=False,
        )
        _evaluation_batch(attempts, policy=policy, predictor=predictor)
        posthoc_transitions += attempts.product_transition_calls
        posthoc_completions += attempts.terminal_completions
        delayed_manifests.append((checkpoint, attempts.manifest_sha256))
    guided_utilities = tuple(_potential(value) for value in guided_final_evaluations)
    posthoc_utilities = tuple(_potential(value) for value in posthoc_final_evaluations)
    guided_mean = float(np.mean(guided_utilities))
    posthoc_mean = float(np.mean(posthoc_utilities))
    expected_transitions = (
        design.particle_count * schedule.sample_steps
        + design.particle_count
        * sum(schedule.sample_steps - checkpoint for checkpoint in schedule.checkpoints)
    )
    expected_completions = design.particle_count * (len(schedule.checkpoints) + 1)
    if (
        guided_transitions != expected_transitions
        or posthoc_transitions != expected_transitions
        or guided_completions != expected_completions
        or posthoc_completions != expected_completions
    ):
        raise UgiHeLaPotencyGuidanceSeamError("matched potency compute schedule changed")
    content = {
        "seed": assignment.seed,
        "assignment_sha256": assignment.assignment_sha256,
        "guided_checkpoint_records": tuple(checkpoint_records),
        "delayed_posthoc_checkpoint_manifest_sha256s": tuple(delayed_manifests),
        "guided_terminal_manifest_sha256": _lock_manifest(guided_final.terminals),
        "posthoc_terminal_manifest_sha256": _lock_manifest(posthoc_final.terminals),
        "guided_terminal_utilities": guided_utilities,
        "posthoc_terminal_utilities": posthoc_utilities,
        "guided_mean_terminal_utility": guided_mean,
        "posthoc_mean_terminal_utility": posthoc_mean,
        "paired_terminal_utility_delta": guided_mean - posthoc_mean,
        "guided_eligible_fraction": sum(
            value.action == "potency_guidance"
            for value in guided_final_evaluations
            if value is not None
        )
        / design.particle_count,
        "posthoc_eligible_fraction": sum(
            value.action == "potency_guidance"
            for value in posthoc_final_evaluations
            if value is not None
        )
        / design.particle_count,
        "product_transition_calls_per_arm": expected_transitions,
        "terminal_completions_per_arm": expected_completions,
        "scheduled_potency_attempts_per_arm": expected_completions,
        "synthesis_calls": 0,
        "proposal_calls": 0,
    }
    return PotencySeedResult(**content, receipt_sha256=_sha256_payload(asdict_like(content)))


def asdict_like(value: Any) -> Any:
    """Recursively canonicalize dataclass-bearing content for receipts."""

    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): asdict_like(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [asdict_like(item) for item in value]
    if isinstance(value, list):
        return [asdict_like(item) for item in value]
    return value


@dataclass(frozen=True)
class PotencyDiagnosticExecutionReview:
    """Separate receipt required before any real nonzero execution."""

    receipt_sha256: str
    all_eight_schedules_frozen: bool
    policy_reviewed: bool
    matched_compute_reviewed: bool
    nonzero_diagnostic_execution_authorized: bool
    candidate_selection_authorized: bool = False
    synthesis_guidance_authorized: bool = False

    def require(self) -> None:
        if (
            len(self.receipt_sha256) != 64
            or not self.all_eight_schedules_frozen
            or not self.policy_reviewed
            or not self.matched_compute_reviewed
            or not self.nonzero_diagnostic_execution_authorized
            or self.candidate_selection_authorized
            or self.synthesis_guidance_authorized
        ):
            raise UgiHeLaPotencyGuidanceSeamError(
                "standalone potency execution lacks a valid separate review"
            )


def run_eight_seed_potency_diagnostic(
    qualification: GroupedSMCScheduleQualification,
    *,
    lane: RestartableGuidanceLane,
    policy: HeLaPotencyDiagnosticPolicy,
    predictor: HeLaBatchPredictor,
    design: ParticleGroupDesign,
    schedule: GuidanceSchedule,
    device: str,
    historical_seed_manifest: str,
    execution_review: PotencyDiagnosticExecutionReview,
) -> dict[str, Any]:
    """Run all gates, then all seeds; never select a favorable seed."""

    execution_review.require()
    assignments = qualification.by_seed()
    if tuple(sorted(assignments)) != EXPECTED_SEEDS:
        raise UgiHeLaPotencyGuidanceSeamError("frozen eight-seed schedule changed")
    gates = []
    for seed in EXPECTED_SEEDS:
        gates.append(
            run_lambda_zero_identity_gate(
                assignments[seed],
                lane=lane,
                design=design,
                schedule=schedule,
                device=device,
                historical_productive_manifest=(
                    historical_seed_manifest if seed == EXPECTED_SEEDS[0] else None
                ),
            )
        )
    if not all(gate.bitwise_identity_passed for gate in gates):
        raise UgiHeLaPotencyGuidanceSeamError("global lambda-zero gate did not pass")
    results = [
        run_potency_seed(
            assignments[seed],
            lane=lane,
            policy=policy,
            predictor=predictor,
            design=design,
            schedule=schedule,
            guidance_strength=0.25,
            device=device,
        )
        for seed in EXPECTED_SEEDS
    ]
    by_seed = {value.seed: value for value in results}
    evaluation_deltas = [by_seed[seed].paired_terminal_utility_delta for seed in EVALUATION_SEEDS]
    primary = float(np.mean(evaluation_deltas))
    replay_nonuniform = sum(
        record.nonuniform_program_count
        for result in results
        for record in result.guided_checkpoint_records
        if record.replay_verified
    )
    safeguards = {
        "all_eight_lambda_zero_gates_passed": True,
        "all_eight_seeds_retained": tuple(by_seed) == EXPECTED_SEEDS,
        "matched_compute_counts": all(
            value.product_transition_calls_per_arm == 1280
            and value.terminal_completions_per_arm == 256
            for value in results
        ),
        "zero_synthesis_calls": sum(value.synthesis_calls for value in results) == 0,
        "zero_proposal_calls": sum(value.proposal_calls for value in results) == 0,
    }
    positive = (
        primary > 0.0
        and sum(value >= 0.0 for value in evaluation_deltas) >= 3
        and replay_nonuniform >= 1
        and all(safeguards.values())
    )
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "standalone_hela_potency_diagnostic_complete_not_candidate_selecting",
        "assignment_seeds": list(EXPECTED_SEEDS),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "lambda_zero_gates": [asdict(value) for value in gates],
        "seed_results": [asdict(value) for value in results],
        "primary_endpoint": {
            "evaluation_seed_paired_deltas": evaluation_deltas,
            "mean_paired_delta": primary,
            "nonnegative_evaluation_seeds": sum(value >= 0.0 for value in evaluation_deltas),
            "replay_verified_nonuniform_guidance_events": replay_nonuniform,
            "positive_diagnostic_signal": positive,
        },
        "secondary": {
            "calibration_seed_paired_deltas": [
                by_seed[seed].paired_terminal_utility_delta for seed in CALIBRATION_SEEDS
            ],
            "guided_eligible_fraction_by_seed": {
                str(seed): by_seed[seed].guided_eligible_fraction for seed in EXPECTED_SEEDS
            },
            "posthoc_eligible_fraction_by_seed": {
                str(seed): by_seed[seed].posthoc_eligible_fraction for seed in EXPECTED_SEEDS
            },
        },
        "safeguards": safeguards,
        "scope": {
            "diagnostic_only": True,
            "candidate_selection": False,
            "prospective_candidate_lock": False,
            "synthesis_guidance": False,
            "proposal_guidance": False,
        },
    }
    return {**content, "result_sha256": _sha256_payload(content)}


__all__ = [
    "CALIBRATION_SEEDS",
    "EVALUATION_SEEDS",
    "EXPECTED_SEEDS",
    "LambdaZeroSeedGate",
    "PotencyCheckpointRecord",
    "PotencyDiagnosticExecutionReview",
    "PotencySeedResult",
    "TerminalAttemptBatch",
    "UgiHeLaPotencyGuidanceSeamError",
    "run_eight_seed_potency_diagnostic",
    "run_lambda_zero_identity_gate",
    "run_potency_seed",
]
