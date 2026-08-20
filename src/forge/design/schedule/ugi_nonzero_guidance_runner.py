"""Development runner for matched Ugi synthesis-guidance experiments.

This module wires restartable product trajectories to terminal-only route
rollouts and within-morphology-program SMC ancestry.  It deliberately owns no
chemistry, model, cache, or cloud implementation.  Those dependencies are
injected through :class:`RestartableGuidanceLane`, which keeps the orchestration
device agnostic and suitable for a later local or remote wrapper.

Nonzero production execution is fail-closed behind a separate authenticated
execution-review receipt.  Small fake lanes may exercise the orchestration in
tests without claiming that any production gate has passed.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from forge.core.hashing import sha256_json as _sha256_payload
from forge.core.io import stable_json as _stable_json
from forge.data.r1_prime_audit import sha256_file
from forge.design.ugi_matched_budget_orchestration import (
    LockedMatchedTerminal,
    RouteComputeUsage,
)
from forge.design.ugi_matched_planner_cache_binding import (
    LazyMatchedPlannerCacheBinding,
)
from forge.design.ugi_restartable_terminal_support_adapter import (
    canonical_morphology_program_bytes,
)
from forge.design.ugi_synthesis_guidance import (
    RolloutDisposition,
    keyed_random_seed,
    select_smc_ancestry,
)

RUNNER_CONFIG_SCHEMA_VERSION = "phase1_ugi_nonzero_guidance_runner_config.v1"
EXECUTION_REVIEW_SCHEMA_VERSION = "phase1_ugi_nonzero_guidance_execution_review.v1"
GROUPED_SMC_SCHEDULE_SCHEMA_VERSION = "phase1_ugi_grouped_smc_schedule_qualification.v1"
RUNNER_PLAN_SCHEMA_VERSION = "forge.ugi_nonzero_guidance_runner_plan.v1"
RUN_RESULT_SCHEMA_VERSION = "forge.ugi_nonzero_guidance_development_run.v1"


class UgiNonzeroGuidanceRunnerError(RuntimeError):
    """Raised when guidance orchestration violates the frozen contract."""


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise UgiNonzeroGuidanceRunnerError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_integer(value: Any, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        raise UgiNonzeroGuidanceRunnerError(f"{label} must be a {qualifier} integer")
    return value


def _require_finite_nonnegative(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise UgiNonzeroGuidanceRunnerError(f"{label} must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class ParticleGroupDesign:
    """Frozen morphology-program replication design for within-group ancestry."""

    morphology_program_count: int
    particles_per_program: int

    def __post_init__(self) -> None:
        _require_nonnegative_integer(
            self.morphology_program_count,
            label="morphology_program_count",
            positive=True,
        )
        _require_nonnegative_integer(
            self.particles_per_program,
            label="particles_per_program",
            positive=True,
        )
        if self.particles_per_program < 2:
            raise UgiNonzeroGuidanceRunnerError(
                "each morphology program needs at least two particles for nontrivial ancestry"
            )

    @property
    def particle_count(self) -> int:
        return self.morphology_program_count * self.particles_per_program

    def expanded_programs(self, programs: Sequence[bytes]) -> tuple[bytes, ...]:
        if len(programs) != self.morphology_program_count:
            raise UgiNonzeroGuidanceRunnerError(
                "morphology program count differs from the frozen particle-group design"
            )
        if any(not isinstance(value, bytes) or not value for value in programs):
            raise UgiNonzeroGuidanceRunnerError("morphology programs must be nonempty bytes")
        if len(set(programs)) != len(programs):
            raise UgiNonzeroGuidanceRunnerError(
                "morphology program groups must have distinct canonical byte identities"
            )
        return tuple(program for program in programs for _ in range(self.particles_per_program))

    def group_slice(self, program_index: int) -> slice:
        _require_nonnegative_integer(program_index, label="program_index")
        if program_index >= self.morphology_program_count:
            raise UgiNonzeroGuidanceRunnerError("program_index exceeds the group design")
        start = program_index * self.particles_per_program
        return slice(start, start + self.particles_per_program)


@dataclass(frozen=True)
class GuidanceSchedule:
    """Sampling, checkpoint and final-pool contract for one matched seed."""

    sample_steps: int
    checkpoints: tuple[int, ...]
    checkpoint_betas: tuple[float, ...]
    rollouts_per_particle_checkpoint: int
    final_selection_count: int

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.sample_steps, label="sample_steps", positive=True)
        if self.sample_steps < 2:
            raise UgiNonzeroGuidanceRunnerError("sample_steps must be at least two")
        if (
            not self.checkpoints
            or tuple(sorted(set(self.checkpoints))) != self.checkpoints
            or any(value <= 0 or value >= self.sample_steps for value in self.checkpoints)
        ):
            raise UgiNonzeroGuidanceRunnerError(
                "checkpoints must be unique increasing interior sampling steps"
            )
        if len(self.checkpoint_betas) != len(self.checkpoints):
            raise UgiNonzeroGuidanceRunnerError("each checkpoint requires one annealing beta")
        previous = 0.0
        for value in self.checkpoint_betas:
            value = _require_finite_nonnegative(value, label="checkpoint beta")
            if not previous < value <= 1.0:
                raise UgiNonzeroGuidanceRunnerError(
                    "checkpoint betas must be strictly increasing and at most one"
                )
            previous = value
        _require_nonnegative_integer(
            self.rollouts_per_particle_checkpoint,
            label="rollouts_per_particle_checkpoint",
            positive=True,
        )
        if self.rollouts_per_particle_checkpoint != 1:
            raise UgiNonzeroGuidanceRunnerError(
                "the frozen first experiment permits one rollout per particle/checkpoint"
            )
        _require_nonnegative_integer(
            self.final_selection_count,
            label="final_selection_count",
            positive=True,
        )


@dataclass(frozen=True)
class GuidanceComputeBudget:
    """Exact per-arm ceilings for productive, shadow and route computation."""

    product_transition_calls: int
    terminal_completions: int
    logical_planner_calls: int
    logical_verifier_calls: int
    final_candidates: int

    def __post_init__(self) -> None:
        for name in (
            "product_transition_calls",
            "terminal_completions",
            "logical_planner_calls",
            "logical_verifier_calls",
            "final_candidates",
        ):
            _require_nonnegative_integer(getattr(self, name), label=name)

    def validate_exact_schedule(
        self,
        design: ParticleGroupDesign,
        schedule: GuidanceSchedule,
    ) -> None:
        particles = design.particle_count
        shadow_transitions = particles * sum(
            schedule.sample_steps - checkpoint for checkpoint in schedule.checkpoints
        )
        expected_transitions = particles * schedule.sample_steps + shadow_transitions
        expected_completions = particles * (len(schedule.checkpoints) + 1)
        if self.product_transition_calls != expected_transitions:
            raise UgiNonzeroGuidanceRunnerError(
                "product-transition budget differs from the exact productive-plus-shadow schedule"
            )
        if self.terminal_completions != expected_completions:
            raise UgiNonzeroGuidanceRunnerError(
                "terminal-completion budget differs from checkpoints plus productive finals"
            )
        if self.final_candidates != schedule.final_selection_count:
            raise UgiNonzeroGuidanceRunnerError(
                "final-candidate budget differs from the frozen selection count"
            )
        if self.logical_planner_calls % expected_completions:
            raise UgiNonzeroGuidanceRunnerError(
                "logical planner calls must divide evenly across terminal assessments"
            )
        if self.logical_verifier_calls % expected_completions:
            raise UgiNonzeroGuidanceRunnerError(
                "logical verifier calls must divide evenly across terminal assessments"
            )

    def route_budget_per_terminal(
        self,
        design: ParticleGroupDesign,
        schedule: GuidanceSchedule,
    ) -> tuple[int, int]:
        self.validate_exact_schedule(design, schedule)
        return (
            self.logical_planner_calls // self.terminal_completions,
            self.logical_verifier_calls // self.terminal_completions,
        )


@dataclass(frozen=True)
class NonzeroGuidanceExecutionReview:
    """Separate fail-closed authorization receipt for real nonzero execution."""

    receipt_sha256: str
    route_completion_utility_frozen: bool
    current_l3_snapshot_frozen: bool
    cumulative_source_runtime_qualified: bool
    production_zero_guidance_passed: bool
    particle_group_amendment_frozen: bool
    nonzero_guidance_authorized: bool
    biological_guidance_authorized: bool
    candidate_selection_authorized: bool
    sealed_holdout_access_authorized: bool

    def __post_init__(self) -> None:
        _require_sha256(self.receipt_sha256, label="execution-review receipt")
        values = tuple(
            getattr(self, name)
            for name in (
                "route_completion_utility_frozen",
                "current_l3_snapshot_frozen",
                "cumulative_source_runtime_qualified",
                "production_zero_guidance_passed",
                "particle_group_amendment_frozen",
                "nonzero_guidance_authorized",
                "biological_guidance_authorized",
                "candidate_selection_authorized",
                "sealed_holdout_access_authorized",
            )
        )
        if not all(isinstance(value, bool) for value in values):
            raise UgiNonzeroGuidanceRunnerError("execution-review gates must be boolean")
        if any(
            (
                self.biological_guidance_authorized,
                self.candidate_selection_authorized,
                self.sealed_holdout_access_authorized,
            )
        ):
            raise UgiNonzeroGuidanceRunnerError(
                "this runner never authorizes biology, candidate selection, or holdout access"
            )

    @property
    def prerequisite_gates_passed(self) -> bool:
        return all(
            (
                self.route_completion_utility_frozen,
                self.current_l3_snapshot_frozen,
                self.cumulative_source_runtime_qualified,
                self.production_zero_guidance_passed,
                self.particle_group_amendment_frozen,
            )
        )

    def require_nonzero_execution_authorized(self) -> None:
        if not self.prerequisite_gates_passed or not self.nonzero_guidance_authorized:
            raise UgiNonzeroGuidanceRunnerError(
                "nonzero guidance is blocked until scalar, current L3, cumulative-source, "
                "zero-guidance, group-amendment, and explicit authorization gates all pass"
            )


@dataclass(frozen=True)
class FrozenSeedProgramAssignment:
    """One exact 16-program/64-particle schedule from the qualification receipt."""

    seed: int
    schedule_sha256: str
    programs: tuple[bytes, ...]
    program_sha256s: tuple[str, ...]
    stochastic_particle_seeds: tuple[int, ...]
    qualification_file_sha256: str | None = None
    qualification_result_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.seed, label="assignment seed")
        _require_sha256(self.schedule_sha256, label="assignment schedule_sha256")
        if not self.programs or len(self.programs) != len(self.program_sha256s):
            raise UgiNonzeroGuidanceRunnerError(
                "each seed assignment requires aligned nonempty programs and hashes"
            )
        if len(set(self.programs)) != len(self.programs):
            raise UgiNonzeroGuidanceRunnerError("each seed assignment requires distinct programs")
        for payload, expected_sha256 in zip(
            self.programs,
            self.program_sha256s,
            strict=True,
        ):
            _require_sha256(expected_sha256, label="morphology_program_sha256")
            if hashlib.sha256(payload).hexdigest() != expected_sha256:
                raise UgiNonzeroGuidanceRunnerError(
                    "frozen morphology-program bytes differ from their schedule hash"
                )
        if not self.stochastic_particle_seeds or len(set(self.stochastic_particle_seeds)) != len(
            self.stochastic_particle_seeds
        ):
            raise UgiNonzeroGuidanceRunnerError(
                "stochastic particle seeds must be unique within one seed assignment"
            )
        for value in self.stochastic_particle_seeds:
            _require_nonnegative_integer(value, label="stochastic particle seed")
        qualification_hashes = (
            self.qualification_file_sha256,
            self.qualification_result_sha256,
        )
        if any(value is None for value in qualification_hashes) and any(
            value is not None for value in qualification_hashes
        ):
            raise UgiNonzeroGuidanceRunnerError(
                "assignment qualification file and result hashes must be paired"
            )
        for value in qualification_hashes:
            if value is not None:
                _require_sha256(value, label="assignment qualification hash")

    @property
    def assignment_sha256(self) -> str:
        return _sha256_payload(
            {
                "seed": self.seed,
                "schedule_sha256": self.schedule_sha256,
                "program_sha256s": self.program_sha256s,
                "stochastic_particle_seeds": self.stochastic_particle_seeds,
                "qualification_file_sha256": self.qualification_file_sha256,
                "qualification_result_sha256": self.qualification_result_sha256,
            }
        )


@dataclass(frozen=True)
class GroupedSMCScheduleQualification:
    """Authenticated exact assignments from the nonexecuting grouped-SMC audit."""

    file_sha256: str
    result_sha256: str
    assignments: tuple[FrozenSeedProgramAssignment, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.file_sha256, label="schedule qualification file_sha256")
        _require_sha256(self.result_sha256, label="schedule qualification result_sha256")
        if len(self.assignments) != 8:
            raise UgiNonzeroGuidanceRunnerError(
                "the primary schedule requires three calibration and five evaluation seeds"
            )
        if len({value.seed for value in self.assignments}) != len(self.assignments):
            raise UgiNonzeroGuidanceRunnerError("schedule seeds must be unique")

    def by_seed(self) -> dict[int, FrozenSeedProgramAssignment]:
        return {value.seed: value for value in self.assignments}


class RestartableGuidanceLane(Protocol):
    """Device-agnostic dependency boundary for one restartable generator lane."""

    def initialize(
        self,
        programs: tuple[bytes, ...],
        *,
        seed: int,
        particle_seeds: tuple[int, ...],
        device: str,
    ) -> GuidanceStateReceipt: ...

    def advance(self, state: Any, *, target_step: int) -> GuidanceStateReceipt: ...

    def snapshot(self, state: Any) -> GuidanceStateReceipt: ...

    def complete_terminal(
        self,
        state: Any,
        *,
        particle_index: int,
        seed: int,
        checkpoint_index: int,
    ) -> GuidanceTerminalCompletionReceipt: ...

    def apply_ancestry(
        self,
        state: Any,
        ancestors: Sequence[int],
    ) -> GuidanceStateReceipt: ...


@dataclass(frozen=True)
class GuidanceStateReceipt:
    """One restartable-lane state operation and its realized device compute."""

    state: Any
    product_transition_calls: int
    gpu_device_seconds: float = 0.0
    consumed_particle_seed_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_integer(
            self.product_transition_calls,
            label="state-operation product_transition_calls",
        )
        _require_finite_nonnegative(
            self.gpu_device_seconds,
            label="state-operation gpu_device_seconds",
        )
        if self.consumed_particle_seed_manifest_sha256 is not None:
            _require_sha256(
                self.consumed_particle_seed_manifest_sha256,
                label="consumed_particle_seed_manifest_sha256",
            )


@dataclass(frozen=True)
class GuidanceTerminalCompletionReceipt:
    """One terminal-completion attempt and its realized transition/device use."""

    terminal: LockedMatchedTerminal | None
    product_transition_calls: int
    terminal_completions: int = 1
    gpu_device_seconds: float = 0.0
    error_detail: str | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_integer(
            self.product_transition_calls,
            label="terminal product_transition_calls",
        )
        if self.terminal_completions != 1:
            raise UgiNonzeroGuidanceRunnerError(
                "each terminal receipt must account for exactly one completion attempt"
            )
        _require_finite_nonnegative(
            self.gpu_device_seconds,
            label="terminal gpu_device_seconds",
        )
        if self.terminal is None:
            if not isinstance(self.error_detail, str) or not self.error_detail:
                raise UgiNonzeroGuidanceRunnerError(
                    "failed terminal completion requires a nonempty error detail"
                )
        elif self.error_detail is not None:
            raise UgiNonzeroGuidanceRunnerError(
                "successful terminal completion cannot carry an error detail"
            )


@dataclass(frozen=True)
class GuidanceRouteEvaluation:
    """Binary/neutral route utility plus realized logical and physical compute."""

    route_completion_utility: float | None
    value_policy_id: str
    usage: RouteComputeUsage
    assessment_receipt_sha256: str
    route_dossier_sha256: str | None
    wall_seconds: float = 0.0
    gpu_device_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.route_completion_utility not in {None, 0.0, 1.0}:
            raise UgiNonzeroGuidanceRunnerError(
                "route-completion utility must be binary or explicitly censored"
            )
        if not isinstance(self.value_policy_id, str) or not self.value_policy_id:
            raise UgiNonzeroGuidanceRunnerError("value_policy_id must be nonempty")
        if not isinstance(self.usage, RouteComputeUsage):
            raise UgiNonzeroGuidanceRunnerError("route usage must be typed")
        _require_sha256(
            self.assessment_receipt_sha256,
            label="assessment_receipt_sha256",
        )
        if self.route_dossier_sha256 is not None:
            _require_sha256(self.route_dossier_sha256, label="route_dossier_sha256")
        if self.route_completion_utility == 1.0 and self.route_dossier_sha256 is None:
            raise UgiNonzeroGuidanceRunnerError(
                "a positive exact-dossier utility requires a dossier hash"
            )
        _require_finite_nonnegative(self.wall_seconds, label="wall_seconds")
        _require_finite_nonnegative(self.gpu_device_seconds, label="gpu_device_seconds")

    @property
    def censored(self) -> bool:
        return self.route_completion_utility is None


@dataclass(frozen=True)
class GuidanceCacheIsolationContract:
    """Immutable cache identities supplied to every arm-specific assessment."""

    base_snapshot_sha256: str
    planner_context_sha256: str
    cache_preflight_sha256: str
    guided_clone_id: str
    post_hoc_clone_id: str
    isolated_overlay_roots: bool
    production_adapter_qualified: bool

    def __post_init__(self) -> None:
        for name in (
            "base_snapshot_sha256",
            "planner_context_sha256",
            "cache_preflight_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        for name in ("guided_clone_id", "post_hoc_clone_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise UgiNonzeroGuidanceRunnerError(f"{name} must be nonempty")
        if self.guided_clone_id == self.post_hoc_clone_id:
            raise UgiNonzeroGuidanceRunnerError(
                "guided and post-hoc assessments require distinct cache clones"
            )
        if self.isolated_overlay_roots is not True:
            raise UgiNonzeroGuidanceRunnerError(
                "guided and post-hoc assessments require isolated cache overlays"
            )
        if not isinstance(self.production_adapter_qualified, bool):
            raise UgiNonzeroGuidanceRunnerError("production_adapter_qualified must be boolean")

    def clone_id(self, treatment_arm: str) -> str:
        if treatment_arm == "guided":
            return self.guided_clone_id
        if treatment_arm == "post_hoc":
            return self.post_hoc_clone_id
        raise UgiNonzeroGuidanceRunnerError("unsupported treatment arm")


@dataclass(frozen=True)
class GuidanceCacheFinalization:
    """Before/after proof produced only after both cache overlays are sealed."""

    audit_sha256: str
    base_snapshot_sha256: str
    planner_context_sha256: str
    cache_preflight_sha256: str
    guided_clone_id: str
    post_hoc_clone_id: str
    base_unchanged: bool
    overlay_roots_isolated: bool
    overlay_start_states_identical: bool

    def __post_init__(self) -> None:
        for name in (
            "audit_sha256",
            "base_snapshot_sha256",
            "planner_context_sha256",
            "cache_preflight_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if not self.guided_clone_id or not self.post_hoc_clone_id:
            raise UgiNonzeroGuidanceRunnerError("cache finalization clone IDs must be nonempty")
        if self.guided_clone_id == self.post_hoc_clone_id:
            raise UgiNonzeroGuidanceRunnerError(
                "cache finalization reused one clone across matched arms"
            )
        if not all(
            (
                self.base_unchanged,
                self.overlay_roots_isolated,
                self.overlay_start_states_identical,
            )
        ):
            raise UgiNonzeroGuidanceRunnerError(
                "cache finalization does not prove immutable isolated overlays"
            )

    def validate_against(self, contract: GuidanceCacheIsolationContract) -> None:
        if (
            self.base_snapshot_sha256 != contract.base_snapshot_sha256
            or self.planner_context_sha256 != contract.planner_context_sha256
            or self.cache_preflight_sha256 != contract.cache_preflight_sha256
            or self.guided_clone_id != contract.guided_clone_id
            or self.post_hoc_clone_id != contract.post_hoc_clone_id
        ):
            raise UgiNonzeroGuidanceRunnerError(
                "final cache audit differs from the frozen cache contract"
            )


def finalize_lazy_matched_cache_binding(
    binding: LazyMatchedPlannerCacheBinding,
    contract: GuidanceCacheIsolationContract,
) -> GuidanceCacheFinalization:
    """Adapt the qualified file-cache binding's final audit to this runner."""

    if not isinstance(binding, LazyMatchedPlannerCacheBinding):
        raise UgiNonzeroGuidanceRunnerError(
            "production cache finalization requires LazyMatchedPlannerCacheBinding"
        )
    if binding.preflight.preflight_sha256 != contract.cache_preflight_sha256:
        raise UgiNonzeroGuidanceRunnerError("cache preflight changed before execution")
    audit = binding.finalize()
    result = GuidanceCacheFinalization(
        audit_sha256=audit.audit_sha256,
        base_snapshot_sha256=audit.base_before.snapshot_sha256,
        planner_context_sha256=audit.context_sha256,
        cache_preflight_sha256=binding.preflight.preflight_sha256,
        guided_clone_id=audit.guided_clone_id,
        post_hoc_clone_id=audit.post_hoc_clone_id,
        base_unchanged=audit.base_before == audit.base_after,
        overlay_roots_isolated=True,
        overlay_start_states_identical=audit.guided_before == audit.post_hoc_before,
    )
    result.validate_against(contract)
    return result


@dataclass(frozen=True)
class GuidanceAssessmentContext:
    """One route call's matched budget, RNG, cache and lock provenance."""

    treatment_arm: str
    assessment_phase: str
    checkpoint: int
    route_seed: int
    reservation: RouteComputeUsage
    base_snapshot_sha256: str
    planner_context_sha256: str
    cache_preflight_sha256: str
    cache_clone_id: str
    post_hoc_productive_lock_sha256: str | None

    def __post_init__(self) -> None:
        if self.treatment_arm not in {"guided", "post_hoc"}:
            raise UgiNonzeroGuidanceRunnerError("unsupported route-assessment arm")
        if self.assessment_phase not in {"checkpoint_shadow", "productive_final"}:
            raise UgiNonzeroGuidanceRunnerError("unsupported route-assessment phase")
        _require_nonnegative_integer(self.checkpoint, label="checkpoint")
        _require_nonnegative_integer(self.route_seed, label="route_seed")
        if not isinstance(self.reservation, RouteComputeUsage):
            raise UgiNonzeroGuidanceRunnerError("route reservation must be typed")
        for name in (
            "base_snapshot_sha256",
            "planner_context_sha256",
            "cache_preflight_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if not isinstance(self.cache_clone_id, str) or not self.cache_clone_id:
            raise UgiNonzeroGuidanceRunnerError("cache_clone_id must be nonempty")
        if self.treatment_arm == "post_hoc":
            _require_sha256(
                self.post_hoc_productive_lock_sha256,
                label="post_hoc_productive_lock_sha256",
            )
        elif self.post_hoc_productive_lock_sha256 is not None:
            raise UgiNonzeroGuidanceRunnerError(
                "guided assessments cannot carry the post-hoc productive lock"
            )


RouteEvaluator = Callable[
    [LockedMatchedTerminal, GuidanceAssessmentContext],
    GuidanceRouteEvaluation,
]
CanonicalIdentity = Callable[[bytes], str]


@dataclass(frozen=True)
class ProductiveAdmissionRecord:
    """Route-blind validity, identity and dedup decision sealed before assessment."""

    terminal_id: str
    terminal_sha256: str
    generation_trace_sha256: str
    canonical_identity: str | None
    disposition: str
    representative_terminal_id: str | None
    representative_index: int | None


@dataclass(frozen=True)
class CheckpointGroupRecord:
    """One program-local rollout, incremental potential and ancestry decision."""

    checkpoint: int
    program_index: int
    global_particle_indices: tuple[int, ...]
    dispositions: tuple[str, ...]
    route_completion_utilities: tuple[float | None, ...]
    shadow_terminal_ids: tuple[str | None, ...]
    shadow_terminal_sha256s: tuple[str | None, ...]
    shadow_trace_sha256s: tuple[str | None, ...]
    value_policy_ids: tuple[str | None, ...]
    evaluation_receipt_sha256s: tuple[str | None, ...]
    source_assessment_receipt_sha256s: tuple[str | None, ...]
    route_dossier_sha256s: tuple[str | None, ...]
    shadow_batch_manifest_sha256: str
    base_cache_snapshot_sha256: str
    planner_context_sha256: str
    cache_preflight_sha256: str
    cache_clone_id: str
    tempered_before: tuple[float, ...]
    tempered_targets: tuple[float, ...]
    incremental_log_weights: tuple[float, ...]
    ancestry_probabilities: tuple[float, ...]
    local_ancestors: tuple[int, ...]
    global_ancestors: tuple[int, ...]
    carried_tempered_after: tuple[float, ...]
    resampled: bool
    keyed_seed: int | None
    terminal_completions: int
    product_transition_calls: int
    logical_planner_calls: int
    physical_planner_calls: int
    verifier_calls: int
    physical_verifier_calls: int
    planner_cache_hits: int
    verifier_cache_hits: int
    invalid_terminal_count: int
    nonexact_l1_count: int
    censored_assessment_count: int
    wall_seconds: float
    gpu_device_seconds: float


@dataclass(frozen=True)
class SelectedCandidate:
    """One constitutionally deduplicated productive final retained for selection."""

    canonical_identity: str
    terminal_id: str
    route_completion_utility: float | None
    support_bonus: bool
    censored: bool
    tie_priority: int


@dataclass(frozen=True)
class ProductiveAssessmentRecord:
    """Row-level endpoint receipt retained before deduplication or selection."""

    terminal_id: str
    terminal_sha256: str
    generation_trace_sha256: str
    canonical_identity: str | None
    representative_terminal_id: str | None
    disposition: str
    route_completion_utility: float | None
    value_policy_id: str | None
    source_assessment_receipt_sha256: str | None
    route_dossier_sha256: str | None
    evaluation_receipt_sha256: str
    route_seed: int | None
    usage: RouteComputeUsage
    censored: bool


@dataclass(frozen=True)
class ArmRunResult:
    """One arm's productive pool, shadow records and deterministic final selection."""

    arm: str
    productive_lock_manifest_sha256: str
    productive_terminal_ids: tuple[str, ...]
    productive_canonical_identities: tuple[str | None, ...]
    productive_route_completion_utilities: tuple[float | None, ...]
    productive_admission_manifest_sha256: str
    productive_admissions: tuple[ProductiveAdmissionRecord, ...]
    productive_assessments: tuple[ProductiveAssessmentRecord, ...]
    productive_pool_size: int
    unique_productive_pool_size: int
    duplicate_productive_count: int
    assessed_productive_count: int
    censored_productive_count: int
    selected: tuple[SelectedCandidate, ...]
    checkpoint_groups: tuple[CheckpointGroupRecord, ...]
    shadow_assessment_started_after_productive_lock: bool
    shadow_assessments_selecting: bool
    base_cache_snapshot_sha256: str
    planner_context_sha256: str
    cache_preflight_sha256: str
    cache_clone_id: str
    isolated_cache_overlay: bool
    realized_terminal_completions: int
    realized_logical_planner_calls: int
    realized_physical_planner_calls: int
    realized_verifier_calls: int
    realized_physical_verifier_calls: int
    planner_cache_hits: int
    verifier_cache_hits: int
    invalid_terminal_count: int
    nonexact_l1_count: int
    censored_assessment_count: int
    realized_product_transition_calls: int
    wall_seconds: float
    gpu_device_seconds: float


@dataclass(frozen=True)
class DevelopmentGuidanceRun:
    """Deterministic matched-arm development result; never a production authorization."""

    base_seed: int
    program_assignment_sha256: str
    qualified_schedule_sha256: str | None
    particle_seed_manifest_sha256: str
    guidance_strength: float
    device: str
    particle_group_design: ParticleGroupDesign
    schedule: GuidanceSchedule
    budget: GuidanceComputeBudget
    cache_isolation: GuidanceCacheIsolationContract
    cache_finalization: GuidanceCacheFinalization
    guided: ArmRunResult
    post_hoc: ArmRunResult
    production_execution: bool
    biological_guidance: bool
    private_holdout_accessed: bool


def development_guidance_run_to_dict(value: DevelopmentGuidanceRun) -> dict[str, Any]:
    """Return the stable local/remote output contract for one development run."""

    if not isinstance(value, DevelopmentGuidanceRun):
        raise UgiNonzeroGuidanceRunnerError("guidance result must be typed")
    content = {
        "schema_version": RUN_RESULT_SCHEMA_VERSION,
        "status": "development_matched_guidance_complete_not_production_authorized",
        "run": asdict(value),
        "scope": {
            "production_execution": value.production_execution,
            "biological_guidance": value.biological_guidance,
            "private_holdout_accessed": value.private_holdout_accessed,
            "prospective_candidate_lock": False,
            "development_endpoint_subset_selected": True,
            "success_probability": None,
        },
    }
    return {**content, "result_sha256": _sha256_payload(content)}


def _validate_terminal(terminal: LockedMatchedTerminal) -> None:
    if not isinstance(terminal, LockedMatchedTerminal) or not terminal.terminal_locked:
        raise UgiNonzeroGuidanceRunnerError("the lane must return a sealed matched terminal")


def _lock_manifest(terminals: Sequence[LockedMatchedTerminal]) -> str:
    digest = hashlib.sha256()
    for terminal in terminals:
        _validate_terminal(terminal)
        for value in (
            terminal.terminal_id.encode(),
            terminal.terminal_bytes,
            terminal.generation_trace_bytes,
        ):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class _GroupRolloutBatch:
    dispositions: tuple[RolloutDisposition, ...]
    utilities: tuple[float | None, ...]
    terminal_ids: tuple[str | None, ...]
    terminal_sha256s: tuple[str | None, ...]
    trace_sha256s: tuple[str | None, ...]
    value_policy_ids: tuple[str | None, ...]
    evaluation_receipt_sha256s: tuple[str | None, ...]
    source_assessment_receipt_sha256s: tuple[str | None, ...]
    route_dossier_sha256s: tuple[str | None, ...]
    manifest_sha256: str
    terminal_completions: int
    product_transition_calls: int
    usage: RouteComputeUsage
    censored_assessment_count: int
    wall_seconds: float
    gpu_device_seconds: float


def _run_group_rollouts(
    *,
    lane: RestartableGuidanceLane,
    state: Any,
    particle_offset: int,
    particle_count: int,
    program_index: int,
    checkpoint: int,
    base_seed: int,
    evaluator: RouteEvaluator,
    expected_value_policy_id: str,
    treatment_arm: str,
    cache_contract: GuidanceCacheIsolationContract,
    post_hoc_productive_lock_sha256: str | None,
    planner_calls_per_terminal: int,
    verifier_calls_per_terminal: int,
    expected_product_transition_calls_per_completion: int,
) -> _GroupRolloutBatch:
    dispositions: list[RolloutDisposition] = []
    utilities: list[float | None] = []
    terminal_ids: list[str | None] = []
    terminal_sha256s: list[str | None] = []
    trace_sha256s: list[str | None] = []
    value_policy_ids: list[str | None] = []
    evaluation_receipt_sha256s: list[str | None] = []
    source_assessment_receipt_sha256s: list[str | None] = []
    route_dossier_sha256s: list[str | None] = []
    usage = RouteComputeUsage()
    completions = 0
    product_transition_calls = 0
    censored_assessments = 0
    wall_seconds = 0.0
    gpu_device_seconds = 0.0
    ceiling = RouteComputeUsage(
        logical_planner_calls=particle_count * planner_calls_per_terminal,
        physical_planner_calls=particle_count * planner_calls_per_terminal,
        logical_verifier_calls=particle_count * verifier_calls_per_terminal,
        physical_verifier_calls=particle_count * verifier_calls_per_terminal,
    )
    for local_index in range(particle_count):
        seed = keyed_random_seed(
            base_seed,
            arm="matched_checkpoint_completion",
            program_index=program_index,
            particle_index=local_index,
            checkpoint_index=checkpoint,
            rollout_index=0,
        )
        started = time.perf_counter()
        try:
            completion = lane.complete_terminal(
                state,
                particle_index=particle_offset + local_index,
                seed=seed,
                checkpoint_index=checkpoint,
            )
        except Exception as error:
            raise UgiNonzeroGuidanceRunnerError(
                "terminal lane raised without a compute receipt"
            ) from error
        wall_seconds += time.perf_counter() - started
        if not isinstance(completion, GuidanceTerminalCompletionReceipt):
            raise UgiNonzeroGuidanceRunnerError(
                "terminal lane must return a typed completion receipt"
            )
        if completion.product_transition_calls != expected_product_transition_calls_per_completion:
            raise UgiNonzeroGuidanceRunnerError(
                "terminal completion transition use differs from the exact schedule"
            )
        completions += completion.terminal_completions
        product_transition_calls += completion.product_transition_calls
        gpu_device_seconds += completion.gpu_device_seconds
        terminal = completion.terminal
        if terminal is None:
            dispositions.append(RolloutDisposition.COMPLETION_ERROR)
            utilities.append(None)
            terminal_ids.append(None)
            terminal_sha256s.append(None)
            trace_sha256s.append(None)
            value_policy_ids.append(None)
            evaluation_receipt_sha256s.append(None)
            source_assessment_receipt_sha256s.append(None)
            route_dossier_sha256s.append(None)
            continue
        _validate_terminal(terminal)
        terminal_ids.append(terminal.terminal_id)
        terminal_sha256s.append(terminal.terminal_sha256)
        trace_sha256s.append(terminal.generation_trace_sha256)
        if not terminal.terminal_valid:
            dispositions.append(RolloutDisposition.INVALID_TERMINAL)
            utilities.append(None)
            value_policy_ids.append(None)
            evaluation_receipt_sha256s.append(None)
            source_assessment_receipt_sha256s.append(None)
            route_dossier_sha256s.append(None)
            continue
        if not terminal.exact_l1:
            dispositions.append(RolloutDisposition.NONEXACT_L1)
            utilities.append(None)
            value_policy_ids.append(None)
            evaluation_receipt_sha256s.append(None)
            source_assessment_receipt_sha256s.append(None)
            route_dossier_sha256s.append(None)
            continue
        remaining = ceiling.remaining_after(usage)
        if remaining.logical_planner_calls < planner_calls_per_terminal:
            dispositions.append(RolloutDisposition.BUDGET_EXHAUSTED)
            utilities.append(None)
            value_policy_ids.append(None)
            evaluation_receipt_sha256s.append(None)
            source_assessment_receipt_sha256s.append(None)
            route_dossier_sha256s.append(None)
            continue
        reservation = RouteComputeUsage(
            logical_planner_calls=planner_calls_per_terminal,
            physical_planner_calls=planner_calls_per_terminal,
            logical_verifier_calls=verifier_calls_per_terminal,
            physical_verifier_calls=verifier_calls_per_terminal,
        )
        context = GuidanceAssessmentContext(
            treatment_arm=treatment_arm,
            assessment_phase="checkpoint_shadow",
            checkpoint=checkpoint,
            route_seed=keyed_random_seed(
                base_seed,
                namespace="matched_route_assessment",
                program_index=program_index,
                particle_index=local_index,
                checkpoint=checkpoint,
            ),
            reservation=reservation,
            base_snapshot_sha256=cache_contract.base_snapshot_sha256,
            planner_context_sha256=cache_contract.planner_context_sha256,
            cache_preflight_sha256=cache_contract.cache_preflight_sha256,
            cache_clone_id=cache_contract.clone_id(treatment_arm),
            post_hoc_productive_lock_sha256=post_hoc_productive_lock_sha256,
        )
        started = time.perf_counter()
        evaluation = evaluator(terminal, context)
        observed_wall_seconds = time.perf_counter() - started
        if evaluation.value_policy_id != expected_value_policy_id:
            raise UgiNonzeroGuidanceRunnerError("route evaluator changed the frozen value policy")
        if not evaluation.usage.fits_within(remaining) or not evaluation.usage.fits_within(
            RouteComputeUsage(
                logical_planner_calls=planner_calls_per_terminal,
                physical_planner_calls=planner_calls_per_terminal,
                logical_verifier_calls=verifier_calls_per_terminal,
                physical_verifier_calls=verifier_calls_per_terminal,
            )
        ):
            raise UgiNonzeroGuidanceRunnerError(
                "checkpoint route evaluator exceeded its per-terminal or group budget"
            )
        usage = usage.plus(evaluation.usage)
        wall_seconds += max(evaluation.wall_seconds, observed_wall_seconds)
        gpu_device_seconds += evaluation.gpu_device_seconds
        dispositions.append(RolloutDisposition.ASSESSED)
        utilities.append(evaluation.route_completion_utility)
        value_policy_ids.append(evaluation.value_policy_id)
        source_assessment_receipt_sha256s.append(evaluation.assessment_receipt_sha256)
        route_dossier_sha256s.append(evaluation.route_dossier_sha256)
        evaluation_receipt_sha256s.append(
            _sha256_payload(
                {
                    "terminal_sha256": terminal.terminal_sha256,
                    "generation_trace_sha256": terminal.generation_trace_sha256,
                    "value_policy_id": evaluation.value_policy_id,
                    "source_assessment_receipt_sha256": (evaluation.assessment_receipt_sha256),
                    "route_dossier_sha256": evaluation.route_dossier_sha256,
                    "route_completion_utility": evaluation.route_completion_utility,
                    "usage": evaluation.usage.__dict__,
                    "wall_seconds": evaluation.wall_seconds,
                    "gpu_device_seconds": evaluation.gpu_device_seconds,
                    "assessment_arm": treatment_arm,
                    "assessment_phase": "checkpoint_shadow",
                    "checkpoint": checkpoint,
                    "route_seed": context.route_seed,
                    "base_snapshot_sha256": context.base_snapshot_sha256,
                    "planner_context_sha256": context.planner_context_sha256,
                    "cache_preflight_sha256": context.cache_preflight_sha256,
                    "cache_clone_id": context.cache_clone_id,
                    "post_hoc_productive_lock_sha256": (context.post_hoc_productive_lock_sha256),
                }
            )
        )
        censored_assessments += int(evaluation.censored)
    manifest_rows = [
        {
            "particle_index": particle_offset + index,
            "disposition": disposition.value,
            "terminal_id": terminal_ids[index],
            "terminal_sha256": terminal_sha256s[index],
            "generation_trace_sha256": trace_sha256s[index],
            "value_policy_id": value_policy_ids[index],
            "evaluation_receipt_sha256": evaluation_receipt_sha256s[index],
            "source_assessment_receipt_sha256": (source_assessment_receipt_sha256s[index]),
            "route_dossier_sha256": route_dossier_sha256s[index],
            "route_completion_utility": utilities[index],
        }
        for index, disposition in enumerate(dispositions)
    ]
    return _GroupRolloutBatch(
        dispositions=tuple(dispositions),
        utilities=tuple(utilities),
        terminal_ids=tuple(terminal_ids),
        terminal_sha256s=tuple(terminal_sha256s),
        trace_sha256s=tuple(trace_sha256s),
        value_policy_ids=tuple(value_policy_ids),
        evaluation_receipt_sha256s=tuple(evaluation_receipt_sha256s),
        source_assessment_receipt_sha256s=tuple(source_assessment_receipt_sha256s),
        route_dossier_sha256s=tuple(route_dossier_sha256s),
        manifest_sha256=_sha256_payload(manifest_rows),
        terminal_completions=completions,
        product_transition_calls=product_transition_calls,
        usage=usage,
        censored_assessment_count=censored_assessments,
        wall_seconds=wall_seconds,
        gpu_device_seconds=gpu_device_seconds,
    )


def _group_checkpoint(
    *,
    lane: RestartableGuidanceLane,
    state: Any,
    design: ParticleGroupDesign,
    schedule: GuidanceSchedule,
    base_seed: int,
    guidance_strength: float,
    checkpoint_position: int,
    program_index: int,
    tempered_before: np.ndarray,
    planner_calls_per_terminal: int,
    verifier_calls_per_terminal: int,
    evaluator: RouteEvaluator,
    expected_value_policy_id: str,
    treatment_arm: str,
    cache_contract: GuidanceCacheIsolationContract,
    post_hoc_productive_lock_sha256: str | None,
) -> tuple[CheckpointGroupRecord, np.ndarray]:
    checkpoint = schedule.checkpoints[checkpoint_position]
    beta = schedule.checkpoint_betas[checkpoint_position]
    group = design.group_slice(program_index)
    count = design.particles_per_program
    batch = _run_group_rollouts(
        lane=lane,
        state=state,
        particle_offset=group.start,
        particle_count=count,
        base_seed=base_seed,
        program_index=program_index,
        checkpoint=checkpoint,
        evaluator=evaluator,
        expected_value_policy_id=expected_value_policy_id,
        treatment_arm=treatment_arm,
        cache_contract=cache_contract,
        post_hoc_productive_lock_sha256=post_hoc_productive_lock_sha256,
        planner_calls_per_terminal=planner_calls_per_terminal,
        verifier_calls_per_terminal=verifier_calls_per_terminal,
        expected_product_transition_calls_per_completion=(schedule.sample_steps - checkpoint),
    )
    if batch.terminal_completions != count:
        raise UgiNonzeroGuidanceRunnerError("checkpoint shadow completion budget was not filled")
    local_before = tempered_before[group].copy()
    utilities = np.asarray(
        [np.nan if value is None else value for value in batch.utilities],
        dtype=np.float64,
    )
    censored = ~np.isfinite(utilities)
    targets = local_before.copy()
    targets[~censored] = beta * utilities[~censored]
    increments = targets - local_before
    increments[censored] = 0.0
    ancestry = select_smc_ancestry(
        np.zeros(count, dtype=np.float64),
        increments,
        np.zeros(count, dtype=np.float64),
        current_beta=1.0,
        previous_beta=0.0,
        guidance_strength=guidance_strength,
        base_seed=base_seed,
        arm="guided",
        program_index=program_index,
        checkpoint_index=checkpoint,
        rollout_index=0,
    )
    local_ancestors = ancestry.ancestors.astype(np.int64)
    global_indices = np.arange(group.start, group.stop, dtype=np.int64)
    global_ancestors = global_indices[local_ancestors]
    carried = targets[local_ancestors]
    record = CheckpointGroupRecord(
        checkpoint=checkpoint,
        program_index=program_index,
        global_particle_indices=tuple(int(value) for value in global_indices),
        dispositions=tuple(value.value for value in batch.dispositions),
        route_completion_utilities=tuple(
            None if not np.isfinite(value) else float(value) for value in utilities
        ),
        shadow_terminal_ids=batch.terminal_ids,
        shadow_terminal_sha256s=batch.terminal_sha256s,
        shadow_trace_sha256s=batch.trace_sha256s,
        value_policy_ids=batch.value_policy_ids,
        evaluation_receipt_sha256s=batch.evaluation_receipt_sha256s,
        source_assessment_receipt_sha256s=(batch.source_assessment_receipt_sha256s),
        route_dossier_sha256s=batch.route_dossier_sha256s,
        shadow_batch_manifest_sha256=batch.manifest_sha256,
        base_cache_snapshot_sha256=cache_contract.base_snapshot_sha256,
        planner_context_sha256=cache_contract.planner_context_sha256,
        cache_preflight_sha256=cache_contract.cache_preflight_sha256,
        cache_clone_id=cache_contract.clone_id(treatment_arm),
        tempered_before=tuple(float(value) for value in local_before),
        tempered_targets=tuple(float(value) for value in targets),
        incremental_log_weights=tuple(float(value) for value in increments),
        ancestry_probabilities=tuple(float(value) for value in ancestry.probabilities),
        local_ancestors=tuple(int(value) for value in local_ancestors),
        global_ancestors=tuple(int(value) for value in global_ancestors),
        carried_tempered_after=tuple(float(value) for value in carried),
        resampled=ancestry.resampled,
        keyed_seed=ancestry.keyed_seed,
        terminal_completions=batch.terminal_completions,
        product_transition_calls=batch.product_transition_calls,
        logical_planner_calls=batch.usage.logical_planner_calls,
        physical_planner_calls=batch.usage.physical_planner_calls,
        verifier_calls=batch.usage.logical_verifier_calls,
        physical_verifier_calls=batch.usage.physical_verifier_calls,
        planner_cache_hits=batch.usage.planner_cache_hits,
        verifier_cache_hits=batch.usage.verifier_cache_hits,
        invalid_terminal_count=sum(
            value is RolloutDisposition.INVALID_TERMINAL for value in batch.dispositions
        ),
        nonexact_l1_count=sum(
            value is RolloutDisposition.NONEXACT_L1 for value in batch.dispositions
        ),
        censored_assessment_count=batch.censored_assessment_count,
        wall_seconds=batch.wall_seconds,
        gpu_device_seconds=batch.gpu_device_seconds,
    )
    return record, carried


@dataclass(frozen=True)
class _ProductivePoolCompletion:
    terminals: tuple[LockedMatchedTerminal, ...]
    terminal_completions: int
    product_transition_calls: int
    wall_seconds: float
    gpu_device_seconds: float


def _complete_productive_pool(
    *,
    lane: RestartableGuidanceLane,
    state: Any,
    particle_count: int,
    sample_steps: int,
    particle_seeds: tuple[int, ...],
) -> _ProductivePoolCompletion:
    terminals = []
    terminal_completions = 0
    product_transition_calls = 0
    wall_seconds = 0.0
    gpu_device_seconds = 0.0
    if len(particle_seeds) != particle_count:
        raise UgiNonzeroGuidanceRunnerError(
            "productive completion requires one frozen seed per particle"
        )
    for particle_index in range(particle_count):
        seed = particle_seeds[particle_index] + 1
        started = time.perf_counter()
        completion = lane.complete_terminal(
            state,
            particle_index=particle_index,
            seed=seed,
            checkpoint_index=sample_steps,
        )
        wall_seconds += time.perf_counter() - started
        if not isinstance(completion, GuidanceTerminalCompletionReceipt):
            raise UgiNonzeroGuidanceRunnerError(
                "productive lane must return a typed completion receipt"
            )
        if completion.terminal is None:
            raise UgiNonzeroGuidanceRunnerError(
                "productive terminal completion failed; the seed cannot be selected"
            )
        if completion.product_transition_calls != 0:
            raise UgiNonzeroGuidanceRunnerError(
                "productive finals must be completed from the frozen terminal step"
            )
        terminal = completion.terminal
        _validate_terminal(terminal)
        terminals.append(terminal)
        terminal_completions += completion.terminal_completions
        product_transition_calls += completion.product_transition_calls
        gpu_device_seconds += completion.gpu_device_seconds
    return _ProductivePoolCompletion(
        terminals=tuple(terminals),
        terminal_completions=terminal_completions,
        product_transition_calls=product_transition_calls,
        wall_seconds=wall_seconds,
        gpu_device_seconds=gpu_device_seconds,
    )


def _build_productive_admission(
    terminals: Sequence[LockedMatchedTerminal],
    *,
    canonical_identity: CanonicalIdentity,
    final_count: int,
    base_seed: int,
) -> tuple[tuple[ProductiveAdmissionRecord, ...], str]:
    """Seal route-blind invalid/L1/identity/dedup decisions before assessment."""

    provisional: list[tuple[str | None, str]] = []
    members_by_identity: dict[str, list[int]] = {}
    for terminal_index, terminal in enumerate(terminals):
        if not terminal.terminal_valid:
            provisional.append((None, "invalid_terminal"))
            continue
        if not terminal.exact_l1:
            provisional.append((None, "nonexact_l1"))
            continue
        identity = canonical_identity(bytes(terminal.terminal_bytes))
        if not isinstance(identity, str) or not identity:
            raise UgiNonzeroGuidanceRunnerError(
                "productive admission requires a route-blind canonical identity"
            )
        provisional.append((identity, "valid_exact_l1"))
        members_by_identity.setdefault(identity, []).append(terminal_index)
    representatives: dict[str, int] = {}
    for identity, indices in members_by_identity.items():
        representatives[identity] = min(
            indices,
            key=lambda index: (
                keyed_random_seed(
                    base_seed,
                    namespace="route_blind_productive_representative",
                    canonical_constitutional_identity=identity,
                    terminal_id=terminals[index].terminal_id,
                ),
                terminals[index].terminal_id,
            ),
        )
    if len(representatives) < final_count:
        raise UgiNonzeroGuidanceRunnerError(
            "route-blind constitutional deduplication leaves fewer candidates "
            "than the frozen final count"
        )
    records: list[ProductiveAdmissionRecord] = []
    for terminal_index, (terminal, (identity, disposition)) in enumerate(
        zip(terminals, provisional, strict=True)
    ):
        representative_index = None if identity is None else representatives[identity]
        representative_terminal_id = (
            None if representative_index is None else terminals[representative_index].terminal_id
        )
        if identity is not None:
            disposition = (
                "canonical_representative"
                if representative_index == terminal_index
                else "duplicate_not_assessed"
            )
        records.append(
            ProductiveAdmissionRecord(
                terminal_id=terminal.terminal_id,
                terminal_sha256=terminal.terminal_sha256,
                generation_trace_sha256=terminal.generation_trace_sha256,
                canonical_identity=identity,
                disposition=disposition,
                representative_terminal_id=representative_terminal_id,
                representative_index=representative_index,
            )
        )
    manifest_rows = [asdict(record) for record in records]
    return tuple(records), _sha256_payload(manifest_rows)


def _assess_terminal_pool(
    terminals: Sequence[LockedMatchedTerminal],
    admissions: Sequence[ProductiveAdmissionRecord],
    *,
    arm: str,
    checkpoint: int,
    planner_calls_per_terminal: int,
    verifier_calls_per_terminal: int,
    evaluator: RouteEvaluator,
    expected_value_policy_id: str,
    base_seed: int,
    cache_contract: GuidanceCacheIsolationContract,
    productive_lock_sha256: str,
) -> tuple[
    tuple[float | None, ...],
    tuple[ProductiveAssessmentRecord, ...],
    RouteComputeUsage,
    int,
    int,
    int,
    float,
    float,
]:
    if len(terminals) != len(admissions):
        raise UgiNonzeroGuidanceRunnerError(
            "productive terminals and route-blind admission rows are misaligned"
        )
    usage = RouteComputeUsage()
    invalid_count = sum(value.disposition == "invalid_terminal" for value in admissions)
    nonexact_count = sum(value.disposition == "nonexact_l1" for value in admissions)
    censored_count = 0
    wall_seconds = 0.0
    gpu_device_seconds = 0.0
    treatment_arm = "guided" if arm == "guided_final" else "post_hoc"
    evaluated: dict[
        int,
        tuple[
            GuidanceRouteEvaluation,
            GuidanceAssessmentContext,
            str,
        ],
    ] = {}
    for terminal_index, (terminal, admission) in enumerate(zip(terminals, admissions, strict=True)):
        if admission.disposition != "canonical_representative":
            continue
        reservation = RouteComputeUsage(
            logical_planner_calls=planner_calls_per_terminal,
            physical_planner_calls=planner_calls_per_terminal,
            logical_verifier_calls=verifier_calls_per_terminal,
            physical_verifier_calls=verifier_calls_per_terminal,
        )
        context = GuidanceAssessmentContext(
            treatment_arm=treatment_arm,
            assessment_phase="productive_final",
            checkpoint=checkpoint,
            route_seed=keyed_random_seed(
                base_seed,
                namespace="matched_route_assessment",
                terminal_index=terminal_index,
                checkpoint=checkpoint,
            ),
            reservation=reservation,
            base_snapshot_sha256=cache_contract.base_snapshot_sha256,
            planner_context_sha256=cache_contract.planner_context_sha256,
            cache_preflight_sha256=cache_contract.cache_preflight_sha256,
            cache_clone_id=cache_contract.clone_id(treatment_arm),
            post_hoc_productive_lock_sha256=(
                productive_lock_sha256 if treatment_arm == "post_hoc" else None
            ),
        )
        started = time.perf_counter()
        result = evaluator(terminal, context)
        observed_wall_seconds = time.perf_counter() - started
        if result.value_policy_id != expected_value_policy_id:
            raise UgiNonzeroGuidanceRunnerError(
                "productive terminal assessment violates the frozen binary policy"
            )
        per_terminal_ceiling = RouteComputeUsage(
            logical_planner_calls=planner_calls_per_terminal,
            physical_planner_calls=planner_calls_per_terminal,
            logical_verifier_calls=verifier_calls_per_terminal,
            physical_verifier_calls=verifier_calls_per_terminal,
        )
        if not result.usage.fits_within(per_terminal_ceiling):
            raise UgiNonzeroGuidanceRunnerError(
                "productive terminal assessment exceeded its per-terminal route budget"
            )
        usage = usage.plus(result.usage)
        censored_count += int(result.censored)
        wall_seconds += max(result.wall_seconds, observed_wall_seconds)
        gpu_device_seconds += result.gpu_device_seconds
        evaluation_receipt_sha256 = _sha256_payload(
            {
                "terminal_sha256": terminal.terminal_sha256,
                "generation_trace_sha256": terminal.generation_trace_sha256,
                "disposition": "assessed",
                "route_completion_utility": result.route_completion_utility,
                "value_policy_id": result.value_policy_id,
                "source_assessment_receipt_sha256": (result.assessment_receipt_sha256),
                "route_dossier_sha256": result.route_dossier_sha256,
                "usage": result.usage.__dict__,
                "route_seed": context.route_seed,
                "cache_clone_id": context.cache_clone_id,
                "productive_admission_disposition": admission.disposition,
                "canonical_identity": admission.canonical_identity,
            }
        )
        evaluated[terminal_index] = (result, context, evaluation_receipt_sha256)

    values: list[float | None] = []
    records: list[ProductiveAssessmentRecord] = []
    for terminal_index, (terminal, admission) in enumerate(zip(terminals, admissions, strict=True)):
        representative_index = admission.representative_index
        if representative_index is None:
            values.append(None)
            receipt = _sha256_payload(
                {
                    "terminal_sha256": terminal.terminal_sha256,
                    "generation_trace_sha256": terminal.generation_trace_sha256,
                    "disposition": admission.disposition,
                    "route_completion_utility": None,
                }
            )
            records.append(
                ProductiveAssessmentRecord(
                    terminal_id=terminal.terminal_id,
                    terminal_sha256=terminal.terminal_sha256,
                    generation_trace_sha256=terminal.generation_trace_sha256,
                    canonical_identity=None,
                    representative_terminal_id=None,
                    disposition=admission.disposition,
                    route_completion_utility=None,
                    value_policy_id=None,
                    source_assessment_receipt_sha256=None,
                    route_dossier_sha256=None,
                    evaluation_receipt_sha256=receipt,
                    route_seed=None,
                    usage=RouteComputeUsage(),
                    censored=False,
                )
            )
            continue
        result, context, representative_receipt = evaluated[representative_index]
        values.append(result.route_completion_utility)
        is_representative = terminal_index == representative_index
        receipt = (
            representative_receipt
            if is_representative
            else _sha256_payload(
                {
                    "terminal_sha256": terminal.terminal_sha256,
                    "generation_trace_sha256": terminal.generation_trace_sha256,
                    "disposition": "duplicate_shared_assessment",
                    "canonical_identity": admission.canonical_identity,
                    "representative_terminal_id": admission.representative_terminal_id,
                    "representative_evaluation_receipt_sha256": representative_receipt,
                }
            )
        )
        records.append(
            ProductiveAssessmentRecord(
                terminal_id=terminal.terminal_id,
                terminal_sha256=terminal.terminal_sha256,
                generation_trace_sha256=terminal.generation_trace_sha256,
                canonical_identity=admission.canonical_identity,
                representative_terminal_id=admission.representative_terminal_id,
                disposition=(
                    "assessed_representative"
                    if is_representative
                    else "duplicate_shared_assessment"
                ),
                route_completion_utility=result.route_completion_utility,
                value_policy_id=result.value_policy_id,
                source_assessment_receipt_sha256=result.assessment_receipt_sha256,
                route_dossier_sha256=result.route_dossier_sha256,
                evaluation_receipt_sha256=receipt,
                route_seed=context.route_seed,
                usage=result.usage if is_representative else RouteComputeUsage(),
                censored=result.censored,
            )
        )
    return (
        tuple(values),
        tuple(records),
        usage,
        invalid_count,
        nonexact_count,
        censored_count,
        wall_seconds,
        gpu_device_seconds,
    )


def _tie_priority(base_seed: int, canonical_identity: str) -> int:
    return keyed_random_seed(
        base_seed,
        namespace="matched_final_selection_tie",
        canonical_constitutional_identity=canonical_identity,
    )


def _select_final_candidates(
    admissions: Sequence[ProductiveAdmissionRecord],
    assessments: Sequence[ProductiveAssessmentRecord],
    *,
    final_count: int,
    base_seed: int,
) -> tuple[SelectedCandidate, ...]:
    if len(admissions) != len(assessments):
        raise UgiNonzeroGuidanceRunnerError("productive admissions and assessments are misaligned")

    def utility_rank(value: SelectedCandidate) -> int:
        if value.support_bonus:
            return 0
        if value.censored:
            return 2
        return 1

    candidates: list[SelectedCandidate] = []
    for admission, assessment in zip(admissions, assessments, strict=True):
        if admission.disposition != "canonical_representative":
            continue
        identity = admission.canonical_identity
        if identity is None or assessment.disposition != "assessed_representative":
            raise UgiNonzeroGuidanceRunnerError(
                "canonical representative lacks its one route assessment"
            )
        utility = assessment.route_completion_utility
        candidates.append(
            SelectedCandidate(
                canonical_identity=identity,
                terminal_id=admission.terminal_id,
                route_completion_utility=(None if utility is None else float(utility)),
                support_bonus=utility == 1.0,
                censored=utility is None,
                tie_priority=_tie_priority(base_seed, identity),
            )
        )
    ranked = sorted(
        candidates,
        key=lambda value: (
            utility_rank(value),
            value.tie_priority,
            value.canonical_identity,
            value.terminal_id,
        ),
    )
    if len(ranked) < final_count:
        raise UgiNonzeroGuidanceRunnerError(
            "constitutional deduplication leaves fewer candidates than the frozen final count"
        )
    return tuple(ranked[:final_count])


def run_development_matched_guidance(
    assignment: FrozenSeedProgramAssignment,
    *,
    lane: RestartableGuidanceLane,
    evaluator: RouteEvaluator,
    canonical_identity: CanonicalIdentity,
    design: ParticleGroupDesign,
    schedule: GuidanceSchedule,
    budget: GuidanceComputeBudget,
    guidance_strength: float,
    device: str,
    expected_value_policy_id: str,
    cache_contract: GuidanceCacheIsolationContract,
    cache_finalizer: Callable[[], GuidanceCacheFinalization],
) -> DevelopmentGuidanceRun:
    """Run deterministic fake/development SMC and matched post-hoc arms.

    This entrypoint is intentionally development-only.  A production wrapper
    must first authenticate a :class:`NonzeroGuidanceExecutionReview` and call
    :meth:`NonzeroGuidanceExecutionReview.require_nonzero_execution_authorized`.
    """

    if not isinstance(assignment, FrozenSeedProgramAssignment):
        raise UgiNonzeroGuidanceRunnerError("program assignment must be typed")
    base_seed = assignment.seed
    guidance_strength = _require_finite_nonnegative(
        guidance_strength,
        label="guidance_strength",
    )
    if not isinstance(device, str) or not device:
        raise UgiNonzeroGuidanceRunnerError("device must be a nonempty string")
    if not isinstance(expected_value_policy_id, str) or not expected_value_policy_id:
        raise UgiNonzeroGuidanceRunnerError("expected value-policy ID must be nonempty")
    if not isinstance(cache_contract, GuidanceCacheIsolationContract):
        raise UgiNonzeroGuidanceRunnerError("cache isolation contract must be typed")
    if not callable(cache_finalizer):
        raise UgiNonzeroGuidanceRunnerError("cache finalizer must be callable")
    budget.validate_exact_schedule(design, schedule)
    if len(assignment.programs) != design.morphology_program_count:
        raise UgiNonzeroGuidanceRunnerError(
            "program assignment count differs from the particle-group design"
        )
    if len(assignment.stochastic_particle_seeds) != design.particle_count:
        raise UgiNonzeroGuidanceRunnerError(
            "particle-seed assignment count differs from the particle-group design"
        )
    planner_per_terminal, verifier_per_terminal = budget.route_budget_per_terminal(
        design,
        schedule,
    )
    expanded = design.expanded_programs(assignment.programs)
    particle_seed_manifest_sha256 = _sha256_payload(assignment.stochastic_particle_seeds)

    generation_wall = {"guided": 0.0, "post_hoc": 0.0}
    generation_gpu = {"guided": 0.0, "post_hoc": 0.0}
    generation_transitions = {"guided": 0, "post_hoc": 0}

    def accept_state_receipt(
        value: GuidanceStateReceipt,
        *,
        treatment_arm: str,
        expected_transition_calls: int,
        observed_wall_seconds: float,
        expected_particle_seed_manifest_sha256: str | None = None,
    ) -> Any:
        if not isinstance(value, GuidanceStateReceipt):
            raise UgiNonzeroGuidanceRunnerError("restartable lane must return typed state receipts")
        if value.product_transition_calls != expected_transition_calls:
            raise UgiNonzeroGuidanceRunnerError(
                "state-operation transition use differs from the exact schedule"
            )
        if value.consumed_particle_seed_manifest_sha256 != expected_particle_seed_manifest_sha256:
            raise UgiNonzeroGuidanceRunnerError(
                "restartable lane did not attest the exact frozen particle-seed manifest"
            )
        generation_transitions[treatment_arm] += value.product_transition_calls
        generation_wall[treatment_arm] += observed_wall_seconds
        generation_gpu[treatment_arm] += value.gpu_device_seconds
        return value.state

    started = time.perf_counter()
    guided_initialization = lane.initialize(
        expanded,
        seed=base_seed,
        particle_seeds=assignment.stochastic_particle_seeds,
        device=device,
    )
    guided_state = accept_state_receipt(
        guided_initialization,
        treatment_arm="guided",
        expected_transition_calls=0,
        observed_wall_seconds=time.perf_counter() - started,
        expected_particle_seed_manifest_sha256=particle_seed_manifest_sha256,
    )
    started = time.perf_counter()
    post_hoc_initialization = lane.initialize(
        expanded,
        seed=base_seed,
        particle_seeds=assignment.stochastic_particle_seeds,
        device=device,
    )
    post_hoc_state = accept_state_receipt(
        post_hoc_initialization,
        treatment_arm="post_hoc",
        expected_transition_calls=0,
        observed_wall_seconds=time.perf_counter() - started,
        expected_particle_seed_manifest_sha256=particle_seed_manifest_sha256,
    )
    post_hoc_snapshots: dict[int, Any] = {}
    tempered = np.zeros(design.particle_count, dtype=np.float64)
    guided_checkpoint_records: list[CheckpointGroupRecord] = []

    previous_step = 0
    for checkpoint_position, checkpoint in enumerate(schedule.checkpoints):
        expected_advance_calls = design.particle_count * (checkpoint - previous_step)
        started = time.perf_counter()
        guided_advance = lane.advance(guided_state, target_step=checkpoint)
        guided_state = accept_state_receipt(
            guided_advance,
            treatment_arm="guided",
            expected_transition_calls=expected_advance_calls,
            observed_wall_seconds=time.perf_counter() - started,
        )
        started = time.perf_counter()
        post_hoc_advance = lane.advance(post_hoc_state, target_step=checkpoint)
        post_hoc_state = accept_state_receipt(
            post_hoc_advance,
            treatment_arm="post_hoc",
            expected_transition_calls=expected_advance_calls,
            observed_wall_seconds=time.perf_counter() - started,
        )
        started = time.perf_counter()
        snapshot = lane.snapshot(post_hoc_state)
        post_hoc_snapshots[checkpoint] = accept_state_receipt(
            snapshot,
            treatment_arm="post_hoc",
            expected_transition_calls=0,
            observed_wall_seconds=time.perf_counter() - started,
        )
        global_ancestors = np.arange(design.particle_count, dtype=np.int64)
        next_tempered = tempered.copy()
        for program_index in range(design.morphology_program_count):
            record, carried = _group_checkpoint(
                lane=lane,
                state=guided_state,
                design=design,
                schedule=schedule,
                base_seed=base_seed,
                guidance_strength=guidance_strength,
                checkpoint_position=checkpoint_position,
                program_index=program_index,
                tempered_before=tempered,
                planner_calls_per_terminal=planner_per_terminal,
                verifier_calls_per_terminal=verifier_per_terminal,
                evaluator=evaluator,
                expected_value_policy_id=expected_value_policy_id,
                treatment_arm="guided",
                cache_contract=cache_contract,
                post_hoc_productive_lock_sha256=None,
            )
            group = design.group_slice(program_index)
            global_ancestors[group] = np.asarray(record.global_ancestors, dtype=np.int64)
            next_tempered[group] = carried
            guided_checkpoint_records.append(record)
        started = time.perf_counter()
        ancestry_receipt = lane.apply_ancestry(
            guided_state,
            tuple(int(v) for v in global_ancestors),
        )
        guided_state = accept_state_receipt(
            ancestry_receipt,
            treatment_arm="guided",
            expected_transition_calls=0,
            observed_wall_seconds=time.perf_counter() - started,
        )
        tempered = next_tempered
        previous_step = checkpoint

    expected_final_advance_calls = design.particle_count * (schedule.sample_steps - previous_step)
    started = time.perf_counter()
    guided_final_advance = lane.advance(guided_state, target_step=schedule.sample_steps)
    guided_state = accept_state_receipt(
        guided_final_advance,
        treatment_arm="guided",
        expected_transition_calls=expected_final_advance_calls,
        observed_wall_seconds=time.perf_counter() - started,
    )
    started = time.perf_counter()
    post_hoc_final_advance = lane.advance(
        post_hoc_state,
        target_step=schedule.sample_steps,
    )
    post_hoc_state = accept_state_receipt(
        post_hoc_final_advance,
        treatment_arm="post_hoc",
        expected_transition_calls=expected_final_advance_calls,
        observed_wall_seconds=time.perf_counter() - started,
    )
    guided_productive_completion = _complete_productive_pool(
        lane=lane,
        state=guided_state,
        particle_count=design.particle_count,
        sample_steps=schedule.sample_steps,
        particle_seeds=assignment.stochastic_particle_seeds,
    )
    post_hoc_productive_completion = _complete_productive_pool(
        lane=lane,
        state=post_hoc_state,
        particle_count=design.particle_count,
        sample_steps=schedule.sample_steps,
        particle_seeds=assignment.stochastic_particle_seeds,
    )
    guided_productive = guided_productive_completion.terminals
    post_hoc_productive = post_hoc_productive_completion.terminals
    for treatment_arm, completion in (
        ("guided", guided_productive_completion),
        ("post_hoc", post_hoc_productive_completion),
    ):
        generation_transitions[treatment_arm] += completion.product_transition_calls
        generation_wall[treatment_arm] += completion.wall_seconds
        generation_gpu[treatment_arm] += completion.gpu_device_seconds
    guided_lock = _lock_manifest(guided_productive)
    post_hoc_lock = _lock_manifest(post_hoc_productive)
    guided_admissions, guided_admission_manifest = _build_productive_admission(
        guided_productive,
        canonical_identity=canonical_identity,
        final_count=schedule.final_selection_count,
        base_seed=base_seed,
    )
    post_hoc_admissions, post_hoc_admission_manifest = _build_productive_admission(
        post_hoc_productive,
        canonical_identity=canonical_identity,
        final_count=schedule.final_selection_count,
        base_seed=base_seed,
    )

    (
        guided_final_utilities,
        guided_final_records,
        guided_final_usage,
        guided_final_invalid,
        guided_final_nonexact,
        guided_final_censored,
        guided_final_wall,
        guided_final_gpu,
    ) = _assess_terminal_pool(
        guided_productive,
        guided_admissions,
        arm="guided_final",
        checkpoint=schedule.sample_steps,
        planner_calls_per_terminal=planner_per_terminal,
        verifier_calls_per_terminal=verifier_per_terminal,
        evaluator=evaluator,
        expected_value_policy_id=expected_value_policy_id,
        base_seed=base_seed,
        cache_contract=cache_contract,
        productive_lock_sha256=guided_lock,
    )
    (
        post_hoc_final_utilities,
        post_hoc_final_records,
        post_hoc_final_usage,
        post_hoc_final_invalid,
        post_hoc_final_nonexact,
        post_hoc_final_censored,
        post_hoc_final_wall,
        post_hoc_final_gpu,
    ) = _assess_terminal_pool(
        post_hoc_productive,
        post_hoc_admissions,
        arm="post_hoc_final",
        checkpoint=schedule.sample_steps,
        planner_calls_per_terminal=planner_per_terminal,
        verifier_calls_per_terminal=verifier_per_terminal,
        evaluator=evaluator,
        expected_value_policy_id=expected_value_policy_id,
        base_seed=base_seed,
        cache_contract=cache_contract,
        productive_lock_sha256=post_hoc_lock,
    )

    guided_selected = _select_final_candidates(
        guided_admissions,
        guided_final_records,
        final_count=schedule.final_selection_count,
        base_seed=base_seed,
    )
    post_hoc_selected = _select_final_candidates(
        post_hoc_admissions,
        post_hoc_final_records,
        final_count=schedule.final_selection_count,
        base_seed=base_seed,
    )

    # Matched shadow compute is deliberately delayed until the productive
    # post-hoc pool, traces and endpoint selection are all sealed.  These
    # records are discarded after accounting and can never enlarge or rank the
    # final candidate pool.
    post_hoc_checkpoint_records: list[CheckpointGroupRecord] = []
    dummy_tempered = np.zeros(design.particle_count, dtype=np.float64)
    for checkpoint_position, checkpoint in enumerate(schedule.checkpoints):
        snapshot = post_hoc_snapshots[checkpoint]
        for program_index in range(design.morphology_program_count):
            record, _ = _group_checkpoint(
                lane=lane,
                state=snapshot,
                design=design,
                schedule=schedule,
                base_seed=base_seed,
                guidance_strength=0.0,
                checkpoint_position=checkpoint_position,
                program_index=program_index,
                tempered_before=dummy_tempered,
                planner_calls_per_terminal=planner_per_terminal,
                verifier_calls_per_terminal=verifier_per_terminal,
                evaluator=evaluator,
                expected_value_policy_id=expected_value_policy_id,
                treatment_arm="post_hoc",
                cache_contract=cache_contract,
                post_hoc_productive_lock_sha256=post_hoc_lock,
            )
            post_hoc_checkpoint_records.append(record)
    cache_finalization = cache_finalizer()
    if not isinstance(cache_finalization, GuidanceCacheFinalization):
        raise UgiNonzeroGuidanceRunnerError(
            "cache finalizer must return a typed finalization receipt"
        )
    cache_finalization.validate_against(cache_contract)

    def arm_result(
        arm: str,
        terminals: Sequence[LockedMatchedTerminal],
        admissions: tuple[ProductiveAdmissionRecord, ...],
        admission_manifest_sha256: str,
        utilities: tuple[float | None, ...],
        lock_manifest: str,
        checkpoints: Sequence[CheckpointGroupRecord],
        *,
        shadow_delayed: bool,
        final_usage: RouteComputeUsage,
        final_records: tuple[ProductiveAssessmentRecord, ...],
        final_invalid: int,
        final_nonexact: int,
        final_censored: int,
        final_wall: float,
        final_gpu: float,
    ) -> ArmRunResult:
        shadow_completions = sum(value.terminal_completions for value in checkpoints)
        shadow_transitions = sum(value.product_transition_calls for value in checkpoints)
        shadow_usage = RouteComputeUsage(
            logical_planner_calls=sum(value.logical_planner_calls for value in checkpoints),
            physical_planner_calls=sum(value.physical_planner_calls for value in checkpoints),
            logical_verifier_calls=sum(value.verifier_calls for value in checkpoints),
            physical_verifier_calls=sum(value.physical_verifier_calls for value in checkpoints),
        )
        total_usage = final_usage.plus(shadow_usage)
        identities = tuple(value.canonical_identity for value in admissions)
        valid_identities = [value for value in identities if value is not None]
        result = ArmRunResult(
            arm=arm,
            productive_lock_manifest_sha256=lock_manifest,
            productive_terminal_ids=tuple(value.terminal_id for value in terminals),
            productive_canonical_identities=identities,
            productive_route_completion_utilities=utilities,
            productive_admission_manifest_sha256=admission_manifest_sha256,
            productive_admissions=admissions,
            productive_assessments=final_records,
            productive_pool_size=len(terminals),
            unique_productive_pool_size=len(set(valid_identities)),
            duplicate_productive_count=len(valid_identities) - len(set(valid_identities)),
            assessed_productive_count=sum(
                value.disposition == "assessed_representative" and not value.censored
                for value in final_records
            ),
            censored_productive_count=final_censored,
            selected=(guided_selected if arm == "guided" else post_hoc_selected),
            checkpoint_groups=tuple(checkpoints),
            shadow_assessment_started_after_productive_lock=shadow_delayed,
            shadow_assessments_selecting=False,
            base_cache_snapshot_sha256=cache_contract.base_snapshot_sha256,
            planner_context_sha256=cache_contract.planner_context_sha256,
            cache_preflight_sha256=cache_contract.cache_preflight_sha256,
            cache_clone_id=cache_contract.clone_id(arm),
            isolated_cache_overlay=cache_contract.isolated_overlay_roots,
            realized_terminal_completions=(
                guided_productive_completion.terminal_completions
                if arm == "guided"
                else post_hoc_productive_completion.terminal_completions
            )
            + shadow_completions,
            realized_logical_planner_calls=total_usage.logical_planner_calls,
            realized_physical_planner_calls=total_usage.physical_planner_calls,
            realized_verifier_calls=total_usage.logical_verifier_calls,
            realized_physical_verifier_calls=total_usage.physical_verifier_calls,
            planner_cache_hits=total_usage.planner_cache_hits,
            verifier_cache_hits=total_usage.verifier_cache_hits,
            invalid_terminal_count=final_invalid
            + sum(value.invalid_terminal_count for value in checkpoints),
            nonexact_l1_count=final_nonexact
            + sum(value.nonexact_l1_count for value in checkpoints),
            censored_assessment_count=final_censored
            + sum(value.censored_assessment_count for value in checkpoints),
            realized_product_transition_calls=(generation_transitions[arm] + shadow_transitions),
            wall_seconds=(
                generation_wall[arm] + final_wall + sum(value.wall_seconds for value in checkpoints)
            ),
            gpu_device_seconds=(
                generation_gpu[arm]
                + final_gpu
                + sum(value.gpu_device_seconds for value in checkpoints)
            ),
        )
        if result.realized_terminal_completions != budget.terminal_completions:
            raise UgiNonzeroGuidanceRunnerError(
                f"{arm} terminal completion ledger differs from the exact schedule"
            )
        if result.realized_product_transition_calls != budget.product_transition_calls:
            raise UgiNonzeroGuidanceRunnerError(
                f"{arm} transition ledger differs from the exact schedule"
            )
        return result

    return DevelopmentGuidanceRun(
        base_seed=base_seed,
        program_assignment_sha256=assignment.assignment_sha256,
        qualified_schedule_sha256=(
            assignment.schedule_sha256
            if assignment.qualification_result_sha256 is not None
            else None
        ),
        particle_seed_manifest_sha256=particle_seed_manifest_sha256,
        guidance_strength=guidance_strength,
        device=device,
        particle_group_design=design,
        schedule=schedule,
        budget=budget,
        cache_isolation=cache_contract,
        cache_finalization=cache_finalization,
        guided=arm_result(
            "guided",
            guided_productive,
            guided_admissions,
            guided_admission_manifest,
            guided_final_utilities,
            guided_lock,
            guided_checkpoint_records,
            shadow_delayed=False,
            final_usage=guided_final_usage,
            final_records=guided_final_records,
            final_invalid=guided_final_invalid,
            final_nonexact=guided_final_nonexact,
            final_censored=guided_final_censored,
            final_wall=guided_final_wall,
            final_gpu=guided_final_gpu,
        ),
        post_hoc=arm_result(
            "post_hoc",
            post_hoc_productive,
            post_hoc_admissions,
            post_hoc_admission_manifest,
            post_hoc_final_utilities,
            post_hoc_lock,
            post_hoc_checkpoint_records,
            shadow_delayed=True,
            final_usage=post_hoc_final_usage,
            final_records=post_hoc_final_records,
            final_invalid=post_hoc_final_invalid,
            final_nonexact=post_hoc_final_nonexact,
            final_censored=post_hoc_final_censored,
            final_wall=post_hoc_final_wall,
            final_gpu=post_hoc_final_gpu,
        ),
        production_execution=False,
        biological_guidance=False,
        private_holdout_accessed=False,
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise UgiNonzeroGuidanceRunnerError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiNonzeroGuidanceRunnerError(f"{label} must be a JSON object")
    return value


def load_grouped_smc_schedule_qualification(
    path: Path,
) -> GroupedSMCScheduleQualification:
    """Load and rederive every exact seed-to-program assignment."""

    value = _read_json(path, label="grouped SMC schedule qualification")
    if value.get("schema_version") != GROUPED_SMC_SCHEDULE_SCHEMA_VERSION:
        raise UgiNonzeroGuidanceRunnerError("unsupported grouped SMC schedule schema")
    if value.get("status") != "grouped_smc_schedule_qualified_nonexecuting":
        raise UgiNonzeroGuidanceRunnerError("grouped SMC schedule is not qualified")
    claimed_result_sha256 = _require_sha256(
        value.get("result_sha256"),
        label="grouped SMC result_sha256",
    )
    content = {key: item for key, item in value.items() if key != "result_sha256"}
    if _sha256_payload(content) != claimed_result_sha256:
        raise UgiNonzeroGuidanceRunnerError("grouped SMC result checksum mismatch")
    qualification_file_sha256 = sha256_file(path)
    expected_design = {
        "programs_per_seed": 16,
        "particles_per_program": 4,
        "particles_per_seed": 64,
        "ancestry_group_key": ["seed", "program_index"],
        "within_program_ancestry_has_four_choices": True,
        "cross_program_ancestry_forbidden": True,
    }
    design = value.get("design")
    if not isinstance(design, Mapping) or any(
        design.get(key) != expected for key, expected in expected_design.items()
    ):
        raise UgiNonzeroGuidanceRunnerError("grouped SMC design changed")
    integration = value.get("integration_contract")
    if not isinstance(integration, Mapping) or any(
        integration.get(key) is not True
        for key in (
            "sampler_receives_each_program_four_times",
            "ancestry_selection_invoked_separately_per_program_group",
            "particle_index_must_not_be_reinterpreted_as_program_index",
            "sixty_four_unique_programs_forbidden",
        )
    ):
        raise UgiNonzeroGuidanceRunnerError("grouped SMC integration contract changed")
    schedules = value.get("seed_schedules")
    if not isinstance(schedules, list) or len(schedules) != 8:
        raise UgiNonzeroGuidanceRunnerError("grouped SMC schedule must contain eight seeds")
    assignments: list[FrozenSeedProgramAssignment] = []
    all_programs: set[bytes] = set()
    all_particle_seeds: set[int] = set()
    for schedule in schedules:
        if not isinstance(schedule, Mapping) or set(schedule) != {
            "seed",
            "programs",
            "particles",
            "schedule_sha256",
        }:
            raise UgiNonzeroGuidanceRunnerError("seed schedule is malformed")
        schedule_content = {key: schedule[key] for key in ("seed", "programs", "particles")}
        schedule_sha256 = _require_sha256(
            schedule.get("schedule_sha256"),
            label="seed schedule_sha256",
        )
        if _sha256_payload(schedule_content) != schedule_sha256:
            raise UgiNonzeroGuidanceRunnerError("seed schedule checksum mismatch")
        seed = _require_nonnegative_integer(schedule.get("seed"), label="schedule seed")
        program_records = schedule.get("programs")
        particle_records = schedule.get("particles")
        if not isinstance(program_records, list) or len(program_records) != 16:
            raise UgiNonzeroGuidanceRunnerError("seed schedule needs 16 program records")
        if not isinstance(particle_records, list) or len(particle_records) != 64:
            raise UgiNonzeroGuidanceRunnerError("seed schedule needs 64 particle records")
        programs: list[bytes] = []
        program_sha256s: list[str] = []
        for program_index, record in enumerate(program_records):
            if not isinstance(record, Mapping) or record.get("program_index") != program_index:
                raise UgiNonzeroGuidanceRunnerError("program records are not canonically ordered")
            payload = canonical_morphology_program_bytes(record.get("program"))
            program_sha256 = _require_sha256(
                record.get("morphology_program_sha256"),
                label="morphology_program_sha256",
            )
            if hashlib.sha256(payload).hexdigest() != program_sha256:
                raise UgiNonzeroGuidanceRunnerError("program hash changed")
            if payload in all_programs:
                raise UgiNonzeroGuidanceRunnerError(
                    "frozen morphology programs must be disjoint across seeds"
                )
            all_programs.add(payload)
            programs.append(payload)
            program_sha256s.append(program_sha256)
        stochastic_seeds: list[int] = []
        for global_index, record in enumerate(particle_records):
            if not isinstance(record, Mapping):
                raise UgiNonzeroGuidanceRunnerError("particle assignment is malformed")
            program_index = global_index // 4
            within_index = global_index % 4
            if (
                record.get("global_particle_index") != global_index
                or record.get("program_index") != program_index
                or record.get("within_program_particle_index") != within_index
                or record.get("morphology_program_sha256") != program_sha256s[program_index]
                or record.get("ancestry_group") != {"seed": seed, "program_index": program_index}
            ):
                raise UgiNonzeroGuidanceRunnerError(
                    "particle assignment violates program-major four-particle grouping"
                )
            particle_seed = _require_nonnegative_integer(
                record.get("stochastic_particle_seed"),
                label="stochastic particle seed",
            )
            if particle_seed in all_particle_seeds:
                raise UgiNonzeroGuidanceRunnerError(
                    "stochastic particle seed is reused across frozen schedules"
                )
            all_particle_seeds.add(particle_seed)
            stochastic_seeds.append(particle_seed)
        assignments.append(
            FrozenSeedProgramAssignment(
                seed=seed,
                schedule_sha256=schedule_sha256,
                programs=tuple(programs),
                program_sha256s=tuple(program_sha256s),
                stochastic_particle_seeds=tuple(stochastic_seeds),
                qualification_file_sha256=qualification_file_sha256,
                qualification_result_sha256=claimed_result_sha256,
            )
        )
    return GroupedSMCScheduleQualification(
        file_sha256=qualification_file_sha256,
        result_sha256=claimed_result_sha256,
        assignments=tuple(assignments),
    )


def _pinned_path(repo: Path, record: Mapping[str, Any], *, label: str) -> Path:
    if set(record) != {"path", "sha256"}:
        raise UgiNonzeroGuidanceRunnerError(f"{label} pin must contain path and sha256")
    path = (repo / str(record["path"])).resolve()
    try:
        relative = path.relative_to(repo.resolve())
    except ValueError as error:
        raise UgiNonzeroGuidanceRunnerError(f"{label} path escapes the repository") from error
    lowered = "/".join(relative.parts).lower()
    if "holdout" in lowered or "sealed" in lowered:
        raise UgiNonzeroGuidanceRunnerError(f"{label} path is forbidden for this runner")
    if sha256_file(path) != record["sha256"]:
        raise UgiNonzeroGuidanceRunnerError(f"{label} hash changed")
    return path


def load_execution_review(path: Path) -> NonzeroGuidanceExecutionReview:
    """Load one canonical execution-review artifact without granting authority."""

    canonical = path.read_bytes()
    value = _read_json(path, label="execution review")
    if canonical != (_stable_json(value) + "\n").encode():
        raise UgiNonzeroGuidanceRunnerError("execution review must be canonical JSON")
    if value.get("schema_version") != EXECUTION_REVIEW_SCHEMA_VERSION:
        raise UgiNonzeroGuidanceRunnerError("unsupported execution-review schema")
    gates = value.get("gates")
    scope = value.get("scope")
    if not isinstance(gates, dict) or not isinstance(scope, dict):
        raise UgiNonzeroGuidanceRunnerError("execution review is malformed")
    expected_gate_fields = {
        "route_completion_utility_frozen",
        "current_l3_snapshot_frozen",
        "cumulative_source_runtime_qualified",
        "production_zero_guidance_passed",
        "particle_group_amendment_frozen",
    }
    if set(gates) != expected_gate_fields:
        raise UgiNonzeroGuidanceRunnerError("execution-review gate set changed")
    expected_scope_fields = {
        "nonzero_guidance_authorized",
        "biological_guidance_authorized",
        "candidate_selection_authorized",
        "sealed_holdout_access_authorized",
    }
    if set(scope) != expected_scope_fields:
        raise UgiNonzeroGuidanceRunnerError("execution-review scope changed")
    return NonzeroGuidanceExecutionReview(
        receipt_sha256=hashlib.sha256(canonical).hexdigest(),
        **gates,
        **scope,
    )


def build_nonzero_guidance_runner_plan(
    repo: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Validate the device-agnostic runner contract without executing guidance."""

    repo = repo.resolve()
    config = _read_json(config_path, label="nonzero-guidance runner config")
    if config.get("schema_version") != RUNNER_CONFIG_SCHEMA_VERSION:
        raise UgiNonzeroGuidanceRunnerError("unsupported nonzero-guidance runner config")
    if config.get("scope") != "development_runner_preflight_nonzero_execution_blocked":
        raise UgiNonzeroGuidanceRunnerError("runner scope changed")
    execution = config.get("execution")
    if execution != {
        "development_fake_execution_allowed": True,
        "production_nonzero_execution": False,
        "biological_guidance": False,
        "development_endpoint_subset_selection": True,
        "prospective_candidate_lock": False,
        "sealed_holdout_access": False,
        "cloud_wrapper_included": False,
    }:
        raise UgiNonzeroGuidanceRunnerError("runner execution guards changed")
    design_value = config.get("particle_group_design")
    schedule_value = config.get("schedule")
    budget_value = config.get("matched_compute")
    if not all(isinstance(value, dict) for value in (design_value, schedule_value, budget_value)):
        raise UgiNonzeroGuidanceRunnerError("runner design, schedule, or budget is malformed")
    design = ParticleGroupDesign(
        morphology_program_count=design_value.get("morphology_program_count"),
        particles_per_program=design_value.get("particles_per_program"),
    )
    if design.particle_count != design_value.get("particles_per_seed"):
        raise UgiNonzeroGuidanceRunnerError("particle-group product differs from total particles")
    if design.morphology_program_count != 16 or design.particles_per_program != 4:
        raise UgiNonzeroGuidanceRunnerError(
            "the v2 amendment must freeze 16 programs x 4 particles"
        )
    schedule = GuidanceSchedule(
        sample_steps=schedule_value.get("sample_steps"),
        checkpoints=tuple(schedule_value.get("synthesis_checkpoints", ())),
        checkpoint_betas=tuple(schedule_value.get("checkpoint_betas", ())),
        rollouts_per_particle_checkpoint=schedule_value.get(
            "terminal_rollouts_per_particle_checkpoint"
        ),
        final_selection_count=schedule_value.get("final_selection_count"),
    )
    expected_schedule = GuidanceSchedule(
        sample_steps=8,
        checkpoints=(2, 4, 6),
        checkpoint_betas=(0.25, 0.5, 0.75),
        rollouts_per_particle_checkpoint=1,
        final_selection_count=32,
    )
    if schedule != expected_schedule:
        raise UgiNonzeroGuidanceRunnerError("runner schedule differs from the v2 amendment")
    budget = GuidanceComputeBudget(
        product_transition_calls=budget_value.get("product_transition_calls"),
        terminal_completions=budget_value.get("terminal_completions"),
        logical_planner_calls=budget_value.get("logical_planner_calls"),
        logical_verifier_calls=budget_value.get("logical_verifier_calls"),
        final_candidates=budget_value.get("final_candidates"),
    )
    expected_budget = GuidanceComputeBudget(
        product_transition_calls=1280,
        terminal_completions=256,
        logical_planner_calls=768,
        logical_verifier_calls=3328,
        final_candidates=32,
    )
    if budget != expected_budget:
        raise UgiNonzeroGuidanceRunnerError("matched-compute budget changed")
    budget.validate_exact_schedule(design, schedule)
    expected_post_hoc_contract = {
        "productive_finals_per_seed": 64,
        "productive_final_bytes_and_generation_traces_sealed_before_shadow_assessment": True,
        "endpoint_subset_sealed_before_shadow_assessment": True,
        "checkpoint_shadow_completions_per_seed": 192,
        "shadow_completions_enter_productive_pool": False,
        "shadow_assessments_influence_generation": False,
        "shadow_assessments_influence_endpoint_selection": False,
    }
    if config.get("post_hoc_contract") != expected_post_hoc_contract:
        raise UgiNonzeroGuidanceRunnerError("post-hoc shadow contract changed")
    expected_final_selection_contract = {
        "invalid_or_nonexact_l1_excluded_before_identity": True,
        "canonical_identity_source": "immutable_terminal_bytes_only",
        "constitutional_deduplication_precedes_productive_route_assessment": True,
        "one_route_blind_representative_assessed_per_constitutional_identity": True,
        "representative_choice_may_not_use_route_outcome": True,
        "duplicates_share_representative_assessment": True,
        "duplicates_add_zero_route_compute": True,
        "rank_order": [
            "exact_dossier_1",
            "observed_incomplete_0",
            "censored_null",
        ],
        "tie_break": "keyed_random_seed_and_canonical_constitutional_identity",
        "identical_rule_across_guided_and_post_hoc": True,
    }
    if config.get("final_selection_contract") != expected_final_selection_contract:
        raise UgiNonzeroGuidanceRunnerError("route-blind final-selection contract changed")
    pins = config.get("inputs")
    if not isinstance(pins, dict) or not pins:
        raise UgiNonzeroGuidanceRunnerError("runner inputs must be hash-pinned")
    expected_inputs = {
        "grouped_smc_schedule_qualification",
        "matched_preregistration_v2",
        "route_completion_utility_qualification",
        "current_source_zero_guidance_requalification",
        "matched_planner_cache_binding_qualification",
        "selected_restartable_generator",
        "production_terminal_route_evaluator",
        "runner_source",
        "runner_tests",
        "runner_cli",
    }
    if set(pins) != expected_inputs:
        raise UgiNonzeroGuidanceRunnerError("runner input set changed")
    paths = {
        label: _pinned_path(repo, record, label=label)
        for label, record in pins.items()
        if isinstance(record, Mapping)
    }
    if set(paths) != set(pins):
        raise UgiNonzeroGuidanceRunnerError("runner input pin is malformed")
    qualification = load_grouped_smc_schedule_qualification(
        paths["grouped_smc_schedule_qualification"]
    )
    frozen_seed_assignments = config.get("seed_assignments")
    if not isinstance(frozen_seed_assignments, Mapping) or set(frozen_seed_assignments) != {
        "calibration",
        "evaluation",
    }:
        raise UgiNonzeroGuidanceRunnerError("seed-assignment roles are malformed")
    expected_role_seeds = {
        "calibration": tuple(value.seed for value in qualification.assignments[:3]),
        "evaluation": tuple(value.seed for value in qualification.assignments[3:]),
    }
    qualification_by_seed = qualification.by_seed()
    for role, expected_seeds in expected_role_seeds.items():
        records = frozen_seed_assignments.get(role)
        if (
            not isinstance(records, list)
            or tuple(value.get("seed") if isinstance(value, Mapping) else None for value in records)
            != expected_seeds
        ):
            raise UgiNonzeroGuidanceRunnerError(
                f"{role} seed assignments differ from the frozen schedule"
            )
        for record in records:
            assignment = qualification_by_seed[record["seed"]]
            if set(record) != {
                "seed",
                "schedule_sha256",
                "program_sha256s",
                "particle_seed_manifest_sha256",
            }:
                raise UgiNonzeroGuidanceRunnerError(f"{role} seed assignment fields changed")
            if (
                record["schedule_sha256"] != assignment.schedule_sha256
                or tuple(record["program_sha256s"]) != assignment.program_sha256s
                or record["particle_seed_manifest_sha256"]
                != _sha256_payload(assignment.stochastic_particle_seeds)
            ):
                raise UgiNonzeroGuidanceRunnerError(
                    f"{role} seed assignment no longer matches the qualification"
                )
    cache_policy = config.get("cache_isolation")
    if cache_policy != {
        "one_immutable_base_snapshot": True,
        "guided_and_post_hoc_overlay_roots_distinct": True,
        "identical_empty_overlay_start_required": True,
        "runner_clone_ids_authoritative": True,
        "final_before_after_audit_required": True,
        "production_adapter_wired": False,
    }:
        raise UgiNonzeroGuidanceRunnerError("cache-isolation policy changed")
    if config.get("authorization") != {
        "nonzero_guidance_authorized": False,
        "remaining_blockers": [
            "grouped_lambda_zero_bitwise_execution",
            "production_cache_and_generator_adapter_qualification",
            "fresh_current_l3_execution_review",
        ],
    }:
        raise UgiNonzeroGuidanceRunnerError(
            "development preflight must keep nonzero production execution blocked"
        )
    result = {
        "schema_version": RUNNER_PLAN_SCHEMA_VERSION,
        "status": "device_agnostic_runner_prepared_nonzero_execution_blocked",
        "config": {
            "path": str(config_path.resolve().relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "particle_group_design": {
            "morphology_program_count": design.morphology_program_count,
            "particles_per_program": design.particles_per_program,
            "particles_per_seed": design.particle_count,
            "ancestry_scope": "within_frozen_morphology_program_only",
        },
        "schedule": {
            "sample_steps": schedule.sample_steps,
            "synthesis_checkpoints": list(schedule.checkpoints),
            "checkpoint_betas": list(schedule.checkpoint_betas),
            "terminal_rollouts_per_particle_checkpoint": (
                schedule.rollouts_per_particle_checkpoint
            ),
            "final_selection_count": schedule.final_selection_count,
        },
        "matched_compute": budget.__dict__,
        "post_hoc_contract": expected_post_hoc_contract,
        "final_selection_contract": expected_final_selection_contract,
        "grouped_smc_schedule": {
            "file_sha256": qualification.file_sha256,
            "result_sha256": qualification.result_sha256,
            "calibration_seeds": list(expected_role_seeds["calibration"]),
            "evaluation_seeds": list(expected_role_seeds["evaluation"]),
            "assignments": [
                {
                    "seed": value.seed,
                    "schedule_sha256": value.schedule_sha256,
                    "assignment_sha256": value.assignment_sha256,
                    "program_sha256s": list(value.program_sha256s),
                    "particle_seed_manifest_sha256": _sha256_payload(
                        value.stochastic_particle_seeds
                    ),
                }
                for value in qualification.assignments
            ],
        },
        "cache_isolation": cache_policy,
        "authorization": config.get("authorization"),
        "inputs": {
            label: {
                "path": str(path.relative_to(repo)),
                "sha256": sha256_file(path),
            }
            for label, path in sorted(paths.items())
        },
        "scope": execution,
    }
    return {**result, "plan_sha256": _sha256_payload(result)}


__all__ = [
    "ArmRunResult",
    "CheckpointGroupRecord",
    "DevelopmentGuidanceRun",
    "GuidanceComputeBudget",
    "GuidanceRouteEvaluation",
    "GuidanceSchedule",
    "NonzeroGuidanceExecutionReview",
    "ParticleGroupDesign",
    "RestartableGuidanceLane",
    "SelectedCandidate",
    "UgiNonzeroGuidanceRunnerError",
    "build_nonzero_guidance_runner_plan",
    "development_guidance_run_to_dict",
    "load_execution_review",
    "run_development_matched_guidance",
]
