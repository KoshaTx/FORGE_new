"""Diagnostic matched-budget orchestration for future Ugi guidance experiments.

This module coordinates synthetic guided and post-hoc arms without defining a
synthesis scalar or running chemistry. Productive generation uses shared keyed
substreams, route assessment uses arm-specific keyed substreams, and post-hoc
assessment is deferred until every admitted terminal and trace has been sealed.

The contract is intentionally conservative: schedule entries reserve a common
per-arm route budget before either arm runs. Budget exhaustion therefore censors
one identical suffix in both arms without dummy calls or extra compute.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from forge.core.seeds import keyed_seed


class UgiMatchedBudgetError(RuntimeError):
    """Raised when diagnostic matched-arm execution violates its contract."""


class MatchedArm(str, Enum):
    """The two causally distinct arms in the matched comparison."""

    GUIDED = "guided"
    POST_HOC = "post_hoc"


class MatchedDisposition(str, Enum):
    """One terminal's nonselecting orchestration disposition."""

    ASSESSED = "assessed"
    INVALID_TERMINAL = "invalid_terminal"
    NONEXACT_L1 = "nonexact_l1"
    SHARED_BUDGET_CENSORED = "shared_budget_censored"


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UgiMatchedBudgetError(f"{name} must be a nonnegative integer")
    return value


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise UgiMatchedBudgetError(f"{name} must be a nonempty string")
    return value


def _sha256_string(name: str, value: Any) -> str:
    value = _nonempty_string(name, value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise UgiMatchedBudgetError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class RouteComputeUsage:
    """Logical and physical route-compute calls for one assessment."""

    logical_planner_calls: int = 0
    physical_planner_calls: int = 0
    logical_verifier_calls: int = 0
    physical_verifier_calls: int = 0

    def __post_init__(self) -> None:
        for name in (
            "logical_planner_calls",
            "physical_planner_calls",
            "logical_verifier_calls",
            "physical_verifier_calls",
        ):
            _nonnegative_integer(name, getattr(self, name))
        if self.physical_planner_calls > self.logical_planner_calls:
            raise UgiMatchedBudgetError(
                "physical_planner_calls cannot exceed logical_planner_calls"
            )
        if self.physical_verifier_calls > self.logical_verifier_calls:
            raise UgiMatchedBudgetError(
                "physical_verifier_calls cannot exceed logical_verifier_calls"
            )

    @property
    def planner_cache_hits(self) -> int:
        """Logical planner calls served without physical execution."""

        return self.logical_planner_calls - self.physical_planner_calls

    @property
    def verifier_cache_hits(self) -> int:
        """Logical verifier calls served without physical execution."""

        return self.logical_verifier_calls - self.physical_verifier_calls

    def plus(self, other: RouteComputeUsage) -> RouteComputeUsage:
        """Return field-wise usage addition."""

        return RouteComputeUsage(
            logical_planner_calls=self.logical_planner_calls + other.logical_planner_calls,
            physical_planner_calls=self.physical_planner_calls + other.physical_planner_calls,
            logical_verifier_calls=self.logical_verifier_calls + other.logical_verifier_calls,
            physical_verifier_calls=self.physical_verifier_calls + other.physical_verifier_calls,
        )

    def fits_within(self, ceiling: RouteComputeUsage) -> bool:
        """Return whether every usage field is no larger than its ceiling."""

        return all(
            getattr(self, name) <= getattr(ceiling, name)
            for name in (
                "logical_planner_calls",
                "physical_planner_calls",
                "logical_verifier_calls",
                "physical_verifier_calls",
            )
        )

    def remaining_after(self, used: RouteComputeUsage) -> RouteComputeUsage:
        """Subtract used calls from a ceiling, failing on overuse."""

        if not used.fits_within(self):
            raise UgiMatchedBudgetError("route usage exceeds its declared ceiling")
        return RouteComputeUsage(
            logical_planner_calls=self.logical_planner_calls - used.logical_planner_calls,
            physical_planner_calls=min(
                self.logical_planner_calls - used.logical_planner_calls,
                self.physical_planner_calls - used.physical_planner_calls,
            ),
            logical_verifier_calls=self.logical_verifier_calls - used.logical_verifier_calls,
            physical_verifier_calls=min(
                self.logical_verifier_calls - used.logical_verifier_calls,
                self.physical_verifier_calls - used.physical_verifier_calls,
            ),
        )


@dataclass(frozen=True)
class MatchedBudgetLimits:
    """One shared budget ceiling applied independently to both arms."""

    productive_generation_calls: int
    terminal_completions: int
    final_candidates: int
    route: RouteComputeUsage

    def __post_init__(self) -> None:
        for name in (
            "productive_generation_calls",
            "terminal_completions",
            "final_candidates",
        ):
            _nonnegative_integer(name, getattr(self, name))


@dataclass(frozen=True)
class MatchedScheduleEntry:
    """One common morphology/checkpoint unit with a frozen budget reservation."""

    unit_id: str
    morphology_program: bytes
    program_index: int
    particle_index: int
    checkpoint_index: int
    generator_checkpoint_sha256: str
    closure_checkpoint_sha256: str
    rollout_index: int
    productive_generation_calls: int
    route_reservation: RouteComputeUsage

    def __post_init__(self) -> None:
        _nonempty_string("unit_id", self.unit_id)
        if not isinstance(self.morphology_program, bytes) or not self.morphology_program:
            raise UgiMatchedBudgetError("morphology_program must be nonempty bytes")
        for name in (
            "program_index",
            "particle_index",
            "checkpoint_index",
            "rollout_index",
            "productive_generation_calls",
        ):
            _nonnegative_integer(name, getattr(self, name))
        _sha256_string("generator_checkpoint_sha256", self.generator_checkpoint_sha256)
        _sha256_string("closure_checkpoint_sha256", self.closure_checkpoint_sha256)
        if self.productive_generation_calls == 0:
            raise UgiMatchedBudgetError("productive_generation_calls must be positive")

    @property
    def morphology_program_sha256(self) -> str:
        """Return the exact shared morphology-program digest."""

        return hashlib.sha256(self.morphology_program).hexdigest()


@dataclass(frozen=True)
class MatchedGenerationRequest:
    """Shared productive-generation request passed to either arm."""

    arm: MatchedArm
    entry: MatchedScheduleEntry
    productive_seed: int


@dataclass(frozen=True)
class LockedMatchedTerminal:
    """A sealed terminal candidate and trace returned by a generation callback."""

    unit_id: str
    morphology_program_sha256: str
    checkpoint_index: int
    generator_checkpoint_sha256: str
    closure_checkpoint_sha256: str
    terminal_id: str
    terminal_locked: bool
    terminal_valid: bool
    exact_l1: bool
    terminal_bytes: bytes
    generation_trace_bytes: bytes
    payload: Any = None

    def __post_init__(self) -> None:
        _nonempty_string("unit_id", self.unit_id)
        _sha256_string("morphology_program_sha256", self.morphology_program_sha256)
        _sha256_string("generator_checkpoint_sha256", self.generator_checkpoint_sha256)
        _sha256_string("closure_checkpoint_sha256", self.closure_checkpoint_sha256)
        _nonempty_string("terminal_id", self.terminal_id)
        _nonnegative_integer("checkpoint_index", self.checkpoint_index)
        if not all(
            isinstance(value, bool)
            for value in (self.terminal_locked, self.terminal_valid, self.exact_l1)
        ):
            raise UgiMatchedBudgetError("terminal admission fields must be boolean")
        if self.exact_l1 and not self.terminal_valid:
            raise UgiMatchedBudgetError("an invalid terminal cannot be exact L1")
        for name in ("terminal_bytes", "generation_trace_bytes"):
            value = getattr(self, name)
            if not isinstance(value, bytes) or not value:
                raise UgiMatchedBudgetError(f"{name} must be nonempty bytes")

    @property
    def terminal_sha256(self) -> str:
        return hashlib.sha256(self.terminal_bytes).hexdigest()

    @property
    def generation_trace_sha256(self) -> str:
        return hashlib.sha256(self.generation_trace_bytes).hexdigest()


DiagnosticField = tuple[str, str]


@dataclass(frozen=True)
class DiagnosticRouteAssessment:
    """Nonselecting fake structured fields plus realized route usage."""

    value_policy_id: str
    diagnostic_fields: tuple[DiagnosticField, ...]
    usage: RouteComputeUsage

    def __post_init__(self) -> None:
        _nonempty_string("value_policy_id", self.value_policy_id)
        if not self.diagnostic_fields:
            raise UgiMatchedBudgetError("diagnostic_fields cannot be empty")
        keys = []
        for field in self.diagnostic_fields:
            if (
                not isinstance(field, tuple)
                or len(field) != 2
                or not all(isinstance(value, str) and value for value in field)
            ):
                raise UgiMatchedBudgetError(
                    "diagnostic_fields must be nonempty string key/value pairs"
                )
            keys.append(field[0])
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise UgiMatchedBudgetError("diagnostic_fields must have unique keys in sorted order")


@dataclass(frozen=True)
class MatchedAssessmentContext:
    """Arm-specific route substream and frozen call ceilings."""

    arm: MatchedArm
    route_seed: int
    remaining_budget: RouteComputeUsage
    unit_reservation: RouteComputeUsage
    cache_snapshot_sha256: str
    cache_clone_id: str
    post_hoc_lock_manifest_sha256: str | None


@dataclass(frozen=True)
class MatchedArmRecord:
    """One arm's auditable outcome for one common schedule unit."""

    unit_id: str
    morphology_program_sha256: str
    checkpoint_index: int
    generator_checkpoint_sha256: str
    closure_checkpoint_sha256: str
    productive_seed: int
    route_seed: int | None
    cache_snapshot_sha256: str
    cache_clone_id: str
    terminal_id: str | None
    terminal_sha256: str | None
    generation_trace_sha256: str | None
    disposition: MatchedDisposition
    route_assessed: bool
    diagnostic_fields: tuple[DiagnosticField, ...] | None
    value_policy_id: str | None
    route_usage: RouteComputeUsage
    detail: str | None


@dataclass(frozen=True)
class MatchedArmLedger:
    """Realized productive and route calls for one arm."""

    productive_generation_calls: int
    terminal_completions: int
    final_candidates: int
    route: RouteComputeUsage

    @property
    def planner_cache_hits(self) -> int:
        return self.route.planner_cache_hits

    @property
    def verifier_cache_hits(self) -> int:
        return self.route.verifier_cache_hits


@dataclass(frozen=True)
class MatchedBudgetRun:
    """Deterministic two-arm result with a shared censored suffix."""

    schedule_sha256: str
    cache_snapshot_sha256: str
    guided_cache_clone_id: str
    post_hoc_cache_clone_id: str
    zero_guidance: bool
    zero_guidance_bitwise_equivalent: bool | None
    post_hoc_lock_manifest_sha256: str
    guided_records: tuple[MatchedArmRecord, ...]
    post_hoc_records: tuple[MatchedArmRecord, ...]
    guided_ledger: MatchedArmLedger
    post_hoc_ledger: MatchedArmLedger
    shared_censored_unit_ids: tuple[str, ...]


def _schedule_sha256(schedule: Sequence[MatchedScheduleEntry]) -> str:
    payload = [
        {
            "unit_id": entry.unit_id,
            "morphology_program_sha256": entry.morphology_program_sha256,
            "program_index": entry.program_index,
            "particle_index": entry.particle_index,
            "checkpoint_index": entry.checkpoint_index,
            "generator_checkpoint_sha256": entry.generator_checkpoint_sha256,
            "closure_checkpoint_sha256": entry.closure_checkpoint_sha256,
            "rollout_index": entry.rollout_index,
            "productive_generation_calls": entry.productive_generation_calls,
            "route_reservation": {
                "logical_planner_calls": entry.route_reservation.logical_planner_calls,
                "physical_planner_calls": entry.route_reservation.physical_planner_calls,
                "logical_verifier_calls": entry.route_reservation.logical_verifier_calls,
                "physical_verifier_calls": entry.route_reservation.physical_verifier_calls,
            },
        }
        for entry in schedule
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _lock_manifest_sha256(
    terminals: Sequence[tuple[MatchedScheduleEntry, LockedMatchedTerminal]],
) -> str:
    digest = hashlib.sha256()
    for entry, terminal in terminals:
        for value in (
            entry.unit_id.encode(),
            entry.morphology_program_sha256.encode(),
            str(entry.checkpoint_index).encode(),
            entry.generator_checkpoint_sha256.encode(),
            entry.closure_checkpoint_sha256.encode(),
            terminal.terminal_id.encode(),
            terminal.terminal_bytes,
            terminal.generation_trace_bytes,
        ):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def _validate_terminal(
    entry: MatchedScheduleEntry,
    terminal: LockedMatchedTerminal,
) -> None:
    if not terminal.terminal_locked:
        raise UgiMatchedBudgetError(
            f"unit {entry.unit_id}: route assessment requires a sealed terminal"
        )
    if terminal.unit_id != entry.unit_id:
        raise UgiMatchedBudgetError(f"unit {entry.unit_id}: terminal unit_id mismatch")
    if terminal.morphology_program_sha256 != entry.morphology_program_sha256:
        raise UgiMatchedBudgetError(
            f"unit {entry.unit_id}: morphology program changed across the arm boundary"
        )
    if terminal.checkpoint_index != entry.checkpoint_index:
        raise UgiMatchedBudgetError(f"unit {entry.unit_id}: checkpoint index mismatch")
    if terminal.generator_checkpoint_sha256 != entry.generator_checkpoint_sha256:
        raise UgiMatchedBudgetError(f"unit {entry.unit_id}: generator checkpoint hash mismatch")
    if terminal.closure_checkpoint_sha256 != entry.closure_checkpoint_sha256:
        raise UgiMatchedBudgetError(f"unit {entry.unit_id}: closure checkpoint hash mismatch")


def _cache_clone_id(
    cache_snapshot_sha256: str,
    arm: MatchedArm,
    schedule_sha256: str,
) -> str:
    payload = {
        "arm": arm.value,
        "cache_snapshot_sha256": cache_snapshot_sha256,
        "schedule_sha256": schedule_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _route_seed(base_seed: int, arm: MatchedArm, entry: MatchedScheduleEntry) -> int:
    return keyed_seed(
        base_seed,
        namespace="matched_route_assessment",
        arm=arm.value,
        program_index=entry.program_index,
        particle_index=entry.particle_index,
        checkpoint_index=entry.checkpoint_index,
        rollout_index=entry.rollout_index,
    )


def _productive_seed(base_seed: int, entry: MatchedScheduleEntry) -> int:
    return keyed_seed(
        base_seed,
        namespace="matched_productive_generation",
        morphology_program_sha256=entry.morphology_program_sha256,
        program_index=entry.program_index,
        particle_index=entry.particle_index,
        checkpoint_index=entry.checkpoint_index,
        rollout_index=entry.rollout_index,
    )


def _route_record(
    *,
    entry: MatchedScheduleEntry,
    productive_seed: int,
    terminal: LockedMatchedTerminal,
    route_seed: int | None,
    cache_snapshot_sha256: str,
    cache_clone_id: str,
    assessment: DiagnosticRouteAssessment | None,
) -> MatchedArmRecord:
    if not terminal.terminal_valid:
        disposition = MatchedDisposition.INVALID_TERMINAL
    elif not terminal.exact_l1:
        disposition = MatchedDisposition.NONEXACT_L1
    else:
        disposition = MatchedDisposition.ASSESSED
    return MatchedArmRecord(
        unit_id=entry.unit_id,
        morphology_program_sha256=entry.morphology_program_sha256,
        checkpoint_index=entry.checkpoint_index,
        generator_checkpoint_sha256=entry.generator_checkpoint_sha256,
        closure_checkpoint_sha256=entry.closure_checkpoint_sha256,
        productive_seed=productive_seed,
        route_seed=route_seed,
        cache_snapshot_sha256=cache_snapshot_sha256,
        cache_clone_id=cache_clone_id,
        terminal_id=terminal.terminal_id,
        terminal_sha256=terminal.terminal_sha256,
        generation_trace_sha256=terminal.generation_trace_sha256,
        disposition=disposition,
        route_assessed=assessment is not None,
        diagnostic_fields=(assessment.diagnostic_fields if assessment else None),
        value_policy_id=(assessment.value_policy_id if assessment else None),
        route_usage=(assessment.usage if assessment else RouteComputeUsage()),
        detail=(None if assessment else "route evaluator not called"),
    )


def _censored_record(
    entry: MatchedScheduleEntry,
    base_seed: int,
    *,
    cache_snapshot_sha256: str,
    cache_clone_id: str,
) -> MatchedArmRecord:
    return MatchedArmRecord(
        unit_id=entry.unit_id,
        morphology_program_sha256=entry.morphology_program_sha256,
        checkpoint_index=entry.checkpoint_index,
        generator_checkpoint_sha256=entry.generator_checkpoint_sha256,
        closure_checkpoint_sha256=entry.closure_checkpoint_sha256,
        productive_seed=_productive_seed(base_seed, entry),
        route_seed=None,
        cache_snapshot_sha256=cache_snapshot_sha256,
        cache_clone_id=cache_clone_id,
        terminal_id=None,
        terminal_sha256=None,
        generation_trace_sha256=None,
        disposition=MatchedDisposition.SHARED_BUDGET_CENSORED,
        route_assessed=False,
        diagnostic_fields=None,
        value_policy_id=None,
        route_usage=RouteComputeUsage(),
        detail="shared preregistered budget reservation exhausted",
    )


def _assess(
    *,
    arm: MatchedArm,
    entry: MatchedScheduleEntry,
    terminal: LockedMatchedTerminal,
    base_seed: int,
    used: RouteComputeUsage,
    limits: MatchedBudgetLimits,
    post_hoc_lock_manifest_sha256: str | None,
    cache_snapshot_sha256: str,
    cache_clone_id: str,
    assess_terminal: Callable[
        [LockedMatchedTerminal, MatchedAssessmentContext], DiagnosticRouteAssessment
    ],
) -> tuple[DiagnosticRouteAssessment | None, RouteComputeUsage, int | None]:
    if not terminal.terminal_valid or not terminal.exact_l1:
        return None, used, None
    if arm is MatchedArm.POST_HOC and not post_hoc_lock_manifest_sha256:
        raise UgiMatchedBudgetError(
            "post-hoc route assessment requires a sealed terminal lock manifest"
        )
    route_seed = _route_seed(base_seed, arm, entry)
    remaining = limits.route.remaining_after(used)
    assessment = assess_terminal(
        terminal,
        MatchedAssessmentContext(
            arm=arm,
            route_seed=route_seed,
            remaining_budget=remaining,
            unit_reservation=entry.route_reservation,
            cache_snapshot_sha256=cache_snapshot_sha256,
            cache_clone_id=cache_clone_id,
            post_hoc_lock_manifest_sha256=post_hoc_lock_manifest_sha256,
        ),
    )
    if not assessment.usage.fits_within(entry.route_reservation):
        raise UgiMatchedBudgetError(
            f"unit {entry.unit_id}: route evaluator exceeded its frozen reservation"
        )
    updated = used.plus(assessment.usage)
    if not updated.fits_within(limits.route):
        raise UgiMatchedBudgetError(
            f"unit {entry.unit_id}: route evaluator exceeded the shared arm budget"
        )
    return assessment, updated, route_seed


def run_diagnostic_matched_budget_arms(
    schedule: Sequence[MatchedScheduleEntry],
    *,
    budget_limits: MatchedBudgetLimits,
    base_seed: int,
    zero_guidance: bool,
    cache_snapshot_sha256: str,
    generate_terminal: Callable[[MatchedGenerationRequest], LockedMatchedTerminal],
    assess_terminal: Callable[
        [LockedMatchedTerminal, MatchedAssessmentContext], DiagnosticRouteAssessment
    ],
) -> MatchedBudgetRun:
    """Run a synthetic matched guided/post-hoc common-prefix contract.

    This function does not select, filter, rank, or resample candidates. Guided
    assessments are available immediately after their sealed rollout terminal;
    post-hoc assessments occur only after the complete admitted post-hoc lock
    manifest is sealed. At zero guidance, productive terminal and trace bytes
    must match exactly across arms.
    """

    _nonnegative_integer("base_seed", base_seed)
    _sha256_string("cache_snapshot_sha256", cache_snapshot_sha256)
    if not isinstance(zero_guidance, bool):
        raise UgiMatchedBudgetError("zero_guidance must be boolean")
    schedule = tuple(schedule)
    if not schedule:
        raise UgiMatchedBudgetError("matched schedule cannot be empty")
    unit_ids = [entry.unit_id for entry in schedule]
    coordinates = [
        (
            entry.program_index,
            entry.particle_index,
            entry.checkpoint_index,
            entry.rollout_index,
        )
        for entry in schedule
    ]
    if len(unit_ids) != len(set(unit_ids)):
        raise UgiMatchedBudgetError("matched schedule unit IDs must be unique")
    if len(coordinates) != len(set(coordinates)):
        raise UgiMatchedBudgetError("matched schedule coordinates must be unique")

    schedule_sha256 = _schedule_sha256(schedule)
    cache_clone_ids = {
        arm: _cache_clone_id(cache_snapshot_sha256, arm, schedule_sha256) for arm in MatchedArm
    }
    if len(set(cache_clone_ids.values())) != len(cache_clone_ids):
        raise UgiMatchedBudgetError("matched arms must use distinct cache clone IDs")

    reserved_route = RouteComputeUsage()
    productive_calls = 0
    admitted_count = 0
    guided_route = RouteComputeUsage()
    post_hoc_route = RouteComputeUsage()
    post_hoc_terminals: list[tuple[MatchedScheduleEntry, LockedMatchedTerminal, int]] = []
    guided_records_by_unit: dict[str, MatchedArmRecord] = {}
    post_hoc_records_by_unit: dict[str, MatchedArmRecord] = {}
    censored_entries: tuple[MatchedScheduleEntry, ...] = ()

    for schedule_index, entry in enumerate(schedule):
        proposed_productive = productive_calls + entry.productive_generation_calls
        proposed_count = admitted_count + 1
        proposed_reserved_route = reserved_route.plus(entry.route_reservation)
        if (
            proposed_productive > budget_limits.productive_generation_calls
            or proposed_count > budget_limits.terminal_completions
            or proposed_count > budget_limits.final_candidates
            or not proposed_reserved_route.fits_within(budget_limits.route)
        ):
            censored_entries = schedule[schedule_index:]
            break

        productive_seed = _productive_seed(base_seed, entry)
        guided = generate_terminal(
            MatchedGenerationRequest(
                arm=MatchedArm.GUIDED,
                entry=entry,
                productive_seed=productive_seed,
            )
        )
        _validate_terminal(entry, guided)
        guided_assessment, guided_route, guided_route_seed = _assess(
            arm=MatchedArm.GUIDED,
            entry=entry,
            terminal=guided,
            base_seed=base_seed,
            used=guided_route,
            limits=budget_limits,
            post_hoc_lock_manifest_sha256=None,
            cache_snapshot_sha256=cache_snapshot_sha256,
            cache_clone_id=cache_clone_ids[MatchedArm.GUIDED],
            assess_terminal=assess_terminal,
        )
        guided_records_by_unit[entry.unit_id] = _route_record(
            entry=entry,
            productive_seed=productive_seed,
            terminal=guided,
            route_seed=guided_route_seed,
            cache_snapshot_sha256=cache_snapshot_sha256,
            cache_clone_id=cache_clone_ids[MatchedArm.GUIDED],
            assessment=guided_assessment,
        )
        post_hoc = generate_terminal(
            MatchedGenerationRequest(
                arm=MatchedArm.POST_HOC,
                entry=entry,
                productive_seed=productive_seed,
            )
        )
        _validate_terminal(entry, post_hoc)
        post_hoc_terminals.append((entry, post_hoc, productive_seed))

        if zero_guidance and (
            guided.terminal_bytes != post_hoc.terminal_bytes
            or guided.generation_trace_bytes != post_hoc.generation_trace_bytes
            or guided.terminal_valid != post_hoc.terminal_valid
            or guided.exact_l1 != post_hoc.exact_l1
        ):
            raise UgiMatchedBudgetError(
                f"unit {entry.unit_id}: zero-guidance terminal/trace bytes diverged"
            )

        productive_calls = proposed_productive
        admitted_count = proposed_count
        reserved_route = proposed_reserved_route

    lock_manifest = _lock_manifest_sha256(
        [(entry, terminal) for entry, terminal, _ in post_hoc_terminals]
    )
    for entry, terminal, productive_seed in post_hoc_terminals:
        assessment, post_hoc_route, route_seed = _assess(
            arm=MatchedArm.POST_HOC,
            entry=entry,
            terminal=terminal,
            base_seed=base_seed,
            used=post_hoc_route,
            limits=budget_limits,
            post_hoc_lock_manifest_sha256=lock_manifest,
            cache_snapshot_sha256=cache_snapshot_sha256,
            cache_clone_id=cache_clone_ids[MatchedArm.POST_HOC],
            assess_terminal=assess_terminal,
        )
        post_hoc_records_by_unit[entry.unit_id] = _route_record(
            entry=entry,
            productive_seed=productive_seed,
            terminal=terminal,
            route_seed=route_seed,
            cache_snapshot_sha256=cache_snapshot_sha256,
            cache_clone_id=cache_clone_ids[MatchedArm.POST_HOC],
            assessment=assessment,
        )
        guided_record = guided_records_by_unit[entry.unit_id]
        post_hoc_record = post_hoc_records_by_unit[entry.unit_id]
        if (
            guided_record.route_assessed
            and post_hoc_record.route_assessed
            and guided_record.value_policy_id != post_hoc_record.value_policy_id
        ):
            raise UgiMatchedBudgetError(
                f"unit {entry.unit_id}: arms used different diagnostic value policies"
            )

    for entry in censored_entries:
        guided_records_by_unit[entry.unit_id] = _censored_record(
            entry,
            base_seed,
            cache_snapshot_sha256=cache_snapshot_sha256,
            cache_clone_id=cache_clone_ids[MatchedArm.GUIDED],
        )
        post_hoc_records_by_unit[entry.unit_id] = _censored_record(
            entry,
            base_seed,
            cache_snapshot_sha256=cache_snapshot_sha256,
            cache_clone_id=cache_clone_ids[MatchedArm.POST_HOC],
        )

    guided_records = tuple(guided_records_by_unit[entry.unit_id] for entry in schedule)
    post_hoc_records = tuple(post_hoc_records_by_unit[entry.unit_id] for entry in schedule)
    return MatchedBudgetRun(
        schedule_sha256=schedule_sha256,
        cache_snapshot_sha256=cache_snapshot_sha256,
        guided_cache_clone_id=cache_clone_ids[MatchedArm.GUIDED],
        post_hoc_cache_clone_id=cache_clone_ids[MatchedArm.POST_HOC],
        zero_guidance=zero_guidance,
        zero_guidance_bitwise_equivalent=(True if zero_guidance else None),
        post_hoc_lock_manifest_sha256=lock_manifest,
        guided_records=guided_records,
        post_hoc_records=post_hoc_records,
        guided_ledger=MatchedArmLedger(
            productive_generation_calls=productive_calls,
            terminal_completions=admitted_count,
            final_candidates=admitted_count,
            route=guided_route,
        ),
        post_hoc_ledger=MatchedArmLedger(
            productive_generation_calls=productive_calls,
            terminal_completions=admitted_count,
            final_candidates=admitted_count,
            route=post_hoc_route,
        ),
        shared_censored_unit_ids=tuple(entry.unit_id for entry in censored_entries),
    )
