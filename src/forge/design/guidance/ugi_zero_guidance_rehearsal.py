"""Additive zero-guidance rehearsal for matched Ugi generation and routing.

This module composes the hash-pinned matched runner with a dependency-injected
restartable generator/closure adapter, typed terminal-route assessment and two
isolated planner-cache overlays.  It exercises the real seams at guidance
strength zero while defining no synthesis scalar and performing no selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.core.io import stable_json as _stable_json
from forge.design import ugi_matched_budget_orchestration as matched_runner
from forge.design.ugi_matched_budget_orchestration import (
    DiagnosticRouteAssessment,
    LockedMatchedTerminal,
    MatchedArm,
    MatchedArmRecord,
    MatchedAssessmentContext,
    MatchedBudgetLimits,
    MatchedBudgetRun,
    MatchedGenerationRequest,
    MatchedScheduleEntry,
    RouteComputeUsage,
)
from forge.design.ugi_matched_planner_cache_binding import (
    MatchedPlannerCacheBindingPreflight,
    preflight_lazy_matched_planner_cache_binding,
)
from forge.route.engine.planner import RoutePlanner
from forge.route.engine.planner_cache import FilePlannerCache, PlannerCacheContext
from forge.route.engine.planner_cache_snapshot import (
    MatchedPlannerCacheOverlayAudit,
    OverlayFilePlannerCache,
    planner_cache_context_sha256,
)
from forge.route.terminals import terminal_assessment as terminal_route_assessment
from forge.route.terminals.terminal_assessment import (
    QualifiedUgiL1Reverifier,
    UgiTerminalRouteAssessmentReceipt,
    ValidatedUgiTerminalPayload,
    assess_locked_ugi_terminal_routes,
    required_three_role_route_reservation,
)

ZERO_GUIDANCE_REHEARSAL_SCHEMA_VERSION = "forge.ugi_zero_guidance_rehearsal.v1"
ZERO_GUIDANCE_REHEARSAL_PREFLIGHT_SCHEMA_VERSION = "forge.ugi_zero_guidance_rehearsal_preflight.v1"
HASH_PINNED_MATCHED_RUNNER_SHA256 = (
    "0643853cc15f92faf88c0c731db8e5691fbc899123386d2f4fd7e49806608cf5"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class UgiZeroGuidanceRehearsalError(RuntimeError):
    """Raised when a zero-guidance rehearsal contract is violated."""


def _module_sha256(module: Any, *, label: str) -> str:
    path_value = getattr(module, "__file__", None)
    if not isinstance(path_value, str) or not path_value:
        raise UgiZeroGuidanceRehearsalError(f"{label} source path is unavailable")
    path = Path(path_value)
    if not path.is_file():
        raise UgiZeroGuidanceRehearsalError(f"{label} source file is unavailable")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_terminal_route_assessment_source_sha256() -> str:
    """Return the currently imported typed assessment implementation digest."""

    return _module_sha256(terminal_route_assessment, label="terminal-route assessment")


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise UgiZeroGuidanceRehearsalError(
            f"{label} must contain 64 lowercase hexadecimal characters"
        )
    return value


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise UgiZeroGuidanceRehearsalError(f"{label} must be a nonempty string")
    return value


def _route_usage_to_dict(value: RouteComputeUsage) -> dict[str, int]:
    return {
        "logical_planner_calls": value.logical_planner_calls,
        "physical_planner_calls": value.physical_planner_calls,
        "logical_verifier_calls": value.logical_verifier_calls,
        "physical_verifier_calls": value.physical_verifier_calls,
    }


def _budget_limits_to_dict(value: MatchedBudgetLimits) -> dict[str, Any]:
    return {
        "productive_generation_calls": value.productive_generation_calls,
        "terminal_completions": value.terminal_completions,
        "final_candidates": value.final_candidates,
        "route": _route_usage_to_dict(value.route),
    }


def _sum_route_usage(values: Sequence[RouteComputeUsage]) -> RouteComputeUsage:
    output = RouteComputeUsage()
    for value in values:
        output = output.plus(value)
    return output


@dataclass(frozen=True)
class RestartableGeneratorClosureIdentity:
    """Hash ownership for the selected restartable generator/closure seam."""

    generator_checkpoint_sha256: str
    closure_checkpoint_sha256: str
    production_generator_manifest_sha256: str
    restartable_equivalence_receipt_sha256: str
    generator_implementation_sha256: str
    terminal_decoder_id: str

    def __post_init__(self) -> None:
        for name in (
            "generator_checkpoint_sha256",
            "closure_checkpoint_sha256",
            "production_generator_manifest_sha256",
            "restartable_equivalence_receipt_sha256",
            "generator_implementation_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        _require_nonempty(self.terminal_decoder_id, label="terminal_decoder_id")

    def to_dict(self) -> dict[str, str]:
        return {
            "generator_checkpoint_sha256": self.generator_checkpoint_sha256,
            "closure_checkpoint_sha256": self.closure_checkpoint_sha256,
            "production_generator_manifest_sha256": (self.production_generator_manifest_sha256),
            "restartable_equivalence_receipt_sha256": (self.restartable_equivalence_receipt_sha256),
            "generator_implementation_sha256": self.generator_implementation_sha256,
            "terminal_decoder_id": self.terminal_decoder_id,
        }


@dataclass(frozen=True)
class RestartableGeneratorClosureAdapter:
    """Dependency-injected productive terminal generator with frozen identity."""

    identity: RestartableGeneratorClosureIdentity
    generate_locked_terminal: Callable[[MatchedGenerationRequest], LockedMatchedTerminal]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RestartableGeneratorClosureIdentity):
            raise UgiZeroGuidanceRehearsalError("generator/closure identity is malformed")
        if not callable(self.generate_locked_terminal):
            raise UgiZeroGuidanceRehearsalError("generate_locked_terminal must be callable")


RoutePlannerFactory = Callable[
    [
        OverlayFilePlannerCache,
        PlannerCacheContext,
        MatchedAssessmentContext,
        LockedMatchedTerminal,
    ],
    RoutePlanner,
]


@dataclass(frozen=True)
class QualifiedRoutePlannerFactoryAdapter:
    """Dependency-injected planner factory bound to one qualification receipt."""

    qualification_sha256: str
    build_planner: RoutePlannerFactory

    def __post_init__(self) -> None:
        _require_sha256(self.qualification_sha256, label="route qualification")
        if not callable(self.build_planner):
            raise UgiZeroGuidanceRehearsalError("build_planner must be callable")


@dataclass(frozen=True)
class ZeroGuidanceRehearsalContract:
    """Fully pinned inputs for one real, nonselecting zero-guidance rehearsal."""

    base_seed: int
    guidance_strength: float
    assessment_at_utc: str
    budget_limits: MatchedBudgetLimits
    generator_identity: RestartableGeneratorClosureIdentity
    planner_context_sha256: str
    expected_cache_snapshot_sha256: str
    terminal_route_assessment_source_sha256: str
    route_qualification_sha256: str
    matched_runner_source_sha256: str = HASH_PINNED_MATCHED_RUNNER_SHA256

    def __post_init__(self) -> None:
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise UgiZeroGuidanceRehearsalError("base_seed must be a nonnegative integer")
        if self.base_seed < 0:
            raise UgiZeroGuidanceRehearsalError("base_seed must be a nonnegative integer")
        if (
            isinstance(self.guidance_strength, bool)
            or not isinstance(self.guidance_strength, (int, float))
            or not math.isfinite(float(self.guidance_strength))
            or float(self.guidance_strength) != 0.0
        ):
            raise UgiZeroGuidanceRehearsalError("this entrypoint permits guidance_strength=0 only")
        _require_nonempty(self.assessment_at_utc, label="assessment_at_utc")
        if not isinstance(self.budget_limits, MatchedBudgetLimits):
            raise UgiZeroGuidanceRehearsalError("matched budget limits are malformed")
        if not isinstance(self.generator_identity, RestartableGeneratorClosureIdentity):
            raise UgiZeroGuidanceRehearsalError("generator identity is malformed")
        for name in (
            "planner_context_sha256",
            "expected_cache_snapshot_sha256",
            "terminal_route_assessment_source_sha256",
            "route_qualification_sha256",
            "matched_runner_source_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if self.matched_runner_source_sha256 != HASH_PINNED_MATCHED_RUNNER_SHA256:
            raise UgiZeroGuidanceRehearsalError("matched runner source pin changed")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "forge.ugi_zero_guidance_rehearsal_contract.v1",
            "base_seed": self.base_seed,
            "guidance_strength": 0,
            "assessment_at_utc": self.assessment_at_utc,
            "budget_limits": _budget_limits_to_dict(self.budget_limits),
            "generator_identity": self.generator_identity.to_dict(),
            "planner_context_sha256": self.planner_context_sha256,
            "expected_cache_snapshot_sha256": self.expected_cache_snapshot_sha256,
            "terminal_route_assessment_source_sha256": (
                self.terminal_route_assessment_source_sha256
            ),
            "route_qualification_sha256": self.route_qualification_sha256,
            "matched_runner_source_sha256": self.matched_runner_source_sha256,
        }

    @property
    def contract_sha256(self) -> str:
        return _sha256_payload(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "contract_sha256": self.contract_sha256}


@dataclass(frozen=True)
class ZeroGuidanceScheduleManifest:
    """Immutable canonical manifest of the exact public schedule inputs."""

    content_json: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.content_json, label="schedule manifest content")
        _require_sha256(self.manifest_sha256, label="schedule manifest checksum")
        try:
            content = json.loads(self.content_json)
        except json.JSONDecodeError as error:
            raise UgiZeroGuidanceRehearsalError(
                "schedule manifest content is invalid JSON"
            ) from error
        if self.content_json != _stable_json(content):
            raise UgiZeroGuidanceRehearsalError("schedule manifest content is not canonical JSON")
        if (
            not isinstance(content, dict)
            or content.get("schema_version")
            != "forge.ugi_zero_guidance_rehearsal_schedule_manifest.v1"
            or not isinstance(content.get("entries"), list)
            or content.get("entry_count") != len(content["entries"])
        ):
            raise UgiZeroGuidanceRehearsalError("schedule manifest content is malformed")
        if self.manifest_sha256 != _sha256_payload(content):
            raise UgiZeroGuidanceRehearsalError("schedule manifest checksum mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            **json.loads(self.content_json),
            "manifest_sha256": self.manifest_sha256,
        }


def _schedule_manifest(
    schedule: Sequence[MatchedScheduleEntry],
) -> ZeroGuidanceScheduleManifest:
    entries = [
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
            "route_reservation": _route_usage_to_dict(entry.route_reservation),
        }
        for entry in schedule
    ]
    content = {
        "schema_version": "forge.ugi_zero_guidance_rehearsal_schedule_manifest.v1",
        "entry_count": len(entries),
        "entries": entries,
    }
    return ZeroGuidanceScheduleManifest(
        content_json=_stable_json(content),
        manifest_sha256=_sha256_payload(content),
    )


@dataclass(frozen=True)
class ZeroGuidanceRehearsalPreflight:
    """Input and cache seal completed before the first generator callback."""

    contract: ZeroGuidanceRehearsalContract
    schedule_manifest: ZeroGuidanceScheduleManifest
    cache_binding: MatchedPlannerCacheBindingPreflight

    def __post_init__(self) -> None:
        if not isinstance(self.contract, ZeroGuidanceRehearsalContract):
            raise UgiZeroGuidanceRehearsalError("rehearsal contract is malformed")
        if not isinstance(self.schedule_manifest, ZeroGuidanceScheduleManifest):
            raise UgiZeroGuidanceRehearsalError("schedule manifest is malformed")
        if not isinstance(self.cache_binding, MatchedPlannerCacheBindingPreflight):
            raise UgiZeroGuidanceRehearsalError("cache-binding preflight is malformed")
        if self.cache_binding.base.snapshot_sha256 != self.contract.expected_cache_snapshot_sha256:
            raise UgiZeroGuidanceRehearsalError("cache snapshot changed before generation")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ZERO_GUIDANCE_REHEARSAL_PREFLIGHT_SCHEMA_VERSION,
            "contract": self.contract.to_dict(),
            "schedule_manifest": self.schedule_manifest.to_dict(),
            "cache_binding": self.cache_binding.to_dict(),
            "completed_before_productive_generation": True,
        }

    @property
    def preflight_sha256(self) -> str:
        return _sha256_payload(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "preflight_sha256": self.preflight_sha256}


@dataclass(frozen=True)
class ChemistryProjectionPairReceipt:
    """Arm-paired constitutional chemistry projection for one generated unit."""

    unit_id: str
    terminal_bytes_identical: bool
    generation_trace_bytes_identical: bool
    projection_status: str
    guided_projection_sha256: str | None
    post_hoc_projection_sha256: str | None
    chemistry_projection_identical: bool | None

    def __post_init__(self) -> None:
        _require_nonempty(self.unit_id, label="unit_id")
        if self.terminal_bytes_identical is not True:
            raise UgiZeroGuidanceRehearsalError("zero-guidance terminal bytes differ")
        if self.generation_trace_bytes_identical is not True:
            raise UgiZeroGuidanceRehearsalError("zero-guidance generation traces differ")
        if self.projection_status == "validated_exact_l1":
            _require_sha256(self.guided_projection_sha256, label="guided projection")
            _require_sha256(self.post_hoc_projection_sha256, label="post-hoc projection")
            if self.chemistry_projection_identical is not True:
                raise UgiZeroGuidanceRehearsalError("zero-guidance chemistry projections differ")
        elif self.projection_status in {"invalid_terminal", "nonexact_l1"}:
            if any(
                value is not None
                for value in (
                    self.guided_projection_sha256,
                    self.post_hoc_projection_sha256,
                    self.chemistry_projection_identical,
                )
            ):
                raise UgiZeroGuidanceRehearsalError(
                    "nonassessable terminal cannot claim a chemistry projection"
                )
        else:
            raise UgiZeroGuidanceRehearsalError("unsupported projection status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "terminal_bytes_identical": self.terminal_bytes_identical,
            "generation_trace_bytes_identical": self.generation_trace_bytes_identical,
            "projection_status": self.projection_status,
            "guided_projection_sha256": self.guided_projection_sha256,
            "post_hoc_projection_sha256": self.post_hoc_projection_sha256,
            "chemistry_projection_identical": self.chemistry_projection_identical,
        }


def _arm_record_to_dict(value: MatchedArmRecord) -> dict[str, Any]:
    return {
        "unit_id": value.unit_id,
        "morphology_program_sha256": value.morphology_program_sha256,
        "checkpoint_index": value.checkpoint_index,
        "generator_checkpoint_sha256": value.generator_checkpoint_sha256,
        "closure_checkpoint_sha256": value.closure_checkpoint_sha256,
        "productive_seed": value.productive_seed,
        "route_seed": value.route_seed,
        "cache_snapshot_sha256": value.cache_snapshot_sha256,
        "cache_clone_id": value.cache_clone_id,
        "terminal_id": value.terminal_id,
        "terminal_sha256": value.terminal_sha256,
        "generation_trace_sha256": value.generation_trace_sha256,
        "disposition": value.disposition.value,
        "route_assessed": value.route_assessed,
        "diagnostic_fields": (
            [list(item) for item in value.diagnostic_fields]
            if value.diagnostic_fields is not None
            else None
        ),
        "value_policy_id": value.value_policy_id,
        "route_usage": _route_usage_to_dict(value.route_usage),
        "detail": value.detail,
    }


def _matched_run_to_dict(value: MatchedBudgetRun) -> dict[str, Any]:
    return {
        "schedule_sha256": value.schedule_sha256,
        "cache_snapshot_sha256": value.cache_snapshot_sha256,
        "guided_cache_clone_id": value.guided_cache_clone_id,
        "post_hoc_cache_clone_id": value.post_hoc_cache_clone_id,
        "zero_guidance": value.zero_guidance,
        "zero_guidance_bitwise_equivalent": value.zero_guidance_bitwise_equivalent,
        "post_hoc_lock_manifest_sha256": value.post_hoc_lock_manifest_sha256,
        "guided_records": [_arm_record_to_dict(item) for item in value.guided_records],
        "post_hoc_records": [_arm_record_to_dict(item) for item in value.post_hoc_records],
        "guided_ledger": {
            "productive_generation_calls": value.guided_ledger.productive_generation_calls,
            "terminal_completions": value.guided_ledger.terminal_completions,
            "final_candidates": value.guided_ledger.final_candidates,
            "route": _route_usage_to_dict(value.guided_ledger.route),
        },
        "post_hoc_ledger": {
            "productive_generation_calls": value.post_hoc_ledger.productive_generation_calls,
            "terminal_completions": value.post_hoc_ledger.terminal_completions,
            "final_candidates": value.post_hoc_ledger.final_candidates,
            "route": _route_usage_to_dict(value.post_hoc_ledger.route),
        },
        "shared_censored_unit_ids": list(value.shared_censored_unit_ids),
    }


@dataclass(frozen=True)
class ZeroGuidanceRehearsalResult:
    """Complete receipts for one matched zero-guidance seam rehearsal."""

    preflight: ZeroGuidanceRehearsalPreflight
    matched_run: MatchedBudgetRun
    chemistry_projections: tuple[ChemistryProjectionPairReceipt, ...]
    guided_route_receipts: tuple[UgiTerminalRouteAssessmentReceipt, ...]
    post_hoc_route_receipts: tuple[UgiTerminalRouteAssessmentReceipt, ...]
    cache_audit: MatchedPlannerCacheOverlayAudit

    def __post_init__(self) -> None:
        if self.matched_run.zero_guidance is not True:
            raise UgiZeroGuidanceRehearsalError("rehearsal did not use zero guidance")
        if self.matched_run.zero_guidance_bitwise_equivalent is not True:
            raise UgiZeroGuidanceRehearsalError("zero-guidance arm identity was not proven")
        if not self.chemistry_projections or not any(
            item.projection_status == "validated_exact_l1" for item in self.chemistry_projections
        ):
            raise UgiZeroGuidanceRehearsalError(
                "rehearsal requires at least one validated chemistry projection"
            )
        if len(self.guided_route_receipts) != len(self.post_hoc_route_receipts):
            raise UgiZeroGuidanceRehearsalError("matched route receipt counts differ")
        if not self.guided_route_receipts:
            raise UgiZeroGuidanceRehearsalError(
                "rehearsal requires at least one paired route assessment"
            )
        if self.cache_audit.guided_clone_id != self.matched_run.guided_cache_clone_id:
            raise UgiZeroGuidanceRehearsalError("guided cache clone receipt mismatch")
        if self.cache_audit.post_hoc_clone_id != self.matched_run.post_hoc_cache_clone_id:
            raise UgiZeroGuidanceRehearsalError("post-hoc cache clone receipt mismatch")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ZERO_GUIDANCE_REHEARSAL_SCHEMA_VERSION,
            "status": "zero_guidance_rehearsal_complete",
            "preflight": self.preflight.to_dict(),
            "matched_run": _matched_run_to_dict(self.matched_run),
            "chemistry_projections": [item.to_dict() for item in self.chemistry_projections],
            "chemistry_projection_identity_proven": True,
            "guided_route_receipts": [item.to_dict() for item in self.guided_route_receipts],
            "post_hoc_route_receipts": [item.to_dict() for item in self.post_hoc_route_receipts],
            "cache_audit": self.cache_audit.to_dict(),
            "scope": {
                "guidance_strength": 0,
                "synthesis_scalar_defined": False,
                "nonzero_synthesis_guidance": False,
                "biological_guidance": False,
                "candidate_selection": False,
                "private_holdout_accessed": False,
            },
        }

    @property
    def result_sha256(self) -> str:
        return _sha256_payload(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "result_sha256": self.result_sha256}

    @property
    def canonical_bytes(self) -> bytes:
        return (_stable_json(self.to_dict()) + "\n").encode()


def _validate_exact_rehearsal_budget(
    schedule: tuple[MatchedScheduleEntry, ...],
    *,
    contract: ZeroGuidanceRehearsalContract,
    planner_context: PlannerCacheContext,
) -> None:
    if not schedule:
        raise UgiZeroGuidanceRehearsalError("rehearsal schedule cannot be empty")
    required = required_three_role_route_reservation(planner_context.budget_limits)
    for entry in schedule:
        if entry.generator_checkpoint_sha256 != (
            contract.generator_identity.generator_checkpoint_sha256
        ):
            raise UgiZeroGuidanceRehearsalError(
                f"unit {entry.unit_id}: selected generator checkpoint hash drifted"
            )
        if entry.closure_checkpoint_sha256 != (
            contract.generator_identity.closure_checkpoint_sha256
        ):
            raise UgiZeroGuidanceRehearsalError(
                f"unit {entry.unit_id}: selected closure checkpoint hash drifted"
            )
        if entry.productive_generation_calls != 1:
            raise UgiZeroGuidanceRehearsalError(
                f"unit {entry.unit_id}: restartable completion must own one productive call"
            )
        if entry.route_reservation != required:
            raise UgiZeroGuidanceRehearsalError(
                f"unit {entry.unit_id}: route reservation differs from the typed three-role ceiling"
            )
    expected_route = _sum_route_usage(tuple(entry.route_reservation for entry in schedule))
    expected_productive = sum(entry.productive_generation_calls for entry in schedule)
    limits = contract.budget_limits
    if (
        limits.productive_generation_calls != expected_productive
        or limits.terminal_completions != len(schedule)
        or limits.final_candidates != len(schedule)
        or limits.route != expected_route
    ):
        raise UgiZeroGuidanceRehearsalError(
            "matched budget differs from the exact uncensored rehearsal schedule"
        )


def _chemistry_projection(terminal: LockedMatchedTerminal) -> tuple[str, str | None]:
    if not terminal.terminal_valid:
        return "invalid_terminal", None
    if not terminal.exact_l1:
        return "nonexact_l1", None
    if not isinstance(terminal.payload, ValidatedUgiTerminalPayload):
        raise UgiZeroGuidanceRehearsalError(
            "exact-L1 terminal lacks a typed validated chemistry payload"
        )
    decoded = ValidatedUgiTerminalPayload.from_bytes(terminal.terminal_bytes)
    if decoded != terminal.payload:
        raise UgiZeroGuidanceRehearsalError("terminal bytes and typed chemistry payload disagree")
    return "validated_exact_l1", decoded.sha256


def _projection_receipts(
    schedule: Sequence[MatchedScheduleEntry],
    terminals: dict[tuple[MatchedArm, str], LockedMatchedTerminal],
) -> tuple[ChemistryProjectionPairReceipt, ...]:
    receipts = []
    for entry in schedule:
        try:
            guided = terminals[(MatchedArm.GUIDED, entry.unit_id)]
            post_hoc = terminals[(MatchedArm.POST_HOC, entry.unit_id)]
        except KeyError as error:
            raise UgiZeroGuidanceRehearsalError(
                f"unit {entry.unit_id}: one productive arm terminal is missing"
            ) from error
        guided_status, guided_projection = _chemistry_projection(guided)
        post_hoc_status, post_hoc_projection = _chemistry_projection(post_hoc)
        if guided_status != post_hoc_status:
            raise UgiZeroGuidanceRehearsalError(
                f"unit {entry.unit_id}: chemistry projection availability differs across arms"
            )
        receipts.append(
            ChemistryProjectionPairReceipt(
                unit_id=entry.unit_id,
                terminal_bytes_identical=(guided.terminal_bytes == post_hoc.terminal_bytes),
                generation_trace_bytes_identical=(
                    guided.generation_trace_bytes == post_hoc.generation_trace_bytes
                ),
                projection_status=guided_status,
                guided_projection_sha256=guided_projection,
                post_hoc_projection_sha256=post_hoc_projection,
                chemistry_projection_identical=(
                    guided_projection == post_hoc_projection
                    if guided_projection is not None
                    else None
                ),
            )
        )
    return tuple(receipts)


def _diagnostic_from_receipt(
    receipt: UgiTerminalRouteAssessmentReceipt,
) -> DiagnosticRouteAssessment:
    return DiagnosticRouteAssessment(
        value_policy_id=receipt.component_budget_policy_sha256,
        diagnostic_fields=(
            ("assessment_sha256", receipt.assessment_sha256),
            ("product_value_sha256", receipt.product_value_sha256),
            ("route_complete", "true" if receipt.product_value.route_complete else "false"),
        ),
        usage=receipt.realized_route_usage,
    )


def _validate_run_receipts(
    run: MatchedBudgetRun,
    schedule: tuple[MatchedScheduleEntry, ...],
    contract: ZeroGuidanceRehearsalContract,
    diagnostics: dict[tuple[MatchedArm, str], DiagnosticRouteAssessment],
    route_receipts: dict[tuple[MatchedArm, str], UgiTerminalRouteAssessmentReceipt],
) -> None:
    if run.cache_snapshot_sha256 != contract.expected_cache_snapshot_sha256:
        raise UgiZeroGuidanceRehearsalError("runner cache snapshot hash drifted")
    if run.zero_guidance is not True or run.zero_guidance_bitwise_equivalent is not True:
        raise UgiZeroGuidanceRehearsalError("runner did not prove zero-guidance identity")
    if run.shared_censored_unit_ids:
        raise UgiZeroGuidanceRehearsalError("exact rehearsal budget unexpectedly censored units")
    if run.guided_ledger != run.post_hoc_ledger:
        raise UgiZeroGuidanceRehearsalError("matched arm ledgers differ at zero guidance")
    if (
        run.guided_ledger.productive_generation_calls
        != contract.budget_limits.productive_generation_calls
        or run.guided_ledger.terminal_completions != len(schedule)
        or run.guided_ledger.final_candidates != len(schedule)
    ):
        raise UgiZeroGuidanceRehearsalError("realized generation budget drifted")
    for arm, records in (
        (MatchedArm.GUIDED, run.guided_records),
        (MatchedArm.POST_HOC, run.post_hoc_records),
    ):
        if len(records) != len(schedule):
            raise UgiZeroGuidanceRehearsalError("matched runner record count drifted")
        for entry, record in zip(schedule, records, strict=True):
            if record.unit_id != entry.unit_id:
                raise UgiZeroGuidanceRehearsalError("matched runner record order drifted")
            key = (arm, entry.unit_id)
            receipt = route_receipts.get(key)
            diagnostic = diagnostics.get(key)
            if record.route_assessed != (receipt is not None):
                raise UgiZeroGuidanceRehearsalError(
                    f"unit {entry.unit_id}: route receipt admission disagrees with runner"
                )
            if receipt is not None and (
                diagnostic is None
                or record.diagnostic_fields != diagnostic.diagnostic_fields
                or record.value_policy_id != diagnostic.value_policy_id
                or record.route_usage != receipt.realized_route_usage
            ):
                raise UgiZeroGuidanceRehearsalError(
                    f"unit {entry.unit_id}: runner route receipt was altered"
                )
    guided = {
        item.unit_id: item for item in route_receipts.values() if item.arm is MatchedArm.GUIDED
    }
    post_hoc = {
        item.unit_id: item for item in route_receipts.values() if item.arm is MatchedArm.POST_HOC
    }
    if set(guided) != set(post_hoc):
        raise UgiZeroGuidanceRehearsalError("route-assessed unit sets differ across arms")
    for unit_id in guided:
        if guided[unit_id].product_value_sha256 != post_hoc[unit_id].product_value_sha256:
            raise UgiZeroGuidanceRehearsalError(
                f"unit {unit_id}: structured route values differ across zero-guidance arms"
            )


def run_zero_guidance_route_rehearsal(
    schedule: Sequence[MatchedScheduleEntry],
    *,
    contract: ZeroGuidanceRehearsalContract,
    generator: RestartableGeneratorClosureAdapter,
    planner_context: PlannerCacheContext,
    base_cache: FilePlannerCache,
    guided_overlay: FilePlannerCache,
    post_hoc_overlay: FilePlannerCache,
    l1_reverifier: QualifiedUgiL1Reverifier,
    planner_factory: QualifiedRoutePlannerFactoryAdapter,
) -> ZeroGuidanceRehearsalResult:
    """Exercise the complete matched route seam at guidance strength zero only."""

    if not isinstance(contract, ZeroGuidanceRehearsalContract):
        raise UgiZeroGuidanceRehearsalError("rehearsal contract is malformed")
    if not isinstance(generator, RestartableGeneratorClosureAdapter):
        raise UgiZeroGuidanceRehearsalError("restartable generator adapter is malformed")
    if generator.identity != contract.generator_identity:
        raise UgiZeroGuidanceRehearsalError("generator/closure adapter identity drifted")
    if not isinstance(planner_context, PlannerCacheContext):
        raise UgiZeroGuidanceRehearsalError("planner context is malformed")
    if planner_cache_context_sha256(planner_context) != contract.planner_context_sha256:
        raise UgiZeroGuidanceRehearsalError("planner context hash drifted")
    if current_terminal_route_assessment_source_sha256() != (
        contract.terminal_route_assessment_source_sha256
    ):
        raise UgiZeroGuidanceRehearsalError("terminal-route assessment source hash drifted")
    if _module_sha256(matched_runner, label="matched runner") != (
        contract.matched_runner_source_sha256
    ):
        raise UgiZeroGuidanceRehearsalError("hash-pinned matched runner source drifted")
    if not isinstance(l1_reverifier, QualifiedUgiL1Reverifier):
        raise UgiZeroGuidanceRehearsalError("L1 reverifier is malformed")
    if not isinstance(planner_factory, QualifiedRoutePlannerFactoryAdapter):
        raise UgiZeroGuidanceRehearsalError("qualified planner factory is malformed")
    if planner_factory.qualification_sha256 != contract.route_qualification_sha256:
        raise UgiZeroGuidanceRehearsalError("route/source qualification hash drifted")

    frozen_schedule = tuple(schedule)
    _validate_exact_rehearsal_budget(
        frozen_schedule,
        contract=contract,
        planner_context=planner_context,
    )
    binding = preflight_lazy_matched_planner_cache_binding(
        base_cache,
        guided_overlay,
        post_hoc_overlay,
        planner_context,
        assessment_at_utc=contract.assessment_at_utc,
    )
    preflight = ZeroGuidanceRehearsalPreflight(
        contract=contract,
        schedule_manifest=_schedule_manifest(frozen_schedule),
        cache_binding=binding.preflight,
    )

    generated: dict[tuple[MatchedArm, str], LockedMatchedTerminal] = {}
    route_receipts: dict[tuple[MatchedArm, str], UgiTerminalRouteAssessmentReceipt] = {}
    diagnostics: dict[tuple[MatchedArm, str], DiagnosticRouteAssessment] = {}

    def generate(request: MatchedGenerationRequest) -> LockedMatchedTerminal:
        if generator.identity != contract.generator_identity:
            raise UgiZeroGuidanceRehearsalError(
                "generator/closure adapter identity changed during execution"
            )
        key = (request.arm, request.entry.unit_id)
        if key in generated:
            raise UgiZeroGuidanceRehearsalError("duplicate productive generator callback")
        terminal = generator.generate_locked_terminal(request)
        if not isinstance(terminal, LockedMatchedTerminal):
            raise UgiZeroGuidanceRehearsalError(
                "restartable generator returned an untyped terminal"
            )
        generated[key] = terminal
        return terminal

    def assess(
        terminal: LockedMatchedTerminal,
        assessment_context: MatchedAssessmentContext,
    ) -> DiagnosticRouteAssessment:
        key = (assessment_context.arm, terminal.unit_id)
        if key in route_receipts:
            raise UgiZeroGuidanceRehearsalError("duplicate terminal-route callback")
        cache = binding.bind(assessment_context)
        if planner_factory.qualification_sha256 != contract.route_qualification_sha256:
            raise UgiZeroGuidanceRehearsalError(
                "route/source qualification changed during execution"
            )
        planner = planner_factory.build_planner(
            cache,
            planner_context,
            assessment_context,
            terminal,
        )
        if not callable(getattr(planner, "assess", None)):
            raise UgiZeroGuidanceRehearsalError(
                "planner factory returned an unsupported route planner"
            )
        receipt = assess_locked_ugi_terminal_routes(
            terminal,
            l1_reverifier=l1_reverifier,
            planner=planner,
            planner_context=planner_context,
            assessment_context=assessment_context,
            assessment_at_utc=contract.assessment_at_utc,
        )
        diagnostic = _diagnostic_from_receipt(receipt)
        route_receipts[key] = receipt
        diagnostics[key] = diagnostic
        return diagnostic

    run = matched_runner.run_diagnostic_matched_budget_arms(
        frozen_schedule,
        budget_limits=contract.budget_limits,
        base_seed=contract.base_seed,
        zero_guidance=True,
        cache_snapshot_sha256=contract.expected_cache_snapshot_sha256,
        generate_terminal=generate,
        assess_terminal=assess,
    )
    projections = _projection_receipts(frozen_schedule, generated)
    _validate_run_receipts(
        run,
        frozen_schedule,
        contract,
        diagnostics,
        route_receipts,
    )
    cache_audit = binding.finalize()
    if cache_audit.guided_after != cache_audit.post_hoc_after:
        raise UgiZeroGuidanceRehearsalError(
            "matched cache overlays contain different zero-guidance assessments"
        )

    guided_receipts = tuple(
        route_receipts[(MatchedArm.GUIDED, entry.unit_id)]
        for entry in frozen_schedule
        if (MatchedArm.GUIDED, entry.unit_id) in route_receipts
    )
    post_hoc_receipts = tuple(
        route_receipts[(MatchedArm.POST_HOC, entry.unit_id)]
        for entry in frozen_schedule
        if (MatchedArm.POST_HOC, entry.unit_id) in route_receipts
    )
    return ZeroGuidanceRehearsalResult(
        preflight=preflight,
        matched_run=run,
        chemistry_projections=projections,
        guided_route_receipts=guided_receipts,
        post_hoc_route_receipts=post_hoc_receipts,
        cache_audit=cache_audit,
    )
