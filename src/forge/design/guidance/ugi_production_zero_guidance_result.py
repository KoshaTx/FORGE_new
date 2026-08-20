"""Byte-owned production provenance for a zero-guidance Ugi rehearsal.

The zero-guidance composer intentionally owns route receipts rather than the
terminal-aware planner's support audit.  This additive wrapper retains both:
the complete canonical composer result and every per-arm
``ProductionTerminalSupportAudit`` captured while planners are constructed.
It performs no generation, routing, scalarization, biology or selection.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.core.io import stable_json as _stable_json
from forge.design.ugi_matched_budget_orchestration import (
    LockedMatchedTerminal,
    MatchedArm,
    MatchedAssessmentContext,
)
from forge.design.ugi_production_terminal_route_evaluator import (
    ProductionQualifiedRoutePlanner,
    ProductionTerminalSupportAudit,
)
from forge.design.ugi_zero_guidance_rehearsal import (
    ZERO_GUIDANCE_REHEARSAL_SCHEMA_VERSION,
    ZeroGuidanceRehearsalResult,
)
from forge.route.engine.planner import RoutePlanner
from forge.route.engine.planner_cache import PlannerCacheContext
from forge.route.engine.planner_cache_snapshot import OverlayFilePlannerCache

PRODUCTION_ZERO_GUIDANCE_EXECUTION_RESULT_SCHEMA_VERSION = (
    "forge.production_zero_guidance_execution_result.v1"
)
_EXPECTED_COMPOSER_SCOPE = {
    "guidance_strength": 0,
    "synthesis_scalar_defined": False,
    "nonzero_synthesis_guidance": False,
    "biological_guidance": False,
    "candidate_selection": False,
    "private_holdout_accessed": False,
}
_EXPECTED_WRAPPER_SCOPE = {
    **_EXPECTED_COMPOSER_SCOPE,
    "production_generator_executed": True,
    "production_route_source_executed": True,
    "support_audits_retained": True,
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class UgiProductionZeroGuidanceResultError(RuntimeError):
    """Raised when production rehearsal provenance is incomplete or divergent."""


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise UgiProductionZeroGuidanceResultError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise UgiProductionZeroGuidanceResultError(
            f"{label} must contain 64 lowercase hexadecimal characters"
        )
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise UgiProductionZeroGuidanceResultError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


@dataclass(frozen=True)
class AuthenticatedZeroGuidanceComposerResult:
    """Canonical full composer result authenticated by its embedded checksum."""

    value: dict[str, Any]
    canonical_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.value, dict) or not isinstance(self.canonical_bytes, bytes):
            raise UgiProductionZeroGuidanceResultError("composer result is malformed")
        if (
            not self.canonical_bytes
            or self.canonical_bytes != (_stable_json(self.value) + "\n").encode()
        ):
            raise UgiProductionZeroGuidanceResultError(
                "composer result bytes are not canonical JSON"
            )
        expected_fields = {
            "schema_version",
            "status",
            "preflight",
            "matched_run",
            "chemistry_projections",
            "chemistry_projection_identity_proven",
            "guided_route_receipts",
            "post_hoc_route_receipts",
            "cache_audit",
            "scope",
            "result_sha256",
        }
        if set(self.value) != expected_fields:
            raise UgiProductionZeroGuidanceResultError(
                "composer result has an unsupported top-level schema"
            )
        if (
            self.value.get("schema_version") != ZERO_GUIDANCE_REHEARSAL_SCHEMA_VERSION
            or self.value.get("status") != "zero_guidance_rehearsal_complete"
            or self.value.get("chemistry_projection_identity_proven") is not True
            or self.value.get("scope") != _EXPECTED_COMPOSER_SCOPE
        ):
            raise UgiProductionZeroGuidanceResultError(
                "composer result is not a completed zero-guidance rehearsal"
            )
        observed_sha256 = _require_sha256(
            self.value.get("result_sha256"), label="composer result_sha256"
        )
        content = {key: value for key, value in self.value.items() if key != "result_sha256"}
        if observed_sha256 != _sha256_payload(content):
            raise UgiProductionZeroGuidanceResultError("composer result checksum mismatch")
        matched = self.value.get("matched_run")
        if not isinstance(matched, dict):
            raise UgiProductionZeroGuidanceResultError("composer matched run is malformed")
        guided = matched.get("guided_records")
        post_hoc = matched.get("post_hoc_records")
        if (
            matched.get("zero_guidance") is not True
            or matched.get("zero_guidance_bitwise_equivalent") is not True
            or matched.get("shared_censored_unit_ids") != []
            or not isinstance(guided, list)
            or not isinstance(post_hoc, list)
            or not guided
            or len(guided) != len(post_hoc)
        ):
            raise UgiProductionZeroGuidanceResultError(
                "composer matched run is censored, nonidentical or incomplete"
            )
        if (
            not isinstance(self.value.get("guided_route_receipts"), list)
            or not isinstance(self.value.get("post_hoc_route_receipts"), list)
            or not self.value["guided_route_receipts"]
            or len(self.value["guided_route_receipts"])
            != len(self.value["post_hoc_route_receipts"])
        ):
            raise UgiProductionZeroGuidanceResultError(
                "composer route receipt pairs are incomplete"
            )
        guided_units = [row.get("unit_id") if isinstance(row, dict) else None for row in guided]
        post_hoc_units = [row.get("unit_id") if isinstance(row, dict) else None for row in post_hoc]
        if (
            any(not isinstance(unit_id, str) or not unit_id for unit_id in guided_units)
            or guided_units != post_hoc_units
            or len(guided_units) != len(set(guided_units))
        ):
            raise UgiProductionZeroGuidanceResultError(
                "composer arm records are not paired in one unique schedule order"
            )

    @classmethod
    def from_canonical_bytes(
        cls,
        value: bytes,
    ) -> AuthenticatedZeroGuidanceComposerResult:
        """Authenticate one byte-owned composer result without discarding fields."""

        if not isinstance(value, bytes) or not value:
            raise UgiProductionZeroGuidanceResultError(
                "composer canonical result must be nonempty bytes"
            )
        try:
            decoded = json.loads(value, object_pairs_hook=_reject_duplicate_pairs)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UgiProductionZeroGuidanceResultError(
                "composer result bytes are not valid JSON"
            ) from error
        if not isinstance(decoded, dict):
            raise UgiProductionZeroGuidanceResultError(
                "composer result bytes must encode an object"
            )
        return cls(value=decoded, canonical_bytes=value)

    @classmethod
    def from_result(
        cls,
        value: ZeroGuidanceRehearsalResult,
    ) -> AuthenticatedZeroGuidanceComposerResult:
        """Own and authenticate every byte of a typed composer result."""

        if not isinstance(value, ZeroGuidanceRehearsalResult):
            raise UgiProductionZeroGuidanceResultError(
                "composer result must be a ZeroGuidanceRehearsalResult"
            )
        return cls.from_canonical_bytes(value.canonical_bytes)

    @property
    def result_sha256(self) -> str:
        return self.value["result_sha256"]

    @property
    def matched_run(self) -> dict[str, Any]:
        return self.value["matched_run"]


@dataclass(frozen=True)
class ArmTerminalSupportAuditSnapshot:
    """One arm/unit coordinate and its complete typed support audit."""

    arm: MatchedArm
    unit_id: str
    audit: ProductionTerminalSupportAudit

    def __post_init__(self) -> None:
        if not isinstance(self.arm, MatchedArm):
            raise UgiProductionZeroGuidanceResultError("support audit arm is malformed")
        _require_nonempty(self.unit_id, label="support audit unit_id")
        if not isinstance(self.audit, ProductionTerminalSupportAudit):
            raise UgiProductionZeroGuidanceResultError(
                "support audit snapshot must retain a ProductionTerminalSupportAudit"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "unit_id": self.unit_id,
            "audit": self.audit.to_dict(),
        }


PlannerBuilder = Callable[
    [
        OverlayFilePlannerCache,
        PlannerCacheContext,
        MatchedAssessmentContext,
        LockedMatchedTerminal,
    ],
    RoutePlanner,
]


class ProductionTerminalSupportAuditCollector:
    """Transparent planner-factory wrapper retaining support audits once."""

    def __init__(self, delegate: PlannerBuilder):
        if not callable(delegate):
            raise UgiProductionZeroGuidanceResultError("planner builder must be callable")
        self._delegate = delegate
        self._snapshots: dict[tuple[MatchedArm, str], ArmTerminalSupportAuditSnapshot] = {}

    def build_planner(
        self,
        cache: OverlayFilePlannerCache,
        planner_context: PlannerCacheContext,
        assessment_context: MatchedAssessmentContext,
        terminal: LockedMatchedTerminal,
    ) -> RoutePlanner:
        """Delegate planner construction and retain its immutable support audit."""

        if not isinstance(assessment_context, MatchedAssessmentContext):
            raise UgiProductionZeroGuidanceResultError(
                "collector requires a MatchedAssessmentContext"
            )
        if not isinstance(terminal, LockedMatchedTerminal):
            raise UgiProductionZeroGuidanceResultError("collector requires a LockedMatchedTerminal")
        planner = self._delegate(cache, planner_context, assessment_context, terminal)
        if not isinstance(planner, ProductionQualifiedRoutePlanner):
            raise UgiProductionZeroGuidanceResultError(
                "production planner builder did not return a qualified route planner"
            )
        key = (assessment_context.arm, terminal.unit_id)
        if key in self._snapshots:
            raise UgiProductionZeroGuidanceResultError(
                "duplicate production support audit for one arm/unit"
            )
        snapshot = ArmTerminalSupportAuditSnapshot(
            arm=assessment_context.arm,
            unit_id=terminal.unit_id,
            audit=planner.support_audit,
        )
        if snapshot.audit.cache_clone_id != assessment_context.cache_clone_id:
            raise UgiProductionZeroGuidanceResultError(
                "support audit clone differs from the matched assessment context"
            )
        if snapshot.audit.terminal_sha256 != terminal.terminal_sha256:
            raise UgiProductionZeroGuidanceResultError(
                "support audit differs from the planner's locked terminal"
            )
        self._snapshots[key] = snapshot
        return planner

    @property
    def snapshots(self) -> tuple[ArmTerminalSupportAuditSnapshot, ...]:
        """Return insertion-ordered immutable snapshots after composer execution."""

        return tuple(self._snapshots.values())


def _audit_pair_payload(value: ProductionTerminalSupportAudit) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("cache_clone_id")
    return payload


@dataclass(frozen=True)
class ProductionZeroGuidanceExecutionResult:
    """Complete composer result plus paired full support-audit provenance."""

    composer_result: AuthenticatedZeroGuidanceComposerResult
    support_audits: tuple[ArmTerminalSupportAuditSnapshot, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.composer_result, AuthenticatedZeroGuidanceComposerResult):
            raise UgiProductionZeroGuidanceResultError(
                "execution result lacks an authenticated composer result"
            )
        if not isinstance(self.support_audits, tuple) or any(
            not isinstance(item, ArmTerminalSupportAuditSnapshot) for item in self.support_audits
        ):
            raise UgiProductionZeroGuidanceResultError(
                "execution support audits must be a tuple of typed snapshots"
            )
        matched = self.composer_result.matched_run
        records_by_key: dict[tuple[MatchedArm, str], dict[str, Any]] = {}
        expected_order: list[tuple[MatchedArm, str]] = []
        for arm, field in (
            (MatchedArm.GUIDED, "guided_records"),
            (MatchedArm.POST_HOC, "post_hoc_records"),
        ):
            for row in matched[field]:
                if not isinstance(row, dict):
                    raise UgiProductionZeroGuidanceResultError("composer arm record is malformed")
                unit_id = row.get("unit_id")
                key = (arm, unit_id)
                if key in records_by_key:
                    raise UgiProductionZeroGuidanceResultError(
                        "composer contains a duplicate arm/unit record"
                    )
                records_by_key[key] = row
                if row.get("route_assessed") is True:
                    expected_order.append(key)
        observed_order = [(item.arm, item.unit_id) for item in self.support_audits]
        if observed_order != expected_order:
            raise UgiProductionZeroGuidanceResultError(
                "support audit snapshots do not exactly cover assessed records in arm order"
            )
        snapshots = {(item.arm, item.unit_id): item for item in self.support_audits}
        guided_units = [unit_id for arm, unit_id in expected_order if arm is MatchedArm.GUIDED]
        post_hoc_units = [unit_id for arm, unit_id in expected_order if arm is MatchedArm.POST_HOC]
        if not guided_units or guided_units != post_hoc_units:
            raise UgiProductionZeroGuidanceResultError(
                "support audits are not paired across the zero-guidance arms"
            )
        guided_clone = matched.get("guided_cache_clone_id")
        post_hoc_clone = matched.get("post_hoc_cache_clone_id")
        if (
            not isinstance(guided_clone, str)
            or not isinstance(post_hoc_clone, str)
            or not guided_clone
            or not post_hoc_clone
            or guided_clone == post_hoc_clone
        ):
            raise UgiProductionZeroGuidanceResultError(
                "composer cache clone identities are missing or nonisolated"
            )
        for unit_id in guided_units:
            guided = snapshots[(MatchedArm.GUIDED, unit_id)]
            post_hoc = snapshots[(MatchedArm.POST_HOC, unit_id)]
            guided_record = records_by_key[(MatchedArm.GUIDED, unit_id)]
            post_hoc_record = records_by_key[(MatchedArm.POST_HOC, unit_id)]
            if (
                guided.audit.cache_clone_id != guided_clone
                or post_hoc.audit.cache_clone_id != post_hoc_clone
                or guided_record.get("cache_clone_id") != guided.audit.cache_clone_id
                or post_hoc_record.get("cache_clone_id") != post_hoc.audit.cache_clone_id
                or guided_record.get("terminal_sha256") != guided.audit.terminal_sha256
                or post_hoc_record.get("terminal_sha256") != post_hoc.audit.terminal_sha256
            ):
                raise UgiProductionZeroGuidanceResultError(
                    f"unit {unit_id}: support audit does not bind to the composer record"
                )
            if _audit_pair_payload(guided.audit) != _audit_pair_payload(post_hoc.audit):
                raise UgiProductionZeroGuidanceResultError(
                    f"unit {unit_id}: paired support audits differ beyond cache clone identity"
                )

    @property
    def paired_support_count(self) -> int:
        return sum(item.arm is MatchedArm.GUIDED for item in self.support_audits)

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PRODUCTION_ZERO_GUIDANCE_EXECUTION_RESULT_SCHEMA_VERSION,
            "status": "production_zero_guidance_execution_complete",
            "composer_result_sha256": self.composer_result.result_sha256,
            "composer_result": self.composer_result.value,
            "support_audits": [item.to_dict() for item in self.support_audits],
            "paired_support_count": self.paired_support_count,
            "paired_support_identity_except_cache_clone_proven": True,
            "scope": dict(_EXPECTED_WRAPPER_SCOPE),
        }

    @property
    def result_sha256(self) -> str:
        return _sha256_payload(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "result_sha256": self.result_sha256}

    @property
    def canonical_bytes(self) -> bytes:
        return (_stable_json(self.to_dict()) + "\n").encode()


def build_production_zero_guidance_execution_result(
    composer_result: ZeroGuidanceRehearsalResult,
    support_audits: Sequence[ArmTerminalSupportAuditSnapshot],
) -> ProductionZeroGuidanceExecutionResult:
    """Seal one typed composer result with its complete per-arm support audits."""

    authenticated = AuthenticatedZeroGuidanceComposerResult.from_result(composer_result)
    snapshots = tuple(support_audits)
    matched = authenticated.matched_run
    expected_order = [
        (arm, row["unit_id"])
        for arm, field in (
            (MatchedArm.GUIDED, "guided_records"),
            (MatchedArm.POST_HOC, "post_hoc_records"),
        )
        for row in matched[field]
        if row.get("route_assessed") is True
    ]
    by_key = {(item.arm, item.unit_id): item for item in snapshots}
    if len(by_key) != len(snapshots):
        raise UgiProductionZeroGuidanceResultError(
            "support audit snapshots contain duplicate arm/unit coordinates"
        )
    try:
        ordered = tuple(by_key[key] for key in expected_order)
    except KeyError as error:
        raise UgiProductionZeroGuidanceResultError(
            "support audit snapshots do not cover every assessed composer record"
        ) from error
    if set(by_key) != set(expected_order):
        raise UgiProductionZeroGuidanceResultError(
            "support audit snapshots contain an unassessed composer coordinate"
        )
    return ProductionZeroGuidanceExecutionResult(
        composer_result=authenticated,
        support_audits=ordered,
    )


__all__ = [
    "ArmTerminalSupportAuditSnapshot",
    "AuthenticatedZeroGuidanceComposerResult",
    "PRODUCTION_ZERO_GUIDANCE_EXECUTION_RESULT_SCHEMA_VERSION",
    "ProductionTerminalSupportAuditCollector",
    "ProductionZeroGuidanceExecutionResult",
    "UgiProductionZeroGuidanceResultError",
    "build_production_zero_guidance_execution_result",
]
