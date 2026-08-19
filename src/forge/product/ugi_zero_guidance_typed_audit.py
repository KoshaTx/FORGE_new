"""Typed outcome accounting for a matched Ugi guidance qualification.

The matched runner deliberately represents invalid terminals, nonexact L1
products, duplicates and planner censoring differently.  This module preserves
those distinctions when a zero-guidance run is summarized.  It does not run a
generator, invoke a planner, authorize nonzero guidance or select a prospective
candidate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload

RUN_SCHEMA_VERSION = "forge.ugi_zero_guidance_qualification_run.v1"
SUPPORT_AUDIT_SCHEMA_VERSION = "phase1_ugi_production_zero_guidance_support_audits.v1"


class UgiZeroGuidanceTypedAuditError(RuntimeError):
    """Raised when zero-guidance outcome classes do not form an exact partition."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UgiZeroGuidanceTypedAuditError(f"{label} must be a mapping")
    return value


def _require_sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise UgiZeroGuidanceTypedAuditError(f"{label} must be a sequence")
    return value


def build_zero_guidance_run_artifact(run: Any) -> dict[str, Any]:
    """Serialize one typed development run with qualification-only scope.

    ``run`` is intentionally duck-typed to avoid duplicating the matched
    runner's dataclass.  The caller must supply the complete dataclass payload
    through ``dataclasses.asdict`` before this function is invoked.
    """

    payload = _require_mapping(run, label="development run")
    if payload.get("guidance_strength") != 0.0:
        raise UgiZeroGuidanceTypedAuditError("qualification requires lambda zero")
    for name in ("production_execution", "biological_guidance", "private_holdout_accessed"):
        if payload.get(name) is not False:
            raise UgiZeroGuidanceTypedAuditError(f"qualification requires {name}=false")
    content = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "zero_guidance_qualification_complete_not_production_authorized",
        "run": dict(payload),
        "scope": {
            "qualification_only": True,
            "production_execution": False,
            "nonzero_guidance_executed": False,
            "candidate_selection": False,
            "prospective_candidate_lock": False,
            "biological_guidance": False,
            "sealed_holdout_access": False,
            "deterministic_qualification_subset_materialized": True,
            "synthesis_success_probability": None,
        },
    }
    return {**content, "result_sha256": _sha256_payload(content)}


def build_support_audit_artifact(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Persist every invoked planner assessment in deterministic order."""

    canonical: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(records):
        record = dict(_require_mapping(raw, label=f"support record {ordinal}"))
        if record.get("ordinal") != ordinal:
            raise UgiZeroGuidanceTypedAuditError(
                "support records must use contiguous execution-order ordinals"
            )
        if record.get("arm") not in {"guided", "post_hoc"}:
            raise UgiZeroGuidanceTypedAuditError("support record arm is unsupported")
        if record.get("assessment_phase") not in {
            "checkpoint_shadow",
            "productive_final",
        }:
            raise UgiZeroGuidanceTypedAuditError("support record assessment phase is unsupported")
        potential = _require_mapping(
            record.get("potential"), label=f"support record {ordinal} potential"
        )
        utility = potential.get("route_completion_utility")
        if utility not in {None, 0.0, 1.0}:
            raise UgiZeroGuidanceTypedAuditError(
                "support record route utility must be binary or censored"
            )
        if potential.get("true_planner_censor") != (utility is None):
            raise UgiZeroGuidanceTypedAuditError(
                "support record censor flag disagrees with its route utility"
            )
        if record.get("scalar_value") is not None:
            raise UgiZeroGuidanceTypedAuditError(
                "zero-guidance support records cannot define a scalar success value"
            )
        if record.get("success_probability") is not None:
            raise UgiZeroGuidanceTypedAuditError(
                "route completion cannot be serialized as synthesis-success probability"
            )
        canonical.append(record)
    manifest = _sha256_payload(canonical)
    content = {
        "schema_version": SUPPORT_AUDIT_SCHEMA_VERSION,
        "record_count": len(canonical),
        "records": canonical,
        "records_manifest_sha256": manifest,
        "scalar_value": None,
        "success_probability": None,
    }
    return {**content, "result_sha256": _sha256_payload(content)}


def _checkpoint_counts(arm: Mapping[str, Any]) -> dict[str, int]:
    groups = _require_sequence(arm.get("checkpoint_groups"), label="checkpoint groups")
    attempts = sum(
        int(_require_mapping(item, label="checkpoint group").get("terminal_completions"))
        for item in groups
    )
    invalid = sum(int(item.get("invalid_terminal_count")) for item in groups)
    nonexact = sum(int(item.get("nonexact_l1_count")) for item in groups)
    censored = sum(int(item.get("censored_assessment_count")) for item in groups)
    utilities = [
        utility
        for raw in groups
        for utility in _require_sequence(
            _require_mapping(raw, label="checkpoint group").get("route_completion_utilities"),
            label="checkpoint utilities",
        )
    ]
    complete = sum(value == 1.0 for value in utilities)
    incomplete = sum(value == 0.0 for value in utilities)
    null_count = sum(value is None for value in utilities)
    if null_count != invalid + nonexact + censored:
        raise UgiZeroGuidanceTypedAuditError(
            "checkpoint null utilities do not match invalid, nonexact and censored rows"
        )
    counts = {
        "attempts": attempts,
        "completion_error": 0,
        "invalid_terminal": invalid,
        "nonexact_l1": nonexact,
        "budget_exhausted_before_planner": 0,
        "route_complete": complete,
        "route_incomplete": incomplete,
        "true_planner_censor": censored,
        "duplicate_not_assessed": 0,
    }
    if attempts != sum(value for key, value in counts.items() if key != "attempts"):
        raise UgiZeroGuidanceTypedAuditError(
            "checkpoint outcome classes do not partition completion attempts"
        )
    return counts


def _productive_counts(arm: Mapping[str, Any]) -> dict[str, int]:
    admissions = _require_sequence(arm.get("productive_admissions"), label="productive admissions")
    assessments = _require_sequence(
        arm.get("productive_assessments"), label="productive assessments"
    )
    if len(admissions) != len(assessments):
        raise UgiZeroGuidanceTypedAuditError("productive admissions and assessments are misaligned")
    admission_counts = Counter(
        _require_mapping(item, label="productive admission").get("disposition")
        for item in admissions
    )
    allowed_admissions = {
        "invalid_terminal",
        "nonexact_l1",
        "canonical_representative",
        "duplicate_not_assessed",
    }
    if set(admission_counts) - allowed_admissions:
        raise UgiZeroGuidanceTypedAuditError("productive admission disposition changed")
    representative_assessments = [
        _require_mapping(item, label="productive assessment")
        for item in assessments
        if _require_mapping(item, label="productive assessment").get("disposition")
        == "assessed_representative"
    ]
    complete = sum(
        item.get("route_completion_utility") == 1.0 for item in representative_assessments
    )
    incomplete = sum(
        item.get("route_completion_utility") == 0.0 for item in representative_assessments
    )
    censored = sum(
        item.get("route_completion_utility") is None for item in representative_assessments
    )
    if censored != sum(bool(item.get("censored")) for item in representative_assessments):
        raise UgiZeroGuidanceTypedAuditError(
            "productive representative censor flags disagree with utilities"
        )
    representatives = admission_counts["canonical_representative"]
    if representatives != complete + incomplete + censored:
        raise UgiZeroGuidanceTypedAuditError(
            "productive representatives do not each have one typed route assessment"
        )
    counts = {
        "attempts": len(admissions),
        "completion_error": 0,
        "invalid_terminal": admission_counts["invalid_terminal"],
        "nonexact_l1": admission_counts["nonexact_l1"],
        "budget_exhausted_before_planner": 0,
        "canonical_representatives": representatives,
        "duplicate_not_assessed": admission_counts["duplicate_not_assessed"],
        "route_complete": complete,
        "route_incomplete": incomplete,
        "true_planner_censor": censored,
    }
    partition = (
        counts["completion_error"]
        + counts["invalid_terminal"]
        + counts["nonexact_l1"]
        + counts["budget_exhausted_before_planner"]
        + counts["duplicate_not_assessed"]
        + counts["route_complete"]
        + counts["route_incomplete"]
        + counts["true_planner_censor"]
    )
    if counts["attempts"] != partition:
        raise UgiZeroGuidanceTypedAuditError(
            "productive outcome classes do not partition completion attempts"
        )
    return counts


def derive_typed_counts(
    run_artifact: Mapping[str, Any],
    support_audit_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive and cross-check typed counts from the run and support ledger."""

    run = _require_mapping(run_artifact.get("run"), label="run payload")
    records = _require_sequence(support_audit_artifact.get("records"), label="support records")
    by_arm_phase = Counter(
        (
            _require_mapping(item, label="support record").get("arm"),
            _require_mapping(item, label="support record").get("assessment_phase"),
        )
        for item in records
    )
    result: dict[str, Any] = {}
    expected_record_count = 0
    for arm_name in ("guided", "post_hoc"):
        arm = _require_mapping(run.get(arm_name), label=f"{arm_name} run")
        checkpoint = _checkpoint_counts(arm)
        productive = _productive_counts(arm)
        checkpoint_assessments = (
            checkpoint["route_complete"]
            + checkpoint["route_incomplete"]
            + checkpoint["true_planner_censor"]
        )
        productive_assessments = (
            productive["route_complete"]
            + productive["route_incomplete"]
            + productive["true_planner_censor"]
        )
        if by_arm_phase[(arm_name, "checkpoint_shadow")] != checkpoint_assessments:
            raise UgiZeroGuidanceTypedAuditError(
                f"{arm_name} checkpoint support records do not match route assessments"
            )
        if by_arm_phase[(arm_name, "productive_final")] != productive_assessments:
            raise UgiZeroGuidanceTypedAuditError(
                f"{arm_name} productive support records do not match route assessments"
            )
        expected_record_count += checkpoint_assessments + productive_assessments
        all_attempts = {
            "attempts": checkpoint["attempts"] + productive["attempts"],
            "completion_error": 0,
            "invalid_terminal": (checkpoint["invalid_terminal"] + productive["invalid_terminal"]),
            "nonexact_l1": checkpoint["nonexact_l1"] + productive["nonexact_l1"],
            "budget_exhausted_before_planner": 0,
            "route_complete": checkpoint["route_complete"] + productive["route_complete"],
            "route_incomplete": (checkpoint["route_incomplete"] + productive["route_incomplete"]),
            "true_planner_censor": (
                checkpoint["true_planner_censor"] + productive["true_planner_censor"]
            ),
            "duplicate_not_assessed": productive["duplicate_not_assessed"],
        }
        if all_attempts["attempts"] != sum(
            value for key, value in all_attempts.items() if key != "attempts"
        ):
            raise UgiZeroGuidanceTypedAuditError(
                f"{arm_name} all-attempt outcome classes do not partition attempts"
            )
        result[arm_name] = {
            "checkpoint_shadow": checkpoint,
            "productive_final": productive,
            "all_attempts": all_attempts,
        }
    if len(records) != expected_record_count:
        raise UgiZeroGuidanceTypedAuditError(
            "support ledger size differs from all invoked route assessments"
        )
    censor_reasons = Counter()
    for item in records:
        record = _require_mapping(item, label="support record")
        potential = _require_mapping(record.get("potential"), label="support potential")
        if not potential.get("true_planner_censor"):
            continue
        for reason in _require_sequence(potential.get("reasons"), label="censor reasons"):
            censor_reasons[str(reason)] += 1
    return {
        "by_arm": result,
        "support_audit_record_count": len(records),
        "planner_censor_reasons": {
            "budget_exhausted_after_planner_invocation": censor_reasons["budget_exhausted"],
            "invalid_input": censor_reasons["invalid_input"],
            "execution_error": censor_reasons["execution_error"],
        },
    }


__all__ = [
    "RUN_SCHEMA_VERSION",
    "SUPPORT_AUDIT_SCHEMA_VERSION",
    "UgiZeroGuidanceTypedAuditError",
    "build_support_audit_artifact",
    "build_zero_guidance_run_artifact",
    "derive_typed_counts",
]
