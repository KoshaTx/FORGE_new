"""Requalify guidance prerequisites against source-qualified fresh-pool v6."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi_source_guidance_prerequisite_requalification_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_source_guidance_prerequisite_requalification.v1"


class UgiSourceGuidancePrerequisiteRequalificationError(ValueError):
    """Raised when the immutable guidance prerequisites cannot be requalified."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise UgiSourceGuidancePrerequisiteRequalificationError(
            f"invalid {label}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise UgiSourceGuidancePrerequisiteRequalificationError(f"{label} must be an object")
    return value


def _pin(repo: Path, record: Mapping[str, Any], *, label: str) -> Path:
    if set(record) != {"path", "sha256"}:
        raise UgiSourceGuidancePrerequisiteRequalificationError(f"{label} pin is malformed")
    path = (repo / str(record.get("path"))).resolve()
    try:
        relative = path.relative_to(repo.resolve())
    except ValueError as error:
        raise UgiSourceGuidancePrerequisiteRequalificationError(
            f"{label} escapes repository"
        ) from error
    lowered = "/".join(relative.parts).lower()
    if "holdout" in lowered or "sealed" in lowered:
        raise UgiSourceGuidancePrerequisiteRequalificationError(f"{label} is forbidden")
    if sha256_file(path) != record.get("sha256"):
        raise UgiSourceGuidancePrerequisiteRequalificationError(f"{label} hash changed")
    return path


def _utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise UgiSourceGuidancePrerequisiteRequalificationError(
            f"{label} must be an ISO-8601 UTC timestamp"
        )
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise UgiSourceGuidancePrerequisiteRequalificationError(f"invalid {label}") from error


def build_source_guidance_prerequisite_requalification(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    """Build a development-only, fail-closed prerequisite receipt."""

    repo = repo.resolve()
    config = _load(config_path, label="guidance-prerequisite config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiSourceGuidancePrerequisiteRequalificationError(
            "unsupported guidance-prerequisite config"
        )
    expected_policy = {
        "purpose": (
            "Requalify immutable zero-guidance and matched-budget prerequisites "
            "against source-qualified fresh-pool v6."
        ),
        "development_only": True,
        "historical_artifacts_rewritten": False,
        "production_runtime_replayed": False,
        "scalarization_authorized": False,
        "nonzero_guidance_authorized": False,
        "candidate_selection_authorized": False,
        "sealed_holdout_access_authorized": False,
    }
    if config.get("policy") != expected_policy:
        raise UgiSourceGuidancePrerequisiteRequalificationError(
            "guidance-prerequisite policy changed"
        )
    inputs = config.get("inputs")
    required = {
        "fresh_pool_v6_config",
        "fresh_pool_v6_result",
        "fresh_pool_v6_component_values",
        "fresh_pool_v6_product_values",
        "value_source",
        "source_qualification",
        "zero_guidance_config",
        "zero_guidance_result",
        "zero_guidance_execution_manifest",
        "matched_budget_config",
        "matched_budget_result",
        "audit_source",
        "audit_runner",
        "audit_tests",
    }
    if not isinstance(inputs, dict) or set(inputs) != required:
        raise UgiSourceGuidancePrerequisiteRequalificationError(
            "guidance-prerequisite inputs changed"
        )
    paths = {
        label: _pin(repo, record, label=label)
        for label, record in inputs.items()
        if isinstance(record, Mapping)
    }
    if set(paths) != required:
        raise UgiSourceGuidancePrerequisiteRequalificationError(
            "guidance-prerequisite pin is malformed"
        )

    fresh_config = _load(paths["fresh_pool_v6_config"], label="fresh-pool v6 config")
    fresh_result = _load(paths["fresh_pool_v6_result"], label="fresh-pool v6 result")
    qualification = _load(paths["source_qualification"], label="source qualification")
    if (
        fresh_config.get("schema_version") != "phase1_ugi3_fresh_pool_route_coverage_config.v6"
        or fresh_result.get("schema_version") != "phase1_ugi3_fresh_pool_route_coverage.v6"
        or fresh_result.get("config_sha256") != sha256_file(paths["fresh_pool_v6_config"])
        or fresh_result.get("status") != "immutable_v5_values_requalified_under_current_source"
        or fresh_result.get("summary", {}).get("value_records_changed") != 0
        or fresh_result.get("summary", {}).get("source_lineage_requalified") is not True
        or fresh_result.get("artifacts", {})
        .get("component_synthesis_values.json.gz", {})
        .get("sha256")
        != sha256_file(paths["fresh_pool_v6_component_values"])
        or fresh_result.get("artifacts", {})
        .get("product_synthesis_values.json.gz", {})
        .get("sha256")
        != sha256_file(paths["fresh_pool_v6_product_values"])
        or fresh_result.get("adjudication", {}).get("zero_guidance_requalification_may_advance")
        is not True
        or fresh_result.get("adjudication", {}).get("nonzero_guidance_authorized") is not False
        or fresh_result.get("adjudication", {}).get("sealed_holdout_accessed") is not False
    ):
        raise UgiSourceGuidancePrerequisiteRequalificationError("fresh-pool v6 is not qualified")
    if (
        qualification.get("status") != "behavior_preserving_synthesis_value_source_qualified"
        or qualification.get("qualified_source", {}).get("sha256")
        != sha256_file(paths["value_source"])
        or qualification.get("adjudication", {}).get("sealed_holdout_accessed") is not False
    ):
        raise UgiSourceGuidancePrerequisiteRequalificationError(
            "synthesis-value source is not qualified"
        )

    zero_config = _load(paths["zero_guidance_config"], label="zero-guidance config")
    zero_result = _load(paths["zero_guidance_result"], label="zero-guidance result")
    zero_manifest = _load(paths["zero_guidance_execution_manifest"], label="zero-guidance manifest")
    zero_scope = zero_result.get("scope", {})
    manifest_scope = zero_manifest.get("scope", {})
    if (
        zero_config.get("schema_version")
        != "phase1_ugi_production_zero_guidance_rehearsal_config.v1"
        or zero_config.get("execution", {}).get("guidance_strength") != 0
        or zero_config.get("execution", {}).get("program_count") != 12
        or zero_result.get("schema_version") != "forge.production_zero_guidance_execution_result.v1"
        or zero_result.get("status") != "production_zero_guidance_execution_complete"
        or zero_result.get("paired_support_count") != 12
        or zero_result.get("paired_support_identity_except_cache_clone_proven") is not True
        or zero_scope.get("guidance_strength") != 0
        or any(
            zero_scope.get(field) is not False
            for field in (
                "biological_guidance",
                "candidate_selection",
                "nonzero_synthesis_guidance",
                "private_holdout_accessed",
                "synthesis_scalar_defined",
            )
        )
        or zero_manifest.get("schema_version") != "phase1_ugi_production_zero_guidance_execution.v1"
        or zero_manifest.get("status") != "production_zero_guidance_rehearsal_complete"
        or zero_manifest.get("paired_support_count") != 12
        or zero_manifest.get("outputs", {}).get("result.json", {}).get("sha256")
        != sha256_file(paths["zero_guidance_result"])
        or any(
            manifest_scope.get(field) is not False
            for field in (
                "biological_guidance",
                "candidate_selection",
                "nonzero_synthesis_guidance",
                "private_holdout_accessed",
                "synthesis_scalar_defined",
            )
        )
        or manifest_scope.get("guidance_strength") != 0
    ):
        raise UgiSourceGuidancePrerequisiteRequalificationError(
            "zero-guidance prerequisite is not qualified"
        )

    matched_config = _load(paths["matched_budget_config"], label="matched-budget config")
    matched_result = _load(paths["matched_budget_result"], label="matched-budget result")
    matched_contract = matched_result.get("matched_contract", {})
    if (
        matched_config.get("schema_version") != "phase1_ugi_matched_budget_orchestration_config.v1"
        or matched_config.get("scope") != "synthetic_nonselecting_matched_budget_orchestration_only"
        or matched_config.get("zero_guidance") is not True
        or matched_config.get("production_synthesis_guidance") is not False
        or matched_config.get("biological_guidance") is not False
        or matched_config.get("candidate_selection") is not False
        or matched_config.get("sealed_holdout_access") is not False
        or matched_result.get("schema_version") != "phase1_ugi_matched_budget_orchestration.v1"
        or matched_result.get("status") != "complete"
        or matched_result.get("decision")
        != "diagnostic_matched_budget_orchestration_contract_qualified"
        or matched_contract.get("zero_guidance") is not True
        or matched_contract.get("zero_guidance_bitwise_equivalent") is not True
        or matched_result.get("real_route_planner_executed") is not False
        or matched_result.get("scalar_synthesis_value_defined") is not False
        or matched_result.get("production_synthesis_guidance") is not False
        or matched_result.get("biological_guidance") is not False
        or matched_result.get("candidate_selection_performed") is not False
        or matched_result.get("sealed_holdout_accessed") is not False
    ):
        raise UgiSourceGuidancePrerequisiteRequalificationError(
            "matched-budget prerequisite is not qualified"
        )

    assessment = config.get("assessment")
    if not isinstance(assessment, dict) or set(assessment) != {"assessment_as_of_utc"}:
        raise UgiSourceGuidancePrerequisiteRequalificationError("assessment record changed")
    assessment_time = _utc(assessment["assessment_as_of_utc"], label="assessment_as_of_utc")
    l3_window = zero_config.get("assessment", {}).get("l3_window", {})
    if l3_window.get("interval_semantics") != "accessed_inclusive_expires_exclusive":
        raise UgiSourceGuidancePrerequisiteRequalificationError("L3 interval semantics changed")
    accessed = _utc(l3_window.get("accessed_utc"), label="L3 accessed_utc")
    expires = _utc(l3_window.get("expires_utc"), label="L3 expires_utc")
    l3_current_at_frozen_assessment = accessed <= assessment_time < expires

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "v6_guidance_prerequisites_requalified_nonzero_guidance_blocked",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "policy": expected_policy,
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "assessment": {
            "assessment_as_of_utc": assessment["assessment_as_of_utc"],
            "l3_accessed_utc": l3_window["accessed_utc"],
            "l3_expires_utc": l3_window["expires_utc"],
            "l3_current_at_frozen_assessment_only": l3_current_at_frozen_assessment,
        },
        "prerequisites": {
            "fresh_pool_v6_source_qualified": True,
            "zero_guidance_prerequisite_requalified_against_v6": True,
            "zero_guidance_paired_supports": 12,
            "matched_budget_prerequisite_requalified_against_v6": True,
            "matched_budget_contract_is_synthetic_only": True,
        },
        "open_gates": {
            "production_cumulative_source_vnext_runtime_qualified": False,
            "scalar_value_policy_frozen": False,
            "l3_refresh_required_before_future_production_execution": True,
        },
        "adjudication": {
            "historical_artifacts_rewritten": False,
            "zero_guidance_prerequisites_requalified": True,
            "nonzero_guidance_authorized": False,
            "candidate_selection_authorized": False,
            "sealed_holdout_accessed": False,
            "stop_reason": (
                "Stop before nonzero guidance: no production synthesis scalar is frozen, "
                "the vNext cumulative-source runtime is not yet qualified, and L3 must be "
                "refreshed for any future production execution."
            ),
        },
    }
