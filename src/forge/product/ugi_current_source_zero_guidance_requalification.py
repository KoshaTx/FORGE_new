"""Freeze current Ugi route evidence and replay locked zero-guidance values.

This additive qualification never reruns molecular generation.  It authenticates
the source-qualified cumulative route source at one explicit decision time,
freezes a decision horizon inside every retained L3 window, and re-evaluates the
already locked zero-guidance terminals through their original support-boundary
qualifications.  No scalar success probability, selection, biology, or holdout
access is introduced.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge.bio.ugi_semantic_annotations import ROLE_NAMES
from forge.core.hashing import sha256_json as _sha256_payload
from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_generated_terminal_support import (
    QualifiedGeneratedUgiTerminalSupport,
    RoleHandleRecheck,
)
from forge.product.ugi_terminal_route_assessment import ExactL1ForwardVerification
from forge.route.planner import (
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteTarget,
)
from forge.route.ugi3_cumulative_production_source import (
    load_cumulative_production_ugi3_source,
)
from forge.route.ugi3_production_planner_qualification import (
    derive_production_internal_role_manifest,
)
from forge.route.ugi3_source_qualified_cumulative_inputs import (
    source_qualified_cumulative_paths,
)
from forge.route.ugi3_support_boundary import MolecularSupportState, TargetQualification
from forge.value.synthesis import (
    ProductSynthesisValue,
    component_synthesis_value_from_assessment,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_current_source_zero_guidance_requalification_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_current_source_zero_guidance_requalification.v1"
_EXPECTED_POLICY = {
    "historical_artifacts_rewritten": False,
    "generator_rerun": False,
    "locked_terminals_reassessed": True,
    "nonzero_guidance_authorized": False,
    "selection_authorized": False,
    "biology_authorized": False,
    "sealed_holdout_access_authorized": False,
    "synthesis_success_probability_defined": False,
}


class UgiCurrentSourceZeroGuidanceRequalificationError(RuntimeError):
    """Raised when the current route/L3/zero-guidance chain cannot fail closed."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise UgiCurrentSourceZeroGuidanceRequalificationError(
            f"invalid {label}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise UgiCurrentSourceZeroGuidanceRequalificationError(f"{label} must be an object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiCurrentSourceZeroGuidanceRequalificationError(f"{label} pin is malformed")
    path = (repo / str(record.get("path"))).resolve()
    try:
        relative = path.relative_to(repo)
    except ValueError as error:
        raise UgiCurrentSourceZeroGuidanceRequalificationError(
            f"{label} escapes repository"
        ) from error
    lowered = "/".join(relative.parts).lower()
    if "holdout" in lowered or "sealed" in lowered:
        raise UgiCurrentSourceZeroGuidanceRequalificationError(f"{label} is forbidden")
    if sha256_file(path) != record.get("sha256"):
        raise UgiCurrentSourceZeroGuidanceRequalificationError(f"{label} hash changed")
    return path


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise UgiCurrentSourceZeroGuidanceRequalificationError(f"{label} is malformed")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise UgiCurrentSourceZeroGuidanceRequalificationError(f"{label} is malformed") from error
    if parsed.tzinfo != timezone.utc:
        raise UgiCurrentSourceZeroGuidanceRequalificationError(f"{label} is not UTC")
    return parsed


def _support_from_dict(value: Any) -> QualifiedGeneratedUgiTerminalSupport:
    if not isinstance(value, dict):
        raise UgiCurrentSourceZeroGuidanceRequalificationError("support snapshot is malformed")
    qualifications = value.get("root_qualifications")
    rechecks = value.get("handle_rechecks")
    targets = value.get("root_targets")
    if not all(isinstance(item, list) for item in (qualifications, rechecks, targets)):
        raise UgiCurrentSourceZeroGuidanceRequalificationError(
            "support snapshot lists are malformed"
        )
    return QualifiedGeneratedUgiTerminalSupport(
        terminal_sha256=value.get("terminal_sha256"),
        generator_checkpoint_sha256=value.get("generator_checkpoint_sha256"),
        product_smiles=value.get("product_smiles"),
        l1_reverification=ExactL1ForwardVerification.from_dict(value.get("l1_reverification")),
        handle_rechecks=tuple(RoleHandleRecheck(**item) for item in rechecks),
        root_targets=tuple(RouteTarget.from_dict(item) for item in targets),
        root_qualifications=tuple(
            TargetQualification(
                exact_l1_eligible=item.get("exact_l1_eligible"),
                supported_ugi_role=item.get("supported_ugi_role"),
                role_handle_qualified=item.get("role_handle_qualified"),
                molecular_support_state=MolecularSupportState(item.get("molecular_support_state")),
                declared_exclusion_code=item.get("declared_exclusion_code"),
                declared_exclusion_policy_locator=item.get("declared_exclusion_policy_locator"),
            )
            for item in qualifications
        ),
    )


def build_current_source_zero_guidance_requalification(
    repo: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Authenticate current L3 evidence and replay all locked guided-arm values."""

    repo = repo.resolve()
    config = _load(config_path, label="current-source requalification config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiCurrentSourceZeroGuidanceRequalificationError("unsupported config schema")
    if config.get("policy") != _EXPECTED_POLICY:
        raise UgiCurrentSourceZeroGuidanceRequalificationError("policy changed")
    inputs = config.get("inputs")
    expected_inputs = {
        "source_qualified_inputs_result",
        "source_qualified_inputs_config",
        "historical_zero_guidance_config_v2",
        "historical_zero_guidance_execution",
        "historical_zero_guidance_composer",
        "historical_zero_guidance_manifest",
        "synthesis_source_qualification",
        "route_completion_utility_qualification",
        "cumulative_source",
        "source_qualified_paths",
        "support_boundary",
        "synthesis_value",
        "builder_source",
        "builder_runner",
        "builder_tests",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_inputs:
        raise UgiCurrentSourceZeroGuidanceRequalificationError("input set changed")
    paths = {label: _pin(repo, record, label=label) for label, record in inputs.items()}

    source_result = _load(
        paths["source_qualified_inputs_result"], label="source-qualified inputs result"
    )
    if source_result.get("status") != "source_qualified_cumulative_inputs_reproduced":
        raise UgiCurrentSourceZeroGuidanceRequalificationError(
            "source-qualified cumulative inputs are not complete"
        )
    source_output = paths["source_qualified_inputs_result"].parent
    assessment_at = config.get("assessment_at_utc")
    decision_horizon = config.get("decision_horizon")
    if not isinstance(decision_horizon, dict) or set(decision_horizon) != {
        "start_utc",
        "end_utc",
        "interval_semantics",
    }:
        raise UgiCurrentSourceZeroGuidanceRequalificationError("decision horizon is malformed")
    if decision_horizon["interval_semantics"] != "start_inclusive_end_exclusive":
        raise UgiCurrentSourceZeroGuidanceRequalificationError("decision horizon semantics changed")
    if assessment_at != decision_horizon["start_utc"]:
        raise UgiCurrentSourceZeroGuidanceRequalificationError(
            "assessment time must equal decision-horizon start"
        )
    start = _parse_utc(decision_horizon["start_utc"], label="decision start")
    end = _parse_utc(decision_horizon["end_utc"], label="decision end")
    if start >= end:
        raise UgiCurrentSourceZeroGuidanceRequalificationError(
            "decision horizon must have positive duration"
        )
    source, metadata = load_cumulative_production_ugi3_source(
        repo_root=repo,
        assessment_as_of_utc=assessment_at,
        paths=source_qualified_cumulative_paths(repo, source_output),
    )
    expected_source_sha256 = config.get("expected_cumulative_source_inputs_sha256")
    if metadata.get("inputs_sha256") != expected_source_sha256:
        raise UgiCurrentSourceZeroGuidanceRequalificationError(
            "cumulative source identity differs from the frozen expectation"
        )
    window = metadata.get("unified_l3_window")
    if not isinstance(window, dict):
        raise UgiCurrentSourceZeroGuidanceRequalificationError("unified L3 window is missing")
    accessed = _parse_utc(window.get("accessed_utc"), label="unified L3 accessed")
    expires = _parse_utc(window.get("expires_utc"), label="unified L3 expires")
    if start < accessed or end > expires:
        raise UgiCurrentSourceZeroGuidanceRequalificationError(
            "decision horizon exceeds the authenticated unified L3 interval"
        )

    execution = _load(
        paths["historical_zero_guidance_execution"], label="historical execution result"
    )
    composer = _load(paths["historical_zero_guidance_composer"], label="historical composer result")
    support_audits = execution.get("support_audits")
    receipts = composer.get("guided_route_receipts")
    if not isinstance(support_audits, list) or not isinstance(receipts, list):
        raise UgiCurrentSourceZeroGuidanceRequalificationError(
            "historical zero-guidance artifacts are malformed"
        )
    supports = {
        item.get("unit_id"): item.get("audit", {}).get("support")
        for item in support_audits
        if isinstance(item, dict) and item.get("arm") == "guided"
    }
    prior = {
        item.get("unit_id"): item
        for item in receipts
        if isinstance(item, dict) and isinstance(item.get("unit_id"), str)
    }
    if not supports or set(supports) != set(prior):
        raise UgiCurrentSourceZeroGuidanceRequalificationError(
            "locked support and route-receipt units do not match"
        )

    internal_roles = derive_production_internal_role_manifest(
        repo_root=repo,
        source_metadata=metadata,
    ).registry
    budget = config.get("per_component_budget")
    if not isinstance(budget, dict):
        raise UgiCurrentSourceZeroGuidanceRequalificationError("budget is malformed")
    limits = PlannerBudgetLimits.from_dict(budget)
    replays: list[dict[str, Any]] = []
    for unit_id in sorted(supports):
        support = _support_from_dict(supports[unit_id])
        assessor = RecursiveRouteAssessor(
            support.wrap_source(source, authenticated_internal_roles=internal_roles)
        )
        components = tuple(
            (
                target.role,
                component_synthesis_value_from_assessment(
                    assessor.assess(target, PlannerBudgetLedger(limits))
                ),
            )
            for target in support.root_targets
        )
        if tuple(role for role, _ in components) != ROLE_NAMES:
            raise UgiCurrentSourceZeroGuidanceRequalificationError(
                f"{unit_id}: component roles changed"
            )
        value = ProductSynthesisValue(
            product_smiles=support.product_smiles,
            l1_forward_consistent=support.l1_reverification.exact_product_reconstructed,
            components=components,
        )
        value_dict = value.to_dict()
        prior_value = prior[unit_id].get("product_value")
        if value_dict != prior_value:
            raise UgiCurrentSourceZeroGuidanceRequalificationError(
                f"{unit_id}: current-source value differs from the locked zero-guidance value"
            )
        replays.append(
            {
                "unit_id": unit_id,
                "terminal_sha256": support.terminal_sha256,
                "prior_product_value_sha256": prior[unit_id].get("product_value_sha256"),
                "replayed_product_value_sha256": _sha256_payload(value_dict),
                "route_complete": value.route_complete,
                "component_outcomes": {
                    role: component.assessment_outcome.value for role, component in components
                },
                "product_value_exactly_reproduced": True,
            }
        )

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "current_source_l3_and_zero_guidance_route_values_requalified",
        "config": {
            "path": str(config_path.resolve().relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "current_cumulative_source": {
            "assessment_at_utc": assessment_at,
            "metadata_sha256": _sha256_payload(metadata),
            "inputs_sha256": metadata["inputs_sha256"],
            "layer_order": metadata["layer_order"],
            "unified_l3_window": window,
            "l3_windows": metadata["l3_windows"],
            "decision_horizon": decision_horizon,
            "all_l3_windows_cover_decision_horizon": True,
        },
        "zero_guidance_v2_route_requalification": {
            "locked_unit_count": len(replays),
            "exact_product_value_reproduction_count": len(replays),
            "all_product_values_exactly_reproduced": True,
            "generator_rerun": False,
            "support_boundary_preserved": True,
            "replays": replays,
        },
        "interpretation": {
            "source_supersession_changed_scientific_values": False,
            "zero_guidance_generator_identity_inherited_not_reexecuted": True,
            "nonzero_guidance_still_requires_grouped_particle_schedule": True,
            "synthesis_value_is_not_success_probability": True,
        },
        "scope": {
            "nonzero_guidance_authorized": False,
            "candidate_selection": False,
            "biology_used": False,
            "sealed_holdout_accessed": False,
            "success_probability": None,
        },
    }
    return {**result, "result_sha256": _sha256_payload(result)}


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UgiCurrentSourceZeroGuidanceRequalificationError",
    "build_current_source_zero_guidance_requalification",
]
