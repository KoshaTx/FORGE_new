"""Qualify the targeted exact-evidence overlay across the frozen Ugi registry."""

from __future__ import annotations

import gzip
import io
from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import read_json_object
from forge.core.io import stable_json as _stable_json
from forge.route.planner import (
    AssessmentOutcome,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
)
from forge.route.ugi3_exact_evidence_source import load_exact_evidence_only_source
from forge.route.ugi3_hybrid_search import (
    build_hybrid_search_diagnostic,
    load_bounded_hybrid_source,
)
from forge.route.ugi3_targeted_exact_overlay import load_targeted_exact_overlay

CONFIG_SCHEMA_VERSION = "phase1_ugi3_targeted_exact_overlay_diagnostic_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_targeted_exact_overlay_diagnostic.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi3_targeted_exact_overlay_assessments.v1"


class Ugi3TargetedExactOverlayDiagnosticError(ValueError):
    """Raised when the versioned overlay diagnostic cannot be reproduced."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3TargetedExactOverlayDiagnosticError, label=label)


def _gzip_json_bytes(value: Any) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write((_stable_json(value) + "\n").encode())
    return output.getvalue()


def _validate_expected(observed: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise Ugi3TargetedExactOverlayDiagnosticError(f"{label} fields mismatch")
        for key, value in expected.items():
            _validate_expected(observed[key], value, label=f"{label}.{key}")
        return
    if observed != expected:
        raise Ugi3TargetedExactOverlayDiagnosticError(
            f"{label} mismatch: expected {expected!r}, observed {observed!r}"
        )


def _verify_top_level_inputs(
    config: Mapping[str, Any],
    input_paths: Mapping[str, Path],
) -> dict[str, str]:
    configured = config.get("inputs")
    if not isinstance(configured, dict) or set(configured) != set(input_paths):
        raise Ugi3TargetedExactOverlayDiagnosticError(
            "configured and supplied overlay inputs differ"
        )
    hashes: dict[str, str] = {}
    for name, path in input_paths.items():
        record = configured[name]
        if not isinstance(record, dict):
            raise Ugi3TargetedExactOverlayDiagnosticError(f"input {name} is malformed")
        observed = sha256_file(path)
        if observed != record.get("expected_sha256"):
            raise Ugi3TargetedExactOverlayDiagnosticError(f"{name} hash mismatch")
        hashes[name] = observed
    return hashes


def _verify_hybrid_reproduction(
    *,
    config: Mapping[str, Any],
    fresh_result: Mapping[str, Any],
    stored_result: Mapping[str, Any],
    fresh_ledger: bytes,
    stored_ledger: bytes,
) -> dict[str, Any]:
    if fresh_ledger != stored_ledger:
        raise Ugi3TargetedExactOverlayDiagnosticError("stored hybrid ledger is not reproducible")
    if fresh_result == stored_result:
        return {"mode": "exact", "ledger_byte_identical": True}
    migration = config.get("hybrid_result_environment_migration")
    if not isinstance(migration, dict):
        raise Ugi3TargetedExactOverlayDiagnosticError("stored hybrid result is not reproducible")
    stored_versions = migration.get("stored_software_versions")
    current_versions = migration.get("current_software_versions")
    if (
        stored_result.get("planner_cache_context", {}).get("software_versions") != stored_versions
        or fresh_result.get("planner_cache_context", {}).get("software_versions")
        != current_versions
    ):
        raise Ugi3TargetedExactOverlayDiagnosticError(
            "hybrid environment migration does not match observed software versions"
        )
    normalized_fresh = deepcopy(fresh_result)
    normalized_fresh["planner_cache_context"]["software_versions"] = stored_versions
    if normalized_fresh != stored_result:
        raise Ugi3TargetedExactOverlayDiagnosticError(
            "hybrid rebuild differs beyond the declared software-version migration"
        )
    if migration.get("require_assessment_ledger_byte_identity") is not True:
        raise Ugi3TargetedExactOverlayDiagnosticError(
            "hybrid environment migration must require byte-identical assessment evidence"
        )
    return {
        "mode": "software_version_only_migration",
        "stored_software_versions": stored_versions,
        "current_software_versions": current_versions,
        "ledger_byte_identical": True,
        "other_result_differences": 0,
    }


def _load_hybrid_source(
    *,
    hybrid_config: Mapping[str, Any],
    hybrid_input_paths: Mapping[str, Path],
):
    policy = hybrid_config.get("search_policy")
    if not isinstance(policy, dict):
        raise Ugi3TargetedExactOverlayDiagnosticError("hybrid search policy is malformed")
    unavailable = policy.get("unavailable_l3_statuses")
    if not isinstance(unavailable, list) or any(
        not isinstance(value, str) for value in unavailable
    ):
        raise Ugi3TargetedExactOverlayDiagnosticError("hybrid unavailable statuses are malformed")
    exact_source = load_exact_evidence_only_source(
        component_program_path=hybrid_input_paths["component_program"],
        component_program_result_path=hybrid_input_paths["component_program_result"],
        component_dossier_path=hybrid_input_paths["component_dossier"],
        component_dossier_result_path=hybrid_input_paths["component_dossier_result"],
        step_ledger_path=hybrid_input_paths["step_ledger"],
        step_result_path=hybrid_input_paths["step_result"],
        terminal_procurement_path=hybrid_input_paths["terminal_procurement"],
        assessment_as_of_utc=policy.get("assessment_as_of_utc"),
        unavailable_procurement_statuses=frozenset(unavailable),
    )
    return load_bounded_hybrid_source(
        exact_source=exact_source,
        readiness_ledger_path=hybrid_input_paths["readiness_ledger"],
        readiness_result_path=hybrid_input_paths["readiness_result"],
        readiness_config_path=hybrid_input_paths["readiness_config"],
        transfer_ledger_path=hybrid_input_paths["transfer_ledger"],
        transfer_result_path=hybrid_input_paths["transfer_result"],
        transfer_config_path=hybrid_input_paths["transfer_config"],
        upstream_registry_path=hybrid_input_paths["upstream_registry"],
        oxidation_variant_path=hybrid_input_paths["oxidation_variant"],
        assessment_as_of_utc=policy.get("assessment_as_of_utc"),
        retrieval_l3_expiry_days=policy.get("retrieval_l3_expiry_days"),
        unavailable_l3_statuses=frozenset(unavailable),
    )


def build_targeted_exact_overlay_diagnostic(
    *,
    config_path: Path,
    input_paths: Mapping[str, Path],
    hybrid_input_paths: Mapping[str, Path],
    targeted_audit_input_paths: Mapping[str, Path],
    cache_root: Path,
) -> tuple[dict[str, Any], bytes]:
    """Rebuild both prerequisites and assess all 424 admitted components."""

    config = _load_json(config_path, label="targeted overlay diagnostic config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3TargetedExactOverlayDiagnosticError("unsupported config schema")
    input_hashes = _verify_top_level_inputs(config, input_paths)

    fresh_hybrid, fresh_hybrid_ledger = build_hybrid_search_diagnostic(
        config_path=input_paths["hybrid_config"],
        input_paths=hybrid_input_paths,
        cache_root=cache_root,
    )
    stored_hybrid = _load_json(input_paths["hybrid_result"], label="stored hybrid result")
    hybrid_reproduction = _verify_hybrid_reproduction(
        config=config,
        fresh_result=fresh_hybrid,
        stored_result=stored_hybrid,
        fresh_ledger=fresh_hybrid_ledger,
        stored_ledger=input_paths["hybrid_ledger"].read_bytes(),
    )

    hybrid_config = _load_json(input_paths["hybrid_config"], label="hybrid config")
    hybrid_source, hybrid_metadata = _load_hybrid_source(
        hybrid_config=hybrid_config,
        hybrid_input_paths=hybrid_input_paths,
    )
    overlay, overlay_metadata = load_targeted_exact_overlay(
        base_source=hybrid_source,
        audit_config_path=input_paths["targeted_audit_config"],
        audit_input_paths=targeted_audit_input_paths,
        stored_audit_result_path=input_paths["targeted_audit_result"],
        stored_route_ledger_path=input_paths["targeted_route_ledger"],
        stored_product_ledger_path=input_paths["targeted_product_ledger"],
    )

    policy = config.get("assessment_policy")
    if not isinstance(policy, dict):
        raise Ugi3TargetedExactOverlayDiagnosticError("assessment policy is malformed")
    limits = PlannerBudgetLimits.from_dict(policy.get("per_component_budget"))
    overlay_keys = {
        (record.target.role, record.target.canonical_smiles): record
        for record in overlay.overlay_records
    }
    baseline_assessor = RecursiveRouteAssessor(hybrid_source)
    overlay_assessor = RecursiveRouteAssessor(overlay)
    baseline_outcomes: Counter[str] = Counter()
    overlay_outcomes: Counter[str] = Counter()
    overlay_role_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    newly_closed_by_channel: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    newly_closed = 0
    verifier_calls = 0
    for admitted in hybrid_source.admitted_components:
        baseline_budget = PlannerBudgetLedger(limits=limits)
        baseline = baseline_assessor.assess(admitted.target, baseline_budget)
        overlay_budget = PlannerBudgetLedger(limits=limits)
        assessment = overlay_assessor.assess(admitted.target, overlay_budget)
        baseline_outcomes[baseline.outcome.value] += 1
        overlay_outcomes[assessment.outcome.value] += 1
        overlay_role_outcomes[admitted.target.role][assessment.outcome.value] += 1
        verifier_calls += overlay_budget.verifier_calls
        overlay_record = overlay_keys.get((admitted.target.role, admitted.target.canonical_smiles))
        became_complete = (
            baseline.outcome is not AssessmentOutcome.COMPLETE
            and assessment.outcome is AssessmentOutcome.COMPLETE
        )
        if became_complete:
            newly_closed += 1
            if overlay_record is None:
                raise Ugi3TargetedExactOverlayDiagnosticError(
                    "unattributed component closure detected"
                )
            newly_closed_by_channel[overlay_record.channel] += 1
        rows.append(
            {
                "component_id": admitted.component_id,
                "role": admitted.target.role,
                "canonical_smiles": admitted.target.canonical_smiles,
                "baseline_outcome": baseline.outcome.value,
                "overlay_outcome": assessment.outcome.value,
                "targeted_overlay_channel": (
                    None if overlay_record is None else overlay_record.channel
                ),
                "targeted_overlay_route_id": (
                    None if overlay_record is None else overlay_record.route_id
                ),
                "newly_closed": became_complete,
                "logical_planner_calls": overlay_budget.logical_planner_calls,
                "expansions": overlay_budget.expansions,
                "verifier_calls": overlay_budget.verifier_calls,
                "evidence_ids": sorted(
                    {
                        evidence_id
                        for event in assessment.trace
                        for evidence_id in event.evidence_ids
                    }
                ),
            }
        )

    summary = {
        "admitted_components": len(hybrid_source.admitted_components),
        "baseline_outcomes": dict(sorted(baseline_outcomes.items())),
        "overlay_outcomes": dict(sorted(overlay_outcomes.items())),
        "overlay_outcomes_by_role": {
            role: dict(sorted(outcomes.items()))
            for role, outcomes in sorted(overlay_role_outcomes.items())
        },
        "newly_closed_components": newly_closed,
        "newly_closed_by_overlay_channel": dict(sorted(newly_closed_by_channel.items())),
        "selected_exact_overlay_targets": len(overlay_keys),
        "overlay_verifier_calls": verifier_calls,
        "unresolved_abstentions": overlay_metadata["unresolved_abstentions"],
    }
    expected = config.get("expected_summary")
    if not isinstance(expected, dict):
        raise Ugi3TargetedExactOverlayDiagnosticError("expected summary is malformed")
    _validate_expected(summary, expected, label="summary")

    ledger_payload = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "rows": sorted(rows, key=lambda row: row["component_id"]),
    }
    ledger = _gzip_json_bytes(ledger_payload)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "config_sha256": sha256_file(config_path),
        "input_sha256": dict(sorted(input_hashes.items())),
        "prerequisite_metadata": {
            "hybrid": hybrid_metadata,
            "hybrid_reproduction": hybrid_reproduction,
            "targeted_overlay": overlay_metadata,
        },
        "summary": summary,
        "artifacts": {
            "assessment_ledger.json.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "sha256": sha256_bytes(ledger),
            }
        },
        "claims_boundary": {
            "scope": "bounded_exact_evidence_overlay_diagnostic",
            "dynamic_general_route_planner": False,
            "family_or_analogue_evidence_can_close": False,
            "production_synthesis_guidance": False,
            "synthesis_success_probability": False,
        },
    }
    return result, ledger
