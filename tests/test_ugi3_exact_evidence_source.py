from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from forge.route.planner import (
    AssessmentOutcome,
    AvailabilityState,
    EvidenceTier,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteTarget,
)
from forge.route.ugi3_exact_evidence_source import (
    ExactEvidenceOnlyUgi3Source,
    build_exact_evidence_source_diagnostic,
    load_exact_evidence_only_source,
    procurement_availability,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_exact_evidence_source.json"
INPUT_PATHS = {
    "adapter_source": REPO / "src/forge/route/ugi3_exact_evidence_source.py",
    "component_dossier": REPO
    / "results/phase1/ugi3_complete_computational_dossiers/component_dossier_ledger.csv.gz",
    "component_dossier_result": REPO
    / "results/phase1/ugi3_complete_computational_dossiers/result.json",
    "component_program": REPO / "results/m0_09/agile_virtual_ugi3_component_program_ledger.csv.gz",
    "component_program_result": REPO / "results/m0_09/agile_virtual_ugi3_component_programs.json",
    "l1_variant": REPO / "configs/assembly/ugi_variant.yaml",
    "planner_cache_source": REPO / "src/forge/route/planner_cache.py",
    "planner_contract_source": REPO / "src/forge/route/planner.py",
    "step_ledger": REPO
    / "results/phase1/ugi3_exact_source_forward_verification/step_verification_ledger.csv.gz",
    "step_result": REPO / "results/phase1/ugi3_exact_source_forward_verification/result.json",
    "terminal_procurement": REPO / "configs/route/m0_09_ugi3_virtual_terminal_procurement.json",
    "upstream_reaction_registry": REPO
    / "configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json",
}
LIMITS = PlannerBudgetLimits(
    maximum_depth=4,
    maximum_logical_planner_calls=1,
    maximum_expansions=4,
    maximum_product_candidates=4,
    maximum_verifier_calls=0,
    maximum_elapsed_milliseconds=0,
)


@pytest.fixture(scope="module")
def frozen_source() -> ExactEvidenceOnlyUgi3Source:
    return load_exact_evidence_only_source(
        component_program_path=INPUT_PATHS["component_program"],
        component_program_result_path=INPUT_PATHS["component_program_result"],
        component_dossier_path=INPUT_PATHS["component_dossier"],
        component_dossier_result_path=INPUT_PATHS["component_dossier_result"],
        step_ledger_path=INPUT_PATHS["step_ledger"],
        step_result_path=INPUT_PATHS["step_result"],
        terminal_procurement_path=INPUT_PATHS["terminal_procurement"],
        assessment_as_of_utc="2026-08-01T00:00:00Z",
        unavailable_procurement_statuses=frozenset({"current_item_level_unavailable"}),
    )


def test_procurement_states_are_distinct_and_time_pinned() -> None:
    snapshot = {
        "accessed_utc": "2026-08-01T00:00:00Z",
        "expiry_days": 30,
    }
    current = {"current_item_level_procurement_closed": True}
    unavailable = {
        "current_item_level_procurement_closed": False,
        "procurement_status": "current_item_level_unavailable",
    }
    unassessed = {
        "current_item_level_procurement_closed": False,
        "procurement_status": "current_item_level_vendor_verified",
    }
    statuses = frozenset({"current_item_level_unavailable"})
    assert (
        procurement_availability(
            snapshot=snapshot,
            record=current,
            assessment_as_of_utc="2026-08-02T00:00:00Z",
            unavailable_procurement_statuses=statuses,
        )
        is AvailabilityState.CURRENT_CLOSED
    )
    assert (
        procurement_availability(
            snapshot=snapshot,
            record=unavailable,
            assessment_as_of_utc="2026-08-02T00:00:00Z",
            unavailable_procurement_statuses=statuses,
        )
        is AvailabilityState.UNAVAILABLE
    )
    assert (
        procurement_availability(
            snapshot=snapshot,
            record=unassessed,
            assessment_as_of_utc="2026-08-02T00:00:00Z",
            unavailable_procurement_statuses=statuses,
        )
        is AvailabilityState.UNASSESSED
    )
    expired_at = datetime(2026, 8, 1, tzinfo=timezone.utc) + timedelta(days=31)
    assert (
        procurement_availability(
            snapshot=snapshot,
            record=current,
            assessment_as_of_utc=expired_at.isoformat().replace("+00:00", "Z"),
            unavailable_procurement_statuses=statuses,
        )
        is AvailabilityState.EXPIRED
    )


def test_family_and_provenance_records_remain_missing_knowledge(
    frozen_source: ExactEvidenceOnlyUgi3Source,
) -> None:
    assessor = RecursiveRouteAssessor(frozen_source)
    family = next(
        item
        for item in frozen_source.admitted_components
        if item.program_status == "reaction_family_projected_program"
    )
    unresolved = next(
        item
        for item in frozen_source.admitted_components
        if item.program_status == "procurement_or_route_search_required"
    )
    family_result = assessor.assess(family.target, PlannerBudgetLedger(limits=LIMITS))
    unresolved_result = assessor.assess(unresolved.target, PlannerBudgetLedger(limits=LIMITS))
    assert family_result.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
    assert family_result.route_tree.evidence[0].tier is EvidenceTier.FAMILY_PROJECTED
    assert unresolved_result.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
    assert unresolved_result.route_tree.evidence[0].tier is EvidenceTier.PROVENANCE_ONLY


def test_exact_open_component_fails_closed_at_unassessed_l3(
    frozen_source: ExactEvidenceOnlyUgi3Source,
) -> None:
    admitted = next(
        item
        for item in frozen_source.admitted_components
        if item.existing_evidence_tier == "exact_source_l2_verified_l3_open"
    )
    result = RecursiveRouteAssessor(frozen_source).assess(
        admitted.target,
        PlannerBudgetLedger(limits=LIMITS),
    )
    assert result.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
    terminal_evidence = [
        evidence
        for event in result.trace
        for evidence in event.evidence_ids
        if evidence.startswith("l3-procurement:")
    ]
    assert terminal_evidence
    assert any(
        child.evidence and child.evidence[0].availability is AvailabilityState.UNASSESSED
        for node in result.route_tree.children
        for child in node.children
    )


def test_expired_and_explicitly_unavailable_l3_fail_closed_end_to_end(
    tmp_path: Path,
) -> None:
    expired_source = load_exact_evidence_only_source(
        component_program_path=INPUT_PATHS["component_program"],
        component_program_result_path=INPUT_PATHS["component_program_result"],
        component_dossier_path=INPUT_PATHS["component_dossier"],
        component_dossier_result_path=INPUT_PATHS["component_dossier_result"],
        step_ledger_path=INPUT_PATHS["step_ledger"],
        step_result_path=INPUT_PATHS["step_result"],
        terminal_procurement_path=INPUT_PATHS["terminal_procurement"],
        assessment_as_of_utc="2026-08-29T20:49:01Z",
        unavailable_procurement_statuses=frozenset({"current_item_level_unavailable"}),
    )
    accepted = next(
        item
        for item in expired_source.admitted_components
        if item.existing_evidence_tier == "accepted_terminal_l3_closed"
    )
    expired = RecursiveRouteAssessor(expired_source).assess(
        accepted.target,
        PlannerBudgetLedger(limits=LIMITS),
    )
    assert expired.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
    assert expired.route_tree.evidence[0].availability is AvailabilityState.EXPIRED

    procurement = json.loads(INPUT_PATHS["terminal_procurement"].read_text())
    oleylamine = next(
        record
        for record in procurement["records"]
        if record["canonical_smiles"] == "CCCCCCCC/C=C\\CCCCCCCCN"
    )
    oleylamine["procurement_status"] = "current_item_level_unavailable"
    unavailable_path = tmp_path / "unavailable_procurement.json"
    unavailable_path.write_text(json.dumps(procurement))
    unavailable_source = load_exact_evidence_only_source(
        component_program_path=INPUT_PATHS["component_program"],
        component_program_result_path=INPUT_PATHS["component_program_result"],
        component_dossier_path=INPUT_PATHS["component_dossier"],
        component_dossier_result_path=INPUT_PATHS["component_dossier_result"],
        step_ledger_path=INPUT_PATHS["step_ledger"],
        step_result_path=INPUT_PATHS["step_result"],
        terminal_procurement_path=unavailable_path,
        assessment_as_of_utc="2026-08-01T00:00:00Z",
        unavailable_procurement_statuses=frozenset({"current_item_level_unavailable"}),
    )
    open_component = next(
        item
        for item in unavailable_source.admitted_components
        if item.existing_evidence_tier == "exact_source_l2_verified_l3_open"
    )
    unavailable = RecursiveRouteAssessor(unavailable_source).assess(
        open_component.target,
        PlannerBudgetLedger(limits=LIMITS),
    )
    assert unavailable.outcome is AssessmentOutcome.OUTSIDE_SUPPORT


def test_exact_closed_and_accepted_terminal_components_close(
    frozen_source: ExactEvidenceOnlyUgi3Source,
) -> None:
    assessor = RecursiveRouteAssessor(frozen_source)
    for tier in (
        "accepted_terminal_l3_closed",
        "exact_source_l2_verified_l3_closed",
    ):
        admitted = next(
            item
            for item in frozen_source.admitted_components
            if item.existing_evidence_tier == tier
        )
        result = assessor.assess(admitted.target, PlannerBudgetLedger(limits=LIMITS))
        assert result.outcome is AssessmentOutcome.COMPLETE


def test_unknown_and_invalid_targets_fail_closed(
    frozen_source: ExactEvidenceOnlyUgi3Source,
) -> None:
    assessor = RecursiveRouteAssessor(frozen_source)
    outside = assessor.assess(
        RouteTarget(role="amine_head", canonical_smiles="CCN"),
        PlannerBudgetLedger(limits=LIMITS),
    )
    invalid = assessor.assess(
        RouteTarget(role="not_a_ugi_role", canonical_smiles="CCN"),
        PlannerBudgetLedger(limits=LIMITS),
    )
    assert outside.outcome is AssessmentOutcome.OUTSIDE_SUPPORT
    assert invalid.outcome is AssessmentOutcome.INVALID_INPUT


def test_frozen_diagnostic_reproduces_all_component_classifications(tmp_path: Path) -> None:
    result, ledger = build_exact_evidence_source_diagnostic(
        config_path=CONFIG,
        input_paths=INPUT_PATHS,
        cache_root=tmp_path / "cache",
    )
    assert result["summary"]["admitted_components"] == 93
    assert result["summary"]["tier_crosswalk_agreement"] == 93
    assert result["summary"]["complete_flag_agreement"] == 93
    assert result["summary"]["first_pass_cache_misses"] == 93
    assert result["summary"]["second_pass_cache_hits"] == 93
    assert result["artifacts"]["assessment_ledger_sha256"]
    assert ledger.startswith(b"\x1f\x8b")
