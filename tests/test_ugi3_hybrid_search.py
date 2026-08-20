from __future__ import annotations

from pathlib import Path

import pytest

from forge.route.engine.planner import (
    AssessmentOutcome,
    EvidenceTier,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
)
from forge.route.engine.ugi3_hybrid_search import (
    SearchChannel,
    build_hybrid_search_diagnostic,
    load_bounded_hybrid_source,
)
from forge.route.sources.ugi3_exact_evidence_source import load_exact_evidence_only_source

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_hybrid_search.json"
INPUT_PATHS = {
    "component_dossier": REPO
    / "results/phase1/ugi3_complete_computational_dossiers/component_dossier_ledger.csv.gz",
    "component_dossier_result": REPO
    / "results/phase1/ugi3_complete_computational_dossiers/result.json",
    "component_program": REPO / "results/m0_09/agile_virtual_ugi3_component_program_ledger.csv.gz",
    "component_program_result": REPO / "results/m0_09/agile_virtual_ugi3_component_programs.json",
    "exact_adapter_source": REPO / "src/forge/route/ugi3_exact_evidence_source.py",
    "exact_evidence_ledger": REPO
    / "results/phase1/ugi3_exact_evidence_source/assessment_ledger.json.gz",
    "exact_evidence_result": REPO / "results/phase1/ugi3_exact_evidence_source/result.json",
    "hybrid_adapter_source": REPO / "src/forge/route/ugi3_hybrid_search.py",
    "l1_variant": REPO / "configs/assembly/ugi_variant.yaml",
    "oxidation_variant": REPO
    / "configs/route/variants/ugi3_upstream_primary_alcohol_oxidation_exact_source_v1.json",
    "planner_cache_source": REPO / "src/forge/route/planner_cache.py",
    "planner_contract_source": REPO / "src/forge/route/planner.py",
    "qualified_forward_source": REPO / "src/forge/route/qualified_forward.py",
    "readiness_config": REPO / "configs/route/phase1_ugi3_production_registry_route_readiness.json",
    "readiness_ledger": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/component_readiness_ledger.csv.gz",
    "readiness_result": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/result.json",
    "step_ledger": REPO
    / "results/phase1/ugi3_exact_source_forward_verification/step_verification_ledger.csv.gz",
    "step_result": REPO / "results/phase1/ugi3_exact_source_forward_verification/result.json",
    "terminal_procurement": REPO / "configs/route/m0_09_ugi3_virtual_terminal_procurement.json",
    "transfer_config": REPO / "configs/route/m0_09_hydrophobic_motif_transfer.json",
    "transfer_ledger": REPO / "results/m0_09/hydrophobic_motif_transfer_ledger.csv.gz",
    "transfer_result": REPO / "results/m0_09/hydrophobic_motif_transfer.json",
    "upstream_registry": REPO / "configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json",
}
LIMITS = PlannerBudgetLimits(
    maximum_depth=4,
    maximum_logical_planner_calls=1,
    maximum_expansions=4,
    maximum_product_candidates=4,
    maximum_verifier_calls=1,
    maximum_elapsed_milliseconds=0,
)


def _load_exact_source(*, assessment_as_of_utc: str):
    return load_exact_evidence_only_source(
        component_program_path=INPUT_PATHS["component_program"],
        component_program_result_path=INPUT_PATHS["component_program_result"],
        component_dossier_path=INPUT_PATHS["component_dossier"],
        component_dossier_result_path=INPUT_PATHS["component_dossier_result"],
        step_ledger_path=INPUT_PATHS["step_ledger"],
        step_result_path=INPUT_PATHS["step_result"],
        terminal_procurement_path=INPUT_PATHS["terminal_procurement"],
        assessment_as_of_utc=assessment_as_of_utc,
        unavailable_procurement_statuses=frozenset({"current_item_level_unavailable"}),
    )


def _load_hybrid_source(*, assessment_as_of_utc: str):
    return load_bounded_hybrid_source(
        exact_source=_load_exact_source(assessment_as_of_utc=assessment_as_of_utc),
        readiness_ledger_path=INPUT_PATHS["readiness_ledger"],
        readiness_result_path=INPUT_PATHS["readiness_result"],
        readiness_config_path=INPUT_PATHS["readiness_config"],
        transfer_ledger_path=INPUT_PATHS["transfer_ledger"],
        transfer_result_path=INPUT_PATHS["transfer_result"],
        transfer_config_path=INPUT_PATHS["transfer_config"],
        upstream_registry_path=INPUT_PATHS["upstream_registry"],
        oxidation_variant_path=INPUT_PATHS["oxidation_variant"],
        assessment_as_of_utc=assessment_as_of_utc,
        retrieval_l3_expiry_days=30,
        unavailable_l3_statuses=frozenset({"current_item_level_unavailable"}),
    )


@pytest.fixture(scope="module")
def hybrid_source():
    source, _ = _load_hybrid_source(assessment_as_of_utc="2026-08-01T00:00:00Z")
    return source


def test_frozen_hybrid_diagnostic_is_exact_and_fail_closed(tmp_path: Path) -> None:
    result, _ = build_hybrid_search_diagnostic(
        config_path=CONFIG,
        input_paths=INPUT_PATHS,
        cache_root=tmp_path / "cache",
    )
    summary = result["summary"]
    assert summary["admitted_components"] == 424
    assert summary["baseline_outcomes"] == {
        "complete": 40,
        "missing_knowledge": 53,
        "outside_support": 331,
    }
    assert summary["hybrid_outcomes"] == {
        "complete": 41,
        "missing_knowledge": 277,
        "outside_support": 106,
    }
    assert summary["newly_closed_components"] == 1
    assert summary["newly_closed_original_missing_targets"] == 0
    assert summary["original_missing_targets_remaining"] == 53
    assert summary["admitted_qualified_templates"] == 0
    assert summary["first_pass_verifier_calls"] == 1
    assert result["claims_boundary"] == {
        "bounded_hybrid_search_diagnostic": True,
        "general_dynamic_planner_qualified": False,
        "production_synthesis_guidance_authorized_by_this_result": False,
        "family_projection_can_close": False,
        "motif_similarity_can_close": False,
        "provenance_can_close": False,
        "handle_qualification_can_close": False,
        "qualified_template_closure_demonstrated": False,
        "assessment_is_synthesis_success_probability": False,
    }


def test_only_exact_identity_retrieval_adds_a_closure(hybrid_source) -> None:
    retrieved = next(
        item
        for item in hybrid_source.admitted_components
        if item.channel is SearchChannel.EXACT_IDENTITY_RETRIEVAL
    )
    budget = PlannerBudgetLedger(limits=LIMITS)
    assessment = RecursiveRouteAssessor(hybrid_source).assess(retrieved.target, budget)
    assert assessment.outcome is AssessmentOutcome.COMPLETE
    assert budget.verifier_calls == 1
    assert assessment.route_tree.evidence[0].tier is EvidenceTier.EXACT_SOURCE
    assert assessment.route_tree.children[0].evidence[0].tier is EvidenceTier.ACCEPTED_TERMINAL


def test_family_and_provenance_search_signals_never_close(hybrid_source) -> None:
    assessor = RecursiveRouteAssessor(hybrid_source)
    family = next(
        item
        for item in hybrid_source.admitted_components
        if item.channel is SearchChannel.FAMILY_PROJECTION
    )
    provenance = next(
        item
        for item in hybrid_source.admitted_components
        if item.channel is SearchChannel.PROVENANCE_PRIORITY
    )
    family_result = assessor.assess(family.target, PlannerBudgetLedger(limits=LIMITS))
    provenance_result = assessor.assess(
        provenance.target,
        PlannerBudgetLedger(limits=LIMITS),
    )
    assert family_result.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
    assert family_result.route_tree.evidence[0].tier is EvidenceTier.FAMILY_PROJECTED
    assert provenance_result.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
    assert provenance_result.route_tree.evidence[0].tier is EvidenceTier.PROVENANCE_ONLY


def test_no_admitted_route_remains_outside_support(hybrid_source) -> None:
    outside = next(
        item
        for item in hybrid_source.admitted_components
        if item.channel is SearchChannel.NO_ADMITTED_ROUTE
    )
    assessment = RecursiveRouteAssessor(hybrid_source).assess(
        outside.target,
        PlannerBudgetLedger(limits=LIMITS),
    )
    assert assessment.outcome is AssessmentOutcome.OUTSIDE_SUPPORT


def test_retrieved_route_fails_closed_when_l3_expires() -> None:
    source, metadata = _load_hybrid_source(assessment_as_of_utc="2026-08-30T00:00:00Z")
    retrieved = next(
        item
        for item in source.admitted_components
        if item.channel is SearchChannel.EXACT_IDENTITY_RETRIEVAL
    )
    assessment = RecursiveRouteAssessor(source).assess(
        retrieved.target,
        PlannerBudgetLedger(limits=LIMITS),
    )
    assert metadata["exact_retrieval_l3_state"] == "expired"
    assert assessment.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
