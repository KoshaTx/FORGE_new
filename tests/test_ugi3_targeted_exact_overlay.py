from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.data.r1_prime_audit import sha256_file
from forge.route.audit.ugi3_targeted_exact_overlay_diagnostic import (
    Ugi3TargetedExactOverlayDiagnosticError,
    _verify_hybrid_reproduction,
)
from forge.route.engine.planner import (
    AssessmentOutcome,
    KnowledgeDisposition,
    KnowledgeResult,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteTarget,
)
from forge.route.evidence.ugi3_targeted_exact_overlay import (
    Ugi3TargetedExactOverlayError,
    load_targeted_exact_overlay,
)

REPO = Path(__file__).resolve().parents[1]
AUDIT_CONFIG = REPO / "configs/route/phase1_ugi3_targeted_aldehyde_evidence_audit_v1.json"
AUDIT_INPUT_PATHS = {
    "audit_source": REPO / "src/forge/route/ugi3_targeted_aldehyde_evidence.py",
    "cli_source": REPO / "scripts/phase1_audit_ugi3_targeted_aldehyde_evidence.py",
    "evidence_pack": REPO / "configs/route/phase1_ugi3_targeted_aldehyde_evidence_v1.json",
    "kovalerchik_source": REPO
    / "data/source_cache/phase1_targeted_l2/KOVALERCHIK_2022/marinedrugs-20-00265-v2.pdf",
    "mo_source": REPO
    / "data/source_cache/phase1_targeted_l2/TYPHONOSIDES/supporting_information.pdf",
    "busta_source": REPO
    / "data/source_cache/phase1_targeted_l2/BUSTA_2016/BustaEtAl_2016_Phytochem.pdf",
    "route_readiness_ledger": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/component_readiness_ledger.csv.gz",
    "route_readiness_result": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/result.json",
    "product_gap_ledger": REPO
    / "results/phase1/ugi3_l2_coverage_priority_audit_v1/product_gap_ledger.csv.gz",
    "priority_result": REPO / "results/phase1/ugi3_l2_coverage_priority_audit_v1/result.json",
    "terminal_procurement": REPO / "configs/route/m0_09_ugi3_virtual_terminal_procurement.json",
    "upstream_registry": REPO / "configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json",
    "oxidation_variant": REPO
    / "configs/route/variants/ugi3_upstream_primary_alcohol_oxidation_exact_source_v1.json",
    "qualified_forward_source": REPO / "src/forge/route/qualified_forward.py",
}
STORED_RESULT = REPO / "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/result.json"
STORED_ROUTE_LEDGER = REPO / (
    "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/" "route_verification_ledger.csv.gz"
)
STORED_PRODUCT_LEDGER = REPO / (
    "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/"
    "product_closure_impact_ledger.csv.gz"
)
OVERLAY_DIAGNOSTIC_RESULT = (
    REPO / "results/phase1/ugi3_targeted_exact_overlay_diagnostic_v1/result.json"
)
OVERLAY_DIAGNOSTIC_LEDGER = REPO / (
    "results/phase1/ugi3_targeted_exact_overlay_diagnostic_v1/" "assessment_ledger.json.gz"
)


class RecordingFailClosedSource:
    def __init__(self) -> None:
        self.calls: list[RouteTarget] = []

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        self.calls.append(target)
        return KnowledgeResult(
            disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
            evidence=(),
            detail="base source has no exact evidence",
        )


def _budget() -> PlannerBudgetLedger:
    return PlannerBudgetLedger(
        PlannerBudgetLimits(
            maximum_depth=4,
            maximum_logical_planner_calls=4,
            maximum_expansions=4,
            maximum_product_candidates=4,
            maximum_verifier_calls=4,
            maximum_elapsed_milliseconds=10_000,
        )
    )


def _load(base: RecordingFailClosedSource):
    return load_targeted_exact_overlay(
        base_source=base,
        audit_config_path=AUDIT_CONFIG,
        audit_input_paths=AUDIT_INPUT_PATHS,
        stored_audit_result_path=STORED_RESULT,
        stored_route_ledger_path=STORED_ROUTE_LEDGER,
        stored_product_ledger_path=STORED_PRODUCT_LEDGER,
    )


def test_overlay_closes_only_four_authenticated_exact_targets() -> None:
    base = RecordingFailClosedSource()
    overlay, metadata = _load(base)

    assert metadata["selected_exact_targets"] == 4
    assert metadata["selected_direct_terminals"] == 2
    assert metadata["selected_recursive_routes"] == 2
    assert metadata["retained_unselected_exact_route_alternatives"] == 1
    assert metadata["unresolved_abstentions"] == 1

    expected = {
        "CCCCCCCCCCC=O": (0, 0),
        "CCCCCCCCCCCCCCCCCC=O": (0, 0),
        "CCCCCCCCCCCCCCCCCCCC=O": (1, 1),
        "CCCCCCCCCCCCCCCCCCCCCC=O": (1, 1),
    }
    for record in overlay.overlay_records:
        budget = _budget()
        assessment = RecursiveRouteAssessor(overlay).assess(record.target, budget)
        expansions, verifier_calls = expected[record.target.canonical_smiles]
        assert assessment.outcome is AssessmentOutcome.COMPLETE
        assert budget.expansions == expansions
        assert budget.verifier_calls == verifier_calls
        if expansions:
            assert len(assessment.route_tree.children) == 1
            assert assessment.route_tree.children[0].outcome is AssessmentOutcome.COMPLETE

    assert base.calls == []


def test_unresolved_and_unrelated_targets_delegate_unchanged() -> None:
    base = RecordingFailClosedSource()
    overlay, _ = _load(base)

    unresolved = RecursiveRouteAssessor(overlay).assess(
        overlay.unresolved_target,
        _budget(),
    )
    unrelated_target = RouteTarget(
        role="isocyanide_tail",
        canonical_smiles="[C-]#[N+]CCCCCCCCCCCCC",
    )
    unrelated = RecursiveRouteAssessor(overlay).assess(unrelated_target, _budget())

    assert unresolved.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
    assert unrelated.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
    assert base.calls == [overlay.unresolved_target, unrelated_target]


def test_overlay_rejects_nonreproducible_stored_result(tmp_path: Path) -> None:
    stored = json.loads(STORED_RESULT.read_text())
    stored["summary"]["generated_products_complete_after"] = 25
    tampered = tmp_path / "result.json"
    tampered.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        Ugi3TargetedExactOverlayError,
        match="not reproducible",
    ):
        load_targeted_exact_overlay(
            base_source=RecordingFailClosedSource(),
            audit_config_path=AUDIT_CONFIG,
            audit_input_paths=AUDIT_INPUT_PATHS,
            stored_audit_result_path=tampered,
            stored_route_ledger_path=STORED_ROUTE_LEDGER,
            stored_product_ledger_path=STORED_PRODUCT_LEDGER,
        )


def test_overlay_rejects_unowned_route_ledger(tmp_path: Path) -> None:
    tampered_ledger = tmp_path / "route.csv.gz"
    tampered_ledger.write_bytes(STORED_ROUTE_LEDGER.read_bytes() + b"tampered")

    with pytest.raises(Ugi3TargetedExactOverlayError, match="route ledger hash mismatch"):
        load_targeted_exact_overlay(
            base_source=RecordingFailClosedSource(),
            audit_config_path=AUDIT_CONFIG,
            audit_input_paths=AUDIT_INPUT_PATHS,
            stored_audit_result_path=STORED_RESULT,
            stored_route_ledger_path=tampered_ledger,
            stored_product_ledger_path=STORED_PRODUCT_LEDGER,
        )


def test_full_registry_overlay_artifact_is_owned_and_fail_closed() -> None:
    result = json.loads(OVERLAY_DIAGNOSTIC_RESULT.read_text())
    assert result["summary"] == {
        "admitted_components": 424,
        "baseline_outcomes": {
            "complete": 41,
            "missing_knowledge": 277,
            "outside_support": 106,
        },
        "newly_closed_by_overlay_channel": {
            "targeted_exact_procurement": 2,
            "targeted_exact_route": 2,
        },
        "newly_closed_components": 4,
        "overlay_outcomes": {
            "complete": 45,
            "missing_knowledge": 274,
            "outside_support": 105,
        },
        "overlay_outcomes_by_role": {
            "amine_head": {
                "complete": 17,
                "missing_knowledge": 189,
                "outside_support": 58,
            },
            "isocyanide_tail": {
                "complete": 6,
                "missing_knowledge": 37,
                "outside_support": 10,
            },
            "oxoester_aldehyde_body_tail": {
                "complete": 22,
                "missing_knowledge": 48,
                "outside_support": 37,
            },
        },
        "overlay_verifier_calls": 3,
        "selected_exact_overlay_targets": 4,
        "unresolved_abstentions": 1,
    }
    assert result["artifacts"]["assessment_ledger.json.gz"]["sha256"] == sha256_file(
        OVERLAY_DIAGNOSTIC_LEDGER
    )
    assert result["claims_boundary"]["family_or_analogue_evidence_can_close"] is False
    assert result["claims_boundary"]["production_synthesis_guidance"] is False


def test_hybrid_environment_migration_rejects_any_chemical_result_change() -> None:
    stored = json.loads((REPO / "results/phase1/ugi3_hybrid_search/result.json").read_text())
    fresh = json.loads(json.dumps(stored))
    fresh["planner_cache_context"]["software_versions"] = [
        ["python", "3.14.2"],
        ["rdkit", "2026.03.4"],
    ]
    fresh["summary"]["hybrid_outcomes"]["complete"] = 42
    config = {
        "hybrid_result_environment_migration": {
            "stored_software_versions": [
                ["python", "3.14.2"],
                ["rdkit", "2025.09.6"],
            ],
            "current_software_versions": [
                ["python", "3.14.2"],
                ["rdkit", "2026.03.4"],
            ],
            "require_assessment_ledger_byte_identity": True,
        }
    }
    ledger = b"byte-identical-assessment-evidence"

    with pytest.raises(
        Ugi3TargetedExactOverlayDiagnosticError,
        match="differs beyond",
    ):
        _verify_hybrid_reproduction(
            config=config,
            fresh_result=fresh,
            stored_result=stored,
            fresh_ledger=ledger,
            stored_ledger=ledger,
        )
