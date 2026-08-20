from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.data.r1_prime_audit import sha256_file
from forge.route.engine.planner import (
    AssessmentOutcome,
    KnowledgeDisposition,
    KnowledgeResult,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteTarget,
)
from forge.route.evidence.ugi3_targeted_role_gap_evidence import (
    Ugi3TargetedRoleGapEvidenceError,
    build_targeted_role_gap_evidence_audit,
)
from forge.route.assessment.ugi3_targeted_role_gap_overlay import load_targeted_role_gap_overlay

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_targeted_role_gap_evidence_audit_v1.json"
INPUT_PATHS = {
    "agile_supplement": REPO / "data/vendor/agile_supplementary_information.pdf",
    "audit_source": REPO / "src/forge/route/ugi3_targeted_role_gap_evidence.py",
    "evidence_pack": REPO / "configs/route/phase1_ugi3_targeted_role_gap_evidence_v1.json",
    "paper_reviews": REPO / "configs/route/m0_09_lnpdb_paper_reviews.json",
    "product_impact_ledger": REPO
    / "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/product_closure_impact_ledger.csv.gz",
    "qualifier_source": REPO / "scripts/phase1_audit_ugi3_targeted_role_gap_evidence.py",
    "readiness_ledger": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/component_readiness_ledger.csv.gz",
    "targeted_aldehyde_result": REPO
    / "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/result.json",
    "targeted_overlay_result": REPO
    / "results/phase1/ugi3_targeted_exact_overlay_diagnostic_v1/result.json",
}
STORED_RESULT = REPO / "results/phase1/ugi3_targeted_role_gap_evidence_audit_v1/result.json"
STORED_LEDGER = REPO / (
    "results/phase1/ugi3_targeted_role_gap_evidence_audit_v1/" "product_impact_ledger.csv.gz"
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
            maximum_depth=2,
            maximum_logical_planner_calls=2,
            maximum_expansions=2,
            maximum_product_candidates=2,
            maximum_verifier_calls=2,
            maximum_elapsed_milliseconds=0,
        )
    )


def test_role_gap_audit_admits_head_and_preserves_isocyanide_abstention() -> None:
    first, first_ledger = build_targeted_role_gap_evidence_audit(
        config_path=CONFIG,
        input_paths=INPUT_PATHS,
    )
    second, second_ledger = build_targeted_role_gap_evidence_audit(
        config_path=CONFIG,
        input_paths=INPUT_PATHS,
    )

    assert first == second
    assert first_ledger == second_ledger
    assert first["summary"]["registry_complete_components_after"] == 46
    assert first["summary"]["generated_products_complete_after"] == 29
    assert first["summary"]["newly_complete_generated_products"] == 5
    assert first["summary"]["unresolved_isocyanide_occurrences_after_head_closure"] == 68
    assert first["summary"]["products_one_exact_isocyanide_gap_from_completion"] == 4
    assert first["claims_boundary"]["homologue_evidence_can_close_exact_isocyanide_route"] is False


def test_role_gap_overlay_closes_only_the_exact_head() -> None:
    base = RecordingFailClosedSource()
    overlay, metadata = load_targeted_role_gap_overlay(
        base_source=base,
        audit_config_path=CONFIG,
        audit_input_paths=INPUT_PATHS,
        stored_audit_result_path=STORED_RESULT,
        stored_product_ledger_path=STORED_LEDGER,
    )
    head = RecursiveRouteAssessor(overlay).assess(overlay.head_target, _budget())
    isocyanide = RecursiveRouteAssessor(overlay).assess(
        overlay.unresolved_isocyanide_target,
        _budget(),
    )

    assert metadata["selected_exact_head_terminals"] == 1
    assert metadata["unresolved_isocyanide_abstentions"] == 1
    assert head.outcome is AssessmentOutcome.COMPLETE
    assert isocyanide.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
    assert base.calls == [overlay.unresolved_isocyanide_target]


def test_homologue_list_cannot_be_promoted_to_exact_target(tmp_path: Path) -> None:
    evidence = json.loads(INPUT_PATHS["evidence_pack"].read_text())
    evidence["unresolved_isocyanide_target"]["nonclosing_homologue_evidence"][
        "exact_neighbor_products"
    ].append("[C-]#[N+]CCCCCCCCCCCCC")
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    config = json.loads(CONFIG.read_text())
    config["inputs"]["evidence_pack"]["expected_sha256"] = sha256_file(evidence_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    inputs = dict(INPUT_PATHS)
    inputs["evidence_pack"] = evidence_path

    with pytest.raises(Ugi3TargetedRoleGapEvidenceError, match="recorded as absent"):
        build_targeted_role_gap_evidence_audit(
            config_path=config_path,
            input_paths=inputs,
        )
