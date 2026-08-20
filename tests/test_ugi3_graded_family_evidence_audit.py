from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

import pytest

from experiments.archive.phase1.synthesis_value_audits.ugi3_graded_family_evidence_audit import (
    EXACT_COMPLETE,
    FAMILY_ALL_CURRENT,
    FAMILY_NO_CURRENT,
    FAMILY_PARTIAL_CURRENT,
    INCOMPATIBLE,
    MISSING_KNOWLEDGE,
    OUTSIDE_SUPPORT,
    Ugi3GradedFamilyEvidenceAuditError,
    build_graded_family_evidence_audit,
    classify_graded_evidence,
)
from forge.corpus.r1_prime_audit import sha256_bytes, sha256_file
from forge.synthesis.engine.planner import AssessmentOutcome, RouteTarget
from forge.synthesis.value.contracts import (
    BurdenEstimate,
    ComponentSynthesisValue,
    EvidenceSupport,
    ForwardConsistency,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_graded_family_evidence_audit_v1.json"


def _rows(payload: bytes) -> list[dict[str, str]]:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
        with io.TextIOWrapper(compressed) as handle:
            return list(csv.DictReader(handle))


def _value(outcome: AssessmentOutcome) -> ComponentSynthesisValue:
    complete = outcome is AssessmentOutcome.COMPLETE
    leaf_field = {
        AssessmentOutcome.COMPLETE: "current_terminal_leaf_count",
        AssessmentOutcome.MISSING_KNOWLEDGE: "missing_knowledge_leaf_count",
        AssessmentOutcome.OUTSIDE_SUPPORT: "outside_support_leaf_count",
        AssessmentOutcome.INCOMPATIBLE: "incompatible_leaf_count",
    }[outcome]
    leaf_counts = {
        "current_terminal_leaf_count": 0,
        "unavailable_terminal_leaf_count": 0,
        "unassessed_terminal_leaf_count": 0,
        "missing_knowledge_leaf_count": 0,
        "outside_support_leaf_count": 0,
        "incompatible_leaf_count": 0,
        "budget_exhausted_leaf_count": 0,
        "invalid_input_leaf_count": 0,
        "execution_error_leaf_count": 0,
    }
    leaf_counts[leaf_field] = 1
    return ComponentSynthesisValue(
        target=RouteTarget(role="amine_head", canonical_smiles="CN"),
        assessment_outcome=outcome,
        forward_consistency=ForwardConsistency.NOT_APPLICABLE,
        evidence_support=(
            EvidenceSupport.NOT_APPLICABLE if complete else EvidenceSupport.PROVENANCE_ONLY
        ),
        route_step_count=0,
        maximum_route_depth=0,
        leaf_count=1,
        protection_burden=BurdenEstimate.unknown(),
        purification_burden=BurdenEstimate.unknown(),
        **leaf_counts,
    )


def test_categorical_classifier_preserves_evidence_and_typed_abstentions() -> None:
    exact = classify_graded_evidence(
        exact_value=_value(AssessmentOutcome.COMPLETE),
        program_family="",
        projected_leaf_count=0,
        current_projected_leaf_count=0,
    )
    all_current = classify_graded_evidence(
        exact_value=_value(AssessmentOutcome.MISSING_KNOWLEDGE),
        program_family="example_family",
        projected_leaf_count=2,
        current_projected_leaf_count=2,
    )
    partial = classify_graded_evidence(
        exact_value=_value(AssessmentOutcome.MISSING_KNOWLEDGE),
        program_family="example_family",
        projected_leaf_count=2,
        current_projected_leaf_count=1,
    )
    none_current = classify_graded_evidence(
        exact_value=_value(AssessmentOutcome.MISSING_KNOWLEDGE),
        program_family="example_family",
        projected_leaf_count=2,
        current_projected_leaf_count=0,
    )
    missing = classify_graded_evidence(
        exact_value=_value(AssessmentOutcome.MISSING_KNOWLEDGE),
        program_family="",
        projected_leaf_count=0,
        current_projected_leaf_count=0,
    )
    outside = classify_graded_evidence(
        exact_value=_value(AssessmentOutcome.OUTSIDE_SUPPORT),
        program_family="",
        projected_leaf_count=0,
        current_projected_leaf_count=0,
    )
    incompatible = classify_graded_evidence(
        exact_value=_value(AssessmentOutcome.INCOMPATIBLE),
        program_family="",
        projected_leaf_count=0,
        current_projected_leaf_count=0,
    )

    assert exact[0] == EXACT_COMPLETE
    assert all_current[0] == FAMILY_ALL_CURRENT
    assert partial[0] == FAMILY_PARTIAL_CURRENT
    assert none_current[0] == FAMILY_NO_CURRENT
    assert missing[0] == MISSING_KNOWLEDGE
    assert outside[0] == OUTSIDE_SUPPORT
    assert incompatible[0] == INCOMPATIBLE
    assert len({missing[0], outside[0], incompatible[0]}) == 3
    with pytest.raises(
        Ugi3GradedFamilyEvidenceAuditError,
        match="cannot replace a typed non-missing outcome",
    ):
        classify_graded_evidence(
            exact_value=_value(AssessmentOutcome.OUTSIDE_SUPPORT),
            program_family="example_family",
            projected_leaf_count=1,
            current_projected_leaf_count=1,
        )


def test_final_v5_graded_audit_is_deterministic_nonselecting_and_nonprobabilistic() -> None:
    first = build_graded_family_evidence_audit(REPO, CONFIG)
    second = build_graded_family_evidence_audit(REPO, CONFIG)
    assert first == second
    result, ledger = first
    assert result["status"] == "complete_nonselecting_graded_family_evidence_audit"
    assert result["artifacts"]["graded_family_evidence_ledger.csv.gz"]["sha256"] == (
        sha256_bytes(ledger)
    )
    assert result["declared_evidence_order"] == [
        EXACT_COMPLETE,
        FAMILY_ALL_CURRENT,
        FAMILY_PARTIAL_CURRENT,
        FAMILY_NO_CURRENT,
    ]
    assert result["summary"]["components"] == 1919
    assert result["summary"]["components_by_graded_evidence_class"] == {
        EXACT_COMPLETE: 45,
        FAMILY_ALL_CURRENT: 32,
        FAMILY_PARTIAL_CURRENT: 57,
        FAMILY_NO_CURRENT: 120,
        MISSING_KNOWLEDGE: 1620,
        OUTSIDE_SUPPORT: 45,
        INCOMPATIBLE: 0,
        "budget_exhausted": 0,
        "invalid_input": 0,
        "execution_error": 0,
    }
    assert result["summary"]["one_gap_components"] == 401
    assert result["summary"]["one_gap_product_occurrences"] == 897
    assert result["summary"]["one_gap_components_by_graded_evidence_class"] == {
        EXACT_COMPLETE: 0,
        FAMILY_ALL_CURRENT: 32,
        FAMILY_PARTIAL_CURRENT: 57,
        FAMILY_NO_CURRENT: 120,
        MISSING_KNOWLEDGE: 168,
        OUTSIDE_SUPPORT: 24,
        INCOMPATIBLE: 0,
        "budget_exhausted": 0,
        "invalid_input": 0,
        "execution_error": 0,
    }
    assert result["summary"]["one_gap_product_occurrences_by_graded_evidence_class"] == {
        EXACT_COMPLETE: 0,
        FAMILY_ALL_CURRENT: 177,
        FAMILY_PARTIAL_CURRENT: 92,
        FAMILY_NO_CURRENT: 246,
        MISSING_KNOWLEDGE: 287,
        OUTSIDE_SUPPORT: 95,
        INCOMPATIBLE: 0,
        "budget_exhausted": 0,
        "invalid_input": 0,
        "execution_error": 0,
    }
    assert result["adjudication"] == {
        "graded_family_evidence_audit_completed": True,
        "family_projection_promoted_to_exact": False,
        "family_scope_qualification_performed": False,
        "scalar_synthesis_value_defined": False,
        "synthesis_success_probability_defined": False,
        "synthesis_guidance_run": False,
        "prospective_candidate_selection_changed": False,
        "candidate_lock_authorized": False,
        "holdout_revealed": False,
    }

    rows = _rows(ledger)
    assert len(rows) == 1919
    projected = [row for row in rows if row["program_family"]]
    assert len(projected) == 209
    assert {row["exact_assessment_outcome"] for row in projected} == {MISSING_KNOWLEDGE}
    assert all(row["exact_scope_still_required"] == "True" for row in projected)
    assert all(row["exact_route_complete"] == "False" for row in projected)
    assert {row["family_scope_status"] for row in projected} == {"not_qualified_by_this_audit"}
    assert not ({"scalar_value", "success_probability"} & set(rows[0]))


def test_config_pins_every_frozen_input() -> None:
    import json

    config = json.loads(CONFIG.read_text())
    for specification in config["inputs"].values():
        assert sha256_file(REPO / specification["path"]) == specification["sha256"]
