from pathlib import Path

from experiments.phase1.synthesis_guidance.guidance.ugi_source_guidance_prerequisite_requalification import (
    build_source_guidance_prerequisite_requalification,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_source_guidance_prerequisite_requalification_v1.json"


def test_requalifies_only_zero_guidance_prerequisites_against_v6() -> None:
    result = build_source_guidance_prerequisite_requalification(REPO, CONFIG)

    assert result["status"] == "v6_guidance_prerequisites_requalified_nonzero_guidance_blocked"
    assert result["prerequisites"]["fresh_pool_v6_source_qualified"] is True
    assert result["prerequisites"]["zero_guidance_prerequisite_requalified_against_v6"] is True
    assert result["prerequisites"]["zero_guidance_paired_supports"] == 12
    assert result["prerequisites"]["matched_budget_prerequisite_requalified_against_v6"] is True
    assert result["open_gates"]["scalar_value_policy_frozen"] is False
    assert result["open_gates"]["production_cumulative_source_vnext_runtime_qualified"] is False
    assert result["adjudication"]["nonzero_guidance_authorized"] is False
    assert result["adjudication"]["candidate_selection_authorized"] is False
    assert result["adjudication"]["sealed_holdout_accessed"] is False


def test_l3_statement_is_frozen_assessment_only() -> None:
    result = build_source_guidance_prerequisite_requalification(REPO, CONFIG)

    assert result["assessment"]["l3_current_at_frozen_assessment_only"] is True
    assert result["open_gates"]["l3_refresh_required_before_future_production_execution"] is True
