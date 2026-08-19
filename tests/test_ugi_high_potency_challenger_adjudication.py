from __future__ import annotations

from pathlib import Path

from forge.potency.ugi_high_potency_challenger_adjudication import (
    adjudicate_high_potency_challenger,
)

REPO = Path(__file__).resolve().parents[1]


def test_fresh_challenger_closes_potency_tilt_without_reopening_applicability() -> None:
    result = adjudicate_high_potency_challenger(
        signal_path=REPO
        / "results/phase1/ugi_morphology_high_potency_challenger_v1_retry1/result.json",
        proposal_path=REPO / "results/phase1/ugi_morphology_high_potency_proposal_v1/result.json",
        generation_path=REPO
        / "results/phase1/ugi_high_potency_challenger_terminal_generation_v1/result.json",
        ranking_path=REPO
        / "results/phase1/ugi_high_potency_challenger_continuous_ranking_v1/result.json",
    )
    gate = result["fresh_matched_terminal_gate"]
    assert gate["equal_oracle_budget"] == 38
    assert gate["authorized_tail_lane"]["support_unique_conservative_high"] == 27
    assert gate["authorized_tail_lane"]["potency_unique_conservative_high"] == 23
    assert gate["promotion_gate_passed"] is False
    assert result["decision"]["potency_morphology_tilting_branch_closed"] is True
    assert result["decision"]["additional_same_data_potency_proposals_authorized"] is False
