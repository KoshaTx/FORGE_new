from __future__ import annotations

from pathlib import Path

from forge.bio.ugi_potency_promotion_failure_audit import (
    audit_potency_promotion_failure,
)

REPO = Path(__file__).resolve().parents[1]


def test_frozen_potency_failure_is_attributed_without_promoting_tilt() -> None:
    result = audit_potency_promotion_failure(
        confirmatory_path=REPO / "results/phase1/ugi_morphology_potency_matched_v1/result.json",
        continuous_path=REPO
        / "results/phase1/ugi_continuous_novelty_matched_ranking_v1/result.json",
        signal_path=REPO / "results/phase1/ugi_morphology_potency_signal_v1/result.json",
    )
    diagnosis = result["diagnosis"]
    assert diagnosis["initial_exact_identity_policy_failure"]["support_eligible_terminals"] == 0
    assert diagnosis["authorized_familiar_head_new_tail_lane"]["support_eligible"] == 78
    assert diagnosis["authorized_familiar_head_new_tail_lane"]["potency_eligible"] == 76
    assert diagnosis["authorized_familiar_head_new_tail_lane"]["promotion_gate_passed"] is False
    assert result["decision"]["current_potency_tilting_promoted"] is False
    assert result["decision"]["future_promotion_remains_possible"] is True
