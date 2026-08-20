from __future__ import annotations

from pathlib import Path

from forge.design.guidance.ugi_promoted_morphology_proposal import (
    build_promoted_morphology_proposal,
)

REPO = Path(__file__).resolve().parents[1]


def test_promoted_proposal_preserves_complete_support() -> None:
    result, _ = build_promoted_morphology_proposal(
        REPO, REPO / "configs/model/phase1_ugi_promoted_morphology_proposal_v1.json"
    )
    assert result["support"]["programs"] == 57190
    assert result["support"]["all_proposal_probabilities_positive"] is True
    assert result["decision"]["applicability_proposal_promoted"] is True
    assert result["decision"]["potency_guidance_authorized"] is False
