from __future__ import annotations

from pathlib import Path

from experiments.phase1.synthesis_guidance.schedule.ugi_morphology_proposal_challenger_schedule import (
    build_morphology_proposal_challenger_schedule,
)

REPO = Path(__file__).resolve().parents[1]


def test_frozen_challenger_schedule_is_support_preserving() -> None:
    result, schedule = build_morphology_proposal_challenger_schedule(
        REPO,
        REPO / "configs/model/phase1_ugi_morphology_proposal_challenger_schedule_v1.json",
    )
    assert result["proposal"]["candidate_id"] == "rho-0.50_power-1.50"
    assert result["proposal"]["all_programs_positive"] is True
    assert result["decision"]["production_replacement_authorized"] is False
    assert len(schedule["records"]) == 3072
