from pathlib import Path

from forge.value.ugi_synthesis_guidance_failure_audit import (
    build_synthesis_guidance_failure_audit,
)

REPO = Path(__file__).resolve().parents[1]


def test_frozen_failure_audit_rejects_synthesis_tilt() -> None:
    result = build_synthesis_guidance_failure_audit(
        REPO, REPO / "configs/route/phase1_ugi_synthesis_guidance_failure_audit_v1.json"
    )
    assert result["matched_result"]["absolute_difference"] == -1
    assert result["decision"]["midtrajectory_synthesis_tilting_promoted"] is False
    assert result["decision"]["proposal_engine_retained"] is True
