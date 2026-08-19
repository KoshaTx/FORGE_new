from pathlib import Path

from forge.value.synthesis_source_qualification import build_synthesis_source_qualification

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_synthesis_value_source_qualification_v1.json"


def test_source_qualification_owns_all_behavior_preserving_replays() -> None:
    result = build_synthesis_source_qualification(REPO, CONFIG)

    assert result["status"] == "behavior_preserving_synthesis_value_source_qualified"
    assert result["summary"] == {
        "synthesis_value_and_fresh_pool_replays": 4,
        "exact_route_replays": 2,
        "all_behavior_preserving": True,
    }
    assert result["adjudication"]["fresh_pool_vnext_may_advance"] is True
    assert result["adjudication"]["zero_guidance_requalification_may_advance"] is False
    assert result["adjudication"]["nonzero_guidance_authorized"] is False
