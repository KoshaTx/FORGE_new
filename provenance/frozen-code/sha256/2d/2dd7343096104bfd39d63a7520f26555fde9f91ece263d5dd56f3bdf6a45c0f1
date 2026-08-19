from pathlib import Path

from forge.value.synthesis_source_exact_route_replay import (
    build_synthesis_source_exact_route_replay,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_synthesis_source_exact_route_replay_v1.json"


def test_exact_c18_and_c16_source_replays_are_behavior_preserving() -> None:
    result = build_synthesis_source_exact_route_replay(REPO, CONFIG)

    assert result["status"] == "exact_route_source_replays_behavior_preserving"
    assert result["summary"] == {
        "replays": 2,
        "byte_identical_step_ledgers": 2,
        "byte_identical_assessments": 2,
        "semantically_identical_summaries": 2,
        "behavior_preserving": True,
    }
    assert result["adjudication"]["historical_artifacts_rewritten"] is False
    assert result["adjudication"]["sealed_holdout_accessed"] is False
    assert all(item["behavior_preserving"] for item in result["replays"].values())
