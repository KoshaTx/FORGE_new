from __future__ import annotations

from forge.route.aizynthfinder_single_step_recovery import (
    SCORE_SCHEMA_VERSION as AIZYNTH_SCORE_SCHEMA_VERSION,
)
from forge.route.graph2edits_single_step_recovery import (
    SCORE_SCHEMA_VERSION as GRAPH2EDITS_SCORE_SCHEMA_VERSION,
)
from forge.route.single_step_recovery_combination import combine_recovery_scores


def _score(schema: str, *, recovered: set[int]) -> dict[str, object]:
    return {
        "schema_version": schema,
        "score_sha256": schema.ljust(64, "0")[:64],
        "per_exact_target": [
            {
                "target_id": f"target-{index}",
                "primary_stratum": "held_reaction_families",
                "transformation": ("amine_formylation" if index >= 29 else "esterification"),
                "first_exact_recovery_rank": 1 if index in recovered else None,
            }
            for index in range(36)
        ],
    }


def test_hybrid_union_is_computed_without_route_promotion() -> None:
    result = combine_recovery_scores(
        _score(AIZYNTH_SCORE_SCHEMA_VERSION, recovered=set(range(19))),
        _score(GRAPH2EDITS_SCORE_SCHEMA_VERSION, recovered=set(range(29))),
    )
    assert result["summary"]["top_k_union_recovery"]["5"]["numerator"] == 29
    assert len(result["summary"]["unrecovered_by_either_engine"]) == 7
    assert result["interpretation"]["all_joint_misses_are_amine_formylation"] is True
    assert result["scientific_authority"]["may_enter_synthesis_value"] is False
