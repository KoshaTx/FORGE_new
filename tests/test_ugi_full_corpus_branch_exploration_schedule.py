from __future__ import annotations

from pathlib import Path

from forge.product.ugi_full_corpus_branch_exploration_schedule import (
    build_full_corpus_branch_exploration_schedule,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_full_corpus_branch_exploration_schedule_v1.json"


def test_full_corpus_schedule_restores_realistic_branch_support() -> None:
    result, _, schedule = build_full_corpus_branch_exploration_schedule(REPO, CONFIG)

    assert result["support"]["role_state_counts"] == [106, 39, 40]
    assert result["support"]["programs"] == 165360
    assert result["support"]["previous_programs"] == 57190
    assert result["support"]["aldehyde_branch_node_count_support"] == list(range(7, 25))
    assert result["support"]["isocyanide_branch_node_count_support"] == [4, *range(6, 25)]
    assert result["support"]["required_aldehyde_reference_state"] == [16, 1, 0, 1]
    assert result["schedule"]["draws"] == 4096
    assert result["schedule"]["branch_class_draw_counts"] == {
        "aldehyde_origin_branched": 1284,
        "both_tail_origins_branched": 650,
        "isocyanide_origin_branched": 2162,
    }
    assert result["schedule"]["required_aldehyde_reference_state_draws"] > 0
    assert len(schedule["records"]) == 4096
    assert result["decision"]["generator_retraining_required"] is False
    assert result["decision"]["promoted_applicability_proposal_extrapolated_to_new_states"] is False


def test_full_corpus_schedule_is_deterministic() -> None:
    first, first_ledger, first_schedule = build_full_corpus_branch_exploration_schedule(
        REPO, CONFIG
    )
    second, second_ledger, second_schedule = build_full_corpus_branch_exploration_schedule(
        REPO, CONFIG
    )

    assert first["result_sha256"] == second["result_sha256"]
    assert first_ledger == second_ledger
    assert first_schedule == second_schedule
