from __future__ import annotations

from forge.synthesis.engine.graph2edits_single_step_recovery import (
    RESULT_SCHEMA_VERSION,
    score_graph2edits_recovery,
)


def test_graph2edits_recovery_scoring_is_non_authoritative() -> None:
    rows = [
        {
            "target_id": f"exact-{index}",
            "primary_stratum": "held_reaction_families",
            "proposals": [{"rank": 1, "canonical_reactants": ["C"]}],
        }
        for index in range(36)
    ]
    rows.extend(
        {
            "target_id": f"other-{index}",
            "primary_stratum": "linear_aldehydes",
            "proposals": [],
        }
        for index in range(84)
    )
    truth = {
        "schema_version": "forge.single_step_benchmark_scoring_truth.v1",
        "lane_input": False,
        "records": [
            {
                "target_id": f"exact-{index}",
                "truth_kind": "documented_exact_forward_unique_reactant_multiset",
                "canonical_reactant_multiset": ["C"],
                "transformation": "fixture",
            }
            for index in range(36)
        ]
        + [
            {
                "target_id": f"adversarial-{index}",
                "truth_kind": "valid_connected_wrong_handle_role_swap_control",
            }
            for index in range(12)
        ],
    }
    score = score_graph2edits_recovery(
        proposal_result={
            "schema_version": RESULT_SCHEMA_VERSION,
            "hidden_truth_loaded_during_proposal_execution": False,
            "result_sha256": "a" * 64,
            "artifacts": {"proposal_ledger": {"sha256": "b" * 64}},
        },
        proposal_rows=rows,
        truth_payload=truth,
        truth_path="truth.json.gz",
        truth_sha256="c" * 64,
    )
    assert score["summary"]["known_route_top_k"]["1"]["fraction"] == 1.0
    assert score["scientific_authority"]["route_closure_authorized"] is False
    assert score["scientific_authority"]["may_enter_synthesis_value"] is False
