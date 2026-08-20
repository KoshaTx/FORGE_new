from __future__ import annotations

from pathlib import Path

from forge.route.engine.aizynthfinder_single_step_recovery import (
    PROPOSAL_RESULT_SCHEMA_VERSION,
    proposal_result,
    score_frozen_proposals,
)

REPO = Path(__file__).resolve().parents[1]


def _truth_records() -> list[dict[str, object]]:
    exact = [
        {
            "target_id": f"exact-{index}",
            "truth_kind": "documented_exact_forward_unique_reactant_multiset",
            "canonical_reactant_multiset": ["C"],
            "transformation": "fixture",
        }
        for index in range(36)
    ]
    adversarial = [
        {
            "target_id": f"adversarial-{index}",
            "truth_kind": "valid_connected_wrong_handle_role_swap_control",
        }
        for index in range(12)
    ]
    return exact + adversarial


def test_proposal_result_never_claims_route_authority() -> None:
    rows = [{"target_id": f"t-{index}", "proposals": []} for index in range(120)]
    result = proposal_result(
        config_path="config.json",
        config_sha256="a" * 64,
        target_path="targets.json.gz",
        target_sha256="b" * 64,
        runtime_path="runtime.json",
        runtime_sha256="c" * 64,
        ledger_path="ledger.jsonl.gz",
        ledger_sha256="d" * 64,
        rows=rows,
        maximum_proposals=20,
    )
    assert result["hidden_truth_loaded_during_proposal_execution"] is False
    assert result["scientific_authority"]["route_closure_authorized"] is False
    assert result["scientific_authority"]["may_enter_synthesis_value"] is False


def test_scoring_reports_exact_recovery_without_promoting_it() -> None:
    rows = []
    for index in range(36):
        rows.append(
            {
                "target_id": f"exact-{index}",
                "primary_stratum": "known_exact_l2_routes",
                "proposals": [
                    {"rank": 1, "canonical_reactants": ["CC"]},
                    {"rank": 2, "canonical_reactants": ["C"]},
                ],
            }
        )
    rows.extend(
        {
            "target_id": f"adversarial-{index}",
            "primary_stratum": "adversarial_incompatibles",
            "proposals": [],
        }
        for index in range(12)
    )
    proposal = {
        "schema_version": PROPOSAL_RESULT_SCHEMA_VERSION,
        "hidden_truth_loaded_during_proposal_execution": False,
        "result_sha256": "e" * 64,
        "artifacts": {"proposal_ledger": {"sha256": "f" * 64}},
    }
    score = score_frozen_proposals(
        proposal_result_payload=proposal,
        proposal_rows=rows,
        truth_payload={
            "schema_version": "forge.single_step_benchmark_scoring_truth.v1",
            "lane_input": False,
            "records": _truth_records(),
        },
        truth_path="truth.json.gz",
        truth_sha256="0" * 64,
    )
    assert score["summary"]["known_route_top_k"]["1"]["fraction"] == 0.0
    assert score["summary"]["known_route_top_k"]["5"]["fraction"] == 1.0
    assert score["scientific_authority"]["may_enter_synthesis_value"] is False
