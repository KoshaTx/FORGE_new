from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from forge.potency.oracle.oracle_graph_transfer_decision import (
    OracleGraphTransferDecisionError,
    adjudicate_transfer_lane,
    run_transfer_decision,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/bio/m0_07_oracle_graph_transfer_decision.json"
TRANSFER_RESULT_PATH = REPO / "results/m0_07/oracle_graph_transfer_result.json"
FREEZE_RESULT_PATH = REPO / "results/m0_07/oracle_freeze_result.json"


def _inputs() -> tuple[dict, dict, dict]:
    config = json.loads(CONFIG_PATH.read_text())
    transfer = json.loads(TRANSFER_RESULT_PATH.read_text())
    freeze = json.loads(FREEZE_RESULT_PATH.read_text())
    return config, transfer, freeze


def test_completed_transfer_lane_is_nonselected_and_cannot_guide() -> None:
    config, transfer, freeze = _inputs()
    result = adjudicate_transfer_lane(transfer, freeze, config)
    assert result["status"] == "label_free_r0_transfer_frozen_as_nonselected_diagnostic"
    assert result["selected_cross_representation_model"]["lane"] == "supervised_graph"
    assert {row["selection_rank"] for row in result["transfer_candidates"]} == {12, 14}
    assert all(row["equal_endpoint_test_r2"] < 0.0 for row in result["transfer_candidates"])
    assert result["decision"] == {
        "lane_selected_by_frozen_policy": False,
        "lane_role": "nonselected_diagnostic_pretraining_ablation",
        "guidance_oracle_authorized": False,
        "production_checkpoint_authorized": False,
        "silent_promotion_prohibited": True,
        "negative_outer_test_r2_used_for_selection": False,
        "negative_outer_test_r2_interpretation": (
            "post-selection evidence that the transfer lane did not improve held-component "
            "generalization"
        ),
        "reopen_requires": (
            "a new hash-pinned cross-representation freeze with complete held-component "
            "evidence and an explicit applicability-policy decision"
        ),
    }


def test_transfer_lane_cannot_silently_replace_the_frozen_winner() -> None:
    config, transfer, freeze = _inputs()
    tampered = copy.deepcopy(freeze)
    transfer_candidate = next(
        row for row in tampered["candidate_ranking"] if row["lane"] == "label_free_r0_transfer"
    )
    original_winner = next(
        row for row in tampered["candidate_ranking"] if row["selection_rank"] == 1
    )
    original_winner["selection_rank"] = transfer_candidate["selection_rank"]
    transfer_candidate["selection_rank"] = 1
    tampered["selected_model"] = transfer_candidate
    with pytest.raises(
        OracleGraphTransferDecisionError,
        match="selected cross-representation model changed",
    ):
        adjudicate_transfer_lane(transfer, tampered, config)


def test_transfer_outer_test_metrics_cannot_change_selection() -> None:
    config, transfer, freeze = _inputs()
    tampered = copy.deepcopy(freeze)
    for row in tampered["candidate_ranking"]:
        if row["lane"] == "label_free_r0_transfer":
            row["equal_endpoint_mean_equal_scheme_test_r2"] = 100.0
    with pytest.raises(
        OracleGraphTransferDecisionError,
        match="differs between matrix and freeze",
    ):
        adjudicate_transfer_lane(transfer, tampered, config)
    assert (
        freeze["selection_contract"]["outer_test_metrics_used_for_architecture_selection"] is False
    )


def test_end_to_end_transfer_decision_is_hash_pinned(tmp_path: Path) -> None:
    output = tmp_path / "decision.json"
    result = run_transfer_decision(CONFIG_PATH, output, REPO)
    persisted = json.loads(output.read_text())
    assert persisted == result
    assert result["matrix_completeness"]["fits_rerun_for_adjudication"] is False
    assert result["inputs"]["transfer_matrix_result"]["sha256"] == (
        "c6661597e2683a99046bed341eb70604deba0e0799db96bd664967c63c8e9862"
    )
    assert result["inputs"]["oracle_freeze_result"]["sha256"] == (
        "8476be6e013e12715bdb55a7ef6d73162b01010f7b560297c6e8406028cda338"
    )
