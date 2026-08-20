from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.route.sources.l2_supervision_decision import (
    CONFIG_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    L2SupervisionDecisionError,
    build_l2_supervision_decision,
    load_config,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/m0_09_l2_supervision_decision.json"
RESULT = REPO / "results/m0_09/l2_supervision_decision.json"


def test_config_freezes_operational_closure_before_generation() -> None:
    config = load_config(CONFIG)
    assert config["schema_version"] == CONFIG_SCHEMA_VERSION
    thresholds = config["future_operational_closure_thresholds"]
    assert thresholds[
        "weighted_high_priority_motif_route_support_minimum"
    ] == pytest.approx(0.9)
    assert thresholds[
        "eligible_candidate_complete_route_fraction_minimum"
    ] == pytest.approx(0.9)
    assert thresholds[
        "eligible_candidate_missing_route_knowledge_fraction_maximum"
    ] == pytest.approx(0.05)
    assert thresholds["consecutive_saturation_rounds_required"] == 2
    assert config["claims_boundary"][
        "thresholds_apply_only_to_future_frozen_candidate_and_motif_sets"
    ]
    safeguards = config["future_training_safeguards"]
    assert safeguards["broad_corpus_replay_required"] is True
    assert safeguards["masked_l1_loss_for_unannotated_structures"] is True
    assert safeguards["ugi_only_finetuning_without_replay_allowed"] is False
    assert safeguards[
        "raw_route_model_likelihood_allowed_as_synthesis_value"
    ] is False


def test_decision_selects_hierarchical_joint_hybrid_architecture() -> None:
    result = build_l2_supervision_decision(CONFIG, REPO)
    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert result["gates"]["joint_product_l1"]["passed"] is True
    assert result["gates"]["hybrid_l2"]["passed"] is True
    assert (
        result["gates"][
            "monolithic_complete_route_decoder_necessary_conditions"
        ]["passed"]
        is False
    )
    decision = result["decision"]
    assert decision["recommended_architecture"] == (
        "hierarchical_joint_product_and_l1_with_hybrid_recursive_l2"
    )
    assert decision["monolithic_joint_product_complete_route_decoder_supported"] is False
    assert decision["l2_model_built"] is False
    assert result["evidence"]["cross_platform_transfer"][
        "lx_2024_direct_aldehyde_transfers"
    ] == 15
    assert result["inputs"][0]["input_id"] == "decision_config"
    assert result["future_training_safeguards"][
        "guidance_must_change_transition_probabilities_before_candidate_lock"
    ] is True


def test_input_hash_mismatch_fails_cleanly(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["inputs"]["paper_route_reviews"]["sha256"] = "0" * 64
    bad_config = tmp_path / "bad.json"
    bad_config.write_text(json.dumps(config))
    with pytest.raises(L2SupervisionDecisionError, match="hash mismatch"):
        build_l2_supervision_decision(bad_config, REPO)


def test_generated_decision_matches_frozen_policy() -> None:
    if not RESULT.exists():
        pytest.skip("L2 supervision decision has not been generated")
    result = json.loads(RESULT.read_text())
    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert result["decision"]["paper_mining_policy"] == "targeted_gap_driven_only"
    assert result["decision"]["current_operational_closure_achieved"] is False
