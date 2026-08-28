from __future__ import annotations

import pytest

from experiments.phase1.product_l1.evaluation.ugi_transformer_morphology_projection_comparison import (
    INPUT_LABELS,
    UgiTransformerMorphologyProjectionComparisonError,
    _program_rows_match,
    _projection_decision,
    _validate_config,
)


def _config() -> dict[str, object]:
    return {
        "schema_version": "forge.ugi_transformer_morphology_projection_comparison_config.v1",
        "inputs": {label: {} for label in INPUT_LABELS},
        "arms": {
            "reduced_projection": {
                "arm_id": "bl_core_constrained_repeat_aware",
                "checkpoint_step": 1700,
                "method_id": "forge_mixed_transformer_reduced_projection_seed0",
                "role_morphology_conditioning": False,
            },
            "full_role_morphology": {
                "arm_id": "full_role_morphology_transformer",
                "checkpoint_step": 1700,
                "method_id": "forge_mixed_transformer_full_role_morphology_seed0",
                "role_morphology_conditioning": True,
            },
        },
        "profiles": {
            "full": {
                "device": "cuda",
                "program_count": 3072,
                "sample_steps": 32,
                "batch_size": 32,
                "flow_seed": 20260825,
                "terminal_decoder_mode": "argmax",
                "terminal_decoder_seed": None,
                "terminal_decode_policy": "strict_valence_topology_argmax",
            }
        },
        "policy": {
            "candidate_selection": False,
            "flow_seed_matched": True,
            "method_blind_assessment": True,
            "negative_results_reported": True,
            "oracle_calls": 0,
            "paired_program_order": True,
            "repairs_or_retries": False,
            "route_calls": 0,
            "training_calls": 0,
        },
        "decision_rule": {
            "primary_metric": "exact_l1_yield_per_attempt",
            "absolute_margin": 0.02,
        },
    }


def test_projection_comparison_contract_is_exact_and_nontraining() -> None:
    runtime = _validate_config(_config(), profile="full")

    assert runtime["program_count"] == 3072
    assert runtime["terminal_decoder_seed"] is None


def test_projection_comparison_rejects_a_changed_arm() -> None:
    config = _config()
    config["arms"]["full_role_morphology"]["checkpoint_step"] = 1500

    with pytest.raises(
        UgiTransformerMorphologyProjectionComparisonError,
        match="arms changed",
    ):
        _validate_config(config, profile="full")


def test_projection_decision_uses_the_frozen_absolute_margin() -> None:
    assert (
        _projection_decision(0.021, margin=0.02)
        == "full_role_morphology_improves_exact_l1_under_seed0_gate"
    )
    assert _projection_decision(0.02, margin=0.02) == "no_material_seed0_exact_l1_change"
    assert (
        _projection_decision(-0.021, margin=0.02)
        == "reduced_projection_improves_exact_l1_under_seed0_gate"
    )


def test_paired_program_gate_checks_both_order_and_content() -> None:
    left = [
        {"pipeline_index": 0, "program": {"node_counts": [1, 2, 3]}},
        {"pipeline_index": 1, "program": {"node_counts": [2, 3, 4]}},
    ]
    assert _program_rows_match(left, [dict(row) for row in left])
    assert not _program_rows_match(left, list(reversed(left)))
