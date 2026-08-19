from __future__ import annotations

import base64
import json

import numpy as np
import pytest
import torch

from forge.potency.ugi_dynamic_controller_analysis import (
    BinomialRidge,
    Observation,
    UgiDynamicControllerAnalysisError,
    controller_decision,
    decode_program_b64,
    morphology_features,
    partial_state_features,
    prediction_metrics,
    stable_group_folds,
)


def _program() -> dict[str, tuple[int, int, int]]:
    return {
        "node_counts": (2, 3, 2),
        "junction_budgets": (1, 0, 0),
        "cycle_ranks": (1, 0, 0),
        "attachment_counts": (1, 1, 1),
    }


def test_decode_program_and_m0_have_exact_twelve_coordinates() -> None:
    payload = json.dumps(
        {"schema_version": "forge.ugi_morphology_program_bytes.v1", "program": _program()},
        sort_keys=True,
    ).encode()
    decoded = decode_program_b64(base64.b64encode(payload).decode())
    names, features = morphology_features(decoded)
    assert decoded == _program()
    assert len(names) == len(features) == 12
    assert names[0] == "amine_node_counts"
    assert names[-1] == "isocyanide_attachment_counts"


def test_partial_features_use_categorical_channels_and_role_blocks() -> None:
    channels = {
        "decoration_anchors": torch.tensor([[0, 2]]),
        "decoration_atoms": torch.tensor([[3, 4]]),
        "decoration_bonds": torch.tensor([[0, 1]]),
        "nodes": torch.tensor([[0, 3, 4, 0, 2, 1, 0]]),
        "offspring": torch.tensor([[1, 0, 1, 1, 0, 1, 0]]),
        "parent_bonds": torch.tensor([[0, 1, 0, 1, 0, 0, 1]]),
    }
    names, features = partial_state_features(_program(), channels, checkpoint=4, sample_steps=8)
    assert len(names) == len(features)
    assert np.all(np.isfinite(features))
    assert features[names.index("checkpoint_is_4")] == 1.0
    assert features[names.index("checkpoint_is_2")] == 0.0
    assert features[names.index("active_decoration_fraction")] == 0.5
    assert "seed" not in " ".join(names)
    assert "terminal" not in " ".join(names)


def test_group_folds_never_split_duplicate_program_hashes() -> None:
    groups = ["a", "b", "a", "c", "b", "d", "e"]
    folds = stable_group_folds(groups, folds=3, salt="test")
    assert folds[0] == folds[2]
    assert folds[1] == folds[4]
    assert len(set(folds.tolist())) == 3


def test_binomial_ridge_learns_monotone_signal() -> None:
    features = np.arange(12, dtype=float)[:, None]
    trials = np.full(12, 4.0)
    successes = np.asarray([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=float)
    model = BinomialRidge(l2=0.1).fit(features, successes, trials)
    predictions = model.predict(features)
    assert predictions[-1] > predictions[0]
    assert np.all((predictions > 0) & (predictions < 1))


def test_binomial_metrics_count_trials_not_fractional_rows() -> None:
    observations = [
        Observation("a", "g1", (0.0,), 1, 4, (0, 1, 2, 3), 2),
        Observation("b", "g2", (1.0,), 3, 4, (4, 5, 6, 7), 2),
    ]
    metrics = prediction_metrics(observations, np.asarray([0.25, 0.75]))
    assert metrics["trials"] == 8
    assert metrics["successes"] == 4
    assert metrics["prevalence"] == 0.5
    assert metrics["brier"] == pytest.approx(0.1875)


def test_controller_defaults_to_terminal_screening_when_gates_fail() -> None:
    base = {
        "metrics": {"auroc": 0.5, "calibration_slope": 0.0},
        "allocation": {
            "selected_program_group_fraction": 0.25,
            "support_rate_relative_improvement": 0.0,
            "diverse_yield_relative_improvement": 0.0,
            "selected_support_rate": 0.04,
        },
        "brier_difference_bootstrap": {"ci95_lower": -0.01},
    }
    m0 = {
        **base,
        "brier_relative_reduction": 0.0,
        "average_precision_lift": 1.0,
    }
    m1 = {
        **base,
        "brier_relative_reduction_over_m0": 0.0,
        "average_precision_lift_over_m0": 0.0,
        "support_yield_relative_improvement_over_m0": 0.0,
    }
    gates = {
        "m0_minimum_brier_relative_reduction": 0.05,
        "m0_minimum_average_precision_lift": 1.25,
        "minimum_auroc": 0.6,
        "minimum_calibration_slope": 0.5,
        "maximum_calibration_slope": 1.5,
        "minimum_program_coverage": 0.25,
        "m0_minimum_support_yield_relative_improvement": 0.25,
        "minimum_diverse_yield_relative_improvement": 0.15,
        "m1_minimum_brier_relative_reduction_over_m0": 0.1,
        "m1_minimum_average_precision_lift_over_m0": 0.2,
        "m1_minimum_support_yield_relative_improvement_over_m0": 0.2,
        "simplicity_tolerance": 0.1,
    }
    decision = controller_decision(m0, m1, gates=gates)
    assert decision["selected_next_controller"] == "plain_terminal_screening"
    assert decision["smc_execution_authorized"] is False
    assert decision["potency_guidance_authorized"] is False


def test_bad_checkpoint_channels_fail_closed() -> None:
    with pytest.raises(UgiDynamicControllerAnalysisError):
        partial_state_features(_program(), {}, checkpoint=2, sample_steps=8)
