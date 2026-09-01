from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.phase1.product_l1.evaluation.ugi_group_balanced_program_prior_comparison import (
    UgiGroupBalancedProgramPriorComparisonError,
    _passes_promotion,
    _runtime,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/ugi_group_balanced_program_prior_comparison_seed0_v1.json"


def test_group_balanced_comparison_config_preserves_scientific_contract() -> None:
    config = json.loads(CONFIG.read_text())
    runtime = _runtime(config, profile="smoke", device="cpu")

    assert runtime["program_count"] == 8
    assert config["policy"]["component_identity_conditioning"] is False
    assert config["policy"]["repairs_or_retries"] is False
    assert config["policy"]["training_calls"] == 0


def test_group_balanced_comparison_rejects_relaxed_retry_policy() -> None:
    config = json.loads(CONFIG.read_text())
    changed = copy.deepcopy(config)
    changed["policy"]["repairs_or_retries"] = True

    with pytest.raises(UgiGroupBalancedProgramPriorComparisonError, match="policy changed"):
        _runtime(changed, profile="smoke", device="cpu")


def test_promotion_requires_realism_and_all_noninferiority_checks() -> None:
    config = json.loads(CONFIG.read_text())
    gate = config["promotion_gate"]
    metrics = {
        "effective_component_count": 25.0,
        "exact_l1_yield_per_attempt": 0.95,
        "local_support_qualified_exact_l1_yield_per_attempt": 0.94,
        "mean_pairwise_ecfp4_distance": 0.42,
        "realism_c2st_auc": 0.90,
        "unique_exact_l1_products_per_attempt": 0.80,
        "valid_fraction_per_attempt": 0.98,
    }
    deltas = {
        "exact_l1_yield_per_attempt": 0.0,
        "local_support_qualified_exact_l1_yield_per_attempt": 0.0,
        "mean_pairwise_ecfp4_distance": 0.0,
        "realism_c2st_auc": -0.03,
        "unique_exact_l1_products_per_attempt": 0.0,
        "valid_fraction_per_attempt": 0.0,
    }

    passes, checks = _passes_promotion(deltas, metrics, metrics, gate)
    assert passes is True
    assert checks["effective_component_count_retained_ratio"] == pytest.approx(1.0)

    failing = dict(deltas)
    failing["realism_c2st_auc"] = -0.019
    passes, checks = _passes_promotion(failing, metrics, metrics, gate)
    assert passes is False
    assert checks["realism_improved"] is False


def test_promotion_rejects_effective_component_collapse() -> None:
    config = json.loads(CONFIG.read_text())
    gate = config["promotion_gate"]
    control = {
        "effective_component_count": 25.0,
        "exact_l1_yield_per_attempt": 0.95,
        "local_support_qualified_exact_l1_yield_per_attempt": 0.94,
        "mean_pairwise_ecfp4_distance": 0.42,
        "realism_c2st_auc": 0.90,
        "unique_exact_l1_products_per_attempt": 0.80,
        "valid_fraction_per_attempt": 0.98,
    }
    treatment = dict(control, effective_component_count=19.0, realism_c2st_auc=0.87)
    deltas = {
        key: treatment[key] - control[key] for key in control if key != "effective_component_count"
    }

    passes, checks = _passes_promotion(deltas, control, treatment, gate)
    assert passes is False
    assert checks["effective_component_count_retained_ratio"] == pytest.approx(0.76)
    assert checks["effective_component_count_preserved"] is False
