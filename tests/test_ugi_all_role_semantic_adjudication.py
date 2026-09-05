from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments._runtime.errors import StageError
from experiments.phase1.multireaction.ugi_all_role_semantic_adjudication import (
    UgiAllRoleSemanticAdjudicationError,
    _validate_comparison,
    _validate_config,
)
from experiments.phase1.product_l1.stages import (
    _require_successful_ugi_all_role_semantic_quantitative_preflight,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/ugi_all_role_semantic_adjudication_seed0_v1.json"
PREFLIGHT_CONFIG = (
    REPO
    / "configs/multireaction/ugi_complete_semantic_preflight_adjudication_seed0_v1.json"
)


def test_all_role_adjudication_freezes_the_full_realism_and_retention_gate() -> None:
    config = _validate_config(json.loads(CONFIG.read_text()))

    assert config["expected_attempts"] == 3072
    assert config["promotion_gate"]["exact_l1_absolute_minimum"] == 0.95
    assert config["promotion_gate"]["unique_exact_l1_minimum_retained_ratio"] == 0.95
    assert config["promotion_gate"]["effective_component_count_minimum_retained_ratio"] == 0.95
    assert config["visual_review"]["required_after_quantitative_gate"] is True


def test_all_role_adjudication_rejects_dropped_realism_metric() -> None:
    config = json.loads(CONFIG.read_text())
    changed = copy.deepcopy(config)
    changed["metric_panel"]["primary_metrics"].remove("role_rbf_mmd2")

    with pytest.raises(ValueError):
        _validate_config(changed)


def test_all_role_adjudication_rejects_unblinded_visual_review() -> None:
    config = json.loads(CONFIG.read_text())
    changed = copy.deepcopy(config)
    changed["visual_review"]["blinded"] = False

    with pytest.raises(UgiAllRoleSemanticAdjudicationError, match="visual-review"):
        _validate_config(changed)


def test_complete_semantic_preflight_freezes_the_same_full_gate_at_256_attempts() -> None:
    config = _validate_config(json.loads(PREFLIGHT_CONFIG.read_text()))

    assert config["expected_attempts"] == 256
    assert config["promotion_gate"] == _validate_config(
        json.loads(CONFIG.read_text())
    )["promotion_gate"]
    assert config["metric_panel"] == _validate_config(json.loads(CONFIG.read_text()))[
        "metric_panel"
    ]


def _preflight_comparison() -> dict[str, object]:
    summary = {
        "fixed_state_failures": 0,
        "repairs": {},
        "ugi_all_role_semantic_joint_support_applied": True,
        "ugi_amine_semantic_joint_support_applied": True,
    }
    return {
        "schema_version": "forge.ugi_all_role_semantic_program_comparison_result.v1",
        "status": "complete",
        "profile": "h100_preflight",
        "programs_per_method": 256,
        "structural_checks": {"paired": True, "no_repair": True},
        "methods": {
            "amine_semantic": {"sampling_summary": summary},
            "all_role_semantic": {"sampling_summary": summary},
        },
    }


def test_preflight_comparison_validation_accepts_only_the_h100_profile() -> None:
    assert _validate_comparison(_preflight_comparison(), expected_attempts=256)[
        "profile"
    ] == "h100_preflight"

    changed = copy.deepcopy(_preflight_comparison())
    changed["profile"] = "full"
    with pytest.raises(UgiAllRoleSemanticAdjudicationError, match="inadmissible"):
        _validate_comparison(changed, expected_attempts=256)


def test_quantitative_preflight_guard_fails_closed_on_any_failed_gate() -> None:
    adjudication = {
        "schema_version": "forge.ugi_all_role_semantic_adjudication.v1",
        "status": "complete",
        "expected_attempts_per_arm": 256,
        "quantitative_decision": "pass_seed0_quantitative_gate",
        "promotion_checks": {
            "realism": True,
            "exact_l1": True,
            "effective_count": True,
        },
    }
    _require_successful_ugi_all_role_semantic_quantitative_preflight(
        adjudication,
        expected_attempts=256,
    )

    failed = copy.deepcopy(adjudication)
    failed["promotion_checks"]["effective_count"] = False
    with pytest.raises(StageError, match="quantitative preflight"):
        _require_successful_ugi_all_role_semantic_quantitative_preflight(
            failed,
            expected_attempts=256,
        )
