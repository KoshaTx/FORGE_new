from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.phase1.multireaction.ugi_amine_semantic_joint_support_adjudication import (
    UgiAmineSemanticJointSupportAdjudicationError,
    _validate_comparison,
    _validate_config,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO
    / "configs/multireaction/ugi_amine_semantic_joint_support_adjudication_seed0_v2.json"
)


def _comparison() -> dict[str, object]:
    sampling = {
        "fixed_state_failures": 0,
        "repairs": {},
        "ugi_amine_semantic_joint_support_applied": True,
    }
    return {
        "schema_version": "forge.ugi_amine_semantic_program_comparison_result.v1",
        "status": "complete",
        "profile": "full",
        "programs_per_method": 3072,
        "methods": {
            "count_only": {"sampling_summary": sampling},
            "amine_semantic": {"sampling_summary": sampling},
        },
        "structural_checks": {
            "attempt_denominator_matched": True,
            "component_identity_conditioning_absent": True,
            "no_repairs_or_retries": True,
        },
    }


def test_joint_support_adjudication_freezes_the_measured_ugi_panel() -> None:
    config = _validate_config(json.loads(CONFIG.read_text()))

    assert config["metric_panel"]["primary_metrics"] == [
        "role_normalized_wasserstein",
        "role_energy_distance",
        "role_rbf_mmd2",
    ]
    assert config["metric_panel"]["weightings"] == [
        "attempt_weighted",
        "unique_product_weighted",
    ]
    assert config["arms"]["count_only"]["method_id"] == "forge_seed0_count_only_ugi_program"
    assert (
        config["arms"]["amine_semantic"]["method_id"]
        == "forge_seed0_amine_semantic_ugi_program"
    )
    assert config["promotion_gate"]["require_uniform_role_panel_improvement"] is True


def test_joint_support_adjudication_rejects_the_old_sequential_decoder() -> None:
    comparison = _comparison()
    old = copy.deepcopy(comparison)
    old["methods"]["amine_semantic"]["sampling_summary"][
        "ugi_amine_semantic_joint_support_applied"
    ] = False

    with pytest.raises(
        UgiAmineSemanticJointSupportAdjudicationError,
        match="joint-support no-repair decoder",
    ):
        _validate_comparison(old, expected_attempts=3072)

    assert _validate_comparison(comparison, expected_attempts=3072)["status"] == "complete"


def test_joint_support_adjudication_rejects_gate_relaxation() -> None:
    changed = json.loads(CONFIG.read_text())
    changed["promotion_gate"]["exact_l1_noninferiority_margin"] = 0.10

    with pytest.raises(
        UgiAmineSemanticJointSupportAdjudicationError,
        match="promotion gate changed",
    ):
        _validate_config(changed)
