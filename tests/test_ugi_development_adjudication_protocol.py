from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.phase1.multireaction.ugi_development_adjudication_protocol import (
    UgiDevelopmentAdjudicationProtocolError,
    _validate_bound_inputs,
    _validate_protocol,
)
from forge.core.hashing import resolve_pin

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPO
    / "configs/multireaction/ugi_atom_local_chemistry_preflight_adjudication_protocol_seed0_v1.json"
)
TRUST_REGION_PROTOCOL = (
    REPO / "configs/multireaction/"
    "ugi_atom_trust_region_preflight_adjudication_protocol_seed0_v1.json"
)
CONTEXT_SUPPORT_PROTOCOL = (
    REPO / "configs/multireaction/ugi_context_support_preflight_adjudication_protocol_seed0_v1.json"
)
WHOLE_HEAD_PROTOCOL = (
    REPO
    / "configs/multireaction/ugi_whole_head_support_preflight_adjudication_protocol_seed0_v1.json"
)
BINARY_WHOLE_HEAD_PROTOCOL = (
    REPO
    / "configs/multireaction/"
    "ugi_binary_whole_head_support_preflight_adjudication_protocol_seed0_v1.json"
)
WHOLE_HEAD_TRAJECTORY_TRUST_REGION_PROTOCOL = (
    REPO
    / "configs/multireaction/"
    "ugi_whole_head_trajectory_trust_region_preflight_adjudication_protocol_seed0_v1.json"
)


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


def _trust_region_protocol() -> dict:
    return json.loads(TRUST_REGION_PROTOCOL.read_text())


def _context_support_protocol() -> dict:
    return json.loads(CONTEXT_SUPPORT_PROTOCOL.read_text())


def _whole_head_protocol() -> dict:
    return json.loads(WHOLE_HEAD_PROTOCOL.read_text())


def _binary_whole_head_protocol() -> dict:
    return json.loads(BINARY_WHOLE_HEAD_PROTOCOL.read_text())


def _whole_head_trajectory_trust_region_protocol() -> dict:
    return json.loads(WHOLE_HEAD_TRAJECTORY_TRUST_REGION_PROTOCOL.read_text())


def test_atom_local_preflight_protocol_is_output_blind_and_fully_gated() -> None:
    protocol = _validate_protocol(_protocol())
    inputs = {
        label: resolve_pin(pin, REPO, label=label) for label, pin in protocol["inputs"].items()
    }

    _validate_bound_inputs(protocol, inputs)

    assert protocol["expected_attempts"] == 256
    assert protocol["policy"] == {
        "candidate_selection": False,
        "component_identity_conditioning": False,
        "include_invalid_or_failed_attempts_in_visual_review": True,
        "indices_selected_before_generation": True,
        "repair_or_retry": False,
        "route_or_oracle_calls": 0,
        "training_calls": 0,
    }
    assert protocol["arms"]["atom_local_chemistry_mog"]["method_id"] == (
        "forge_seed0_mog_atom_local_chemistry_complete_semantic_ugi_program"
    )
    assert len(protocol["visual_review"]["attempt_indices"]) == 24


def test_atom_local_preflight_protocol_rejects_post_outcome_index_changes() -> None:
    changed = copy.deepcopy(_protocol())
    changed["visual_review"]["attempt_indices"][0] = 12

    with pytest.raises(UgiDevelopmentAdjudicationProtocolError, match="output-blind draw"):
        _validate_protocol(changed)


def test_atom_local_preflight_protocol_rejects_weakened_exact_l1_gate() -> None:
    changed = copy.deepcopy(_protocol())
    changed["promotion_gate"]["exact_l1_absolute_minimum"] = 0.90

    with pytest.raises(UgiDevelopmentAdjudicationProtocolError, match="promotion gate"):
        _validate_protocol(changed)


def test_atom_trust_region_preflight_protocol_binds_exact_intervention() -> None:
    protocol = _validate_protocol(_trust_region_protocol())
    inputs = {
        label: resolve_pin(pin, REPO, label=label) for label, pin in protocol["inputs"].items()
    }

    _validate_bound_inputs(protocol, inputs)

    comparison_config = json.loads(inputs["comparison_config"].read_text())
    assert comparison_config["semantic_guidance"][
        "local_chemistry_atom_total_variation_radius"
    ] == pytest.approx(0.05)
    assert protocol["arms"]["atom_trust_region_mog"]["method_id"] == (
        "forge_seed0_mog_atom_trust_region_complete_semantic_ugi_program"
    )
    assert len(protocol["visual_review"]["attempt_indices"]) == 24


def test_context_support_protocol_binds_novelty_neutral_all_role_intervention() -> None:
    protocol = _validate_protocol(_context_support_protocol())
    inputs = {
        label: resolve_pin(pin, REPO, label=label) for label, pin in protocol["inputs"].items()
    }

    policy = _validate_bound_inputs(protocol, inputs)

    assert policy.local_chemistry_score_mode == "support_tier"
    assert policy.uses_local_chemistry_atoms is True
    assert policy.uses_local_chemistry_bonds is True
    assert protocol["arms"]["context_support_mog"]["method_id"] == (
        "forge_seed0_mog_context_support_complete_semantic_ugi_program"
    )
    assert len(protocol["visual_review"]["attempt_indices"]) == 24


def test_whole_head_protocol_binds_one_completed_head_score() -> None:
    protocol = _validate_protocol(_whole_head_protocol())
    inputs = {
        label: resolve_pin(pin, REPO, label=label) for label, pin in protocol["inputs"].items()
    }

    policy = _validate_bound_inputs(protocol, inputs)

    assert policy.local_chemistry_score_mode == "whole_head_support_tier"
    assert policy.uses_coordinate_local_chemistry_atoms is False
    assert policy.uses_local_chemistry_bonds is True
    assert protocol["visual_review"]["criteria"] == [
        "head_arrangement_preference",
        "overall_lipid_plausibility_preference",
        "unsupported_ring_or_heteroatom_pathology",
    ]


def test_binary_whole_head_protocol_binds_equal_supported_arrangement_scores() -> None:
    protocol = _validate_protocol(_binary_whole_head_protocol())
    inputs = {
        label: resolve_pin(pin, REPO, label=label) for label, pin in protocol["inputs"].items()
    }

    policy = _validate_bound_inputs(protocol, inputs)

    assert policy.local_chemistry_score_mode == "whole_head_support_binary"
    assert policy.uses_coordinate_local_chemistry_atoms is False
    assert protocol["arms"]["binary_whole_head_support_mog"]["method_id"] == (
        "forge_seed0_mog_binary_whole_head_support_complete_semantic_ugi_program"
    )
    assert len(protocol["visual_review"]["attempt_indices"]) == 24


def test_whole_head_trajectory_trust_region_protocol_binds_component_level_radius() -> None:
    protocol = _validate_protocol(_whole_head_trajectory_trust_region_protocol())
    inputs = {
        label: resolve_pin(pin, REPO, label=label) for label, pin in protocol["inputs"].items()
    }

    policy = _validate_bound_inputs(protocol, inputs)

    assert policy.local_chemistry_score_mode == "whole_head_support_binary"
    assert policy.whole_head_total_variation_radius == pytest.approx(0.05)
    assert protocol["arms"]["whole_head_trajectory_trust_region_mog"]["method_id"] == (
        "forge_seed0_mog_binary_whole_head_tv050_complete_semantic_ugi_program"
    )
    assert len(protocol["visual_review"]["attempt_indices"]) == 24
