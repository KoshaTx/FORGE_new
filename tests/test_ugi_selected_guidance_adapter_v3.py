from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import forge.design.flow.ugi_selected_guidance_adapter_v3 as adapter_v3
from forge.design.flow.ugi_selected_guidance_adapter import _sha256_payload
from forge.design.flow.ugi_selected_guidance_adapter_v3 import (
    EQUIVALENCE_RESULT_FILE_SHA256,
    EQUIVALENCE_RESULT_LOGICAL_SHA256,
    EQUIVALENCE_ROWS_FILE_SHA256,
    EQUIVALENCE_ROWS_LOGICAL_SHA256,
    EXPECTED_BASE_V2_ADAPTER_IDENTITY_SHA256,
    EXPECTED_V2_GENERATOR_SOURCE_SHA256,
    EXPECTED_V2_GUIDANCE_SOURCE_SHA256,
    UgiSelectedGuidanceAdapterV3Error,
    _bind_equivalence_receipt,
    _validate_comparison_rows,
    build_selected_model_restartable_guidance_lane_v3,
    load_selected_v2_equivalence_binding,
)
from forge.design.sampling.ugi_selected_restartable_generator_v2 import (
    _PENDING_EQUIVALENCE_SHA256,
)

REPO = Path(__file__).resolve().parents[1]
ROWS = REPO / adapter_v3.EQUIVALENCE_ROWS_PATH
V2_GENERATOR_SOURCE = REPO / adapter_v3.V2_GENERATOR_SOURCE_PATH
V2_GUIDANCE_SOURCE = REPO / adapter_v3.V2_GUIDANCE_SOURCE_PATH


@pytest.fixture(scope="module")
def lane():
    return build_selected_model_restartable_guidance_lane_v3(REPO)


def test_equivalence_binding_validates_receipt_rows_and_frozen_sources() -> None:
    binding = load_selected_v2_equivalence_binding(REPO)

    assert binding.result_file_sha256 == EQUIVALENCE_RESULT_FILE_SHA256
    assert binding.result_logical_sha256 == EQUIVALENCE_RESULT_LOGICAL_SHA256
    assert binding.rows_file_sha256 == EQUIVALENCE_ROWS_FILE_SHA256
    assert binding.rows_logical_sha256 == EQUIVALENCE_ROWS_LOGICAL_SHA256
    assert binding.base_v2_adapter_identity_sha256 == (EXPECTED_BASE_V2_ADAPTER_IDENTITY_SHA256)
    assert binding.v2_generator_source_sha256 == EXPECTED_V2_GENERATOR_SOURCE_SHA256
    assert binding.v2_guidance_source_sha256 == EXPECTED_V2_GUIDANCE_SOURCE_SHA256


def test_comparison_rows_fail_semantically_even_under_rehashed_payload() -> None:
    rows = json.loads(ROWS.read_text())
    rows[0]["equal"] = False

    with pytest.raises(UgiSelectedGuidanceAdapterV3Error, match="equality status"):
        _validate_comparison_rows(rows, expected_logical_sha256=_sha256_payload(rows))


def test_comparison_rows_require_complete_canonical_coverage() -> None:
    rows = json.loads(ROWS.read_text())
    rows[0], rows[1] = rows[1], rows[0]

    with pytest.raises(UgiSelectedGuidanceAdapterV3Error, match="ordering or coverage"):
        _validate_comparison_rows(rows, expected_logical_sha256=_sha256_payload(rows))


def test_v3_replaces_only_pending_receipt_token_and_binds_complete_identity(lane) -> None:
    bound = lane.bound_closure_identity
    pending = replace(
        bound,
        restartable_equivalence_receipt_sha256=_PENDING_EQUIVALENCE_SHA256,
    )
    changed = {key for key in pending.to_dict() if pending.to_dict()[key] != bound.to_dict()[key]}

    assert changed == {"restartable_equivalence_receipt_sha256"}
    assert bound.restartable_equivalence_receipt_sha256 == EQUIVALENCE_RESULT_FILE_SHA256
    assert lane.selected_lane.adapter.generate_locked_terminal is lane.selected_lane.callback
    assert lane.base_v2_adapter_identity_sha256 == EXPECTED_BASE_V2_ADAPTER_IDENTITY_SHA256
    expected_identity = _sha256_payload(
        {
            "schema_version": adapter_v3.SELECTED_GUIDANCE_ADAPTER_V3_SCHEMA_VERSION,
            "base_v2_adapter_identity_sha256": lane.base_v2_adapter_identity_sha256,
            "bound_closure_identity": bound.to_dict(),
            "bound_closure_identity_sha256": lane.bound_closure_identity_sha256,
            "equivalence_binding": lane.equivalence_binding.identity_dict(),
            "selected_bindings_sha256": lane.selected_lane.bindings.canonical_sha256,
            "v3_adapter_source_sha256": lane._v3_source_sha256,
        }
    )
    assert lane.adapter_identity_sha256 == expected_identity


def test_pending_token_replacement_fails_on_an_already_bound_lane(lane) -> None:
    with pytest.raises(UgiSelectedGuidanceAdapterV3Error, match="exact pending token"):
        _bind_equivalence_receipt(lane.selected_lane, lane.equivalence_binding)


def test_runtime_guard_rehashes_bound_receipt(monkeypatch: pytest.MonkeyPatch, lane) -> None:
    original = adapter_v3._file_sha256

    def changed_receipt(path: Path) -> str:
        if path == lane.equivalence_binding.result_path:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(adapter_v3, "_file_sha256", changed_receipt)
    with pytest.raises(UgiSelectedGuidanceAdapterV3Error, match="receipt changed"):
        lane._require_bound_artifacts_unchanged()


def test_initialize_and_state_operations_enter_runtime_guard(
    monkeypatch: pytest.MonkeyPatch,
    lane,
) -> None:
    def reject(_self) -> None:
        raise UgiSelectedGuidanceAdapterV3Error("sentinel v3 runtime guard")

    monkeypatch.setattr(type(lane), "_require_bound_artifacts_unchanged", reject)
    with pytest.raises(UgiSelectedGuidanceAdapterV3Error, match="sentinel"):
        lane.initialize((), seed=0, particle_seeds=(), device="cpu")
    with pytest.raises(UgiSelectedGuidanceAdapterV3Error, match="sentinel"):
        lane._require_current_state(object())


def test_frozen_v2_sources_remain_byte_identical() -> None:
    assert adapter_v3._file_sha256(V2_GENERATOR_SOURCE) == (EXPECTED_V2_GENERATOR_SOURCE_SHA256)
    assert adapter_v3._file_sha256(V2_GUIDANCE_SOURCE) == EXPECTED_V2_GUIDANCE_SOURCE_SHA256
