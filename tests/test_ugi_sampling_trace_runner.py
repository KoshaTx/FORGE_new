from copy import deepcopy
from pathlib import Path

import pytest

from experiments.phase1.multireaction.ugi_sampling_trace import (
    UgiSamplingTraceError,
    _authenticate_record,
    _compare_historical,
    _complete_decisions,
    _complete_endpoint,
    _read,
    _validate,
)

ROOT = Path(__file__).resolve().parents[1]


def test_final_config_authentication_accepts_receipt_metadata_and_detects_changes(tmp_path):
    from forge.core.hashing import PinError, pin_record

    path = tmp_path / "config.json"
    path.write_text('{"same": true}')
    record = pin_record(path, tmp_path)
    assert "bytes" in record
    assert _authenticate_record(record, tmp_path, label="config") == path
    path.write_text('{"same": false}')
    with pytest.raises(PinError, match="changed"):
        _authenticate_record(record, tmp_path, label="config")


def test_replay_requires_original_complete_batch_and_frozen_policy():
    config = _read(ROOT / "configs/multireaction/ugi_sampling_trace_v1.json")
    comparison = _read(ROOT / config["inputs"]["comparison_config"]["path"])
    assert _validate(config, comparison)["batch_size"] == 128
    for field, value in (("program_count", 16), ("selected_attempts", [0, 0]), ("cpu_threads", 0)):
        changed = deepcopy(config)
        changed["runtime"][field] = value
        with pytest.raises(UgiSamplingTraceError):
            _validate(changed, comparison)
    changed = deepcopy(config)
    changed["policy"]["repairs_or_retries"] = True
    with pytest.raises(UgiSamplingTraceError, match="policy"):
        _validate(changed, comparison)
    changed = deepcopy(config)
    changed["checkpoint"]["step"] += 1
    with pytest.raises(UgiSamplingTraceError, match="checkpoint"):
        _validate(changed, comparison)


def test_historical_comparison_includes_topology_and_invalid_attempts():
    rows = [
        {
            "sample_index": 0,
            "canonical_smiles": "CC",
            "valid": True,
            "sampled_topology": {"a": [1]},
        },
        {
            "sample_index": 1,
            "canonical_smiles": None,
            "valid": False,
            "constraint_abstention_reason": "support",
        },
    ]
    saved = {
        "samples": [
            {
                "pipeline_index": 0,
                "smiles": "CC",
                "valid": True,
                "sampled_topology": {"a": [1]},
                "constraint_abstention_reason": None,
            },
            {
                "pipeline_index": 1,
                "smiles": None,
                "valid": False,
                "sampled_topology": None,
                "constraint_abstention_reason": "support",
            },
        ]
    }
    assert _compare_historical(rows, saved)["equal"]
    changed = deepcopy(saved)
    changed["samples"][0]["sampled_topology"]["a"] = [2]
    assert _compare_historical(rows, changed)["different_attempts"] == [0]
    changed = deepcopy(saved)
    changed["samples"][1]["constraint_abstention_reason"] = "other"
    assert _compare_historical(rows, changed)["different_attempts"] == [1]
    with pytest.raises(UgiSamplingTraceError):
        _compare_historical(rows, {"samples": []})
    with pytest.raises(UgiSamplingTraceError):
        _compare_historical([], saved)


def test_empty_or_partial_endpoint_never_establishes_passivity():
    assert not _complete_endpoint({"sampler_returns": []}, 2)
    captured = {
        "terminal_state_scope": "final_batch",
        "batch_offset": 0,
        "total_attempts": 2,
        "batch_record_ids": ["a", "b"],
        "terminal_states": {"nodes": [[1], [2]], "bonds": [[0], [0]]},
        "generator_states": {
            "generator": {"state": "x"},
            "topology_generator": {"state": "y"},
            "topology_conditioned_chemistry_generator": None,
            "terminal_generator": None,
        },
    }
    assert _complete_endpoint({"sampler_returns": [captured]}, 2)
    assert not _complete_endpoint({"sampler_returns": [captured, captured]}, 2)
    for key, value in (
        ("batch_offset", 2),
        ("total_attempts", 4),
        ("terminal_states", {}),
        ("generator_states", {}),
        ("batch_record_ids", ["a"]),
    ):
        changed = deepcopy(captured)
        changed[key] = value
        assert not _complete_endpoint({"sampler_returns": [changed]}, 2)


def test_outcome_only_trace_cannot_establish_decision_coverage():
    def event(kind, selector):
        return {"kind": kind, "selector": selector, "context": {"attempt_index": 0}}

    outcomes = [event("attempt_outcome", "sample_synthesis_program_products")]
    rows = [{"valid": True}]
    assert not _complete_decisions(outcomes, [0], rows)
    events = outcomes + [
        event("selector_outcome", "decode_ugi_exact_topology"),
        event("selector_outcome", "_strict_terminal_record"),
        event("categorical_choice", "_sample_amine_semantic_topology"),
        event("categorical_choice", "_sample_constructive_ester_offspring"),
        event("terminal_assignment", "_select_ugi_amine_atom_states"),
        event("terminal_assignment", "_ugi_ester_motif_constraints"),
    ]
    assert _complete_decisions(events, [0], rows)
    for index in range(len(events)):
        assert not _complete_decisions(events[:index] + events[index + 1 :], [0], rows)
    assert _complete_decisions(
        outcomes, [0], [{"valid": False, "constraint_abstention_reason": "support"}]
    )
    assert not _complete_decisions(outcomes, [0], [{"valid": False}])
