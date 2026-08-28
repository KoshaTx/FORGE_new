from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments._runtime.modal import modal_request_plan
from experiments.phase1.multireaction.h100_training_profile import (
    _candidate_runtime,
    _equivalence,
    _passes_tolerance,
    _repartition_draws,
)
from forge.core.hashing import sha256_file

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/shared_mixed_h100_training_profile_v1.json"
SPEC = REPO / "experiments/phase1/multireaction/shared_mixed_h100_training_profile.json"
TF32_CONFIG = REPO / "configs/multireaction/shared_mixed_h100_tf32_profile_v1.json"
TF32_SPEC = REPO / "experiments/phase1/multireaction/shared_mixed_h100_tf32_profile.json"
PRODUCTION_PREFLIGHT_CONFIG = (
    REPO / "configs/multireaction/shared_mixed_primary_seed0_h100_preflight_v1.json"
)
PRODUCTION_CONFIG = REPO / "configs/multireaction/shared_mixed_primary_seed0_production_v1.json"
PRODUCTION_SPEC = (
    REPO / "experiments/phase1/multireaction/shared_mixed_primary_seed0_h100.json"
)
EVALUATION_CONFIG = (
    REPO / "configs/multireaction/shared_mixed_primary_seed0_evaluation_v1.json"
)
EVALUATION_SPEC = (
    REPO
    / "experiments/phase1/multireaction/shared_mixed_primary_seed0_evaluation_h100.json"
)


def test_h100_profile_pins_and_upload_closure_are_authenticated() -> None:
    config = json.loads(CONFIG.read_text())
    spec = json.loads(SPEC.read_text())
    stage = spec["stages"][0]
    assert stage["resources"]["gpu_type"] == "H100!"
    assert stage["resources"]["precision"] == "float32"
    assert stage["config"]["sha256"] == str(sha256_file(CONFIG))
    assert stage["inputs"] == config["inputs"]
    for pin in config["inputs"].values():
        assert str(sha256_file(REPO / pin["path"])) == pin["sha256"]

    request = modal_request_plan(REPO, SPEC, profile="smoke", replicate=0, device=None)
    assert request["resource_envelope"]["gpu_type"] == "H100!"
    assert set(config["inputs"]).issubset(
        {
            "production_design",
            "production_cache",
            "production_training_config",
            "cpu_performance_result",
            "smoke_training_result",
        }
    )
    uploaded = request["uploads"]
    assert config["inputs"]["production_cache"]["path"] in uploaded


def test_batch_geometry_preserves_examples_but_is_repartitioned() -> None:
    draws = (
        (
            np.arange(0, 32, dtype=np.int64),
            np.arange(32, 64, dtype=np.int64),
            np.arange(64, 96, dtype=np.int64),
            np.arange(96, 128, dtype=np.int64),
        ),
    )
    repartitioned = _repartition_draws(draws, micro_batch_size=64)
    assert len(repartitioned[0]) == 2
    assert np.array_equal(np.concatenate(repartitioned[0]), np.arange(128))

    baseline = {
        "micro_batch_size": 32,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 128,
    }
    candidate = {
        "micro_batch_size": 64,
        "gradient_accumulation_steps": 2,
    }
    assert _candidate_runtime(baseline, candidate)["effective_batch_size"] == 128


def test_tf32_is_an_explicit_qualified_candidate_not_the_reference_math_mode() -> None:
    config = json.loads(TF32_CONFIG.read_text())
    spec = json.loads(TF32_SPEC.read_text())
    assert spec["stages"][0]["config"]["sha256"] == str(sha256_file(TF32_CONFIG))
    candidates = config["profile"]["candidates"]
    assert candidates[0]["id"] == "fp32_eager"
    assert candidates[0]["allow_tf32"] is False
    assert [candidate["id"] for candidate in candidates[1:]] == [
        "tf32_eager",
        "tf32_compile",
    ]
    assert all(candidate["allow_tf32"] is True for candidate in candidates[1:])


def test_seed0_production_is_fail_closed_on_the_exact_final_architecture() -> None:
    preflight = json.loads(PRODUCTION_PREFLIGHT_CONFIG.read_text())
    production = json.loads(PRODUCTION_CONFIG.read_text())
    spec = json.loads(PRODUCTION_SPEC.read_text())
    assert spec["profiles"] == ["full"]
    assert spec["replicates"] == {"full": 1}
    assert [stage["id"] for stage in spec["stages"]] == ["preflight", "study"]
    assert spec["stages"][1]["needs"] == ["preflight"]
    assert all(stage["resources"]["gpu_type"] == "H100!" for stage in spec["stages"])
    assert spec["stages"][0]["config"]["sha256"] == str(
        sha256_file(PRODUCTION_PREFLIGHT_CONFIG)
    )
    assert spec["stages"][1]["config"]["sha256"] == str(sha256_file(PRODUCTION_CONFIG))
    assert preflight["profile"]["model_overrides"] == {"repeat_group_conditioning": True}
    assert preflight["profile"]["expected_parameter_count"] == 5_328_685
    assert production["authorization"]["profiles"] == ["full"]
    assert production["full"]["training"]["precision"] == "float32"
    assert production["full"]["training"]["micro_batch_size"] == 32
    assert production["full"]["training"]["gradient_accumulation_steps"] == 4
    assert production["full"]["evaluation"]["heldout_samples"] == 3072

    request = modal_request_plan(
        REPO, PRODUCTION_SPEC, profile="full", replicate=0, device=None
    )
    assert request["resource_envelope"]["gpu_type"] == "H100!"
    assert request["resource_envelope"]["timeout_seconds"] == 176400


def test_seed0_recovered_evaluation_is_preflighted_and_cannot_retrain() -> None:
    config = json.loads(EVALUATION_CONFIG.read_text())
    spec = json.loads(EVALUATION_SPEC.read_text())
    assert spec["profiles"] == ["full"]
    assert spec["replicates"] == {"full": 1}
    assert [stage["id"] for stage in spec["stages"]] == [
        "layout_preflight",
        "evaluation",
    ]
    assert spec["stages"][1]["needs"] == ["layout_preflight"]
    assert all("training" not in stage["implementation"] for stage in spec["stages"])
    assert spec["stages"][0]["resources"]["device"] == "cpu"
    assert spec["stages"][1]["resources"]["gpu_type"] == "H100!"
    assert all(
        stage["config"]["sha256"] == str(sha256_file(EVALUATION_CONFIG))
        for stage in spec["stages"]
    )
    assert config["full"]["checkpoint_steps"] == [100, 500, 1000, 1500, 1700]
    assert config["full"]["heldout_samples"] == 3072
    assert config["full"]["terminal_decode_policy"] == "strict_valence_topology_argmax"
    evaluation_inputs = spec["stages"][1]["inputs"]
    assert evaluation_inputs["checkpoint_archive"]["sha256"] == str(
        sha256_file(REPO / evaluation_inputs["checkpoint_archive"]["path"])
    )

    request = modal_request_plan(
        REPO, EVALUATION_SPEC, profile="full", replicate=0, device=None
    )
    assert request["resource_envelope"]["gpu_type"] == "H100!"
    assert evaluation_inputs["checkpoint_archive"]["path"] in request["uploads"]


def test_equivalence_gate_uses_parameter_updates_not_large_parameter_baseline() -> None:
    initial = torch.tensor([1000.0, -1000.0])
    reference_final = initial + torch.tensor([0.01, -0.02])
    candidate_final = initial + torch.tensor([0.011, -0.019])
    reference = {"losses": [2.0, 1.0]}
    candidate = {"losses": [2.001, 1.001]}
    values = _equivalence(
        reference,
        initial,
        reference_final,
        candidate,
        initial.clone(),
        candidate_final,
    )
    assert values["initial_parameters_exact"] is True
    assert values["parameter_update_relative_l2_difference"] > 0.05
    assert not _passes_tolerance(
        values,
        {
            "loss_max_absolute_difference": 0.01,
            "loss_max_relative_difference": 0.01,
            "parameter_update_relative_l2_difference": 0.05,
            "parameter_update_max_absolute_difference": 0.01,
            "parameter_update_cosine_similarity": 0.9,
        },
    )
