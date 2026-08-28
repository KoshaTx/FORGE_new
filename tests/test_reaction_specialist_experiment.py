from __future__ import annotations

import json
from pathlib import Path

from experiments._runtime.spec import ExperimentSpec
from experiments.catalog import SPECIFICATIONS
from experiments.phase1.product_l1.evaluation.reaction_specialist_ugi_v0_comparison import (
    _load_topology_policy,
    _validate,
)
from forge.core.hashing import sha256_file
from forge.model.reaction_specialization import exact_exposure_schedule

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "multireaction"
SPEC = REPO / "experiments" / "phase1" / "multireaction" / "reaction_specialists_seed0_h100.json"
RECOVERY_SPEC = (
    REPO
    / "experiments"
    / "phase1"
    / "multireaction"
    / "reaction_specialist_ugi_v0_evaluation_recovery_seed0_h100.json"
)
TOPOLOGY_CONFIG = (
    REPO / "configs/model/phase1_reaction_topology_specialist_ugi_v0_comparison_v2.json"
)
TOPOLOGY_SPEC = (
    REPO
    / "experiments/phase1/multireaction/reaction_topology_specialist_ugi_seed0_h100_v2.json"
)
BASE_TRAINING = (
    REPO
    / "runs/phase1-shared-mixed-role-morphology-seed0-h100"
    / "7763bd4f949c7724cf28a75f8cef37bafc31fa775c4c304ce33f1c813154e134"
    / "stages/study/artifacts/training/result.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_specialist_exposure_targets_match_authenticated_base_evidence() -> None:
    training = _read(BASE_TRAINING)
    observed = training["arms"]["full_role_morphology_transformer"][
        "examples_seen_by_program"
    ]
    expected_final = {
        "ugi_3cr_agile": (2469, (32, 32, 32)),
        "bl_2023_repeated_aza_michael": (2416, (32, 32, 16)),
        "lx_2024_repeated_reductive_amination": (2416, (32, 32, 16)),
    }
    for filename in (
        "reaction_specialist_ugi_v1.json",
        "reaction_specialist_bl_v1.json",
        "reaction_specialist_lx_v1.json",
    ):
        config = _read(CONFIGS / filename)
        program = config["target_program"]
        assert config["exposure"]["existing_examples"] == observed[program]
        assert config["exposure"]["target_examples"] == 384000
        schedule = exact_exposure_schedule(
            existing_examples=config["exposure"]["existing_examples"],
            target_examples=config["exposure"]["target_examples"],
            effective_batch_size=config["full"]["effective_batch_size"],
            micro_batch_size=config["full"]["micro_batch_size"],
        )
        assert (len(schedule.optimizer_steps), schedule.final_microbatches) == expected_final[program]


def test_specialist_descriptor_is_fail_closed_and_parallel_after_preflight() -> None:
    group = _read(SPEC)
    assert group["schema_version"] == "forge.modal_experiment_group.v1"
    assert [member["id"] for member in group["parallel"]] == ["ugi", "bl", "lx"]
    members = [group["preflight"], *group["parallel"]]
    specs: dict[str, dict] = {}
    for member in members:
        # The catalog paths are intentionally pinned by the group; derive the admitted path by
        # matching the declared experiment id rather than trusting a second hand-written mapping.
        candidates = list((REPO / "experiments/phase1/multireaction").glob("*.json"))
        spec_path = next(
            path
            for path in candidates
            if _read(path).get("experiment_id") == member["experiment"]
        )
        assert sha256_file(spec_path) == member["spec_sha256"]
        specs[member["id"]] = _read(spec_path)

    assert specs["preflight"]["stages"][0]["implementation"].endswith("h100-preflight.v1")
    assert [stage["id"] for stage in specs["ugi"]["stages"]] == [
        "specialize_ugi",
        "compare_ugi_v0",
    ]
    assert specs["ugi"]["stages"][1]["needs"] == ["specialize_ugi"]
    for member_id, spec in specs.items():
        assert all(stage["resources"]["gpu_type"] == "H100!" for stage in spec["stages"])
        if member_id != "preflight":
            assert spec["metadata"]["requires_verified_group_preflight"] is True
        for stage in spec["stages"]:
            config_path = REPO / stage["config"]["path"]
            assert sha256_file(config_path) == stage["config"]["sha256"]


def test_specialist_comparison_uses_exact_program_decoder_and_matched_assessors() -> None:
    config = _read(REPO / "configs/model/phase1_reaction_specialist_ugi_v0_comparison_v1.json")
    assert config["matched_ugi_exposure"] == 384000
    assert (
        config["profiles"]["full"]["current_terminal_decode_policy"]
        == "strict_program_topology_argmax"
    )
    assert config["profiles"]["full"]["program_count"] == 3072
    assert config["policy"]["method_blind_assessment"] is True
    assert config["policy"]["repairs_or_retries"] is False


def test_specialist_recovery_is_hash_pinned_evaluation_only() -> None:
    spec = _read(RECOVERY_SPEC)
    assert spec["profiles"] == ["smoke", "full"]
    assert spec["metadata"]["training_calls"] == 0
    assert len(spec["stages"]) == 1
    stage = spec["stages"][0]
    assert stage["implementation"] == "evaluate.ugi.reaction-specialist-v0-comparison.v1"
    assert stage["needs"] == []
    assert stage["resources"]["gpu_type"] == "H100!"

    config_path = REPO / stage["config"]["path"]
    config = _read(config_path)
    assert sha256_file(config_path) == stage["config"]["sha256"]
    assert config["inputs"] == stage["inputs"]
    assert config["policy"]["training_calls"] == 0
    runtime = _validate(config, profile="smoke", device="cuda")
    assert runtime["program_count"] == 4
    checkpoint = REPO / config["inputs"]["specialist_checkpoint"]["path"]
    result = REPO / config["inputs"]["specialist_result"]["path"]
    assert checkpoint.with_name("result.json") == result
    assert sha256_file(checkpoint) == config["inputs"]["specialist_checkpoint"]["sha256"]
    assert sha256_file(result) == config["inputs"]["specialist_result"]["sha256"]


def test_topology_specialist_is_preflight_gated_and_uses_pinned_exact_program_support() -> None:
    config = _read(TOPOLOGY_CONFIG)
    runtime = _validate(config, profile="full", device="cuda")
    inputs = {label: REPO / pin["path"] for label, pin in config["inputs"].items()}
    policy = _load_topology_policy(inputs)

    assert runtime["current_terminal_decode_policy"] == (
        "strict_ugi_program_coupled_conditional"
    )
    assert runtime["terminal_decoder_seed"] == runtime["flow_seed"] + 1
    assert policy.allowed_ring_sizes == (5, 6, 7)
    assert policy.maximum_heavy_degree == 4
    assert policy.maximum_adjacent_branch_run_by_role == (2, 1, 1)

    spec = ExperimentSpec.load(TOPOLOGY_SPEC)
    assert spec.experiment_id == "phase1-reaction-topology-specialist-ugi-seed0-h100-v2"
    document = _read(TOPOLOGY_SPEC)
    assert [stage["id"] for stage in document["stages"]] == [
        "preflight",
        "specialize_ugi",
        "compare_ugi_v0",
    ]
    assert document["stages"][1]["needs"] == ["preflight"]
    assert document["stages"][2]["needs"] == ["specialize_ugi"]
    assert document["stages"][2]["implementation"].endswith(
        "reaction-topology-specialist-v0-comparison.v2"
    )
    assert SPECIFICATIONS[spec.experiment_id] == str(TOPOLOGY_SPEC.relative_to(REPO))
    for stage in document["stages"]:
        assert stage["resources"]["gpu_type"] == "H100!"
        assert sha256_file(REPO / stage["config"]["path"]) == stage["config"]["sha256"]
