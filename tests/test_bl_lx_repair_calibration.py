from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.phase1.multireaction.mechanism_study import _deep_merge, _study_arms
from experiments.phase1.multireaction.production_training import (
    SynthesisProgramProductionTrainingError,
    _StratifiedProgramSampler,
)
from forge.model.synthesis_program_sampling import TERMINAL_DECODE_POLICIES

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/bl_lx_repair_calibration_v1.json"
BL_CORE_CONFIG = REPO / "configs/multireaction/bl_core_constraint_calibration_v1.json"
UGI_EXPOSURE_CONFIG = REPO / "configs/multireaction/ugi_train_exposure_calibration_v1.json"
BL_CORE_PRODUCTION_CONFIG = REPO / "configs/multireaction/bl_core_constrained_production_v1.json"
MECHANISM_CONFIG = REPO / "configs/multireaction/transformer_mechanism_study_v1.json"


def test_mechanism_ablations_are_one_factor_changes_from_final_forge() -> None:
    config = json.loads(MECHANISM_CONFIG.read_text())
    programs = (
        "ugi_3cr_agile",
        "bl_2023_repeated_aza_michael",
        "lx_2024_repeated_reductive_amination",
    )
    arms = _study_arms(config, programs)
    assert set(arms) == {
        "full_transformer",
        "input_only_program",
        "no_role_loss",
        "no_core_loss",
        "no_routed_adapters",
        "no_gradient_conflict_control",
        "fact_matched",
        "fact_generous",
    }
    for arm in arms.values():
        overrides = arm["model_overrides"]
        assert overrides["repeat_group_conditioning"] is True
        assert overrides["semantic_objective"]["repeat_consistency_weight"] == 0.25
        assert "batch_program_counts" not in arm
    assert arms["no_role_loss"]["model_overrides"]["semantic_objective"] == {
        "repeat_consistency_weight": 0.25,
        "role_consistency_weight": 0.0,
    }
    assert arms["no_core_loss"]["model_overrides"]["semantic_objective"] == {
        "repeat_consistency_weight": 0.25,
        "core_consistency_weight": 0.0,
    }
    for profile in ("smoke", "full"):
        assert (
            config[profile]["evaluation"]["terminal_decode_policy"]
            == "strict_valence_topology_argmax"
        )

    production = json.loads(
        (REPO / "experiments/phase1/multireaction/transformer_mechanism_study.json").read_text()
    )
    preflight = json.loads(
        (
            REPO
            / "experiments/phase1/multireaction/transformer_mechanism_study_h100_preflight.json"
        ).read_text()
    )
    assert production["replicates"]["full"] == 3
    assert production["stages"][0]["resources"]["gpu_type"] == "H100!"
    assert preflight["profiles"] == ["smoke"]
    assert preflight["stages"][0]["resources"]["gpu_type"] == "H100!"
    preflight_config = json.loads((REPO / preflight["stages"][0]["config"]["path"]).read_text())
    assert preflight_config["smoke"]["training"]["device"] == "cuda"
    assert preflight_config["smoke"]["evaluation"]["device"] == "cuda"


def test_repair_study_is_two_models_by_two_decoders_with_ugi_anti_regression() -> None:
    config = json.loads(CONFIG.read_text())
    programs = (
        "ugi_3cr_agile",
        "bl_2023_repeated_aza_michael",
        "lx_2024_repeated_reductive_amination",
    )
    arms = _study_arms(config, programs)
    assert set(arms) == {"corrected_layout_reference", "repeat_aware_transformer"}
    assert arms["corrected_layout_reference"].get("model_overrides") is None
    intervention = arms["repeat_aware_transformer"]["model_overrides"]
    assert intervention["repeat_group_conditioning"] is True
    assert intervention["semantic_objective"]["repeat_consistency_weight"] == 0.25
    assert all(arm["evaluation_programs"] == list(programs) for arm in arms.values())
    for profile in ("smoke", "full"):
        evaluation = config[profile]["evaluation"]
        assert evaluation["calibration_only"] is True
        assert "heldout_samples" not in evaluation
        assert tuple(evaluation["terminal_decode_policies"]) == TERMINAL_DECODE_POLICIES
    assert len(arms) * len(TERMINAL_DECODE_POLICIES) == 4


def test_repeat_override_preserves_the_frozen_role_and_core_objectives() -> None:
    base = {
        "semantic_objective": {
            "role_consistency_weight": 0.25,
            "core_consistency_weight": 0.25,
            "state_balancing": "equal_present_semantic_state_mass",
        }
    }
    override = {
        "semantic_objective": {"repeat_consistency_weight": 0.25},
        "repeat_group_conditioning": True,
    }
    merged = _deep_merge(base, override)
    assert merged["semantic_objective"] == {
        "role_consistency_weight": 0.25,
        "core_consistency_weight": 0.25,
        "repeat_consistency_weight": 0.25,
        "state_balancing": "equal_present_semantic_state_mass",
    }
    assert merged["repeat_group_conditioning"] is True


def test_local_and_h100_descriptors_pin_the_same_calibration_contract() -> None:
    config_sha = "2879dc167d4c50346301a629690337be698d0d493f5d6dffe37da94e115f59c1"
    for name, profile, device in (
        ("bl_lx_repair_calibration_smoke.json", "smoke", "cpu"),
        ("bl_lx_repair_calibration_h100.json", "full", "cuda"),
    ):
        descriptor = json.loads((REPO / "experiments/phase1/multireaction" / name).read_text())
        assert descriptor["profiles"] == [profile]
        stage = descriptor["stages"][0]
        assert stage["config"]["sha256"] == config_sha
        assert stage["resources"]["device"] == device
        assert (
            stage["outputs"]["evaluation_result"]["schema_version"]
            == "forge.bl_lx_repair_calibration_result.v1"
        )


def test_bl_core_calibration_changes_one_model_factor_and_reads_calibration_only() -> None:
    config = json.loads(BL_CORE_CONFIG.read_text())
    programs = (
        "ugi_3cr_agile",
        "bl_2023_repeated_aza_michael",
        "lx_2024_repeated_reductive_amination",
    )
    arms = _study_arms(config, programs)
    assert set(arms) == {"bl_core_constrained_repeat_aware"}
    arm = arms["bl_core_constrained_repeat_aware"]
    assert arm["program_mass"] == {program: 1 / 3 for program in programs}
    assert arm["model_overrides"] == {
        "repeat_group_conditioning": True,
        "semantic_objective": {"repeat_consistency_weight": 0.25},
    }
    assert config["comparison"]["changed_factor"] == (
        "BL semantic reaction-core atoms and bonds are adapter-fixed"
    )
    for profile in ("smoke", "full"):
        assert config[profile]["evaluation"]["calibration_only"] is True
        assert "heldout_samples" not in config[profile]["evaluation"]


def test_bl_core_descriptors_pin_one_local_and_one_unlaunched_h100_seed() -> None:
    config_sha = "134c7be45ab1138fb85e7430a97298352029af943433daae4b6a06c6dade5407"
    for name, profile, device in (
        ("bl_core_constraint_calibration_smoke.json", "smoke", "cpu"),
        ("bl_core_constraint_calibration_h100.json", "full", "cuda"),
    ):
        descriptor = json.loads((REPO / "experiments/phase1/multireaction" / name).read_text())
        assert descriptor["profiles"] == [profile]
        assert descriptor["replicates"][profile] == 1
        stage = descriptor["stages"][0]
        assert stage["config"]["sha256"] == config_sha
        assert stage["resources"]["device"] == device
        assert stage["outputs"]["evaluation_result"]["schema_version"] == (
            "forge.bl_core_constraint_calibration_result.v1"
        )
    h100 = json.loads(
        (
            REPO / "experiments/phase1/multireaction/bl_core_constraint_calibration_h100.json"
        ).read_text()
    )
    assert h100["metadata"]["paid_launch_requires_separate_approval"] is True


def test_bl_core_h100_preflight_exercises_smoke_profile_on_exact_h100() -> None:
    descriptor = json.loads(
        (
            REPO
            / "experiments/phase1/multireaction/bl_core_constraint_calibration_h100_preflight.json"
        ).read_text()
    )
    assert descriptor["profiles"] == ["smoke"]
    assert descriptor["replicates"] == {"smoke": 1}
    stage = descriptor["stages"][0]
    assert stage["config"] == {
        "path": "configs/multireaction/bl_core_constraint_calibration_h100_preflight_v1.json",
        "sha256": "377940ae5e1bc2efc35ff0991ef66e9dc0798bc594d2381051c927fda23ece27",
    }
    assert stage["resources"]["device"] == "cuda"
    assert stage["resources"]["gpu_type"] == "H100!"
    assert descriptor["metadata"]["optimizer_steps"] == 2
    assert descriptor["metadata"]["scientific_result"] is False
    config = json.loads((REPO / stage["config"]["path"]).read_text())
    assert config["smoke"]["training"]["device"] == "cuda"
    assert config["smoke"]["evaluation"]["device"] == "cuda"


def test_ugi_exposure_arm_appends_only_ugi_rows_and_preserves_all_folds() -> None:
    config = json.loads(UGI_EXPOSURE_CONFIG.read_text())
    programs = (
        "ugi_3cr_agile",
        "bl_2023_repeated_aza_michael",
        "lx_2024_repeated_reductive_amination",
    )
    arms = _study_arms(config, programs)
    assert set(arms) == {"bl_core_constrained_ugi_exposure"}
    arm = arms["bl_core_constrained_ugi_exposure"]
    assert arm["program_mass"] == {program: 1 / 3 for program in programs}
    assert config["historical_batch_program_counts"] == {
        "ugi_3cr_agile": 11,
        "bl_2023_repeated_aza_michael": 11,
        "lx_2024_repeated_reductive_amination": 10,
    }
    assert arm["batch_program_counts"] == {
        "ugi_3cr_agile": 21,
        "bl_2023_repeated_aza_michael": 11,
        "lx_2024_repeated_reductive_amination": 10,
    }
    assert config["full"]["training"]["micro_batch_size"] == 42
    assert config["full"]["training"]["effective_batch_size"] == 168
    for profile in ("smoke", "full"):
        assert config[profile]["evaluation"]["calibration_only"] is True
        assert "heldout_samples" not in config[profile]["evaluation"]


def test_stratified_sampler_preserves_historical_and_expanded_batch_counts() -> None:
    common = {
        "program_states": (1, 2, 3),
        "program_ids": ("ugi", "bl", "lx"),
        "indices": (
            np.asarray([101], dtype=np.int64),
            np.asarray([202], dtype=np.int64),
            np.asarray([303], dtype=np.int64),
        ),
        "probabilities": (
            np.asarray([1.0]),
            np.asarray([1.0]),
            np.asarray([1.0]),
        ),
    }
    historical = _StratifiedProgramSampler(**common)
    expanded = _StratifiedProgramSampler(**common, fixed_batch_counts=(21, 11, 10))

    assert historical.batch_counts(32) == (11, 11, 10)
    assert expanded.batch_counts(42) == (21, 11, 10)
    selected = expanded.sample(42, np.random.default_rng(9))
    values, counts = np.unique(selected, return_counts=True)
    assert dict(zip(values.tolist(), counts.tolist(), strict=True)) == {
        101: 21,
        202: 11,
        303: 10,
    }
    with pytest.raises(
        SynthesisProgramProductionTrainingError,
        match="do not sum to the runtime microbatch size",
    ):
        expanded.sample(41, np.random.default_rng(9))


def test_padding_aware_sampler_preserves_marginals_and_reduces_batch_maximum() -> None:
    family_indices = tuple(
        np.asarray([offset + 10, offset + 100], dtype=np.int64) for offset in (0, 200, 400)
    )
    probabilities = tuple(np.asarray([0.9, 0.1]) for _ in family_indices)
    cumulative = tuple(np.asarray([0.9, 1.0]) for _ in family_indices)
    sampler = _StratifiedProgramSampler(
        program_states=(1, 2, 3),
        program_ids=("ugi", "bl", "lx"),
        indices=family_indices,
        probabilities=probabilities,
        size_sorted_indices=family_indices,
        size_cumulative_probabilities=cumulative,
    )
    random_rng = np.random.default_rng(219)
    coupled_rng = np.random.default_rng(219)
    random_maxima = []
    coupled_maxima = []
    coupled_large = np.zeros(3, dtype=np.int64)
    samples = 20_000
    for _ in range(samples):
        random_draw = sampler.sample(3, random_rng)
        coupled_draw = sampler.sample(3, coupled_rng, padding_aware_quantile_bins=10)
        random_maxima.append(max(int(value) % 200 for value in random_draw))
        coupled_maxima.append(max(int(value) % 200 for value in coupled_draw))
        for family, values in enumerate(family_indices):
            coupled_large[family] += int(values[1] in coupled_draw)

    assert np.allclose(coupled_large / samples, np.full(3, 0.1), atol=0.01)
    assert np.mean(coupled_maxima) < np.mean(random_maxima) * 0.7
    first = sampler.sample(3, np.random.default_rng(221), padding_aware_quantile_bins=10)
    second = sampler.sample(3, np.random.default_rng(221), padding_aware_quantile_bins=10)
    assert np.array_equal(first, second)


def test_ugi_exposure_descriptors_pin_local_preflight_and_full_contracts() -> None:
    cases = (
        (
            "ugi_train_exposure_calibration_smoke.json",
            "smoke",
            "cpu",
            "configs/multireaction/ugi_train_exposure_calibration_v1.json",
            "042cf0b780e994a864f859257a48550f8ce3d05dc6b92cb8a605e45589fcefac",
        ),
        (
            "ugi_train_exposure_calibration_h100_preflight.json",
            "smoke",
            "cuda",
            "configs/multireaction/ugi_train_exposure_calibration_h100_preflight_v1.json",
            "6b941ef99583360fce35a4ab6655208c173bb7cf4c6632d6194e167dd52c1542",
        ),
        (
            "ugi_train_exposure_calibration_h100.json",
            "full",
            "cuda",
            "configs/multireaction/ugi_train_exposure_calibration_v1.json",
            "042cf0b780e994a864f859257a48550f8ce3d05dc6b92cb8a605e45589fcefac",
        ),
    )
    for name, profile, device, config_path, config_sha in cases:
        descriptor = json.loads((REPO / "experiments/phase1/multireaction" / name).read_text())
        assert descriptor["profiles"] == [profile]
        assert descriptor["replicates"] == {profile: 1}
        stage = descriptor["stages"][0]
        assert stage["config"] == {"path": config_path, "sha256": config_sha}
        assert stage["resources"]["device"] == device
        assert stage["outputs"]["evaluation_result"]["schema_version"] == (
            "forge.ugi_train_exposure_calibration_result.v1"
        )
        assert descriptor["metadata"]["batch_program_counts"] == {
            "ugi_3cr_agile": 21,
            "bl_2023_repeated_aza_michael": 11,
            "lx_2024_repeated_reductive_amination": 10,
        }


def test_bl_core_production_promotes_standard_batch_and_strict_decoder() -> None:
    config = json.loads(BL_CORE_PRODUCTION_CONFIG.read_text())
    programs = (
        "ugi_3cr_agile",
        "bl_2023_repeated_aza_michael",
        "lx_2024_repeated_reductive_amination",
    )
    arms = _study_arms(config, programs)
    assert set(arms) == {"bl_core_constrained_repeat_aware"}
    arm = arms["bl_core_constrained_repeat_aware"]
    assert arm["role"] == "final_forge_conditioned"
    assert "batch_program_counts" not in arm
    assert arm["program_mass"] == {program: 1 / 3 for program in programs}
    assert arm["model_overrides"] == {
        "repeat_group_conditioning": True,
        "semantic_objective": {"repeat_consistency_weight": 0.25},
    }
    assert config["full"]["training"]["micro_batch_size"] == 32
    assert config["full"]["training"]["effective_batch_size"] == 128
    assert config["full"]["evaluation"] == {
        "device": "cuda",
        "checkpoint_steps": [100, 500, 1000, 1500, 1700],
        "sample_steps": 32,
        "batch_size": 32,
        "calibration_samples": 512,
        "heldout_samples": 3072,
        "component_disjoint_record_limit": None,
        "terminal_decode_policy": "strict_valence_topology_argmax",
    }
    assert config["promotion"]["rejected_schedule"] == "bl_core_constrained_ugi_exposure"


def test_bl_core_production_descriptors_freeze_three_exact_h100_seeds() -> None:
    production_sha = "14fbf0fc4401131bdc9b34a41afe5335817dbe78b74f0e3e5d4a3706efdb93e3"
    preflight_sha = "407b61af3cb78f970acb956f6129a8e8dbf82c1e63d9405a7e6cde1611ebe477"
    cases = (
        (
            "bl_core_constrained_production_smoke.json",
            "smoke",
            "cpu",
            production_sha,
            1,
        ),
        (
            "bl_core_constrained_production_h100_preflight.json",
            "smoke",
            "cuda",
            preflight_sha,
            1,
        ),
        (
            "bl_core_constrained_production_h100.json",
            "full",
            "cuda",
            production_sha,
            3,
        ),
    )
    for name, profile, device, config_sha, replicates in cases:
        descriptor = json.loads((REPO / "experiments/phase1/multireaction" / name).read_text())
        stage = descriptor["stages"][0]
        assert descriptor["profiles"] == [profile]
        assert descriptor["replicates"] == {profile: replicates}
        assert stage["config"]["sha256"] == config_sha
        assert stage["resources"]["device"] == device
        assert stage["outputs"]["evaluation_result"]["schema_version"] == (
            "forge.synthesis_program_production_evaluation_result.v1"
        )
    h100 = json.loads(
        (
            REPO / "experiments/phase1/multireaction/bl_core_constrained_production_h100.json"
        ).read_text()
    )
    assert h100["stages"][0]["resources"]["gpu_type"] == "H100!"
    assert h100["metadata"]["decoder"] == "strict_valence_topology_argmax"
