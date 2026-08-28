from __future__ import annotations

from pathlib import Path

import pytest

from experiments._runtime.spec import ExperimentSpec
from experiments.phase1.multireaction.mechanism_study import _deep_merge, _study_arms
from experiments.phase1.multireaction.production_evaluation import (
    SynthesisProgramProductionEvaluationError,
    _validate_evaluation_budget,
)
from forge.core.io import read_json_object
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.synthesis_program_training import build_synthesis_program_flow

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/shared_bias_end_to_end_retraining_v1.json"
PREFLIGHT = REPO / "configs/multireaction/shared_bias_end_to_end_h100_preflight_v1.json"
DESIGN = REPO / "results/phase1/shared_synthesis_program_mixed_design_v1/design.json"
CACHE = REPO / "results/phase1/shared_synthesis_program_mixed_cache_v1/cache.npz"
EXPERIMENT = REPO / "experiments/phase1/multireaction/shared_bias_end_to_end_seed0_h100.json"


def test_retraining_contract_is_end_to_end_matched_and_fixed_at_step_9000() -> None:
    config = read_json_object(CONFIG)
    design = read_json_object(DESIGN)
    programs = tuple(design["programs"])
    arms = _study_arms(config, programs)

    assert set(arms) == {
        "shared_bias_global_source_control",
        "shared_bias_program_role_source",
    }
    assert {arm["source_marginal_mode"] for arm in arms.values()} == {
        "global",
        "program_role_full_support",
    }
    for arm in arms.values():
        assert arm["program_mass"] == {program: 1 / 3 for program in programs}
        model = arm["model_overrides"]
        assert int(model.get("specialist_adapter_dim", 0)) == 0
        assert model["program_routed_output_heads"] is True
        assert model["maximum_children"] == 3
        assert model["semantic_objective"] == {
            "repeat_consistency_weight": 0.25,
            "offspring_weight": 1.0,
            "junction_consistency_weight": 0.5,
            "chemistry_loss_balancing": "equal_present_role_mass",
            "topology_conditioned_chemistry_weight": 1.0,
        }
    assert config["full"]["training"]["optimizer_steps"] == 9000
    assert config["full"]["training"]["checkpoint_steps"] == [100, 500, 1700, 4500, 9000]
    assert config["full"]["training"]["effective_batch_size"] == 128


def test_full_shared_model_parameter_count_matches_frozen_launch_gate() -> None:
    config = read_json_object(CONFIG)
    design = read_json_object(DESIGN)
    arms = _study_arms(config, tuple(design["programs"]))
    model_config = _deep_merge(
        design["model"], arms["shared_bias_program_role_source"]["model_overrides"]
    )
    with SynthesisProgramProductionCache(CACHE) as cache:
        model = build_synthesis_program_flow(
            vocabulary=cache.vocabulary,
            node_classes=len(cache.atom_vocabulary),
            model_config=model_config,
            device="cpu",
        )
    assert sum(parameter.numel() for parameter in model.parameters()) == int(
        config["expected_full_parameter_count"]
    )


def test_paid_experiment_is_fail_closed_through_exact_h100_preflight() -> None:
    preflight = read_json_object(PREFLIGHT)
    experiment = ExperimentSpec.load(EXPERIMENT)
    assert preflight["execution_scope"] == "h100_preflight"
    assert preflight["full"]["training"]["optimizer_steps"] == 3
    assert experiment.profiles == ("full",)
    assert tuple(stage.stage_id for stage in experiment.stages) == (
        "h100_preflight",
        "production",
    )
    assert experiment.stages[1].needs == ("h100_preflight",)
    assert all(stage.resources.gpu_type == "H100!" for stage in experiment.stages)


def test_preflight_budget_is_explicitly_execution_only_without_weakening_production() -> None:
    preflight = read_json_object(PREFLIGHT)
    production = read_json_object(CONFIG)
    design = read_json_object(DESIGN)
    frozen = design["evaluation"]["native_sampling"]

    preflight_evaluation = {
        "execution_scope": preflight["execution_scope"],
        "full": preflight["full"]["evaluation"],
    }
    production_evaluation = {
        "execution_scope": production["execution_scope"],
        "full": production["full"]["evaluation"],
    }
    assert (
        _validate_evaluation_budget(preflight_evaluation, design, profile="full")
        == "h100_preflight"
    )
    assert (
        _validate_evaluation_budget(production_evaluation, design, profile="full") == "production"
    )
    assert production_evaluation["full"]["sample_steps"] == int(frozen["flow_steps"])
    assert production_evaluation["full"]["heldout_samples"] == int(
        frozen["heldout_samples_per_supported_program_at_final_checkpoint_per_seed"]
    )

    invalid_production = {
        "execution_scope": "production",
        "full": dict(preflight_evaluation["full"]),
    }
    with pytest.raises(
        SynthesisProgramProductionEvaluationError,
        match="full evaluation budget differs from the frozen design",
    ):
        _validate_evaluation_budget(invalid_production, design, profile="full")
