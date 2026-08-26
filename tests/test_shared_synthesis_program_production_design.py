from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.phase1.multireaction.production_design import (
    SynthesisProgramProductionDesignError,
    freeze_shared_production_comparison_design,
    validate_production_design_contract,
)
from forge.core.hashing import sha256_file

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/shared_production_comparison_design_v1.json"


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_frozen_design_uses_real_weight_fields_and_matches_compute(tmp_path: Path) -> None:
    result = freeze_shared_production_comparison_design(
        CONFIG,
        REPO,
        tmp_path / "result.json",
    )

    assert result["status"] == "design_frozen_launch_blocked"
    assert all(result["gates"].values())
    assert result["production_training_authorized"] is False
    assert result["production_sampling_authorized"] is False
    assert result["corpus"]["ugi_3cr_agile"]["weight_field"] == ("family_balance_weight_raw")
    assert result["corpus"]["bl_2023_repeated_aza_michael"]["weight_field"] == (
        "source_balanced_weight"
    )
    assert (
        result["corpus"]["lx_2024_repeated_reductive_amination"]["folds"]["train"]["records"] == 70
    )
    compute = result["matched_compute"]
    assert compute["arms"] == 4
    assert compute["replicates"] == 3
    assert compute["optimizer_steps_per_arm_per_replicate"] == 1700
    assert compute["examples_per_arm_per_replicate"] == 217_600
    assert compute["total_planned_optimizer_steps"] == 20_400
    assert result["evaluation_contract"]["route_calls"] == 0
    assert result["evaluation_contract"]["oracle_calls"] == 0


def test_design_receipt_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "one" / "result.json"
    second = tmp_path / "two" / "result.json"
    freeze_shared_production_comparison_design(CONFIG, REPO, first)
    freeze_shared_production_comparison_design(CONFIG, REPO, second)
    assert sha256_file(first) == sha256_file(second)


def test_design_rejects_a_raw_count_or_unequal_program_prior() -> None:
    config = _config()
    programs = config["programs"]
    assert isinstance(programs, dict)
    ugi = programs["ugi_3cr_agile"]
    assert isinstance(ugi, dict)
    ugi["weight_field"] = "reaction_family"
    with pytest.raises(
        SynthesisProgramProductionDesignError,
        match="source-balanced weight fields",
    ):
        validate_production_design_contract(config)

    config = _config()
    training = config["training"]
    assert isinstance(training, dict)
    arms = training["arms"]
    assert isinstance(arms, dict)
    shared = arms["shared_three_program_conditioned"]
    assert isinstance(shared, dict)
    mass = shared["program_mass"]
    assert isinstance(mass, dict)
    mass.update(
        {
            "ugi_3cr_agile": 0.5,
            "bl_2023_repeated_aza_michael": 0.25,
            "lx_2024_repeated_reductive_amination": 0.25,
        }
    )
    with pytest.raises(SynthesisProgramProductionDesignError, match="frozen program prior"):
        validate_production_design_contract(config)


def test_design_rejects_holdout_selection_gate_relaxation_and_candidate_use() -> None:
    for mutation, message in (
        (("evaluation", "held_reaction_family", "hard_gate", True), "held reaction family"),
        (("evaluation", "candidate_selection", None, True), "select candidates"),
        (
            ("execution", "production_launch_authorized", None, True),
            "cannot authorize a production launch",
        ),
    ):
        config = copy.deepcopy(_config())
        section, key, nested, value = mutation
        parent = config[section]
        assert isinstance(parent, dict)
        if nested is None:
            parent[key] = value
        else:
            child = parent[key]
            assert isinstance(child, dict)
            child[nested] = value
        with pytest.raises(SynthesisProgramProductionDesignError, match=message):
            validate_production_design_contract(config)


def test_design_reports_coverage_and_precision_and_forbids_degenerate_metric() -> None:
    config = _config()
    evaluation = config["evaluation"]
    assert isinstance(evaluation, dict)
    metrics = evaluation["per_family_metrics"]
    assert isinstance(metrics, list)
    assert "exact_l1_decomposition_coverage" in metrics
    assert "exact_forward_replay_precision" in metrics
    assert evaluation["forbidden_metrics"] == ["reductive_amination_substructure_hit_rate"]


def test_design_accepts_only_the_qualified_transformer_balancing_contract() -> None:
    config = _config()
    config["model"] = {
        "architecture": "reaction_program_graph_transformer",
        "hidden_dim": 192,
        "layers": 6,
        "attention_heads": 8,
        "expert_count": 3,
        "adapter_dim": 64,
        "dropout": 0.1,
        "bond_classes": 4,
        "maximum_heavy_atoms": 194,
        "maximum_closures": 3,
        "source_probability_floor": 0.00001,
        "component_ids_enter_neural_tensors": False,
        "fragment_tokens_enter_neural_tensors": False,
        "semantic_objective": {
            "role_consistency_weight": 0.25,
            "core_consistency_weight": 0.25,
            "state_balancing": "equal_present_semantic_state_mass",
        },
        "gradient_balancing": {
            "method": "equal_family_mass_deterministic_pcgrad",
            "group_identity": "source_program_state",
            "norm_amplification": False,
        },
    }
    assert all(validate_production_design_contract(config).values())
    config["model"]["gradient_balancing"]["norm_amplification"] = True
    with pytest.raises(SynthesisProgramProductionDesignError, match="family-balancing"):
        validate_production_design_contract(config)
