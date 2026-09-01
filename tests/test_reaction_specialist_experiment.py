from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments._runtime.errors import StageError
from experiments._runtime.spec import ExperimentSpec
from experiments.catalog import SPECIFICATIONS
from experiments.phase1.product_l1 import stages as product_l1_stages
from experiments.phase1.product_l1.evaluation.reaction_specialist_ugi_v0_comparison import (
    _load_topology_policy,
    _validate,
)
from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
    _sample_supported_programs,
)
from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
    _validate as _validate_chemistry_comparison,
)
from forge.core.hashing import sha256_file
from forge.model.reaction_specialization import exact_exposure_schedule
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_joint_lipid_prior import UgiJointLipidPrior
from forge.potency.annotations import ROLE_NAMES

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
    REPO / "experiments/phase1/multireaction/reaction_topology_specialist_ugi_seed0_h100_v2.json"
)
BASE_TRAINING = (
    REPO
    / "runs/phase1-shared-mixed-role-morphology-seed0-h100"
    / "7763bd4f949c7724cf28a75f8cef37bafc31fa775c4c304ce33f1c813154e134"
    / "stages/study/artifacts/training/result.json"
)
CHEMISTRY_CONFIG = REPO / "configs/multireaction/reaction_chemistry_specialist_ugi_seed0_v3.json"
CHEMISTRY_COMPARISON_CONFIG = (
    REPO / "configs/multireaction/ugi_chemistry_specialist_base_comparison_seed0_v3.json"
)
CHEMISTRY_SPEC = (
    REPO / "experiments/phase1/multireaction/ugi_chemistry_specialist_seed0_h100_v3.json"
)
CHEMISTRY_COMPARISON_CONFIG_V4 = (
    REPO / "configs/multireaction/ugi_chemistry_specialist_base_comparison_seed0_recovery_v4.json"
)
CHEMISTRY_SPEC_V4 = (
    REPO / "experiments/phase1/multireaction/"
    "ugi_chemistry_specialist_evaluation_recovery_seed0_h100_v4.json"
)
CONTEXTUAL_CHEMISTRY_CONFIG = (
    REPO / "configs/multireaction/reaction_contextual_chemistry_specialist_ugi_seed0_v4.json"
)
CONTEXTUAL_CHEMISTRY_SPEC = (
    REPO / "experiments/phase1/multireaction/ugi_contextual_chemistry_specialist_seed0_h100_v4.json"
)
JOINT_LIPID_CONFIG = (
    REPO / "configs/multireaction/reaction_joint_lipid_specialist_ugi_seed0_v5.json"
)
JOINT_LIPID_COMPARISON_CONFIG = (
    REPO / "configs/multireaction/ugi_joint_lipid_specialist_base_comparison_seed0_v5.json"
)
JOINT_LIPID_SPEC = (
    REPO / "experiments/phase1/multireaction/ugi_joint_lipid_specialist_seed0_h100_v5.json"
)
ROLE_LOCAL_CONFIG = REPO / "configs/multireaction/reaction_role_local_decoder_ugi_seed0_v6.json"
ROLE_LOCAL_COMPARISON_CONFIG = (
    REPO / "configs/multireaction/ugi_role_local_specialist_base_comparison_seed0_v6.json"
)
ROLE_LOCAL_SPEC = (
    REPO / "experiments/phase1/multireaction/ugi_role_local_decoder_seed0_h100_v6.json"
)
MEASURED_ONLY_FULL_CONFIG = (
    REPO / "configs/multireaction/reaction_measured_only_full_model_ugi_seed0_v7.json"
)
MEASURED_ONLY_FULL_COMPARISON_CONFIG = (
    REPO / "configs/multireaction/ugi_measured_only_full_model_base_comparison_seed0_v7.json"
)
MEASURED_ONLY_FULL_SPEC = (
    REPO / "experiments/phase1/multireaction/ugi_measured_only_full_model_seed0_h100_v7.json"
)
STRUCTURED_TOPOLOGY_CONFIG = (
    REPO / "configs/multireaction/reaction_structured_topology_specialist_ugi_seed0_v8.json"
)
STRUCTURED_TOPOLOGY_COMPARISON_CONFIG = (
    REPO / "configs/multireaction/ugi_structured_topology_specialist_base_comparison_seed0_v8.json"
)
STRUCTURED_TOPOLOGY_SPEC = (
    REPO / "experiments/phase1/multireaction/ugi_structured_topology_specialist_seed0_h100_v8.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_specialist_exposure_targets_match_authenticated_base_evidence() -> None:
    training = _read(BASE_TRAINING)
    observed = training["arms"]["full_role_morphology_transformer"]["examples_seen_by_program"]
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
        assert (len(schedule.optimizer_steps), schedule.final_microbatches) == expected_final[
            program
        ]


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
            path for path in candidates if _read(path).get("experiment_id") == member["experiment"]
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

    assert runtime["current_terminal_decode_policy"] == ("strict_ugi_program_coupled_conditional")
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


def test_chemistry_specialist_freezes_topology_and_has_paired_base_gate() -> None:
    training = _read(
        REPO / "results/phase1/shared_bias_parallel_program_role_seed0_v2/training_result.json"
    )
    config = _read(CHEMISTRY_CONFIG)
    comparison = _read(CHEMISTRY_COMPARISON_CONFIG)
    observed = training["arms"]["shared_bias_program_role_source"]["examples_seen_by_program"][
        "ugi_3cr_agile"
    ]

    assert config["exposure"]["existing_examples"] == observed == 384006
    assert config["exposure"]["target_examples"] - observed == 66464
    assert config["chemistry_specialization"] == {
        "balance_by_role": True,
        "topology_conditioning": "target_parents_and_closure_endpoints",
        "trainable_outputs": ["nodes", "parent_bonds", "closure_bonds"],
    }
    runtime = _validate_chemistry_comparison(comparison, profile="full", device="cuda")
    assert runtime["program_count"] == 3072
    assert comparison["policy"]["paired_random_stream"] is True
    assert comparison["policy"]["repairs_or_retries"] is False

    spec = ExperimentSpec.load(CHEMISTRY_SPEC)
    assert spec.experiment_id == "phase1-ugi-chemistry-specialist-seed0-h100-v3"
    document = _read(CHEMISTRY_SPEC)
    assert [stage["id"] for stage in document["stages"]] == [
        "preflight",
        "specialize_ugi",
        "compare_base",
    ]
    assert document["stages"][1]["needs"] == ["preflight"]
    assert document["stages"][2]["needs"] == ["specialize_ugi"]
    assert SPECIFICATIONS[spec.experiment_id] == str(CHEMISTRY_SPEC.relative_to(REPO))
    for stage in document["stages"]:
        assert stage["resources"]["gpu_type"] == "H100!"
        assert sha256_file(REPO / stage["config"]["path"]) == stage["config"]["sha256"]


def test_chemistry_specialist_v4_draws_only_from_bound_measured_ugi_support() -> None:
    config = _read(CHEMISTRY_COMPARISON_CONFIG_V4)
    runtime = _validate_chemistry_comparison(config, profile="full", device="cuda")
    assert "program_draw" not in config["inputs"]
    assert runtime["program_count"] == 3072
    assert runtime["layout_seed"] == 2026083003

    qualified = REPO / config["inputs"]["qualified_reactions"]["path"]
    assignments = REPO / config["inputs"]["ugi_assignments"]["path"]
    policy = UgiEsterChemotypePolicy.from_qualified_registry(
        qualified,
        training_assignments_path=assignments,
        reaction_id="ugi_3cr_agile",
        expected_sha256=sha256_file(qualified),
        expected_training_assignments_sha256=sha256_file(assignments),
    )
    programs, document = _sample_supported_programs(
        REPO / config["inputs"]["production_cache"]["path"],
        policy,
        count=64,
        seed=int(runtime["layout_seed"]),
    )
    assert document["rows"] == len(programs) == 64
    assert document["source"] == "measured_training_joint_count_prior"
    for program in programs:
        for index, role in enumerate(ROLE_NAMES):
            assert policy.minimum_topology_exterior_atoms(role) <= program.node_counts[index]
            assert program.node_counts[index] <= policy.maximum_exterior_atoms(role)

    spec = ExperimentSpec.load(CHEMISTRY_SPEC_V4)
    assert spec.experiment_id == "phase1-ugi-chemistry-specialist-evaluation-recovery-seed0-h100-v4"
    document = _read(CHEMISTRY_SPEC_V4)
    assert [stage["id"] for stage in document["stages"]] == [
        "comparison_preflight",
        "compare_base",
    ]
    assert document["stages"][0]["needs"] == []
    assert document["stages"][1]["needs"] == ["comparison_preflight"]
    assert document["metadata"]["training_calls"] == 0
    assert SPECIFICATIONS[spec.experiment_id] == str(CHEMISTRY_SPEC_V4.relative_to(REPO))
    for stage in document["stages"]:
        assert stage["resources"]["gpu_type"] == "H100!"
        assert sha256_file(REPO / stage["config"]["path"]) == stage["config"]["sha256"]


def test_contextual_chemistry_specialist_is_fail_closed_and_topology_frozen() -> None:
    config = _read(CONTEXTUAL_CHEMISTRY_CONFIG)
    assert config["schema_version"].endswith("contextual_chemistry_specialization_config.v4")
    assert config["contextual_chemistry_layers"] == 2
    assert config["contextual_chemistry_adapter_dim"] == 128
    assert config["chemistry_specialization"]["context_features"] == [
        "precursor_role",
        "reaction_core_position",
        "typed_one_hop_tree_neighbors",
        "typed_closure_neighbors",
        "heavy_degree",
    ]
    assert config["exposure"]["target_examples"] - config["exposure"]["existing_examples"] == 66464

    spec = ExperimentSpec.load(CONTEXTUAL_CHEMISTRY_SPEC)
    assert spec.experiment_id == "phase1-ugi-contextual-chemistry-specialist-seed0-h100-v4"
    document = _read(CONTEXTUAL_CHEMISTRY_SPEC)
    assert [stage["id"] for stage in document["stages"]] == [
        "preflight",
        "comparison_preflight",
        "specialize_ugi",
        "compare_base",
    ]
    assert document["stages"][1]["needs"] == ["preflight"]
    assert document["stages"][2]["needs"] == ["comparison_preflight"]
    assert document["stages"][3]["needs"] == ["specialize_ugi"]
    assert SPECIFICATIONS[spec.experiment_id] == str(CONTEXTUAL_CHEMISTRY_SPEC.relative_to(REPO))
    for stage in document["stages"]:
        assert stage["resources"]["gpu_type"] == "H100!"
        assert sha256_file(REPO / stage["config"]["path"]) == stage["config"]["sha256"]


def test_joint_lipid_specialist_uses_identity_free_measured_exploration_mixture() -> None:
    config = _read(JOINT_LIPID_CONFIG)
    comparison = _read(JOINT_LIPID_COMPARISON_CONFIG)
    policy = config["joint_lipid_prior"]
    assert policy == {
        "component_ids_enter_neural_tensors": False,
        "exploration_mass": 0.5,
        "exploration_weighting": "frozen_source_weights",
        "measured_mass": 0.5,
        "measured_weighting": "uniform_unique_constitutional_product",
    }
    prior = UgiJointLipidPrior.from_training_assignments(
        REPO / config["inputs"]["ugi_assignments"]["path"],
        measured_mass=policy["measured_mass"],
        exploration_mass=policy["exploration_mass"],
        expected_sha256=config["inputs"]["ugi_assignments"]["sha256"],
    )
    assert prior.to_mapping()["measured_training_products"] == 480
    assert prior.to_mapping()["component_identity_conditioning"] is False
    runtime = _validate_chemistry_comparison(comparison, profile="full", device="cuda")
    assert runtime["program_count"] == 3072

    spec = ExperimentSpec.load(JOINT_LIPID_SPEC)
    assert spec.experiment_id == "phase1-ugi-joint-lipid-specialist-seed0-h100-v5"
    document = _read(JOINT_LIPID_SPEC)
    assert [stage["id"] for stage in document["stages"]] == [
        "preflight",
        "comparison_preflight",
        "specialize_ugi",
        "compare_base",
    ]
    assert document["stages"][1]["needs"] == ["preflight"]
    assert document["stages"][2]["needs"] == ["comparison_preflight"]
    assert document["stages"][3]["needs"] == [
        "specialize_ugi",
        "comparison_preflight",
    ]
    assert SPECIFICATIONS[spec.experiment_id] == str(JOINT_LIPID_SPEC.relative_to(REPO))
    for stage in document["stages"]:
        assert stage["resources"]["gpu_type"] == "H100!"
        assert sha256_file(REPO / stage["config"]["path"]) == stage["config"]["sha256"]


def test_role_local_decoder_targets_attributed_ugi_topology_failure() -> None:
    config = _read(ROLE_LOCAL_CONFIG)
    comparison = _read(ROLE_LOCAL_COMPARISON_CONFIG)
    assert config["schema_version"] == (
        "forge.reaction_program_role_local_specialization_config.v6"
    )
    assert config["role_local_decoder"] == {
        "core_updated": False,
        "cross_role_attention": True,
        "local_context": [
            "node_hidden",
            "parent_hidden",
            "mean_child_hidden",
            "cross_role_summary",
        ],
        "trainable_outputs": [
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        ],
    }
    assert config["exposure"]["target_examples"] - config["exposure"]["existing_examples"] == 66464
    runtime = _validate_chemistry_comparison(comparison, profile="full", device="cuda")
    assert runtime["program_count"] == 3072
    assert comparison["policy"]["paired_program_order"] is True
    assert comparison["policy"]["repairs_or_retries"] is False

    spec = ExperimentSpec.load(ROLE_LOCAL_SPEC)
    assert spec.experiment_id == "phase1-ugi-role-local-decoder-seed0-h100-v6"
    document = _read(ROLE_LOCAL_SPEC)
    assert [stage["id"] for stage in document["stages"]] == [
        "preflight",
        "comparison_preflight",
        "specialize_ugi",
        "compare_base",
    ]
    assert document["stages"][1]["needs"] == ["preflight"]
    assert document["stages"][2]["needs"] == ["comparison_preflight"]
    assert document["stages"][3]["needs"] == [
        "specialize_ugi",
        "comparison_preflight",
    ]
    assert SPECIFICATIONS[spec.experiment_id] == str(ROLE_LOCAL_SPEC.relative_to(REPO))
    for stage in document["stages"]:
        assert stage["resources"]["gpu_type"] == "H100!"
        assert sha256_file(REPO / stage["config"]["path"]) == stage["config"]["sha256"]


def test_measured_only_full_model_diagnostic_has_exact_train_measure_and_scope() -> None:
    config = _read(MEASURED_ONLY_FULL_CONFIG)
    comparison = _read(MEASURED_ONLY_FULL_COMPARISON_CONFIG)
    assert config["schema_version"] == (
        "forge.reaction_program_measured_only_full_finetune_config.v7"
    )
    assert config["measured_only_finetune"] == {
        "initialization": "authenticated_shared_checkpoint",
        "optimizer_state": "reset",
        "training_rows": "source_adjudicated_measured_train_only",
        "trainable_scope": "all_model_parameters",
    }
    policy = config["joint_lipid_prior"]
    assert policy["measured_mass"] == 1.0
    assert policy["exploration_mass"] == 0.0
    assert policy["component_ids_enter_neural_tensors"] is False
    assert config["exposure"]["target_examples"] - config["exposure"]["existing_examples"] == 66464

    prior = UgiJointLipidPrior.from_training_assignments(
        REPO / config["inputs"]["ugi_assignments"]["path"],
        measured_mass=policy["measured_mass"],
        exploration_mass=policy["exploration_mass"],
        expected_sha256=config["inputs"]["ugi_assignments"]["sha256"],
    )
    prior_document = prior.to_mapping()
    assert prior_document["measured_training_products"] == 480
    assert prior_document["measured_mass"] == 1.0
    assert prior_document["exploration_mass"] == 0.0
    assert prior_document["component_identity_conditioning"] is False

    runtime = _validate_chemistry_comparison(comparison, profile="full", device="cuda")
    assert runtime["program_count"] == 3072
    assert comparison["policy"]["paired_program_order"] is True
    assert comparison["policy"]["repairs_or_retries"] is False

    spec = ExperimentSpec.load(MEASURED_ONLY_FULL_SPEC)
    assert spec.experiment_id == "phase1-ugi-measured-only-full-model-seed0-h100-v7"
    document = _read(MEASURED_ONLY_FULL_SPEC)
    assert [stage["id"] for stage in document["stages"]] == [
        "preflight",
        "comparison_preflight",
        "specialize_ugi",
        "compare_base",
    ]
    assert document["stages"][1]["needs"] == ["preflight"]
    assert document["stages"][2]["needs"] == ["comparison_preflight"]
    assert document["stages"][3]["needs"] == [
        "specialize_ugi",
        "comparison_preflight",
    ]
    assert document["metadata"]["measured_training_products"] == 480
    assert document["metadata"]["full_model_trainable"] is True
    assert SPECIFICATIONS[spec.experiment_id] == str(MEASURED_ONLY_FULL_SPEC.relative_to(REPO))
    for stage in document["stages"]:
        assert stage["resources"]["gpu_type"] == "H100!"
        assert sha256_file(REPO / stage["config"]["path"]) == stage["config"]["sha256"]


def test_structured_topology_specialist_trains_only_inside_the_ugi_grammar() -> None:
    config = _read(STRUCTURED_TOPOLOGY_CONFIG)
    comparison = _read(STRUCTURED_TOPOLOGY_COMPARISON_CONFIG)
    assert config["schema_version"] == (
        "forge.reaction_program_structured_topology_specialization_config.v8"
    )
    assert config["structured_topology_specialization"] == {
        "maximum_children": 3,
        "tree_weight": 1.0,
        "closure_weight": 1.0,
        "decoder": "exact_ugi_role_grammar",
        "support_coordinates": [
            "node_count",
            "junction_budget",
            "cycle_rank",
            "attachment_count",
            "maximum_adjacent_branch_run",
            "feasible_ring_edges",
        ],
    }
    assert config["exposure"]["target_examples"] - config["exposure"]["existing_examples"] == 66464
    runtime = _validate_chemistry_comparison(comparison, profile="full", device="cuda")
    assert runtime["program_count"] == 3072
    assert comparison["profiles"]["h100_preflight"]["program_count"] == 256
    assert comparison["policy"]["paired_program_order"] is True
    assert comparison["policy"]["repairs_or_retries"] is False

    spec = ExperimentSpec.load(STRUCTURED_TOPOLOGY_SPEC)
    assert spec.experiment_id == "phase1-ugi-structured-topology-specialist-seed0-h100-v8"
    document = _read(STRUCTURED_TOPOLOGY_SPEC)
    assert [stage["id"] for stage in document["stages"]] == [
        "preflight",
        "specialize_ugi",
        "compare_base",
    ]
    assert document["stages"][1]["needs"] == ["preflight"]
    assert document["stages"][2]["needs"] == ["specialize_ugi"]
    assert document["metadata"]["frozen_shared_transformer"] is True
    assert document["metadata"]["frozen_chemistry_outputs"] is True
    assert document["metadata"]["grammar_normalized_topology"] is True
    assert SPECIFICATIONS[spec.experiment_id] == str(STRUCTURED_TOPOLOGY_SPEC.relative_to(REPO))
    for stage in document["stages"]:
        assert stage["resources"]["gpu_type"] == "H100!"
        assert sha256_file(REPO / stage["config"]["path"]) == stage["config"]["sha256"]


def test_chemistry_specialist_recovery_uses_real_preflight_and_authenticates_gate(
    monkeypatch,
) -> None:
    calls: list[dict] = []

    def capture(context, **kwargs):
        calls.append(kwargs)
        return "captured"

    monkeypatch.setattr(
        product_l1_stages,
        "_ugi_chemistry_specialist_base_comparison_stage",
        capture,
    )
    sentinel = object()
    assert (
        product_l1_stages.evaluate_ugi_chemistry_specialist_base_comparison_preflight_recovery_stage(
            sentinel
        )
        == "captured"
    )
    assert calls.pop()["run_profile"] == "h100_preflight"
    assert (
        product_l1_stages.evaluate_ugi_chemistry_specialist_base_comparison_recovery_stage(sentinel)
        == "captured"
    )
    assert calls.pop()["gate_dependency_stage"] == "comparison_preflight"

    config = _read(CHEMISTRY_COMPARISON_CONFIG_V4)
    passing = {
        "status": "complete",
        "profile": "h100_preflight",
        "programs_per_method": 64,
        "preflight_gates": {
            "base_valid_fraction_sufficient": True,
            "specialist_valid_fraction_sufficient": True,
            "c2st_estimable": True,
            "program_support_abstentions_zero": True,
        },
    }
    product_l1_stages._require_successful_ugi_chemistry_preflight(config, passing)
    failing = json.loads(json.dumps(passing))
    failing["preflight_gates"]["program_support_abstentions_zero"] = False
    with pytest.raises(StageError, match="64-program h100_preflight"):
        product_l1_stages._require_successful_ugi_chemistry_preflight(config, failing)
