from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.phase1.product_l1.evaluation.ugi_all_role_semantic_program_comparison import (
    UgiAllRoleSemanticProgramComparisonError,
    _load_draw,
    _local_chemistry_method_id,
    _runtime,
    _semantic_support_audit,
)
from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
    _topology_policy,
)
from forge.core.hashing import sha256_file
from forge.core.io import read_json_object
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_mog_semantic_guidance import UgiMogSemanticGuidancePolicy

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/ugi_all_role_semantic_program_comparison_seed0_v1.json"
JOINT_REALISM_CONFIG = (
    REPO / "configs/multireaction/ugi_mog_joint_realism_program_comparison_seed0_v2.json"
)
LOCAL_CHEMISTRY_CONFIG = (
    REPO / "configs/multireaction/ugi_local_chemistry_mog_vs_complete_semantic_seed0_v1.json"
)
ATOM_LOCAL_CHEMISTRY_CONFIG = (
    REPO / "configs/multireaction/ugi_atom_local_chemistry_mog_vs_complete_semantic_seed0_v1.json"
)
ATOM_TRUST_REGION_CONFIG = (
    REPO / "configs/multireaction/ugi_atom_trust_region_mog_vs_complete_semantic_seed0_v1.json"
)
CONTEXT_SUPPORT_CONFIG = (
    REPO / "configs/multireaction/ugi_context_support_mog_vs_complete_semantic_seed0_v1.json"
)
WHOLE_HEAD_SUPPORT_CONFIG = (
    REPO / "configs/multireaction/ugi_whole_head_support_mog_vs_complete_semantic_seed0_v1.json"
)
BINARY_WHOLE_HEAD_SUPPORT_CONFIG = (
    REPO
    / "configs/multireaction/ugi_binary_whole_head_support_mog_vs_complete_semantic_seed0_v1.json"
)
WHOLE_HEAD_TRAJECTORY_TRUST_REGION_CONFIG = (
    REPO
    / "configs/multireaction/"
    "ugi_whole_head_trajectory_trust_region_vs_complete_semantic_seed0_v1.json"
)
WHOLE_HEAD_TRAJECTORY_TRUST_REGION_V2_CONFIG = (
    REPO
    / "configs/multireaction/"
    "ugi_whole_head_trajectory_trust_region_vs_complete_semantic_seed0_v2.json"
)
DRAW = REPO / "results/phase1/ugi_all_role_semantic_program_draw_seed0_v1/program_draw.json"


def test_all_role_comparison_freezes_the_no_selection_contract() -> None:
    config = json.loads(CONFIG.read_text())
    runtime = _runtime(config, profile="smoke", device="cpu")

    assert runtime["program_count"] == 16
    assert config["policy"]["component_identity_conditioning"] is False
    assert config["policy"]["repairs_or_retries"] is False
    assert config["preflight_gate"]["exact_l1_absolute_minimum"] == 0.95


def test_all_role_comparison_rejects_relaxed_exact_l1_gate() -> None:
    config = json.loads(CONFIG.read_text())
    changed = copy.deepcopy(config)
    changed["preflight_gate"]["exact_l1_absolute_minimum"] = 0.5

    with pytest.raises(UgiAllRoleSemanticProgramComparisonError, match="gate changed"):
        _runtime(changed, profile="smoke", device="cpu")


def test_joint_realism_comparison_pairs_the_same_mog_decoder_and_random_seed() -> None:
    config = json.loads(JOINT_REALISM_CONFIG.read_text())
    runtime = _runtime(config, profile="smoke", device="cpu")

    assert runtime["baseline_terminal_decode_policy"] == runtime["treatment_terminal_decode_policy"]
    assert runtime["baseline_terminal_decoder_seed"] == runtime["treatment_terminal_decoder_seed"]
    assert "joint_realism_rank_weight" not in config["baseline_semantic_guidance"]
    assert config["semantic_guidance"]["joint_realism_rank_weight"] > 0


def test_joint_realism_comparison_rejects_realism_in_the_baseline() -> None:
    config = json.loads(JOINT_REALISM_CONFIG.read_text())
    changed = copy.deepcopy(config)
    changed["baseline_semantic_guidance"]["joint_realism_rank_weight"] = 1.0
    changed["baseline_semantic_guidance"]["joint_realism_bandwidth"] = 1.0

    with pytest.raises(UgiAllRoleSemanticProgramComparisonError, match="baseline cannot"):
        _runtime(changed, profile="smoke", device="cpu")


def test_local_chemistry_comparison_pairs_complete_semantics_and_hash_pinned_inputs() -> None:
    config = json.loads(LOCAL_CHEMISTRY_CONFIG.read_text())
    runtime = _runtime(config, profile="smoke", device="cpu")

    assert runtime["baseline_terminal_decode_policy"] == runtime["treatment_terminal_decode_policy"]
    assert runtime["baseline_terminal_decoder_seed"] == runtime["treatment_terminal_decoder_seed"]
    assert "local_chemistry_rank_weight" not in config["baseline_semantic_guidance"]
    assert config["semantic_guidance"]["local_chemistry_rank_weight"] > 0
    assert set(("semantic_atoms", "semantic_bonds")).issubset(config["inputs"])


def test_local_chemistry_comparison_rejects_missing_train_reference() -> None:
    config = json.loads(LOCAL_CHEMISTRY_CONFIG.read_text())
    changed = copy.deepcopy(config)
    del changed["inputs"]["semantic_atoms"]

    with pytest.raises(UgiAllRoleSemanticProgramComparisonError, match="config changed"):
        _runtime(changed, profile="smoke", device="cpu")


def test_atom_local_chemistry_has_distinct_identity_and_disables_bond_ranking() -> None:
    config = json.loads(ATOM_LOCAL_CHEMISTRY_CONFIG.read_text())
    policy = UgiMogSemanticGuidancePolicy.from_mapping(config["semantic_guidance"])

    assert policy.uses_local_chemistry_atoms is True
    assert policy.uses_local_chemistry_bonds is False
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_atom_local_chemistry_complete_semantic_ugi_program"
    )


def test_atom_trust_region_has_distinct_identity_and_frozen_radius() -> None:
    config = json.loads(ATOM_TRUST_REGION_CONFIG.read_text())
    policy = UgiMogSemanticGuidancePolicy.from_mapping(config["semantic_guidance"])

    assert policy.uses_local_chemistry_atoms is True
    assert policy.uses_local_chemistry_bonds is False
    assert policy.local_chemistry_atom_total_variation_radius == 0.05
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_atom_trust_region_complete_semantic_ugi_program"
    )


def test_context_support_has_distinct_identity_and_no_frequency_reward() -> None:
    config = json.loads(CONTEXT_SUPPORT_CONFIG.read_text())
    policy = UgiMogSemanticGuidancePolicy.from_mapping(config["semantic_guidance"])

    assert policy.local_chemistry_score_mode == "support_tier"
    assert policy.uses_local_chemistry_atoms is True
    assert policy.uses_local_chemistry_bonds is True
    assert policy.local_chemistry_atom_total_variation_radius is None
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_context_support_complete_semantic_ugi_program"
    )


def test_whole_head_support_has_distinct_identity_and_single_head_objective() -> None:
    config = json.loads(WHOLE_HEAD_SUPPORT_CONFIG.read_text())
    policy = UgiMogSemanticGuidancePolicy.from_mapping(config["semantic_guidance"])

    assert policy.local_chemistry_score_mode == "whole_head_support_tier"
    assert policy.uses_local_chemistry_atoms is True
    assert policy.uses_coordinate_local_chemistry_atoms is False
    assert policy.uses_local_chemistry_bonds is True
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_whole_head_support_complete_semantic_ugi_program"
    )
    _runtime(config, profile="smoke", device="cpu")


def test_tiered_whole_head_group_entropy_guard_has_distinct_identity() -> None:
    config = json.loads(WHOLE_HEAD_SUPPORT_CONFIG.read_text())
    config["semantic_guidance"]["whole_head_group_effective_count_retention"] = 0.99
    policy = UgiMogSemanticGuidancePolicy.from_mapping(config["semantic_guidance"])

    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_whole_head_support_complete_semantic_ugi_program_grp0990"
    )


def test_binary_whole_head_support_has_distinct_identity_and_no_specificity_reward() -> None:
    config = json.loads(BINARY_WHOLE_HEAD_SUPPORT_CONFIG.read_text())
    policy = UgiMogSemanticGuidancePolicy.from_mapping(config["semantic_guidance"])

    assert policy.local_chemistry_score_mode == "whole_head_support_binary"
    assert policy.uses_local_chemistry_atoms is True
    assert policy.uses_coordinate_local_chemistry_atoms is False
    assert policy.uses_local_chemistry_bonds is True
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_binary_whole_head_support_complete_semantic_ugi_program"
    )
    _runtime(config, profile="smoke", device="cpu")


def test_whole_head_trajectory_trust_region_has_distinct_identity() -> None:
    config = json.loads(WHOLE_HEAD_TRAJECTORY_TRUST_REGION_CONFIG.read_text())
    policy = UgiMogSemanticGuidancePolicy.from_mapping(config["semantic_guidance"])

    assert policy.local_chemistry_score_mode == "whole_head_support_binary"
    assert policy.whole_head_total_variation_radius == pytest.approx(0.05)
    assert policy.local_chemistry_atom_total_variation_radius is None
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_binary_whole_head_tv050_hard_bond_complete_semantic_ugi_program"
    )
    _runtime(config, profile="smoke", device="cpu")


def test_whole_head_trajectory_trust_region_v2_does_not_rank_bonds() -> None:
    config = json.loads(WHOLE_HEAD_TRAJECTORY_TRUST_REGION_V2_CONFIG.read_text())
    policy = UgiMogSemanticGuidancePolicy.from_mapping(config["semantic_guidance"])

    assert policy.whole_head_total_variation_radius == pytest.approx(0.05)
    assert policy.uses_local_chemistry_atoms is True
    assert policy.uses_local_chemistry_bonds is False
    assert policy.effective_local_chemistry_bond_rank_weight == 0.0
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_binary_whole_head_tv050_hard_bond_complete_semantic_ugi_program"
    )
    _runtime(config, profile="smoke", device="cpu")


def test_whole_head_soft_bond_attribution_has_distinct_identity() -> None:
    config = json.loads(
        (
            REPO
            / "configs/multireaction/"
            "ugi_whole_head_trajectory_trust_region_cpu_attribution_seed0_v3.json"
        ).read_text()
    )
    policy = UgiMogSemanticGuidancePolicy.from_mapping(config["semantic_guidance"])

    assert policy.whole_head_hard_bond_support is False
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_binary_whole_head_tv050_soft_bond_complete_semantic_ugi_program"
    )
    _runtime(config, profile="h100_preflight", device="cpu")


def test_whole_head_entropy_guard_has_distinct_identity() -> None:
    config = json.loads(
        (
            REPO
            / "configs/multireaction/"
            "ugi_whole_head_entropy_guard_cpu_prescreen_seed0_v4.json"
        ).read_text()
    )
    policy = UgiMogSemanticGuidancePolicy.from_mapping(config["semantic_guidance"])

    assert policy.whole_head_candidate_effective_count_retention == pytest.approx(0.99)
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_binary_whole_head_tv050_eff0990_hard_bond"
        "_complete_semantic_ugi_program"
    )
    _runtime(config, profile="h100_preflight", device="cpu")


def test_whole_head_group_entropy_guard_has_distinct_identity() -> None:
    config = json.loads(
        (
            REPO
            / "configs/multireaction/"
            "ugi_whole_head_group_entropy_cpu_prescreen_seed0_v5.json"
        ).read_text()
    )
    policy = UgiMogSemanticGuidancePolicy.from_mapping(config["semantic_guidance"])

    assert policy.whole_head_group_effective_count_retention == pytest.approx(0.99)
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_binary_whole_head_tv050_grp0990_hard_bond"
        "_complete_semantic_ugi_program"
    )
    _runtime(config, profile="h100_preflight", device="cpu")


def test_whole_head_distance_prescreen_has_distinct_identity() -> None:
    config = json.loads(
        (
            REPO
            / "configs/multireaction/ugi_whole_head_distance_cpu_prescreen_seed0_v6.json"
        ).read_text()
    )
    policy = UgiMogSemanticGuidancePolicy.from_mapping(config["semantic_guidance"])

    assert policy.local_chemistry_score_mode == "whole_head_support_distance"
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_whole_head_distance_tv050_complete_semantic_ugi_program"
    )
    _runtime(config, profile="h100_preflight", device="cpu")


def test_donor_slack_is_explicit_in_method_identity() -> None:
    config = json.loads(
        (
            REPO
            / "configs/multireaction/"
            "ugi_donor_slack_head_distance_topology_vs_amine_joint_support_cpu_seed0_v1.json"
        ).read_text()
    )
    policy = UgiMogSemanticGuidancePolicy.from_mapping(config["semantic_guidance"])

    assert policy.amine_hydrogen_bond_donors_tolerance == 1
    assert policy.amine_heavy_branch_atoms_tolerance == 0
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_whole_head_distance_complete_semantic_ugi_program"
        "_toposupport_donorslack1"
    )
    _runtime(config, profile="h100_preflight", device="cpu")


def test_all_role_draw_loads_paired_baseline_and_treatment_targets() -> None:
    programs, baseline, treatment = _load_draw(DRAW, count=16)

    assert len(programs) == len(baseline) == len(treatment) == 16
    assert all(target.amine.nitrogen_atoms >= 1 for target in treatment)
    assert all(target.tail_pair.aldehyde_ester_short_side_carbons == 6 for target in treatment)


def test_all_role_smoke_draw_has_nonempty_declared_support() -> None:
    config = json.loads(CONFIG.read_text())
    inputs = {label: REPO / value["path"] for label, value in config["inputs"].items()}
    programs, _, treatment = _load_draw(DRAW, count=16)
    ester_policy = UgiEsterChemotypePolicy.from_qualified_registry(
        inputs["qualified_reactions"],
        training_assignments_path=inputs["ugi_assignments"],
        reaction_id="ugi_3cr_agile",
        expected_sha256=sha256_file(inputs["qualified_reactions"]),
        expected_training_assignments_sha256=sha256_file(inputs["ugi_assignments"]),
    )
    audit = _semantic_support_audit(
        programs,
        treatment,
        topology_policy=_topology_policy(inputs),
        ester_policy=ester_policy,
        local_chemistry_support=LocalChemistrySupport.from_mapping(
            read_json_object(
                inputs["role_morphology_policy"],
                error=ValueError,
                label="role-local chemistry support",
            )
        ),
    )

    assert audit["all_pairs_supported"] is True
    assert audit["minimum_amine_topologies_per_pair"] >= 1
    assert audit["minimum_aldehyde_topologies_per_pair"] >= 1


def test_local_chemistry_method_id_records_nondefault_entropy_floor() -> None:
    policy = UgiMogSemanticGuidancePolicy(
        amine_heavy_atom_graph_diameter_tolerance=0,
        amine_carbon_skeleton_diameter_tolerance=0,
        aldehyde_ester_side_carbons_tolerance=0,
        model_rank_weight=1.0,
        semantic_rank_weight=1.0,
        rank_temperature=0.75,
        uniform_probability_mass=0.35,
        local_chemistry_rank_weight=1.0,
        local_chemistry_score_mode="whole_head_support_binary",
        local_chemistry_bond_maximum_depth_bucket=64,
        local_chemistry_unsaturation_position_only=True,
        tail_unsaturation_count_strategy="frequency_resampled",
        whole_head_topology_support=True,
    )
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_binary_whole_head_support_complete_semantic_ugi_program"
        "_bonddepth064_unsatpos_toposupport_frequencyresampled_unif0350"
    )


def test_local_chemistry_method_id_records_tiered_topology_support() -> None:
    policy = UgiMogSemanticGuidancePolicy(
        amine_heavy_atom_graph_diameter_tolerance=0,
        amine_carbon_skeleton_diameter_tolerance=0,
        aldehyde_ester_side_carbons_tolerance=0,
        model_rank_weight=1.0,
        semantic_rank_weight=1.0,
        rank_temperature=0.75,
        uniform_probability_mass=0.35,
        local_chemistry_rank_weight=1.0,
        local_chemistry_score_mode="whole_head_support_tier",
        whole_head_topology_support=True,
        whole_head_topology_score_mode="tier",
    )
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_whole_head_support_complete_semantic_ugi_program"
        "_toposupporttier_unif0350"
    )


def test_local_chemistry_method_id_distinguishes_topology_only_context_support() -> None:
    policy = UgiMogSemanticGuidancePolicy(
        amine_heavy_atom_graph_diameter_tolerance=0,
        amine_carbon_skeleton_diameter_tolerance=0,
        aldehyde_ester_side_carbons_tolerance=0,
        model_rank_weight=1.0,
        semantic_rank_weight=1.0,
        rank_temperature=0.75,
        uniform_probability_mass=0.25,
        local_chemistry_rank_weight=1.0,
        local_chemistry_score_mode="support_tier",
        whole_head_topology_support=True,
    )
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_context_support_complete_semantic_ugi_program_toposupport"
    )


def test_local_chemistry_method_id_distinguishes_topology_ranked_chemistry_argmax() -> None:
    policy = UgiMogSemanticGuidancePolicy(
        amine_heavy_atom_graph_diameter_tolerance=0,
        amine_carbon_skeleton_diameter_tolerance=0,
        aldehyde_ester_side_carbons_tolerance=0,
        model_rank_weight=1.0,
        semantic_rank_weight=1.0,
        rank_temperature=0.75,
        uniform_probability_mass=0.25,
        local_chemistry_rank_weight=1.0,
        local_chemistry_score_mode="support_tier",
        whole_head_topology_support=True,
        terminal_chemistry_selection_mode="model_argmax",
    )
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_context_support_complete_semantic_ugi_program"
        "_toposupport_chemargmax"
    )


def test_local_chemistry_method_id_distinguishes_topology_only_rank_objective() -> None:
    policy = UgiMogSemanticGuidancePolicy(
        amine_heavy_atom_graph_diameter_tolerance=0,
        amine_carbon_skeleton_diameter_tolerance=0,
        aldehyde_ester_side_carbons_tolerance=0,
        model_rank_weight=1.0,
        semantic_rank_weight=1.0,
        rank_temperature=0.75,
        uniform_probability_mass=0.25,
        local_chemistry_rank_weight=0.0,
        local_chemistry_score_mode="support_tier",
        whole_head_topology_support=True,
        whole_head_topology_rank_weight=1.0,
    )
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_context_support_complete_semantic_ugi_program"
        "_toposupport_topow1000"
    )


def test_local_chemistry_method_id_distinguishes_topology_only_with_bond_argmax() -> None:
    policy = UgiMogSemanticGuidancePolicy(
        amine_heavy_atom_graph_diameter_tolerance=0,
        amine_carbon_skeleton_diameter_tolerance=0,
        aldehyde_ester_side_carbons_tolerance=0,
        model_rank_weight=1.0,
        semantic_rank_weight=1.0,
        rank_temperature=0.75,
        uniform_probability_mass=0.25,
        local_chemistry_rank_weight=0.0,
        local_chemistry_score_mode="support_tier",
        whole_head_topology_support=True,
        whole_head_topology_rank_weight=1.0,
        terminal_bond_selection_mode="model_argmax",
    )
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_context_support_complete_semantic_ugi_program"
        "_toposupport_topow1000_bondargmax"
    )


def test_local_chemistry_method_id_records_nondefault_chemistry_weight() -> None:
    policy = UgiMogSemanticGuidancePolicy(
        amine_heavy_atom_graph_diameter_tolerance=0,
        amine_carbon_skeleton_diameter_tolerance=0,
        aldehyde_ester_side_carbons_tolerance=0,
        model_rank_weight=1.0,
        semantic_rank_weight=1.0,
        rank_temperature=0.75,
        uniform_probability_mass=0.35,
        local_chemistry_rank_weight=2.0,
        local_chemistry_score_mode="whole_head_support_binary",
        local_chemistry_bond_maximum_depth_bucket=64,
        local_chemistry_unsaturation_position_only=True,
        tail_unsaturation_count_strategy="frequency_resampled",
        whole_head_topology_support=True,
    )
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_binary_whole_head_support_complete_semantic_ugi_program"
        "_bonddepth064_unsatpos_toposupport_frequencyresampled_unif0350_chemw2000"
    )


def test_local_chemistry_method_id_records_bond_specific_entropy_and_weight() -> None:
    policy = UgiMogSemanticGuidancePolicy(
        amine_heavy_atom_graph_diameter_tolerance=0,
        amine_carbon_skeleton_diameter_tolerance=0,
        aldehyde_ester_side_carbons_tolerance=0,
        model_rank_weight=1.0,
        semantic_rank_weight=1.0,
        rank_temperature=0.75,
        uniform_probability_mass=0.35,
        local_chemistry_rank_weight=1.0,
        local_chemistry_bond_rank_weight=2.0,
        local_chemistry_bond_uniform_probability_mass=0.05,
        local_chemistry_score_mode="whole_head_support_binary",
        local_chemistry_bond_maximum_depth_bucket=64,
        local_chemistry_unsaturation_position_only=True,
        tail_unsaturation_count_strategy="frequency_resampled",
        whole_head_topology_support=True,
    )
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_binary_whole_head_support_complete_semantic_ugi_program"
        "_bonddepth064_unsatpos_toposupport_frequencyresampled_unif0350"
        "_bondunif0050_bondw2000"
    )


def test_local_chemistry_method_id_records_terminal_offset_position_basis() -> None:
    policy = UgiMogSemanticGuidancePolicy(
        amine_heavy_atom_graph_diameter_tolerance=0,
        amine_carbon_skeleton_diameter_tolerance=0,
        aldehyde_ester_side_carbons_tolerance=0,
        model_rank_weight=1.0,
        semantic_rank_weight=1.0,
        rank_temperature=0.75,
        uniform_probability_mass=0.35,
        local_chemistry_rank_weight=1.0,
        local_chemistry_bond_rank_weight=1.0,
        local_chemistry_bond_uniform_probability_mass=0.3,
        local_chemistry_score_mode="whole_head_support_binary",
        local_chemistry_bond_maximum_depth_bucket=64,
        local_chemistry_unsaturation_position_only=True,
        local_chemistry_bond_position_basis="terminal_offset",
        tail_unsaturation_count_strategy="frequency_resampled",
        whole_head_topology_support=True,
    )
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_binary_whole_head_support_complete_semantic_ugi_program"
        "_bonddepth064_unsatpos_terminaloffset_toposupport_frequencyresampled_unif0350"
        "_bondunif0300"
    )


def test_local_chemistry_method_id_records_joint_terminal_pattern() -> None:
    policy = UgiMogSemanticGuidancePolicy(
        amine_heavy_atom_graph_diameter_tolerance=0,
        amine_carbon_skeleton_diameter_tolerance=0,
        aldehyde_ester_side_carbons_tolerance=0,
        model_rank_weight=1.0,
        semantic_rank_weight=1.0,
        rank_temperature=0.75,
        uniform_probability_mass=0.35,
        local_chemistry_rank_weight=1.0,
        local_chemistry_bond_rank_weight=1.0,
        local_chemistry_bond_uniform_probability_mass=0.25,
        local_chemistry_score_mode="whole_head_support_binary",
        local_chemistry_bond_maximum_depth_bucket=64,
        local_chemistry_unsaturation_position_only=True,
        local_chemistry_bond_position_basis="terminal_offset",
        tail_unsaturation_count_strategy="frequency_resampled",
        tail_unsaturation_position_strategy="measured_joint_terminal_pattern",
        whole_head_topology_support=True,
    )
    assert _local_chemistry_method_id(policy) == (
        "forge_seed0_mog_binary_whole_head_support_complete_semantic_ugi_program"
        "_bonddepth064_unsatpos_terminaloffset_toposupport_frequencyresampled"
        "_measuredjointterminalpattern_unif0350_bondunif0250"
    )
