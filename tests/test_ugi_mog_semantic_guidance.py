from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from forge.model.defog_feasibility import AtomState
from forge.model.synthesis_program_sampling import _sample_mog_chemistry_allowed
from forge.model.ugi_all_role_semantic_program import (
    UgiAllRoleSemanticTarget,
    UgiMeasuredJointAllRoleSemanticPrior,
    UgiTailPairSemanticTarget,
)
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget
from forge.model.ugi_mog_semantic_guidance import (
    UgiMogSemanticGuidanceError,
    UgiMogSemanticGuidancePolicy,
    replace_tail_unsaturation_semantics,
)
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.model.ugi_role_chemistry_prior import UgiRoleChemistryPrior


def _policy(**overrides: object) -> UgiMogSemanticGuidancePolicy:
    values: dict[str, object] = {
        "amine_heavy_atom_graph_diameter_tolerance": 1,
        "amine_carbon_skeleton_diameter_tolerance": 1,
        "aldehyde_ester_side_carbons_tolerance": 1,
        "model_rank_weight": 1.0,
        "semantic_rank_weight": 1.0,
        "rank_temperature": 0.75,
        "uniform_probability_mass": 0.25,
    }
    values.update(overrides)
    return UgiMogSemanticGuidancePolicy.from_mapping(values)


def _joint_policy() -> UgiMogSemanticGuidancePolicy:
    return _policy(
        joint_realism_rank_weight=1.0,
        joint_realism_bandwidth=1.0,
    )


def _program(*, amine_nodes: int = 8) -> UgiMorphologyProgram:
    return UgiMorphologyProgram(
        node_counts=(amine_nodes, 15, 10),
        junction_budgets=(1, 0, 0),
        cycle_ranks=(0, 0, 0),
        attachment_counts=(1, 2, 1),
    )


def _target(*, amine_diameter: int = 6) -> UgiAllRoleSemanticTarget:
    return UgiAllRoleSemanticTarget(
        amine=UgiAmineSemanticTarget(
            heavy_atom_graph_diameter=amine_diameter,
            carbon_skeleton_diameter=amine_diameter - 1,
            nitrogen_atoms=2,
            oxygen_atoms=0,
        ),
        tail_pair=UgiTailPairSemanticTarget(
            aldehyde_ester_short_side_carbons=4,
            aldehyde_ester_long_side_carbons=10,
            aldehyde_carbon_carbon_double_bonds=0,
            aldehyde_carbon_carbon_triple_bonds=0,
            isocyanide_carbon_carbon_double_bonds=0,
            isocyanide_carbon_carbon_triple_bonds=0,
            isocyanide_carbon_skeleton_diameter=10,
            aldehyde_alkoxy_handle_side_carbons=4,
            aldehyde_acyl_side_carbons=10,
        ),
    )


def _prior() -> UgiMeasuredJointAllRoleSemanticPrior:
    return UgiMeasuredJointAllRoleSemanticPrior(
        support=((_program(), _target()), (_program(amine_nodes=10), _target(amine_diameter=8))),
        probabilities=np.asarray((0.75, 0.25), dtype=np.float64),
        audit={
            "training_fold": "train",
            "reference": "source-adjudicated measured Ugi train products only",
            "component_or_family_ids_in_sampled_target": False,
            "stored_component_graphs_in_sampled_target": False,
            "smiles_or_fragment_tokens_in_sampled_target": False,
            "joint_support_sha256": "frozen-test-support",
        },
    )


def _chemistry_prior() -> UgiRoleChemistryPrior:
    return UgiRoleChemistryPrior(
        reaction_id="ugi_3cr_agile",
        roles=("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail"),
        maximum_depth_bucket=6,
        smoothing=1.0,
        minimum_context_count=1,
        assignments_path=Path("assignments.csv.gz"),
        assignments_sha256="0" * 64,
        semantic_atoms_path=Path("semantic_atoms.csv.gz"),
        semantic_atoms_sha256="1" * 64,
        semantic_bonds_path=Path("semantic_bonds.csv.gz"),
        semantic_bonds_sha256="2" * 64,
        representative_components_by_role=(
            ("amine_head", 1),
            ("isocyanide_tail", 1),
            ("oxoester_aldehyde_body_tail", 1),
        ),
        representative_pairs_sha256="3" * 64,
        atom_counts=(("amine_head", 1, 2, 0, "N", 0, 0, 0, 8),),
        bond_counts=(("isocyanide_tail", 1, 0, "C", "C", 0, 8),),
        bond_terminal_counts=(
            ("isocyanide_tail", 0, 0, "C", "C", 2, 1),
            ("isocyanide_tail", 0, 0, "C", "C", 0, 7),
        ),
        role_unsaturation_count_support=(
            ("oxoester_aldehyde_body_tail", 0, 0),
            ("oxoester_aldehyde_body_tail", 0, 1),
            ("isocyanide_tail", 0, 0),
            ("isocyanide_tail", 1, 0),
        ),
        role_unsaturation_pattern_frequencies=(
            ("oxoester_aldehyde_body_tail", (), (), 1),
            ("oxoester_aldehyde_body_tail", (), (0,), 1),
            ("isocyanide_tail", (), (), 1),
            ("isocyanide_tail", (4,), (), 1),
        ),
    )


def test_semantic_bands_preserve_composition_and_total_tail_carbons() -> None:
    policy = _policy()
    target = UgiAmineSemanticTarget(8, 6, 2, 1)
    amine = policy.amine_targets_within_band(target)
    tails = policy.aldehyde_directional_pairs_within_band(
        alkoxy_handle_carbons=4,
        acyl_carbons=10,
    )

    assert len(amine) == 9
    assert {(value.nitrogen_atoms, value.oxygen_atoms) for value in amine} == {(2, 1)}
    assert {
        (value.heavy_atom_graph_diameter, value.carbon_skeleton_diameter) for value in amine
    } == {(heavy, carbon) for heavy in (7, 8, 9) for carbon in (5, 6, 7)}
    assert tails == ((3, 11), (4, 10), (5, 9))
    assert {sum(value) for value in tails} == {14}


def test_semantic_bands_preserve_substitution_semantics_exactly() -> None:
    target = UgiAmineSemanticTarget(8, 6, 2, 1, 1, 2)

    band = _policy().amine_targets_within_band(target)

    assert {(value.hydrogen_bond_donors, value.heavy_branch_atoms) for value in band} == {(1, 2)}


def test_semantic_bands_allow_bounded_ranked_substitution_slack() -> None:
    target = UgiAmineSemanticTarget(8, 6, 2, 1, 1, 2)
    policy = _policy(
        amine_hydrogen_bond_donors_tolerance=1,
        amine_heavy_branch_atoms_tolerance=1,
    )

    band = policy.amine_targets_within_band(target)

    assert {(value.hydrogen_bond_donors, value.heavy_branch_atoms) for value in band} == {
        (donors, branches) for donors in (0, 1, 2) for branches in (1, 2, 3)
    }
    requested = target
    one_step = UgiAmineSemanticTarget(8, 6, 2, 1, 2, 2)
    two_steps = UgiAmineSemanticTarget(8, 6, 2, 1, 2, 3)
    assert policy.amine_distance(requested, target) == 0.0
    assert policy.amine_distance(one_step, target) == 1.0
    assert policy.amine_distance(two_steps, target) == 2.0
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy


def test_ranked_law_prefers_jointly_better_candidate_and_retains_entropy_floor() -> None:
    policy = _policy()
    probabilities = policy.probabilities(
        model_scores=(4.0, 2.0, 1.0),
        semantic_distances=(0.0, 1.0, 2.0),
    )

    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]
    assert np.all(probabilities >= policy.uniform_probability_mass / 3.0)


def test_joint_realism_is_train_only_identity_free_and_prefers_supported_joint_state() -> None:
    policy = _joint_policy()
    with pytest.raises(UgiMogSemanticGuidanceError, match="without a bound train reference"):
        policy.joint_realism_scores(_program(), (_target(),))

    bound = policy.bind_joint_realism(_prior())
    scores = bound.joint_realism_scores(
        _program(),
        (_target(), _target(amine_diameter=14)),
    )
    assert scores is not None
    assert scores[0] > scores[1]
    audit = bound.joint_realism_audit()
    assert audit is not None
    assert audit["training_fold"] == "train"
    assert audit["component_identity_conditioning"] is False
    assert audit["stored_component_graphs"] is False
    assert audit["calibration_or_heldout_access"] is False


def test_joint_realism_rank_changes_only_the_supported_candidate_law() -> None:
    policy = _joint_policy().bind_joint_realism(_prior())
    probabilities = policy.probabilities(
        model_scores=(1.0, 1.0),
        semantic_distances=(0.0, 0.0),
        joint_realism_scores=(2.0, -2.0),
    )

    assert probabilities[0] > probabilities[1]
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= policy.uniform_probability_mass / 2.0)


def test_local_chemistry_is_a_separate_ranked_objective_with_audited_binding() -> None:
    policy = _policy(local_chemistry_rank_weight=3.0)
    with pytest.raises(UgiMogSemanticGuidanceError, match="bound measured-train prior"):
        policy.chemistry_probabilities(
            model_scores=(2.0, 1.0),
            semantic_distances=(0.0, 0.0),
            local_chemistry_scores=(-2.0, 0.0),
        )

    bound = policy.bind_local_chemistry(_chemistry_prior())
    probabilities = bound.chemistry_probabilities(
        model_scores=(2.0, 1.0),
        semantic_distances=(0.0, 0.0),
        local_chemistry_scores=(-2.0, 0.0),
    )

    assert probabilities[1] > probabilities[0]
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= policy.uniform_probability_mass / 2.0)
    audit = bound.local_chemistry_audit()
    assert audit is not None
    assert audit["component_identity_conditioning"] is False
    assert audit["component_graph_conditioning"] is False
    assert audit["fragment_vocabulary_conditioning"] is False


def test_terminal_offset_bond_scoring_uses_the_distal_train_context() -> None:
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_bond_maximum_depth_bucket=64,
        local_chemistry_unsaturation_position_only=True,
        local_chemistry_bond_position_basis="terminal_offset",
        local_chemistry_score_mode="support_tier",
    ).bind_local_chemistry(replace(_chemistry_prior(), maximum_bond_depth_bucket=64))

    scores = policy.bond_local_scores(
        role="isocyanide_tail",
        depth=12,
        terminal_offset=0,
        in_ring=False,
        symbols=("C", "C"),
        bond_classes=4,
    )

    assert scores[0] == 4.0
    assert scores[2] == 4.0
    assert policy.to_mapping()["local_chemistry_bond_position_basis"] == "terminal_offset"


def test_terminal_offset_basis_requires_position_specific_tail_scoring() -> None:
    with pytest.raises(UgiMogSemanticGuidanceError):
        _policy(
            local_chemistry_rank_weight=1.0,
            local_chemistry_bond_position_basis="terminal_offset",
        )


def test_local_chemistry_policy_can_rank_topology_without_applying_chemistry_term() -> None:
    policy = _policy(local_chemistry_rank_weight=1.0).bind_local_chemistry(_chemistry_prior())

    probabilities = policy.probabilities(
        model_scores=(2.0, 1.0),
        semantic_distances=(0.0, 1.0),
    )

    assert probabilities[0] > probabilities[1]


def test_local_atom_ranking_can_leave_bond_choices_to_the_transformer() -> None:
    policy = _policy(
        local_chemistry_rank_weight=3.0,
        local_chemistry_bond_rank_weight=0.0,
    ).bind_local_chemistry(_chemistry_prior())

    atom_probabilities = policy.atom_chemistry_probabilities(
        model_scores=(2.0, 1.0),
        semantic_distances=(0.0, 0.0),
        local_chemistry_scores=(-2.0, 0.0),
    )
    bond_probabilities = policy.probabilities(
        model_scores=(2.0, 1.0),
        semantic_distances=(0.0, 0.0),
    )

    assert policy.uses_local_chemistry is True
    assert policy.uses_local_chemistry_atoms is True
    assert policy.uses_local_chemistry_bonds is False
    assert atom_probabilities[1] > atom_probabilities[0]
    assert bond_probabilities[0] > bond_probabilities[1]
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy


def test_bond_ranking_can_use_a_separate_entropy_mixture() -> None:
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_bond_rank_weight=2.0,
        local_chemistry_bond_uniform_probability_mass=0.05,
    ).bind_local_chemistry(_chemistry_prior())

    shared = policy.chemistry_probabilities(
        model_scores=(1.0, 1.0),
        semantic_distances=(0.0, 0.0),
        local_chemistry_scores=(0.0, 4.0),
    )
    bonds = policy.bond_chemistry_probabilities(
        model_scores=(1.0, 1.0),
        semantic_distances=(0.0, 0.0),
        local_chemistry_scores=(0.0, 4.0),
    )

    assert bonds[1] > shared[1]
    assert np.isclose(bonds.sum(), 1.0)
    assert np.all(bonds >= 0.05 / 2.0)
    assert policy.local_chemistry_audit()[
        "local_chemistry_bond_uniform_probability_mass"
    ] == 0.05
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy

    with pytest.raises(UgiMogSemanticGuidanceError):
        _policy(local_chemistry_bond_uniform_probability_mass=0.05)


def test_atom_local_chemistry_trust_region_bounds_probability_shift() -> None:
    policy = _policy(
        local_chemistry_rank_weight=3.0,
        local_chemistry_bond_rank_weight=0.0,
        local_chemistry_atom_total_variation_radius=0.05,
    ).bind_local_chemistry(_chemistry_prior())
    baseline = policy.probabilities(
        model_scores=(2.0, 1.0),
        semantic_distances=(0.0, 0.0),
    )
    unrestricted = policy.chemistry_probabilities(
        model_scores=(2.0, 1.0),
        semantic_distances=(0.0, 0.0),
        local_chemistry_scores=(-2.0, 0.0),
    )
    trusted = policy.atom_chemistry_probabilities(
        model_scores=(2.0, 1.0),
        semantic_distances=(0.0, 0.0),
        local_chemistry_scores=(-2.0, 0.0),
    )

    trusted_tv = 0.5 * float(np.abs(trusted - baseline).sum())
    unrestricted_tv = 0.5 * float(np.abs(unrestricted - baseline).sum())
    assert np.isclose(trusted_tv, 0.05)
    assert trusted_tv < unrestricted_tv
    assert np.isclose(trusted.sum(), 1.0)
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy


def test_support_tier_scores_do_not_reward_frequency_within_one_context() -> None:
    prior = replace(
        _chemistry_prior(),
        atom_counts=(
            ("amine_head", 1, 2, 0, "C", 0, 0, 0, 100),
            ("amine_head", 1, 2, 0, "N", 0, 0, 0, 1),
        ),
        bond_counts=(
            ("amine_head", 1, 0, "C", "N", 0, 100),
            ("amine_head", 1, 0, "C", "N", 1, 1),
        ),
    )
    vocabulary = (
        # Both atom states have exact-context evidence despite unequal counts.
        AtomState("C", 0, False, 0),
        AtomState("N", 0, False, 0),
    )

    atom_tiers = prior.atom_support_tiers(
        role="amine_head",
        depth=1,
        degree=2,
        in_ring=False,
        atom_vocabulary=vocabulary,
    )
    bond_tiers = prior.bond_support_tiers(
        role="amine_head",
        depth=1,
        in_ring=False,
        symbols=("C", "N"),
        bond_classes=4,
    )

    assert atom_tiers.tolist() == [4.0, 4.0]
    assert bond_tiers[:2].tolist() == [4.0, 4.0]
    assert (
        prior.edge_symbol_support_tier(
            role="amine_head", depth=1, in_ring=False, symbols=("C", "N")
        )
        == 4.0
    )
    assert (
        prior.edge_symbol_support_tier(
            role="amine_head", depth=1, in_ring=False, symbols=("N", "N")
        )
        == 0.0
    )


def test_support_tier_policy_uses_bottleneck_not_summed_frequency() -> None:
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_score_mode="support_tier",
    ).bind_local_chemistry(_chemistry_prior())

    assert policy.aggregate_local_scores((4.0, 4.0, 2.0)) == 2.0
    assert policy.local_chemistry_audit()["score_mode"] == "support_tier"
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy


def test_whole_head_support_mode_disables_repeated_coordinate_rewards() -> None:
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_bond_rank_weight=0.0,
        local_chemistry_score_mode="whole_head_support_tier",
    ).bind_local_chemistry(_chemistry_prior())

    assert policy.uses_local_chemistry_atoms
    assert not policy.uses_coordinate_local_chemistry_atoms
    assert not policy.uses_local_chemistry_bonds
    assert policy.aggregate_local_scores((4.0,)) == 4.0
    assert (
        policy.local_chemistry_audit()["component_score_aggregation"]
        == "one novelty-neutral support tier for the completed amine-head arrangement"
    )
    with pytest.raises(
        UgiMogSemanticGuidanceError,
        match="whole-head support cannot score an isolated atom",
    ):
        policy.atom_local_scores(
            role="amine_head",
            depth=1,
            degree=2,
            in_ring=False,
            atom_vocabulary=(AtomState("C", 0, False, 0),),
        )
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy


def test_binary_whole_head_support_does_not_rank_supported_specificity_tiers() -> None:
    prior = _chemistry_prior()
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_bond_rank_weight=0.0,
        local_chemistry_score_mode="whole_head_support_binary",
    ).bind_local_chemistry(prior)

    assert not policy.uses_coordinate_local_chemistry_atoms
    assert policy.aggregate_local_scores((1.0,)) == 1.0
    assert (
        policy.local_chemistry_audit()["component_score_aggregation"]
        == "one binary measured-support indicator for the completed amine-head arrangement"
    )
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy


def test_whole_head_topology_support_is_identity_free_and_round_trips() -> None:
    exact_signature = (2, 0, 0, (0, 2, 0, 0, 0), (1, 2), (1,), (), (1, 2))
    prior = replace(
        _chemistry_prior(),
        amine_head_topology_exact_signatures=frozenset({exact_signature}),
    )
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_bond_rank_weight=0.0,
        local_chemistry_score_mode="whole_head_support_binary",
        whole_head_topology_support=True,
    ).bind_local_chemistry(prior)

    supported = policy.amine_head_topology_support_score(
        nodes=(0, 1),
        depths_by_node={0: 1, 1: 2},
        neighbors={0: {1, 2}, 1: {0}, 2: {0}},
        ring_nodes=set(),
    )
    unsupported = policy.amine_head_topology_support_score(
        nodes=(0, 1),
        depths_by_node={0: 1, 1: 3},
        neighbors={0: {1, 2}, 1: {0}, 2: {0}},
        ring_nodes=set(),
    )

    assert supported == 1.0
    assert unsupported == 0.0
    assert policy.local_chemistry_audit()["whole_head_topology_support"] is True
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy

    context_policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_score_mode="support_tier",
        whole_head_topology_support=True,
    ).bind_local_chemistry(prior)
    assert context_policy.uses_coordinate_local_chemistry_atoms
    assert context_policy.amine_head_topology_support_score(
        nodes=(0, 1),
        depths_by_node={0: 1, 1: 2},
        neighbors={0: {1, 2}, 1: {0}, 2: {0}},
        ring_nodes=set(),
    ) == 1.0
    assert UgiMogSemanticGuidancePolicy.from_mapping(context_policy.to_mapping()) == context_policy


def test_whole_head_topology_tier_preserves_exact_vs_coarse_evidence() -> None:
    exact_signature = (2, 0, 0, (0, 2, 0, 0, 0), (1, 2), (1,), (), (1, 2))
    coarse_signature = (2, 0, 0, (0, 2, 0, 0, 0), 2, (1,), 0, 1)
    prior = replace(
        _chemistry_prior(),
        amine_head_topology_exact_signatures=frozenset({exact_signature}),
        amine_head_topology_coarse_signatures=frozenset({coarse_signature}),
    )
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_score_mode="whole_head_support_tier",
        whole_head_topology_support=True,
        whole_head_topology_score_mode="tier",
    ).bind_local_chemistry(prior)

    assert policy.amine_head_topology_support_score(
        nodes=(0, 1),
        depths_by_node={0: 1, 1: 2},
        neighbors={0: {1, 2}, 1: {0}, 2: {0}},
        ring_nodes=set(),
    ) == 2.0
    assert policy.local_chemistry_audit()["whole_head_topology_score_mode"] == "tier"
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy


def test_topology_support_can_restore_deterministic_terminal_chemistry() -> None:
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_score_mode="support_tier",
        whole_head_topology_support=True,
        terminal_chemistry_selection_mode="model_argmax",
    ).bind_local_chemistry(_chemistry_prior())

    assert policy.uses_local_reference
    assert not policy.uses_ranked_terminal_chemistry
    assert not policy.uses_local_chemistry_atoms
    assert not policy.uses_local_chemistry_bonds
    assert policy.local_chemistry_audit()["terminal_chemistry_selection_mode"] == "model_argmax"
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy

    with pytest.raises(UgiMogSemanticGuidanceError):
        _policy(terminal_chemistry_selection_mode="model_argmax")


def test_topology_support_rank_is_independent_of_terminal_local_chemistry() -> None:
    policy = _policy(
        local_chemistry_rank_weight=0.0,
        local_chemistry_score_mode="support_tier",
        whole_head_topology_support=True,
        whole_head_topology_rank_weight=1.0,
    ).bind_local_chemistry(_chemistry_prior())

    assert policy.uses_ranked_terminal_chemistry
    assert policy.uses_local_reference
    assert not policy.uses_local_chemistry
    assert policy.effective_whole_head_topology_rank_weight == 1.0
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy


def test_terminal_bond_argmax_preserves_stochastic_atom_readout() -> None:
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        terminal_bond_selection_mode="model_argmax",
    )

    assert policy.uses_ranked_terminal_chemistry
    assert not policy.uses_ranked_terminal_bonds
    assert policy.uses_local_chemistry_atoms
    assert not policy.uses_local_chemistry_bonds
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy


def test_whole_head_trust_region_bounds_complete_assignment_law() -> None:
    policy = _policy(
        local_chemistry_rank_weight=3.0,
        local_chemistry_bond_rank_weight=0.0,
        local_chemistry_score_mode="whole_head_support_binary",
        whole_head_total_variation_radius=0.05,
    ).bind_local_chemistry(_chemistry_prior())
    baseline = policy.probabilities(
        model_scores=(2.0, 1.0, 0.0),
        semantic_distances=(0.0, 0.0, 0.0),
    )
    unrestricted = policy.chemistry_probabilities(
        model_scores=(2.0, 1.0, 0.0),
        semantic_distances=(0.0, 0.0, 0.0),
        local_chemistry_scores=(0.0, 0.0, 1.0),
    )
    trusted = policy.whole_head_chemistry_probabilities(
        model_scores=(2.0, 1.0, 0.0),
        semantic_distances=(0.0, 0.0, 0.0),
        local_chemistry_scores=(0.0, 0.0, 1.0),
    )

    trusted_tv = 0.5 * float(np.abs(trusted - baseline).sum())
    unrestricted_tv = 0.5 * float(np.abs(unrestricted - baseline).sum())
    assert np.isclose(trusted_tv, 0.05)
    assert trusted_tv < unrestricted_tv
    assert np.isclose(trusted.sum(), 1.0)
    assert policy.local_chemistry_audit()["whole_head_total_variation_radius"] == 0.05
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy


def test_whole_head_and_coordinate_local_trust_regions_are_not_interchangeable() -> None:
    with pytest.raises(UgiMogSemanticGuidanceError):
        _policy(
            local_chemistry_rank_weight=1.0,
            local_chemistry_score_mode="whole_head_support_binary",
            local_chemistry_atom_total_variation_radius=0.05,
        )
    with pytest.raises(UgiMogSemanticGuidanceError):
        _policy(
            local_chemistry_rank_weight=1.0,
            local_chemistry_score_mode="support_tier",
            whole_head_total_variation_radius=0.05,
        )


def test_whole_head_effective_count_guard_preserves_candidate_entropy() -> None:
    policy = _policy(
        local_chemistry_rank_weight=3.0,
        local_chemistry_score_mode="whole_head_support_binary",
        whole_head_total_variation_radius=0.05,
        whole_head_candidate_effective_count_retention=0.99,
    ).bind_local_chemistry(_chemistry_prior())
    baseline = policy.probabilities(
        model_scores=(4.0, 3.0, 2.0, 1.0, 0.0),
        semantic_distances=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    trusted = policy.whole_head_chemistry_probabilities(
        model_scores=(4.0, 3.0, 2.0, 1.0, 0.0),
        semantic_distances=(0.0, 0.0, 0.0, 0.0, 0.0),
        local_chemistry_scores=(0.0, 0.0, 0.0, 0.0, 1.0),
    )

    def effective_count(probabilities: np.ndarray) -> float:
        return float(np.exp(-np.sum(probabilities * np.log(probabilities))))

    assert effective_count(trusted) >= 0.99 * effective_count(baseline) - 1e-12
    assert 0.5 * float(np.abs(trusted - baseline).sum()) <= 0.05 + 1e-12
    assert np.isclose(trusted.sum(), 1.0)
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy

    with pytest.raises(UgiMogSemanticGuidanceError):
        _policy(
            local_chemistry_rank_weight=1.0,
            local_chemistry_score_mode="support_tier",
            whole_head_candidate_effective_count_retention=0.99,
        )


def test_whole_head_group_guard_preserves_arrangement_class_entropy() -> None:
    policy = _policy(
        local_chemistry_rank_weight=3.0,
        local_chemistry_score_mode="whole_head_support_binary",
        whole_head_total_variation_radius=0.05,
        whole_head_group_effective_count_retention=0.99,
    ).bind_local_chemistry(_chemistry_prior())
    groups = ("a", "a", "a", "b", "c")
    baseline = policy.probabilities(
        model_scores=(4.0, 3.0, 2.0, 1.0, 0.0),
        semantic_distances=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    trusted = policy.whole_head_chemistry_probabilities(
        model_scores=(4.0, 3.0, 2.0, 1.0, 0.0),
        semantic_distances=(0.0, 0.0, 0.0, 0.0, 0.0),
        local_chemistry_scores=(1.0, 1.0, 1.0, 0.0, 0.0),
        group_labels=groups,
    )

    def grouped_effective_count(probabilities: np.ndarray) -> float:
        masses = np.asarray(
            [
                sum(probabilities[index] for index, group in enumerate(groups) if group == label)
                for label in ("a", "b", "c")
            ]
        )
        return float(np.exp(-np.sum(masses * np.log(masses))))

    assert grouped_effective_count(trusted) >= 0.99 * grouped_effective_count(baseline) - 1e-12
    assert 0.5 * float(np.abs(trusted - baseline).sum()) <= 0.05 + 1e-12
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy

    with pytest.raises(UgiMogSemanticGuidanceError):
        policy.whole_head_chemistry_probabilities(
            model_scores=(2.0, 1.0),
            semantic_distances=(0.0, 0.0),
            local_chemistry_scores=(1.0, 0.0),
        )


def test_whole_head_group_guard_does_not_require_total_variation_bound() -> None:
    policy = _policy(
        local_chemistry_rank_weight=3.0,
        local_chemistry_score_mode="whole_head_support_tier",
        whole_head_group_effective_count_retention=0.99,
    ).bind_local_chemistry(_chemistry_prior())
    groups = ("a", "a", "a", "b", "c")
    baseline = policy.probabilities(
        model_scores=(4.0, 3.0, 2.0, 1.0, 0.0),
        semantic_distances=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    guarded = policy.whole_head_chemistry_probabilities(
        model_scores=(4.0, 3.0, 2.0, 1.0, 0.0),
        semantic_distances=(0.0, 0.0, 0.0, 0.0, 0.0),
        local_chemistry_scores=(4.0, 4.0, 4.0, 1.0, 0.0),
        group_labels=groups,
    )

    def grouped_effective_count(probabilities: np.ndarray) -> float:
        grouped = np.asarray(
            [probabilities[:3].sum(), probabilities[3], probabilities[4]],
            dtype=np.float64,
        )
        positive = grouped[grouped > 0]
        return float(np.exp(-np.sum(positive * np.log(positive))))

    assert grouped_effective_count(guarded) >= 0.99 * grouped_effective_count(baseline) - 1e-12
    assert not np.allclose(guarded, baseline)


def test_whole_head_hard_bond_support_can_be_disabled_explicitly() -> None:
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_bond_rank_weight=0.0,
        local_chemistry_score_mode="whole_head_support_binary",
        whole_head_total_variation_radius=0.05,
        whole_head_hard_bond_support=False,
    ).bind_local_chemistry(_chemistry_prior())

    assert policy.whole_head_hard_bond_support is False
    assert policy.local_chemistry_audit()["whole_head_hard_bond_support"] is False
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy

    with pytest.raises(UgiMogSemanticGuidanceError):
        _policy(
            local_chemistry_rank_weight=1.0,
            local_chemistry_score_mode="support_tier",
            whole_head_hard_bond_support=False,
        )


def test_whole_head_distance_mode_scores_nearby_novel_arrangements() -> None:
    prior = replace(
        _chemistry_prior(),
        amine_head_basic_signatures=frozenset({(2, 1, 0, 0, 0, 1, 1, (("N", "C"),))}),
    )
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_score_mode="whole_head_support_distance",
        whole_head_total_variation_radius=0.05,
    ).bind_local_chemistry(prior)

    score = policy.amine_head_arrangement_support_score(
        nodes=(0, 1),
        symbols_by_node={0: "C", 1: "N", 2: "C"},
        depths_by_node={0: 1, 1: 2},
        neighbors={0: {1}, 1: {0, 2}},
        ring_nodes=set(),
    )

    assert 0.0 < score <= 1.0
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy


def test_local_chemistry_rank_cannot_resurrect_a_hard_masked_state() -> None:
    policy = _policy(local_chemistry_rank_weight=3.0).bind_local_chemistry(_chemistry_prior())

    selected = _sample_mog_chemistry_allowed(
        np.asarray((1.0, 0.0), dtype=np.float64),
        np.asarray((True, False), dtype=np.bool_),
        np.asarray((-10.0, 10.0), dtype=np.float64),
        policy=policy,
        generator=np.random.default_rng(9),
    )

    assert selected == 0


def test_unsaturation_position_scoring_requires_fine_bond_depth() -> None:
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_bond_maximum_depth_bucket=64,
        local_chemistry_unsaturation_position_only=True,
        local_chemistry_score_mode="whole_head_support_binary",
    )

    assert policy.local_chemistry_unsaturation_position_only is True
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy
    with pytest.raises(UgiMogSemanticGuidanceError):
        _policy(
            local_chemistry_rank_weight=1.0,
            local_chemistry_unsaturation_position_only=True,
            local_chemistry_score_mode="whole_head_support_binary",
        )


def test_unsaturation_position_support_floor_requires_position_scoring() -> None:
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_bond_maximum_depth_bucket=64,
        local_chemistry_unsaturation_position_only=True,
        local_chemistry_unsaturation_minimum_support_tier=4,
        local_chemistry_score_mode="whole_head_support_binary",
    )

    assert policy.local_chemistry_unsaturation_minimum_support_tier == 4
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy
    with pytest.raises(UgiMogSemanticGuidanceError):
        _policy(
            local_chemistry_rank_weight=1.0,
            local_chemistry_bond_maximum_depth_bucket=64,
            local_chemistry_unsaturation_minimum_support_tier=4,
            local_chemistry_score_mode="whole_head_support_binary",
        )


def test_unsaturation_count_slack_uses_only_nearby_measured_role_support() -> None:
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_bond_maximum_depth_bucket=64,
        local_chemistry_unsaturation_position_only=True,
        tail_unsaturation_count_tolerance=1,
    ).bind_local_chemistry(replace(_chemistry_prior(), maximum_bond_depth_bucket=64))

    options = policy.tail_unsaturation_count_options(
        role="oxoester_aldehyde_body_tail",
        requested_double_count=0,
        requested_triple_count=1,
    )
    replaced = replace_tail_unsaturation_semantics(
        _target(),
        role="oxoester_aldehyde_body_tail",
        double_count=0,
        triple_count=1,
    )

    assert options == ((0, 0), (0, 1))
    assert replaced.tail_pair.aldehyde_carbon_carbon_triple_bonds == 1
    assert replaced.tail_pair.isocyanide_carbon_carbon_triple_bonds == 0
    assert policy.local_chemistry_audit()["tail_unsaturation_count_tolerance"] == 1
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy

    with pytest.raises(UgiMogSemanticGuidanceError):
        _policy(tail_unsaturation_count_tolerance=1)


def test_grouped_downward_unsaturation_slack_never_adds_or_swaps_bond_types() -> None:
    policy = _policy(
        local_chemistry_rank_weight=1.0,
        local_chemistry_bond_maximum_depth_bucket=64,
        local_chemistry_unsaturation_position_only=True,
        tail_unsaturation_count_tolerance=1,
        tail_unsaturation_count_strategy="grouped_downward",
    ).bind_local_chemistry(replace(_chemistry_prior(), maximum_bond_depth_bucket=64))

    saturated = policy.tail_unsaturation_count_options(
        role="oxoester_aldehyde_body_tail",
        requested_double_count=0,
        requested_triple_count=0,
    )
    triple = policy.tail_unsaturation_count_options(
        role="oxoester_aldehyde_body_tail",
        requested_double_count=0,
        requested_triple_count=1,
    )

    assert saturated == ((0, 0),)
    assert triple == ((0, 0), (0, 1))
    assert UgiMogSemanticGuidancePolicy.from_mapping(policy.to_mapping()) == policy

    flat_downward = replace(policy, tail_unsaturation_count_strategy="flat_downward")
    assert flat_downward.tail_unsaturation_count_options(
        role="oxoester_aldehyde_body_tail",
        requested_double_count=0,
        requested_triple_count=0,
    ) == ((0, 0),)
    assert UgiMogSemanticGuidancePolicy.from_mapping(flat_downward.to_mapping()) == flat_downward

    frequency_downward = replace(policy, tail_unsaturation_count_strategy="frequency_downward")
    assert frequency_downward.tail_unsaturation_count_options(
        role="oxoester_aldehyde_body_tail",
        requested_double_count=0,
        requested_triple_count=1,
    ) == ((0, 0), (0, 1))
    assert UgiMogSemanticGuidancePolicy.from_mapping(frequency_downward.to_mapping()) == (
        frequency_downward
    )
    assert frequency_downward.local_chemistry_prior is not None
    assert frequency_downward.local_chemistry_prior.unsaturation_count_frequencies(
        "oxoester_aldehyde_body_tail"
    ) == ((0, 0, 1), (0, 1, 1))

    frequency_resampled = replace(
        policy,
        tail_unsaturation_count_strategy="frequency_resampled",
        tail_unsaturation_count_tolerance=0,
    )
    assert frequency_resampled.tail_unsaturation_count_options(
        role="oxoester_aldehyde_body_tail",
        requested_double_count=0,
        requested_triple_count=0,
    ) == ((0, 0), (0, 1))
    assert UgiMogSemanticGuidancePolicy.from_mapping(frequency_resampled.to_mapping()) == (
        frequency_resampled
    )

    joint_pattern = replace(
        frequency_resampled,
        local_chemistry_bond_position_basis="terminal_offset",
        tail_unsaturation_position_strategy="measured_joint_terminal_pattern",
    )
    assert joint_pattern.local_chemistry_prior is not None
    assert joint_pattern.local_chemistry_prior.unsaturation_pattern_frequencies(
        "oxoester_aldehyde_body_tail"
    ) == (((), (), 1), ((), (0,), 1))
    assert UgiMogSemanticGuidancePolicy.from_mapping(joint_pattern.to_mapping()) == joint_pattern

    ranked_joint_pattern = replace(
        joint_pattern,
        tail_unsaturation_position_strategy="measured_joint_terminal_pattern_ranked",
    )
    assert UgiMogSemanticGuidancePolicy.from_mapping(
        ranked_joint_pattern.to_mapping()
    ) == ranked_joint_pattern

    with pytest.raises(UgiMogSemanticGuidanceError):
        replace(
            frequency_resampled,
            tail_unsaturation_position_strategy="measured_joint_terminal_pattern",
        )

    smoothed_frequency_resampled = replace(
        policy,
        tail_unsaturation_count_strategy="smoothed_frequency_resampled",
        tail_unsaturation_count_tolerance=0,
    )
    assert smoothed_frequency_resampled.tail_unsaturation_count_options(
        role="oxoester_aldehyde_body_tail",
        requested_double_count=0,
        requested_triple_count=0,
    ) == ((0, 0), (0, 1))
    assert UgiMogSemanticGuidancePolicy.from_mapping(
        smoothed_frequency_resampled.to_mapping()
    ) == smoothed_frequency_resampled

    lightly_smoothed = replace(
        smoothed_frequency_resampled,
        tail_unsaturation_frequency_pseudocount=0.1,
    )
    assert UgiMogSemanticGuidancePolicy.from_mapping(lightly_smoothed.to_mapping()) == (
        lightly_smoothed
    )
    with pytest.raises(UgiMogSemanticGuidanceError):
        replace(frequency_resampled, tail_unsaturation_frequency_pseudocount=0.1)


def test_legacy_policy_mapping_remains_exactly_backward_compatible() -> None:
    mapping = _policy().to_mapping()

    assert "joint_realism_rank_weight" not in mapping
    assert UgiMogSemanticGuidancePolicy.from_mapping(mapping).to_mapping() == mapping


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("amine_heavy_atom_graph_diameter_tolerance", 3),
        ("semantic_rank_weight", -1.0),
        ("rank_temperature", 0.0),
        ("uniform_probability_mass", 0.0),
        ("uniform_probability_mass", 1.0),
    ),
)
def test_policy_rejects_unbounded_or_deterministic_requests(field: str, value: object) -> None:
    with pytest.raises(UgiMogSemanticGuidanceError):
        _policy(**{field: value})
