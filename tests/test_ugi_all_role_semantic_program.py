from __future__ import annotations

import numpy as np
import pytest

from forge.model.ugi_all_role_semantic_program import (
    UgiAllRoleSemanticProgramError,
    UgiAllRoleSemanticTarget,
    UgiMeasuredAllRoleSemanticPrior,
    UgiMeasuredJointAllRoleSemanticPrior,
    UgiMeasuredRoleFactorizedSemanticPrior,
    UgiTailPairSemanticTarget,
    all_role_semantic_target,
    build_equal_group_conditional_distribution,
    build_equal_group_joint_distribution,
    directional_all_role_semantic_target,
)
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget
from forge.model.ugi_morphology_program import UgiMorphologyProgram


def _program() -> UgiMorphologyProgram:
    return UgiMorphologyProgram(
        node_counts=(6, 17, 12),
        junction_budgets=(1, 1, 0),
        cycle_ranks=(0, 0, 0),
        attachment_counts=(1, 1, 1),
    )


def _target(*, aldehyde_double_bonds: int = 0) -> UgiAllRoleSemanticTarget:
    return UgiAllRoleSemanticTarget(
        amine=UgiAmineSemanticTarget(5, 4, 2, 0),
        tail_pair=UgiTailPairSemanticTarget(
            aldehyde_ester_short_side_carbons=6,
            aldehyde_ester_long_side_carbons=10,
            aldehyde_carbon_carbon_double_bonds=aldehyde_double_bonds,
            aldehyde_carbon_carbon_triple_bonds=0,
            isocyanide_carbon_carbon_double_bonds=0,
            isocyanide_carbon_carbon_triple_bonds=0,
            isocyanide_carbon_skeleton_diameter=12,
        ),
    )


def test_all_role_target_is_graph_derived_and_identity_free() -> None:
    target = all_role_semantic_target(
        amine_smiles="CN1CCN(CCN)CC1",
        aldehyde_smiles="O=CCCCCC(=O)OCCCCCCCCCC",
        isocyanide_smiles="CCCCCCCCCCCC[N+]#[C-]",
    )

    assert target.amine.nitrogen_atoms == 3
    assert target.tail_pair.aldehyde_ester_short_side_carbons >= 1
    assert target.tail_pair.aldehyde_ester_long_side_carbons >= 1
    assert target.tail_pair.isocyanide_carbon_carbon_double_bonds == 0
    assert set(target.to_mapping()) == {"amine", "tail_pair"}
    assert not any(
        token in repr(target.to_mapping()).lower()
        for token in ("smiles", "component_id", "family_id")
    )


def test_all_role_target_round_trips_strict_json_mapping() -> None:
    target = _target(aldehyde_double_bonds=1)

    assert UgiAllRoleSemanticTarget.from_mapping(target.to_mapping()) == target
    changed = target.to_mapping()
    changed["tail_pair"]["aldehyde_carbon_carbon_double_bonds"] = True
    with pytest.raises(UgiAllRoleSemanticProgramError, match="must contain integers"):
        UgiAllRoleSemanticTarget.from_mapping(changed)


def test_directional_target_preserves_alkoxy_aldehyde_and_acyl_sides() -> None:
    target = directional_all_role_semantic_target(
        amine_smiles="CN1CCN(CCN)CC1",
        aldehyde_smiles="CCCCCCCCCCCCCCCCCC(=O)OCCCCCC=O",
        isocyanide_smiles="CCCCCCCCCCCC[N+]#[C-]",
    )

    assert target.tail_pair.aldehyde_alkoxy_handle_side_carbons == 6
    assert target.tail_pair.aldehyde_acyl_side_carbons == 18
    assert UgiAllRoleSemanticTarget.from_mapping(target.to_mapping()) == target


def test_legacy_all_role_target_mapping_keeps_direction_unspecified() -> None:
    target = _target()

    mapping = target.to_mapping()
    assert "aldehyde_alkoxy_handle_side_carbons" not in mapping["tail_pair"]
    assert UgiAllRoleSemanticTarget.from_mapping(mapping) == target


def test_joint_conditional_distribution_balances_family_triples_not_rows() -> None:
    program = (
        _program().node_counts,
        _program().junction_budgets,
        _program().cycle_ranks,
        _program().attachment_counts,
    )
    saturated = _target(aldehyde_double_bonds=0)
    unsaturated = _target(aldehyde_double_bonds=1)
    support = build_equal_group_conditional_distribution(
        [
            (program, "repeated|pair|tail", saturated),
            (program, "repeated|pair|tail", saturated),
            (program, "single|pair|tail", unsaturated),
        ]
    )
    targets, probabilities = support[program]
    mass = {target: probability for target, probability in zip(targets, probabilities, strict=True)}

    assert mass[saturated] == pytest.approx(0.5)
    assert mass[unsaturated] == pytest.approx(0.5)


def test_joint_program_semantic_distribution_balances_groups_without_refactorization() -> None:
    first = _program()
    second = UgiMorphologyProgram(
        node_counts=(7, 17, 12),
        junction_budgets=(1, 1, 0),
        cycle_ranks=(0, 0, 0),
        attachment_counts=(1, 1, 1),
    )
    saturated = _target(aldehyde_double_bonds=0)
    unsaturated = _target(aldehyde_double_bonds=1)
    support, probabilities = build_equal_group_joint_distribution(
        [
            (
                (
                    first.node_counts,
                    first.junction_budgets,
                    first.cycle_ranks,
                    first.attachment_counts,
                ),
                "large",
                saturated,
            ),
            (
                (
                    first.node_counts,
                    first.junction_budgets,
                    first.cycle_ranks,
                    first.attachment_counts,
                ),
                "large",
                saturated,
            ),
            (
                (
                    second.node_counts,
                    second.junction_budgets,
                    second.cycle_ranks,
                    second.attachment_counts,
                ),
                "small",
                unsaturated,
            ),
        ]
    )
    mass = {
        (program, target): probability
        for (program, target), probability in zip(support, probabilities, strict=True)
    }

    assert mass[(first, saturated)] == pytest.approx(0.5)
    assert mass[(second, unsaturated)] == pytest.approx(0.5)


def test_joint_prior_draws_one_target_per_complete_program_without_retry() -> None:
    program = _program()
    key = (
        program.node_counts,
        program.junction_budgets,
        program.cycle_ranks,
        program.attachment_counts,
    )
    target = _target()
    prior = UgiMeasuredAllRoleSemanticPrior(
        support={key: ((target,), np.asarray([1.0], dtype=np.float64))},
        audit={},
    )

    assert prior.sample_for_programs((program, program), seed=17) == (target, target)


def test_joint_prior_fails_closed_without_complete_program_support() -> None:
    prior = UgiMeasuredAllRoleSemanticPrior(support={}, audit={})

    with pytest.raises(UgiAllRoleSemanticProgramError, match="lacks measured semantic support"):
        prior.sample_for_programs((_program(),), seed=17)


def test_joint_program_semantic_prior_draws_pairs_without_retry() -> None:
    program = _program()
    target = _target()
    prior = UgiMeasuredJointAllRoleSemanticPrior(
        support=((program, target),),
        probabilities=np.asarray([1.0], dtype=np.float64),
        audit={},
    )

    programs, targets = prior.sample(count=2, seed=17)
    assert programs == (program, program)
    assert targets == (target, target)


def test_role_factorized_prior_samples_identity_free_role_semantics() -> None:
    program = _program()
    one = np.asarray([1.0], dtype=np.float64)
    prior = UgiMeasuredRoleFactorizedSemanticPrior(
        amine_support={(6, 1, 0, 1): ((_target().amine,), one)},
        aldehyde_support={(17, 1, 0, 1): (((6, 10, 0, 0, 6, 10),), one)},
        isocyanide_support={(12, 0, 0, 1): (((0, 0, 12),), one)},
        audit={},
    )

    sampled = prior.sample_for_programs((program,), seed=17)[0]
    assert sampled.amine == _target().amine
    assert sampled.tail_pair.aldehyde_alkoxy_handle_side_carbons == 6
    assert sampled.tail_pair.aldehyde_acyl_side_carbons == 10
    assert sampled.tail_pair.isocyanide_carbon_skeleton_diameter == 12
    assert not any(
        token in repr(sampled.to_mapping()).lower()
        for token in ("smiles", "component_id", "family_id")
    )
