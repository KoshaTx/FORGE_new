from __future__ import annotations

import numpy as np
import pytest

from forge.model.ugi_amine_semantic_program import (
    UgiAmineSemanticProgramError,
    UgiAmineSemanticTarget,
    UgiMeasuredAmineSemanticPrior,
    amine_local_substitution_metrics,
    amine_program_key,
    amine_semantic_target,
    build_equal_family_conditional_distribution,
)
from forge.model.ugi_morphology_program import UgiMorphologyProgram


def _program(nodes: int) -> UgiMorphologyProgram:
    return UgiMorphologyProgram(
        node_counts=(nodes, 18, 12),
        junction_budgets=(1, 1, 0),
        cycle_ranks=(0, 0, 0),
        attachment_counts=(1, 1, 1),
    )


def test_amine_semantic_target_is_graph_derived_and_identity_free() -> None:
    target = amine_semantic_target("CN1CCN(CCN)CC1")

    assert target.nitrogen_atoms == 3
    assert target.oxygen_atoms == 0
    assert target.heavy_atom_graph_diameter >= 4
    assert target.carbon_skeleton_diameter >= 2
    assert set(target.to_mapping()) == {
        "heavy_atom_graph_diameter",
        "carbon_skeleton_diameter",
        "nitrogen_atoms",
        "oxygen_atoms",
    }


def test_substitution_aware_amine_target_adds_only_local_integer_semantics() -> None:
    target = amine_semantic_target("CN1CCN(CCN)CC1", include_substitution_semantics=True)

    assert target.hydrogen_bond_donors == 1
    assert target.heavy_branch_atoms == 2
    assert UgiAmineSemanticTarget.from_mapping(target.to_mapping()) == target
    assert UgiAmineSemanticTarget.from_key(target.key) == target
    assert set(target.to_mapping()).difference(UgiAmineSemanticTarget(4, 3, 2, 0).to_mapping()) == {
        "hydrogen_bond_donors",
        "heavy_branch_atoms",
    }


def test_local_substitution_metrics_match_saturated_neutral_head_support() -> None:
    assert amine_local_substitution_metrics(("N", "C", "N", "C", "O"), (1, 2, 3, 3, 1)) == (2, 2)


def test_amine_semantic_target_fails_closed_on_unrepresented_elements() -> None:
    with pytest.raises(UgiAmineSemanticProgramError, match="only C/N/O"):
        amine_semantic_target("NCCS")


def test_amine_semantic_target_rejects_boolean_or_float_json_values() -> None:
    valid = UgiAmineSemanticTarget(4, 3, 2, 0).to_mapping()
    for invalid_value in (True, 4.0):
        changed = {**valid, "heavy_atom_graph_diameter": invalid_value}
        with pytest.raises(UgiAmineSemanticProgramError, match="must contain integers"):
            UgiAmineSemanticTarget.from_mapping(changed)


def test_conditional_distribution_balances_families_not_rows() -> None:
    program = amine_program_key(_program(6))
    compact = UgiAmineSemanticTarget(4, 3, 2, 0)
    long = UgiAmineSemanticTarget(7, 6, 2, 0)
    support = build_equal_family_conditional_distribution(
        [
            (program, "repeated", compact),
            (program, "repeated", compact),
            (program, "single", long),
        ]
    )
    targets, probabilities = support[program]
    mass = {target: probability for target, probability in zip(targets, probabilities, strict=True)}

    assert mass[compact] == pytest.approx(0.5)
    assert mass[long] == pytest.approx(0.5)


def test_semantic_prior_samples_once_per_matching_coarse_program() -> None:
    program = _program(6)
    key = amine_program_key(program)
    target = UgiAmineSemanticTarget(4, 3, 2, 1)
    prior = UgiMeasuredAmineSemanticPrior(
        support={key: ((target,), np.asarray([1.0], dtype=np.float64))},
        audit={},
    )

    assert prior.sample_for_programs((program, program), seed=7) == (target, target)


def test_semantic_prior_fails_closed_without_conditional_support() -> None:
    prior = UgiMeasuredAmineSemanticPrior(support={}, audit={})

    with pytest.raises(UgiAmineSemanticProgramError, match="lacks measured semantic support"):
        prior.sample_for_programs((_program(6),), seed=7)
