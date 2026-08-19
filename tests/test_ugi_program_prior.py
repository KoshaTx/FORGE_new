from __future__ import annotations

import json

import numpy as np
import pytest

from forge.product.ugi_morphology_program import UgiMorphologyProgram
from forge.product.ugi_program_prior import (
    UgiProgramPriorError,
    blend_program_priors,
    build_component_family_balanced_program_prior,
    build_weighted_program_prior,
    load_program_prior,
    sample_weighted_program_prior,
    validate_program_prior_support,
)

BOUNDS = {
    "maximum_component_atoms": 12,
    "maximum_total_atoms": 30,
    "maximum_junction_budget": 3,
    "maximum_cycle_rank": 1,
    "maximum_attachment_count": 2,
}


def _programs() -> tuple[UgiMorphologyProgram, ...]:
    return (
        UgiMorphologyProgram((4, 8, 9), (1, 0, 1), (0, 0, 0), (1, 1, 1)),
        UgiMorphologyProgram((6, 10, 7), (2, 1, 0), (1, 0, 0), (1, 1, 1)),
    )


def test_weighted_program_prior_preserves_role_marginals_without_components() -> None:
    prior = build_weighted_program_prior(_programs(), np.asarray([0.75, 0.25]))

    assert validate_program_prior_support(prior, BOUNDS)["status"] == "pass"
    assert np.allclose(prior.probabilities_by_role[0], [0.75, 0.25])
    samples = sample_weighted_program_prior(
        prior,
        count=32,
        rng=np.random.default_rng(7),
        bounds=BOUNDS,
    )
    assert len(samples) == 32
    assert all(sample.node_count <= BOUNDS["maximum_total_atoms"] for sample in samples)


def test_invalid_weighted_program_prior_is_rejected() -> None:
    with pytest.raises(UgiProgramPriorError):
        build_weighted_program_prior(_programs(), np.asarray([1.0, -1.0]))


def test_component_family_balanced_prior_and_blend() -> None:
    programs = (
        UgiMorphologyProgram((4, 8, 8), (0, 0, 0), (0, 0, 0)),
        UgiMorphologyProgram((4, 8, 8), (0, 1, 0), (0, 0, 0)),
        UgiMorphologyProgram((4, 8, 8), (0, 1, 0), (0, 0, 0)),
    )
    roles = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
    keys = tuple({role: f"{role}-{index}" for role in roles} for index in range(len(programs)))
    families = (
        {role: f"{role}-family-a" for role in roles},
        {role: f"{role}-family-b" for role in roles},
        {role: f"{role}-family-b" for role in roles},
    )
    component_prior, counts = build_component_family_balanced_program_prior(
        programs, keys, families
    )
    assert counts == {role: 3 for role in roles}
    aldehyde_index = roles.index("oxoester_aldehyde_body_tail")
    positive = sum(
        probability
        for state, probability in zip(
            component_prior.support_by_role[aldehyde_index],
            component_prior.probabilities_by_role[aldehyde_index],
            strict=True,
        )
        if state[1] > 0
    )
    assert np.isclose(positive, 0.5)
    realism = build_weighted_program_prior(programs, np.asarray([1.0, 0.0, 0.0]))
    blended = blend_program_priors(realism, component_prior, exploration_mass=0.5)
    blended_positive = sum(
        probability
        for state, probability in zip(
            blended.support_by_role[aldehyde_index],
            blended.probabilities_by_role[aldehyde_index],
            strict=True,
        )
        if state[1] > 0
    )
    assert np.isclose(blended_positive, 0.25)


def test_program_prior_artifact_round_trip(tmp_path) -> None:
    prior = build_weighted_program_prior(_programs(), np.asarray([0.75, 0.25]))
    role_priors = {}
    for role, support, probabilities in zip(
        ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail"),
        prior.support_by_role,
        prior.probabilities_by_role,
        strict=True,
    ):
        role_priors[role] = [
            {
                "node_count": state[0],
                "junction_budget": state[1],
                "cycle_rank": state[2],
                "attachment_count": state[3],
                "probability": float(probability),
            }
            for state, probability in zip(support, probabilities, strict=True)
        ]
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "phase1_ugi_program_prior.v1",
                "status": "pass",
                "bounds": BOUNDS,
                "role_priors": role_priors,
            }
        )
    )

    loaded, bounds = load_program_prior(path)

    assert bounds == BOUNDS
    assert loaded.support_by_role == prior.support_by_role
    assert all(
        np.allclose(left, right)
        for left, right in zip(
            loaded.probabilities_by_role,
            prior.probabilities_by_role,
            strict=True,
        )
    )
