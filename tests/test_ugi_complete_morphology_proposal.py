from __future__ import annotations

import numpy as np
import pytest

from forge.product.ugi_complete_morphology_proposal import (
    UgiCompleteMorphologyProposalError,
    enumerate_complete_program_support,
    support_preserving_probabilities,
)
from forge.product.ugi_program_prior import UgiProgramPrior


def _prior() -> UgiProgramPrior:
    return UgiProgramPrior(
        support_by_role=(
            ((2, 0, 0, 1), (3, 0, 0, 1)),
            ((4, 0, 0, 1),),
            ((5, 0, 0, 1), (6, 0, 0, 1)),
        ),
        probabilities_by_role=(
            np.asarray([0.25, 0.75]),
            np.asarray([1.0]),
            np.asarray([0.4, 0.6]),
        ),
    )


def test_complete_support_is_cartesian_and_preserves_broad_mass() -> None:
    records = enumerate_complete_program_support(_prior(), {"maximum_total_atoms": 20})
    assert len(records) == 4
    assert len({row["program_sha256"] for row in records}) == 4
    assert sum(row["broad_prior_probability"] for row in records) == pytest.approx(1.0)


def test_total_size_bound_filters_and_renormalizes_support() -> None:
    records = enumerate_complete_program_support(_prior(), {"maximum_total_atoms": 11})
    assert len(records) == 1
    assert records[0]["broad_prior_probability"] == pytest.approx(1.0)


def test_support_preserving_mixture_keeps_every_program_and_exact_importance_ratio() -> None:
    broad = np.asarray([0.1, 0.2, 0.3, 0.4])
    scores = np.asarray([0.01, 0.1, 0.5, 0.9])
    proposal, importance = support_preserving_probabilities(
        broad,
        scores,
        mixture_rho=0.5,
        score_floor=1e-6,
        score_power=1.0,
    )
    assert proposal.sum() == pytest.approx(1.0)
    assert np.all(proposal > 0.0)
    assert np.allclose(importance, broad / proposal)
    assert float(np.max(importance)) <= 2.0


def test_invalid_support_preserving_request_fails_closed() -> None:
    with pytest.raises(UgiCompleteMorphologyProposalError):
        support_preserving_probabilities(
            np.asarray([0.0, 1.0]),
            np.asarray([0.1, 0.2]),
            mixture_rho=0.5,
            score_floor=1e-6,
            score_power=1.0,
        )
