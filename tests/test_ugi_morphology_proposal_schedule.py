from __future__ import annotations

import numpy as np
import pytest

from experiments.phase1.synthesis_guidance.schedule.ugi_morphology_proposal_schedule import (
    UgiMorphologyProposalScheduleError,
    _program_dict,
)


def test_program_dict_is_role_complete() -> None:
    program = _program_dict(
        {
            "node_counts": [3, 14, 15],
            "junction_budgets": [1, 0, 0],
            "cycle_ranks": [0, 0, 0],
            "attachment_counts": [1, 1, 1],
        }
    )
    assert program["node_counts"] == (3, 14, 15)


def test_program_dict_fails_on_missing_role_vector() -> None:
    with pytest.raises(UgiMorphologyProposalScheduleError):
        _program_dict(
            {
                "node_counts": [3, 14],
                "junction_budgets": [1, 0, 0],
                "cycle_ranks": [0, 0, 0],
                "attachment_counts": [1, 1, 1],
            }
        )


def test_support_preserving_mixture_has_positive_mass() -> None:
    scores = np.asarray([0.001, 0.01, 0.1])
    rho = 0.5
    prior = 1.0 / len(scores)
    weighted = scores / scores.sum()
    proposal = (1.0 - rho) * prior + rho * weighted
    assert np.all(proposal > 0.0)
    assert proposal.sum() == pytest.approx(1.0)
