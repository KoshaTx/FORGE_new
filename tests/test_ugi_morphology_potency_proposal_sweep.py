import numpy as np

from forge.potency.ugi_morphology_potency_proposal_sweep import (
    _calibrated_utility,
    _candidate_distribution,
)


def test_candidate_distribution_preserves_support() -> None:
    support = np.asarray([0.2, 0.3, 0.5])
    value = np.asarray([0.0, 0.5, 1.0])
    proposal = _candidate_distribution(support, value, gamma=0.5, beta=2.0)
    assert np.all(proposal > 0.0)
    assert np.isclose(proposal.sum(), 1.0)
    assert proposal[-1] > support[-1]


def test_calibrated_utility_is_neutral_below_median() -> None:
    calibration = np.asarray([0.0, 1.0, 2.0, 3.0])
    values = np.asarray([-1.0, 0.0, 1.0, 2.0, 4.0])
    utility = _calibrated_utility(values, calibration)
    assert np.all((0.0 <= utility) & (utility <= 1.0))
    assert np.all(utility[:2] == 0.0)
    assert utility[-1] == 1.0
