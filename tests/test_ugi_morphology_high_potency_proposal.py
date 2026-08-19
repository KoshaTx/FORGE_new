import numpy as np

from forge.potency.ugi_morphology_potency_proposal_sweep import _candidate_distribution


def test_candidate_distribution_preserves_full_support() -> None:
    support = np.asarray([0.2, 0.3, 0.5])
    value = np.asarray([0.0, 0.4, 1.0])
    proposal = _candidate_distribution(support, value, gamma=0.5, beta=2.0)
    assert np.all(proposal > 0.0)
    assert np.isclose(proposal.sum(), 1.0)
    assert proposal[-1] > support[-1]
