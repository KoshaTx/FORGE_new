from __future__ import annotations

import numpy as np

from experiments.phase1.hela_potency.morphology.ugi_morphology_proposal_strength_sweep import (
    _cluster_bootstrap,
    _effective_count,
)


def test_effective_count_handles_normalized_and_unnormalized_mass() -> None:
    assert _effective_count(np.asarray([0.5, 0.5])) == 2.0
    assert _effective_count(np.asarray([2.0, 2.0])) == 2.0
    assert _effective_count(np.asarray([1.0, 0.0])) == 1.0


def test_cluster_bootstrap_is_deterministic_and_detects_enrichment() -> None:
    groups = ["a", "b", "c", "d"]
    support = np.asarray([1.0, 1.0, 0.0, 0.0])
    broad = np.full(4, 0.25)
    candidates = np.asarray([[0.4, 0.4, 0.1, 0.1]])
    first = _cluster_bootstrap(
        groups=groups,
        support=support,
        broad=broad,
        candidates=candidates,
        replicates=200,
        seed=17,
    )
    second = _cluster_bootstrap(
        groups=groups,
        support=support,
        broad=broad,
        candidates=candidates,
        replicates=200,
        seed=17,
    )
    assert first == second
    assert first[0]["support_rate_ci95"][0] >= 0.0
    assert first[0]["support_rate_ci95"][1] <= 1.0
