from __future__ import annotations

import numpy as np

from forge.potency.ugi_morphology_proposal_challenger_adjudication import (
    paired_cluster_bootstrap,
)


def test_paired_cluster_bootstrap_is_deterministic() -> None:
    kwargs = {
        "groups": ["a", "b", "c", "d"],
        "support": np.asarray([1.0, 1.0, 0.0, 0.0]),
        "current": np.asarray([0.3, 0.3, 0.2, 0.2]),
        "challenger": np.asarray([0.4, 0.4, 0.1, 0.1]),
        "replicates": 200,
        "seed": 91,
    }
    first = paired_cluster_bootstrap(**kwargs)
    second = paired_cluster_bootstrap(**kwargs)
    assert first == second
    assert first["absolute_difference_ci95"][1] >= 0.0
