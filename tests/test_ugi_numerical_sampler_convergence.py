from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.archive.producers.phase1_evaluate_ugi_numerical_sampler_convergence import (
    _assert_matched_programs,
    _distribution_distance,
)


def test_distribution_distance_has_expected_extremes() -> None:
    identical = _distribution_distance(np.asarray([2.0, 1.0]), np.asarray([4.0, 2.0]))
    disjoint = _distribution_distance(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]))

    assert identical == {"total_variation": 0.0, "jensen_shannon_nats": 0.0}
    assert disjoint["total_variation"] == 1.0
    assert disjoint["jensen_shannon_nats"] == pytest.approx(math.log(2.0))


def test_program_matching_does_not_require_identical_numerical_trajectory() -> None:
    common = {
        "product_id": "program-prior-000001",
        "program": {
            "node_counts": [4, 8, 9],
            "junction_budgets": [0, 0, 0],
            "cycle_ranks": [0, 0, 0],
            "attachment_counts": [1, 1, 1],
        },
        "source_stratum": "training_fold_weighted_program_prior",
        "branch_class": "linear_tail_origins",
    }
    left = {"samples": [{**common, "offspring_by_role": {"amine_head": [1, 0]}}]}
    right = {"samples": [{**common, "offspring_by_role": {"amine_head": [0, 1]}}]}

    _assert_matched_programs(left, right)


def test_program_matching_rejects_a_changed_morphology_condition() -> None:
    left = {
        "samples": [
            {
                "product_id": "program-prior-000001",
                "program": {"node_counts": [4, 8, 9]},
                "source_stratum": "prior",
                "branch_class": "linear",
            }
        ]
    }
    right = {
        "samples": [
            {
                "product_id": "program-prior-000001",
                "program": {"node_counts": [4, 8, 10]},
                "source_stratum": "prior",
                "branch_class": "linear",
            }
        ]
    }

    with pytest.raises(ValueError, match="differs at program"):
        _assert_matched_programs(left, right)
