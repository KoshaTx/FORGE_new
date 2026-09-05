from __future__ import annotations

import numpy as np
import pytest

from forge.model.ugi_morphology_diversity import (
    UgiMorphologyDiversityError,
    categorical_kl,
    expected_unique_modes,
    sample_tempered_indices,
    select_morphology_temperature,
    summarize_sampled_modes,
    tempered_probabilities,
)
from forge.model.ugi_topology_diversity import (
    role_topology_signature,
    summarize_topology_diversity,
)


def test_temperature_flattens_probabilities_without_changing_support() -> None:
    source = np.asarray((0.7, 0.2, 0.08, 0.02), dtype=np.float64)

    treatment = tempered_probabilities(source, temperature=2.0)

    assert np.isclose(treatment.sum(), 1.0)
    assert np.all(treatment > 0)
    assert treatment[0] < source[0]
    assert treatment[-1] > source[-1]
    assert expected_unique_modes(treatment, count=64) > expected_unique_modes(source, count=64)
    assert categorical_kl(treatment, source) > 0


def test_temperature_selection_is_prespecified_and_fail_closed() -> None:
    source = np.asarray((0.7, 0.2, 0.08, 0.02), dtype=np.float64)
    selection = select_morphology_temperature(
        source,
        count=64,
        candidate_temperatures=(1.0, 1.5, 2.0, 3.0),
        minimum_expected_unique_ratio=1.05,
        maximum_kl=0.4,
    )

    assert selection.temperature > 1.0
    assert selection.expected_unique_ratio >= 1.05
    assert selection.kl_from_source <= 0.4
    assert selection.to_mapping()["target_attained"] is True

    with pytest.raises(UgiMorphologyDiversityError, match="no prespecified"):
        select_morphology_temperature(
            source,
            count=64,
            candidate_temperatures=(1.0, 1.01),
            minimum_expected_unique_ratio=2.0,
            maximum_kl=1e-8,
        )


def test_tempered_draw_is_iid_deterministic_and_attempt_weighted() -> None:
    source = np.asarray((0.5, 0.3, 0.2), dtype=np.float64)

    first = sample_tempered_indices(source, count=64, seed=17)
    second = sample_tempered_indices(source, count=64, seed=17)
    summary = summarize_sampled_modes(first, support_size=3)

    assert np.array_equal(first, second)
    assert summary["attempts"] == 64
    assert summary["unique_modes"] == 3
    assert 1.0 <= summary["effective_mode_count"] <= 3.0
    assert summary["maximum_mode_multiplicity"] == max(
        int(np.sum(first == index)) for index in range(3)
    )


def test_topology_signature_distinguishes_branch_placement() -> None:
    shallow_branch = role_topology_signature((2, 0, 1, 0), attachment_count=1, cycle_rank=0)
    deep_branch = role_topology_signature((1, 2, 0, 0), attachment_count=1, cycle_rank=0)

    assert shallow_branch != deep_branch


def _attempt(amine_offspring: tuple[int, ...]) -> dict[str, object]:
    return {
        "valid": True,
        "program": {
            "attachment_counts": [1, 2, 1],
            "cycle_ranks": [0, 0, 0],
            "junction_budgets": [1, 0, 0],
            "node_counts": [4, 3, 3],
        },
        "sampled_topology": {
            "offspring_by_role": [
                list(amine_offspring),
                [1, 0, 0],
                [1, 1, 0],
            ]
        },
    }


def test_topology_diversity_reports_shape_not_molecule_identity() -> None:
    rows = [
        _attempt((2, 0, 1, 0)),
        _attempt((2, 0, 1, 0)),
        _attempt((1, 2, 0, 0)),
        {"valid": False},
    ]

    summary = summarize_topology_diversity(rows)

    assert summary["attempts"] == 4
    assert summary["valid_topologies"] == 3
    assert summary["unique_complete_topology_signatures"] == 2
    assert 1.0 < summary["effective_complete_topology_count"] < 2.0
    assert summary["identity_fields_used"] is False
    assert summary["invalid_attempts_retained_in_denominator"] is True
