from __future__ import annotations

import numpy as np
import pytest

from forge.model.ugi_measured_joint_program_prior import (
    UgiMeasuredJointProgramPriorError,
    build_group_balanced_program_distribution,
    summarize_program_distribution,
)
from forge.model.ugi_morphology_program import UgiMorphologyProgram


def _program(amine: int, aldehyde: int, isocyanide: int) -> UgiMorphologyProgram:
    return UgiMorphologyProgram(
        node_counts=(amine, aldehyde, isocyanide),
        junction_budgets=(0, 1, 0),
        cycle_ranks=(0, 0, 0),
        attachment_counts=(1, 1, 1),
    )


def test_group_balanced_distribution_does_not_weight_groups_by_row_count() -> None:
    first = _program(4, 18, 14)
    second = _program(6, 22, 14)
    third = _program(8, 20, 14)

    programs, probabilities = build_group_balanced_program_distribution(
        {"large_family": [first, second], "small_family": [third]}
    )
    mass = {
        program.node_counts: probability
        for program, probability in zip(programs, probabilities.tolist(), strict=True)
    }

    assert mass == {
        first.node_counts: pytest.approx(0.25),
        second.node_counts: pytest.approx(0.25),
        third.node_counts: pytest.approx(0.50),
    }
    assert np.isclose(probabilities.sum(), 1.0)


def test_program_summary_reports_tail_size_and_asymmetry() -> None:
    programs = (_program(4, 18, 14), _program(8, 22, 14))
    summary = summarize_program_distribution(programs, np.asarray([0.25, 0.75]))

    assert summary["expected_node_count_by_role"] == {
        "amine_head": pytest.approx(7.0),
        "oxoester_aldehyde_body_tail": pytest.approx(21.0),
        "isocyanide_tail": pytest.approx(14.0),
    }
    assert summary["expected_two_tail_exterior_node_total"] == pytest.approx(35.0)
    assert summary["expected_two_tail_exterior_node_asymmetry"] == pytest.approx(7.0)


def test_group_balanced_distribution_rejects_empty_groups() -> None:
    with pytest.raises(UgiMeasuredJointProgramPriorError, match="empty group"):
        build_group_balanced_program_distribution({"empty": []})
