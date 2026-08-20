from __future__ import annotations

import numpy as np
import pytest

from forge.design.schedule.ugi_matched_morphology_allocation_schedule import (
    UgiMatchedMorphologyAllocationScheduleError,
    inverse_cdf_indices,
)


def test_inverse_cdf_uses_one_common_uniform_coordinate() -> None:
    uniforms = [0.10, 0.60, 0.99]
    broad = inverse_cdf_indices([0.5, 0.5], uniforms)
    tilted = inverse_cdf_indices([0.2, 0.8], uniforms)
    assert broad.tolist() == [0, 1, 1]
    assert tilted.tolist() == [0, 1, 1]
    assert inverse_cdf_indices([0.5, 0.5], uniforms).tolist() == broad.tolist()


@pytest.mark.parametrize(
    ("probabilities", "uniforms"),
    [
        ([0.0, 1.0], [0.5]),
        ([0.4, 0.4], [0.5]),
        ([0.5, 0.5], [1.0]),
        ([0.5, np.nan], [0.5]),
    ],
)
def test_inverse_cdf_fails_closed_on_invalid_support(
    probabilities: list[float], uniforms: list[float]
) -> None:
    with pytest.raises(UgiMatchedMorphologyAllocationScheduleError):
        inverse_cdf_indices(probabilities, uniforms)


def test_inverse_cdf_boundary_assignment_is_deterministic() -> None:
    selected = inverse_cdf_indices([0.25, 0.75], [0.0, 0.249999, 0.25, 0.999999])
    assert selected.tolist() == [0, 0, 1, 1]
