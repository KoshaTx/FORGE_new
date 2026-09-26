"""Protect partial-versus-complete component novelty and malformed overlap handling."""

import pytest

from experiments.phase1.multireaction.iclr22_cal_strata import classify


@pytest.mark.parametrize(
    "overlap,expected,roles",
    [
        (["a", "b"], "all_components_seen", []),
        (["a"], "some_components_unseen", ["tail"]),
        ([], "all_components_unseen", ["head", "tail"]),
    ],
)
def test_repeated_arms_do_not_change_identity_stratum(overlap, expected, roles):
    row = {
        "component_instances": [["head", "a", 1], ["tail", "b", 2]],
        "component_identity_overlap_TRAIN": overlap,
        "component_disjoint_reference_eligible": not overlap,
    }
    assert classify(row) == (expected, roles)


def test_foreign_overlap_identity_is_rejected():
    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        classify(
            {
                "component_instances": [["head", "a", 1]],
                "component_identity_overlap_TRAIN": ["wrong"],
                "component_disjoint_reference_eligible": False,
            }
        )


def test_wholly_disjoint_label_cannot_be_relaxed_to_partial():
    with pytest.raises(ValueError, match="wholly-disjoint predicate"):
        classify(
            {
                "component_instances": [["head", "a", 1], ["tail", "b", 1]],
                "component_identity_overlap_TRAIN": ["a"],
                "component_disjoint_reference_eligible": True,
            }
        )
