from __future__ import annotations

from forge.route.ugi3_agile_template_saturation_stress import projected_leaf_candidates


def test_projected_leaf_candidates_excludes_internal_intermediate() -> None:
    steps = [
        {
            "reactants": ["acid", "diol"],
            "product": "alcohol_intermediate",
        },
        {
            "reactants": ["alcohol_intermediate"],
            "product": "aldehyde_target",
        },
    ]
    assert projected_leaf_candidates(steps) == ("acid", "diol")


def test_projected_leaf_candidates_deduplicates_shared_leaf() -> None:
    steps = [
        {"reactants": ["leaf"], "product": "middle"},
        {"reactants": ["leaf", "middle"], "product": "target"},
    ]
    assert projected_leaf_candidates(steps) == ("leaf",)
