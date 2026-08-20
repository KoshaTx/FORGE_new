from __future__ import annotations

import numpy as np

from forge.potency.applicability.ugi_applicability_recalibration import (
    _group_id,
    _meta_fold,
    _normalized_radius,
    _passes,
    _select_threshold,
)


def _row(label: str, scheme: str, radius: float, truth: float, prediction: float) -> dict:
    return {
        "label": label,
        "scheme": scheme,
        "normalized_radius": radius,
        "y_true": truth,
        "y_pred": prediction,
        "absolute_error": abs(truth - prediction),
        "conformal_q90": 1.0,
    }


def test_meta_fold_is_deterministic_and_bounded() -> None:
    observed = [_meta_fold("A1B2C3", salt="frozen", folds=5) for _ in range(3)]
    assert observed[0] == observed[1] == observed[2]
    assert 0 <= observed[0] < 5


def test_group_id_matches_the_held_component_unit() -> None:
    label = "A10B12C5"
    assert _group_id(label, "held_head_5fold") == "amine:10"
    assert _group_id(label, "held_aldehyde_5fold") == "aldehyde:12"
    assert _group_id(label, "held_isocyanide_5fold") == "isocyanide:5"
    assert _group_id(label, "held_aldehyde_isocyanide_pair_5fold") == "aldehyde:12|isocyanide:5"


def test_normalized_radius_uses_worst_view_and_modality() -> None:
    thresholds = {
        view: {kind: {"interpolative_max": 2.0} for kind in ("fingerprint", "descriptor")}
        for view in ("product", "amine", "aldehyde", "isocyanide")
    }
    row = {f"{view}_{kind}_distance": 1.0 for view in thresholds for kind in thresholds[view]}
    row["aldehyde_descriptor_distance"] = 3.0

    assert _normalized_radius(row, thresholds) == 1.5


def test_select_threshold_returns_widest_passing_radius() -> None:
    schemes = ("s1", "s2")
    rows = []
    for scheme_index, scheme in enumerate(schemes):
        for index in range(24):
            truth = float(index + scheme_index)
            prediction = truth + (0.1 if index < 18 else 5.0)
            radius = 1.0 if index < 18 else 2.0
            rows.append(_row(f"label-{index}", scheme, radius, truth, prediction))
    criteria = {
        "minimum_overall_coverage": 0.5,
        "minimum_per_scheme_coverage": 0.5,
        "minimum_mae_relative_reduction": 0.1,
        "minimum_per_scheme_spearman": 0.2,
        "minimum_top_quartile_enrichment": 0.0,
        "minimum_per_scheme_r2": 0.1,
        "maximum_per_scheme_coverage90_gap": 0.2,
    }

    selected, curve = _select_threshold(rows, (1.0, 2.0), schemes, criteria)

    assert selected == 1.0
    assert _passes(curve[0], criteria)
    assert not _passes(curve[1], criteria)


def test_exact_novelty_is_not_an_input_to_radius() -> None:
    thresholds = {
        view: {kind: {"interpolative_max": 1.0} for kind in ("fingerprint", "descriptor")}
        for view in ("product", "amine", "aldehyde", "isocyanide")
    }
    base = {f"{view}_{kind}_distance": 0.5 for view in thresholds for kind in thresholds[view]}

    seen = _normalized_radius({**base, "exact_identity": "seen"}, thresholds)
    novel = _normalized_radius({**base, "exact_identity": "new"}, thresholds)

    assert np.isclose(seen, novel)
