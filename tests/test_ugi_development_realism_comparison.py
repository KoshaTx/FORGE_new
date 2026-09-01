from __future__ import annotations

import pytest

from experiments.phase1.multireaction.ugi_development_realism_comparison import (
    PANEL_PRIMARY_METRICS,
    PANEL_WEIGHTINGS,
    UgiDevelopmentRealismComparisonError,
    _panel_comparison,
    _validate_metric_panel,
)


def _panel() -> dict[str, object]:
    return {
        "primary_metrics": list(PANEL_PRIMARY_METRICS),
        "weightings": list(PANEL_WEIGHTINGS),
        "secondary_metrics": ["role_c2st_auc"],
        "decision_rule": "descriptive_directional_consistency",
        "direction": "lower_is_better",
        "inference": "none_seed0_development_only",
    }


def _summary(value: float) -> dict[str, float]:
    return {
        f"{prefix}_{metric}": value
        for prefix in ("attempt", "unique")
        for metric in PANEL_PRIMARY_METRICS
    }


def test_metric_panel_classifies_all_primary_distances_in_both_weightings() -> None:
    result = _panel_comparison(
        {
            "base": _summary(1.0),
            "better": _summary(0.9),
            "worse": _summary(1.1),
            "mixed": {**_summary(0.9), "unique_role_rbf_mmd2": 1.1},
        },
        baseline_arm="base",
        panel=_validate_metric_panel(_panel()),
    )

    comparisons = result["comparisons"]
    assert comparisons["better"]["classification"] == "uniformly_improved"
    assert comparisons["better"]["primary_metrics_improved"] == 6
    assert comparisons["worse"]["classification"] == "uniformly_worsened"
    assert comparisons["worse"]["primary_metrics_improved"] == 0
    assert comparisons["mixed"]["classification"] == "mixed"
    assert comparisons["mixed"]["primary_metrics_improved"] == 5


def test_metric_panel_rejects_c2st_as_a_primary_metric() -> None:
    panel = _panel()
    panel["primary_metrics"] = [*PANEL_PRIMARY_METRICS, "role_c2st_auc"]

    with pytest.raises(UgiDevelopmentRealismComparisonError, match="primary metric panel"):
        _validate_metric_panel(panel)
