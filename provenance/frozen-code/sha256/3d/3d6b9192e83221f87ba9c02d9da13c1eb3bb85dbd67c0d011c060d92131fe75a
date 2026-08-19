from __future__ import annotations

import numpy as np

from forge.bio.ugi_interpolative_conformal import _scheme_metrics


def test_scheme_metrics_require_minimum_fold_size() -> None:
    rows = [
        {
            "fold": 0,
            "y_true": float(index),
            "y_pred": float(index),
            "covered90": True,
        }
        for index in range(19)
    ]
    assert _scheme_metrics(rows, minimum_test_rows_per_fold=20)["eligible_folds"] == 0


def test_scheme_metrics_report_foldwise_gate_inputs() -> None:
    rows = []
    for fold in range(3):
        for index in range(25):
            truth = float(index)
            rows.append(
                {
                    "fold": fold,
                    "y_true": truth,
                    "y_pred": truth + 0.1 * np.sin(index),
                    "covered90": index != 0,
                }
            )
    result = _scheme_metrics(rows, minimum_test_rows_per_fold=20)
    assert result["eligible_folds"] == 3
    assert result["positive_r2_fold_fraction"] == 1.0
    assert result["mean_test_spearman_rho"] > 0.99
    assert 0.0 <= result["mean_absolute_90pct_coverage_gap"] <= 0.1
