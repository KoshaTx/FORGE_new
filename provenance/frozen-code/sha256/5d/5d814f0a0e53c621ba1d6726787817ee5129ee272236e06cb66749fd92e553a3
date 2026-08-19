from __future__ import annotations

import numpy as np
import pytest

from forge.bio.ugi_morphology_enriched_oracle_audit import (
    UgiMorphologyEnrichedOracleAuditError,
    effective_sample_size,
    weighted_prediction_metrics,
)


def test_effective_sample_size_is_scale_invariant() -> None:
    weights = np.asarray([1.0, 2.0, 3.0])
    assert effective_sample_size(weights) == pytest.approx(effective_sample_size(10.0 * weights))


def test_equal_weights_recover_standard_error_metrics() -> None:
    observed = np.asarray([1.0, 2.0, 4.0, 8.0])
    predicted = np.asarray([1.0, 3.0, 3.0, 7.0])
    metrics = weighted_prediction_metrics(
        observed,
        predicted,
        np.ones(4),
        np.asarray([True, True, False, True]),
    )
    residual = observed - predicted
    assert metrics["mae"] == pytest.approx(float(np.mean(np.abs(residual))))
    assert metrics["rmse"] == pytest.approx(float(np.sqrt(np.mean(residual**2))))
    assert metrics["effective_sample_size"] == pytest.approx(4.0)
    assert metrics["conformal90_coverage"] == pytest.approx(0.75)


def test_positive_weighting_changes_metrics_without_changing_records() -> None:
    observed = np.asarray([0.0, 1.0, 10.0])
    predicted = np.asarray([0.0, 1.0, 0.0])
    uniform = weighted_prediction_metrics(observed, predicted, np.ones(3), np.ones(3, dtype=bool))
    shifted = weighted_prediction_metrics(
        observed,
        predicted,
        np.asarray([10.0, 10.0, 1.0]),
        np.ones(3, dtype=bool),
    )
    assert shifted["records"] == uniform["records"] == 3
    assert float(shifted["mae"]) < float(uniform["mae"])


def test_nonpositive_importance_weight_fails_closed() -> None:
    with pytest.raises(UgiMorphologyEnrichedOracleAuditError):
        effective_sample_size(np.asarray([1.0, 0.0]))
