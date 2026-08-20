import numpy as np

from experiments.phase1.hela_potency.morphology.ugi_morphology_potency_signal import (
    prediction_metrics,
)


def test_prediction_metrics_detect_rank_and_top_quartile_signal() -> None:
    observed = np.arange(8, dtype=np.float64)
    metrics = prediction_metrics(observed, observed.copy())
    assert metrics["midrank_spearman"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["top_quartile_observed_gain"] > 0.0


def test_weighted_metrics_are_finite() -> None:
    observed = np.asarray([0.0, 1.0, 2.0, 4.0])
    predicted = np.asarray([0.1, 0.8, 2.2, 3.0])
    weights = np.asarray([1.0, 1.0, 2.0, 5.0])
    metrics = prediction_metrics(observed, predicted, weights)
    assert np.isfinite(list(metrics.values())).all()
    assert metrics["midrank_spearman"] > 0.0
