import numpy as np

from forge.potency.ugi_morphology_high_potency_challenger import classification_metrics


def test_classification_metrics_reward_correct_ordering() -> None:
    labels = np.asarray([0, 0, 1, 1])
    strong = classification_metrics(labels, np.asarray([0.1, 0.2, 0.8, 0.9]))
    flat = classification_metrics(labels, np.asarray([0.5, 0.5, 0.5, 0.5]))
    assert strong["roc_auc"] == 1.0
    assert strong["average_precision"] > flat["average_precision"]
    assert strong["log_loss"] < flat["log_loss"]
