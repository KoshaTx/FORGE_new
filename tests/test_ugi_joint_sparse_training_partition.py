from __future__ import annotations

import pytest

from experiments.phase1.product_l1.training.ugi_joint_sparse_training import (
    UgiJointSparseTrainingError,
    _merge_fold_values,
    _training_partition,
)

FOLDS = ("train", "calibration", "heldout")


def test_default_development_partition_is_disjoint() -> None:
    partition = _training_partition(
        {},
        FOLDS,
        {"early_stopping": {"patience": 4}},
    )

    assert partition.mode == "development_split"
    assert partition.training_folds == ("train",)
    assert partition.diagnostic_folds == ("calibration",)
    assert partition.selection_mode == "calibration_early_stopping"
    assert partition.overlapping_folds == ()


def test_production_refit_requires_all_folds_and_fixed_final_step() -> None:
    config = {
        "training_partition": {
            "mode": "production_refit_all_folds",
            "training_folds": list(FOLDS),
            "diagnostic_folds": list(FOLDS),
            "selection_mode": "fixed_final_step",
        }
    }
    partition = _training_partition(
        config,
        FOLDS,
        {"early_stopping": {"patience": 0}},
    )

    assert partition.training_folds == FOLDS
    assert partition.overlapping_folds == tuple(sorted(FOLDS))


@pytest.mark.parametrize(
    ("training_folds", "selection_mode", "patience", "message"),
    [
        (("train", "calibration"), "fixed_final_step", 0, "every available"),
        (FOLDS, "calibration_early_stopping", 0, "fixed final step"),
        (FOLDS, "fixed_final_step", 1, "data-dependent early stopping"),
    ],
)
def test_production_refit_rejects_selection_leakage(
    training_folds: tuple[str, ...],
    selection_mode: str,
    patience: int,
    message: str,
) -> None:
    config = {
        "training_partition": {
            "mode": "production_refit_all_folds",
            "training_folds": list(training_folds),
            "diagnostic_folds": list(FOLDS),
            "selection_mode": selection_mode,
        }
    }

    with pytest.raises(UgiJointSparseTrainingError, match=message):
        _training_partition(
            config,
            FOLDS,
            {"early_stopping": {"patience": patience}},
        )


def test_development_partition_rejects_training_calibration_overlap() -> None:
    config = {
        "training_partition": {
            "mode": "development_split",
            "training_folds": ["train", "calibration"],
            "diagnostic_folds": ["calibration"],
            "selection_mode": "calibration_early_stopping",
        }
    }

    with pytest.raises(UgiJointSparseTrainingError, match="must be disjoint"):
        _training_partition(config, FOLDS, {"early_stopping": {"patience": 4}})


def test_merge_fold_values_preserves_declared_fold_and_record_order() -> None:
    values = {
        "train": ("t0", "t1"),
        "calibration": ("c0",),
        "heldout": ("h0", "h1"),
    }

    assert _merge_fold_values(values, ("heldout", "train")) == (
        "h0",
        "h1",
        "t0",
        "t1",
    )
