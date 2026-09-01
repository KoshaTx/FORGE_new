from __future__ import annotations

import csv
import gzip
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from forge.core.hashing import sha256_file
from forge.model.ugi_joint_lipid_prior import (
    UgiJointLipidPrior,
    UgiJointLipidPriorError,
)


class _Cache:
    def __init__(self) -> None:
        self.vocabulary = SimpleNamespace(program_to_index={"ugi_3cr_agile": 1, "bl": 2})
        self.arrays = {
            "source_weights": np.asarray([1.0, 1.0, 1.0, 2.0, 6.0, 1.0]),
        }
        self._record_ids = ("held", "measured-a", "measured-b", "virtual-a", "virtual-b", "bl")

    def __len__(self) -> int:
        return len(self._record_ids)

    def indices(self, *, program_id: str, fold: str) -> np.ndarray:
        assert fold == "train"
        if program_id == "ugi_3cr_agile":
            return np.asarray([1, 2, 3, 4], dtype=np.int64)
        if program_id == "bl":
            return np.asarray([5], dtype=np.int64)
        return np.asarray([], dtype=np.int64)

    def record_id(self, index: int) -> str:
        return self._record_ids[index]


def _write_assignments(path: Path) -> None:
    with gzip.open(path, mode="wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "product_id",
                "primary_product_fold",
                "is_source_adjudicated_measured_product",
            ),
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "product_id": "measured-a",
                    "primary_product_fold": "train",
                    "is_source_adjudicated_measured_product": "true",
                },
                {
                    "product_id": "measured-b",
                    "primary_product_fold": "train",
                    "is_source_adjudicated_measured_product": "1",
                },
                {
                    "product_id": "held",
                    "primary_product_fold": "heldout",
                    "is_source_adjudicated_measured_product": "true",
                },
                {
                    "product_id": "virtual-a",
                    "primary_product_fold": "train",
                    "is_source_adjudicated_measured_product": "false",
                },
            )
        )


def test_joint_lipid_measure_preserves_declared_strata_and_excludes_holdout(
    tmp_path: Path,
) -> None:
    assignments = tmp_path / "assignments.csv.gz"
    _write_assignments(assignments)
    prior = UgiJointLipidPrior.from_training_assignments(
        assignments,
        measured_mass=0.5,
        exploration_mass=0.5,
        expected_sha256=sha256_file(assignments),
    )

    measure, receipt = prior.training_measure(_Cache())  # type: ignore[arg-type]

    assert np.isclose(measure.sum(), 1.0)
    assert np.allclose(measure[[1, 2]], [0.25, 0.25])
    assert np.allclose(measure[[3, 4]], [0.125, 0.375])
    assert measure[0] == measure[5] == 0.0
    assert receipt["realized_mass"] == {"measured": 0.5, "exploration": 0.5}
    assert receipt["measured_rows"] == 2
    assert receipt["exploration_rows"] == 2
    assert receipt["product_ids_enter_neural_tensors"] is False


def test_joint_lipid_policy_receipt_does_not_expose_product_ids(tmp_path: Path) -> None:
    assignments = tmp_path / "assignments.csv.gz"
    _write_assignments(assignments)
    prior = UgiJointLipidPrior.from_training_assignments(
        assignments,
        measured_mass=0.75,
        exploration_mass=0.25,
    )

    receipt = repr(prior.to_mapping())

    assert "measured-a" not in receipt
    assert "measured-b" not in receipt
    assert "component_identity_conditioning': False" in receipt


def test_joint_lipid_measure_fails_if_a_measured_training_product_is_missing(
    tmp_path: Path,
) -> None:
    assignments = tmp_path / "assignments.csv.gz"
    _write_assignments(assignments)
    prior = UgiJointLipidPrior.from_training_assignments(
        assignments,
        measured_mass=0.5,
        exploration_mass=0.5,
    )
    altered = UgiJointLipidPrior(
        **{
            **prior.__dict__,
            "measured_training_record_ids": (*prior.measured_training_record_ids, "missing"),
        }
    )

    with pytest.raises(UgiJointLipidPriorError, match="lacks 1 measured"):
        altered.training_measure(_Cache())  # type: ignore[arg-type]
