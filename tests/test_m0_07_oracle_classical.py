from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forge.bio.oracle_classical import (
    OracleClassicalError,
    Partition,
    aggregate_selection_metrics,
    build_estimator,
    build_feature_bundle,
    conformal_radius,
    derive_fit_seed,
    evaluate_partition,
    load_classical_config,
)

REPO = Path(__file__).resolve().parents[1]


def test_feature_bundle_is_deterministic_and_structure_only() -> None:
    frame = pd.DataFrame(
        {
            "label": ["A1B1C1", "A2B1C1", "A3B1C1"],
            "model_smiles": ["CCN", "CCO", "CCCC(=O)OCC"],
        }
    )
    config = load_classical_config(REPO / "configs/bio/m0_07_oracle_classical.json")

    first = build_feature_bundle(frame, config["representations"])
    second = build_feature_bundle(frame, config["representations"])

    assert first.labels == ("A1B1C1", "A2B1C1", "A3B1C1")
    assert first.matrices["morgan_count"].shape == (3, 1024)
    assert first.matrices["rdkit_expert"].shape[0] == 3
    assert first.matrices["rdkit_expert"].shape[1] > 200
    assert first.matrices["morgan_plus_rdkit_expert"].shape[1] == (
        first.matrices["morgan_count"].shape[1] + first.matrices["rdkit_expert"].shape[1]
    )
    for name in first.matrices:
        np.testing.assert_equal(first.matrices[name], second.matrices[name])


def test_fit_seed_is_stable_and_context_specific() -> None:
    first = derive_fit_seed(1729, "morgan", "ridge", "hela", "held_head", 0)
    second = derive_fit_seed(1729, "morgan", "ridge", "hela", "held_head", 0)
    changed = derive_fit_seed(1729, "morgan", "ridge", "hela", "held_head", 1)

    assert first == second
    assert first != changed
    assert 0 < first < 2**31 - 1


def test_conformal_radius_uses_finite_sample_rank() -> None:
    residuals = np.asarray([1.0, 2.0, 3.0, 4.0])

    assert conformal_radius(residuals, 0.8) == 4.0
    assert conformal_radius(residuals, 0.5) == 3.0


def test_ridge_preprocessing_is_fit_only_on_passed_training_rows() -> None:
    estimator = build_estimator("ridge", {"alpha": 1.0}, seed=1729)
    train_x = np.asarray([[0.0], [2.0], [4.0]])
    train_y = np.asarray([0.0, 1.0, 2.0])

    estimator.fit(train_x, train_y)

    assert estimator.named_steps["scale"].mean_[0] == pytest.approx(2.0)


def test_evaluate_partition_uses_calibration_only_for_intervals() -> None:
    features = np.arange(12, dtype=np.float64).reshape(-1, 1)
    targets = np.arange(12, dtype=np.float64)
    labels = [f"L{index}" for index in range(12)]
    partition = Partition(
        scheme="held_group",
        fold=0,
        train_indices=np.arange(0, 6),
        calibration_indices=np.arange(6, 9),
        test_indices=np.arange(9, 12),
    )

    metric, predictions = evaluate_partition(
        features,
        targets,
        labels,
        partition,
        representation="descriptor",
        model_name="ridge",
        endpoint="test",
        model_config={"alpha": 1.0},
        global_seed=1729,
        coverages=[0.8, 0.9, 0.95],
    )

    assert metric["train_rows"] == 6
    assert metric["calibration_rows"] == 3
    assert metric["test_rows"] == 3
    assert len(predictions) == 3
    assert {row["label"] for row in predictions} == {"L9", "L10", "L11"}
    assert metric["conformal_q90"] >= 0.0


def test_selection_aggregation_equal_weights_schemes() -> None:
    schemes = [f"scheme_{index}" for index in range(7)]
    rows = []
    for model, offset in (("good", 0.5), ("bad", -0.5)):
        for scheme in schemes:
            rows.append(
                {
                    "endpoint": "hela",
                    "representation": "morgan",
                    "model": model,
                    "scheme": scheme,
                    "test_r2": offset,
                    "test_rmse": 1.0 - offset,
                    "test_mae": 0.8 - offset,
                    "test_pearson_r": offset,
                    "test_spearman_rho": offset,
                    "test_coverage90": 0.9,
                    "test_mean_interval_width90": 2.0,
                }
            )

    aggregate, leaders = aggregate_selection_metrics(rows, schemes)

    assert len(aggregate) == 2
    assert leaders["hela"]["model"] == "good"
    assert leaders["hela"]["eligible_schemes"] == 7


def test_config_rejects_random_split_selection(tmp_path: Path) -> None:
    config = json.loads((REPO / "configs/bio/m0_07_oracle_classical.json").read_text())
    config["evaluation"]["model_selection_uses_random_split"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    with pytest.raises(OracleClassicalError, match="random split"):
        load_classical_config(path)


@pytest.mark.needs_vendor
def test_frozen_classical_result_when_present() -> None:
    result_path = REPO / "results/m0_07/oracle_classical_result.json"
    if not result_path.exists():
        pytest.skip("M0-07 classical oracle matrix has not been generated")
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == "m0_07_oracle_classical.v1"
    assert result["status"] == "classical_lane_complete_oracle_not_frozen"
    assert result["summary"]["curated_records"] == 1100
    assert result["summary"]["virtual_candidate_records"] == 12276
    assert result["summary"]["model_fits"] == 768
    assert not result["decision"]["oracle_model_frozen"]
    assert "lantern_random" not in result["evaluation_contract"]["selection_eligible_schemes"]
    assert result["applicability"]["interpretation"] == (
        "structural_proximity_only_not_biological_confidence"
    )

    for name, metadata in result["artifacts"].items():
        path = REPO / "results/m0_07" / name
        payload = path.read_bytes()
        assert len(payload) == metadata["bytes"]
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]

    with gzip.open(REPO / "results/m0_07/oracle_classical_metrics.csv.gz", "rt") as handle:
        metrics = list(csv.DictReader(handle))
    assert len(metrics) == 768
