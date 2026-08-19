from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from forge.potency.oracle_freeze import (
    OracleFreezeError,
    aggregate_candidates,
    build_applicability_policy,
    build_nested_selection_audit,
    freeze_oracle,
)

CONFIG_PATH = Path("configs/bio/m0_07_oracle_freeze.json")


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _candidate_rows(
    *,
    lane: str,
    representation: str,
    model: str,
    base_r2: float,
) -> list[dict]:
    rows = []
    config = _config()
    for endpoint_index, endpoint in enumerate(config["selection"]["endpoints"]):
        for scheme in config["selection"]["eligible_schemes"]:
            folds = [0] if scheme == "lantern_scaffold_balanced" else list(range(5))
            for fold in folds:
                rows.append(
                    {
                        "lane": lane,
                        "representation": representation,
                        "model": model,
                        "endpoint": endpoint,
                        "scheme": scheme,
                        "fold": fold,
                        "calibration_r2": base_r2 - 0.01 * endpoint_index,
                        "calibration_rmse": 2.0 - base_r2,
                        "calibration_spearman_rho": 0.4,
                        "test_r2": base_r2 - 0.01 * endpoint_index,
                        "test_rmse": 2.0 - base_r2,
                        "test_spearman_rho": 0.4,
                        "test_coverage90": 0.9,
                    }
                )
        rows.append(
            {
                "lane": lane,
                "representation": representation,
                "model": model,
                "endpoint": endpoint,
                "scheme": "lantern_random",
                "fold": 0,
                "calibration_r2": 0.99,
                "calibration_rmse": 0.1,
                "calibration_spearman_rho": 0.99,
                "test_r2": 0.99,
                "test_rmse": 0.1,
                "test_spearman_rho": 0.99,
                "test_coverage90": 0.9,
            }
        )
    return rows


def test_candidate_selection_uses_equal_endpoint_held_out_evidence() -> None:
    config = _config()
    rows = [
        *_candidate_rows(
            lane="classical",
            representation="morgan",
            model="rf",
            base_r2=0.2,
        ),
        *_candidate_rows(
            lane="graph",
            representation="role_dmpnn",
            model="ensemble",
            base_r2=0.3,
        ),
    ]
    candidates, scheme_metrics = aggregate_candidates(rows, config)
    assert candidates[0]["candidate_id"] == "graph::role_dmpnn::ensemble"
    assert candidates[0]["selection_rank"] == 1
    assert len(candidates) == 2
    assert "lantern_random" not in scheme_metrics[candidates[0]["candidate_id"]]["expt_Hela"]


def test_outer_test_metrics_cannot_select_the_oracle() -> None:
    config = _config()
    calibration_leader = _candidate_rows(
        lane="graph",
        representation="calibration_leader",
        model="ensemble",
        base_r2=0.3,
    )
    test_only_leader = _candidate_rows(
        lane="graph",
        representation="test_only_leader",
        model="ensemble",
        base_r2=0.2,
    )
    for row in calibration_leader:
        row["test_r2"] = -1.0
        row["test_rmse"] = 10.0
    for row in test_only_leader:
        row["test_r2"] = 0.99
        row["test_rmse"] = 0.1
    candidates, _ = aggregate_candidates(
        [*calibration_leader, *test_only_leader],
        config,
    )
    assert candidates[0]["representation"] == "calibration_leader"
    nested = build_nested_selection_audit(
        [*calibration_leader, *test_only_leader],
        config,
    )
    assert {row["selected_candidate_id"] for row in nested["fold_selections"]} == {
        "graph::calibration_leader::ensemble"
    }
    assert all(row["outer_test_used_for_selection"] is False for row in nested["fold_selections"])


def test_missing_selection_scheme_fails_closed() -> None:
    config = _config()
    rows = _candidate_rows(
        lane="graph",
        representation="role_dmpnn",
        model="ensemble",
        base_r2=0.3,
    )
    rows = [
        row
        for row in rows
        if not (row["endpoint"] == "expt_Hela" and row["scheme"] == "held_aldehyde_5fold")
    ]
    with pytest.raises(OracleFreezeError, match="has schemes"):
        aggregate_candidates(rows, config)


def test_missing_selection_fold_fails_closed() -> None:
    config = _config()
    rows = _candidate_rows(
        lane="graph",
        representation="role_dmpnn",
        model="ensemble",
        base_r2=0.3,
    )
    rows = [
        row
        for row in rows
        if not (
            row["endpoint"] == "expt_Hela"
            and row["scheme"] == "held_aldehyde_5fold"
            and row["fold"] == 4
        )
    ]
    with pytest.raises(OracleFreezeError, match="has folds"):
        aggregate_candidates(rows, config)


def test_unexpected_selection_fold_fails_closed() -> None:
    config = _config()
    rows = _candidate_rows(
        lane="graph",
        representation="role_dmpnn",
        model="ensemble",
        base_r2=0.3,
    )
    source = next(
        row
        for row in rows
        if row["endpoint"] == "expt_Hela"
        and row["scheme"] == "held_isocyanide_5fold"
        and row["fold"] == 4
    )
    rows.append({**source, "fold": 5})
    with pytest.raises(OracleFreezeError, match="has folds"):
        aggregate_candidates(rows, config)


def test_applicability_policy_abstains_only_where_required_gate_fails() -> None:
    config = _config()
    candidate = "graph::role_dmpnn::ensemble"
    metrics = {
        candidate: {
            endpoint: {
                scheme: {
                    "folds": 5,
                    "mean_test_r2": (0.05 if scheme == "held_aldehyde_5fold" else 0.3),
                    "mean_test_rmse": 1.0,
                    "mean_test_spearman_rho": 0.4,
                    "mean_absolute_90pct_coverage_gap": 0.02,
                    "positive_r2_fold_fraction": 0.8,
                }
                for scheme in config["selection"]["eligible_schemes"]
            }
            for endpoint in config["selection"]["endpoints"]
        }
    }
    policy = build_applicability_policy(
        candidate,
        metrics,
        metrics[candidate],
        config,
    )
    for endpoint in policy["endpoints"].values():
        assert endpoint["domains"]["unseen_aldehyde"]["action"] == "abstain"
        assert (
            endpoint["domains"]["unseen_isocyanide"]["action"]
            == "conformal_lower_confidence_guidance"
        )
        assert endpoint["domains"]["unseen_aldehyde_and_isocyanide"]["action"] == "abstain"
    assert policy["unsupported_domains"]["three_unseen_components"]["action"] == "abstain"


def _write_metric_file(path: Path, *, r2: float) -> str:
    fieldnames = [
        "representation",
        "model",
        "endpoint",
        "scheme",
        "fold",
        "calibration_r2",
        "calibration_rmse",
        "calibration_spearman_rho",
        "test_r2",
        "test_rmse",
        "test_spearman_rho",
        "test_coverage90",
    ]
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in _candidate_rows(
            lane="ignored",
            representation="representation",
            model="model",
            base_r2=r2,
        ):
            source = {key: row[key] for key in fieldnames}
            writer.writerow(source)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_end_to_end_freeze_pins_inputs_and_rejects_tampered_metrics(
    tmp_path: Path,
) -> None:
    config = _config()
    for lane_index, (lane, specification) in enumerate(config["inputs"].items()):
        result_path = tmp_path / f"{lane}_result.json"
        metrics_path = tmp_path / f"{lane}_metrics.csv.gz"
        metrics_hash = _write_metric_file(metrics_path, r2=0.2 + lane_index * 0.05)
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": specification["result_schema"],
                    "status": specification["result_status"],
                    "artifacts": {"metrics": {"sha256": metrics_hash}},
                }
            )
        )
        specification["expected_candidates"] = [["representation", "model"]]
        specification["result_path"] = result_path.name
        specification["metrics_path"] = metrics_path.name
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output_path = tmp_path / "freeze.json"
    result = freeze_oracle(config_path, output_path, tmp_path)
    assert result["decision"]["one_representation_and_model_frozen_across_endpoints"]
    assert not result["decision"]["production_checkpoint_frozen"]
    assert result["selected_model"]["lane"] == "label_free_r0_transfer"
    assert output_path.is_file()

    first_lane = next(iter(config["inputs"]))
    metrics_path = tmp_path / config["inputs"][first_lane]["metrics_path"]
    with metrics_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(OracleFreezeError, match="not pinned"):
        freeze_oracle(config_path, output_path, tmp_path)
