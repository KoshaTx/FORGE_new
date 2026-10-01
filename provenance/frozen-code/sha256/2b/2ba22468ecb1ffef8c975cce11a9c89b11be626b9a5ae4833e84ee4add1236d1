from __future__ import annotations

import json
from pathlib import Path

import pytest
from forge_paper import (
    ExperimentMatrix,
    ExperimentMatrixError,
    diagnose_experiment_matrix,
)

from cli import build_parser

REPO = Path(__file__).resolve().parents[1]
MATRIX = REPO / "configs/reproduction/natbiotech_v1_experiments.json"
PROTOCOL = REPO / "configs/multireaction/common_ugi_baseline_protocol_v1.json"


def test_v1_matrix_covers_every_retained_baseline_and_ablation_family() -> None:
    matrix = ExperimentMatrix.load(MATRIX)
    ids = {entry.entry_id for entry in matrix.entries}
    assert matrix.paper_id == "forge-natbiotech-v1"
    assert {
        "transformer_four_arm_production",
        "finite_catalogue_oracle",
        "production_adjudication",
        "common_ugi_assessor",
        "conditional_cross_role_dependence",
        "fact_matched_and_generous",
        "learned_inventory_selector",
        "transformer_mechanism_ablations",
        "rgfn",
        "defog_unconditional",
        "genmol_safe",
        "common_route_evidence_assessment",
        "paper_v1_result_renderer",
    } <= ids
    assert not any("mpnn" in entry.entry_id.lower() for entry in matrix.entries)
    by_id = {entry.entry_id: entry for entry in matrix.entries}
    assert by_id["ou_dag_chem"].requirement == "context"
    assert by_id["synflownet"].requirement == "context"
    assert by_id["syncogen"].requirement == "context"


def test_common_ugi_protocol_uses_one_attempt_denominator_and_three_seeds() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["randomness"]["training_and_sampling_seeds"] == [
        20260825,
        20260826,
        20260827,
    ]
    assert protocol["sampling"]["attempts_per_seed"] == 3072
    assert protocol["sampling"]["attempt_denominator_includes_invalid_or_failed_generation"]
    assert protocol["sampling"]["repair_or_retry_after_failed_attempt"] is False
    assert protocol["evaluation"]["coverage_and_precision_reported_separately"] is True
    assert protocol["scope"]["training_folds"] == ["train"]


def test_diagnosis_reports_source_freeze_transition_without_claiming_results() -> None:
    diagnosis = diagnose_experiment_matrix(REPO, MATRIX)
    rows = {row["id"]: row for row in diagnosis["entries"]}
    source_matches = (
        diagnosis["source_freeze"]["current_source_sha256"]
        == diagnosis["source_freeze"]["qualified_source_sha256"]
    )
    assert diagnosis["source_freeze"]["intact"] is source_matches
    assert diagnosis["immediate_production_ready"] is source_matches
    assert diagnosis["setup_complete_for_all_retained_rows"] is True
    assert diagnosis["paper_results_complete"] is False
    assert rows["transformer_four_arm_production"]["observed_state"] == "launch_ready"
    assert rows["finite_catalogue_oracle"]["observed_state"] == "launch_ready"
    assert rows["production_adjudication"]["observed_state"] == "waiting_for_upstream"
    assert rows["transformer_mechanism_ablations"]["observed_state"] == (
        "completed" if source_matches else "launch_ready"
    )
    assert rows["held_reaction_family"]["requirement"] == "optional"


def test_matrix_rejects_unknown_dependencies(tmp_path: Path) -> None:
    document = json.loads(MATRIX.read_text())
    document["entries"][0]["dependencies"] = ["not_a_real_entry"]
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ExperimentMatrixError, match="invalid dependencies"):
        ExperimentMatrix.load(path)


def test_cli_exposes_v1_experiment_readiness() -> None:
    args = build_parser().parse_args(["paper", "experiments", "--strict"])
    assert args.function is not None
    assert args.strict is True
