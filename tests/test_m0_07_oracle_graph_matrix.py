from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from forge.potency.oracle_graph import (
    GraphFeatureVocabulary,
    load_oracle_graph_records,
)
from forge.potency.oracle_graph_jobs import build_graph_job_rows
from forge.potency.oracle_graph_matrix import (
    _inner_split,
    fit_graph_job,
)

REPO = Path(__file__).resolve().parents[1]


@contextmanager
def _single_threaded_torch() -> Iterator[None]:
    """Make the poison comparison bitwise, independent of CPU reduction order."""

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _assignments() -> list[dict[str, str]]:
    with gzip.open(
        REPO / "results/m0_07/oracle_split_assignments.csv.gz",
        "rt",
        newline="",
    ) as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _fixture() -> tuple[
    dict[str, object],
    list[object],
    dict[str, int],
    list[dict[str, str]],
    GraphFeatureVocabulary,
    dict[str, object],
]:
    config = json.loads((REPO / "configs/bio/m0_07_oracle_graph_matrix.json").read_text())
    config["epoch_selection"] = {
        **config["epoch_selection"],
        "maximum_epochs": 2,
        "patience": 1,
    }
    vocabulary = GraphFeatureVocabulary.from_corpus_result(
        REPO / "results/m0_07/oracle_graph_corpus_result.json"
    )
    records = load_oracle_graph_records(
        REPO / "results/m0_07/agile_oracle_curated.csv.gz",
        vocabulary,
    )
    label_to_index = {record.label: index for index, record in enumerate(records)}
    assignments = _assignments()
    rows = build_graph_job_rows(
        assignments,
        architectures=("whole_graph_dmpnn",),
        endpoints=("expt_Hela",),
        seeds=(1729,),
        selection_schemes={"held_aldehyde_5fold"},
        fit_output_root="results/m0_07/graph_fits",
    )
    row = next(
        candidate
        for candidate in rows
        if candidate["scheme"] == "held_aldehyde_5fold" and candidate["fold"] == 0
    )
    return config, records, label_to_index, assignments, vocabulary, row


def test_inner_validation_split_is_deterministic_and_disjoint() -> None:
    labels = [f"A{index}B1C1" for index in range(80)]
    first = _inner_split(
        labels,
        scheme="held_head_5fold",
        fold=2,
        fraction=0.125,
    )
    second = _inner_split(
        labels,
        scheme="held_head_5fold",
        fold=2,
        fraction=0.125,
    )

    assert first == second
    assert len(first[1]) == 10
    assert set(first[0]).isdisjoint(first[1])
    assert sorted((*first[0], *first[1])) == list(range(80))


def test_graph_fit_never_uses_calibration_or_test_targets_for_training() -> None:
    (
        config,
        records,
        label_to_index,
        assignments,
        vocabulary,
        row,
    ) = _fixture()
    arguments = {
        "label_to_index": label_to_index,
        "assignments": assignments,
        "vocabulary": vocabulary,
        "config": config,
        "config_sha256": "c" * 64,
        "input_hashes": {"fixture": "a" * 64},
    }
    held_out = {
        assignment["label"]
        for assignment in assignments
        if assignment["scheme"] == "held_aldehyde_5fold"
        and int(assignment["fold"]) == 0
        and assignment["stage"] in {"calibration", "test"}
    }
    poisoned = [
        replace(
            record,
            targets=(
                record.targets + torch.tensor([10_000.0, -10_000.0])
                if record.label in held_out
                else record.targets.clone()
            ),
        )
        for record in records
    ]
    with _single_threaded_torch():
        first = fit_graph_job(row, records=records, **arguments)
        second = fit_graph_job(row, records=poisoned, **arguments)

    assert first["epoch_selection"] == second["epoch_selection"]
    assert first["refit"] == second["refit"]
    assert first["boundary"] == {
        "calibration_or_test_used_for_scaling": False,
        "calibration_or_test_used_for_epoch_selection": False,
        "calibration_or_test_used_for_weight_updates": False,
        "calibration_and_test_accessed_after_checkpoint_hash": True,
    }
    assert len(first["calibration"]["prediction"]) == 100
    assert len(first["test"]["prediction"]) == 300
    assert first["calibration"]["prediction"] == pytest.approx(
        second["calibration"]["prediction"],
        abs=0.0,
    )
    assert first["test"]["prediction"] == pytest.approx(
        second["test"]["prediction"],
        abs=0.0,
    )
    assert first["calibration"]["truth"] != second["calibration"]["truth"]
    assert first["test"]["truth"] != second["test"]["truth"]
