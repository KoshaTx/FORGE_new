from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from experiments.phase1.multireaction.final_production_adjudication import (
    SAMPLES_SCHEMA,
    _ledger_counts,
    seed_summary,
)
from experiments.phase1.multireaction.production_adjudication import (
    SynthesisProgramProductionAdjudicationError,
)


def test_seed_summary_reports_sample_standard_deviation_without_imputation() -> None:
    summary = seed_summary([0.7, 0.8, 0.9])

    assert summary["by_seed"] == [0.7, 0.8, 0.9]
    assert np.isclose(summary["mean"], 0.8)
    assert np.isclose(summary["sample_standard_deviation"], 0.1)
    assert summary["all_seed_cells_defined"] is True
    assert summary["undefined_cells_imputed"] is False

    incomplete = seed_summary([0.7, None, 0.9])
    assert incomplete["mean"] is None
    assert incomplete["sample_standard_deviation"] is None
    assert incomplete["defined_seed_count"] == 2
    assert incomplete["undefined_cells_imputed"] is False


def test_ledger_counts_uses_attempts_as_the_yield_denominator(tmp_path) -> None:
    path = tmp_path / "samples.jsonl.gz"
    rows = [
        {
            "arm_id": "final",
            "program_id": "ugi",
            "evaluation_split": "heldout",
            "checkpoint_step": 1700,
            "valid": True,
            "exact_l1_program": True,
            "forward_verified_trace_count": 1,
        },
        {
            "arm_id": "final",
            "program_id": "ugi",
            "evaluation_split": "heldout",
            "checkpoint_step": 1700,
            "valid": True,
            "exact_l1_program": False,
            "forward_verified_trace_count": 0,
        },
        {
            "arm_id": "final",
            "program_id": "ugi",
            "evaluation_split": "heldout",
            "checkpoint_step": 1700,
            "valid": False,
            "exact_l1_program": False,
            "forward_verified_trace_count": 0,
        },
        {
            "arm_id": "different-arm",
            "program_id": "ugi",
            "evaluation_split": "heldout",
            "checkpoint_step": 1700,
            "valid": True,
            "exact_l1_program": True,
            "forward_verified_trace_count": 1,
        },
    ]
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps({"schema_version": SAMPLES_SCHEMA, "rows": len(rows)}) + "\n")
        for row in rows:
            stream.write(json.dumps(row) + "\n")

    summary = _ledger_counts(path, arm_id="final", programs=("ugi",), final_step=1700)["ugi"]

    assert summary["attempts"] == 3
    assert summary["valid"] == 2
    assert summary["exact_l1"] == 1
    assert summary["raw_valid_fraction"] == 2 / 3
    assert summary["exact_l1_yield_per_attempt"] == 1 / 3
    assert summary["exact_l1_coverage_among_valid"] == 1 / 2
    assert summary["forward_replay_precision_among_exact_l1"] == 1.0


def test_ledger_counts_rejects_an_unsupported_header(tmp_path) -> None:
    path = tmp_path / "samples.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps({"schema_version": "wrong", "rows": 0}) + "\n")

    with pytest.raises(SynthesisProgramProductionAdjudicationError, match="supported schema"):
        _ledger_counts(path, arm_id="final", programs=("ugi",), final_step=1700)
