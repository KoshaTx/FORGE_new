from __future__ import annotations

import gzip
import json

import pytest

from experiments.phase1.multireaction.local_chemistry_resampling_adjudication import (
    FINAL_ARM,
    SAMPLES_SCHEMA,
    SynthesisProgramProductionAdjudicationError,
    _liability_audit,
)


def _write_samples(path, smiles: str) -> None:
    rows = [
        {
            "arm_id": FINAL_ARM,
            "canonical_smiles": smiles,
            "checkpoint_step": 1700,
            "evaluation_split": "heldout",
            "exact_l1_program": True,
            "local_chemistry_policy_applied": True,
            "program_id": program,
            "valid": True,
        }
        for program in (
            "bl_2023_repeated_aza_michael",
            "lx_2024_repeated_reductive_amination",
            "ugi_3cr_agile",
        )
    ]
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps({"schema_version": SAMPLES_SCHEMA, "rows": len(rows)}) + "\n")
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_liability_audit_counts_attempts_and_clean_structures(tmp_path) -> None:
    path = tmp_path / "samples.jsonl.gz"
    _write_samples(path, "CCCCNCC")

    audit = _liability_audit(path, final_step=1700, attempts_per_program=1)

    for summary in audit.values():
        assert summary["attempts"] == 1
        assert summary["valid"] == 1
        assert summary["exact_l1"] == 1
        assert summary["policy_applied"] == 1
        assert summary["molecules_with_declared_liability"] == 0
        assert summary["declared_liability_fraction_among_valid"] == 0.0


@pytest.mark.parametrize(
    ("smiles", "metric"),
    (
        ("COO", "oxygen_oxygen_bonds"),
        ("CNO", "nitrogen_oxygen_bonds"),
        ("C1CC1", "three_membered_rings"),
    ),
)
def test_liability_audit_detects_declared_local_liability(
    tmp_path, smiles: str, metric: str
) -> None:
    path = tmp_path / "samples.jsonl.gz"
    _write_samples(path, smiles)

    audit = _liability_audit(path, final_step=1700, attempts_per_program=1)

    assert all(summary[metric] > 0 for summary in audit.values())
    assert all(summary["molecules_with_declared_liability"] == 1 for summary in audit.values())


def test_liability_audit_requires_policy_on_every_attempt(tmp_path) -> None:
    path = tmp_path / "samples.jsonl.gz"
    _write_samples(path, "CC")
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    rows[1]["local_chemistry_policy_applied"] = False
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")

    with pytest.raises(SynthesisProgramProductionAdjudicationError, match="not applied"):
        _liability_audit(path, final_step=1700, attempts_per_program=1)
