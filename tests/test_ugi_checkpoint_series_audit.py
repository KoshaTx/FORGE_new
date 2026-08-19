from __future__ import annotations

import json
from pathlib import Path

from forge.product.defog_feasibility import sha256_file
from forge.product.ugi_checkpoint_series_audit import audit_ugi_checkpoint_series


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_checkpoint_series_audit_requires_and_records_frozen_steps(tmp_path: Path) -> None:
    probe = tmp_path / "probe.json"
    _write(
        probe,
        {"schema_version": "phase1_ugi_program_probe.v1", "samples": [{"program": {}}]},
    )
    checkpoint = tmp_path / "checkpoint_step_10.pt"
    checkpoint.write_bytes(b"checkpoint")
    training = tmp_path / "training.json"
    _write(
        training,
        {
            "status": "complete",
            "selection": {"best_step": 10},
            "checkpoint_snapshots": [{"step": 10, "sha256": sha256_file(checkpoint)}],
        },
    )
    descriptors = {
        key: {"median": value}
        for key, value in {
            "heavy_atoms": 40,
            "heteroatoms": 6,
            "logp": 8,
            "molecular_weight": 550,
            "rings": 1,
            "rotatable_bonds": 27,
        }.items()
    }
    sample = tmp_path / "sample.json"
    _write(
        sample,
        {
            "status": "complete",
            "checkpoints": {"joint": str(checkpoint)},
            "matched_staged_result": str(probe),
            "statistics": {
                "valid_molecules": 4,
                "valid_fraction": 1.0,
                "unique_valid_molecules": 3,
                "aromatic_valid_fraction": 0.25,
                "mean_decorations": 0.5,
                "failure_types": {},
            },
            "reference_comparison": {
                "all_frozen_ugi": {
                    "reference_molecules": 5,
                    "reference_descriptors": descriptors,
                    "exact_matches": 1,
                    "nearest_morgan_tanimoto": {"median": 0.6},
                },
                "generated_descriptors": {
                    "heteroatoms": {"median": 5},
                    "logp": {"median": 9},
                    "rings": {"median": 1},
                    "rotatable_bonds": {"median": 28},
                },
            },
        },
    )
    output = tmp_path / "audit.json"
    result = audit_ugi_checkpoint_series(
        tmp_path,
        training_result_path=training,
        program_probe_path=probe,
        sample_result_paths=[sample],
        output_path=output,
    )
    assert result["checkpoints"][0]["step"] == 10
    assert result["checkpoints"][0]["unique_fraction_among_valid"] == 0.75
    assert result["decision"]["production_checkpoint_frozen"] is False
    assert result["inputs"]["training_result"]["sha256"] == sha256_file(training)
    assert json.loads(output.read_text()) == result
