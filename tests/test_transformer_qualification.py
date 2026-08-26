from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.phase1.multireaction import transformer_qualification as qualification
from forge.core.hashing import sha256_file


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _run(tmp_path: Path, *, method: str) -> Path:
    run_dir = tmp_path / "runs" / "qualified"
    checkpoint = run_dir / "stages/training/artifacts/checkpoint.json"
    result = run_dir / "stages/training/artifacts/result.json"
    _write(checkpoint, {"schema_version": "checkpoint"})
    _write(
        result,
        {
            "status": "pass",
            "gates": {"bounded_gate": True},
            "training": {
                "architecture": "reaction_program_graph_transformer",
                "semantic_objective": {
                    "role_consistency_weight": 0.25,
                    "core_consistency_weight": 0.25,
                },
                "gradient_balancing": {"method": method},
                "initial_total_loss": 2.0,
                "final_total_loss": 0.5,
                "final_to_initial_loss_ratio": 0.25,
            },
            "validation": {
                "ugi": {"reconstruction": {"exact_tensor_fraction": 1.0}},
                "bl": {"reconstruction": {"exact_tensor_fraction": 1.0}},
                "lx": {"reconstruction": {"exact_tensor_fraction": 1.0}},
            },
        },
    )
    _write(
        run_dir / "stages/training/manifest.json",
        {
            "artifacts": {
                "checkpoint": {
                    "path": "artifacts/checkpoint.json",
                    "sha256": str(sha256_file(checkpoint)),
                },
                "result": {
                    "path": "artifacts/result.json",
                    "sha256": str(sha256_file(result)),
                },
            }
        },
    )
    _write(
        run_dir / "run.json",
        {
            "experiment_id": "phase1-reaction-program-transformer-overfit",
            "profile": "smoke",
            "status": "complete",
            "run_id": "qualified",
            "source_sha256": "source",
            "stages": {"cache": {}, "training": {}},
        },
    )
    return run_dir


def test_freezer_preserves_bounded_claim_and_pins_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qualification, "verify_run_directory", lambda _path: None)
    run_dir = _run(tmp_path, method="equal_family_mass_deterministic_pcgrad")
    output = tmp_path / "results/qualification.json"
    frozen = qualification.freeze_transformer_qualification(run_dir, tmp_path, output)
    assert frozen["status"] == "qualified_for_frozen_followup_preflight"
    assert frozen["candidate_selection"] is False
    assert frozen["calls"] == {"route": 0, "oracle": 0}
    assert frozen["exact_tensor_reconstruction_by_program"] == {
        "ugi": 1.0,
        "bl": 1.0,
        "lx": 1.0,
    }
    assert output.is_file()


def test_freezer_preserves_failed_gate_as_negative_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qualification, "verify_run_directory", lambda _path: None)
    run_dir = _run(tmp_path, method="equal_family_mass_deterministic_pcgrad")
    result_path = run_dir / "stages/training/artifacts/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["status"] = "fail"
    result["gates"]["bounded_gate"] = False
    _write(result_path, result)
    manifest_path = run_dir / "stages/training/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["result"]["sha256"] = str(sha256_file(result_path))
    _write(manifest_path, manifest)
    frozen = qualification.freeze_transformer_qualification(
        run_dir, tmp_path, tmp_path / "result.json"
    )
    assert frozen["status"] == "not_qualified_for_followup_preflight"
    assert "cannot advance" in frozen["decision"]


def test_freezer_rejects_wrong_balancing_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qualification, "verify_run_directory", lambda _path: None)
    run_dir = _run(tmp_path, method="none")
    with pytest.raises(
        qualification.TransformerQualificationError,
        match="does not implement the Transformer contract",
    ):
        qualification.freeze_transformer_qualification(
            run_dir, tmp_path, tmp_path / "result.json"
        )
