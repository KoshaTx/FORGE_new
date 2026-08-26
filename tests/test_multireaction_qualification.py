from __future__ import annotations

import json
from pathlib import Path

from experiments.phase1.multireaction.qualification import qualify_overfit_run


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value))
    return path


def test_overfit_qualification_preserves_a_negative_exact_l1_gate(tmp_path: Path) -> None:
    config = _write(
        tmp_path / "config.json",
        {
            "schema_version": "forge.multireaction_overfit_qualification_config.v1",
            "inputs": {},
            "gates": {
                "maximum_final_to_initial_loss_ratio": 0.5,
                "minimum_exact_tensor_fraction": 0.5,
                "minimum_valid_fraction": 0.75,
                "minimum_exact_l1_fraction": 0.0625,
            },
        },
    )
    training = _write(
        tmp_path / "training.json",
        {
            "schema_version": "forge.multireaction_training_result.v2",
            "run_kind": "overfit_gate",
            "arms": {
                "program": {
                    "initial_total_loss": 10.0,
                    "final_total_loss": 4.0,
                    "all_losses_finite": True,
                    "evaluation_by_program": {
                        "aza": {
                            "records": 1,
                            "fixed_noise_reconstruction": {"exact_tensor_records": 1},
                        },
                        "reductive": {
                            "records": 1,
                            "fixed_noise_reconstruction": {"exact_tensor_records": 1},
                        },
                    },
                }
            },
        },
    )
    sampling = _write(
        tmp_path / "sampling.json",
        {
            "schema_version": "forge.multireaction_sampling_result.v2",
            "run_kind": "overfit_gate",
            "metrics": {"samples": 16, "valid": 16, "exact_l1_program": 0},
        },
    )

    result = qualify_overfit_run(
        config,
        tmp_path,
        training,
        sampling,
        tmp_path / "result.json",
    )

    assert result["status"] == "overfit_gate_failed"
    assert result["gates"]["exact_l1_generation"] is False
    assert result["gates"]["fixed_noise_tensor_reconstruction"] is True
    assert result["inputs"]["training_result"]["sha256"]
