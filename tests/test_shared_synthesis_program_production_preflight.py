from __future__ import annotations

import copy
import json
import tarfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments._runtime.modal import modal_request_plan
from experiments.phase1.multireaction.accelerator_selection import (
    AcceleratorSelectionError,
    _selection_rows,
    adjudicate_accelerator_benchmarks,
)
from experiments.phase1.multireaction.production_evaluation import (
    SynthesisProgramProductionEvaluationError,
    _validate_archive_members,
)
from experiments.phase1.multireaction.production_preflight import (
    SynthesisProgramProductionPreflightError,
    _accelerator_policy,
    _runtime_matches_accelerator,
    _validate_factorized_layout_schedule,
    _validate_molecule_rendering,
    _validate_static_contract,
)
from experiments.phase1.multireaction.production_randomness import production_seed
from forge.core.hashing import sha256_file

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/shared_production_accelerator_benchmark_v5.json"
TRANSFORMER_CONFIG = (
    REPO / "configs/multireaction/transformer_production_accelerator_benchmark_v2.json"
)
TRANSFORMER_PRODUCTION_SPECS = {
    "A100-40GB": REPO / "experiments/phase1/multireaction/transformer_production.json",
    "H100!": REPO / "experiments/phase1/multireaction/transformer_production_h100.json",
}
SPECS = {
    "L4": (
        REPO / "experiments/phase1/multireaction/shared_production_accelerator_benchmark_l4.json"
    ),
    "A100-40GB": (
        REPO / "experiments/phase1/multireaction/"
        "transformer_production_accelerator_benchmark_a100_40gb.json"
    ),
    "H100!": (
        REPO / "experiments/phase1/multireaction/"
        "transformer_production_accelerator_benchmark_h100.json"
    ),
}


def _documents() -> tuple[dict[str, object], ...]:
    config = json.loads(CONFIG.read_text())
    paths = {label: REPO / value["path"] for label, value in config["inputs"].items()}
    return (
        config,
        json.loads(paths["production_design"].read_text()),
        json.loads(paths["production_training_config"].read_text()),
        json.loads(paths["production_evaluation_config"].read_text()),
        json.loads(paths["smoke_training_result"].read_text()),
        json.loads(paths["smoke_evaluation_result"].read_text()),
        json.loads(paths["test_baseline_report"].read_text()),
    )


def _documents_from(path: Path) -> tuple[dict[str, object], ...]:
    config = json.loads(path.read_text())
    paths = {label: REPO / value["path"] for label, value in config["inputs"].items()}
    return (
        config,
        json.loads(paths["production_design"].read_text()),
        json.loads(paths["production_training_config"].read_text()),
        json.loads(paths["production_evaluation_config"].read_text()),
        json.loads(paths["smoke_training_result"].read_text()),
        json.loads(paths["smoke_evaluation_result"].read_text()),
        json.loads(paths["test_baseline_report"].read_text()),
    )


def test_preflight_static_contract_closes_every_local_gate() -> None:
    gates = _validate_static_contract(*_documents())
    assert all(gates.values())
    assert _validate_molecule_rendering() is True


def test_transformer_preflight_is_bound_to_current_smoke_and_regression_receipts() -> None:
    documents = _documents_from(TRANSFORMER_CONFIG)
    gates = _validate_static_contract(*documents)
    assert all(gates.values())
    assert documents[1]["model"]["architecture"] == "reaction_program_graph_transformer"
    assert documents[2]["smoke"]["micro_batch_size"] == 3


def test_transformer_production_alternatives_change_only_execution_identity() -> None:
    specifications = {
        gpu_type: json.loads(path.read_text())
        for gpu_type, path in TRANSFORMER_PRODUCTION_SPECS.items()
    }
    a100 = specifications["A100-40GB"]
    h100 = specifications["H100!"]
    assert [stage["resources"]["gpu_type"] for stage in a100["stages"]] == [
        "A100-40GB",
        "A100-40GB",
    ]
    assert [stage["resources"]["gpu_type"] for stage in h100["stages"]] == [
        "H100!",
        "H100!",
    ]
    for document in specifications.values():
        document.pop("experiment_id")
        document.pop("description")
        document.pop("metadata")
        for stage in document["stages"]:
            stage["resources"].pop("gpu_type")
    assert a100 == h100


def test_preflight_exercises_every_frozen_layout_cell_and_preserves_seed_contract() -> None:
    class RecordingPrior:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, int]] = []

        def sample(self, program_id: str, *, sample_count: int, seed: int) -> tuple[int, ...]:
            self.calls.append((program_id, sample_count, seed))
            return tuple(range(sample_count))

    design = {
        "training": {
            "replicate_seeds": [20260825, 20260826],
            "arms": {
                "arm": {
                    "program_mass": {"program_a": 0.5, "program_b": 0.5},
                }
            },
        }
    }
    evaluation = {
        "full": {
            "checkpoint_steps": [100, 1700],
            "calibration_samples": 3,
            "heldout_samples": 5,
        }
    }
    prior = RecordingPrior()
    result = _validate_factorized_layout_schedule(prior, design, evaluation)  # type: ignore[arg-type]
    assert result == {"replicates": 2, "layout_cells": 12, "layouts": 44}
    assert len(prior.calls) == 12
    assert (
        production_seed(
            20260827,
            "ugi_only_conditioned",
            100,
            "calibration",
            "ugi_3cr_agile",
            "layout",
        )
        == 3561515994897245985
    )


def test_preflight_refuses_authorization_or_evaluation_budget_drift() -> None:
    documents = list(_documents())
    unauthorized = copy.deepcopy(documents[0])
    unauthorized["authorization"]["authorized"] = False
    with pytest.raises(SynthesisProgramProductionPreflightError, match="not authorized"):
        _validate_static_contract(unauthorized, *documents[1:])

    evaluation = copy.deepcopy(documents[3])
    evaluation["full"]["heldout_samples"] -= 1
    gates = _validate_static_contract(*documents[:3], evaluation, *documents[4:])
    assert gates["full_evaluation_matches_frozen_budget"] is False


@pytest.mark.parametrize("gpu_type", ["L4", "A100-40GB", "H100!"])
def test_modal_benchmark_plan_requests_one_exact_accelerator(gpu_type: str) -> None:
    plan = modal_request_plan(REPO, SPECS[gpu_type], profile="smoke", replicate=0, device=None)
    assert plan["resource_envelope"]["gpu_type"] == gpu_type
    assert plan["resource_envelope"]["memory_mb"] == 32768
    assert plan["resource_envelope"]["timeout_seconds"] == 3600


def test_accelerator_policy_rejects_unfrozen_target_and_wrong_physical_memory() -> None:
    config = json.loads(CONFIG.read_text())
    with pytest.raises(SynthesisProgramProductionPreflightError, match="not authorized"):
        _accelerator_policy(config, "H100!")

    l4 = _accelerator_policy(config, "L4")
    assert _runtime_matches_accelerator(
        SimpleNamespace(name="NVIDIA L4", total_memory=24 * 1024**3), l4
    )
    assert not _runtime_matches_accelerator(
        SimpleNamespace(name="NVIDIA A100-SXM4-80GB", total_memory=80 * 1024**3), l4
    )

    a100 = _accelerator_policy(config, "A100-40GB")
    assert _runtime_matches_accelerator(
        SimpleNamespace(name="NVIDIA A100-SXM4-40GB", total_memory=40 * 1024**3), a100
    )
    assert not _runtime_matches_accelerator(
        SimpleNamespace(name="NVIDIA A100-SXM4-80GB", total_memory=80 * 1024**3), a100
    )

    transformer_config = json.loads(TRANSFORMER_CONFIG.read_text())
    h100 = _accelerator_policy(transformer_config, "H100!")
    assert _runtime_matches_accelerator(
        SimpleNamespace(name="NVIDIA H100 80GB HBM3", total_memory=80 * 1024**3), h100
    )
    assert not _runtime_matches_accelerator(
        SimpleNamespace(name="NVIDIA H200", total_memory=141 * 1024**3), h100
    )


def test_benchmark_projection_is_bound_to_frozen_training_budget() -> None:
    config = json.loads(CONFIG.read_text())
    design_path = REPO / config["inputs"]["production_design"]["path"]
    design = json.loads(design_path.read_text())
    policy = _accelerator_policy(config, "L4")
    assert policy["production_optimizer_steps_per_arm"] == design["training"]["optimizer_steps"]
    assert policy["production_replicates"] == len(design["training"]["replicate_seeds"])
    assert policy["warmup_optimizer_steps"] == 1
    assert policy["measured_optimizer_steps"] == 3


def _benchmark_result(gpu_type: str, *, hours: float, cost: float) -> dict[str, object]:
    return {
        "schema_version": "forge.synthesis_program_production_accelerator_benchmark_result.v1",
        "status": "pass",
        "gates": {"deterministic": True, "memory": True},
        "device": {
            "declared_gpu_type": gpu_type,
            "name": f"NVIDIA {gpu_type}",
            "total_memory_bytes": 24 * 1024**3,
        },
        "peak_reserved_bytes": 1024,
        "projection": {
            "scope": "training_only_excludes_evaluation_and_startup",
            "hours_per_replicate": hours,
            "gpu_cost_usd_all_replicates": cost,
        },
    }


def test_accelerator_selection_uses_cost_then_wall_time() -> None:
    ranking = _selection_rows(
        [
            _benchmark_result("L4", hours=10.0, cost=12.0),
            _benchmark_result("A100-40GB", hours=4.0, cost=12.0),
        ]
    )
    assert [row["gpu_type"] for row in ranking] == ["A100-40GB", "L4"]

    failed = _benchmark_result("L4", hours=10.0, cost=8.0)
    failed["gates"] = {"memory": False}
    with pytest.raises(AcceleratorSelectionError, match="did not pass"):
        _selection_rows([failed])


def test_accelerator_selection_can_prioritize_wall_time() -> None:
    ranking = _selection_rows(
        [
            _benchmark_result("A100-40GB", hours=4.0, cost=8.0),
            _benchmark_result("H100!", hours=2.0, cost=12.0),
        ],
        primary_metric="projected_training_hours_per_replicate",
        tie_breaker_metric="projected_training_gpu_cost_usd_all_replicates",
    )
    assert [row["gpu_type"] for row in ranking] == ["H100!", "A100-40GB"]


def test_accelerator_adjudicator_pins_verified_run_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.phase1.multireaction.accelerator_selection as selection

    config_path = tmp_path / "configs" / "benchmark.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "forge.synthesis_program_production_accelerator_benchmark_config.v1"
                ),
                "benchmark": {
                    "adjudication": {
                        "required_targets": ["L4", "A100-40GB"],
                        "primary_order": "lowest projected cost",
                        "tie_breaker": "lowest wall time",
                        "production_requirement": "preflight selected target",
                    }
                },
            }
        )
    )
    config_pin = {
        "path": "configs/benchmark.json",
        "sha256": str(sha256_file(config_path)),
        "bytes": config_path.stat().st_size,
    }
    run_dirs = []
    for gpu_type, hours, cost in (
        ("L4", 10.0, 20.0),
        ("A100-40GB", 4.0, 18.0),
    ):
        run_id = gpu_type.lower().replace("-", "")
        run_dir = tmp_path / "runs" / gpu_type / run_id
        stage_dir = run_dir / "stages" / "benchmark"
        artifact_dir = stage_dir / "artifacts"
        artifact_dir.mkdir(parents=True)
        result = _benchmark_result(gpu_type, hours=hours, cost=cost)
        result.update(
            {
                "config": config_pin,
                "runtime": {"precision": "float32"},
                "model": {"hidden_dim": 192},
                "math_mode": {"cuda_matmul_allow_tf32": False},
                "design_sha256": "d" * 64,
                "cache": {"sha256": "c" * 64},
            }
        )
        result_path = artifact_dir / "result.json"
        result_path.write_text(json.dumps(result))
        stage = {
            "implementation": (
                "model.shared-synthesis-program-production-accelerator-benchmark.v1"
            ),
            "config": config_pin,
            "external_inputs": {"cache": {"sha256": "c" * 64}},
            "fingerprint": gpu_type,
            "artifacts": {"result": {"path": "artifacts/result.json"}},
        }
        (stage_dir / "manifest.json").write_text(json.dumps(stage))
        run = {
            "run_id": run_id,
            "source_sha256": "s" * 64,
            "stages": {"benchmark": {"fingerprint": gpu_type}},
        }
        (run_dir / "run.json").write_text(json.dumps(run))
        run_dirs.append(run_dir)

    monkeypatch.setattr(selection, "verify_run_directory", lambda _: {"status": "verified"})
    output_path = tmp_path / "results" / "selection.json"
    selected = adjudicate_accelerator_benchmarks(run_dirs, tmp_path, output_path)

    assert selected["selected_gpu_type"] == "A100-40GB"
    assert output_path.is_file()
    assert [row["gpu_type"] for row in selected["evidence"]] == ["A100-40GB", "L4"]
    assert all(row["result"]["sha256"] for row in selected["evidence"])
    assert all(
        row["result"]["path"].startswith("results/evidence/") for row in selected["evidence"]
    )
    assert (output_path.parent / "evidence" / "l4" / "result.json").read_bytes() == (
        run_dirs[0] / "stages" / "benchmark" / "artifacts" / "result.json"
    ).read_bytes()
    assert (output_path.parent / "evidence" / "a100_40gb" / "stage_manifest.json").read_bytes() == (
        run_dirs[1] / "stages" / "benchmark" / "manifest.json"
    ).read_bytes()


def test_checkpoint_archive_rejects_any_unexpected_member(tmp_path: Path) -> None:
    training = {
        "arms": {
            "arm": {
                "checkpoints": [{"filename": "checkpoint.pt"}],
            }
        }
    }
    path = tmp_path / "checkpoints.tar"
    with tarfile.open(path, "w") as archive:
        for name in ("arm/checkpoint.pt", "unexpected.pt"):
            payload = b"checkpoint"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    with tarfile.open(path) as archive:
        with pytest.raises(SynthesisProgramProductionEvaluationError, match="members differ"):
            _validate_archive_members(archive, training)
