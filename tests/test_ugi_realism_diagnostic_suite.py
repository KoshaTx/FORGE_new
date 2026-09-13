"""Provenance and restart invariants for the Ugi diagnostic workbench."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.phase1.multireaction import ugi_realism_diagnostic_suite as suite
from forge.core.hashing import sha256_file
from forge.core.io import read_json_object, write_json


def _pin(path: Path, repo: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(repo)), "sha256": str(sha256_file(path))}


def _stage_result(repo: Path, config: Path, name: str) -> dict:
    return {
        "status": suite.STAGE_STATUSES[name],
        "schema_version": f"forge.{suite.STAGES[name]}.v1",
        "config": _pin(config, repo),
        "inputs": read_json_object(config)["inputs"],
        "sources": {"implementation": _pin(repo / "forge" / "model" / "source.py", repo)},
        "policy": None,
        **(
            {"numerical_complete": True, "reviewer_status": "pending"}
            if name == "evaluator"
            else {}
        ),
    }


@pytest.fixture
def configured(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "forge" / "model").mkdir(parents=True)
    (repo / "forge" / "model" / "source.py").write_text("VALUE = 1\n")
    inputs = repo / "inputs.json"
    write_json(inputs, {"source": "train_only_fixture"})
    stages = {}
    for name in suite.STAGES:
        stage_config = repo / "configs" / f"{name}.json"
        write_json(stage_config, {"inputs": {"fixture": _pin(inputs, repo)}})
        result_path = repo / "collected" / name / "result.json"
        write_json(result_path, _stage_result(repo, stage_config, name))
        (result_path.parent / "packet.svg").write_text("<svg></svg>\n")
        stages[name] = {
            "config": _pin(stage_config, repo),
            "collected_result": _pin(result_path, repo),
        }
    config = repo / "suite.json"
    write_json(
        config,
        {
            "schema_version": suite.CONFIG_SCHEMA,
            "scientific_question": "Locate the first structural error.",
            "stages": stages,
            "policy": suite.POLICY,
        },
    )
    return repo, config, repo / "outputs"


def test_collection_is_deterministic_and_verification_is_read_only(configured: tuple) -> None:
    repo, config, output = configured
    first = suite.run_ugi_realism_diagnostic_suite(repo, config, output, collect=True)
    original = (output / "result.json").read_bytes()
    second = suite.run_ugi_realism_diagnostic_suite(repo, config, output, collect=True)
    assert first == second
    assert original == (output / "result.json").read_bytes()
    before = {path: path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}
    verified = suite.verify_ugi_realism_diagnostic_suite(repo, output / "result.json")
    assert verified["status"] == "verified"
    assert first["model_quality_improvement_established"] is False
    assert first["visual_reviewer_calibration"] == "packet_prepared_judgments_pending"
    assert before == {path: path.stat().st_mtime_ns for path in before}


def test_pending_calibration_cannot_be_changed_into_a_completed_review(configured: tuple) -> None:
    repo, config, output = configured
    result = suite.run_ugi_realism_diagnostic_suite(repo, config, output, collect=True)
    result["visual_reviewer_calibration"] = "calibrated"
    write_json(output / "result.json", result)
    with pytest.raises(suite.UgiRealismDiagnosticSuiteError, match="scope changed"):
        suite.verify_ugi_realism_diagnostic_suite(repo, output / "result.json")


def test_receipt_must_match_declared_collected_result(configured: tuple) -> None:
    repo, config, output = configured
    result = suite.run_ugi_realism_diagnostic_suite(repo, config, output, collect=True)
    replacement = repo / "replacement" / "result.json"
    write_json(replacement, _stage_result(repo, repo / "configs" / "support.json", "support"))
    spec = read_json_object(config)
    spec["stages"]["support"]["collected_result"] = _pin(replacement, repo)
    write_json(config, spec)
    result["config"] = _pin(config, repo)
    write_json(output / "result.json", result)
    with pytest.raises(suite.UgiRealismDiagnosticSuiteError, match="differs from declaration"):
        suite.verify_ugi_realism_diagnostic_suite(repo, output / "result.json")
    with pytest.raises(suite.UgiRealismDiagnosticSuiteError, match="differs from declaration"):
        suite.run_ugi_realism_diagnostic_suite(repo, config, output, collect=True)


@pytest.mark.parametrize("mutation", ["config", "inputs", "sources", "policy", "schema_version"])
def test_collected_execution_provenance_is_verified(configured: tuple, mutation: str) -> None:
    repo, config, output = configured
    spec = read_json_object(config)
    result_path = repo / spec["stages"]["support"]["collected_result"]["path"]
    payload = read_json_object(result_path)
    if mutation in {"config", "inputs", "sources"}:
        pin = payload[mutation] if mutation == "config" else next(iter(payload[mutation].values()))
        pin["sha256"] = "0" * 64
    else:
        payload[mutation] = "changed"
    write_json(result_path, payload)
    spec["stages"]["support"]["collected_result"] = _pin(result_path, repo)
    write_json(config, spec)
    with pytest.raises(ValueError):
        suite.run_ugi_realism_diagnostic_suite(repo, config, output, collect=True)
    assert not (output / "result.json").exists()


@pytest.mark.parametrize(
    "mutation", ["input", "config", "source", "packet", "receipt", "new_source", "new_artifact"]
)
def test_tampering_never_reuses_a_completed_stage(configured: tuple, mutation: str) -> None:
    repo, config, output = configured
    suite.run_ugi_realism_diagnostic_suite(repo, config, output, collect=True)
    targets = {
        "input": repo / "inputs.json",
        "config": repo / "configs" / "support.json",
        "source": repo / "forge" / "model" / "source.py",
        "packet": repo / "collected" / "evaluator" / "packet.svg",
        "receipt": output / "receipts" / "model.json",
        "new_source": repo / "forge" / "model" / "new.py",
        "new_artifact": repo / "collected" / "support" / "unexpected.txt",
    }
    targets[mutation].write_text("changed\n")
    with pytest.raises(ValueError):
        suite.verify_ugi_realism_diagnostic_suite(repo, output / "result.json")
    with pytest.raises(ValueError):
        suite.run_ugi_realism_diagnostic_suite(repo, config, output, collect=True)


def test_incomplete_stage_never_produces_complete_suite(configured: tuple) -> None:
    repo, config, output = configured
    spec = read_json_object(config)
    result_path = repo / spec["stages"]["model"]["collected_result"]["path"]
    write_json(result_path, {"status": "running", "schema_version": "test.model.v1"})
    spec["stages"]["model"]["collected_result"] = _pin(result_path, repo)
    write_json(config, spec)
    with pytest.raises(suite.UgiRealismDiagnosticSuiteError, match="not complete"):
        suite.run_ugi_realism_diagnostic_suite(repo, config, output, collect=True)
    assert not (output / "result.json").exists()
    assert (output / "receipts" / "support.json").exists()
    assert read_json_object(output / "progress.json")["status"] == "failed"
    assert read_json_object(output / "failures" / "model.json")["active_stage"] == "model"


def test_resume_preserves_partial_work_and_does_not_repeat_completed_stages(
    configured: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, config, output = configured
    calls: list[str] = []

    def runner(name: str):
        def run(_repo: Path, _config: Path, stage_output: Path) -> dict:
            calls.append(name)
            assert not stage_output.exists()
            stage_output.mkdir()
            if name == "model" and calls.count("model") == 1:
                (stage_output / "partial.txt").write_text("recoverable failure evidence")
                raise RuntimeError("simulated interrupted local analysis")
            result = _stage_result(_repo, _config, name)
            write_json(stage_output / "result.json", result)
            return result

        return run

    monkeypatch.setattr(suite, "_runner", runner)
    with pytest.raises(RuntimeError, match="simulated"):
        suite.run_ugi_realism_diagnostic_suite(repo, config, output)
    assert calls == ["support", "model"]
    assert not (output / "result.json").exists()
    suite.run_ugi_realism_diagnostic_suite(repo, config, output)
    assert calls == ["support", "model", "model", "evaluator"]
    partial = output / "stages" / "model" / "attempt_0001" / "output" / "partial.txt"
    assert partial.read_text() == "recoverable failure evidence"
    assert (output / "stages" / "model" / "attempt_0002" / "output" / "result.json").exists()
    assert (output / "failures" / "model.json").exists()
    assert (
        suite.verify_ugi_realism_diagnostic_suite(repo, output / "result.json")["status"]
        == "verified"
    )


@pytest.mark.parametrize(
    "field,value",
    [("training_calls", 1), ("remote_compute", True), ("heldout_structure_access", True)],
)
def test_scope_expansion_is_rejected(configured: tuple, field: str, value: object) -> None:
    repo, config, output = configured
    spec = read_json_object(config)
    spec["policy"][field] = value
    write_json(config, spec)
    with pytest.raises(suite.UgiRealismDiagnosticSuiteError, match="local-only policy"):
        suite.run_ugi_realism_diagnostic_suite(repo, config, output)
    assert not output.exists()


def test_output_escape_and_artifact_symlinks_fail_closed(configured: tuple) -> None:
    repo, config, output = configured
    with pytest.raises(suite.UgiRealismDiagnosticSuiteError, match="within the repository"):
        suite.run_ugi_realism_diagnostic_suite(repo, config, repo.parent / "outside", collect=True)
    packet = repo / "collected" / "support" / "packet.svg"
    packet.unlink()
    packet.symlink_to(repo / "inputs.json")
    with pytest.raises(suite.UgiRealismDiagnosticSuiteError, match="symlink"):
        suite.run_ugi_realism_diagnostic_suite(repo, config, output, collect=True)
