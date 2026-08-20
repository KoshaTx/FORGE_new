from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments._runtime.errors import RunExistsError, StageError, VerificationError
from experiments._runtime.registry import StageRegistry
from experiments._runtime.runner import ExperimentRunner, verify_run_directory
from experiments._runtime.stage import ProducedArtifact, RunContext, StageResult
from forge.core.hashing import sha256_file
from forge.core.io import atomic_write, write_json


def pin(path: Path, repo: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(repo)), "sha256": str(sha256_file(path))}


def resources() -> dict[str, object]:
    return {
        "device": "cpu",
        "precision": "float32",
        "cpus": 1,
        "workers": 0,
        "memory_mb": 512,
        "timeout_seconds": 60,
        "gpu_type": None,
    }


def build_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path
    (repo / "forge").mkdir(parents=True)
    atomic_write(repo / "forge" / "example.py", b"VALUE = 1\n")
    atomic_write(repo / "pyproject.toml", b"[project]\nname='fixture'\nversion='0'\n")
    (repo / "configs").mkdir()
    config = repo / "configs" / "stage.json"
    write_json(config, {"schema_version": "test.stage_config.v1", "multiplier": 3})
    (repo / "data").mkdir()
    source = repo / "data" / "source.json"
    write_json(source, {"value": 7})

    experiment = {
        "schema_version": "forge.experiment.v1",
        "experiment_id": "runner-test",
        "description": "Two deterministic fixture stages.",
        "root_seed": 123,
        "profiles": ["smoke"],
        "replicates": {"smoke": 2},
        "stages": [
            {
                "id": "prepare",
                "implementation": "test.prepare.v1",
                "needs": [],
                "config": pin(config, repo),
                "inputs": {"source": pin(source, repo)},
                "outputs": {
                    "prepared": {
                        "path": "prepared.json",
                        "schema_version": "test.prepared.v1",
                    }
                },
                "resources": resources(),
                "determinism": {"mode": "strict", "stream": "prepare"},
            },
            {
                "id": "evaluate",
                "implementation": "test.evaluate.v1",
                "needs": ["prepare"],
                "config": pin(config, repo),
                "inputs": {},
                "outputs": {"score": {"path": "score.json", "schema_version": "test.score.v1"}},
                "resources": resources(),
                "determinism": {"mode": "strict", "stream": "evaluate"},
            },
        ],
        "metadata": {},
        "nonclaims": ["Fixture only."],
    }
    spec = repo / "configs" / "experiment.json"
    write_json(spec, experiment)
    return repo, spec


def fixture_registry() -> StageRegistry:
    registry = StageRegistry()

    def prepare(context: RunContext) -> StageResult:
        source = json.loads(context.input("source").read_text())
        config = context.config()
        seed = context.derive_seed("example", source["value"])
        output = context.output_path("prepared.json")
        write_json(
            output,
            {
                "schema_version": "test.prepared.v1",
                "seed": seed,
                "value": source["value"] * config["multiplier"],
            },
        )
        return StageResult(
            artifacts=(ProducedArtifact("prepared", "prepared.json", "test.prepared.v1", rows=1),),
            summary={"rows": 1},
        )

    def evaluate(context: RunContext) -> StageResult:
        prepared = json.loads(context.dependency("prepare", "prepared").path.read_text())
        output = context.output_path("score.json")
        write_json(output, {"schema_version": "test.score.v1", "score": prepared["value"] + 1})
        return StageResult(
            artifacts=(ProducedArtifact("score", "score.json", "test.score.v1", rows=1),),
            metrics={"score": prepared["value"] + 1},
        )

    registry.register("test.prepare.v1", prepare)
    registry.register("test.evaluate.v1", evaluate)
    return registry


def test_runs_a_dag_and_verifies_every_output(tmp_path: Path) -> None:
    repo, spec = build_repo(tmp_path)
    runner = ExperimentRunner(repo, registry=fixture_registry(), source_paths=("forge",))
    result = runner.run(spec, profile="smoke")
    assert list(result.stage_manifests) == ["prepare", "evaluate"]
    score = json.loads(
        (result.plan.run_dir / "stages" / "evaluate" / "artifacts" / "score.json").read_text()
    )
    assert score["score"] == 22
    assert runner.verify(spec, profile="smoke").plan.run_id == result.plan.run_id
    receipt = verify_run_directory(result.plan.run_dir)
    assert receipt["artifacts"] == 2
    assert receipt["run_id"] == result.plan.run_id


def test_resume_reuses_only_verified_completed_stages(tmp_path: Path) -> None:
    repo, spec = build_repo(tmp_path)
    runner = ExperimentRunner(repo, registry=fixture_registry(), source_paths=("forge",))
    first = runner.run(spec, profile="smoke")
    with pytest.raises(RunExistsError):
        runner.run(spec, profile="smoke")
    resumed = runner.run(spec, profile="smoke", resume=True)
    assert resumed.plan.run_id == first.plan.run_id


def test_tampered_output_is_rejected(tmp_path: Path) -> None:
    repo, spec = build_repo(tmp_path)
    runner = ExperimentRunner(repo, registry=fixture_registry(), source_paths=("forge",))
    result = runner.run(spec, profile="smoke")
    output = result.plan.run_dir / "stages" / "prepare" / "artifacts" / "prepared.json"
    atomic_write(output, b"{}\n")
    with pytest.raises(VerificationError, match="changed"):
        runner.verify(spec, profile="smoke")


def test_replicates_have_distinct_recorded_seed_streams(tmp_path: Path) -> None:
    repo, spec = build_repo(tmp_path)
    runner = ExperimentRunner(repo, registry=fixture_registry(), source_paths=("forge",))
    first = runner.run(spec, profile="smoke", replicate=0)
    second = runner.run(spec, profile="smoke", replicate=1)
    assert first.plan.run_id != second.plan.run_id
    assert first.plan.root_seed != second.plan.root_seed
    assert first.plan.master_seed == second.plan.master_seed == 123
    with pytest.raises(StageError, match="outside"):
        runner.plan(spec, profile="smoke", replicate=2)


def test_strict_reproduction_executes_twice_and_requires_byte_identity(tmp_path: Path) -> None:
    repo, spec = build_repo(tmp_path)
    runner = ExperimentRunner(repo, registry=fixture_registry(), source_paths=("forge",))
    receipt = runner.reproduce(spec, profile="smoke")
    assert receipt["status"] == "reproduced"
    assert receipt["stages"]["prepare"]["byte_identical"] is True
    assert receipt["stages"]["evaluate"]["requirement"] == "byte_identity"


def test_failure_leaves_resumable_partial_stage_but_no_committed_stage(tmp_path: Path) -> None:
    repo, spec = build_repo(tmp_path)
    registry = fixture_registry()

    def fail(_: RunContext) -> StageResult:
        raise RuntimeError("intentional failure")

    document = json.loads(spec.read_text())
    document["stages"][1]["implementation"] = "test.fail.v1"
    write_json(spec, document)
    registry.register("test.fail.v1", fail)
    runner = ExperimentRunner(repo, registry=registry, source_paths=("forge",))
    with pytest.raises(RuntimeError, match="intentional"):
        runner.run(spec, profile="smoke")
    plan = runner.plan(spec, profile="smoke")
    assert not (plan.run_dir / "stages" / "evaluate").exists()
    partial = plan.run_dir / "stages" / ".evaluate.partial"
    assert json.loads((partial / "partial.json").read_text())["status"] == "incomplete"
    assert list((plan.run_dir / "failures" / "evaluate").glob("*.json"))


def test_resume_reenters_matching_partial_stage(tmp_path: Path) -> None:
    repo, spec = build_repo(tmp_path)
    registry = fixture_registry()
    calls = 0

    def once(context: RunContext) -> StageResult:
        nonlocal calls
        calls += 1
        checkpoint = context.work_dir / "checkpoint.json"
        marker = context.output_path("attempt.json")
        if not context.resume:
            write_json(checkpoint, {"step": 7})
            write_json(marker, {"attempt": calls})
            raise RuntimeError("interrupt after checkpoint")
        assert json.loads(checkpoint.read_text()) == {"step": 7}
        assert json.loads(marker.read_text()) == {"attempt": 1}
        write_json(
            context.output_path("score.json"),
            {"schema_version": "test.score.v1", "score": 22},
        )
        return StageResult(
            artifacts=(ProducedArtifact("score", "score.json", "test.score.v1", rows=1),)
        )

    document = json.loads(spec.read_text())
    document["stages"][1]["implementation"] = "test.once.v1"
    write_json(spec, document)
    registry.register("test.once.v1", once)
    runner = ExperimentRunner(repo, registry=registry, source_paths=("forge",))
    with pytest.raises(RuntimeError, match="interrupt"):
        runner.run(spec, profile="smoke")
    completed = runner.run(spec, profile="smoke", resume=True)
    assert completed.stage_manifests["evaluate"]["status"] == "complete"
    assert not (completed.plan.run_dir / "stages" / "evaluate" / "work").exists()
    assert calls == 2
