"""Orchestration integrity; chemistry remains covered by stage tests and semantic replay."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.phase1.multireaction import combinatorial_generation_pipeline as pipeline


@pytest.fixture
def harness(tmp_path, monkeypatch):
    config = {
        "sampling": {"attempts_per_family": 1},
        "population_mode": "fresh_train",
        "source_pins": [],  # This harness replaces the chemistry/source contract with stubs.
    }
    request = tmp_path / "config.json"
    pipeline._write(request, config)
    calls, checks = [], []
    fail = {"stage": None}

    def contract(repo, path):
        return json.loads(path.read_text()), {}

    def stage_config(stage, config, templates, prior):
        return {"stage": stage, "parent": prior, "sampling": config["sampling"]}

    def runner(stage):
        def run(repo, config_path, output):
            calls.append(stage)
            if fail["stage"] == stage:
                raise RuntimeError("deliberate interrupted stage")
            pipeline._write(
                output / "result.json",
                {"config": pipeline._pin(config_path, repo), "stage": stage},
            )

        return run

    def verify_stage(stage, repo, path):
        checks.append(stage)
        assert pipeline._read(path)["stage"] == stage

    def summarize(repo, config, results):
        assert list(results) == list(pipeline.STAGES)
        return {"acceptance_passed": True, "qualified_product_count": 1}, [{"sample_index": 0}]

    monkeypatch.setattr(pipeline, "contract", contract)
    monkeypatch.setattr(pipeline, "stage_config", stage_config)
    monkeypatch.setattr(pipeline, "_verify_stage", verify_stage)
    monkeypatch.setattr(pipeline, "summarize", summarize)
    monkeypatch.setattr(
        pipeline,
        "MODULES",
        {stage: SimpleNamespace(run=runner(stage)) for stage in pipeline.STAGES},
    )
    return SimpleNamespace(root=tmp_path, request=request, calls=calls, checks=checks, fail=fail)


def test_resume_preserves_completed_stages_and_failed_attempts(harness):
    h = harness
    h.fail["stage"] = "occurrence"
    with pytest.raises(RuntimeError, match="interrupted"):
        pipeline.run(h.root, h.request, "run")
    failed = h.root / "run/stages/occurrence/attempt_0001/failure.json"
    saved_failure = failed.read_bytes()
    completed = h.root / "run/stages/graph/complete.json"
    saved_completed = completed.read_bytes()
    h.fail["stage"] = None
    result = pipeline.run(h.root, h.request, "run")
    assert result["acceptance_passed"]
    assert h.calls == [
        "graph",
        "admission",
        "occurrence",
        "occurrence",
        "connection",
        "source_core",
    ]
    assert failed.read_bytes() == saved_failure
    assert completed.read_bytes() == saved_completed
    assert (h.root / "run/stages/occurrence/attempt_0002/output/result.json").exists()
    before = list(h.calls)
    pipeline.run(h.root, h.request, "run")
    assert h.calls == before
    assert pipeline.verify(h.root, h.root / "run/result.json")["status"] == "verified"


def test_changed_request_is_rejected_before_running_stages(harness):
    h = harness
    pipeline.run(h.root, h.request, "run")
    before = list(h.calls)
    pipeline._write(h.request, {"sampling": {"attempts_per_family": 2}, "source_pins": []})
    with pytest.raises(ValueError, match="different request"):
        pipeline.run(h.root, h.request, "run")
    assert h.calls == before


def test_two_writers_cannot_share_output(harness):
    h = harness
    with pipeline._lock(h.root / "run"):
        with pytest.raises(ValueError, match="active writer"):
            pipeline.run(h.root, h.request, "run")
    assert not h.calls


@pytest.mark.parametrize("mutation", ["result_config", "parent", "export", "summary", "stage_set"])
def test_semantic_verifier_rejects_substitution_even_with_updated_hashes(harness, mutation):
    h = harness
    result = pipeline.run(h.root, h.request, "run")
    path = h.root / "run/result.json"
    if mutation in {"result_config", "parent"}:
        slot = result["stages"]["occurrence"]
        stage_result_path = h.root / slot["result"]["path"]
        stage_result = pipeline._read(stage_result_path)
        if mutation == "result_config":
            stage_result["config"] = result["stages"]["admission"]["config"]
        else:
            config_path = h.root / slot["config"]["path"]
            config = pipeline._read(config_path)
            config["parent"] = result["stages"]["graph"]["result"]
            pipeline._write(config_path, config)
            slot["config"] = pipeline._pin(config_path, h.root)
            stage_result["config"] = slot["config"]
        pipeline._write(stage_result_path, stage_result)
        slot["result"] = pipeline._pin(stage_result_path, h.root)
    elif mutation == "export":
        export = h.root / result["qualified_products"]["path"]
        export.write_text('{"sample_index": 7}\n')
        result["qualified_products"] = pipeline._pin(export, h.root)
    elif mutation == "summary":
        result["qualified_product_count"] = 100
    else:
        del result["stages"]["connection"]
    pipeline._write(path, result)
    with pytest.raises(ValueError):
        pipeline.verify(h.root, path)


def test_stage_config_preserves_policies_and_budgets():
    config = {
        "population_mode": "fresh_train",
        "sampling": {"attempts_per_family": 128, "flow_steps": 32, "batch_size": 16},
    }
    templates = {
        s: {
            "schema_version": s,
            "policy": {"gate_changes": False},
            "inputs": {"cache": {"path": "cache", "sha256": "a" * 64}},
            "source_pins": [{"path": "source.py", "sha256": "b" * 64}],
            "maximum_canonical_matches": 4096,
            "use_saved_layouts": True,
        }
        for s in pipeline.STAGES
    }
    before = deepcopy(templates)
    parent = {"path": "prior/result.json", "sha256": "c" * 64}
    for stage in pipeline.STAGES:
        result = pipeline.stage_config(stage, config, templates, parent)
        assert result["policy"] == templates[stage]["policy"]
        if stage == "graph":
            assert result["sampling"] == config["sampling"]
            assert not result["use_saved_layouts"]
        elif stage == "admission":
            assert result["parent_result"] == parent
        else:
            assert result["baseline_result"] == parent
    assert templates == before


@pytest.mark.parametrize("mutation", ["gate", "budget", "source", "chain", "sampling"])
def test_frozen_request_rejects_drift(tmp_path, mutation):
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/multireaction/combinatorial_generation_pipeline_fresh_v2.json"
    config = pipeline._read(path)
    if mutation == "gate":
        config["policy"]["gate_changes"] = True
    elif mutation == "budget":
        config["sampling"]["attempts_per_family"] = 257
    elif mutation == "source":
        config["source_pins"].pop()
    elif mutation == "chain":
        config["reference_results"]["source_core"] = config["reference_results"]["connection"]
    else:
        config["sampling"]["batch_size"] = 32
    changed = tmp_path / "config.json"
    pipeline._write(changed, config)
    with pytest.raises(ValueError):
        pipeline.contract(root, changed)


def test_current_frozen_request_accepts_unchanged_sources():
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/multireaction/combinatorial_generation_pipeline_fresh_v2.json"
    config, templates = pipeline.contract(root, path)
    assert set(templates) == set(pipeline.STAGES)
    assert config["population_mode"] == "fresh_train"
