"""Checkpoint admission and durable shard recovery; no model or chemistry execution."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

MODULE = (
    Path(__file__).resolve().parents[1]
    / "experiments/phase1/multireaction/run_matched_context_null.py"
)
spec = importlib.util.spec_from_file_location("matched_context_execution_test", MODULE)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def complete_state():
    families = ["family_" + str(i) for i in range(22)]
    config = {
        "runtime_source_sha256": "source",
        "inputs": {"population": "pinned"},
        "model": {"semantic_conditioning": "semantic_null"},
    }
    config_pin = {"sha256": "configuration"}
    state = {
        "identity": {
            "config_sha256": "configuration",
            "source_sha256": "source",
            "inputs": config["inputs"],
            "seed": 2026092401,
        },
        "completed_steps": 2794,
        "examples_seen": 8851392,
        "family_presentations": {f: 402336 for f in families},
    }
    return state, config, config_pin, families


def test_exact_full_exposure_state_is_accepted():
    runner.validate_checkpoint(*complete_state())


@pytest.mark.parametrize(
    "mutation", ["step", "seed", "data", "hash", "source", "count", "family", "mode"]
)
def test_checkpoint_refuses_incomplete_or_unmatched_fit(mutation):
    state, config, pin, families = copy.deepcopy(complete_state())
    if mutation == "step":
        state["completed_steps"] = 22
    elif mutation == "seed":
        state["identity"]["seed"] += 1
    elif mutation == "data":
        state["identity"]["inputs"] = {"population": "another"}
    elif mutation == "hash":
        state["identity"]["config_sha256"] = "changed"
    elif mutation == "source":
        state["identity"]["source_sha256"] = "changed"
    elif mutation == "count":
        state["family_presentations"][families[0]] -= 1
    elif mutation == "family":
        state["family_presentations"]["wrong_family"] = state["family_presentations"].pop(
            families[0]
        )
    elif mutation == "mode":
        config["model"]["semantic_conditioning"] = "conditioned"
    with pytest.raises(ValueError, match="complete, exposure-matched"):
        runner.validate_checkpoint(state, config, pin, families)


def closed_shard(tmp_path):
    entry = {"draw": 0, "offset": 16, "seed": 42}
    rows = [{"draw": 0, "index": i} for i in range(16, 24)]
    readouts = tmp_path / "readouts.json"
    readouts.write_text(json.dumps(rows))
    prediction = tmp_path / "predictions.pt"
    prediction.write_bytes(b"fixture tensor artifact")
    states = tmp_path / "states.pt"
    states.write_bytes(b"fixture graph state artifact")

    def pin(path):
        return {"path": path.name, "sha256": runner.fixture.digest(path)}

    receipt = {
        "request_sha256": "frozenrequest",
        "schedule": entry,
        "draw": 0,
        "offset": 16,
        "readouts": pin(readouts),
        "predictions": pin(prediction),
        "states": pin(states),
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    return path, entry, receipt, rows


def test_closed_shard_authenticates_all_outputs_and_exact_eight_rows(tmp_path):
    path, entry, receipt, rows = closed_shard(tmp_path)
    assert runner.read_closed_sampling(path, entry, "frozenrequest", tmp_path) == (receipt, rows)


@pytest.mark.parametrize("mutation", ["request", "schedule", "artifact", "row_count", "index"])
def test_resume_rejects_tampering_or_dropped_rows(tmp_path, mutation):
    path, entry, receipt, rows = closed_shard(tmp_path)
    if mutation == "request":
        receipt["request_sha256"] = "other"
    elif mutation == "schedule":
        receipt["schedule"] = dict(entry, seed=43)
    elif mutation == "artifact":
        (tmp_path / "predictions.pt").write_bytes(b"changed")
    else:
        if mutation == "row_count":
            rows.pop()
        else:
            rows[0]["index"] = 17
        artifact = tmp_path / "readouts.json"
        artifact.write_text(json.dumps(rows))
        receipt["readouts"]["sha256"] = runner.fixture.digest(artifact)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        runner.read_closed_sampling(path, entry, "frozenrequest", tmp_path)


def test_tensor_persistence_never_overwrites_partial_or_closed_file(tmp_path):
    path = tmp_path / "output.pt"
    partial = tmp_path / "output.pt.partial"
    partial.write_bytes(b"preserve failed write")
    with pytest.raises(FileExistsError):
        runner.atomic_tensor(path, torch.tensor([1]))
    assert partial.read_bytes() == b"preserve failed write"
    other = tmp_path / "closed.pt"
    runner.atomic_tensor(other, torch.tensor([2]))
    with pytest.raises(FileExistsError):
        runner.atomic_tensor(other, torch.tensor([3]))
    assert torch.equal(torch.load(other, weights_only=True), torch.tensor([2]))


def test_resume_budget_charges_closed_actual_cpu_and_unclosed_full_reserve(tmp_path):
    first, remaining = runner.open_attempt(tmp_path, 10, "request")
    assert remaining == 10
    runner.fixture.write(
        first.with_name(first.name.replace(".attempt.json", ".closed.json")),
        {"CPU_seconds": 3.2, "complete": False},
    )
    _, remaining = runner.open_attempt(tmp_path, 10, "request")
    assert remaining == 6
    with pytest.raises(ValueError, match="budget exhausted"):
        runner.open_attempt(tmp_path, 10, "request")


def test_resume_rejects_foreign_attempt_budget(tmp_path):
    runner.open_attempt(tmp_path, 10, "other")
    with pytest.raises(ValueError, match="another request"):
        runner.open_attempt(tmp_path, 10, "request")
