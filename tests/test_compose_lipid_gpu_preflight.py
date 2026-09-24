"""Diagnostic backward execution must leave model weights unchanged and remain bounded."""

from dataclasses import asdict

import pytest
import torch

from experiments.phase1.multireaction.compose_lipid_gpu_preflight import (
    check_payload,
    requested_gpu,
    run_preflight,
)
from experiments.phase1.multireaction.compose_lipid_package import qualified_execution
from forge.corpus.qualified_program_cache import (
    QualifiedProgramExample,
    collate_qualified_program_examples,
)
from tests.test_source_instance_coordinates import repeated


def payload():
    graph, vocabulary, atoms = repeated()
    example = QualifiedProgramExample(graph, "synthetic", (("head", "h", 1), ("tail", "t", 3)), ())
    batch = collate_qualified_program_examples([example], maximum_nodes=128, maximum_closures=2)
    return {
        "schema_version": "forge.compose_lipid_gpu_preflight_input.v1",
        "seed": 1,
        "optimizer_steps": 0,
        "training_admitted": False,
        "precision": "float32",
        "deterministic": True,
        "batch_size": 2,
        "repeats": 2,
        "maximum_atoms": graph.node_count,
        "maximum_closures": graph.graph.closure_count,
        "target_ids": [graph.graph.structure_id],
        "node_classes": len(atoms),
        "vocabulary": asdict(vocabulary),
        "model": {
            "architecture": "reaction_program_graph_transformer",
            "hidden_dim": 16,
            "layers": 1,
            "attention_heads": 2,
            "expert_count": 2,
            "adapter_dim": 4,
            "maximum_heavy_atoms": 128,
            "maximum_closures": 2,
            "bond_classes": 3,
            "dropout": 0.1,
            "repeat_group_conditioning": True,
        },
        "semantic_weights": {
            "role_weight": 0.25,
            "core_weight": 0.25,
            "repeat_consistency_weight": 0.25,
        },
        "batch": {
            k: {"dtype": str(v.dtype).removeprefix("torch."), "values": v.tolist()}
            for k, v in batch.items()
        },
    }


def test_backward_diagnostic_is_repeatable_without_optimizer(monkeypatch):
    monkeypatch.setattr(
        torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("No optimizer allowed")
    )
    result = run_preflight(payload(), device="cpu")
    assert result["passed"] and result["model_state_unchanged"]
    assert result["optimizer_steps"] == 0
    assert result["batches"][-1]["maximum_size_stress"]


@pytest.mark.parametrize(
    "key,value",
    [("optimizer_steps", 1), ("batch_size", 33), ("repeats", 4), ("training_admitted", True)],
)
def test_preflight_cannot_expand_into_training(key, value):
    data = payload()
    data[key] = value
    with pytest.raises(ValueError, match="bounded"):
        check_payload(data)


@pytest.mark.parametrize("gpu_type", ["L4", "L40S"])
def test_explicit_gpu_allocation_matches_production(gpu_type):
    request = {"gpu": gpu_type, "timeout_seconds": 600, "retries": 0, "optimizer_steps": 0}
    assert requested_gpu(request) == gpu_type
    model = payload()["model"]
    design = {
        "model": model,
        "execution": {
            "gpu_type": gpu_type,
            "gpu_count": 1,
            "detached": True,
            "automatic_retries": 0,
            "timeout_seconds": 28800,
        },
    }
    result = {
        "schema_version": "forge.compose_lipid_gpu_preflight.v1",
        "passed": True,
        "device": "cuda",
        "model": model,
        "optimizer_steps": 0,
        "model_state_unchanged": True,
        "gpu": f"NVIDIA {gpu_type}",
        "inputs": request,
    }
    assert qualified_execution(design, result) == (gpu_type, 28800)
    result["gpu"] = "NVIDIA H100"
    with pytest.raises(ValueError, match="requested training GPU"):
        qualified_execution(design, result)


@pytest.mark.parametrize(
    "key,value", [("gpu", "any"), ("retries", 1), ("timeout_seconds", 601), ("optimizer_steps", 1)]
)
def test_gpu_request_cannot_expand_execution(key, value):
    request = {"gpu": "L40S", "timeout_seconds": 600, "retries": 0, "optimizer_steps": 0}
    request[key] = value
    with pytest.raises(ValueError, match="explicit"):
        requested_gpu(request)
