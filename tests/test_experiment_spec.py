from __future__ import annotations

from copy import deepcopy

import pytest

from experiments._runtime.errors import SpecError
from experiments._runtime.spec import ExperimentSpec

PIN = {"path": "data/input.json", "sha256": "a" * 64}
RESOURCES = {
    "device": "cpu",
    "precision": "float32",
    "cpus": 1,
    "workers": 0,
    "memory_mb": 512,
    "timeout_seconds": 60,
    "gpu_type": None,
}


def stage(stage_id: str, needs: list[str] | None = None) -> dict[str, object]:
    return {
        "id": stage_id,
        "implementation": f"test.{stage_id}.v1",
        "needs": needs or [],
        "config": PIN,
        "inputs": {"source": PIN},
        "outputs": {"result": {"path": f"{stage_id}.json", "schema_version": "test.result.v1"}},
        "resources": RESOURCES,
        "determinism": {"mode": "strict", "stream": stage_id},
    }


def document() -> dict[str, object]:
    return {
        "schema_version": "forge.experiment.v1",
        "experiment_id": "test-pipeline",
        "description": "A small typed experiment.",
        "root_seed": 17,
        "profiles": ["smoke", "full"],
        "replicates": {"smoke": 1, "full": 5},
        "stages": [stage("prepare"), stage("evaluate", ["prepare"])],
        "metadata": {"owner": "tests"},
        "nonclaims": ["This is not a scientific result."],
    }


def test_loads_a_strict_acyclic_experiment() -> None:
    spec = ExperimentSpec.from_mapping(document())
    assert spec.experiment_id == "test-pipeline"
    assert [item.stage_id for item in spec.topological_stages()] == ["prepare", "evaluate"]
    assert spec.to_mapping() == document()


def test_unknown_field_fails_instead_of_being_ignored() -> None:
    value = document()
    value["typo_seed"] = 19
    with pytest.raises(SpecError, match="unknown"):
        ExperimentSpec.from_mapping(value)


def test_cycle_is_reported_with_the_stage_path() -> None:
    value = document()
    value["stages"] = [stage("left", ["right"]), stage("right", ["left"])]
    with pytest.raises(SpecError, match="cycle"):
        ExperimentSpec.from_mapping(value)


def test_output_cannot_escape_the_private_stage_directory() -> None:
    value = deepcopy(document())
    stages = value["stages"]
    assert isinstance(stages, list)
    first = stages[0]
    assert isinstance(first, dict)
    outputs = first["outputs"]
    assert isinstance(outputs, dict)
    result = outputs["result"]
    assert isinstance(result, dict)
    result["path"] = "../result.json"
    with pytest.raises(SpecError, match="inside"):
        ExperimentSpec.from_mapping(value)


def test_gpu_type_requires_cuda() -> None:
    value = deepcopy(document())
    stages = value["stages"]
    assert isinstance(stages, list)
    first = stages[0]
    assert isinstance(first, dict)
    resources_value = first["resources"]
    assert isinstance(resources_value, dict)
    resources_value["gpu_type"] = "L4"
    with pytest.raises(SpecError, match="only valid"):
        ExperimentSpec.from_mapping(value)
