"""Bind an admitted corpus to the prepared run recipe without submitting compute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.phase1.multireaction.compose_lipid_run import (
    CHECKPOINT_SCHEMA,
    CONFIG_SCHEMA,
    RESULT_SCHEMA,
    read_admitted_config,
)
from experiments.phase1.multireaction.compose_lipid_training import (
    IMPLEMENTATION,
    detached_plan,
    training_files,
)
from forge.core.hashing import resolve_pin
from forge.core.io import write_json
from forge.corpus.compose_lipid_noise_marginals import compile_noise_marginals
from forge.corpus.compose_lipid_source_view import pin
from forge.corpus.compose_lipid_training_data import require_training_admission


def qualified_execution(design: dict, gpu: dict) -> tuple[str, int]:
    """Require the requested production GPU to match the passing diagnostic."""
    execution = design["execution"]
    gpu_type = execution.get("gpu_type")
    timeout = execution.get("timeout_seconds", 86400)
    if (
        gpu_type not in {"L4", "L40S"}
        or execution.get("gpu_count") != 1
        or execution.get("detached") is not True
        or execution.get("automatic_retries") != 0
        or type(timeout) is not int
        or not 1 <= timeout <= 86400
    ):
        raise ValueError(
            "Training requires one explicit GPU, detached execution and a bounded timeout"
        )
    if (
        gpu.get("schema_version") != "forge.compose_lipid_gpu_preflight.v1"
        or gpu.get("passed") is not True
        or gpu.get("device") != "cuda"
        or gpu.get("model") != design["model"]
        or gpu.get("optimizer_steps") != 0
        or gpu.get("model_state_unchanged") is not True
        or gpu.get("gpu") != f"NVIDIA {gpu_type}"
        or gpu.get("inputs", {}).get("gpu") != gpu_type
    ):
        raise ValueError("A passing CUDA diagnostic on the requested training GPU is required")
    return gpu_type, timeout


def build_package(
    repo: Path, output: Path, *, recipe: Path, admission: Path, preflight: Path
) -> Path:
    """Consume final scientific admission; fail before fitting on an unadmitted corpus."""
    repo, output = repo.resolve(), output.resolve()
    output.relative_to(repo)
    if output.exists():
        raise FileExistsError(output)
    admitted = json.loads(admission.read_text())
    inputs = {name: admitted["inputs"][name] for name in ("population", "verification", "measure")}
    inputs["admission"] = pin(repo, admission)
    require_training_admission(repo, **inputs)
    design = json.loads(recipe.read_text())
    gpu = json.loads(preflight.read_text())
    if design.get("schema_version") != "forge.compose_lipid_training_recipe.v1":
        raise ValueError("Unsupported training recipe")
    for name in ("population", "verification"):
        if design["inputs_prepared"][name] != inputs[name]:
            raise ValueError("Update the recipe for the newly admitted population")
    gpu_type, timeout = qualified_execution(design, gpu)
    payload = json.loads(
        resolve_pin(gpu["inputs"]["payload"], repo, label="GPU preflight payload").read_text()
    )
    mapped = design["inputs_prepared"]["mapped_cache"]
    if (
        payload["inputs"]["mapped_cache"] != mapped
        or payload["batch_size"] != design["runtime"]["batch_size"]
    ):
        raise ValueError("GPU diagnostic uses different data or batch size")
    inputs["mapped_cache"] = mapped
    output.mkdir(parents=True)
    noise = compile_noise_marginals(repo, output / "noise-marginals.json", **inputs)
    inputs["noise_marginals"] = pin(repo, noise)
    config = {
        name: design[name]
        for name in ("model", "runtime", "optimizer", "semantic_weights", "gradient_clip_norm")
    }
    config.update(schema_version=CONFIG_SCHEMA, inputs=inputs)
    config_path = output / "training-config.json"
    write_json(config_path, config)
    read_admitted_config(repo, config_path)
    execution_files = training_files(repo, config_path)
    stage_inputs = dict(inputs)
    stage_inputs.update(
        {f"execution-{i}": value for i, value in enumerate(execution_files.values())}
    )
    spec = {
        "schema_version": "forge.experiment.v1",
        "experiment_id": output.name.replace("_", "-"),
        "description": "Bounded initial COMPOSE training segment over all admitted formal families.",
        "root_seed": design["seed"],
        "profiles": ["full"],
        "replicates": {"full": 1},
        "stages": [
            {
                "id": "train",
                "implementation": IMPLEMENTATION,
                "needs": [],
                "config": pin(repo, config_path),
                "inputs": stage_inputs,
                "outputs": {
                    "checkpoint": {"path": "checkpoint.pt", "schema_version": CHECKPOINT_SCHEMA},
                    "result": {"path": "result.json", "schema_version": RESULT_SCHEMA},
                },
                "resources": {
                    "device": "cuda",
                    "precision": "float32",
                    "cpus": 4,
                    "workers": 0,
                    "memory_mb": 16384,
                    "timeout_seconds": timeout,
                    "gpu_type": gpu_type,
                },
                "determinism": {"mode": "strict", "stream": "compose-training"},
            }
        ],
        "metadata": {"recipe": pin(repo, recipe), "gpu_preflight": pin(repo, preflight)},
        "nonclaims": ["This first segment is not a convergence or model-quality claim."],
    }
    spec_path = output / "experiment.json"
    write_json(spec_path, spec)
    plan = detached_plan(repo, spec_path, profile="full", replicate=0)
    write_json(output / "launch-plan.json", plan)
    write_json(
        output / "result.json",
        {
            "schema_version": "forge.compose_lipid_training_package.v1",
            "ready": True,
            "config": pin(repo, config_path),
            "experiment": pin(repo, spec_path),
            "launch_plan": pin(repo, output / "launch-plan.json"),
            "training_launched": False,
        },
    )
    return spec_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "admission", "preflight", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    print(build_package(repo, **vars(args)))


if __name__ == "__main__":
    main()
