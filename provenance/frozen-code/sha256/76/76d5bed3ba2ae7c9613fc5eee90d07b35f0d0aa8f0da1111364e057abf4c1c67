"""COMPOSE stage and admission-checked detached submission through the shared runtime."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from experiments._runtime.modal import (
    launch_modal,
    modal_call_receipt_path,
    modal_request_plan,
    modal_restart_receipt_path,
)
from experiments._runtime.registry import stage
from experiments._runtime.spec import ExperimentSpec
from experiments._runtime.stage import (
    ProducedArtifact,
    RunContext,
    StageResult,
    require_config_inputs,
)
from forge.core.hashing import resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json
from forge.corpus.compose_lipid_training_data import ComposeLipidTrainingData
from forge.model.training_restart import TrainingRestartError

IMPLEMENTATION = "model.compose_lipid.training.v1"


def training_files(repo: Path, config_path: Path) -> dict[str, dict[str, str]]:
    """Authenticate exactly the receipts, tensors and recipes consumed by this trainer.

    Historical provenance links in those receipts are not execution dependencies.
    In particular, walking arbitrary JSON pins would pull unrelated archival files.
    """
    import torch

    from experiments.phase1.multireaction.compose_lipid_run import (
        DATA_INPUTS,
        read_admitted_config,
        training_marginals,
    )

    config = read_admitted_config(repo, config_path)
    files: dict[str, dict[str, str]] = {}

    def add(value: dict[str, str]) -> Path:
        path = resolve_pin(value, repo, label="COMPOSE execution dependency")
        relative = path.relative_to(repo.resolve()).as_posix()
        files[relative] = {"path": relative, "sha256": value["sha256"]}
        return path

    def read(value: dict[str, str]) -> dict:
        return json.loads(add(value).read_text())

    add({"path": str(config_path.relative_to(repo)), "sha256": str(sha256_file(config_path))})
    inputs = config["inputs"]
    for value in inputs.values():
        add(value)
    if "prepared_tensors" in inputs:
        prepared = read(inputs["prepared_tensors"])
        add(prepared["implementation"])
        for shard in prepared["shards"]:
            add(shard["artifact"])
    admission = read(inputs["admission"])
    validation = read(admission["inputs"]["validation"])
    for value in validation["inputs"].values():
        add(value)
    for name, digest in read(validation["inputs"]["source-snapshot.json"]).items():
        add({"path": name, "sha256": digest})
    weights = read(inputs["measure"])
    add(weights["implementation"])
    add(weights["artifact"])
    for value in weights["inputs"].values():
        add(value)
    add(read(weights["inputs"]["evidence_index"])["artifact"])
    population = read(inputs["population"])
    request = read(population["request"])
    lookup = add(population["artifacts"]["preparation.sqlite"])
    connection = sqlite3.connect(lookup.as_uri() + "?mode=ro", uri=True)
    try:
        if "mapped_cache" in inputs:
            mapped = read(inputs["mapped_cache"])
            add(mapped["implementation"])
            add(mapped["inputs"]["atom_vocabulary"])
            for value in mapped["artifacts"].values():
                add(value)
        else:
            remaps = read(population["artifacts"]["vocabulary-remaps.json"])
            add(remaps["atom_vocabulary"])
            for name in remaps["source_to_union_state_indices"]:
                add(request["inputs"][name]["request"])
            for (receipt,) in connection.execute("SELECT receipt FROM shards ORDER BY id"):
                shard = read(json.loads(receipt))
                add(shard["artifact"])
                origin = read(shard["origin_shard"])
                add(origin["artifact"])
    finally:
        connection.close()
    # Check the compiled measure and admission independently of the transport inventory.
    with ComposeLipidTrainingData(
        repo, **{key: inputs[key] for key in DATA_INPUTS}, mapped_cache=inputs.get("mapped_cache")
    ) as data:
        if config["model"]["maximum_heavy_atoms"] < data.maximum_heavy_atoms:
            raise TrainingRestartError("Model support would exclude large molecules")
        training_marginals(repo, config, data, torch.device("cpu"))
    return dict(sorted(files.items()))


@stage(IMPLEMENTATION)
def train(context: RunContext) -> StageResult:
    from experiments.phase1.multireaction.compose_lipid_run import (
        CHECKPOINT_SCHEMA,
        PARALLEL_GPUS,
        RESULT_SCHEMA,
        run_training,
    )

    config = context.config()
    gpu_count = PARALLEL_GPUS.get(config["runtime"].get("parallel_backend"))
    if gpu_count is not None and (
        not (context.resources.gpu_type or "").endswith(f":{gpu_count}")
        or context.resources.cpus < gpu_count * config["runtime"]["cpu_threads"]
    ):
        raise TrainingRestartError("Parallel workers require their GPU count and CPU thread budget")
    require_config_inputs(context, config, labels=set(config["inputs"]))
    if (
        context.resources.precision != "float32"
        or context.resources.workers != 0
        or context.stage.determinism.mode != "strict"
        or context.resources.cpus < config["runtime"]["cpu_threads"]
    ):
        raise TrainingRestartError(
            "Allocated resources differ from deterministic training contract"
        )
    result = run_training(
        context.repo,
        context.config_path,
        context.work_dir,
        context.output_dir,
        device=context.resources.device,
        seed=context.derive_seed("compose-training") % 2**32,
        resume=context.resume,
        commit_progress=context.commit_progress,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact("checkpoint", "checkpoint.pt", CHECKPOINT_SCHEMA),
            ProducedArtifact("result", "result.json", RESULT_SCHEMA),
        ),
        metrics={
            "completed_steps": result["completed_steps"],
            "examples_seen": result["examples_seen"],
        },
        summary={"status": result["status"]},
    )


def detached_plan(repo: Path, spec_path: Path, *, profile: str, replicate: int) -> dict[str, Any]:
    from experiments.phase1.multireaction.compose_lipid_run import (
        CHECKPOINT_SCHEMA,
        PARALLEL_GPUS,
        RESULT_SCHEMA,
        read_admitted_config,
    )

    spec = ExperimentSpec.load(spec_path)
    if len(spec.stages) != 1 or spec.stages[0].implementation != IMPLEMENTATION:
        raise TrainingRestartError("Expected one COMPOSE training stage")
    item = spec.stages[0]
    config_path = item.config.resolve(repo)
    config = read_admitted_config(repo, config_path)
    gpu_count = PARALLEL_GPUS.get(config["runtime"].get("parallel_backend"))
    if gpu_count is not None and (
        not (item.resources.gpu_type or "").endswith(f":{gpu_count}")
        or item.resources.cpus < gpu_count * config["runtime"]["cpu_threads"]
    ):
        raise TrainingRestartError("Parallel workers require their GPU count and CPU thread budget")
    expected_outputs = {
        "checkpoint": {"path": "checkpoint.pt", "schema_version": CHECKPOINT_SCHEMA},
        "result": {"path": "result.json", "schema_version": RESULT_SCHEMA},
    }
    if {key: value.to_mapping() for key, value in item.outputs.items()} != expected_outputs:
        raise TrainingRestartError("COMPOSE stage must publish its checkpoint and result")
    if (
        item.resources.precision != "float32"
        or item.resources.workers != 0
        or item.determinism.mode != "strict"
        or item.resources.cpus < config["runtime"]["cpu_threads"]
    ):
        raise TrainingRestartError("Invalid COMPOSE allocation or determinism policy")
    required = training_files(repo, config_path)
    for name, value in config["inputs"].items():
        if name not in item.inputs or item.inputs[name].to_mapping() != value:
            raise TrainingRestartError(f"Stage input differs from training config: {name}")
    plan = modal_request_plan(repo, spec_path, profile=profile, replicate=replicate, device=None)
    missing = [
        name
        for name, value in required.items()
        if plan["uploads"].get(name, {}).get("sha256") != value["sha256"]
    ]
    if missing:
        raise TrainingRestartError(f"Declare missing execution files in stage.inputs: {missing}")
    return plan


def submit_detached(
    repo: Path,
    spec_path: Path,
    *,
    profile: str,
    replicate: int,
    restart_from: Path | None = None,
    diagnosis: str = "",
) -> Path:
    """Submit once, persist ambiguity, and never retry a possibly billable request."""
    plan = detached_plan(repo, spec_path, profile=profile, replicate=replicate)
    receipt_path = modal_call_receipt_path(repo, plan["request_id"])
    if restart_from is not None:
        if not diagnosis.strip():
            raise TrainingRestartError("Restart requires a recorded failure diagnosis")
        receipt_path = modal_restart_receipt_path(repo, plan, restart_from)
    attempt = repo / "runs" / "_compose_submissions" / receipt_path.stem
    attempt.parent.mkdir(parents=True, exist_ok=True)
    if receipt_path.exists():
        raise TrainingRestartError("Existing call: inspect/collect it; do not submit again")
    try:
        attempt.mkdir()
    except FileExistsError as error:
        raise TrainingRestartError(
            "Prior submission attempt requires diagnosis before any paid retry"
        ) from error
    write_json(attempt / "request.json", plan)
    try:
        status = launch_modal(
            repo,
            spec_path,
            profile=profile,
            replicate=replicate,
            device=None,
            resume=restart_from is not None,
            detached=True,
            **(dict(restart_from=restart_from, diagnosis=diagnosis) if restart_from else {}),
        )
        if status:
            raise TrainingRestartError(
                f"Detached launcher exited {status}; execution status is unknown"
            )
        receipt = read_json_object(receipt_path, error=TrainingRestartError)
        for key in ("request_id", "source_sha256", "spec_sha256", "uploads"):
            if receipt.get(key) != plan[key]:
                raise TrainingRestartError(f"Detached call receipt changed {key}")
        if receipt.get("status") != "launched" or not receipt.get("function_call_id"):
            raise TrainingRestartError("Detached call identifier was not persisted")
    except BaseException as error:
        write_json(
            attempt / "failure.json",
            {"status": "submission_unconfirmed", "error": str(error), "automatic_retry": False},
        )
        raise
    write_json(
        attempt / "submitted.json",
        {
            "call_receipt": str(receipt_path.relative_to(repo)),
            "function_call_id": receipt["function_call_id"],
        },
    )
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "submit", "resume"))
    parser.add_argument("spec", type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--restart-from", type=Path)
    parser.add_argument("--diagnosis", default="")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    spec = (repo / args.spec).resolve()
    spec.relative_to(repo)
    options = {"profile": args.profile, "replicate": args.replicate}
    if args.action == "resume":
        if args.restart_from is None or not args.diagnosis.strip():
            parser.error("resume requires --restart-from RECEIPT --diagnosis TEXT")
        options.update(restart_from=(repo / args.restart_from).resolve(), diagnosis=args.diagnosis)
    elif args.restart_from is not None or args.diagnosis:
        parser.error("restart arguments require the resume action")
    if args.action == "plan":
        print(json.dumps(detached_plan(repo, spec, **options), indent=2, sort_keys=True))
    else:
        print(submit_detached(repo, spec, **options))


if __name__ == "__main__":
    main()
