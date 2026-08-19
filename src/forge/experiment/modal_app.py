#!/usr/bin/env python3
"""Run any registered FORGE experiment DAG on Modal and download its verified artifacts."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import modal

LOCAL_REPO = Path(__file__).resolve().parents[3]
IMAGE_PROJECT = Path("/opt/forge-project")
IMAGE_SOURCE = IMAGE_PROJECT / "src" / "forge"
VOLUME_ROOT = Path("/forge-workspace")
VOLUME_NAME = "forge-experiment-runs"

app = modal.App("forge-experiment-runner")
experiment_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file(
        LOCAL_REPO / "pyproject.toml",
        remote_path=str(IMAGE_PROJECT / "pyproject.toml"),
        copy=True,
    )
    .add_local_file(
        LOCAL_REPO / "uv.lock",
        remote_path=str(IMAGE_PROJECT / "uv.lock"),
        copy=True,
    )
    .uv_sync(
        str(IMAGE_PROJECT),
        extras=["oracle", "torch"],
        extra_options="--no-install-project",
        frozen=True,
    )
    .add_local_dir(
        LOCAL_REPO / "src" / "forge",
        remote_path=str(IMAGE_SOURCE),
        copy=True,
    )
)


def _upload_paths(spec_path: Path) -> dict[str, Path]:
    from forge.experiment.spec import ExperimentSpec

    spec = ExperimentSpec.load(spec_path)
    paths = {str(spec_path.relative_to(LOCAL_REPO)): spec_path}
    for stage in spec.stages:
        for pin in (stage.config, *stage.inputs.values()):
            paths[pin.path] = pin.resolve(LOCAL_REPO)
    paths["pyproject.toml"] = LOCAL_REPO / "pyproject.toml"
    paths["uv.lock"] = LOCAL_REPO / "uv.lock"
    return paths


@app.function(
    image=image,
    cpu=1.0,
    memory=512,
    timeout=300,
    retries=0,
    volumes={str(VOLUME_ROOT): experiment_volume},
)
def execute_experiment(
    request_id: str,
    experiment_path: str,
    profile: str,
    replicate: int,
    device: str | None,
    resume: bool,
    expected_source_sha256: str,
) -> dict[str, Any]:
    """Execute one content-addressed DAG inside its persistent request workspace."""

    repo = VOLUME_ROOT / "jobs" / request_id / "repo"
    source = repo / "src" / "forge"
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(IMAGE_SOURCE, source, dirs_exist_ok=True)
    sys.path.insert(0, str(repo / "src"))

    from forge.core.hashing import sha256_tree
    from forge.experiment import ExperimentRunner, ModalRuntimeBackend

    observed_source = str(sha256_tree(source))
    if observed_source != expected_source_sha256:
        raise RuntimeError(
            f"remote source changed: expected {expected_source_sha256}, found {observed_source}"
        )
    runner = ExperimentRunner(repo, backend=ModalRuntimeBackend())
    result = runner.run(
        repo / experiment_path,
        profile=profile,
        replicate=replicate,
        device=device,
        resume=resume,
    )
    runner.verify(
        repo / experiment_path,
        profile=profile,
        replicate=replicate,
        device=device,
    )
    experiment_volume.commit()
    return {
        "backend": "modal",
        "experiment_id": result.plan.experiment_id,
        "remote_run_path": str(result.plan.run_dir.relative_to(VOLUME_ROOT)),
        "run_id": result.plan.run_id,
        "source_sha256": result.plan.source_sha256,
        "stages": list(result.stage_manifests),
        "status": "complete",
    }


def _download_run(remote_path: str, local_path: Path) -> None:
    from modal.types import FileEntryType

    if local_path.exists():
        from forge.experiment import verify_run_directory

        verify_run_directory(local_path)
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{local_path.name}.", dir=local_path.parent))
    prefix = remote_path.strip("/") + "/"
    try:
        entries = experiment_volume.listdir(remote_path, recursive=True)
        for entry in entries:
            if entry.type != FileEntryType.FILE:
                continue
            normalized = entry.path.lstrip("/")
            if not normalized.startswith(prefix):
                raise RuntimeError(f"Modal returned a file outside the run directory: {entry.path}")
            relative = PurePosixPath(normalized.removeprefix(prefix))
            destination = temporary.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_file = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                dir=destination.parent,
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    experiment_volume.read_file_into_fileobj(entry.path, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_file, destination)
            except BaseException:
                try:
                    os.unlink(temporary_file)
                except FileNotFoundError:
                    pass
                raise

        from forge.experiment import verify_run_directory

        verify_run_directory(temporary)
        os.rename(temporary, local_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


@app.local_entrypoint()
def main(
    experiment: str,
    profile: str,
    replicate: int = 0,
    device: str = "",
    resume: bool = False,
) -> None:
    """Upload verified pins, allocate the declared envelope, execute, and download."""

    from forge.experiment.modal import modal_request_plan

    spec_path = (LOCAL_REPO / experiment).resolve()
    request = modal_request_plan(
        LOCAL_REPO,
        spec_path,
        profile=profile,
        replicate=replicate,
        device=device or None,
    )
    remote_repo = PurePosixPath("jobs") / request["request_id"] / "repo"
    with experiment_volume.batch_upload(force=True) as batch:
        for relative, local in _upload_paths(spec_path).items():
            batch.put_file(local, remote_repo / relative)

    envelope = request["resource_envelope"]
    remote_function = execute_experiment.with_options(
        cpu=float(envelope["cpus"]),
        memory=int(envelope["memory_mb"]),
        gpu=envelope["gpu_type"],
        timeout=min(86_400, int(envelope["timeout_seconds"]) + 300),
    )
    result = remote_function.remote(
        request["request_id"],
        str(spec_path.relative_to(LOCAL_REPO)),
        profile,
        replicate,
        device or None,
        resume,
        request["source_sha256"],
    )
    experiment_volume.reload()
    local_run = LOCAL_REPO / "runs" / result["experiment_id"] / result["run_id"]
    _download_run(result["remote_run_path"], local_run)
    result["local_run_path"] = str(local_run)
    result["verification"] = "downloaded artifacts match remote manifests"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit("run this program with `modal run src/forge/experiment/modal_app.py`")
