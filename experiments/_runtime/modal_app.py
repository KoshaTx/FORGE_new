#!/usr/bin/env python3
"""Run any registered FORGE experiment DAG on Modal and download its verified artifacts."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal

IMAGE_PROJECT = Path("/opt/forge-project")
LOCAL_REPO = Path(__file__).resolve().parents[2] if modal.is_local() else IMAGE_PROJECT
VOLUME_ROOT = Path("/forge-workspace")
VOLUME_NAME = "forge-experiment-runs"

app = modal.App("forge-experiment-runner")
experiment_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Modal imports this entrypoint from its /root mount before the remote function body runs.
    # The project itself is copied into IMAGE_PROJECT, so make that copy importable during
    # function hydration as well as during experiment execution.
    .env({"PYTHONPATH": str(IMAGE_PROJECT)})
    .apt_install("libsm6", "libxext6", "libxrender1")
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
        str(LOCAL_REPO),
        extras=["oracle", "torch"],
        extra_options="--no-install-project",
        frozen=True,
    )
    .add_local_dir(
        LOCAL_REPO / "forge",
        remote_path=str(IMAGE_PROJECT / "forge"),
        copy=True,
    )
    .add_local_dir(
        LOCAL_REPO / "experiments" / "_runtime",
        remote_path=str(IMAGE_PROJECT / "experiments" / "_runtime"),
        copy=True,
    )
    .add_local_dir(
        LOCAL_REPO / "experiments" / "installation_smoke",
        remote_path=str(IMAGE_PROJECT / "experiments" / "installation_smoke"),
        copy=True,
    )
    .add_local_dir(
        LOCAL_REPO / "experiments" / "phase1",
        remote_path=str(IMAGE_PROJECT / "experiments" / "phase1"),
        copy=True,
    )
    .add_local_file(
        LOCAL_REPO / "experiments" / "__init__.py",
        remote_path=str(IMAGE_PROJECT / "experiments" / "__init__.py"),
        copy=True,
    )
    .add_local_file(
        LOCAL_REPO / "experiments" / "catalog.py",
        remote_path=str(IMAGE_PROJECT / "experiments" / "catalog.py"),
        copy=True,
    )
)


def _upload_paths(spec_path: Path) -> dict[str, Path]:
    from experiments._runtime.modal import modal_upload_paths

    return modal_upload_paths(LOCAL_REPO, spec_path)


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
    for relative in (
        "forge",
        "experiments/_runtime",
        "experiments/installation_smoke",
        "experiments/phase1",
    ):
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(IMAGE_PROJECT / relative, destination, dirs_exist_ok=True)
    for relative in ("experiments/__init__.py", "experiments/catalog.py"):
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(IMAGE_PROJECT / relative, destination)
    sys.path.insert(0, str(repo))

    from experiments import load_catalog
    from experiments._runtime import ExperimentRunner, ModalRuntimeBackend
    from experiments._runtime.modal import modal_volume_relative_path
    from experiments._runtime.source import source_fingerprint

    observed_source = source_fingerprint(repo)
    if observed_source != expected_source_sha256:
        raise RuntimeError(
            f"remote source changed: expected {expected_source_sha256}, found {observed_source}"
        )
    load_catalog()
    runner = ExperimentRunner(
        repo,
        backend=ModalRuntimeBackend(progress_commit=experiment_volume.commit),
    )
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
        "remote_run_path": modal_volume_relative_path(result.plan.run_dir, VOLUME_ROOT),
        "run_id": result.plan.run_id,
        "source_sha256": result.plan.source_sha256,
        "stages": list(result.stage_manifests),
        "status": "complete",
    }


def _download_run(remote_path: str, local_path: Path) -> None:
    from modal.types import FileEntryType

    if local_path.exists():
        from experiments._runtime import verify_run_directory

        verify_run_directory(local_path)
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    from experiments._runtime.modal import modal_run_staging_paths

    temporary_root, temporary = modal_run_staging_paths(local_path)
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

        from experiments._runtime import verify_run_directory

        verify_run_directory(temporary)
        os.rename(temporary, local_path)
        temporary_root.rmdir()
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


@app.local_entrypoint()
def main(
    experiment: str = "",
    profile: str = "",
    replicate: int = 0,
    device: str = "",
    resume: bool = False,
    launch_only: bool = False,
    call_receipt: str = "",
    status_only: bool = False,
) -> None:
    """Launch detached work, or inspect and collect a previously launched call."""

    from experiments._runtime.modal import (
        MODAL_CALL_RECEIPT_SCHEMA,
        modal_call_receipt_path,
        modal_request_plan,
    )
    from forge.core.io import read_json_object, write_json

    if call_receipt:
        if experiment or profile or launch_only or resume or device or replicate:
            raise RuntimeError("call-receipt mode cannot include launch arguments")
        receipt_path = (LOCAL_REPO / call_receipt).resolve()
        try:
            receipt_path.relative_to((LOCAL_REPO / "runs" / "_modal_calls").resolve())
        except ValueError as error:
            raise RuntimeError("Modal call receipt escapes runs/_modal_calls") from error
        receipt = read_json_object(
            receipt_path,
            error=RuntimeError,
            label="Modal experiment call receipt",
        )
        if (
            receipt.get("schema_version") != MODAL_CALL_RECEIPT_SCHEMA
            or receipt.get("status") != "launched"
            or not isinstance(receipt.get("function_call_id"), str)
        ):
            raise RuntimeError("invalid Modal experiment call receipt")
        function_call = modal.FunctionCall.from_id(receipt["function_call_id"])
        if status_only:
            try:
                result = function_call.get(timeout=0)
            except (modal.exception.TimeoutError, TimeoutError):
                print(
                    json.dumps(
                        {
                            "function_call_id": receipt["function_call_id"],
                            "request_id": receipt["request_id"],
                            "schema_version": "forge.modal_experiment_call_status.v1",
                            "status": "running",
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return
        else:
            result = function_call.get()
        for key in ("experiment_id", "source_sha256"):
            if result.get(key) != receipt.get(key):
                raise RuntimeError(f"detached Modal result changed {key}")
        local_run = LOCAL_REPO / "runs" / result["experiment_id"] / result["run_id"]
        if not status_only:
            _download_run(result["remote_run_path"], local_run)
            result["local_run_path"] = str(local_run)
            result["verification"] = "downloaded artifacts match remote manifests"
        result["function_call_id"] = receipt["function_call_id"]
        result["request_id"] = receipt["request_id"]
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if status_only:
        raise RuntimeError("--status-only requires --call-receipt")
    if not experiment or not profile:
        raise RuntimeError("launch mode requires --experiment and --profile")

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
    arguments = (
        request["request_id"],
        str(spec_path.relative_to(LOCAL_REPO)),
        profile,
        replicate,
        device or None,
        resume,
        request["source_sha256"],
    )
    if launch_only:
        receipt_path = modal_call_receipt_path(LOCAL_REPO, request["request_id"])
        if receipt_path.exists():
            raise RuntimeError(
                "detached Modal call receipt already exists; inspect or collect it instead of "
                f"launching a duplicate: {receipt_path}"
            )
        function_call = remote_function.spawn(*arguments)
        receipt = {
            "device": device or None,
            "experiment": str(spec_path.relative_to(LOCAL_REPO)),
            "experiment_id": request["experiment_id"],
            "function_call_id": function_call.object_id,
            "launched_utc": datetime.now(timezone.utc).isoformat(),
            "profile": profile,
            "replicate": replicate,
            "request_id": request["request_id"],
            "resource_envelope": request["resource_envelope"],
            "resume": resume,
            "schema_version": MODAL_CALL_RECEIPT_SCHEMA,
            "source_sha256": request["source_sha256"],
            "spec_sha256": request["spec_sha256"],
            "status": "launched",
            "uploads": request["uploads"],
        }
        write_json(receipt_path, receipt)
        receipt["receipt_path"] = str(receipt_path)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    result = remote_function.remote(*arguments)
    local_run = LOCAL_REPO / "runs" / result["experiment_id"] / result["run_id"]
    _download_run(result["remote_run_path"], local_run)
    result["local_run_path"] = str(local_run)
    result["verification"] = "downloaded artifacts match remote manifests"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit("run this program with `modal run experiments/_runtime/modal_app.py`")
