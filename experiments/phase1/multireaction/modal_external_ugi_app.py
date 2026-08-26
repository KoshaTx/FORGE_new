#!/usr/bin/env python3
"""Run one pinned external Ugi baseline request on an exact Modal H100.

Each upstream method receives its own dependency image. Requests and clean checkouts are uploaded to
a content-addressed volume workspace, and every native attempt plus its receipt is downloaded only
after the strict runtime contract passes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import modal

IMAGE_PROJECT = Path("/opt/forge-project")
VOLUME_ROOT = Path("/forge-external-ugi")
VOLUME_NAME = "forge-external-ugi-baselines"
LOCAL_REPO = Path(__file__).resolve().parents[3] if modal.is_local() else IMAGE_PROJECT

app = modal.App("forge-external-ugi-baselines")
artifact_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _forge_image(python_version: str = "3.11") -> modal.Image:
    return (
        modal.Image.debian_slim(python_version=python_version)
        .env({"PYTHONPATH": str(IMAGE_PROJECT)})
        .apt_install("git", "libsm6", "libxext6", "libxrender1")
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
            extras=["torch"],
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
        .add_local_file(
            LOCAL_REPO / "configs" / "baselines" / "external_ugi_v1.json",
            remote_path=str(IMAGE_PROJECT / "configs" / "baselines" / "external_ugi_v1.json"),
            copy=True,
        )
        # The pinned native releases still import pkg_resources. Modern minimal Python images do
        # not include it, so install the final setuptools line that retains the compatibility API.
        .uv_pip_install("setuptools==80.9.0")
    )


rgfn_image = _forge_image().uv_pip_install(
    "dgl==1.1.3",
    "dgllife==0.3.2",
    "gin-config==0.5.0",
    "hydra-core==1.3.2",
    "more-itertools==10.1.0",
    "openbabel-wheel==3.1.1.19",
    "openpyxl==3.1.4",
    "pandas==2.1.4",
    "pydantic==2.6.3",
    "pytdc==1.0.6",
    "meeko==0.5.1",
    "torch-geometric==2.5.3",
    "torchmetrics==1.2.0",
    "wandb==0.15.12",
    "wurlitzer==3.1.0",
    "xlsxwriter==3.2.0",
)

defog_image = _forge_image().uv_pip_install(
    "hydra-core==1.3.2",
    "imageio==2.31.1",
    "matplotlib==3.7.1",
    "networkx==2.8.7",
    "omegaconf==2.3.0",
    "overrides==7.3.1",
    "pandas==1.5.3",
    "pyemd==1.0.0",
    "PyGSP==0.5.1",
    "pytorch-lightning==2.0.4",
    "scipy==1.11.0",
    "seaborn==0.12.2",
    "torch-geometric==2.3.1",
    "torchmetrics==0.11.4",
    "wandb==0.15.4",
)

genmol_image = _forge_image().uv_pip_install(
    "bionemo-moco==0.0.2.1",
    "datasets==2.18.0",
    "easydict==1.13",
    "einops==0.7.0",
    "hydra-core==1.3.2",
    "jaxtyping>=0.2.34",
    "lightning==2.5.1",
    "matplotlib>=3.3.2",
    "numpy==1.26.4",
    "openbabel-wheel==3.1.1.22",
    "pandas==2.1.0",
    "pot>=0.9.5",
    "pytdc==0.4.1",
    "safe-mol==0.1.14",
    "scikit-learn>=1.2.2",
    "transformers==4.52.4",
    "torch==2.6.0",
    "wandb==0.13.5",
)


def _execute(job_key: str, method_id: str, expected_source_sha256: str) -> dict[str, Any]:
    import torch

    from experiments._runtime.source import source_fingerprint
    from experiments.phase1.multireaction.native_baseline_runtime import run_native_baseline
    from forge.core.hashing import sha256_file
    from forge.core.io import write_json

    observed_source_sha256 = source_fingerprint(IMAGE_PROJECT)
    if observed_source_sha256 != expected_source_sha256:
        raise RuntimeError(
            "remote external-baseline source changed: "
            f"expected {expected_source_sha256}, found {observed_source_sha256}"
        )

    workspace = VOLUME_ROOT / "jobs" / job_key
    request_path = workspace / "request" / "request.json"
    checkout = workspace / "checkout"
    output = workspace / "output"
    sys.path.insert(0, str(checkout))
    result = run_native_baseline(
        request_path,
        checkout,
        output,
        manifest_path=IMAGE_PROJECT / "configs" / "baselines" / "external_ugi_v1.json",
    )
    accelerator = torch.cuda.get_device_name(0)
    write_json(
        output / "modal_execution.json",
        {
            "schema_version": "forge.external_ugi_modal_execution.v1",
            "status": "complete",
            "method_id": method_id,
            "job_key": job_key,
            "source_sha256": observed_source_sha256,
            "request_sha256": str(sha256_file(request_path)),
            "upstream_commit": str(result["checkout"]["commit"]),
            "accelerator": accelerator,
        },
    )
    artifact_volume.commit()
    return {
        "status": "complete",
        "method_id": method_id,
        "job_key": job_key,
        "remote_output": str(PurePosixPath("jobs") / job_key / "output"),
        "accelerator": accelerator,
        "source_sha256": observed_source_sha256,
        "result": result,
    }


_FUNCTION_OPTIONS = {
    "cpu": 8.0,
    "memory": 65536,
    "gpu": "H100!",
    "timeout": 86_400,
    "retries": 0,
    "volumes": {str(VOLUME_ROOT): artifact_volume},
}


@app.function(image=rgfn_image, **_FUNCTION_OPTIONS)
def execute_rgfn(job_key: str, expected_source_sha256: str) -> dict[str, Any]:
    return _execute(job_key, "rgfn", expected_source_sha256)


@app.function(image=defog_image, **_FUNCTION_OPTIONS)
def execute_defog(job_key: str, expected_source_sha256: str) -> dict[str, Any]:
    return _execute(job_key, "defog_unconditional", expected_source_sha256)


@app.function(image=genmol_image, **_FUNCTION_OPTIONS)
def execute_genmol(job_key: str, expected_source_sha256: str) -> dict[str, Any]:
    return _execute(job_key, "genmol_safe", expected_source_sha256)


def _download_directory(remote_path: str, local_path: Path) -> None:
    from modal.types import FileEntryType

    if local_path.exists() and any(local_path.iterdir()):
        raise RuntimeError(f"external native output is not empty: {local_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{local_path.name}.", dir=local_path.parent))
    prefix = remote_path.strip("/") + "/"
    try:
        for entry in artifact_volume.listdir(remote_path, recursive=True):
            if entry.type != FileEntryType.FILE:
                continue
            normalized = entry.path.lstrip("/")
            if not normalized.startswith(prefix):
                raise RuntimeError(f"Modal returned an out-of-scope file: {entry.path}")
            relative = PurePosixPath(normalized.removeprefix(prefix))
            destination = staging.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                artifact_volume.read_file_into_fileobj(entry.path, handle)
                handle.flush()
                os.fsync(handle.fileno())
        if not (staging / "result.json").is_file() or not (staging / "receipt.json").is_file():
            raise RuntimeError("external native output is incomplete")
        if local_path.exists():
            local_path.rmdir()
        os.rename(staging, local_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


@app.local_entrypoint()
def main(
    method: str,
    request: str,
    checkout: str,
    output: str,
    launch_only: bool = False,
    resume: str = "",
) -> None:
    from experiments._runtime.source import source_fingerprint
    from forge.core.hashing import sha256_file, sha256_json
    from forge.core.io import read_json_object, write_json

    request_path = Path(request).resolve()
    checkout_path = Path(checkout).resolve()
    output_path = Path(output).resolve()
    payload = read_json_object(request_path, error=RuntimeError, label="external Ugi request")
    observed_method = str(payload.get("method", {}).get("method_id", ""))
    if method != observed_method or method not in {"rgfn", "defog_unconditional", "genmol_safe"}:
        raise RuntimeError(f"external method/request mismatch: {method!r} != {observed_method!r}")
    if output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError(f"external native output is not empty: {output_path}")
    git_head = subprocess.run(
        ["git", "-C", str(checkout_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_commit = str(payload.get("checkout", {}).get("commit", ""))
    if git_head != expected_commit:
        raise RuntimeError(
            f"external checkout changed: expected {expected_commit}, found {git_head}"
        )
    source_sha256 = source_fingerprint(LOCAL_REPO)
    job_key = str(
        sha256_json(
            {
                "method_id": method,
                "request_sha256": str(sha256_file(request_path)),
                "upstream_commit": git_head,
                "source_sha256": source_sha256,
            }
        )
    )
    function = {
        "rgfn": execute_rgfn,
        "defog_unconditional": execute_defog,
        "genmol_safe": execute_genmol,
    }[method]
    if resume:
        if launch_only:
            raise RuntimeError("--launch-only and --resume are mutually exclusive")
        call_receipt_path = Path(resume).resolve()
        call_receipt = read_json_object(
            call_receipt_path,
            error=RuntimeError,
            label="external Ugi Modal call receipt",
        )
        expected_call_receipt = {
            "schema_version": "forge.external_ugi_modal_call.v1",
            "status": "launched",
            "method_id": method,
            "job_key": job_key,
            "source_sha256": source_sha256,
            "request_sha256": str(sha256_file(request_path)),
            "upstream_commit": git_head,
            "output": str(output_path),
        }
        if (
            not isinstance(call_receipt.get("function_call_id"), str)
            or {key: call_receipt.get(key) for key in expected_call_receipt}
            != expected_call_receipt
        ):
            raise RuntimeError("external Ugi Modal call receipt no longer matches this job")
        function_call = modal.FunctionCall.from_id(call_receipt["function_call_id"])
    else:
        call_receipt_path = output_path.parent / f"{output_path.name}.modal_call.json"
        if call_receipt_path.exists():
            raise RuntimeError(
                "external Ugi Modal call receipt already exists; resume it instead of "
                f"launching a duplicate: {call_receipt_path}"
            )
        remote_root = PurePosixPath("jobs") / job_key
        with artifact_volume.batch_upload(force=True) as batch:
            batch.put_directory(request_path.parent, remote_root / "request")
            batch.put_directory(checkout_path, remote_root / "checkout")
        function_call = function.spawn(job_key, source_sha256)
        call_receipt = {
            "schema_version": "forge.external_ugi_modal_call.v1",
            "status": "launched",
            "method_id": method,
            "job_key": job_key,
            "source_sha256": source_sha256,
            "request_sha256": str(sha256_file(request_path)),
            "upstream_commit": git_head,
            "function_call_id": function_call.object_id,
            "output": str(output_path),
        }
        write_json(call_receipt_path, call_receipt)
        print(json.dumps(call_receipt, indent=2, sort_keys=True))
        if launch_only:
            return

    result = function_call.get()
    _download_directory(result["remote_output"], output_path)
    result["local_output"] = str(output_path)
    result["call_receipt"] = str(call_receipt_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(
        "run with `modal run experiments/phase1/multireaction/modal_external_ugi_app.py`"
    )
