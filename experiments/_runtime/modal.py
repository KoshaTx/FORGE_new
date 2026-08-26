"""Local planning and launch utilities for the generic Modal runner."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from experiments._runtime.errors import BackendError
from experiments._runtime.source import source_fingerprint
from experiments._runtime.spec import ExperimentSpec
from forge.core.hashing import sha256_file, sha256_json


def modal_volume_relative_path(path: Path, volume_root: Path) -> str:
    """Return a stable logical path even when Modal resolves a mounted volume symlink."""

    try:
        relative = path.resolve().relative_to(volume_root.resolve())
    except ValueError as exc:
        raise BackendError(f"run path {path} is outside Modal volume {volume_root}") from exc
    if not relative.parts:
        raise BackendError("run path resolves to the Modal volume root, not a run directory")
    return relative.as_posix()


def modal_run_staging_paths(local_run: Path) -> tuple[Path, Path]:
    """Create a private staging root containing a child named as the final run id."""

    staging_root = Path(tempfile.mkdtemp(prefix=f".{local_run.name}.", dir=local_run.parent))
    staged_run = staging_root / local_run.name
    staged_run.mkdir()
    return staging_root, staged_run


def modal_request_plan(
    repo: Path,
    spec_path: Path,
    *,
    profile: str,
    replicate: int,
    device: str | None,
) -> dict[str, Any]:
    """Validate a remote request and describe what will be uploaded and allocated."""

    spec = ExperimentSpec.load(spec_path)
    if profile not in spec.profiles:
        raise BackendError(f"profile {profile!r} is not declared by {spec.experiment_id!r}")
    if not 0 <= replicate < spec.replicates[profile]:
        raise BackendError(
            f"replicate {replicate} is outside [0, {spec.replicates[profile]}) for {profile!r}"
        )
    resources = [
        stage.resources if device is None else stage.resources.with_device(device)
        for stage in spec.stages
    ]
    if any(resource.device == "mps" for resource in resources):
        raise BackendError("Modal does not provide Apple MPS devices")
    gpu_types = {resource.gpu_type for resource in resources if resource.device == "cuda"}
    if None in gpu_types or len(gpu_types) > 1:
        raise BackendError("all CUDA stages in one Modal DAG must declare the same gpu_type")

    uploads: dict[str, Path] = {str(spec_path.relative_to(repo)): spec_path}
    for stage in spec.stages:
        for pin in (stage.config, *stage.inputs.values()):
            uploads[pin.path] = pin.resolve(repo)
    for relative in ("pyproject.toml", "uv.lock"):
        path = repo / relative
        if not path.is_file():
            raise BackendError(f"Modal execution requires {relative}")
        uploads[relative] = path
    source_sha256 = source_fingerprint(repo)
    request_id = str(
        sha256_json(
            {
                "device": device,
                "profile": profile,
                "replicate": replicate,
                "source_sha256": source_sha256,
                "spec_sha256": str(sha256_file(spec_path)),
                "uploads": {
                    relative: str(sha256_file(path)) for relative, path in sorted(uploads.items())
                },
            }
        )
    )
    return {
        "backend": "modal",
        "experiment_id": spec.experiment_id,
        "profile": profile,
        "replicate": replicate,
        "request_id": request_id,
        "resource_envelope": {
            "cpus": max(resource.cpus for resource in resources),
            "gpu_type": next(iter(gpu_types), None),
            "memory_mb": max(resource.memory_mb for resource in resources),
            "timeout_seconds": sum(resource.timeout_seconds for resource in resources),
        },
        "run_id": None,
        "source_sha256": source_sha256,
        "uploads": {
            relative: {
                "bytes": path.stat().st_size,
                "sha256": str(sha256_file(path)),
            }
            for relative, path in sorted(uploads.items())
        },
        "why_run_id_is_pending": (
            "The run id includes the actual remote Python, package, CUDA, and accelerator identity."
        ),
    }


def launch_modal(
    repo: Path,
    spec_path: Path,
    *,
    profile: str,
    replicate: int,
    device: str | None,
    resume: bool,
) -> int:
    """Invoke the one generic Modal program; never fall back to local execution."""

    modal_request_plan(
        repo,
        spec_path,
        profile=profile,
        replicate=replicate,
        device=device,
    )
    command = [
        sys.executable,
        "-m",
        "modal",
        "run",
        str(repo / "experiments" / "_runtime" / "modal_app.py"),
        "--experiment",
        str(spec_path.relative_to(repo)),
        "--profile",
        profile,
        "--replicate",
        str(replicate),
    ]
    if device is not None:
        command.extend(("--device", device))
    if resume:
        command.append("--resume")
    completed = subprocess.run(command, cwd=repo, check=False)
    return int(completed.returncode)


__all__ = [
    "launch_modal",
    "modal_request_plan",
    "modal_run_staging_paths",
    "modal_volume_relative_path",
]
