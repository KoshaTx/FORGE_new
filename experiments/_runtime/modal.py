"""Local planning and launch utilities for the generic Modal runner."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments._runtime.errors import BackendError
from experiments._runtime.source import source_fingerprint
from experiments._runtime.spec import ExperimentSpec
from forge.core.hashing import PinError, resolve_pin, sha256_file, sha256_json
from forge.core.io import read_json_object

MODAL_CALL_RECEIPT_SCHEMA = "forge.modal_experiment_call.v1"


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


def modal_call_receipt_path(repo: Path, request_id: str) -> Path:
    """Return the deterministic local receipt path for one detached request."""

    if len(request_id) != 64 or any(
        character not in "0123456789abcdef" for character in request_id
    ):
        raise BackendError(f"invalid Modal request id: {request_id!r}")
    return repo.resolve() / "runs" / "_modal_calls" / f"{request_id}.json"


def modal_restart_receipt_path(repo: Path, plan: Mapping[str, Any], previous: Path) -> Path:
    """Bind a new attempt to the unchanged request and preserve its parent receipt."""
    previous = previous.resolve()
    previous.relative_to((repo / "runs" / "_modal_calls").resolve())
    receipt = read_json_object(previous, error=BackendError)
    for key in ("request_id", "source_sha256", "spec_sha256", "uploads"):
        if receipt.get(key) != plan[key]:
            raise BackendError(f"Restart changed {key}; use the original source and inputs")
    if receipt.get("status") != "launched" or not receipt.get("function_call_id"):
        raise BackendError("Restart requires a persisted launched-call receipt")
    token = str(sha256_json({"parent": str(sha256_file(previous)), "request": plan["request_id"]}))
    return modal_call_receipt_path(repo, plan["request_id"]).with_name(
        f"{plan['request_id']}-restart-{token[:16]}.json"
    )


def require_terminal_modal_call(call: Any) -> str:
    """Unknown, running and successful calls must never start paid recovery."""
    roots = call.get_call_graph()
    if len(roots) != 1 or roots[0].status.name not in {
        "FAILURE",
        "INIT_FAILURE",
        "TERMINATED",
        "TIMEOUT",
    }:
        raise BackendError("Prior Modal call is not confirmed failed, terminated or timed out")
    return str(roots[0].status.name)


def _config_dependency_uploads(repo: Path, config_path: Path) -> dict[str, Path]:
    """Resolve the recursive ``inputs`` closure of one hash-pinned JSON configuration.

    Experiment specifications pin their stage configs and immediate data inputs.  Assessment
    configs may themselves compose other pinned configs and data.  Modal must upload that closure,
    while the remote assessment code remains responsible for independently verifying every pin.
    Only top-level ``inputs`` mappings are traversed, so arbitrary result/provenance records are
    never mistaken for execution dependencies.
    """

    repo = repo.resolve()
    uploads: dict[str, Path] = {}
    visited: set[Path] = set()

    def visit(path: Path, *, depth: int) -> None:
        resolved = path.resolve()
        if resolved in visited or resolved.suffix.lower() != ".json":
            return
        visited.add(resolved)
        value = read_json_object(
            resolved,
            error=BackendError,
            label=f"Modal configuration dependency {resolved.relative_to(repo)}",
        )
        inputs = value.get("inputs")
        if inputs is None:
            return
        if not isinstance(inputs, Mapping):
            raise BackendError(
                f"Modal configuration inputs must be an object: {resolved.relative_to(repo)}"
            )
        for label, pin in sorted(inputs.items()):
            # Some historical documents use an ``inputs`` object for metadata-bearing evidence
            # records.  They are not ``resolve_pin`` contracts and are loaded only by their owning
            # workflow.  Traverse the runtime's exact two-field pin type and leave all other
            # records alone.
            if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256"}:
                continue
            try:
                dependency = resolve_pin(
                    pin,
                    repo,
                    label=f"{resolved.relative_to(repo)}:{label}",
                )
            except PinError as error:
                raise BackendError(str(error)) from error
            relative = dependency.relative_to(repo).as_posix()
            uploads[relative] = dependency
            # Active experiment configs compose at most two configuration layers below the stage
            # config.  At that boundary, upload every declared input but do not recursively walk
            # into data registries that happen to be JSON and carry their own archival ``inputs``.
            if depth < 2:
                visit(dependency, depth=depth + 1)

    visit(config_path, depth=0)
    return uploads


def modal_upload_paths(repo: Path, spec_path: Path) -> dict[str, Path]:
    """Return the complete authenticated file set required by one remote experiment."""

    repo = repo.resolve()
    spec_path = spec_path.resolve()
    spec = ExperimentSpec.load(spec_path)
    uploads: dict[str, Path] = {str(spec_path.relative_to(repo)): spec_path}
    for stage in spec.stages:
        config_path = stage.config.resolve(repo)
        uploads[stage.config.path] = config_path
        uploads.update(_config_dependency_uploads(repo, config_path))
        for pin in stage.inputs.values():
            uploads[pin.path] = pin.resolve(repo)
    for relative in ("pyproject.toml", "uv.lock"):
        path = repo / relative
        if not path.is_file():
            raise BackendError(f"Modal execution requires {relative}")
        uploads[relative] = path
    return dict(sorted(uploads.items()))


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

    uploads = modal_upload_paths(repo, spec_path)
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
        "spec_sha256": str(sha256_file(spec_path)),
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
    detached: bool = False,
    restart_from: Path | None = None,
    diagnosis: str = "",
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
    ]
    if detached:
        command.append("--detach")
    command.extend(
        [
            str(repo / "experiments" / "_runtime" / "modal_app.py"),
            "--experiment",
            str(spec_path.relative_to(repo)),
            "--profile",
            profile,
            "--replicate",
            str(replicate),
        ]
    )
    if device is not None:
        command.extend(("--device", device))
    if resume:
        command.append("--resume")
    if restart_from is not None:
        if not resume or not detached or not diagnosis.strip():
            raise BackendError("Explicit recovery requires detached resume and a diagnosis")
        command.extend(("--restart-from", str(restart_from), "--diagnosis", diagnosis))
    if detached:
        command.append("--launch-only")
    completed = subprocess.run(command, cwd=repo, check=False)
    return int(completed.returncode)


def inspect_or_collect_modal(repo: Path, receipt_path: Path, *, status_only: bool) -> int:
    """Poll or collect an already detached call without submitting new compute."""

    repo = repo.resolve()
    receipt_path = receipt_path.resolve()
    try:
        relative = receipt_path.relative_to(repo)
    except ValueError as error:
        raise BackendError(
            f"Modal call receipt is outside the repository: {receipt_path}"
        ) from error
    receipt = read_json_object(
        receipt_path,
        error=BackendError,
        label="Modal experiment call receipt",
    )
    if (
        receipt.get("schema_version") != MODAL_CALL_RECEIPT_SCHEMA
        or receipt.get("status") != "launched"
        or not isinstance(receipt.get("function_call_id"), str)
        or not isinstance(receipt.get("request_id"), str)
    ):
        raise BackendError(f"invalid Modal experiment call receipt: {receipt_path}")
    command = [
        sys.executable,
        "-m",
        "modal",
        "run",
        str(repo / "experiments" / "_runtime" / "modal_app.py"),
        "--call-receipt",
        str(relative),
    ]
    if status_only:
        command.append("--status-only")
    completed = subprocess.run(command, cwd=repo, check=False)
    return int(completed.returncode)


__all__ = [
    "MODAL_CALL_RECEIPT_SCHEMA",
    "inspect_or_collect_modal",
    "launch_modal",
    "modal_call_receipt_path",
    "modal_request_plan",
    "modal_run_staging_paths",
    "modal_upload_paths",
    "modal_volume_relative_path",
]
