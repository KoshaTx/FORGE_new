"""Fail-closed qualification for the isolated Graph2Edits runtime.

This module is intentionally separate from proposal and synthesis-value logic.
It authenticates one local runtime, loads the frozen checkpoint, and may run one
explicitly nonbenchmark smoke target.  Passing this qualification never
activates the backend, creates route evidence, or authorizes benchmark access.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RUNTIME_RECEIPT_SCHEMA_VERSION = "forge.graph2edits_runtime_qualification.v1"
RUNTIME_PLATFORM_ID = "python311-macos-arm64-cpu"
NONBENCHMARK_SMOKE_SMILES = "CCOC(=O)C"

_LOCK_RECORD = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)")
_REQUIRED_DIRECT_PINS = {
    "numpy": "1.26.4",
    "omegaconf": "2.3.0",
    "rdkit": "2024.3.6",
    "torch": "2.2.2",
}
_REQUIRED_LOCAL_WHEELS = {
    "syntheseus": "0.8.0",
    "syntheseus-graph2edits": "0.2.0",
}


class Graph2EditsRuntimeQualificationError(RuntimeError):
    """Raised when the isolated runtime cannot be authenticated or loaded."""


def stable_json(value: Any) -> str:
    """Return the canonical JSON representation used for receipt hashes."""

    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash one file without loading large checkpoints into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def parse_hash_lock(path: Path) -> dict[str, str]:
    """Parse exact package versions from a pip-style hash lock."""

    versions: dict[str, str] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise Graph2EditsRuntimeQualificationError(f"could not read runtime lock: {path}") from exc
    for line in lines:
        match = _LOCK_RECORD.match(line)
        if match is None:
            continue
        name = _normalized_distribution_name(match.group(1))
        version = match.group(2)
        previous = versions.setdefault(name, version)
        if previous != version:
            raise Graph2EditsRuntimeQualificationError(
                f"runtime lock contains conflicting versions for {name}"
            )
    if not versions:
        raise Graph2EditsRuntimeQualificationError("runtime lock contains no exact package pins")
    return dict(sorted(versions.items()))


def _artifact_record(path: Path, *, repo_root: Path) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise Graph2EditsRuntimeQualificationError(
            f"runtime artifact lies outside the repository: {path}"
        ) from exc
    if not path.is_file():
        raise Graph2EditsRuntimeQualificationError(f"runtime artifact is missing: {path}")
    return {
        "path": relative.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _require_exact_pin(versions: Mapping[str, str], name: str, expected: str) -> None:
    observed = versions.get(_normalized_distribution_name(name))
    if observed != expected:
        raise Graph2EditsRuntimeQualificationError(
            f"{name} version mismatch: expected {expected}, observed {observed}"
        )


def _installed_versions(expected: Mapping[str, str]) -> dict[str, str]:
    installed: dict[str, str] = {}
    for name, version in sorted(expected.items()):
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise Graph2EditsRuntimeQualificationError(
                f"runtime distribution is missing: {name}"
            ) from exc
        if observed != version:
            raise Graph2EditsRuntimeQualificationError(
                f"installed {name} version mismatch: expected {version}, observed {observed}"
            )
        installed[name] = observed
    return installed


def _pip_check() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PIP_NO_CACHE_DIR"] = "1"
    process = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if process.returncode != 0:
        detail = (process.stdout + process.stderr).strip()
        raise Graph2EditsRuntimeQualificationError(f"pip check failed: {detail}")
    return {
        "passed": True,
        "result": process.stdout.strip(),
    }


def _prediction_record(reaction: Any) -> dict[str, Any]:
    probability = reaction.metadata.get("probability")
    return {
        "reactants": sorted(molecule.smiles for molecule in reaction.reactants),
        "opaque_model_score": probability,
    }


def _run_checkpoint_and_smoke(checkpoint_dir: Path) -> dict[str, Any]:
    """Load the checkpoint and run one hand-authored target twice."""

    try:
        import numpy
        import omegaconf
        import rdkit
        import torch
        from syntheseus import Molecule
        from syntheseus.reaction_prediction.inference import Graph2EditsModel
    except (ImportError, OSError) as exc:
        raise Graph2EditsRuntimeQualificationError(
            "Graph2Edits runtime imports could not be loaded"
        ) from exc

    torch.manual_seed(0)
    numpy.random.seed(0)
    model = Graph2EditsModel(
        model_dir=checkpoint_dir,
        device="cpu",
        max_edit_steps=9,
        use_cache=False,
    )
    parameter_count = sum(parameter.numel() for parameter in model.get_parameters())
    target = Molecule(NONBENCHMARK_SMOKE_SMILES)
    repeated_outputs: list[list[dict[str, Any]]] = []
    for _ in range(2):
        reactions = model([target], num_results=3)[0]
        repeated_outputs.append([_prediction_record(reaction) for reaction in reactions])
    deterministic = stable_json(repeated_outputs[0]) == stable_json(repeated_outputs[1])
    if not deterministic:
        raise Graph2EditsRuntimeQualificationError(
            "repeated nonbenchmark smoke inference was not deterministic"
        )
    return {
        "imports": {
            "numpy": numpy.__version__,
            "omegaconf": omegaconf.__version__,
            "rdkit": rdkit.__version__,
            "torch": torch.__version__,
        },
        "checkpoint_load": {
            "passed": True,
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "parameter_count": parameter_count,
            "device": "cpu",
            "max_edit_steps": 9,
        },
        "smoke_inference": {
            "executed": True,
            "source": "hand-authored_nonbenchmark_runtime_smoke",
            "target_smiles": NONBENCHMARK_SMOKE_SMILES,
            "num_results_requested": 3,
            "repeat_count": 2,
            "byte_identical_normalized_outputs": True,
            "outputs": repeated_outputs[0],
            "scientific_interpretation": "runtime_only_no_route_evidence_or_performance_claim",
        },
    }


def build_runtime_receipt(
    *,
    repo_root: Path,
    candidate_config_path: Path,
    requirements_input_path: Path,
    runtime_lock_path: Path,
) -> dict[str, Any]:
    """Build a deterministic, inactive local runtime qualification receipt."""

    if sys.version_info[:2] != (3, 11):
        raise Graph2EditsRuntimeQualificationError(
            f"qualification requires Python 3.11, observed {platform.python_version()}"
        )
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise Graph2EditsRuntimeQualificationError(
            "this receipt is scoped to native macOS arm64 only"
        )

    try:
        candidate = json.loads(candidate_config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Graph2EditsRuntimeQualificationError(
            f"invalid Graph2Edits candidate config: {candidate_config_path}"
        ) from exc
    if candidate.get("activation_authorized") is not False:
        raise Graph2EditsRuntimeQualificationError("candidate backend must remain inactive")

    locked_versions = parse_hash_lock(runtime_lock_path)
    for name, version in _REQUIRED_DIRECT_PINS.items():
        _require_exact_pin(locked_versions, name, version)
    expected_versions = {**locked_versions, **_REQUIRED_LOCAL_WHEELS}
    installed_versions = _installed_versions(expected_versions)
    pip_check = _pip_check()

    checkpoint_record = candidate["checkpoint_release"]
    runtime_record = candidate["runtime"]
    checkpoint_path = repo_root / checkpoint_record["checkpoint_path"]
    syntheseus_wheel_path = repo_root / runtime_record["syntheseus_wheel_path"]
    graph2edits_wheel_path = repo_root / runtime_record["syntheseus_graph2edits_wheel_path"]
    artifacts = {
        "candidate_config": _artifact_record(candidate_config_path, repo_root=repo_root),
        "checkpoint": _artifact_record(checkpoint_path, repo_root=repo_root),
        "qualification_module": _artifact_record(Path(__file__), repo_root=repo_root),
        "qualification_script": _artifact_record(
            repo_root / "scripts/qualify_graph2edits_runtime.py", repo_root=repo_root
        ),
        "requirements_input": _artifact_record(requirements_input_path, repo_root=repo_root),
        "runtime_lock": _artifact_record(runtime_lock_path, repo_root=repo_root),
        "syntheseus_graph2edits_wheel": _artifact_record(
            graph2edits_wheel_path, repo_root=repo_root
        ),
        "syntheseus_wheel": _artifact_record(syntheseus_wheel_path, repo_root=repo_root),
    }

    expected_artifact_values = {
        "checkpoint": (
            checkpoint_record["checkpoint_size_bytes"],
            checkpoint_record["checkpoint_sha256"],
        ),
        "syntheseus_wheel": (
            runtime_record["syntheseus_wheel_size_bytes"],
            runtime_record["syntheseus_wheel_sha256"],
        ),
        "syntheseus_graph2edits_wheel": (
            runtime_record["syntheseus_graph2edits_wheel_size_bytes"],
            runtime_record["syntheseus_graph2edits_wheel_sha256"],
        ),
    }
    for label, (expected_size, expected_sha256) in expected_artifact_values.items():
        observed = artifacts[label]
        if observed["size_bytes"] != expected_size or observed["sha256"] != expected_sha256:
            raise Graph2EditsRuntimeQualificationError(
                f"candidate config and local {label} receipt disagree"
            )

    model_result = _run_checkpoint_and_smoke(checkpoint_path.parent)
    if model_result["imports"] != {
        "numpy": "1.26.4",
        "omegaconf": "2.3.0",
        "rdkit": "2024.03.6",
        "torch": "2.2.2",
    }:
        raise Graph2EditsRuntimeQualificationError("imported runtime versions are not frozen")

    payload: dict[str, Any] = {
        "schema_version": RUNTIME_RECEIPT_SCHEMA_VERSION,
        "status": "local_cpu_import_checkpoint_and_nonbenchmark_smoke_passed",
        "activation_authorized": False,
        "production_promotion_authorized": False,
        "scientific_authority": {
            "proposal_only": True,
            "may_create_route_evidence": False,
            "may_enter_synthesis_value": False,
            "may_close_route": False,
            "may_support_performance_claim": False,
        },
        "scope_guards": {
            "frozen_120_target_benchmark_executed": False,
            "sealed_holdouts_accessed": False,
            "proposal_or_value_logic_modified": False,
            "candidate_backend_activated": False,
            "performance_metrics_computed": False,
        },
        "platform": {
            "runtime_id": RUNTIME_PLATFORM_ID,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "macos_version": platform.mac_ver()[0],
            "device": "cpu",
            "environment_locator": ".venv-graph2edits-py311-arm64",
            "linux_gpu_qualified": False,
        },
        "dependency_resolution": {
            "locked_versions": locked_versions,
            "local_wheel_versions": dict(sorted(_REQUIRED_LOCAL_WHEELS.items())),
            "installed_versions": installed_versions,
            "installed_versions_sha256": _sha256_payload(installed_versions),
            "pip_check": pip_check,
            "compatibility_decisions": {
                "torch": "2.2.2_from_Syntheseus_documented_environment",
                "numpy": "1.26.4_final_1x_to_avoid_numpy2_abi_transition",
                "rdkit": "2024.3.6_python311_macos_arm64_wheel",
                "omegaconf": "2.3.0_replaces_metadata_incompatible_2.0.6",
                "antlr4-python3-runtime": (
                    "4.9.3_hash_pinned_pure_python_sdist_required_by_omegaconf_2.3.0"
                ),
            },
        },
        "artifacts": artifacts,
        **model_result,
        "remaining_gates": [
            "institutional artifact and patent-derived-data license review",
            "frozen 120-target benchmark execution under separate authorization",
            "production-device repeated-inference qualification",
        ],
    }
    payload["payload_sha256"] = _sha256_payload(payload)
    return payload


def verify_runtime_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    verify_artifacts: bool = True,
    verify_environment: bool = False,
) -> None:
    """Verify receipt integrity and optionally its files/current environment."""

    if receipt.get("schema_version") != RUNTIME_RECEIPT_SCHEMA_VERSION:
        raise Graph2EditsRuntimeQualificationError("unexpected runtime receipt schema")
    stored_hash = receipt.get("payload_sha256")
    unhashed = dict(receipt)
    unhashed.pop("payload_sha256", None)
    if not isinstance(stored_hash, str) or stored_hash != _sha256_payload(unhashed):
        raise Graph2EditsRuntimeQualificationError("runtime receipt payload hash mismatch")
    if receipt.get("activation_authorized") is not False:
        raise Graph2EditsRuntimeQualificationError("runtime receipt cannot activate the backend")
    if receipt.get("production_promotion_authorized") is not False:
        raise Graph2EditsRuntimeQualificationError("runtime receipt cannot promote the backend")
    guards = receipt.get("scope_guards")
    if not isinstance(guards, Mapping):
        raise Graph2EditsRuntimeQualificationError("runtime receipt scope guards are missing")
    for field in (
        "frozen_120_target_benchmark_executed",
        "sealed_holdouts_accessed",
        "proposal_or_value_logic_modified",
        "candidate_backend_activated",
        "performance_metrics_computed",
    ):
        if guards.get(field) is not False:
            raise Graph2EditsRuntimeQualificationError(f"unsafe runtime scope guard: {field}")
    smoke = receipt.get("smoke_inference")
    if not isinstance(smoke, Mapping) or smoke.get("source") != (
        "hand-authored_nonbenchmark_runtime_smoke"
    ):
        raise Graph2EditsRuntimeQualificationError("runtime smoke provenance is not admissible")
    if smoke.get("target_smiles") != NONBENCHMARK_SMOKE_SMILES:
        raise Graph2EditsRuntimeQualificationError("runtime smoke target changed")
    if smoke.get("byte_identical_normalized_outputs") is not True:
        raise Graph2EditsRuntimeQualificationError("runtime smoke was not deterministic")

    if verify_artifacts:
        if repo_root is None:
            raise Graph2EditsRuntimeQualificationError(
                "repo_root is required when verifying runtime artifacts"
            )
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise Graph2EditsRuntimeQualificationError("runtime artifact records are missing")
        for label, record in artifacts.items():
            if not isinstance(record, Mapping):
                raise Graph2EditsRuntimeQualificationError(f"invalid artifact record: {label}")
            path = repo_root / str(record.get("path"))
            if not path.is_file():
                raise Graph2EditsRuntimeQualificationError(f"runtime artifact is missing: {path}")
            if path.stat().st_size != record.get("size_bytes"):
                raise Graph2EditsRuntimeQualificationError(
                    f"runtime artifact byte count mismatch: {label}"
                )
            if sha256_file(path) != record.get("sha256"):
                raise Graph2EditsRuntimeQualificationError(
                    f"runtime artifact SHA-256 mismatch: {label}"
                )

    if verify_environment:
        dependency_resolution = receipt.get("dependency_resolution")
        if not isinstance(dependency_resolution, Mapping):
            raise Graph2EditsRuntimeQualificationError("dependency receipt is missing")
        versions = dependency_resolution.get("installed_versions")
        if not isinstance(versions, Mapping):
            raise Graph2EditsRuntimeQualificationError("installed-version receipt is missing")
        _installed_versions({str(name): str(version) for name, version in versions.items()})
        _pip_check()
