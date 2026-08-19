#!/usr/bin/env python3
"""Run recoverable full-corpus Phase 1 product pretraining on one Modal L4."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
REMOTE_SOURCE_ROOT = Path("/root")
REMOTE_REPO_ROOT = Path("/root/forge_repo")
VOLUME_MOUNT = Path("/forge-output")
DEFAULT_CONFIG = REPO_ROOT / "configs/model/phase1_product_pretrain.json"
ALLOWED_CONFIG_PATHS = (
    "configs/model/phase1_product_pretrain.json",
    "configs/model/phase1_product_pretrain_v2.json",
    "configs/model/phase1_product_pretrain_v3.json",
)
GPU_CLASS = "L4"
VOLUME_NAME = "forge-phase1-training"
REMOTE_SOURCE_FILES = (
    "forge/__init__.py",
    "forge/data/__init__.py",
    "forge/data/r0_splits.py",
    "forge/product/__init__.py",
    "forge/product/defog_feasibility.py",
    "forge/product/lipid_context.py",
    "forge/product/phase1_flow.py",
    "forge/product/sparse_topology_feasibility.py",
)
TRAINING_INPUT_FILES = (
    "configs/model/phase1_product_pretrain.json",
    "configs/model/phase1_product_pretrain_v2.json",
    "configs/model/phase1_product_pretrain_v3.json",
    "data/splits/phase1/manifest.json",
    "results/phase1/product_prelaunch_audit.json",
    "data/splits/m0_03_constitutional/r0_fold_assignments.csv",
    "results/m0_03/r0_constitutional.csv.gz",
    "data/vendor/r1_reaction_enumerated_support_v1.csv",
)

app = modal.App("forge-phase1-product-pretrain")
output_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.11.0",
    "numpy==2.4.2",
    "rdkit==2025.9.6",
)
for relative_source in REMOTE_SOURCE_FILES:
    image = image.add_local_file(
        SOURCE_ROOT / relative_source,
        remote_path=str(REMOTE_SOURCE_ROOT / relative_source),
    )
for relative_input in TRAINING_INPUT_FILES:
    image = image.add_local_file(
        REPO_ROOT / relative_input,
        remote_path=str(REMOTE_REPO_ROOT / relative_input),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_volume_download(remote_path: str, local_path: Path) -> int:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{local_path.name}.", dir=local_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            count = output_volume.read_file_into_fileobj(remote_path, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, local_path)
        return int(count)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _config_run_name(config_path: Path) -> str:
    return f"product_pretrain_full_{_sha256_file(config_path)[:12]}"


@app.function(
    image=image,
    gpu=GPU_CLASS,
    cpu=8.0,
    memory=16384,
    timeout=14_400,
    retries=2,
    volumes={str(VOLUME_MOUNT): output_volume},
)
def train_full_product(config_relative_path: str, run_name: str) -> dict[str, Any]:
    """Train or resume the frozen full-corpus product model."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(REMOTE_SOURCE_ROOT))

    import numpy as np
    import rdkit
    import torch

    from forge.data.r0_splits import sha256_file
    from forge.product.phase1_flow import train_product_pretrain

    if config_relative_path not in ALLOWED_CONFIG_PATHS:
        raise RuntimeError("remote training received an unapproved product config")
    config_path = REMOTE_REPO_ROOT / config_relative_path
    config = json.loads(config_path.read_text())
    expected_run_name = f"product_pretrain_full_{sha256_file(config_path)[:12]}"
    if run_name != expected_run_name:
        raise RuntimeError(
            f"run name {run_name!r} does not match pinned config {expected_run_name!r}"
        )
    output_dir = VOLUME_MOUNT / run_name
    outputs = config["outputs"]
    result_path = output_dir / outputs["result_filename"]
    latest_path = output_dir / outputs["latest_checkpoint_filename"]
    managed_paths = tuple(
        output_dir / outputs[key]
        for key in (
            "result_filename",
            "checkpoint_filename",
            "latest_checkpoint_filename",
            "best_checkpoint_filename",
            "progress_filename",
        )
    )
    if result_path.is_file():
        existing = json.loads(result_path.read_text())
        if existing.get("status") != "complete" or existing.get("config", {}).get(
            "sha256"
        ) != sha256_file(config_path):
            raise RuntimeError("existing completed-run artifact violates the pinned config")
        return {
            "status": "already_complete",
            "run_name": run_name,
            "result": existing,
        }

    resume_checkpoint = latest_path if latest_path.is_file() else None
    partial_without_checkpoint = resume_checkpoint is None and any(
        path.exists() for path in managed_paths
    )
    result = train_product_pretrain(
        config_path,
        REMOTE_REPO_ROOT,
        output_dir,
        smoke=False,
        resume_checkpoint=resume_checkpoint,
        overwrite=partial_without_checkpoint,
    )
    result["cloud_execution"] = {
        "run_name": run_name,
        "gpu_class": GPU_CLASS,
        "volume": VOLUME_NAME,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rdkit": rdkit.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": int(torch.backends.cudnn.version()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "precision": "float32",
            "mixed_precision": False,
        },
        "mounted_sources": {
            f"src/{relative}": sha256_file(REMOTE_SOURCE_ROOT / relative)
            for relative in REMOTE_SOURCE_FILES
        },
        "training_inputs": {
            relative: sha256_file(REMOTE_REPO_ROOT / relative) for relative in TRAINING_INPUT_FILES
        },
        "recovery": {
            "resumed": resume_checkpoint is not None,
            "restarted_uncheckpointed_partial": partial_without_checkpoint,
            "volume_background_commits": True,
        },
    }
    _atomic_json(result_path, result)
    output_volume.commit()
    return {
        "status": "complete",
        "run_name": run_name,
        "resumed": resume_checkpoint is not None,
        "restarted_uncheckpointed_partial": partial_without_checkpoint,
        "result": result,
    }


@app.local_entrypoint()
def main(
    config_path: str = str(DEFAULT_CONFIG),
    local_output: str = "",
) -> None:
    config_file = Path(config_path).resolve()
    allowed = {str((REPO_ROOT / path).resolve()): path for path in ALLOWED_CONFIG_PATHS}
    config_relative_path = allowed.get(str(config_file))
    if config_relative_path is None:
        raise ValueError("full training requires one of the frozen product configs")
    config = json.loads(config_file.read_text())
    run_name = _config_run_name(config_file)
    remote = train_full_product.remote(config_relative_path, run_name)
    result = remote["result"]
    if result.get("status") != "complete":
        raise RuntimeError("remote training did not return a complete result")

    local_output_dir = (
        Path(local_output).resolve()
        if local_output
        else (REPO_ROOT / config["outputs"]["default_full_directory"]).resolve()
    )
    output_names = config["outputs"]
    filenames = (
        output_names["result_filename"],
        output_names["checkpoint_filename"],
        output_names["latest_checkpoint_filename"],
        output_names["best_checkpoint_filename"],
        output_names["progress_filename"],
    )
    downloaded = {}
    for filename in filenames:
        remote_path = f"{run_name}/{filename}"
        local_path = local_output_dir / filename
        downloaded[filename] = {
            "bytes": _atomic_volume_download(remote_path, local_path),
            "sha256": _sha256_file(local_path),
        }

    downloaded_result = json.loads((local_output_dir / output_names["result_filename"]).read_text())
    if downloaded_result != result:
        raise RuntimeError("downloaded result JSON differs from the completed remote result")
    expected_hashes = {
        output_names["checkpoint_filename"]: result["checkpoint"]["sha256"],
        output_names["latest_checkpoint_filename"]: result["latest_checkpoint"]["sha256"],
        output_names["best_checkpoint_filename"]: result["best_checkpoint"]["sha256"],
    }
    for filename, expected in expected_hashes.items():
        observed = downloaded[filename]["sha256"]
        if observed != expected:
            raise RuntimeError(
                f"downloaded {filename} hash mismatch: expected {expected}, observed {observed}"
            )
    print(
        json.dumps(
            {
                "status": remote["status"],
                "run_name": run_name,
                "resumed": remote.get("resumed", False),
                "local_output": str(local_output_dir),
                "downloaded": downloaded,
                "training": result["training"],
                "validation": result["validation"],
                "model": result["model"],
            },
            indent=2,
            sort_keys=True,
        )
    )
